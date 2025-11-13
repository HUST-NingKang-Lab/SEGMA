import torch
import pandas as pd
from tqdm import tqdm
import os
from transformers import EsmTokenizer, EsmForProteinFolding
from utils import get_struc_seq
import os
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
import argparse


def fold_protein(model, tokenizer, data, path, batch_size=10, device='cuda'):
    for batch_start in tqdm(range(0, len(data), batch_size)):
        batch_data = data[batch_start:batch_start + batch_size]
        batch_seqs = [seq for seq, _ in batch_data]
        batch_labels = [label for _, label in batch_data]

        try:
            inputs = tokenizer(batch_seqs, return_tensors="pt", add_special_tokens=False, padding=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            outputs = {k: v.to("cpu").detach() for k, v in outputs.items()}

            # Convert outputs to PDB format and save
            folded_positions = model.output_to_pdb(outputs)
            torch.cuda.empty_cache()
            for folded_position, label in zip(folded_positions, batch_labels):
                with open(f"{path}/{label}.pdb", "w") as f:
                    f.write(folded_position)
        except Exception as e:
            print(f"Error processing batch starting at index {batch_start}: {e}")
            # print sequence in the batch
            for seq, label in batch_data:
                print(f"Label: {label}, Sequence: {seq}")
            continue

def get_batch_structure(pdb_dir, save_path, foldseek_path):
    
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    seq_list = []
    error_num = 0
    
    for pdb_file in tqdm(pdb_files):
        try:
            parsed_seqs = get_struc_seq(foldseek_path, os.path.join(pdb_dir, pdb_file), ["A"], plddt_mask=False)["A"]
        except Exception as e:
            error_num += 1
            continue
        seq, foldseek_seq, combined_seq = parsed_seqs
        
        #get pdb file name
        label = pdb_file.split(".pd")[0]
        seq_list.append((seq, foldseek_seq, combined_seq, label))

    #save to csv file
    print(f"error_num: {error_num}")
    print(f"total: {len(pdb_files)}")
    df = pd.DataFrame(seq_list, columns=["seq", "foldseek_seq", "combined_seq", "value"])
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process input FASTA file and fold proteins.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input FASTA file")
    args = parser.parse_args()

    seqs = []
    records = list(SeqIO.parse(args.input, 'fasta'))
    for record in records:
        seqs.append((str(record.seq), record.id))

    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")
    device = "cuda:0"
    model.to(device)
    os.makedirs("peptide_structure", exist_ok=True)
    fold_protein(model, tokenizer, seqs, "peptide_structure", batch_size=10, device=device)
    foldseek_path = 'foldseek'
    print("get structure sequence of all pdb files")
    pdb_dir = 'peptide_structure'
    save_path = 'candidate_seqs.csv'
    get_batch_structure(pdb_dir, save_path, foldseek_path)
    