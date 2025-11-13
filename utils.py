from torch.utils.data import Dataset
import torch
import torch.nn as nn
import math
import os
import time
import numpy as np
from Bio.PDB import PDBParser, MMCIFParser


class Seq4Transformer(Dataset):
    def __init__(self, seqs, tokenizer, max_len=256):
        self.seqs = seqs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        toks = self.tokenizer(self.seqs[idx], 
                                truncation=True, 
                                padding='max_length', 
                                return_tensors='pt', 
                                max_length=self.max_len)
        return {'input_ids': toks['input_ids'].squeeze(),
                'attention_mask': toks['attention_mask'].squeeze()}

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) / math.sqrt(rank))
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = x @ self.W_a  # batch * rank
        x = x @ self.W_b  # batch * out_dim
        return self.alpha * x
    
class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def extract_plddt(pdb_path: str) -> np.ndarray:
    
    if pdb_path.endswith(".cif"):
        parser = MMCIFParser()
    elif pdb_path.endswith(".pdb"):
        parser = PDBParser()
    else:
        raise ValueError("Invalid file format for plddt extraction. Must be '.cif' or '.pdb'.")
    
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    chain = model["A"]

    # Extract plddt scores
    plddts = []
    for residue in chain:
        residue_plddts = []
        for atom in residue:
            plddt = atom.get_bfactor()
            residue_plddts.append(plddt)
        
        plddts.append(np.mean(residue_plddts))

    plddts = np.array(plddts)
    return plddts


def get_struc_seq(foldseek,
                path,
                chains: list = None,
                process_id: int = 0,
                plddt_mask: bool = "auto",
                plddt_threshold: float = 70.,
                foldseek_verbose: bool = False) -> dict:
   
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    # Check whether the structure is predicted by AlphaFold2
    if plddt_mask == "auto":
        with open(path, "r") as r:
            plddt_mask = True if "alphafold" in r.read().lower() else False
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_mask:
                try:
                    plddts = extract_plddt(path)
                    assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                    
                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)
                
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Failed to mask plddt for {name}")
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict
    
  