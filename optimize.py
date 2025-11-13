
from transformers import EsmTokenizer, EsmModel
from transformers import EsmForMaskedLM
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import numpy as np
from functools import partial
from utils import Seq4Transformer, LoRALayer, LinearWithLoRA
import os
import argparse



def apply_lora(
    model,
    lora_rank=8,
    lora_alpha=16,
    lora_query=True,
    lora_key=False,
    lora_value=True,
    lora_projection=False,
    lora_mlp=False,
    lora_head=True,
):
    # freeze model layers
    for param in model.parameters():
        param.requires_grad = False

    # for adding lora to linear layers
    linear_with_lora = partial(LinearWithLoRA, rank=lora_rank, alpha=lora_alpha)

    # iterate through each transfomer layer
    for layer in model.esm.encoder.layer:
        if lora_query:
            layer.attention.self.query = linear_with_lora(layer.attention.self.query)

        if lora_key:
            layer.attention.self.key = linear_with_lora(layer.attention.self.key)

        if lora_value:
            layer.attention.self.value = linear_with_lora(layer.attention.self.value)

        if lora_projection:
            layer.attention.output.dense = linear_with_lora(
                layer.attention.output.dense
            )

        if lora_mlp:
            layer.output.dense = linear_with_lora(layer.output.dense)
            layer.output.dense = linear_with_lora(layer.output.dense)

    if lora_head:
        model.lm_head.dense = linear_with_lora(model.lm_head.dense)
        model.lm_head.decoder = linear_with_lora(model.lm_head.decoder)  


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def generate_mutations(sequence, model, tokenizer, device):
    # batch_size = len(sequence) // 2
    batch_size = 10
    device = torch.device(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        seq_length = len(sequence) // 2
        heatmap = np.zeros((seq_length, 20))
        vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        masked_sequences = []
        for i in range(seq_length):
            mask_token = '#' + sequence[i * 2 + 1]
            masked_sequence = sequence[:i * 2] + mask_token + sequence[i * 2 + 2:]
            masked_sequences.append(masked_sequence)

        for batch_start in range(0, len(masked_sequences), batch_size):
            batch_sequences = masked_sequences[batch_start:batch_start + batch_size]
            inputs = tokenizer(batch_sequences, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits.cpu().numpy()

            for batch_idx, masked_sequence in enumerate(batch_sequences):
                i = batch_start + batch_idx
                mask_token = '#' + sequence[i * 2 + 1]
                mask_index = tokenizer.convert_tokens_to_ids(mask_token)
                mask_logits = logits[batch_idx, i + 1, mask_index]

                for j, aa in enumerate(vocab):
                    new_token = aa + sequence[i * 2 + 1]
                    new_token_id = tokenizer.convert_tokens_to_ids(new_token)
                    heatmap[i, j] = logits[batch_idx, i + 1, new_token_id] - mask_logits

                heatmap[i] = np.exp(heatmap[i]) / np.sum(np.exp(heatmap[i]))
                heatmap[i] -= heatmap[i][vocab.index(sequence[i * 2])]
    return heatmap

def make_mutation(sequence, heatmap, top_k=20):
    # at most top_k mutations
    seq_length = int(len(sequence) / 2)
    vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    mutation_list = []
    heatmap_flat = heatmap.flatten()
    # get the indices of the top_k values in heatmap_flat, with the value > 0
    top_k_indices = np.argpartition(heatmap_flat, -top_k)[-top_k:]
    # filter the indices with value > 0
    top_k_indices = top_k_indices[heatmap_flat[top_k_indices] > 0]
    # sort the indices by the value
    top_k_indices = top_k_indices[np.argsort(heatmap_flat[top_k_indices])[::-1]]
    for index in top_k_indices:
        row = index // 20
        col = index % 20
        mutated_aa = vocab[col]
        position = row
        mutated_sequence = sequence[:position*2] + mutated_aa + sequence[position*2+1:]
        mutation_list.append(mutated_sequence)
    return mutation_list

# import numerate


def evaluate_sequence(sequences):
    vocab = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # remove any character in the string if it is in vocab
    structure_seq = ''.join([i for i in sequences[0] if i not in vocab])
    with open('current_seqs.fasta', 'w') as f:
        for i, seq in enumerate(sequences):
            # wrrite '>P{i}\n' before each sequence
            f.write(f'>P{i}\n')
            seq = ''.join([i for i in seq if not i.islower()])
            f.write(seq + '\n')
            f.flush()

    os.system('python predict_MIC/APEX_predict.py -i current_seqs.fasta')
    # read the MICs from the csv file
    df = pd.read_csv('predict_MIC/Predicted_MICs.csv', index_col=0)
    standard_MICs = []
    target_MICs = []
    target_bac = ['E. coli ATCC 11775', 'E. coli AIC221',
        'E. coli AIC222', 'P. aeruginosa PA01',
        'P. aeruginosa PA14', 'S. aureus ATCC 12600',
        'S. aureus (ATCC BAA-1556) - MRSA']
    seqs = []
    effective_count = []
    for index, row in df.iterrows():
        index = [index[i]+structure_seq[i].lower() for i in range(len(index))]
        index = ''.join(index)
        seqs.append(index)
        # get the num of columns whose MIC < 32
        count = 0
        median_MICs = []
        all_bac = df.columns.tolist()
        for bac in all_bac:
            if row[bac] < 32:
                count += 1
            median_MICs.append(row[bac])
        effective_count.append(count)
        standard_MICs.append(np.median(median_MICs))
        median_MICs = []
        for bac in target_bac:
            median_MICs.append(row[bac])
        # get the mean MIC of the target bacteria
        target_MICs.append(np.mean(median_MICs))        
    return seqs, standard_MICs, target_MICs, effective_count


def optimize(sequence, model, tokenizer, device, iterations=10, top_k=10):
    # use tree search to optimize the sequence
    current_sequence_space = [sequence]
    current_sequence_space, current_sequence_stand_mic, current_sequence_tar_mic, current_sequence_effcount = evaluate_sequence(current_sequence_space)
    # print(f" Original sequence: {sequence}, MIC: {current_sequence_space_mic[0]}")
    # print(f" Starting optimization for sequence: {sequence}, standard MIC: {current_sequence_stand_mic[0]}, target MIC: {current_sequence_tar_mic[0]}")
    optimize_record = []
    optimize_record.append((current_sequence_space[0], current_sequence_stand_mic[0], 
                            current_sequence_tar_mic[0],current_sequence_effcount[0], '0'))
    for iter in range(iterations):
        # print(f" Optimization round {iter+1}")
        next_sequence_space = []
        next_sequence_stand_mic = []
        next_sequence_tar_mic = []
        next_sequence_effcount = []
        for i, seq in enumerate(current_sequence_space):
            heatmap = generate_mutations(seq, model, tokenizer, device)
            mutations = make_mutation(seq, heatmap, top_k)
            if not mutations:
                next_sequence_space.append(seq)
                next_sequence_stand_mic.append(current_sequence_stand_mic[i])
                next_sequence_tar_mic.append(current_sequence_tar_mic[i])
                next_sequence_effcount.append(current_sequence_effcount[i])
                continue
            mutations, mut_stand_scores, mut_tar_scores, mut_effective_count = evaluate_sequence(mutations)
            if_mutated = False
            for j in range(len(mutations)):
                # ---- only check standard MIC and effective counts----
                if mut_stand_scores[j] < min(current_sequence_stand_mic):
                    if mut_effective_count[j] >= np.max(current_sequence_effcount): 
                        next_sequence_space.append(mutations[j])
                        next_sequence_tar_mic.append(mut_tar_scores[j])
                        next_sequence_stand_mic.append(mut_stand_scores[j])
                        next_sequence_effcount.append(mut_effective_count[j])
                        if_mutated = True
            # if not mutated, keep the original sequence
            if not if_mutated:
                next_sequence_space.append(seq)
                next_sequence_stand_mic.append(current_sequence_stand_mic[i])
                next_sequence_tar_mic.append(current_sequence_tar_mic[i])
                next_sequence_effcount.append(current_sequence_effcount[i])     
        
        # expand instead of replace
        current_sequence_space = current_sequence_space + next_sequence_space
        current_sequence_stand_mic = current_sequence_stand_mic + next_sequence_stand_mic
        current_sequence_tar_mic = current_sequence_tar_mic + next_sequence_tar_mic
        current_sequence_effcount = current_sequence_effcount + next_sequence_effcount
        
        # drop duplicates
        unique_sequences = {}
        for i in range(len(current_sequence_space)):
            if current_sequence_space[i] not in unique_sequences:
                unique_sequences[current_sequence_space[i]] = [current_sequence_stand_mic[i], 
                                                                current_sequence_tar_mic[i],
                                                                current_sequence_effcount[i]]
            else:
                continue
        current_sequence_space = list(unique_sequences.keys())
        current_sequence_stand_mic = [unique_sequences[seq][0] for seq in current_sequence_space]
        current_sequence_tar_mic = [unique_sequences[seq][1] for seq in current_sequence_space]
        current_sequence_effcount = [unique_sequences[seq][2] for seq in current_sequence_space]
        
        # only keep lowest 10 sequences
        if len(current_sequence_space) > 10:
            sorted_indices = np.argsort(current_sequence_stand_mic)
            current_sequence_space = [current_sequence_space[i] for i in sorted_indices[:10]]
            current_sequence_tar_mic = [current_sequence_tar_mic[i] for i in sorted_indices[:10]]
            current_sequence_stand_mic = [current_sequence_stand_mic[i] for i in sorted_indices[:10]]
            current_sequence_effcount = [current_sequence_effcount[i] for i in sorted_indices[:10]]
        
        for i in range(len(current_sequence_space)):
            optimize_record.append((current_sequence_space[i], 
                                    current_sequence_stand_mic[i], 
                                    current_sequence_tar_mic[i],
                                    current_sequence_effcount[i],
                                    str(iter+1)))

    optimize_record_df = pd.DataFrame(optimize_record)
    optimize_record_df.columns = ['sequence', 'standard_MIC', 'target_MIC', 'effective_count', 'iteration']
    optimize_record_df.to_csv('optimize_record.csv')
    optimize_record_df = pd.read_csv('optimize_record.csv')
    optimize_record_df = optimize_record_df.loc[optimize_record_df.groupby('iteration')['standard_MIC'].idxmin()].reset_index(drop=True)
    iter_sequences = {}
    for i, row in optimize_record_df.iterrows():
        iter_sequences[row['iteration']] = row['sequence']
    # rank it by iteration, from lower to higher
    iter_sequences = [iter_sequences[i] for i in range(iterations+1)]
    return iter_sequences

            
def main(input_file, output_file):
    model_name = 'westlake-repl/SaProt_650M_AF2'
    lora_weights = torch.load("lora_weights.pth")
    base_model = EsmForMaskedLM.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(model_name)

    apply_lora(base_model, lora_rank=8, lora_alpha=8, 
                lora_query=True, lora_key=False, 
                lora_value=True, lora_projection=False, 
                lora_mlp=False, lora_head=True)

    model_state_dict = base_model.state_dict()
    model_state_dict.update(lora_weights)
    base_model.load_state_dict(model_state_dict)
    base_model.to('cuda:0')

    candidates = pd.read_csv(input_file)
    candidates = candidates.reset_index(drop=True)
    print(f"Number of candidates: {len(candidates)}")

    optimized_seqs = []
    for i in tqdm(range(len(candidates))):
        id_ = candidates.loc[i, 'ID']
        seq = candidates.loc[i, 'struct_seq']
        optimized_seq = optimize(seq, base_model, tokenizer, 'cuda:0', iterations=18, top_k=10)
        for j in range(len(optimized_seq)):
            optimized_seqs.append((id_, optimized_seq[j], {'iteration': j}))

    optimized_seqs_df = pd.DataFrame(optimized_seqs)
    optimized_seqs_df.columns = ['ID', 'optimized_seq', 'interation_info']
    optimized_seqs_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize sequences using LoRA-enhanced model.")
    parser.add_argument("-i", "--input", required=True, help="Input CSV file with candidate sequences.")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file for optimized sequences.")
    args = parser.parse_args()
    main(args.input, args.output)





