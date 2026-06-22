import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from amp_sigma_pipeline import PipelineConfig, embed_sequences, read_sequences, seed_everything


def read_input_sequences(args: argparse.Namespace) -> pd.DataFrame:
    if args.sequence:
        return pd.DataFrame(
            [{"id": args.sequence_id, "sequence": args.sequence.strip().upper()}]
        )
    if not args.input:
        raise ValueError("Provide either --sequence or --input.")
    return read_sequences(args.input, args.id_col, args.seq_col)


def save_embedding_csv(
    seq_df: pd.DataFrame,
    embeddings: np.ndarray,
    output_csv: str,
) -> None:
    embed_cols = [f"emb_{idx}" for idx in range(embeddings.shape[1])]
    embed_df = pd.DataFrame(embeddings, columns=embed_cols)
    out_df = pd.concat([seq_df[["id", "sequence"]].reset_index(drop=True), embed_df], axis=1)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    seq_df = read_input_sequences(args)
    config = PipelineConfig(
        esmfold_model="",
        esm_model=args.model,
        saprot_model="",
        foldseek_bin="",
        embed_device=args.embed_device,
        max_len=args.max_len,
        embed_batch_size=args.batch_size,
        seed=args.seed,
    )

    embeddings = embed_sequences(seq_df["sequence"].tolist(), args.model, config)

    output_npy = Path(args.output_npy)
    output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, embeddings)

    metadata = seq_df[["id", "sequence"]].copy()
    metadata.insert(0, "embedding_index", range(len(metadata)))
    metadata["embedding_dim"] = embeddings.shape[1]
    metadata["model"] = args.model
    metadata_path = Path(args.metadata_csv)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)

    if args.output_csv:
        save_embedding_csv(seq_df, embeddings, args.output_csv)

    print(f"Saved embeddings: {output_npy} shape={embeddings.shape}")
    print(f"Saved metadata: {metadata_path}")
    if args.output_csv:
        print(f"Saved embedding CSV: {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate pooled protein language-model embeddings from amino-acid sequences."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--sequence", help="Single protein sequence.")
    input_group.add_argument("--input", help="FASTA or CSV containing protein sequences.")
    parser.add_argument("--model", required=True, help="Local path or HF id for an ESM-compatible model.")
    parser.add_argument("--output-npy", default="outputs/protein_embeddings.npy")
    parser.add_argument("--metadata-csv", default="outputs/protein_embedding_metadata.csv")
    parser.add_argument("--output-csv", default=None, help="Optional wide CSV containing embedding columns.")
    parser.add_argument("--sequence-id", default="seq_1", help="ID used with --sequence.")
    parser.add_argument("--id-col", default=None, help="CSV id column. Auto-detected when omitted.")
    parser.add_argument("--seq-col", default=None, help="CSV sequence column. Auto-detected when omitted.")
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

