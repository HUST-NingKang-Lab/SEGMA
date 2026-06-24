import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from spear_pipeline import PipelineConfig, predict, seed_everything, train


def extract_structure_folder(
    pdb_dir: str,
    output_csv: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    from foldseek_util import get_struc_seq

    pdb_path = Path(pdb_dir)
    if not pdb_path.exists():
        raise FileNotFoundError(f"Structure folder not found: {pdb_dir}")

    structure_files = sorted(
        path for path in pdb_path.iterdir() if path.suffix.lower() in {".pdb", ".cif"}
    )
    if not structure_files:
        raise ValueError(f"No .pdb or .cif files found in {pdb_dir}")

    records = []
    error_records = []
    for idx, structure_file in enumerate(tqdm(structure_files, desc="Foldseek 3Di")):
        seq_id = structure_file.stem.replace("|", "__")
        try:
            parsed = get_struc_seq(
                config.foldseek_bin,
                str(structure_file),
                chains=[config.chain],
                process_id=idx,
                plddt_mask=False,
            )
            aa_seq, foldseek_seq, combined_seq = parsed[config.chain]
            records.append(
                {
                    "id": seq_id,
                    "seq": aa_seq,
                    "foldseek_seq": foldseek_seq,
                    "combined_seq": combined_seq,
                    "structure_file": str(structure_file),
                }
            )
        except Exception as exc:
            error_records.append({"id": seq_id, "structure_file": str(structure_file), "error": str(exc)})

    if not records:
        raise RuntimeError("No structure sequences were extracted successfully.")

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    structure_df = pd.DataFrame(records)
    structure_df.to_csv(output_path, index=False)

    if error_records:
        error_path = output_path.with_name(f"{output_path.stem}_errors.csv")
        pd.DataFrame(error_records).to_csv(error_path, index=False)
        print(f"Saved {len(error_records)} failed structures to {error_path}")

    return structure_df


def has_trained_classifiers(model_dir: str) -> bool:
    return any(Path(model_dir).glob("*.joblib"))


def run(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        esmfold_model="",
        esm_model=args.esm_model,
        saprot_model=args.saprot_model,
        foldseek_bin=args.foldseek_bin,
        embed_device=args.embed_device,
        max_len=args.max_len,
        embed_batch_size=args.embed_batch_size,
        chain=args.chain,
        seed=args.seed,
    )
    seed_everything(config.seed)

    extract_structure_folder(args.pdb_dir, args.structure_csv, config)

    if not has_trained_classifiers(args.model_dir):
        if not args.training_csv:
            raise ValueError(
                "No trained classifiers found in model-dir. Provide --training-csv to train them first."
            )
        train(args.training_csv, args.model_dir, config, force_recompute=args.force_recompute)

    predict(args.structure_csv, args.model_dir, args.output, config)
    print(f"Saved predictions to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict AMP scores directly from an existing folder of PDB/CIF structures."
    )
    parser.add_argument("--pdb-dir", required=True, help="Folder containing .pdb or .cif files.")
    parser.add_argument("--foldseek-bin", required=True, help="Path to foldseek executable.")
    parser.add_argument("--esm-model", required=True, help="Local path or HF id for ESM2.")
    parser.add_argument("--saprot-model", required=True, help="Local path or HF id for SaProt.")
    parser.add_argument("--model-dir", default="outputs/model", help="Folder containing or storing classifiers.")
    parser.add_argument("--training-csv", default=None, help="Optional training CSV if classifiers do not exist.")
    parser.add_argument("--structure-csv", default="outputs/structure_sequences_from_pdb.csv")
    parser.add_argument("--output", default="outputs/predictions_from_pdb.csv")
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-recompute", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
