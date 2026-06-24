import argparse
from pathlib import Path

import pandas as pd

from spear_pipeline import PipelineConfig, predict, seed_everything, train


def normalize_structure_table(
    input_csv: str,
    normalized_csv: str,
    id_col: str | None = None,
    seq_col: str | None = None,
    combined_col: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    resolved_seq_col = seq_col or _first_existing(df, ["seq", "sequence", "protein_seq"])
    resolved_combined_col = combined_col or _first_existing(df, ["combined_seq", "structure_seq", "saprot_seq"])
    resolved_id_col = id_col or _first_existing_or_none(df, ["id", "seq_name", "name", "value", "Peptide_ID"])

    out = pd.DataFrame()
    if resolved_id_col is None:
        out["id"] = [f"seq_{idx + 1}" for idx in range(len(df))]
    else:
        out["id"] = df[resolved_id_col].astype(str).map(_safe_id)
    out["seq"] = df[resolved_seq_col].astype(str).str.upper()
    out["combined_seq"] = df[resolved_combined_col].astype(str)

    if "foldseek_seq" in df.columns:
        out["foldseek_seq"] = df["foldseek_seq"].astype(str)

    output_path = Path(normalized_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def has_trained_classifiers(model_dir: str) -> bool:
    return any(Path(model_dir).glob("*.joblib"))


def run(args: argparse.Namespace) -> None:
    config = PipelineConfig(
        esmfold_model="",
        esm_model=args.esm_model,
        saprot_model=args.saprot_model,
        foldseek_bin="",
        embed_device=args.embed_device,
        max_len=args.max_len,
        embed_batch_size=args.embed_batch_size,
        seed=args.seed,
    )
    seed_everything(config.seed)

    normalize_structure_table(
        args.structure_table,
        args.normalized_csv,
        id_col=args.id_col,
        seq_col=args.seq_col,
        combined_col=args.combined_col,
    )

    if not has_trained_classifiers(args.model_dir):
        if not args.training_csv:
            raise ValueError(
                "No trained classifiers found in model-dir. Provide --training-csv to train them first."
            )
        train(args.training_csv, args.model_dir, config, force_recompute=args.force_recompute)

    predict(args.normalized_csv, args.model_dir, args.output, config)
    print(f"Saved predictions to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict AMP scores from an existing Foldseek/SaProt structure-sequence CSV."
    )
    parser.add_argument("--structure-table", required=True, help="CSV containing seq and combined_seq columns.")
    parser.add_argument("--esm-model", required=True, help="Local path or HF id for ESM2.")
    parser.add_argument("--saprot-model", required=True, help="Local path or HF id for SaProt.")
    parser.add_argument("--model-dir", default="outputs/model", help="Folder containing or storing classifiers.")
    parser.add_argument("--training-csv", default=None, help="Optional training CSV if classifiers do not exist.")
    parser.add_argument("--normalized-csv", default="outputs/normalized_structure_table.csv")
    parser.add_argument("--output", default="outputs/predictions_from_structure_table.csv")
    parser.add_argument("--id-col", default=None)
    parser.add_argument("--seq-col", default=None)
    parser.add_argument("--combined-col", default=None)
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-recompute", action="store_true")
    return parser


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    value = _first_existing_or_none(df, candidates)
    if value is None:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return value


def _first_existing_or_none(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _safe_id(value: object) -> str:
    return str(value).replace("|", "__").replace("/", "_").replace("\\", "_").strip()


if __name__ == "__main__":
    run(build_parser().parse_args())

