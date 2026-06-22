import argparse
import json
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import EsmForProteinFolding, EsmModel, EsmTokenizer


@dataclass
class PipelineConfig:
    esmfold_model: str
    esm_model: str
    saprot_model: str
    foldseek_bin: str
    fold_device: str = "cuda:0"
    embed_device: str = "cuda:0"
    max_len: int = 128
    fold_batch_size: int = 8
    embed_batch_size: int = 64
    chain: str = "A"
    seed: int = 0


class SeqDataset(Dataset):
    def __init__(self, seqs: list[str], tokenizer: EsmTokenizer, max_len: int):
        self.seqs = seqs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        toks = self.tokenizer(
            self.seqs[idx],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=self.max_len,
        )
        return {
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_sequences(input_path: str, id_col: str | None, seq_col: str | None) -> pd.DataFrame:
    path = Path(input_path)
    if path.suffix.lower() in {".fa", ".faa", ".fasta"}:
        from Bio import SeqIO

        records = [(record.id, str(record.seq)) for record in SeqIO.parse(str(path), "fasta")]
        return pd.DataFrame(records, columns=["id", "sequence"])

    df = pd.read_csv(path)
    resolved_id_col = id_col or _first_existing_or_none(df, ["id", "seq_name", "name", "value", "Peptide_ID"])
    resolved_seq_col = seq_col or _first_existing(df, ["sequence", "seq", "protein_seq"])
    if resolved_id_col is None:
        out = df[[resolved_seq_col]].copy()
        out.insert(0, "id", [f"seq_{idx + 1}" for idx in range(len(out))])
        out.columns = ["id", "sequence"]
    else:
        out = df[[resolved_id_col, resolved_seq_col]].copy()
        out.columns = ["id", "sequence"]
    out["id"] = out["id"].astype(str).map(_safe_id)
    out["sequence"] = out["sequence"].astype(str).str.upper()
    return out.drop_duplicates(subset=["sequence"]).reset_index(drop=True)


def fold_sequences(
    seq_df: pd.DataFrame,
    output_dir: str,
    config: PipelineConfig,
) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model = EsmForProteinFolding.from_pretrained(config.esmfold_model)
    tokenizer = EsmTokenizer.from_pretrained(config.esmfold_model)
    model.to(config.fold_device)
    model.eval()

    pdb_paths: list[Path] = []
    rows = list(seq_df[["sequence", "id"]].itertuples(index=False, name=None))
    for start in tqdm(range(0, len(rows), config.fold_batch_size), desc="Folding"):
        batch = rows[start : start + config.fold_batch_size]
        batch_seqs = [seq for seq, _ in batch]
        batch_ids = [_safe_id(seq_id) for _, seq_id in batch]
        batch_paths = [output_path / f"{seq_id}.pdb" for seq_id in batch_ids]
        pdb_paths.extend(batch_paths)
        if all(path.exists() for path in batch_paths):
            continue

        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(config.fold_device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        outputs = {key: value.to("cpu").detach() for key, value in outputs.items()}
        for pdb, path in zip(model.output_to_pdb(outputs), batch_paths):
            path.write_text(pdb)
        _empty_cuda_cache()

    return pdb_paths


def extract_structure_sequences(
    seq_df: pd.DataFrame,
    pdb_dir: str,
    output_csv: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    from foldseek_util import get_struc_seq

    records = []
    for idx, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc="Foldseek 3Di"):
        seq_id = _safe_id(row["id"])
        pdb_path = Path(pdb_dir) / f"{seq_id}.pdb"
        parsed = get_struc_seq(
            config.foldseek_bin,
            str(pdb_path),
            chains=[config.chain],
            process_id=idx,
            plddt_mask=False,
        )[config.chain]
        aa_seq, foldseek_seq, combined_seq = parsed
        records.append(
            {
                "id": seq_id,
                "seq": aa_seq,
                "foldseek_seq": foldseek_seq,
                "combined_seq": combined_seq,
            }
        )
    out_df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


def train(
    training_csv: str,
    model_dir: str,
    config: PipelineConfig,
    force_recompute: bool = False,
) -> None:
    seed_everything(config.seed)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(training_csv)
    required = {"seq", "combined_seq", "value"}
    missing = required - set(training.columns)
    if missing:
        raise ValueError(f"Training CSV missing columns: {sorted(missing)}")

    training["label"] = training["value"].astype(str).map(lambda x: 1 if x.startswith("AMP") else 0)
    features = _get_or_make_training_features(training, model_path, config, force_recompute)

    classifiers = {
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42, max_iter=100),
        "RandomForest": RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=-1),
    }
    labels = training["label"].to_numpy()
    for name, clf in classifiers.items():
        clf.fit(features, labels)
        joblib.dump(clf, model_path / f"{name}.joblib")

    (model_path / "config.json").write_text(json.dumps(asdict(config), indent=2))


def predict(
    structure_csv: str,
    model_dir: str,
    output_csv: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    seqs = pd.read_csv(structure_csv)
    required = {"id", "seq", "combined_seq"}
    missing = required - set(seqs.columns)
    if missing:
        raise ValueError(f"Inference CSV missing columns: {sorted(missing)}")

    esm_embeds = embed_sequences(seqs["seq"].tolist(), config.esm_model, config)
    saprot_embeds = embed_sequences(seqs["combined_seq"].tolist(), config.saprot_model, config)
    features = np.concatenate([esm_embeds, saprot_embeds], axis=1)

    pred_df = seqs[["id", "seq", "combined_seq"]].copy()
    score_cols = []
    for clf_path in sorted(Path(model_dir).glob("*.joblib")):
        clf = joblib.load(clf_path)
        col = clf_path.stem
        pred_df[col] = clf.predict_proba(features)[:, 1]
        score_cols.append(col)
    pred_df["mean_score"] = pred_df[score_cols].mean(axis=1)
    pred_df = pred_df.sort_values("mean_score", ascending=False)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)
    return pred_df


def run_all(args: argparse.Namespace) -> None:
    config = _config_from_args(args)
    seed_everything(config.seed)
    seq_df = read_sequences(args.input, args.id_col, args.seq_col)
    work_dir = Path(args.work_dir)
    pdb_dir = work_dir / "structures"
    structure_csv = work_dir / "structure_sequences.csv"
    model_dir = work_dir / "model"
    output_csv = Path(args.output)

    fold_sequences(seq_df, str(pdb_dir), config)
    extract_structure_sequences(seq_df, str(pdb_dir), str(structure_csv), config)
    train(args.training_csv, str(model_dir), config, args.force_recompute)
    predict(str(structure_csv), str(model_dir), str(output_csv), config)


def embed_sequences(seqs: list[str], model_name: str, config: PipelineConfig) -> np.ndarray:
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    embedder = EsmModel.from_pretrained(model_name, output_hidden_states=True)
    dataset = SeqDataset(seqs, tokenizer, config.max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=config.embed_batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    reps = []
    embedder.to(config.embed_device)
    embedder.eval()
    use_cuda_autocast = config.embed_device.startswith("cuda")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Embedding {Path(model_name).name}"):
            input_ids = batch["input_ids"].to(config.embed_device)
            attention_mask = batch["attention_mask"].to(config.embed_device)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_cuda_autocast
                else nullcontext()
            )
            with autocast_context:
                hidden = embedder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            reps.append(_mean_pool_without_special_tokens(hidden, attention_mask).cpu())

    embedder.to("cpu")
    _empty_cuda_cache()
    return torch.cat(reps, dim=0).numpy()


def _get_or_make_training_features(
    training: pd.DataFrame,
    model_path: Path,
    config: PipelineConfig,
    force_recompute: bool,
) -> np.ndarray:
    feature_path = model_path / "training_features.npy"
    if feature_path.exists() and not force_recompute:
        return np.load(feature_path)

    esm_embeds = embed_sequences(training["seq"].tolist(), config.esm_model, config)
    saprot_embeds = embed_sequences(training["combined_seq"].tolist(), config.saprot_model, config)
    features = np.concatenate([esm_embeds, saprot_embeds], axis=1)
    np.save(feature_path, features)
    return features


def _mean_pool_without_special_tokens(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.clone()
    mask[:, 0] = 0
    seq_len = attention_mask.sum(dim=1) - 1
    for i in range(mask.size(0)):
        if seq_len[i] >= 0:
            mask[i, seq_len[i]] = 0
    mask = mask.unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden * mask).sum(dim=1) / denom


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    value = _first_existing_or_none(df, candidates)
    if value is None:
        raise ValueError(f"Could not find any of these columns: {list(candidates)}")
    return value


def _first_existing_or_none(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _safe_id(value: object) -> str:
    return str(value).replace("|", "__").replace("/", "_").replace("\\", "_").strip()


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        esmfold_model=args.esmfold_model,
        esm_model=args.esm_model,
        saprot_model=args.saprot_model,
        foldseek_bin=args.foldseek_bin,
        fold_device=args.fold_device,
        embed_device=args.embed_device,
        max_len=args.max_len,
        fold_batch_size=args.fold_batch_size,
        embed_batch_size=args.embed_batch_size,
        chain=args.chain,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AMP-SiGMA sequence-to-structure-to-embedding prediction pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run folding, 3Di extraction, training, and prediction.")
    _add_common_args(run_parser)
    run_parser.add_argument("--input", required=True, help="Input FASTA/CSV containing protein sequences.")
    run_parser.add_argument("--training-csv", required=True, help="CSV with seq, combined_seq, value columns.")
    run_parser.add_argument("--output", default="outputs/predictions.csv")
    run_parser.add_argument("--work-dir", default="outputs")
    run_parser.add_argument("--id-col", default=None)
    run_parser.add_argument("--seq-col", default=None)
    run_parser.add_argument("--force-recompute", action="store_true")
    run_parser.set_defaults(func=run_all)

    train_parser = subparsers.add_parser("train", help="Train classifiers from an existing structure-sequence CSV.")
    _add_common_args(train_parser)
    train_parser.add_argument("--training-csv", required=True)
    train_parser.add_argument("--model-dir", default="outputs/model")
    train_parser.add_argument("--force-recompute", action="store_true")
    train_parser.set_defaults(func=lambda args: train(args.training_csv, args.model_dir, _config_from_args(args), args.force_recompute))

    pred_parser = subparsers.add_parser("predict", help="Predict from an existing CSV with id, seq, combined_seq.")
    _add_common_args(pred_parser)
    pred_parser.add_argument("--structure-csv", required=True)
    pred_parser.add_argument("--model-dir", default="outputs/model")
    pred_parser.add_argument("--output", default="outputs/predictions.csv")
    pred_parser.set_defaults(func=lambda args: predict(args.structure_csv, args.model_dir, args.output, _config_from_args(args)))

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--esmfold-model", required=True, help="Local path or HF id for ESMFold.")
    parser.add_argument("--esm-model", required=True, help="Local path or HF id for ESM2.")
    parser.add_argument("--saprot-model", required=True, help="Local path or HF id for SaProt.")
    parser.add_argument("--foldseek-bin", required=True, help="Path to foldseek executable.")
    parser.add_argument("--fold-device", default="cuda:0")
    parser.add_argument("--embed-device", default="cuda:0")
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--fold-batch-size", type=int, default=8)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--seed", type=int, default=0)


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
