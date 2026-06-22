import os
import subprocess
import time
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser


def get_struc_seq(
    foldseek: str,
    path: str,
    chains: list[str] | None = None,
    process_id: int = 0,
    plddt_mask: bool | str = "auto",
    plddt_threshold: float = 70.0,
    foldseek_verbose: bool = False,
) -> dict[str, tuple[str, str, str]]:
    """Extract amino-acid, Foldseek 3Di, and SaProt combined sequences.

    Returns:
        A dict keyed by chain id. Each value is ``(aa_seq, foldseek_seq,
        combined_seq)`` where ``combined_seq`` interleaves amino-acid tokens
        with lower-case structural tokens, as expected by SaProt.
    """
    foldseek_path = Path(foldseek)
    pdb_path = Path(path)
    if not foldseek_path.exists():
        raise FileNotFoundError(f"Foldseek not found: {foldseek}")
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {path}")

    tmp_save_path = Path(f"get_struc_seq_{process_id}_{time.time()}.tsv")
    cmd = [
        str(foldseek_path),
        "structureto3didescriptor",
        "--threads",
        "1",
        "--chain-name-mode",
        "1",
        str(pdb_path),
        str(tmp_save_path),
    ]
    if not foldseek_verbose:
        cmd.insert(2, "-v")
        cmd.insert(3, "0")

    subprocess.run(cmd, check=True)

    if plddt_mask == "auto":
        plddt_mask = "alphafold" in pdb_path.read_text(errors="ignore").lower()

    seq_dict: dict[str, tuple[str, str, str]] = {}
    name = pdb_path.name
    try:
        with tmp_save_path.open("r") as handle:
            for line in handle:
                desc, seq, struc_seq = line.split("\t")[:3]
                if plddt_mask:
                    struc_seq = _mask_low_plddt(pdb_path, struc_seq, plddt_threshold)

                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    combined_seq = "".join(a + b.lower() for a, b in zip(seq, struc_seq))
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    finally:
        _remove_if_exists(tmp_save_path)
        _remove_if_exists(Path(str(tmp_save_path) + ".dbtype"))

    return seq_dict


def extract_plddt(pdb_path: str | Path, chain_id: str = "A") -> np.ndarray:
    """Extract residue-level pLDDT scores from PDB B-factors."""
    pdb_path = Path(pdb_path)
    if pdb_path.suffix == ".cif":
        parser = MMCIFParser(QUIET=True)
    elif pdb_path.suffix == ".pdb":
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError("Invalid file format. Expected '.cif' or '.pdb'.")

    structure = parser.get_structure("protein", str(pdb_path))
    chain = structure[0][chain_id]

    plddts = []
    for residue in chain:
        atom_scores = [atom.get_bfactor() for atom in residue]
        plddts.append(float(np.mean(atom_scores)))
    return np.array(plddts)


def _mask_low_plddt(
    pdb_path: Path,
    struc_seq: str,
    plddt_threshold: float,
) -> str:
    plddts = extract_plddt(pdb_path)
    if len(plddts) != len(struc_seq):
        raise ValueError(f"Length mismatch: {len(plddts)} != {len(struc_seq)}")

    np_seq = np.array(list(struc_seq))
    np_seq[np.where(plddts < plddt_threshold)[0]] = "#"
    return "".join(np_seq)


def _remove_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass

