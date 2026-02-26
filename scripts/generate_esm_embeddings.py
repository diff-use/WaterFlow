"""
Generate and cache ESM3 residue-level embeddings for protein structures.

This script:
1. Reads a split file containing PDB entries
2. For each entry, loads the PDB with biotite (ground truth residue list)
3. Runs ESM3 to get per-residue embeddings
4. Aligns ESM embeddings to biotite structure (inserting zeros for skipped residues)
5. Saves embeddings to separate .pt files in cache_dir/esm/

ESM3's PDB parser skips certain non-canonical amino acids (SEC, PYL) entirely,
while biotite's filter_amino_acids() includes them. This script ensures the
output embeddings match the residue count from biotite, using zero vectors
for any residues ESM skipped (non-canonical AAs treated as UNK).

Usage:
    python scripts/generate_esm_embeddings.py \
        --split_file /path/to/split.txt \
        --cache_dir /path/to/cache \
        --base_pdb_dir /path/to/pdbs \
        [--device cuda:0] \
        [--batch_limit 10] \
        [--overwrite]

Output format (per cache file):
    {
        'residue_embeddings': Tensor of shape (N_res, embed_dim),
        'sequence': str,  # biotite-derived sequence (X for non-canonical)
        'pdb_id': str,
    }
"""

import argparse
from pathlib import Path

import biotite.structure as bts
import numpy as np
import torch
from biotite.structure.io.pdb import PDBFile, get_structure
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_complex import ProteinComplex
from loguru import logger
from tqdm import tqdm

from src.utils import normalize_ins_code, setup_logging_for_tqdm

# Standard 3-letter to 1-letter amino acid mapping (20 canonical only)
THREE_TO_ONE = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def parse_split_file(split_file: str, base_pdb_dir: Path) -> list[dict]:
    """
    Parse split file and construct entries with paths.
    """
    entries = []
    with open(split_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("_")
            if len(parts) < 2:
                logger.warning(f"Skipping malformed line: {line}")
                continue

            pdb_id = parts[0]
            pdb_path = base_pdb_dir / pdb_id / f"{pdb_id}_final.pdb"

            entries.append(
                {
                    "pdb_id": pdb_id,
                    "pdb_path": pdb_path,
                    "cache_key": line,
                }
            )

    return entries


def get_biotite_residues(pdb_path: str) -> tuple[list[str], int]:
    """
    Parse PDB with biotite to get ground truth residue list.

    Uses the same residue counting method as geometry cache:
    unique (chain_id, res_id, ins_code) tuples in sorted order.

    Returns:
        biotite_seq: List of single-letter codes (X for non-canonical)
        num_residues: Total residue count
    """
    pdb_file = PDBFile.read(pdb_path)
    atoms = get_structure(pdb_file, model=1, altloc="occupancy")
    atoms = atoms[atoms.element != "H"]
    protein_mask = bts.filter_amino_acids(atoms)
    protein_atoms = atoms[protein_mask]

    # Use same residue counting as geometry cache (dataset.py)
    chain_id_arr = protein_atoms.chain_id
    res_id_arr = protein_atoms.res_id
    ins_code_arr = np.array(
        [normalize_ins_code(x) for x in protein_atoms.ins_code], dtype=object
    )

    # Build unique residue keys in order of first occurrence, then get res_name for each
    residue_keys = list(zip(chain_id_arr, res_id_arr, ins_code_arr))
    unique_res_keys = list(dict.fromkeys(residue_keys))  # preserves order

    # Map residue keys to their 3-letter residue names
    key_to_resname = {}
    for i, key in enumerate(residue_keys):
        if key not in key_to_resname:
            key_to_resname[key] = protein_atoms.res_name[i]

    biotite_seq = [
        THREE_TO_ONE.get(key_to_resname[key], "X") for key in unique_res_keys
    ]

    return biotite_seq, len(biotite_seq)


def align_esm_to_biotite(
    esm_emb: torch.Tensor,
    esm_seq: list[str],
    biotite_seq: list[str],
) -> torch.Tensor:
    """
    Align ESM embeddings to biotite residue list.

    ESM may skip non-canonical residues (SEC, PYL) or map others (MSE->M).
    This function inserts zero vectors for any residues ESM skipped.

    Args:
        esm_emb: (N_esm, embed_dim) ESM embeddings
        esm_seq: ESM's sequence (single-letter codes, no chain breaks)
        biotite_seq: Biotite's sequence (X for non-canonical)

    Returns:
        aligned_emb: (N_biotite, embed_dim) aligned embeddings
    """
    num_biotite = len(biotite_seq)
    embed_dim = esm_emb.shape[1]
    aligned_emb = torch.zeros(num_biotite, embed_dim, dtype=esm_emb.dtype)

    esm_idx = 0
    for bt_idx, bt_aa in enumerate(biotite_seq):
        if bt_aa == "X":
            # Non-canonical residue - ESM skipped it, leave as zeros (UNK)
            continue

        if esm_idx >= len(esm_seq):
            # ESM ran out of residues - remaining are zeros
            break

        # ESM may have mapped MSE->M, so biotite might have X where ESM has M
        # In that case we already handled it above (bt_aa == 'X')
        # Here we just copy the ESM embedding
        aligned_emb[bt_idx] = esm_emb[esm_idx]
        esm_idx += 1

    return aligned_emb


def compute_esm_embeddings(
    pdb_path: str,
    model: ESM3,
) -> dict | None:
    """
    Compute ESM3 residue-level embeddings aligned to biotite structure.

    Uses biotite as ground truth for residue count. ESM embeddings are
    aligned to match, with zero vectors for any non-canonical residues
    that ESM skipped.
    """
    try:
        # 1. Get ground truth residue list from biotite
        biotite_seq, num_residues_biotite = get_biotite_residues(pdb_path)

        # 2. Run ESM (may skip some residues like SEC, PYL)
        complex_obj = ProteinComplex.from_pdb(pdb_path)
        protein = ESMProtein.from_protein_complex(complex_obj)

        with torch.no_grad():
            protein_tensor = model.encode(protein)
            output = model.logits(
                protein_tensor,
                LogitsConfig(return_embeddings=True),
            )

        if output.embeddings is None:
            raise RuntimeError("Model returned no embeddings")

        emb = output.embeddings
        if emb.dim() == 3:
            emb = emb.squeeze(0)
        emb = emb[1:-1].detach().cpu()  # remove BOS/EOS

        # Remove chain breaks from ESM embeddings
        esm_seq_with_breaks = protein.sequence or ""
        is_chainbreak = torch.tensor(
            [aa == "|" for aa in esm_seq_with_breaks], dtype=torch.bool
        )
        esm_emb = emb[~is_chainbreak]
        esm_seq = [aa for aa in esm_seq_with_breaks if aa != "|"]

        # 3. Align ESM embeddings to biotite structure
        aligned_emb = align_esm_to_biotite(esm_emb, esm_seq, biotite_seq)

        return {
            "residue_embeddings": aligned_emb,
            "sequence": "".join(biotite_seq),
            "num_residues": num_residues_biotite,
        }

    except Exception:
        logger.exception(f"Error computing embeddings for {pdb_path}")
        return None


def main() -> None:
    setup_logging_for_tqdm()
    parser = argparse.ArgumentParser(
        description="Generate ESM3 embeddings for protein structures"
    )
    parser.add_argument(
        "--split_file",
        type=str,
        required=True,
        help="Text file with PDB entries (one per line, e.g., '6eey_final')",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Base cache directory; embeddings saved to {cache_dir}/esm/",
    )
    parser.add_argument(
        "--base_pdb_dir",
        type=str,
        default="/sb/wankowicz_lab/data/srivasv/pdb_redo_data",
        help="Base directory containing PDB subdirectories",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use for ESM model"
    )
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing embeddings"
    )

    args = parser.parse_args()

    base_pdb_dir = Path(args.base_pdb_dir)
    cache_dir = Path(args.cache_dir)
    esm_cache_dir = cache_dir / "esm"
    esm_cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Loading ESM3 model...")
    model = ESM3.from_pretrained("esm3-open").to(device)
    model.eval()
    logger.info("Model loaded.")

    entries = parse_split_file(args.split_file, base_pdb_dir)
    logger.info(f"Found {len(entries)} entries in split file")

    if args.batch_limit:
        entries = entries[: args.batch_limit]
        logger.info(f"Processing first {args.batch_limit} entries")

    if not args.overwrite:
        to_process = []
        for entry in entries:
            cache_path = esm_cache_dir / f"{entry['cache_key']}.pt"
            if not cache_path.exists():
                to_process.append(entry)
        logger.info(f"{len(to_process)} entries need ESM embeddings")
        entries = to_process

    if not entries:
        logger.info("No entries to process. Done.")
        return

    success_count = 0
    failures = []

    for entry in tqdm(entries, desc="Computing ESM embeddings"):
        pdb_path = entry["pdb_path"]
        cache_key = entry["cache_key"]
        cache_path = esm_cache_dir / f"{cache_key}.pt"

        if not pdb_path.exists():
            logger.warning(f"PDB file not found: {pdb_path}")
            failures.append((cache_key, "PDB file not found"))
            continue

        result = compute_esm_embeddings(
            str(pdb_path),
            model,
        )

        if result is not None:
            result["pdb_id"] = entry["pdb_id"]
            torch.save(result, cache_path)
            success_count += 1
        else:
            failures.append((cache_key, "Embedding computation failed"))

    logger.info(
        f"Completed: {success_count}/{len(entries)} entries processed successfully"
    )

    # Log failures
    if failures:
        failure_log_path = esm_cache_dir / "embedding_failures.log"
        with open(failure_log_path, "a") as f:
            for cache_key, reason in failures:
                f.write(f"{cache_key}\t{reason}\n")
        logger.info(f"Logged {len(failures)} failures to {failure_log_path}")


if __name__ == "__main__":
    main()
