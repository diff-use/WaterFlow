"""
Precompute SLAE embeddings for protein structures and save to separate cache files.

This script:
1. Reads a split file containing PDB entries
2. For each entry, loads the PDB and converts to atom37 representation
3. Runs SLAE encoder to get atom-level embeddings
4. Aligns embeddings to match geometry cache's atom order
5. Saves embeddings to separate .pt files in cache_dir/slae/

SLAE only produces embeddings for atoms with canonical names (37 standard atom
types). Non-canonical atoms (e.g., from modified residues like 65T, DAL, MLE)
get zero vectors to maintain alignment with the geometry cache.

Usage:
    uv run python -m scripts.generate_slae_embeddings \
        --split_file /path/to/split.txt \
        --cache_dir /path/to/cache \
        --base_pdb_dir /path/to/pdbs \
        [--slae_ckpt /path/to/checkpoint] \
        [--slae_config /path/to/config]

Output format (per cache file):
    {
        'node_embeddings': Tensor (N_geometry_atoms, 128),  # aligned atom embeddings
        'atom37_coords': Tensor (N_res, 37, 3),             # for reference
        'pdb_id': str,
    }
"""
import argparse
import sys
from collections.abc import Iterable
from itertools import batched
from pathlib import Path

import biotite.structure as bts

# TODO: Create SLAE as an installable site-package and have the user install it instead of sys path hacks (pending approval from SLAE authors)

sys.path.insert(0, str(Path(__file__).parent.parent))
# SLAE paths - derive from home directory for portability
SLAE_ROOT = Path.home() / "SLAE_wl"
sys.path.insert(0, str(SLAE_ROOT))  # For SLAE imports
# Default paths for SLAE config and checkpoint
DEFAULT_SLAE_CONFIG = (
    SLAE_ROOT / "SLAE" / "configs" / "encoder" / "protein_encoder.yaml"
)
DEFAULT_SLAE_CKPT = SLAE_ROOT / "checkpoints" / "autoencoder.ckpt"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer  # noqa: E402
from SLAE.io.atom_tensor import atom37_to_atoms, atomarray_to_tensors  # noqa: E402
from SLAE.model.encoder import ProteinEncoder  # noqa: E402
from SLAE.util.constants import PROTEIN_ATOMS  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.dataset import parse_asu_with_biotite  # noqa: E402
from src.utils import (  # noqa: E402
    normalize_ins_code,
    parse_split_file,
    setup_logging_for_tqdm,
)


def load_yaml_dict(config_path: str | Path) -> dict[str, object]:
    """Load YAML config and return as dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config.pop("_target_", None)
    return config


def get_geometry_atom_info(
    protein_atoms: bts.AtomArray,
) -> list[tuple[str, int, str, str]]:
    """
    Build list of (chain_id, res_id, ins_code, atom_name) for each atom in geometry order.

    Geometry order refers to the atom ordering used in our geometry cache (dataset.py),
    which preserves the order atoms appear in the PDB file. This differs from SLAE's
    internal ordering (which may sort residues). We need this mapping to align SLAE
    embeddings back to our geometry representation.

    Args:
        protein_atoms: biotite AtomArray

    Returns:
        List of (chain_id, res_id, ins_code, atom_name) tuples, one per atom
    """
    chains = protein_atoms.chain_id
    res_ids = protein_atoms.res_id
    ins_codes = np.array([normalize_ins_code(x) for x in protein_atoms.ins_code])
    atom_names = [str(name).strip().upper() for name in protein_atoms.atom_name]

    return list(zip(chains, res_ids, ins_codes, atom_names, strict=True))


def align_slae_to_geometry(
    slae_emb: torch.Tensor,
    slae_residue_idx: torch.Tensor,
    slae_atom_type: torch.Tensor,
    slae_chains: np.ndarray,
    slae_residue_ids: torch.Tensor,
    slae_ins_codes: np.ndarray,
    geometry_atom_info: list[tuple[str, int, str, str]],
) -> torch.Tensor:
    """
    Align SLAE atom embeddings to match geometry's atom order.

    SLAE only produces embeddings for atoms with canonical names (37 standard atom
    types). This function creates output embeddings that match geometry's atom
    order, with zero vectors for atoms SLAE doesn't have.

    Uses full atom identity (chain_id, res_id, ins_code, atom_name) for matching
    to handle different residue indexing between geometry (first occurrence order)
    and SLAE (sorted order).

    Args:
        slae_emb: (N_slae, 128) SLAE embeddings for canonical atoms
        slae_residue_idx: (N_slae,) residue index for each SLAE atom
        slae_atom_type: (N_slae,) atom type index (0-36) for each SLAE atom
        slae_chains: (N_res,) chain IDs from atomarray_to_tensors
        slae_residue_ids: (N_res,) residue IDs from atomarray_to_tensors
        slae_ins_codes: (N_res,) insertion codes from atomarray_to_tensors
        geometry_atom_info: List of (chain_id, res_id, ins_code, atom_name) for geometry atoms

    Returns:
        aligned_emb: (N_geometry, 128) embeddings aligned to geometry order
    """
    n_geometry = len(geometry_atom_info)
    n_slae = slae_emb.shape[0]

    # Build SLAE atom keys 
    res_indices = slae_residue_idx.numpy()
    atom_type_indices = slae_atom_type.numpy()

    # Gather per-atom info from per-residue arrays
    slae_atom_chains = slae_chains[res_indices]
    slae_atom_res_ids = slae_residue_ids.numpy()[res_indices]
    slae_atom_ins_codes = np.array([normalize_ins_code(slae_ins_codes[r]) for r in res_indices])
    slae_atom_names = np.array([PROTEIN_ATOMS[t] for t in atom_type_indices])

    # Build lookup: (chain_id, res_id, ins_code, atom_name) -> slae_atom_index
    slae_lookup = {
        (slae_atom_chains[i], slae_atom_res_ids[i], slae_atom_ins_codes[i], slae_atom_names[i]): i
        for i in range(n_slae)
    }

    # Build index array: geometry_idx -> slae_idx (or -1 if not found)
    index_array = np.full(n_geometry, -1, dtype=np.int64)
    for geom_idx, key in enumerate(geometry_atom_info):
        if key in slae_lookup:
            index_array[geom_idx] = slae_lookup[key]

    # Vectorized assignment: use mask for valid indices
    valid_mask = index_array >= 0
    aligned = torch.zeros(n_geometry, slae_emb.shape[1], dtype=slae_emb.dtype)
    if valid_mask.any():
        aligned[valid_mask] = slae_emb[index_array[valid_mask]]

    return aligned


def compute_slae_embeddings_batch(
    coords_list: Iterable[torch.Tensor],
    residue_type_list: Iterable[torch.Tensor],
    residue_id_list: Iterable[torch.Tensor],
    chains_list: Iterable[np.ndarray],
    ins_code_list: Iterable[np.ndarray],
    geometry_atom_info_list: Iterable[list[tuple[str, int, str, str]]],
    encoder: ProteinEncoder,
    featurizer: ProteinGraphFeaturizer,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Compute SLAE node embeddings from atom37 coordinates for a batch of structures.

    Embeddings are aligned to match geometry's atom order, with zero vectors for
    atoms that SLAE doesn't produce embeddings for (non-canonical atoms).

    Args:
        coords_list: List of (N_res, 37, 3) atom37 coordinates tensors
        residue_type_list: List of (N_res,) residue type indices tensors
        residue_id_list: List of (N_res,) residue ID tensors
        chains_list: List of (N_res,) chain ID arrays from atomarray_to_tensors
        ins_code_list: List of (N_res,) insertion code arrays from atomarray_to_tensors
        geometry_atom_info_list: List of geometry atom info (from get_geometry_atom_info)
        encoder: SLAE ProteinEncoder
        featurizer: ProteinGraphFeaturizer
        device: torch device

    Returns:
        embeddings_list: List of (N_geometry_atoms, 128) aligned node embeddings tensors
    """
    # create Data objects for all structures
    data_list = []
    slae_atom_info_list = []  # Store (residue_idx, atom_type, chains, residue_id, ins_code) for alignment

    for coords, residue_type, residue_id, chains, ins_code in zip(
        coords_list, residue_type_list, residue_id_list, chains_list, ins_code_list,
        strict=True,
    ):
        # create PyG Data object with atom37 coords (featurizer will convert to flat)
        data = Data(
            coords=coords,
            residue_type=residue_type,
            residue_id=residue_id,
        )
        data_list.append(data)

        # Get SLAE atom info for later alignment
        _, residue_idx, atom_type = atom37_to_atoms(coords)
        slae_atom_info_list.append(
            (residue_idx, atom_type, chains, residue_id, ins_code)
        )

    batch = Batch.from_data_list(data_list)
    batch = batch.to(device)
    batch = featurizer(batch)

    with torch.no_grad():
        outputs = encoder(batch)
        embeddings = outputs["node_embedding"]  # (total_atoms, 128)

    # split embeddings back into individual structures and align to geometry
    embeddings_list = []
    start_idx = 0
    for (
        residue_idx,
        atom_type,
        chains,
        residue_id,
        ins_code,
    ), geometry_atom_info in zip(slae_atom_info_list, geometry_atom_info_list, strict=True):
        num_slae_atoms = residue_idx.size(0)
        slae_emb = embeddings[start_idx : start_idx + num_slae_atoms].cpu()

        # Align SLAE embeddings to geometry atom order
        aligned_emb = align_slae_to_geometry(
            slae_emb,
            residue_idx,
            atom_type,
            chains,
            residue_id,
            ins_code,
            geometry_atom_info,
        )
        embeddings_list.append(aligned_emb)
        start_idx += num_slae_atoms

    return embeddings_list


def main() -> None:
    setup_logging_for_tqdm()
    parser = argparse.ArgumentParser(
        description="Precompute SLAE embeddings for protein structures"
    )
    parser.add_argument(
        "--split_file",
        type=Path,
        required=True,
        help="Text file with PDB entries (one per line, e.g., '6eey_final')",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        required=True,
        help="Base cache directory; embeddings saved to {cache_dir}/slae/",
    )
    parser.add_argument(
        "--base_pdb_dir",
        type=Path,
        default=Path("/sb/wankowicz_lab/data/srivasv/pdb_redo_data"),
        help="Base directory containing PDB subdirectories",
    )
    parser.add_argument(
        "--slae_ckpt",
        type=Path,
        default=DEFAULT_SLAE_CKPT,
        help="Path to SLAE checkpoint",
    )
    parser.add_argument(
        "--slae_config",
        type=Path,
        default=DEFAULT_SLAE_CONFIG,
        help="Path to SLAE encoder config",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of structures to process in each batch",
    )

    args = parser.parse_args()

    slae_cache_dir = args.cache_dir / "slae"
    slae_cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # load SLAE encoder
    logger.info("Loading SLAE encoder...")
    enc_config = load_yaml_dict(args.slae_config)
    encoder = ProteinEncoder(**enc_config).to(device).eval()

    ckpt = torch.load(args.slae_ckpt, map_location="cpu", weights_only=False)
    encoder.load_state_dict(ckpt["encoder"], strict=False)

    for p in encoder.parameters():
        p.requires_grad = False

    logger.info(f"Loaded encoder from {args.slae_ckpt}")

    # create featurizer
    featurizer = ProteinGraphFeaturizer(radius=8.0, use_atom37=True)

    # parse split file
    entries = parse_split_file(args.split_file, args.base_pdb_dir)
    logger.info(f"Found {len(entries)} entries in split file")

    if args.batch_limit:
        entries = entries[: args.batch_limit]
        logger.info(f"Processing first {args.batch_limit} entries")

    # filter entries that need processing
    if not args.overwrite:
        to_process = []
        for entry in entries:
            cache_path = slae_cache_dir / f"{entry['cache_key']}.pt"
            if not cache_path.exists():
                to_process.append(entry)
        logger.info(f"{len(to_process)} entries need SLAE embeddings")
        entries = to_process

    if not entries:
        logger.info("No entries to process. Done.")
        return

    # process entries in batches
    success_count = 0
    failures = []
    total_batches = (len(entries) + args.batch_size - 1) // args.batch_size

    for batch_idx, entry_batch in enumerate(
        tqdm(
            list(batched(entries, args.batch_size)),
            desc="Computing SLAE embeddings",
            total=total_batches,
        )
    ):
        batch_data = []
        batch_info = []

        for entry in entry_batch:
            pdb_path = entry["pdb_path"]
            cache_key = entry["cache_key"]

            if not pdb_path.exists():
                logger.error(f"PDB file not found: {pdb_path}")
                failures.append((cache_key, "PDB file not found"))
                continue

            try:
                protein_atoms, _ = parse_asu_with_biotite(str(pdb_path))
                coords, residue_type, chains, residue_id, ins_code = (
                    atomarray_to_tensors(protein_atoms)
                )
                # Get geometry atom info for alignment
                geometry_atom_info = get_geometry_atom_info(protein_atoms)
                batch_data.append(
                    (
                        coords,
                        residue_type,
                        residue_id,
                        chains,
                        ins_code,
                        geometry_atom_info,
                    )
                )
                batch_info.append(
                    {
                        "entry": entry,
                        "coords": coords,
                        "n_geometry_atoms": len(geometry_atom_info),
                    }
                )
            except Exception as e:
                logger.exception(f"Error preparing {cache_key}")
                failures.append((cache_key, str(e)))

        # compute embeddings for batch
        if batch_data:
            try:
                (coords_list, 
                 residue_type_list, 
                 residue_id_list, 
                 chains_list, 
                 ins_code_list, 
                 geometry_atom_info_list
                 ) = zip(*batch_data)

                embeddings_list = compute_slae_embeddings_batch(
                    coords_list,
                    residue_type_list,
                    residue_id_list,
                    chains_list,
                    ins_code_list,
                    geometry_atom_info_list,
                    encoder,
                    featurizer,
                    device,
                )

                # save each structure's embeddings
                for info, embeddings in zip(batch_info, embeddings_list, strict=True):
                    cache_path = slae_cache_dir / f"{info['entry']['cache_key']}.pt"

                    result = {
                        "node_embeddings": embeddings,
                        "atom37_coords": info["coords"],
                        "pdb_id": info["entry"]["pdb_id"],
                    }

                    torch.save(result, cache_path)
                    success_count += 1

            except Exception as e:
                logger.exception(f"Error processing batch {batch_idx}")
                for info in batch_info:
                    failures.append((info["entry"]["cache_key"], str(e)))

    logger.info(
        f"Completed: {success_count}/{len(entries)} entries processed successfully"
    )

    if failures:
        failure_log_path = slae_cache_dir / "embedding_failures.log"
        with open(failure_log_path, "a") as f:
            for cache_key, reason in failures:
                f.write(f"{cache_key}\t{reason}\n")
        logger.error(f"Logged {len(failures)} failures to {failure_log_path}")


if __name__ == "__main__":
    main()
