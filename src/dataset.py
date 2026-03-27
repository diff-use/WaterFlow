"""
Dataset utilities for protein-water structure loading and preprocessing.

This module provides:
- PDB parsing with biotite and PyMOL for crystal contacts
- Per-water quality filtering (distance, EDIA, B-factor)
- Structure-level quality checks (CoM distance, clashes, chain interactions)
- ProteinWaterDataset: PyTorch Dataset returning HeteroData graphs
- get_dataloader: Convenience function for DataLoader creation
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import biotite.structure as bts
import numpy as np
import pymol2
import torch
import torch.nn.functional as F
from biotite.structure.io.pdb import get_structure, PDBFile
from loguru import logger
from scipy.spatial.distance import cdist
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_cluster import radius_graph
from torch_geometric.data import Batch, HeteroData
from tqdm import tqdm

from src.constants import EDGE_PP, ELEM_IDX, ELEMENT_VOCAB, NUM_RBF
from src.utils import (
    compute_edge_features,
    normalize_ins_code,
)


def element_onehot(symbols: list[str]) -> Tensor:
    """One-hot encoding with 'other' bucket at end."""
    other_idx = len(ELEMENT_VOCAB)
    indices = torch.tensor(
        [ELEM_IDX.get(s.upper(), other_idx) for s in symbols], dtype=torch.long
    )
    return F.one_hot(indices, num_classes=other_idx + 1).float()


def parse_asu_with_biotite(
    path: str,
) -> tuple[bts.AtomArray, bts.AtomArray]:
    """
    Parse PDB file and extract protein and water atoms.

    Args:
        path: Path to PDB file

    Returns:
        Tuple of (protein_atoms, water_atoms) as biotite AtomArrays.
        Hydrogen atoms are excluded.

    Notes:
        - model=1: Uses first model in PDB (standard for X-ray structures)
        - altloc="occupancy": Selects highest-occupancy alternate conformation
        - Uses filter_amino_acids (not filter_canonical_amino_acids) to include
          modified residues like MSE, SEC that external encoders may handle
    """
    pdb_file = PDBFile.read(path)
    atoms = get_structure(pdb_file, model=1, altloc="occupancy")

    atoms = atoms[atoms.element != "H"]

    protein_mask = bts.filter_amino_acids(atoms)
    water_mask = (atoms.res_name == "HOH") | (atoms.res_name == "WAT")

    protein_atoms = atoms[protein_mask]
    water_atoms = atoms[water_mask]

    return protein_atoms, water_atoms


def get_crystal_contacts_pymol(
    pdb_path: str, cutoff: float = 5.0
) -> dict[str, np.ndarray | list]:
    """
    Extract ASU and symmetry mate atoms within crystal contact distance.

    Uses PyMOL's symexp command to generate symmetry mates and selects
    interface atoms within the specified cutoff distance.

    Args:
        pdb_path: Path to PDB file with crystal symmetry information
        cutoff: Distance cutoff in Angstroms for interface detection

    Returns:
        Dict with keys:
            - 'asu_coords': (N_asu, 3) ASU atom coordinates
            - 'mate_coords': (N_mate, 3) symmetry mate atom coordinates
            - 'asu_atoms': List of PyMOL atom objects for ASU
            - 'mate_atoms': List of PyMOL atom objects for mates
    """
    with pymol2.PyMOL() as pm:
        cmd = pm.cmd
        cmd.reinitialize()
        cmd.feedback("disable", "all", "everything")
        obj = "struct"
        cmd.load(pdb_path, obj)
        cmd.symexp("sym", obj, obj, cutoff)
        cmd.select("interface", f"byres (sym* within {cutoff} of {obj})")

        asu_coords = cmd.get_coords(obj, state=1)
        mate_coords = cmd.get_coords("sym* and interface", state=1)
        asu_atoms = cmd.get_model(obj, state=1).atom
        mate_atoms = cmd.get_model("sym* and interface", state=1).atom

        asu_coords = (
            asu_coords if asu_coords is not None else np.zeros((0, 3), dtype=float)
        )
        mate_coords = (
            mate_coords if mate_coords is not None else np.zeros((0, 3), dtype=float)
        )
        return {
            "asu_coords": asu_coords,
            "mate_coords": mate_coords,
            "asu_atoms": asu_atoms,
            "mate_atoms": mate_atoms,
        }


def match_atoms_to_coords(
    atoms: bts.AtomArray, target_coords: np.ndarray, tolerance: float = 0.01
) -> list[int]:
    """
    Match biotite atoms to target coordinates by nearest neighbor. (needed for mates when parsing with PyMOL)

    Args:
        atoms: Biotite AtomArray with coord attribute
        target_coords: (N, 3) array of target coordinates to match
        tolerance: Maximum distance in Angstroms for a valid match

    Returns:
        List of indices into atoms array for matched atoms
    """
    if target_coords.shape[0] == 0:
        return []

    matched = []
    for i, coord in enumerate(target_coords):
        dists = np.linalg.norm(atoms.coord - coord, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < tolerance:
            matched.append(min_idx)
    return matched


def _make_undirected(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Convert directed edges to undirected by adding reverse edges.

    Args:
        edge_index: (2, E) directed edge index tensor

    Returns:
        (2, E') undirected edge index with reverse edges added and duplicates removed
    """
    if edge_index.numel() == 0:
        return edge_index
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # add reverse edges
    ei = torch.unique(ei.T, dim=0).T  # drop duplicates
    return ei


def _pad_atom_embeddings_for_mates(
    asu_embedding: torch.Tensor,
    total_num_atoms: int,
) -> torch.Tensor:
    """
    Pad ASU-only atom embeddings with zeros for symmetry mate atoms.

    Args:
        asu_embedding: (N_asu, embed_dim) embeddings for ASU atoms only
        total_num_atoms: Total number of atoms including symmetry mates

    Returns:
        (total_num_atoms, embed_dim) padded embeddings with zeros for mate atoms
    """
    if total_num_atoms <= asu_embedding.size(0):
        return asu_embedding
    pad = asu_embedding.new_zeros(
        total_num_atoms - asu_embedding.size(0), asu_embedding.size(1)
    )
    return torch.cat([asu_embedding, pad], dim=0)


def load_slae_embedding(
    embedding_dir: Path,
    cache_key: str,
    num_asu_protein: int,
    total_num_atoms: int,
) -> torch.Tensor:
    """
    Load SLAE atom-level embeddings from cache.

    This is a standalone function to allow reuse outside dataset context.

    Args:
        embedding_dir: Directory containing cached embedding files
        cache_key: Identifier for the cached embedding file
        num_asu_protein: Expected number of ASU protein atoms
        total_num_atoms: Total protein atoms including symmetry mates

    Returns:
        (total_num_atoms, slae_dim) tensor with zeros padded for mate atoms

    Raises:
        FileNotFoundError: If SLAE cache file doesn't exist
        ValueError: If atom count doesn't match expected ASU count
    """
    slae_cache_path = embedding_dir / f"{cache_key}.pt"
    if not slae_cache_path.exists():
        raise FileNotFoundError(
            f"SLAE cache file not found: {slae_cache_path}. "
            "Generate embeddings with scripts/generate_slae_embeddings.py."
        )
    slae_cached = torch.load(slae_cache_path, weights_only=False)
    if "node_embeddings" not in slae_cached:
        raise KeyError(f"Missing 'node_embeddings' in SLAE cache: {slae_cache_path}")
    slae_emb = slae_cached["node_embeddings"]
    if slae_emb.size(0) != num_asu_protein:
        raise ValueError(
            f"SLAE embedding atom count mismatch for {cache_key}: "
            f"expected {num_asu_protein}, got {slae_emb.size(0)}"
        )
    return _pad_atom_embeddings_for_mates(slae_emb, total_num_atoms)


def load_esm_embedding(
    embedding_dir: Path,
    cache_key: str,
    num_protein_residues: int,
) -> torch.Tensor:
    """
    Load ESM residue-level embeddings from cache.

    This is a standalone function to allow reuse outside dataset context.
    Returns raw residue embeddings; broadcasting to atom level is done separately.

    Args:
        embedding_dir: Directory containing cached embedding files
        cache_key: Identifier for the cached embedding file
        num_protein_residues: Expected number of unique residues

    Returns:
        (num_protein_residues, esm_dim) tensor of residue embeddings

    Raises:
        FileNotFoundError: If ESM cache file doesn't exist
        ValueError: If residue count doesn't match expected count
    """
    esm_cache_path = embedding_dir / f"{cache_key}.pt"
    if not esm_cache_path.exists():
        raise FileNotFoundError(
            f"ESM cache file not found: {esm_cache_path}. "
            "Generate embeddings with scripts/generate_esm_embeddings.py."
        )
    esm_cached = torch.load(esm_cache_path, weights_only=False)
    if "residue_embeddings" not in esm_cached:
        raise KeyError(f"Missing 'residue_embeddings' in ESM cache: {esm_cache_path}")
    residue_embeddings = esm_cached["residue_embeddings"]
    if residue_embeddings.size(0) != num_protein_residues:
        raise ValueError(
            f"ESM residue count mismatch for {cache_key}: "
            f"expected {num_protein_residues}, got {residue_embeddings.size(0)}"
        )
    return residue_embeddings


def check_com_distance(
    protein_coords: torch.Tensor,
    water_coords: torch.Tensor,
    max_com_dist: float = 25.0,
) -> tuple[bool, str]:
    """
    Check if protein and water centers of mass are within acceptable distance.

    Large CoM differences indicate atoms are in different frames of reference.

    Args:
        protein_coords: (N, 3) protein atom coordinates
        water_coords: (M, 3) water atom coordinates
        max_com_dist: Maximum allowed distance between CoMs (Angstroms)

    Returns:
        (is_valid, reason) tuple
    """
    if water_coords.size(0) == 0:
        return True, ""

    protein_com = protein_coords.mean(dim=0)
    water_com = water_coords.mean(dim=0)
    com_dist = torch.linalg.norm(protein_com - water_com).item()

    if com_dist > max_com_dist:
        return False, f"CoM distance {com_dist:.1f}A exceeds threshold {max_com_dist}A"
    return True, ""


def check_water_clashes(
    protein_coords: torch.Tensor,
    water_coords: torch.Tensor,
    clash_dist: float = 2.0,
    max_clash_fraction: float = 0.05,
) -> tuple[bool, str]:
    """
    Check if too many waters clash with the protein surface (within a threshold).

    Waters within clash_dist of any protein atom are considered clashing.

    Args:
        protein_coords: (N, 3) protein atom coordinates
        water_coords: (M, 3) water atom coordinates
        clash_dist: Distance threshold for clash detection (Angstroms)
        max_clash_fraction: Maximum allowed fraction of clashing waters (0-1)

    Returns:
        (is_valid, reason) tuple
    """
    if water_coords.size(0) == 0:
        return True, ""

    # compute pairwise distances: (M, N)
    dists = torch.cdist(water_coords, protein_coords)
    min_dists = dists.min(dim=1).values  # closest protein atom to each water

    n_clashing = (min_dists < clash_dist).sum().item()
    clash_fraction = n_clashing / water_coords.size(0)

    if clash_fraction > max_clash_fraction:
        return False, (
            f"Water clash fraction {clash_fraction:.1%} ({n_clashing}/{water_coords.size(0)}) "
            f"exceeds threshold {max_clash_fraction:.0%}"
        )
    return True, ""


def check_chain_interactions(
    protein_atoms: bts.AtomArray,
    interface_dist_threshold: float = 4.0,
) -> tuple[bool, str, str]:
    """
    Check if multi-chain proteins have interacting chains (PPI) vs ASU copies.

    For proteins with >=2 chains, computes minimum inter-chain distance.
    If min distance > threshold, chains are likely ASU copies, not a true PPI.

    Args:
        protein_atoms: biotite AtomArray with chain_id and coord attributes
        interface_dist_threshold: Chains must be within this distance to be
            considered interacting (Angstroms)

    Returns:
        (is_valid, reason, interaction_status) tuple where interaction_status
        is one of: "Single Chain", "Interacting", "Non-Interacting (ASU Copies)"
    """
    chain_ids = np.unique(protein_atoms.chain_id)
    num_chains = len(chain_ids)

    if num_chains < 2:
        return True, "", "Single Chain"

    chain_coords = {
        cid: torch.tensor(
            protein_atoms[protein_atoms.chain_id == cid].coord, dtype=torch.float32
        )
        for cid in chain_ids
    }

    min_interface_dist = float("inf")
    for chain_a, chain_b in itertools.combinations(chain_ids, 2):
        coords_a = chain_coords[chain_a]
        coords_b = chain_coords[chain_b]
        min_d = torch.cdist(coords_a, coords_b).min().item()
        if min_d < min_interface_dist:
            min_interface_dist = min_d

    if min_interface_dist > interface_dist_threshold:
        return (
            False,
            f"Multi-chain ({num_chains} chains) min interface distance {min_interface_dist:.1f}A "
            f"> {interface_dist_threshold}A (likely ASU copies, not PPI)",
            "Non-Interacting (ASU Copies)",
        )

    return True, "", "Interacting"


def check_water_residue_ratio(
    num_waters: int,
    num_residues: int,
    min_ratio: float = 0.8,
) -> tuple[bool, str]:
    """
    Check if water/residue ratio meets minimum threshold.

    Structures with too few waters relative to protein size may be
    poorly resolved or have incomplete solvent modeling.

    Args:
        num_waters: Number of water molecules
        num_residues: Number of protein residues
        min_ratio: Minimum required waters/residues ratio

    Returns:
        (is_valid, reason) tuple
    """
    if num_residues == 0:
        return False, "No residues found"

    ratio = num_waters / num_residues

    if ratio < min_ratio:
        return False, (
            f"Water/residue ratio {ratio:.2f} ({num_waters}/{num_residues}) "
            f"below threshold {min_ratio}"
        )
    return True, ""


def load_edia_for_pdb(
    json_path: str | Path,
) -> dict[tuple[str, int, str], float] | None:
    """
    Load EDIA scores for water molecules from a JSON file.

    Args:
        json_path: Path to JSON file containing EDIA scores for the structure

    Returns:
        Dictionary mapping (chain_id, res_id, ins_code) -> EDIA score for waters,
        or None if file not found or error
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        edia_lookup = {}
        for entry in data:
            # Filter for water molecules only
            if entry.get("compID") in ["HOH", "WAT"]:
                # The identifying information is nested inside the "pdb" key in the JSON
                pdb_info = entry.get("pdb", {})

                chain_id = str(pdb_info.get("strandID", ""))
                res_id = int(pdb_info.get("seqNum", 0))

                # Extract and normalize insertion code, defaulting to an empty string
                raw_ins_code = pdb_info.get("insCode", "")
                ins_code = normalize_ins_code(raw_ins_code) if raw_ins_code else ""

                # Build the lookup key and extract the EDIAm score
                key = (chain_id, res_id, ins_code)
                edia_lookup[key] = float(entry.get("EDIAm", 0.0))

        if not edia_lookup:
            return {}

        return edia_lookup

    except Exception as e:
        logger.warning(f"Warning: Could not load EDIA JSON data for {json_path}: {e}")
        return None


def compute_normalized_bfactors(
    pdb_path: str,
) -> tuple[dict[tuple[str, int, str], float] | None, np.ndarray | None]:
    """
    Extract and normalize B-factors for water molecules.

    B-factors are z-score normalized using statistics from water atoms only
    in the selected structure.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Tuple of:
        - Dictionary mapping (chain_id, res_id, ins_code) -> normalized B-factor for waters
        - Raw B-factor array for waters (for caching if needed)
        Returns (None, None) on error
    """
    try:
        pdb_file = PDBFile.read(pdb_path)
        atoms = pdb_file.get_structure(
            model=1, altloc="occupancy", extra_fields=["b_factor"]
        )

        # filter for water molecules
        water_mask = (atoms.res_name == "HOH") | (atoms.res_name == "WAT")
        water_atoms = atoms[water_mask]

        if not water_atoms:
            return None, None

        # Normalize using water-only B-factor statistics.
        water_mean = np.mean(water_atoms.b_factor)
        water_std = np.std(water_atoms.b_factor)

        # lookup dictionary with one entry per unique water residue
        bfactor_lookup = {}

        for i in range(len(water_atoms)):
            chain_id = str(water_atoms.chain_id[i])
            res_id = int(water_atoms.res_id[i])
            ins_code = normalize_ins_code(water_atoms.ins_code[i])
            key = (chain_id, res_id, ins_code)

            if key not in bfactor_lookup:
                raw_bfactor = water_atoms.b_factor[i]
                # If all water B-factors are identical, assign neutral z-score 0.0.
                normalized = (
                    (raw_bfactor - water_mean) / max(water_std, 1e-3)
                    if water_std > 0
                    else 0.0
                )
                bfactor_lookup[key] = normalized

        return bfactor_lookup, water_atoms.b_factor

    except Exception as e:
        logger.warning(f"Warning: Could not extract B-factors from {pdb_path}: {e}")
        return None, None


def apply_threshold_filter(
    water_keys: list[tuple],
    lookup: dict[tuple, float],
    threshold: float,
    fail_if_below: bool,
) -> np.ndarray:
    """
    Apply a threshold filter using a lookup dictionary.

    Args:
        water_keys: List of per-water residue keys
        lookup: Dict mapping residue key -> value
        threshold: Threshold value for comparison
        fail_if_below: If True, fail when value < threshold (e.g., EDIA).
                       If False, fail when value > threshold (e.g., B-factor).

    Returns:
        Boolean mask where True indicates the water FAILS the filter.
        Waters missing from lookup get np.nan and pass the filter (conservative).
    """
    values = np.array([lookup.get(key, np.nan) for key in water_keys])
    if fail_if_below:
        return values < threshold
    return values > threshold


def filter_waters_by_quality(
    water_coords: np.ndarray,
    water_keys: list[tuple],
    protein_coords: np.ndarray | None,
    edia_lookup: dict[tuple, float] | None,
    bfactor_lookup: dict[tuple, float] | None,
    max_protein_dist: float = 6.0,
    min_edia: float = 0.4,
    max_bfactor_zscore: float = 1.5,
    cache_key: str | None = None,
) -> np.ndarray:
    """
    Filter water atoms based on quality criteria.

    Waters are removed if they fail ANY of the enabled criteria:
    1. Distance from protein surface > max_protein_dist (if protein_coords provided)
    2. EDIA score < min_edia (if edia_lookup provided)
    3. Normalized B-factor > max_bfactor_zscore (if bfactor_lookup provided)

    Args:
        water_coords: (N, 3) array of water coordinates
        water_keys: List of per-water residue keys
        protein_coords: (M, 3) array of protein coordinates, or None to skip distance filtering
        edia_lookup: Dict mapping residue key -> EDIA score, or None to skip EDIA filtering
        bfactor_lookup: Dict mapping residue key -> normalized B-factor, or None to skip B-factor filtering
        max_protein_dist: Maximum allowed distance to protein surface
        min_edia: Minimum allowed EDIA score
        max_bfactor_zscore: Maximum allowed B-factor z-score
        cache_key: Optional identifier for logging (e.g., PDB ID)

    Returns:
        np.ndarray: Boolean mask of waters to keep (True = keep, False = remove)
    """
    n_waters = len(water_keys)

    if n_waters == 0:
        return np.array([], dtype=bool)

    stats = {
        "total": n_waters,
        "removed_distance": 0,
        "removed_edia": 0,
        "removed_bfactor": 0,
    }

    # distance filtering using scipy.spatial.distance.cdist
    dist_fail = np.zeros(n_waters, dtype=bool)
    if protein_coords is not None and len(protein_coords):
        dist_matrix = cdist(water_coords, protein_coords)
        min_dists = dist_matrix.min(axis=1)
        dist_fail = min_dists > max_protein_dist
        stats["removed_distance"] = int(dist_fail.sum())

    # lookup-based filters: (lookup, threshold, fail_if_below, stat_key)
    lookup_filters = [
        (edia_lookup, min_edia, True, "edia"),
        (bfactor_lookup, max_bfactor_zscore, False, "bfactor"),
    ]

    lookup_fail = np.zeros(n_waters, dtype=bool)
    for lookup, threshold, fail_if_below, name in lookup_filters:
        if lookup is not None:
            fail_mask = apply_threshold_filter(
                water_keys, lookup, threshold, fail_if_below
            )
            stats[f"removed_{name}"] = int(fail_mask.sum())
            lookup_fail |= fail_mask

    # combine all failure masks - water is kept only if it passes all enabled filters
    keep_mask = ~(dist_fail | lookup_fail)
    stats["kept"] = int(keep_mask.sum())

    # log filtering statistics
    if cache_key is not None and stats["total"] > 0:
        removed = stats["total"] - stats["kept"]
        if removed > 0:
            logger.info(
                f"  {cache_key}: Filtered {removed}/{stats['total']} waters "
                f"(dist:{stats['removed_distance']}, "
                f"edia:{stats['removed_edia']}, "
                f"bfactor:{stats['removed_bfactor']})"
            )

    return keep_mask


class ProteinWaterDataset(Dataset):
    """
    Dataset for protein crystal contact prediction.

    Returns HeteroData with:
    - 'protein' node type: ASU protein atoms + optionally symmetry mates
    - 'water' node type: water molecules
    - ('protein', 'pp', 'protein') edges
    """

    def __init__(
        self,
        pdb_list_file: str,
        processed_dir: str,
        encoder_type: str = "gvp",
        base_pdb_dir: str | None = None,
        cutoff: float = 8.0,
        include_mates: bool = True,
        geometry_cache_name: str = "geometry",
        preprocess: bool = True,
        duplicate_single_sample: int = 1,
        max_com_dist: float = 25.0,
        max_clash_fraction: float = 0.05,
        clash_dist: float = 2.0,
        interface_dist_threshold: float = 4.0,
        min_water_residue_ratio: float = 0.6,
        max_protein_dist: float = 5.0,
        min_edia: float = 0.4,
        max_bfactor_zscore: float = 1.5,
        filter_by_distance: bool = True,
        filter_by_edia: bool = True,
        filter_by_bfactor: bool = True,
    ):
        """
        Args:
            pdb_list_file: Text file with lines like "<pdb_id>_final"
            processed_dir: Cache root directory. Geometry caches are stored in
                           {processed_dir}/{geometry_cache_name}[_mates] and embedding
                           caches in {processed_dir}/{encoder_name}.
            encoder_type: Encoder used downstream ('gvp', 'slae', or 'esm').
                          Embeddings are loaded only for the selected type.
            base_pdb_dir: Base directory containing PDB subdirectories
            cutoff: Distance cutoff for PP edges and crystal contacts (Angstroms)
            include_mates: If True, include symmetry mate atoms as protein nodes
            geometry_cache_name: Base name for geometry cache directory. When
                                 include_mates=True, "_mates" is appended automatically.
                                 Default is "geometry", resulting in "geometry/" or
                                 "geometry_mates/" subdirectories.
            preprocess: If True, run preprocessing on missing cached files
            duplicate_single_sample: If dataset has 1 sample, duplicate it this many times
            Quality checks (always active):
            max_com_dist: Max allowed distance between protein and water CoM (Angstroms).
                          Structures exceeding this are filtered (different reference frames).
            max_clash_fraction: Max fraction of waters allowed within clash_dist of protein.
                                Structures exceeding this are filtered.
            clash_dist: Distance threshold for water-protein clashes (Angstroms).
            interface_dist_threshold: For multi-chain proteins, min inter-chain distance
                                      must be <= this to be considered interacting.
                                      Structures with larger distances are filtered (ASU copies).
            min_water_residue_ratio: Minimum ratio of waters/residues required.
                                     Structures below this are filtered (poor solvent modeling).

            Per-water filtering (toggleable):
            max_protein_dist: Remove waters farther than this from nearest protein atom (Angstroms).
            min_edia: Remove waters with EDIA score below this threshold.
            max_bfactor_zscore: Remove waters with normalized B-factor (z-score) above this.
            filter_by_distance: Enable/disable distance-from-protein filtering.
            filter_by_edia: Enable/disable EDIA score filtering.
            filter_by_bfactor: Enable/disable B-factor z-score filtering.
                              If a per-water filter is disabled, its threshold is ignored.
        """

        self.cache_dir = Path(processed_dir)
        # Directory-based separation: geometry/ vs geometry_mates/
        cache_suffix = "_mates" if include_mates else ""
        self.geometry_dir = self.cache_dir / f"{geometry_cache_name}{cache_suffix}"
        if base_pdb_dir is None:
            raise ValueError("base_pdb_dir is required")
        self.base_pdb_dir = Path(base_pdb_dir)
        self.cutoff = cutoff
        self.encoder_type = encoder_type
        if self.encoder_type in ("slae", "esm"):
            self.embedding_dir = self.cache_dir / self.encoder_type
        else:
            self.embedding_dir = None
        self.include_mates = include_mates
        self.duplicate_single_sample = duplicate_single_sample

        self.max_com_dist = max_com_dist
        self.max_clash_fraction = max_clash_fraction
        self.clash_dist = clash_dist
        self.interface_dist_threshold = interface_dist_threshold
        self.min_water_residue_ratio = min_water_residue_ratio

        self.max_protein_dist = max_protein_dist
        self.min_edia = min_edia
        self.max_bfactor_zscore = max_bfactor_zscore
        self.filter_by_distance = filter_by_distance
        self.filter_by_edia = filter_by_edia
        self.filter_by_bfactor = filter_by_bfactor

        if self.encoder_type not in {"gvp", "slae", "esm"}:
            raise ValueError(
                f"Unsupported encoder_type '{self.encoder_type}'. "
                "Expected one of: gvp, slae, esm"
            )

        self.entries = self._parse_pdb_list(pdb_list_file)

        if preprocess:
            self._preprocess_all()

        # if single sample and duplication requested, set effective length [this is for experiments to check if the model can memorize a sample]
        if len(self.entries) == 1 and duplicate_single_sample > 1:
            self._effective_length = duplicate_single_sample
            logger.info(
                f"Single sample detected. Duplicating {duplicate_single_sample}x "
            )
        else:
            self._effective_length = len(self.entries)

    def _parse_pdb_list(self, pdb_list_file: str) -> list[dict]:
        """
        Parse PDB list file and construct entries with paths.

        Expected format:
        <pdb_id>_final  (e.g., "6eey_final")

        Constructs path: {base_pdb_dir}/{pdb_id}/{pdb_id}_final.pdb
        """
        entries = []
        with open(pdb_list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if not line.endswith("_final"):
                    logger.warning(f"Warning: Unexpected format: {line}")
                    continue
                pdb_id = line.removesuffix("_final")
                if not pdb_id:
                    logger.warning(f"Warning: Unexpected format: {line}")
                    continue

                pdb_path = self.base_pdb_dir / pdb_id / f"{pdb_id}_final.pdb"

                # Cache key is just the base key - directory separation handles mates
                entries.append(
                    {
                        "pdb_id": pdb_id,
                        "pdb_path": pdb_path,
                        "cache_key": line,
                        "embedding_key": line,  # Same as cache_key for embedding lookup
                    }
                )

        logger.info(f"Loaded {len(entries)} entries from {pdb_list_file}")
        return entries

    def _preprocess_all(self):
        """
        Preprocess all PDB files that don't have cached geometry results.

        Iterates through entries, runs PyMOL crystal contact detection,
        applies quality filters, and caches results. Entries that fail
        preprocessing are logged and removed from the dataset.
        """
        self.geometry_dir.mkdir(parents=True, exist_ok=True)

        to_process = [
            e
            for e in self.entries
            if not (self.geometry_dir / f"{e['cache_key']}.pt").exists()
        ]

        if not to_process:
            logger.info("All entries already preprocessed.")
            return

        logger.info(f"Preprocessing {len(to_process)} entries...")
        failures = []
        for entry in tqdm(to_process, desc="Preprocessing"):
            cache_path = self.geometry_dir / f"{entry['cache_key']}.pt"
            try:
                self._preprocess_one(entry, cache_path)
            except Exception as e:
                logger.warning(f"\nFailed to preprocess {entry['cache_key']}: {e}")
                failures.append((entry["cache_key"], str(e)))

        # write failures to log file
        if failures:
            failure_log_path = self.geometry_dir / "preprocessing_failures.log"
            with open(failure_log_path, "a") as f:
                for pdb_id, reason in failures:
                    f.write(f"{pdb_id}\t{reason}\n")
            logger.info(f"Logged {len(failures)} failures to {failure_log_path}")

        valid_entries = [
            e
            for e in self.entries
            if (self.geometry_dir / f"{e['cache_key']}.pt").exists()
        ]
        n_removed = len(self.entries) - len(valid_entries)
        if n_removed > 0:
            logger.info(f"Filtered out {n_removed} entries without valid cache files.")
        self.entries = valid_entries
        logger.info(f"Dataset contains {len(self.entries)} valid entries.")

    def _preprocess_one(self, entry: dict, cache_path: Path):
        """
        Preprocess a single PDB file.

        Runs PyMOL crystal contact detection and caches:
        - Protein positions, features, residue indices
        - Water positions and features (if any)
        - Symmetry mate positions and features (if any)

        Raises ValueError if structure fails quality filters.
        """
        pdb_path = str(entry["pdb_path"])

        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_path)

        # check inter-chain interactions for multi-chain proteins
        chain_valid, chain_reason, _ = check_chain_interactions(
            protein_atoms,
            interface_dist_threshold=self.interface_dist_threshold,
        )
        if not chain_valid:
            raise ValueError(f"Quality filter failed: {chain_reason}")

        crystal_data = get_crystal_contacts_pymol(pdb_path, self.cutoff)

        # Ensure consistency between biotite and PyMOL parsing.
        # Both parse the same ASU, but may differ in altloc selection, hydrogen
        # handling, or edge cases. Keep only waters present in both representations.
        asu_water_indices = match_atoms_to_coords(
            water_atoms, crystal_data["asu_coords"]
        )
        if asu_water_indices:
            asu_water_mask = np.zeros(len(water_atoms), dtype=bool)
            asu_water_mask[asu_water_indices] = True
            water_atoms = water_atoms[asu_water_mask]
        else:
            water_atoms = water_atoms[:0]

        # Per-water filtering is optional; structure-level quality checks below always run.
        use_distance_filter = self.filter_by_distance
        use_edia_filter = self.filter_by_edia
        use_bfactor_filter = self.filter_by_bfactor
        any_filter_enabled = (
            use_distance_filter or use_edia_filter or use_bfactor_filter
        )

        if any_filter_enabled and water_atoms:
            # load EDIA data only when the EDIA filter is active
            edia_lookup = None
            if use_edia_filter:
                edia_json_path = Path(pdb_path).with_suffix(".json")
                edia_lookup = load_edia_for_pdb(edia_json_path)
                if edia_lookup is None:
                    raise ValueError(
                        f"EDIA filtering enabled but JSON file missing for {entry['pdb_id']}. "
                        f"Expected file: {edia_json_path.name} in the same directory as the PDB."
                    )

            # compute normalized B-factors only when the B-factor filter is active
            bfactor_lookup = None
            if use_bfactor_filter:
                bfactor_lookup, _ = compute_normalized_bfactors(pdb_path)

            # build water keys for filtering
            water_keys = list(
                zip(
                    water_atoms.chain_id.astype(str),
                    water_atoms.res_id.astype(int),
                    np.array(
                        [normalize_ins_code(x) for x in water_atoms.ins_code],
                        dtype=object,
                    ),
                )
            )

            # apply quality filters
            keep_mask = filter_waters_by_quality(
                water_atoms.coord,
                water_keys,
                protein_atoms.coord if use_distance_filter else None,
                edia_lookup,
                bfactor_lookup,
                max_protein_dist=self.max_protein_dist,
                min_edia=self.min_edia,
                max_bfactor_zscore=self.max_bfactor_zscore,
                cache_key=entry["cache_key"],
            )
            water_atoms = water_atoms[keep_mask]

        protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
        water_pos_raw = (
            torch.tensor(water_atoms.coord, dtype=torch.float32)
            if water_atoms
            else torch.zeros((0, 3), dtype=torch.float32)
        )

        # Structure-level quality checks remain active even if all per-water filters are disabled.
        # check center-of-mass distance of protein atoms and water atoms (before centering)
        com_valid, com_reason = check_com_distance(
            protein_pos,
            water_pos_raw,
            max_com_dist=self.max_com_dist,
        )
        if not com_valid:
            raise ValueError(f"Quality filter failed: {com_reason}")

        # check water clashes with protein atoms
        clash_valid, clash_reason = check_water_clashes(
            protein_pos,
            water_pos_raw,
            clash_dist=self.clash_dist,
            max_clash_fraction=self.max_clash_fraction,
        )
        if not clash_valid:
            raise ValueError(f"Quality filter failed: {clash_reason}")

        # center protein positions
        center = protein_pos.mean(dim=0, keepdim=True)
        protein_pos = protein_pos - center

        protein_elements = [str(e).upper() for e in protein_atoms.element]
        protein_feat = element_onehot(protein_elements)

        # compute residue indices (including ins_code to match ESM/SLAE residue counting)
        res_id = protein_atoms.res_id
        chain_id_arr = protein_atoms.chain_id
        ins_code_arr = np.array(
            [normalize_ins_code(x) for x in protein_atoms.ins_code], dtype=object
        )
        residue_keys = list(zip(chain_id_arr, res_id, ins_code_arr))
        unique_res = {k: i for i, k in enumerate(dict.fromkeys(residue_keys))}
        protein_res_idx = torch.tensor(
            [unique_res[k] for k in residue_keys], dtype=torch.long
        )

        # check water/residue ratio
        num_residues = len(unique_res)
        num_waters = len(water_atoms)
        ratio_valid, ratio_reason = check_water_residue_ratio(
            num_waters,
            num_residues,
            min_ratio=self.min_water_residue_ratio,
        )
        if not ratio_valid:
            raise ValueError(f"Quality filter failed: {ratio_reason}")

        # process water atoms
        if water_atoms:
            water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32) - center
            water_elements = [str(e).upper() for e in water_atoms.element]
            water_feat = element_onehot(water_elements)
        else:
            water_pos = torch.zeros((0, 3), dtype=torch.float32)
            water_feat = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)

        # process symmetry mate atoms
        mate_coords = crystal_data["mate_coords"]
        if mate_coords.shape[0] > 0:
            mate_pos = torch.tensor(mate_coords, dtype=torch.float32) - center
            mate_elements = [a.symbol.upper() for a in crystal_data["mate_atoms"]]
            mate_feat = element_onehot(mate_elements)

            # compute mate residue indices (group atoms by actual residue)
            mate_residue_keys = [(a.chain, a.resi) for a in crystal_data["mate_atoms"]]
            unique_mate_res = list(dict.fromkeys(mate_residue_keys))  # preserves order
            mate_res_map = {k: i for i, k in enumerate(unique_mate_res)}
            mate_res_idx = torch.tensor(
                [mate_res_map[k] for k in mate_residue_keys], dtype=torch.long
            )
        else:
            mate_pos = torch.zeros((0, 3), dtype=torch.float32)
            mate_feat = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)
            mate_res_idx = torch.empty(0, dtype=torch.long)

        # Compute final protein data based on include_mates flag
        num_asu_protein = protein_pos.size(0)
        if self.include_mates and mate_pos.size(0) > 0:
            final_protein_pos = torch.cat([protein_pos, mate_pos], dim=0)
            final_protein_feat = torch.cat([protein_feat, mate_feat], dim=0)
            # Offset mate residue indices by max protein residue index
            max_res_idx = (
                protein_res_idx.max().item() if protein_res_idx.numel() > 0 else -1
            )
            offset_mate_res_idx = mate_res_idx + max_res_idx + 1
            final_protein_res_idx = torch.cat(
                [protein_res_idx, offset_mate_res_idx], dim=0
            )
        else:
            final_protein_pos = protein_pos
            final_protein_feat = protein_feat
            final_protein_res_idx = protein_res_idx

        # Compute PP edges and features
        if final_protein_pos.size(0) > 0:
            pp_edge_index = radius_graph(final_protein_pos, r=self.cutoff, loop=False)
            pp_edge_index = _make_undirected(pp_edge_index)
            pp_edge_unit_vectors, pp_edge_rbf = compute_edge_features(
                final_protein_pos,
                pp_edge_index,
                num_gaussians=NUM_RBF,
                cutoff=self.cutoff,
            )
        else:
            pp_edge_index = torch.empty((2, 0), dtype=torch.long)
            pp_edge_unit_vectors, pp_edge_rbf = compute_edge_features(
                final_protein_pos,
                pp_edge_index,
                num_gaussians=NUM_RBF,
                cutoff=self.cutoff,
            )

        # Cache all data including PP edges and features
        torch.save(
            {
                "protein_pos": final_protein_pos,
                "protein_feat": final_protein_feat,
                "protein_res_idx": final_protein_res_idx,
                "water_pos": water_pos,
                "water_feat": water_feat,
                # PP topology and features (precomputed)
                "pp_edge_index": pp_edge_index,
                "pp_edge_unit_vectors": pp_edge_unit_vectors,
                "pp_edge_rbf": pp_edge_rbf,
                # Metadata
                "num_asu_protein": num_asu_protein,
                "num_protein_residues": num_residues,
            },
            cache_path,
        )

    def __len__(self) -> int:
        return self._effective_length

    def _annotate_data_with_embeddings(
        self,
        data: HeteroData,
        cache_key: str,
        asu_protein_res_idx: torch.Tensor,
        num_asu_protein: int,
        num_protein_residues: int,
    ) -> None:
        """
        Load encoder-specific embeddings and attach to data object.

        Only loads embeddings for the encoder type specified at dataset init.
        GVP encoder doesn't require pre-computed embeddings. Embeddings are
        stored using generic attribute names (embedding, embedding_type) for
        consistent access regardless of encoder type.

        Args:
            data: HeteroData object to attach embeddings to (modified in-place)
            cache_key: Identifier for cached embedding files
            asu_protein_res_idx: (N_asu,) residue index per ASU atom
            num_asu_protein: Number of ASU protein atoms
            num_protein_residues: Number of unique protein residues
        """
        if self.encoder_type == "slae":
            data["protein"].embedding = load_slae_embedding(
                embedding_dir=self.embedding_dir,
                cache_key=cache_key,
                num_asu_protein=num_asu_protein,
                total_num_atoms=data["protein"].num_nodes,
            )
            data["protein"].embedding_type = "slae"
        elif self.encoder_type == "esm":
            # Load residue embeddings and broadcast to atom level
            residue_embeddings = load_esm_embedding(
                embedding_dir=self.embedding_dir,
                cache_key=cache_key,
                num_protein_residues=num_protein_residues,
            )
            esm_atom_emb = residue_embeddings[asu_protein_res_idx]
            data["protein"].embedding = _pad_atom_embeddings_for_mates(
                esm_atom_emb, data["protein"].num_nodes
            )
            data["protein"].embedding_type = "esm"

    def __getitem__(self, idx: int) -> HeteroData:
        """
        Load cached data and build graph.

        Returns HeteroData with:
        - 'protein' node type with pos, x, residue_index
        - 'water' node type with pos, x
        - ('protein', 'pp', 'protein') edges with:
            - edge_index: (2, E) topology
            - edge_unit_vectors: (E, 3) unit vectors
            - edge_rbf: (E, 16) RBF features
        - NO water edges (built dynamically in flow model)
        """
        # map idx to actual entry index (handles duplication)
        if len(self.entries) == 0:
            raise IndexError("ProteinWaterDataset is empty; no entries available.")

        actual_idx = idx % len(self.entries)
        entry = self.entries[actual_idx]
        cache_path = self.geometry_dir / f"{entry['cache_key']}.pt"

        if not cache_path.exists():
            raise FileNotFoundError(
                f"Geometry cache file not found: {cache_path}. "
                f"Run with preprocess=True to generate it."
            )

        cached = torch.load(cache_path, weights_only=False)

        # load all data directly from cache (already includes mates if applicable)
        protein_pos = cached["protein_pos"]
        protein_feat = cached["protein_feat"]
        protein_res_idx = cached["protein_res_idx"]
        pp_edge_index = cached["pp_edge_index"]
        pp_edge_unit_vectors = cached["pp_edge_unit_vectors"]
        pp_edge_rbf = cached["pp_edge_rbf"]
        num_asu_protein = cached["num_asu_protein"]
        num_protein_residues = cached["num_protein_residues"]
        water_pos = cached["water_pos"]
        water_feat = cached["water_feat"]

        # extract ASU protein residue indices for embedding loading
        asu_protein_res_idx = protein_res_idx[:num_asu_protein]

        data = HeteroData()

        # compute total num_residues (protein + mates)
        num_residues = (
            int(protein_res_idx.max().item() + 1) if protein_res_idx.numel() > 0 else 0
        )

        data["protein"].x = protein_feat
        data["protein"].pos = protein_pos
        data["protein"].residue_index = protein_res_idx
        data["protein"].num_nodes = protein_pos.size(0)
        data["protein"].num_residues = num_residues
        data["protein"].num_protein_residues = num_protein_residues

        self._annotate_data_with_embeddings(
            data=data,
            cache_key=entry["embedding_key"],  # use base key for embeddings
            asu_protein_res_idx=asu_protein_res_idx,
            num_asu_protein=num_asu_protein,
            num_protein_residues=num_protein_residues,
        )

        data["water"].x = water_feat
        data["water"].pos = water_pos
        data["water"].num_nodes = water_pos.size(0)

        # load PP edges and features from cache
        data[EDGE_PP].edge_index = pp_edge_index
        data[EDGE_PP].edge_unit_vectors = pp_edge_unit_vectors
        data[EDGE_PP].edge_rbf = pp_edge_rbf

        # store metadata (use embedding_key for consistency with existing code)
        data.pdb_id = entry["embedding_key"]
        data.num_asu_protein_atoms = num_asu_protein

        return data


def get_dataloader(
    pdb_list_file: str,
    processed_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 8,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a DataLoader for crystal contact dataset.

    Args:
        pdb_list_file: Path to text file with PDB entries (one per line)
        processed_dir: Cache root directory. Uses:
                      - {processed_dir}/geometry for geometry caches
                      - {processed_dir}/{encoder_name} for embedding caches
        encoder_type: Encoder used downstream ('gvp', 'slae', or 'esm').
                      Embeddings are loaded only for this type.
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of DataLoader workers (default 8)
        pin_memory: Pin memory for faster CPU-GPU transfer (default True)
        prefetch_factor: Number of batches to prefetch per worker (default 4)
        persistent_workers: Keep workers alive between epochs (default True)
        **dataset_kwargs: Additional arguments passed to ProteinWaterDataset
                         (e.g., cutoff, include_mates, duplicate_single_sample)

    Returns:
        DataLoader that yields batched HeteroData objects

    Note:
        For single-protein overfitting, use duplicate_single_sample parameter:
        - duplicate_single_sample=100 creates 100 copies of the sample in the dataset
        - Then batch_size works normally
    """
    dataset = ProteinWaterDataset(
        pdb_list_file=pdb_list_file, processed_dir=processed_dir, **dataset_kwargs
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=lambda batch: Batch.from_data_list(batch),
    )

    return loader
