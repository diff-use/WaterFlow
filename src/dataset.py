#dataset.py

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Dict
from pathlib import Path
import itertools
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData, Batch
from tqdm import tqdm

import biotite.structure as bts
from biotite.structure.io.pdb import PDBFile, get_structure
import pymol2
from torch_cluster import radius_graph
import e3nn
from e3nn.math import soft_one_hot_linspace

from src.utils import atom37_to_atoms

ELEMENT_VOCAB = [
    "C", "N", "O", "S", "P", "SE", "MG", "ZN", "CA", "FE", "NA", "K", "CL", "F", "BR",
]
ELEM_IDX = {e: i for i, e in enumerate(ELEMENT_VOCAB)}


def element_onehot(symbols: List[str]) -> Tensor:
    """One-hot encoding with 'other' bucket at end."""
    other_idx = len(ELEMENT_VOCAB)
    indices = torch.tensor([ELEM_IDX.get(s.upper(), other_idx) for s in symbols], dtype=torch.long) 
    return F.one_hot(indices, num_classes=other_idx + 1).float()


def parse_asu_with_biotite(
    path: str,
    chain_filter: Optional[Sequence[str]] = None,
) -> Tuple[bts.AtomArray, bts.AtomArray]:
    """Parse PDB and return (protein_atoms, water_atoms)."""
    pdb_file = PDBFile.read(path)
    atoms = get_structure(pdb_file, model=1, altloc="occupancy")
    
    if chain_filter is not None:
        mask = np.isin(atoms.chain_id, np.array(chain_filter, dtype=atoms.chain_id.dtype))
        atoms = atoms[mask]
    
    atoms = atoms[atoms.element != "H"]
    
    protein_mask = bts.filter_canonical_amino_acids(atoms)
    water_mask = (atoms.res_name == "HOH") | (atoms.res_name == "WAT")
    
    protein_atoms = atoms[protein_mask]
    water_atoms = atoms[water_mask]
    
    return protein_atoms, water_atoms


def get_crystal_contacts_pymol(pdb_path: str, cutoff: float = 5.0) -> Dict:
    """Get ASU and symmetry mate atoms using PyMOL."""
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
        
        return {
            "asu_coords": asu_coords if asu_coords is not None else np.zeros((0, 3), dtype=float),
            "mate_coords": mate_coords if mate_coords is not None else np.zeros((0, 3), dtype=float),
            "asu_atoms": asu_atoms,
            "mate_atoms": mate_atoms,
        }


def match_atoms_to_coords(
    atoms: bts.AtomArray,
    target_coords: np.ndarray,
    tolerance: float = 0.01
) -> List[int]:
    """Match biotite atoms to PyMOL coordinates, return indices."""
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
    """Symmetrize and deduplicate edges: edge_index shape [2, E]."""
    if edge_index.numel() == 0:
        return edge_index
    ei = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # add reverse edges
    ei = torch.unique(ei.T, dim=0).T  # drop duplicates
    return ei

def check_com_distance(
    protein_coords: torch.Tensor,
    water_coords: torch.Tensor,
    max_com_dist: float = 25.0,
) -> Tuple[bool, str]:
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
) -> Tuple[bool, str]:
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
    protein_atoms: "bts.AtomArray",
    interface_dist_threshold: float = 4.0,
) -> Tuple[bool, str, str]:
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

    # get coordinates per chain
    chain_coords = {
        cid: torch.tensor(protein_atoms[protein_atoms.chain_id == cid].coord, dtype=torch.float32)
        for cid in chain_ids
    }

    min_interface_dist = float('inf')

    for chain_a, chain_b in itertools.combinations(chain_ids, 2):
        coords_a = chain_coords[chain_a]
        coords_b = chain_coords[chain_b]

        # compute pairwise distances between chains
        dists = torch.cdist(coords_a, coords_b)
        min_d = dists.min().item()

        if min_d < min_interface_dist:
            min_interface_dist = min_d

    if min_interface_dist > interface_dist_threshold:
        return (
            False,
            f"Multi-chain ({num_chains} chains) min interface distance {min_interface_dist:.1f}A "
            f"> {interface_dist_threshold}A (likely ASU copies, not PPI)",
            "Non-Interacting (ASU Copies)"
        )

    return True, "", "Interacting"


def check_water_residue_ratio(
    num_waters: int,
    num_residues: int,
    min_ratio: float = 0.8,
) -> Tuple[bool, str]:
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
        base_pdb_dir: str = "/sb/wankowicz_lab/data/srivasv/pdb_redo_data",
        cutoff: float = 8.0,
        include_mates: bool = True,
        preprocess: bool = True,
        duplicate_single_sample: int = 1,
        max_com_dist: float = 25.0,
        max_clash_fraction: float = 0.05,
        clash_dist: float = 2.0,
        interface_dist_threshold: float = 4.0,
        min_water_residue_ratio: float = 0.8,
    ):
        """
        Args:
            pdb_list_file: Text file with lines like "<pdb_id>_final_<chainID>"
            processed_dir: Directory to cache preprocessed .pt files
            base_pdb_dir: Base directory containing PDB subdirectories
            cutoff: Distance cutoff for PP edges and crystal contacts (Angstroms)
            include_mates: If True, include symmetry mate atoms as protein nodes
            preprocess: If True, run preprocessing on missing cached files
            duplicate_single_sample: If dataset has 1 sample, duplicate it this many times
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
        """
        self.processed_dir = Path(processed_dir)
        self.base_pdb_dir = Path(base_pdb_dir)
        self.cutoff = cutoff
        self.include_mates = include_mates
        self.duplicate_single_sample = duplicate_single_sample

        self.max_com_dist = max_com_dist
        self.max_clash_fraction = max_clash_fraction
        self.clash_dist = clash_dist
        self.interface_dist_threshold = interface_dist_threshold
        self.min_water_residue_ratio = min_water_residue_ratio

        self.entries = self._parse_pdb_list(pdb_list_file)

        if preprocess:
            self._preprocess_all()

        # if single sample and duplication requested, set effective length [this is for experiments to check if the model can memorize a sample]
        if len(self.entries) == 1 and duplicate_single_sample > 1:
            self._effective_length = duplicate_single_sample
            print(f"Single sample detected. Duplicating {duplicate_single_sample}x ")
        else:
            self._effective_length = len(self.entries)
    
    def _parse_pdb_list(self, pdb_list_file: str) -> List[Dict]:
        """
        Parse PDB list file and construct entries with paths.

        Supports two formats:
        1. Chain-specific: <pdb_id>_final_<chainID>  (e.g., "6eey_final_A")
        2. Whole PDB: <pdb_id>_final  (e.g., "6eey_final")

        Constructs path: {base_pdb_dir}/{pdb_id}/{pdb_id}_final.pdb
        """
        entries = []
        with open(pdb_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('_')
                if len(parts) < 2:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue

                pdb_id = parts[0]

                if len(parts) >= 3 and parts[1] == "final":
                    chain_id = parts[-1]
                elif len(parts) == 2 and parts[1] == "final":
                    chain_id = None
                else:
                    print(f"Warning: Unexpected format: {line}")
                    continue

                pdb_path = self.base_pdb_dir / pdb_id / f"{pdb_id}_final.pdb"

                entries.append({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'pdb_path': pdb_path,
                    'cache_key': line,
                })

        print(f"Loaded {len(entries)} entries from {pdb_list_file}")
        return entries
    
    def _preprocess_all(self):
        """Preprocess all PDB files that don't have cached results."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        to_process = [
            e for e in self.entries
            if not (self.processed_dir / f"{e['cache_key']}.pt").exists()
        ]

        if not to_process:
            print("All entries already preprocessed.")
            return

        print(f"Preprocessing {len(to_process)} entries...")
        failures = []
        for entry in tqdm(to_process, desc="Preprocessing"):
            cache_path = self.processed_dir / f"{entry['cache_key']}.pt"
            try:
                self._preprocess_one(entry, cache_path)
            except Exception as e:
                print(f"\nFailed to preprocess {entry['cache_key']}: {e}")
                failures.append((entry['cache_key'], str(e)))

        # Write failures to log file
        if failures:
            failure_log_path = self.processed_dir / "preprocessing_failures.log"
            with open(failure_log_path, 'a') as f:
                for pdb_id, reason in failures:
                    f.write(f"{pdb_id}\t{reason}\n")
            print(f"Logged {len(failures)} failures to {failure_log_path}")

        valid_entries = [
            e for e in self.entries
            if (self.processed_dir / f"{e['cache_key']}.pt").exists()
        ]
        n_removed = len(self.entries) - len(valid_entries)
        if n_removed > 0:
            print(f"Filtered out {n_removed} entries without valid cache files.")
        self.entries = valid_entries
        print(f"Dataset contains {len(self.entries)} valid entries.")
    
    def _preprocess_one(self, entry: Dict, cache_path: Path):
        """
        Preprocess a single PDB file.

        Runs PyMOL crystal contact detection and caches:
        - Protein positions, features, residue indices
        - Water positions and features (if any)
        - Symmetry mate positions and features (if any)

        Raises ValueError if structure fails quality filters.
        """
        pdb_path = str(entry['pdb_path'])
        chain_filter = [entry['chain_id']] if entry['chain_id'] is not None else None

        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_path, chain_filter)

        # check inter-chain interactions for multi-chain proteins
        chain_valid, chain_reason, _ = check_chain_interactions(
            protein_atoms,
            interface_dist_threshold=self.interface_dist_threshold,
        )
        if not chain_valid:
            raise ValueError(f"Quality filter failed: {chain_reason}")

        crystal_data = get_crystal_contacts_pymol(pdb_path, self.cutoff)

        # filter water atoms to only those in ASU
        asu_water_indices = match_atoms_to_coords(
            water_atoms, crystal_data["asu_coords"]
        )
        if len(asu_water_indices) > 0:
            asu_water_mask = np.zeros(len(water_atoms), dtype=bool)
            asu_water_mask[asu_water_indices] = True
            water_atoms = water_atoms[asu_water_mask]
        else:
            water_atoms = water_atoms[:0]

        protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
        water_pos_raw = (
            torch.tensor(water_atoms.coord, dtype=torch.float32)
            if len(water_atoms) > 0
            else torch.zeros((0, 3), dtype=torch.float32)
        )

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
        protein_x = element_onehot(protein_elements)
        
        # compute residue indices (using chain_id, res_id only - matches SLAE's atomarray_to_tensors)
        res_id = protein_atoms.res_id
        chain_id_arr = protein_atoms.chain_id
        residue_keys = list(zip(chain_id_arr, res_id))
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
        if len(water_atoms) > 0:
            water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32) - center
            water_elements = [str(e).upper() for e in water_atoms.element]
            water_x = element_onehot(water_elements)
        else:
            water_pos = torch.zeros((0, 3), dtype=torch.float32)
            water_x = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)
        
        # process symmetry mate atoms
        mate_coords = crystal_data["mate_coords"]
        if mate_coords.shape[0] > 0:
            mate_pos = torch.tensor(mate_coords, dtype=torch.float32) - center
            mate_elements = [a.symbol.upper() for a in crystal_data["mate_atoms"]]
            mate_x = element_onehot(mate_elements)

            # compute mate residue indices (group atoms by actual residue)
            mate_residue_keys = [(a.chain, a.resi) for a in crystal_data["mate_atoms"]]
            unique_mate_res = list(dict.fromkeys(mate_residue_keys))  # preserves order
            mate_res_map = {k: i for i, k in enumerate(unique_mate_res)}
            mate_res_idx = torch.tensor(
                [mate_res_map[k] for k in mate_residue_keys], dtype=torch.long
            )
        else:
            mate_pos = torch.zeros((0, 3), dtype=torch.float32)
            mate_x = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)
            mate_res_idx = torch.empty(0, dtype=torch.long)

        # cache all data
        torch.save({
            'protein_pos': protein_pos,
            'protein_x': protein_x,
            'protein_res_idx': protein_res_idx,
            'water_pos': water_pos,
            'water_x': water_x,
            'mate_pos': mate_pos,
            'mate_x': mate_x,
            'mate_res_idx': mate_res_idx,
        }, cache_path)
    
    def __len__(self) -> int:
        return self._effective_length
    
    def __getitem__(self, idx: int) -> HeteroData:
        """
        Load cached data and build graph on-the-fly.

        Returns HeteroData with:
        - 'protein' node type with pos, x, residue_index
        - 'water' node type with pos, x
        - ('protein', 'pp', 'protein') edges with edge_index (topology only)
        - NO water edges (built dynamically in flow model)
        """
        # map idx to actual entry index (handles duplication)
        actual_idx = idx % len(self.entries)
        entry = self.entries[actual_idx]
        cache_path = self.processed_dir / f"{entry['cache_key']}.pt"
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cached file not found: {cache_path}. "
                f"Run with preprocess=True to generate it."
            )
        
        cached = torch.load(cache_path, weights_only=False)

        if 'protein_slae_embedding' in cached and 'protein_atom37_coords' in cached:
            atom37_coords = cached['protein_atom37_coords']
            protein_pos, residue_idx_per_atom, atom_types = atom37_to_atoms(atom37_coords)

            # recenter (TODO: optimize by centering in precompute script)
            center = protein_pos.mean(dim=0, keepdim=True)
            protein_pos = protein_pos - center

            protein_x = F.one_hot(atom_types, num_classes=37).float()
            protein_res_idx = residue_idx_per_atom
            num_asu_protein = protein_pos.size(0)
        else:
            # use original protein atoms from cache 
            protein_pos = cached['protein_pos']
            protein_x = cached['protein_x']
            protein_res_idx = cached['protein_res_idx']
            num_asu_protein = protein_pos.size(0)
        
        # compute num_residues for protein (before adding mates)
        num_protein_residues = int(protein_res_idx.max().item() + 1) if protein_res_idx.numel() > 0 else 0

        # concatenate symmetry mate atoms to protein if mates are included
        if self.include_mates and cached['mate_pos'].size(0) > 0:
            mate_pos = cached['mate_pos']
            mate_x = cached['mate_x']

            protein_pos = torch.cat([protein_pos, mate_pos], dim=0)
            protein_x = torch.cat([protein_x, mate_x], dim=0)

            # load mate residue indices (properly grouped by residue)
            # offset by max protein residue index
            max_res_idx = protein_res_idx.max().item() if protein_res_idx.numel() > 0 else -1
            if 'mate_res_idx' in cached:
                mate_res_idx = cached['mate_res_idx'] + max_res_idx + 1
            else:
                # fallback for old cache files without mate_res_idx
                mate_res_idx = torch.arange(
                    max_res_idx + 1,
                    max_res_idx + 1 + mate_pos.size(0),
                    dtype=torch.long
                )
            protein_res_idx = torch.cat([protein_res_idx, mate_res_idx], dim=0)
        
        water_pos = cached['water_pos']
        water_x = cached['water_x']

        data = HeteroData()

        # compute total num_residues (protein + mates)
        num_residues = int(protein_res_idx.max().item() + 1) if protein_res_idx.numel() > 0 else 0

        data['protein'].x = protein_x
        data['protein'].pos = protein_pos
        data['protein'].residue_index = protein_res_idx
        data['protein'].num_nodes = protein_pos.size(0)
        data['protein'].num_residues = num_residues
        data['protein'].num_protein_residues = num_protein_residues  # excludes mates

        # load SLAE embeddings if available (precomputed by scripts/precompute_slae_embeddings.py)
        if 'protein_slae_embedding' in cached:
            slae_emb = cached['protein_slae_embedding']
            # handle mates: if mates were concatenated during preprocessing, embeddings include them
            if self.include_mates and 'mate_slae_embedding' in cached:
                mate_emb = cached['mate_slae_embedding']
                slae_emb = torch.cat([slae_emb, mate_emb], dim=0)
            data['protein'].slae_embedding = slae_emb
        
        data['water'].x = water_x
        data['water'].pos = water_pos
        data['water'].num_nodes = water_pos.size(0)
        
        if protein_pos.size(0) > 0:
            pp_edge_index = radius_graph(protein_pos, r=self.cutoff, loop=False)
            pp_edge_index = _make_undirected(pp_edge_index)
            data['protein', 'pp', 'protein'].edge_index = pp_edge_index
        else:
            data['protein', 'pp', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # store metadata
        data.pdb_id = entry['cache_key']
        data.num_asu_protein_atoms = num_asu_protein
        
        return data


def get_dataloader(
    pdb_list_file: str,
    processed_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = False,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a DataLoader for crystal contact dataset.

    Args:
        pdb_list_file: Path to text file with PDB entries (one per line)
        processed_dir: Directory for cached preprocessed files
        batch_size: Number of graphs per batch
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
        pdb_list_file=pdb_list_file,
        processed_dir=processed_dir,
        **dataset_kwargs
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda batch: Batch.from_data_list(batch),
    )

    return loader

