#dataset.py

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Dict
from pathlib import Path
import numpy as np

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
from e3nn import o3

ELEMENT_VOCAB = [
    "C", "N", "O", "S", "P", "SE", "MG", "ZN", "CA", "FE", "NA", "K", "CL", "F", "BR",
]
ELEM_IDX = {e: i for i, e in enumerate(ELEMENT_VOCAB)}


def rbf(r: Tensor, num_gaussians: int = 16, cutoff: float = 8.0) -> Tensor:
    """Radial basis function encoding of distances."""
    return e3nn.math.soft_one_hot_linspace(
        r, 
        start=0.0, 
        end=cutoff, 
        number=num_gaussians,
        basis='bessel',
        cutoff=True
    )


def edge_features(
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_rbf: int = 16,
    cutoff: float = 8.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (edge_rbf, edge_vec) = (RBF(dist), unit displacement vectors)."""
    if edge_index.numel() == 0:
        return (
            torch.empty(0, num_rbf, device=src_pos.device),
            torch.empty(0, 1, 3, device=src_pos.device)
        )
    
    s_idx, d_idx = edge_index
    
    disp = src_pos[s_idx] - dst_pos[d_idx]
    dist = torch.clamp(disp.norm(dim=-1, keepdim=True), min=1e-8)
    e_rbf = rbf(dist.squeeze(-1), num_gaussians=num_rbf, cutoff=cutoff)
    e_vec = (disp / dist).unsqueeze(1)
    return e_rbf, e_vec


def element_onehot(symbols: List[str]) -> Tensor:
    """One-hot encoding with 'other' bucket at end."""
    other_idx = len(ELEMENT_VOCAB)
    indices = torch.tensor([ELEM_IDX.get(s.upper(), other_idx) for s in symbols])
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
        cutoff: float = 3.0,
        num_rbf: int = 16,
        include_mates: bool = True,
        preprocess: bool = True,
    ):
        """
        Args:
            pdb_list_file: Text file with lines like "<pdb_id>_final_<chainID>"
            processed_dir: Directory to cache preprocessed .pt files
            base_pdb_dir: Base directory containing PDB subdirectories
            cutoff: Distance cutoff for edges and crystal contacts (Angstroms)
            num_rbf: Number of RBF bins for edge distance encoding
            include_mates: If True, include symmetry mate atoms as protein nodes
            preprocess: If True, run preprocessing on missing cached files
        """
        self.processed_dir = Path(processed_dir)
        self.base_pdb_dir = Path(base_pdb_dir)
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.include_mates = include_mates
        
        self.entries = self._parse_pdb_list(pdb_list_file)
        
        if preprocess:
            self._preprocess_all()
    
    def _parse_pdb_list(self, pdb_list_file: str) -> List[Dict]:
        """
        Parse PDB list file and construct entries with paths.
        
        Expected format per line: <pdb_id>_final_<chainID>
        Constructs path: {base_pdb_dir}/{pdb_id}/{pdb_id}_final.pdb
        """
        entries = []
        with open(pdb_list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                #parse format: <pdb_id>_final_<chainID>
                parts = line.split('_')
                if len(parts) < 3:
                    print(f"Warning: Skipping malformed line: {line}")
                    continue
                
                pdb_id = parts[0]
                chain_id = parts[-1]  # last part is chain ID
                
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
        for entry in tqdm(to_process, desc="Preprocessing"):
            cache_path = self.processed_dir / f"{entry['cache_key']}.pt"
            try:
                self._preprocess_one(entry, cache_path)
            except Exception as e:
                print(f"\nFailed to preprocess {entry['cache_key']}: {e}")
    
    def _preprocess_one(self, entry: Dict, cache_path: Path):
        """
        Preprocess a single PDB file.
        
        Runs expensive PyMOL crystal contact detection and caches:
        - Protein positions, features, residue indices
        - Water positions and features (if any)
        - Symmetry mate positions and features (if any)
        """
        pdb_path = str(entry['pdb_path'])
        chain_filter = [entry['chain_id']]
        
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_path, chain_filter)
        crystal_data = get_crystal_contacts_pymol(pdb_path, self.cutoff)
        
        #filter water atoms to only those in ASU
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
        center = protein_pos.mean(dim=0, keepdim=True)
        protein_pos = protein_pos - center
        
        protein_elements = [str(e).upper() for e in protein_atoms.element]
        protein_x = element_onehot(protein_elements)
        
        #compute residue indices
        res_id = protein_atoms.res_id
        chain_id_arr = protein_atoms.chain_id
        ins_code = getattr(
            protein_atoms, "ins_code", np.array([""] * len(protein_atoms))
        )
        residue_keys = list(zip(chain_id_arr, res_id, ins_code))
        unique_res = {k: i for i, k in enumerate(dict.fromkeys(residue_keys))}
        protein_res_idx = torch.tensor(
            [unique_res[k] for k in residue_keys], dtype=torch.long
        )
        
        #process water atoms
        if len(water_atoms) > 0:
            water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32) - center
            water_elements = [str(e).upper() for e in water_atoms.element]
            water_x = element_onehot(water_elements)
        else:
            water_pos = torch.zeros((0, 3), dtype=torch.float32)
            water_x = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)
        
        #process symmetry mate atoms
        mate_coords = crystal_data["mate_coords"]
        if mate_coords.shape[0] > 0:
            mate_pos = torch.tensor(mate_coords, dtype=torch.float32) - center
            mate_elements = [a.symbol.upper() for a in crystal_data["mate_atoms"]]
            mate_x = element_onehot(mate_elements)
        else:
            mate_pos = torch.zeros((0, 3), dtype=torch.float32)
            mate_x = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)
        
        #cache all data
        torch.save({
            'protein_pos': protein_pos,
            'protein_x': protein_x,
            'protein_res_idx': protein_res_idx,
            'water_pos': water_pos,
            'water_x': water_x,
            'mate_pos': mate_pos,
            'mate_x': mate_x,
        }, cache_path)
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> HeteroData:
        """
        Load cached data and build graph on-the-fly.
        
        Returns HeteroData with:
        - 'protein' node type with pos, x, residue_index
        - 'water' node type with pos, x
        - ('protein', 'pp', 'protein') edges with edge_index, edge_rbf, edge_vec
        - NO water edges (create these during training as needed)
        """
        entry = self.entries[idx]
        cache_path = self.processed_dir / f"{entry['cache_key']}.pt"
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cached file not found: {cache_path}. "
                f"Run with preprocess=True to generate it."
            )
        
        cached = torch.load(cache_path, weights_only=False)
        
        #start with ASU protein atoms
        protein_pos = cached['protein_pos']
        protein_x = cached['protein_x']
        protein_res_idx = cached['protein_res_idx']
        num_asu_protein = protein_pos.size(0)
        
        #concatenate symmetry mate atoms to protein if mates are included
        if self.include_mates and cached['mate_pos'].size(0) > 0:
            mate_pos = cached['mate_pos']
            mate_x = cached['mate_x']
            
            protein_pos = torch.cat([protein_pos, mate_pos], dim=0)
            protein_x = torch.cat([protein_x, mate_x], dim=0)
            
            #assign unique residue IDs to mates
            max_res_idx = protein_res_idx.max().item() if protein_res_idx.numel() > 0 else -1
            mate_res_idx = torch.arange(
                max_res_idx + 1,
                max_res_idx + 1 + mate_pos.size(0),
                dtype=torch.long
            )
            protein_res_idx = torch.cat([protein_res_idx, mate_res_idx], dim=0)
        
        water_pos = cached['water_pos']
        water_x = cached['water_x']
        
        data = HeteroData()
        
        data['protein'].x = protein_x
        data['protein'].pos = protein_pos
        data['protein'].residue_index = protein_res_idx
        data['protein'].num_nodes = protein_pos.size(0)
        
        data['water'].x = water_x
        data['water'].pos = water_pos
        data['water'].num_nodes = water_pos.size(0)
        
        if protein_pos.size(0) > 0:
            pp_edge_index = radius_graph(protein_pos, r=self.cutoff, loop=False)
            pp_edge_index = _make_undirected(pp_edge_index)
            
            pp_edge_rbf, pp_edge_vec = edge_features(
                protein_pos, protein_pos, pp_edge_index, self.num_rbf, self.cutoff
            )
            
            data['protein', 'pp', 'protein'].edge_index = pp_edge_index
            data['protein', 'pp', 'protein'].edge_rbf = pp_edge_rbf
            data['protein', 'pp', 'protein'].edge_vec = pp_edge_vec.squeeze(1)
        else:
            data['protein', 'pp', 'protein'].edge_index = torch.empty((2, 0), dtype=torch.long)
            data['protein', 'pp', 'protein'].edge_rbf = torch.empty((0, self.num_rbf), dtype=torch.float32)
            data['protein', 'pp', 'protein'].edge_vec = torch.empty((0, 3), dtype=torch.float32)
        
        #store metadata
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
                         (e.g., cutoff, num_rbf, include_mates)
    
    Returns:
        DataLoader that yields batched HeteroData objects
        
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

