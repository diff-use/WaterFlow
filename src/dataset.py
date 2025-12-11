# dataset processing with waters

# heterodata object, protein nodes, mate nodes, water nodes -- different convolution weights for mates and proteins to waters
# big question to grapple with is how to deal with the distribution shift with and without mates -- how would that work
# ok screw the dist shift for now, let's see if we can get good metrics from the crystal contacts and the edge updates

# store the unit vectors and rbf embeddings of the distance for the cache -- compute it only for the water edges on the fly

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple, Dict
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from pathlib import Path
import biotite.structure as bts
from biotite.structure.io.pdb import PDBFile, get_structure
import pymol2

import e3nn
from e3nn import o3

from torch_cluster import radius_graph

ELEMENT_VOCAB = [
    "C", "N", "O", "S", "P", "SE", "MG", "ZN", "CA", "FE", "NA", "K", "CL", "F", "BR",
]
ELEM_IDX = {e: i for i, e in enumerate(ELEMENT_VOCAB)}

def rbf(r: Tensor, num_gaussians: int = 16, cutoff: float = 8.0) -> Tensor:
    return e3nn.math.soft_one_hot_linspace(r, 
                                     start=0.0, 
                                     end=cutoff, 
                                     number=num_gaussians,
                                     basis='bessel',
                                     cutoff=True
                                    )

def edge_features(src_pos: torch.Tensor,
                  dst_pos: torch.Tensor,
                  edge_index: torch.Tensor,
                  num_rbf: int = 16,
                  cutoff: float = 8.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (edge_s, edge_v) = (RBF(dist), unit disp vectors) for given edges."""
    if edge_index.numel() == 0:
        return torch.empty(0, num_rbf, device=src_pos.device), torch.empty(0, 1, 3, device=src_pos.device)
    
    s_idx, d_idx = edge_index
    
    assert s_idx.max() < src_pos.size(0), f"Source index {s_idx.max()} >= {src_pos.size(0)}"
    assert d_idx.max() < dst_pos.size(0), f"Destination index {d_idx.max()} >= {dst_pos.size(0)}"
    
    disp = src_pos[s_idx] - dst_pos[d_idx]
    dist = torch.clamp(disp.norm(dim=-1, keepdim=True), min=1e-8)
    e_s = rbf(dist.squeeze(-1), num_gaussians=num_rbf, cutoff=cutoff)
    e_v = (disp / dist).unsqueeze(1)
    return e_s, e_v

def element_onehot(symbols: List[str]) -> Tensor:
    """One-hot encoding with 'other' bucket at end."""
    other_idx = len(ELEMENT_VOCAB)
    out = torch.zeros((len(symbols), other_idx + 1), dtype=torch.float32)
    for i, s in enumerate(symbols):
        j = ELEM_IDX.get(s.upper(), other_idx)
        out[i, j] = 1.0
    return out


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
    
    # Remove hydrogens
    atoms = atoms[atoms.element != "H"]
    
    # Split into protein and water
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

def match_atoms_to_coords(atoms: bts.AtomArray, target_coords: np.ndarray, tolerance: float = 0.01) -> List[int]:
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
    ei = torch.unique(ei.T, dim=0).T  # drop duplicates (unique over columns)
    return ei

def build_hetero_graph(
    pdb_path: str,
    chain_filter: Optional[Sequence[str]] = None,
    cutoff: float = 3.0,
    device: Optional[torch.device] = None,
) -> HeteroData:
    """Build HeteroData with protein, water, mate nodes and radius-graph edges."""
    
    protein_atoms, water_atoms = parse_asu_with_biotite(pdb_path, chain_filter)
    crystal_data = get_crystal_contacts_pymol(pdb_path, cutoff)
    
    asu_water_indices = match_atoms_to_coords(water_atoms, crystal_data["asu_coords"])
    asu_water_mask = np.zeros(len(water_atoms), dtype=bool)
    if len(asu_water_indices) > 0:
        asu_water_mask[asu_water_indices] = True
        water_atoms = water_atoms[asu_water_mask]
    else:
        water_atoms = water_atoms[:0]
    
    protein_pos = torch.tensor(
        protein_atoms.coord.astype(np.float32), dtype=torch.float32
    )
    protein_pos = protein_pos - protein_pos.mean(dim=0, keepdim=True)
    protein_elements = [str(e).upper() for e in protein_atoms.element]
    protein_x = element_onehot(protein_elements)
    
    res_id = protein_atoms.res_id
    chain_id = protein_atoms.chain_id
    ins_code = getattr(
        protein_atoms, "ins_code", np.array([""] * len(protein_atoms))
    )
    residue_keys = list(zip(chain_id, res_id, ins_code))
    unique_res = {k: i for i, k in enumerate(dict.fromkeys(residue_keys))}
    protein_res_idx = torch.tensor(
        [unique_res[k] for k in residue_keys], dtype=torch.long
    )
    
    if len(water_atoms) > 0:
        water_pos = torch.tensor(
            water_atoms.coord.astype(np.float32), dtype=torch.float32
        )
        water_pos = water_pos - protein_pos.mean(dim=0, keepdim=True)
        water_elements = [str(e).upper() for e in water_atoms.element]
        water_x = element_onehot(water_elements)
    else:
        water_pos = torch.zeros((0, 3), dtype=torch.float32)
        water_x = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)
    
    mate_coords = crystal_data["mate_coords"]
    if mate_coords.shape[0] > 0:
        mate_pos = torch.tensor(
            mate_coords.astype(np.float32), dtype=torch.float32
        )
        mate_pos = mate_pos - protein_pos.mean(dim=0, keepdim=True)
        mate_elements = [a.symbol.upper() for a in crystal_data["mate_atoms"]]
        mate_x = element_onehot(mate_elements)
    else:
        mate_pos = torch.zeros((0, 3), dtype=torch.float32)
        mate_x = torch.zeros((0, len(ELEMENT_VOCAB) + 1), dtype=torch.float32)
    
    if protein_pos.size(0) > 0:
        pp_edge_index = radius_graph(
            protein_pos, r=cutoff, loop=False  # [2, E_pp]
        )
        pp_edge_index = _make_undirected(pp_edge_index)
    else:
        pp_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    if mate_pos.size(0) > 0:
        mm_edge_index = radius_graph(
            mate_pos, r=cutoff, loop=False  # [2, E_mm]
        )
        mm_edge_index = _make_undirected(mm_edge_index)
    else:
        mm_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    num_prot = protein_pos.size(0)
    num_mate = mate_pos.size(0)
    if num_prot > 0 and num_mate > 0:
        all_pos = torch.cat([protein_pos, mate_pos], dim=0)
        all_edge_index = radius_graph(
            all_pos, r=cutoff, loop=False  # [2, E_all]
        )
        src, dst = all_edge_index

        pm_mask = (src < num_prot) & (dst >= num_prot)
        mp_mask = (src >= num_prot) & (dst < num_prot)

        prot_idx = torch.cat([src[pm_mask], dst[mp_mask]], dim=0)
        mate_idx = torch.cat([dst[pm_mask], src[mp_mask]], dim=0) - num_prot

        if prot_idx.numel() > 0:
            edge_index_pm = torch.stack([prot_idx, mate_idx], dim=0)
            edge_index_pm = torch.unique(edge_index_pm.T, dim=0).T
        else:
            edge_index_pm = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_pm = torch.empty((2, 0), dtype=torch.long)
    
    data = HeteroData()
    
    data['protein'].pos = protein_pos
    data['protein'].x = protein_x
    data['protein'].residue_index = protein_res_idx
    data['protein'].num_nodes = protein_pos.size(0)
    
    data['water'].pos = water_pos
    data['water'].x = water_x
    data['water'].num_nodes = water_pos.size(0)
    
    data['mate'].pos = mate_pos
    data['mate'].x = mate_x
    data['mate'].num_nodes = mate_pos.size(0)

    data['protein', 'pp', 'protein'].edge_index = pp_edge_index
    data['mate', 'mm', 'mate'].edge_index = mm_edge_index
    data['protein', 'pm', 'mate'].edge_index = edge_index_pm
    
    # if edge_index_pm.numel() > 0: # reverese the flow?
    #     data['mate', 'mp', 'protein'].edge_index = edge_index_pm.flip(0)

    e_s_p, e_v_p = edge_features(protein_pos, protein_pos, pp_edge_index)
    data['protein', 'pp', 'protein'].edge_rbf = e_s_p            # (E, num_rbf)
    data['protein', 'pp', 'protein'].edge_unit_vec = e_v_p.squeeze(1) 

    e_s_m, e_v_m = edge_features(mate_pos, mate_pos, mm_edge_index)
    data['mate', 'mm', 'mate'].edge_rbf = e_s_m            # (E, num_rbf)
    data['mate', 'mm', 'mate'].edge_unit_vec = e_v_m.squeeze(1) 
    
    e_s_pm, e_v_pm = edge_features(protein_pos, mate_pos, edge_index_pm)
    data['protein', 'pm', 'mate'].edge_rbf = e_s_pm            # (E, num_rbf)
    data['protein', 'pm', 'mate'].edge_unit_vec = e_v_pm.squeeze(1) 
    
    return data

#caching, dataset, dataloader building

