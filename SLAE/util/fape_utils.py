import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from SLAE.util.constants import SYMMETRIC_ATOMS, RES3, PROTEIN_ATOMS
def build_bonds(pos: torch.Tensor, 
                edge_index: torch.Tensor, 
                cutoff: float = 2.0) -> torch.Tensor:
    """
    Return indices of putative covalent bonds based solely on distance.
    pos  : [N_atoms, 3]
    edge_index: [2, N_edge] which already take care of batches
    out  : [2, N_bonds]   
    """
    edge_length =  torch.pairwise_distance(pos[edge_index[0, :]], pos[edge_index[1, :]])
    safe_min = 1e-6  # Small positive value
    edge_length = torch.clamp(edge_length, min=safe_min)
    # get pair of atoms with edge_length < cutoff
    mask = edge_length < cutoff
    edge_index = edge_index[:, mask]
    edge_index = torch.unique(edge_index.sort(dim=0)[0], dim=1)
    return edge_index


def bonds_to_angles(bonds: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    bonds : torch.Tensor, dtype *anything integer* (torch.int, torch.long …)
            Shape [2, N_bonds].  Each column (a, b) is an undirected bond.

    Returns
    -------
    angles : torch.Tensor, same dtype & device as `bonds`, shape [N_angles, 3]
             Triplets (i, j, k) with j the central atom and i < k.
    """
    device = bonds.device
    dtype  = torch.long         # could be torch.int32 (torch.int) or int64

    if bonds.numel() == 0:
        return torch.empty((0, 3), dtype=dtype, device=device)

    # ---------- 1.  adjacency list ----------
    n_atoms = int(bonds.max().item()) + 1
    adj = [[] for _ in range(n_atoms)]
    # convert once to Python ints for speed
    for a, b in bonds.t().tolist():
        adj[a].append(b)
        adj[b].append(a)

    # ---------- 2.  build angle triples -----
    angle_list = []
    for j, neigh in enumerate(adj):
        m = len(neigh)
        if m < 2:
            continue
        for x in range(m - 1):
            for y in range(x + 1, m):
                i, k = neigh[x], neigh[y]
                if i > k:                       # enforce i < k uniqueness
                    i, k = k, i
                angle_list.append((i, j, k))

    if angle_list:
        angles = torch.tensor(angle_list, dtype=dtype, device=device)
    else:
        angles = torch.empty((0, 3), dtype=dtype, device=device)

    return angles


_IS_SYMM = {
    (res, atom): True
    for res, atom_list in SYMMETRIC_ATOMS.items()
    for atom in atom_list
}
# ---------------------------------------------------------------------

def symmetric_atom_mask(
    atom_type:     torch.Tensor,   # [N_atoms]  ints 0‑36
    residue_index: torch.Tensor,   # [N_atoms]  ints 0…N_res‑1
    residue_type:  torch.Tensor,   # [N_res]    ints 0…22
) -> torch.Tensor:
    """
    Returns a Boolean [N_atoms] tensor that is True for every atom whose
    *own* label is listed in SYMMETRIC_ATOMS for its residue type.
    """
    device = atom_type.device
    N      = atom_type.size(0)

    # map residue indices → 3‑letter codes once
    res3_codes = [RES3[rid] for rid in residue_type.tolist()]   # Python list

    mask = torch.zeros(N, dtype=torch.bool, device=device)

    for idx in range(N):
        res3 = res3_codes[ residue_index[idx].item() ]          # e.g. "TYR"
        atom = PROTEIN_ATOMS[ atom_type[idx].item() ]                  # e.g. "CD1"
        if (res3, atom) in _IS_SYMM:
            mask[idx] = True
    return mask


def filter_angles(
    angles: torch.Tensor,        # [N_angles, 3]
    sym_mask: torch.Tensor,      # [N_atoms] bool
) -> torch.Tensor:
    keep = ~sym_mask[angles].any(dim=-1)
    return angles[keep]          # [N_filtered, 3]



def make_filtered_frames(
    pos:            torch.Tensor,   # [N_atoms, 3]
    atom_type:      torch.Tensor,   # [N_atoms]
    edge_index:     torch.Tensor,   # [2, N_edges]
    residue_index:  torch.Tensor,   # [N_atoms]
    residue_type:   torch.Tensor,   # [N_residues]
    d_cut: float = 2.0,
) -> torch.Tensor:
    bonds   = build_bonds(pos, edge_index, d_cut)                        # [2, N_bonds]
    angles  = bonds_to_angles(bonds)            # [N_angles, 3]
    symmask = symmetric_atom_mask(atom_type, residue_index, residue_type)
    frames  = filter_angles(angles, symmask)                 # [N_frames, 3]
    return frames

