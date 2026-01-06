
import numpy as np
import torch
from collections import defaultdict
from biotite.structure import AtomArray
from SLAE.util.constants import PROTEIN_ATOMS_INDEX, RES_TYPE_MAP, FILL


"""
Convert an AtomArray into tensors:
    coords: [N, 37, 3]
    residue_type: [N]
    chains: [N]
"""

                              
def atomarray_to_tensors(atom_array: AtomArray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # --- collect residue data ---
    residues = defaultdict(list)
    for atom in atom_array:
        key = (atom.chain_id, atom.res_id)
        residues[key].append(atom)
    
    # --- sort by chain then residue id ---
    sorted_keys = sorted(residues.keys(), key=lambda x: (x[0], x[1]))
    N = len(sorted_keys)

    coords = np.full((N, 37, 3), FILL, dtype=np.float32)
    residue_type = np.zeros(N, dtype=np.int64)
    # chains is now a list of chain IDs (str)
    chains = np.empty(N, dtype=object)
    residue_id = np.zeros(N, dtype=np.int64)

    # chain letter → number map
    # chain_map = {}
    # next_chain_id = 0

    for i, (chain_id, res_id) in enumerate(sorted_keys):
        atoms = residues[(chain_id, res_id)]
        residue_id[i] = res_id

        # assign chain numeric id
        """
        if chain_id not in chain_map:
            chain_map[chain_id] = next_chain_id
            next_chain_id += 1
        chains[i] = chain_map[chain_id]
        """
        chains[i] = chain_id
        

        res_name = atoms[0].res_name.upper()
        residue_type[i] = RES_TYPE_MAP.get(res_name, -1)  # -1 = unknown residue

        for atom in atoms:
            name = atom.atom_name.strip().upper()
            if name in PROTEIN_ATOMS_INDEX:
                coords[i, PROTEIN_ATOMS_INDEX[name], :] = atom.coord

    # convert to torch tensors
    coords = torch.from_numpy(coords) # [N, 37, 3]
    residue_type = torch.from_numpy(residue_type) # [N]
    #chains = torch.from_numpy(chains) # [N]
    residue_id = torch.from_numpy(residue_id) # [N]

    return coords, residue_type, chains, residue_id


def atom37_to_atoms(atom_tensor):
    """
    Given an atom tensor of shape (N_res, 37, 3), return:
    - coords: (N_atoms, 3)
    - residue_index: (N_atoms,) indicating which residue each atom belongs to to
    - atom_type: (N_atoms,) indicating the atom type (0-36)
    """
    # A site is present if ANY of its 3 coords differs from fill_value
    present = (atom_tensor != FILL).any(dim=-1)        # (N_res, n_slots)
    nz = present.nonzero(as_tuple=False)                     # (N_atoms, 2)
    residue_index = nz[:, 0]
    atom_type     = nz[:, 1].long()

    flat      = atom_tensor.reshape(-1, 3)
    flat_mask = present.reshape(-1)                          # (N_res * n_slots,)
    coords    = flat[flat_mask]                              # (N_atoms, 3)

    return coords, residue_index, atom_type
