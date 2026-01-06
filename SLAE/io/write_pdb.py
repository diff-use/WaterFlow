from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from biopandas.pdb import PandasPdb
from SLAE.io.atom_tensor import atom37_to_atoms

from SLAE.util.constants import FILL, VALID_ATOM37_MASK, PROTEIN_ATOMS, RES3, ATOM37_TYPE, ATOM_TYPES

def infer_residue_types(x: torch.Tensor) -> List[str]:
    """
    Infer residue identities using exact mask matching against VALID_ATOM37_MASK.
    If no exact match, returns 'UNK' for that position.
    """
    assert x.ndim == 3 and x.size(1) == 37 and x.size(2) == 3, "x must be (N, 37, 3)"

    # presence mask per residue: (N, 37) of {0.,1.} to match VALID_ATOM37_MASK dtype
    present = (x != FILL).any(dim=-1).float()
    # flip last entry in present to 0 to handle OXT
    present[:, -1] = 0

    valid_mask = VALID_ATOM37_MASK.to(present.device)
    eq = (present.unsqueeze(1) == valid_mask.unsqueeze(0))  # (N, R, 37)
    exact = eq.all(dim=-1)                                         # (N, R)

    # argmax over R with a check whether there was any exact match
    # if none matched: UNK
    best_idx = exact.float().argmax(dim=1)                         # (N,)
    any_match = exact.any(dim=1)                                   # (N,)

    res3_list: List[str] = []
    for i in range(present.size(0)):
        if any_match[i].item():
            res3 = RES3[int(best_idx[i].item())]
        else:
            res3 = "UNK"
        res3_list.append(res3)

    return res3_list

def to_pdb(x: torch.Tensor, out_path: str) -> str:
    """
    Convert an coord tensor (N, 37, 3) to a PDB file and write it to `out_path`.

    Args:
        x (Tensor): Tensor of shape (N, 37, 3).
        out_path (str): Path to save the PDB file (e.g., '/tmp/protein.pdb').

    Returns:
        str: The path to the written PDB file.
    """
    # Find non-missing atoms (use x-coordinate to identify presence)

    coords, res_nums, atom_index = atom37_to_atoms(x)

    coords = coords.cpu().numpy()
    res_nums = res_nums.cpu().numpy()
    res_nums += 1  # convert to 1-based indexing for PDB
    atom_index = atom_index.cpu().numpy()

    # Infer residue types (3-letter codes)
    residue_codes = infer_residue_types(x)
    residue_names = [residue_codes[i - 1] for i in res_nums]


    # Map atom indices to names
    atom_names = [PROTEIN_ATOMS[i] for i in atom_index]


    # Elements
    element_symbols = [ATOM_TYPES[ATOM37_TYPE[a]] for a in atom_index]

    n = len(res_nums)
    df = pd.DataFrame({
        "record_name": ["ATOM"] * n,
        "atom_number": np.arange(1, n + 1),
        "blank_1": [""] * n,
        "atom_name": atom_names,
        "alt_loc": [""] * n,
        "residue_name": residue_names,
        "blank_2": [""] * n,
        "chain_id": ["A"] * n,
        "residue_number": res_nums,
        "insertion": [""] * n,
        "blank_3": [""] * n,
        "x_coord": coords[:, 0],
        "y_coord": coords[:, 1],
        "z_coord": coords[:, 2],
        "occupancy": [1.0] * n,
        "b_factor": [0.0] * n,
        "blank_4": [""] * n,
        "segment_id": [""] * n,
        "element_symbol": element_symbols,
        "charge": [0] * n,
        "line_idx": np.arange(1, n + 1),
    })

    ppdb = PandasPdb()
    ppdb.df["ATOM"] = df
    ppdb.to_pdb(path=out_path, gz=False, append_newline=True)
    return out_path