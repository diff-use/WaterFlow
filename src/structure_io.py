# structure_io.py

"""Merge predicted waters with the protein+het atoms and write the result to PDB or CIF."""

from __future__ import annotations

import biotite.structure as bts
import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, set_structure as _set_structure_cif


# Columns every AtomArray has. Any other column on the kept atoms (e.g. b_factor,
# occupancy) must be mirrored onto the waters so the two arrays concatenate.
_MANDATORY = (
    "chain_id",
    "res_id",
    "ins_code",
    "res_name",
    "hetero",
    "atom_name",
    "element",
)

# Default water B-factor: mean B of kept atoms within this radius (A), plus an
# offset since ordered waters move more than the atoms they touch.
_B_FACTOR_CONTACT_RADIUS = 5.0
_B_FACTOR_WATER_OFFSET = 10.0


def merge_waters(
    atoms: bts.AtomArray,
    positions,
    *,
    chain_id: str | None = None,
    b_factor: float | None = None,
    occupancy: float = 1.0,
) -> bts.AtomArray:
    """Return the kept atoms with predicted waters appended as HOH oxygens.

    Each row of positions becomes one O atom in its own HOH residue, in whatever
    frame positions are given (no re-centering here).

    Args:
        atoms: Non-water atoms to keep (protein + hets), in the output frame.
        positions: (N, 3) water coordinates; numpy array or tensor.
        chain_id: Chain for the waters. Defaults to an unused single character.
        b_factor: One B-factor for every water. Defaults to a per-water estimate
            from the nearby atoms (see _default_b_factors). Written only if atoms
            has a b_factor column.
        occupancy: Occupancy for every water. Written only if atoms has an
            occupancy column.

    Returns:
        The kept atoms followed by the waters, as one AtomArray.
    """
    coords = _to_coords(positions)
    if chain_id is None:
        chain_id = _pick_unused_chain_id(atoms)
    if b_factor is None:
        b_factor = _default_b_factors(atoms, coords)

    waters = _water_array(
        coords,
        chain_id=chain_id,
        b_factor=b_factor,
        occupancy=occupancy,
        template=atoms,
    )
    return atoms + waters


def write_structure(atoms: bts.AtomArray, output_path: str) -> None:
    """Write an AtomArray to PDB or CIF, chosen by the file extension.

    A .cif extension writes mmCIF; anything else writes PDB.
    """
    if str(output_path).endswith(".cif"):
        cif_file = CIFFile()
        _set_structure_cif(cif_file, atoms)
        cif_file.write(str(output_path))
    else:
        pdb_file = PDBFile()
        pdb_file.set_structure(atoms)
        pdb_file.write(str(output_path))


def _water_array(
    coords: np.ndarray,
    *,
    chain_id: str,
    b_factor,
    occupancy: float,
    template: bts.AtomArray,
) -> bts.AtomArray:
    """Build oxygen-only HOH waters with the same columns as template.

    b_factor may be a scalar or a per-water array.
    """
    n = len(coords)

    waters = bts.AtomArray(n)
    waters.coord = coords
    waters.chain_id[:] = chain_id
    waters.res_id[:] = np.arange(1, n + 1)
    waters.ins_code[:] = ""
    waters.res_name[:] = "HOH"
    waters.atom_name[:] = "O"
    waters.element[:] = "O"
    waters.hetero[:] = True

    # template + waters only concatenates if both have the same columns. Give the
    # waters every extra column the kept atoms carry so they line up. A new column
    # starts empty (0 / '' / False), which is fine except for b_factor and
    # occupancy, which get a real value.
    for cat in template.get_annotation_categories():
        if cat in _MANDATORY:
            continue
        waters.add_annotation(cat, dtype=template.get_annotation(cat).dtype)
        if cat == "b_factor":
            waters.get_annotation(cat)[:] = b_factor
        elif cat == "occupancy":
            waters.get_annotation(cat)[:] = occupancy
    return waters


def _pick_unused_chain_id(atoms: bts.AtomArray) -> str:
    """A single-character chain id not used by atoms (falls back to 'W')."""
    used = set(atoms.chain_id.tolist()) if atoms.array_length() else set()
    for c in "WXYZUVTSRQ0123456789":
        if c not in used:
            return c
    return "W"


def _default_b_factors(atoms: bts.AtomArray, coords: np.ndarray) -> np.ndarray:
    """Per-water default B-factor from the local environment.

    Each water takes the mean B-factor of the kept atoms near it, plus an offset.
    A water with no atom in range uses the overall mean. Returns zeros when atoms 
    has no b_factor.
    """

    if len(coords) == 0:
        return np.zeros(0, dtype=np.float32)
    if "b_factor" not in atoms.get_annotation_categories() or atoms.array_length() == 0:
        return np.zeros(len(coords), dtype=np.float32)

    b = atoms.b_factor
    cell_list = bts.CellList(atoms, cell_size=_B_FACTOR_CONTACT_RADIUS)
    within = cell_list.get_atoms(coords, radius=_B_FACTOR_CONTACT_RADIUS, as_mask=True)
    counts = within.sum(axis=1)
    local_mean = (within * b[None, :]).sum(axis=1) / np.maximum(counts, 1)
    base = np.where(counts > 0, local_mean, b.mean())
    return (base + _B_FACTOR_WATER_OFFSET).astype(np.float32)


def _to_coords(positions) -> np.ndarray:
    """Tensor or array positions to an (N, 3) float32 array."""
    if hasattr(positions, "detach"):  # torch.Tensor
        positions = positions.detach().cpu().numpy()
    return np.asarray(positions, dtype=np.float32).reshape(-1, 3)
