# structure_io.py

"""Write predicted waters alongside kept (non-water) atoms to a PDB or CIF file.

Pure structure IO. Callers pass the non-water atoms to keep (protein + hets) and
the predicted water positions **in the same reference frame as those atoms**, and
get back a merged ``AtomArray`` or a written file. Undoing the model's
mean-centering and choosing which atoms to keep are the caller's job, so this
module stays free of dataset- and model-specific conventions.
"""

from __future__ import annotations

import biotite.structure as bts
import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, set_structure as _set_structure_cif


# Annotations every AtomArray carries; anything else on the kept atoms (e.g.
# b_factor, occupancy) is an "extra" the waters must mirror so concatenation works.
_MANDATORY = (
    "chain_id",
    "res_id",
    "ins_code",
    "res_name",
    "hetero",
    "atom_name",
    "element",
)


def merge_waters(
    atoms: bts.AtomArray,
    positions,
    *,
    chain_id: str | None = None,
    b_factor: float | None = None,
    occupancy: float = 1.0,
) -> bts.AtomArray:
    """Return ``atoms`` with predicted waters appended as ``HOH`` oxygens.

    Each row of ``positions`` becomes one ``O`` atom in its own ``HOH`` residue,
    in whatever frame ``positions`` are already given (no re-centering here).

    Args:
        atoms: Non-water atoms to keep (protein + hets), in the output frame.
        positions: (N, 3) predicted water coordinates; numpy array or tensor.
        chain_id: Chain for the waters. Defaults to a single character not already
            used by ``atoms``.
        b_factor: B-factor for every water. Defaults to the mean B-factor of any
            HOH already in ``atoms``, else the mean over ``atoms``, else 0.0.
            Only written when ``atoms`` carries a ``b_factor`` annotation.
        occupancy: Occupancy for every water, written only when ``atoms`` carries
            an ``occupancy`` annotation (biotite defaults it to 1.0 otherwise).

    Returns:
        ``atoms`` followed by the water atoms, as one AtomArray.
    """
    if chain_id is None:
        chain_id = _pick_unused_chain_id(atoms)
    if b_factor is None:
        b_factor = _default_b_factor(atoms)

    waters = _water_array(
        positions,
        chain_id=chain_id,
        b_factor=b_factor,
        occupancy=occupancy,
        template=atoms,
    )
    return atoms + waters


def write_structure(atoms: bts.AtomArray, output_path: str) -> None:
    """Write an AtomArray to a PDB or CIF file, dispatching on the extension.

    ``.cif`` writes mmCIF; anything else writes PDB.
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
    positions,
    *,
    chain_id: str,
    b_factor: float,
    occupancy: float,
    template: bts.AtomArray,
) -> bts.AtomArray:
    """Build oxygen-only HOH waters carrying exactly ``template``'s annotations."""
    coords = np.asarray(_to_numpy(positions), dtype=np.float32).reshape(-1, 3)
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

    # Mirror the kept atoms' extra fields so `template + waters` concatenates.
    for cat in template.get_annotation_categories():
        if cat in _MANDATORY:
            continue
        ref = template.get_annotation(cat)
        waters.add_annotation(cat, dtype=ref.dtype)
        if cat == "b_factor":
            fill = b_factor
        elif cat == "occupancy":
            fill = occupancy
        else:
            fill = _default_for(ref.dtype)
        waters.get_annotation(cat)[:] = fill
    return waters


def _pick_unused_chain_id(atoms: bts.AtomArray) -> str:
    """A single-character chain id not already used by ``atoms`` (falls back to 'W')."""
    used = set(atoms.chain_id.tolist()) if atoms.array_length() else set()
    for c in "WXYZUVTSRQ0123456789":
        if c not in used:
            return c
    return "W"


def _default_b_factor(atoms: bts.AtomArray) -> float:
    """Mean B-factor of any HOH in ``atoms``, else the overall mean, else 0.0."""
    if "b_factor" not in atoms.get_annotation_categories() or atoms.array_length() == 0:
        return 0.0
    hoh = atoms.b_factor[atoms.res_name == "HOH"]
    source = hoh if len(hoh) else atoms.b_factor
    return float(source.mean()) if len(source) else 0.0


def _default_for(dtype: np.dtype):
    """A neutral fill value for an annotation of the given dtype."""
    kind = np.dtype(dtype).kind
    if kind in ("U", "S"):
        return ""
    if kind == "b":
        return False
    return 0


def _to_numpy(positions):
    """Coerce a tensor or array-like of positions to a numpy array."""
    if hasattr(positions, "detach"):  # torch.Tensor
        return positions.detach().cpu().numpy()
    return np.asarray(positions)
