# structure_io.py

"""Merge predicted waters with the protein+het atoms and write the result to PDB or CIF."""

from __future__ import annotations

import string
from collections import namedtuple
from pathlib import Path

import biotite.structure as bts
import numpy as np
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import (
    CIFCategory,
    CIFFile,
    set_structure as _set_structure_cif,
)


# biotite's PDBFile.set_space_group wants an object with .space_group/.z_val
# (its SpaceGroupInfo namedtuple, which isn't importable); this matches it.
_SpaceGroup = namedtuple("_SpaceGroup", ["space_group", "z_val"])


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

# File suffixes (compared lower-case) that select the mmCIF writer.
_CIF_SUFFIXES = (".cif", ".mmcif")

# Single-character chain IDs to try for the waters, in preference order: 'W'
# (water convention) first, then the remaining letters and digits. Single-char
# only, since PDB chain IDs are one column. PDB/mmCIF allow A-Za-z0-9.
_CHAIN_ID_CANDIDATES = "W" + "".join(
    c
    for c in string.ascii_uppercase + string.ascii_lowercase + string.digits
    if c != "W"
)


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

    waters = _water_array(
        coords,
        chain_id=chain_id,
        b_factor=b_factor,
        occupancy=occupancy,
        template=atoms,
    )
    merged = atoms + waters
    # Keep the crystal frame's unit cell: the waters carry no box, so pin the
    # merged array's box to the kept atoms' rather than rely on concat behaviour.
    merged.box = None if atoms.box is None else atoms.box.copy()
    return merged


def write_structure(
    atoms: bts.AtomArray,
    output_path: str,
    *,
    space_group: str | None = None,
) -> None:
    """Write an AtomArray to PDB or CIF, chosen by the file extension.

    A .cif or .mmcif extension (matched case-insensitively) writes mmCIF;
    anything else writes PDB.

    Crystallographic information is preserved when available: the unit cell
    rides on ``atoms.box`` (written by biotite) and the frame is written
    verbatim. The space group is the one thing biotite's AtomArray drops, so
    pass ``space_group`` (the H-M symbol from read_space_group on the input) to
    keep it; without it biotite falls back to P1.

    Args:
        atoms: The structure to write (its ``box`` carries the unit cell).
        output_path: Destination path; the suffix selects the format.
        space_group: Hermann-Mauguin symbol, e.g. "P 1 21 1". Written only when
            given and the output carries a unit cell to attach it to.
    """
    suffix = Path(output_path).suffix.lower()
    if suffix in _CIF_SUFFIXES:
        cif_file = CIFFile()
        _set_structure_cif(cif_file, atoms)
        if space_group:
            cif_file.block["symmetry"] = CIFCategory(
                {"space_group_name_H-M": space_group}
            )
        cif_file.write(str(output_path))
    else:
        pdb_file = PDBFile()
        pdb_file.set_structure(atoms)
        if space_group:
            pdb_file.set_space_group(_SpaceGroup(space_group, 0))
        pdb_file.write(str(output_path))


def read_space_group(path: str) -> str | None:
    """Read the space group (Hermann-Mauguin symbol) from a PDB or CIF file.

    Pair with write_structure so an input's space group survives the round trip
    (the unit cell already rides on the parsed AtomArray's box; this is the part
    biotite drops).

    Args:
        path: Path to the source PDB or CIF file.

    Returns:
        The H-M symbol, or None when the file carries none (no CRYST1 record /
        no symmetry category).
    """
    suffix = Path(path).suffix.lower()
    if suffix in _CIF_SUFFIXES:
        block = CIFFile.read(str(path)).block
        if "symmetry" not in block or "space_group_name_H-M" not in block["symmetry"]:
            return None
        name = block["symmetry"]["space_group_name_H-M"].as_item()
    else:
        try:
            name = PDBFile.read(str(path)).get_space_group().space_group
        except Exception:
            return None
    name = (name or "").strip()
    return name or None


def _water_array(
    coords: np.ndarray,
    *,
    chain_id: str,
    b_factor,
    occupancy: float,
    template: bts.AtomArray,
) -> bts.AtomArray:
    """Build oxygen-only HOH waters with the same columns as template.

    b_factor may be a scalar, a per-water array, or None. When None, a per-water
    default is computed here (see _default_b_factors) -- but only if the template
    actually carries a b_factor column, so no work is done for a template without
    one.
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
            values = (
                _default_b_factors(template, coords) if b_factor is None else b_factor
            )
            waters.get_annotation(cat)[:] = values
        elif cat == "occupancy":
            waters.get_annotation(cat)[:] = occupancy
    return waters


def _pick_unused_chain_id(atoms: bts.AtomArray) -> str:
    """A single-character chain id not used by atoms.

    Tries every valid single-character ID (letters then digits, 'W' first).
    Raises ValueError if all are occupied, rather than reusing an existing chain
    (which would merge waters into it and risk duplicate residue identifiers).
    """
    used = set(atoms.chain_id.tolist()) if atoms.array_length() else set()
    for c in _CHAIN_ID_CANDIDATES:
        if c not in used:
            return c
    raise ValueError(
        "no unused single-character chain id available for the waters; "
        "all letters and digits are already occupied. Pass chain_id explicitly."
    )


def _default_b_factors(atoms: bts.AtomArray, coords: np.ndarray) -> np.ndarray:
    """Per-water default B-factor from the local environment.

    Each water takes the mean B-factor of the kept atoms within
    _B_FACTOR_CONTACT_RADIUS of it, plus _B_FACTOR_WATER_OFFSET. A water with no
    atom in range uses the overall mean.

    Args:
        atoms: Kept atoms carrying a b_factor annotation (the water template).
            Only called when that column is present, so no column check here.
        coords: (N, 3) water coordinates, in the same frame as atoms.

    Returns:
        (N,) float32 array of per-water B-factors; zeros when there are no
        waters or no atoms to average over.
    """
    if len(coords) == 0 or atoms.array_length() == 0:
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
