"""Unit tests for src/structure_io.py -- merging predicted waters and writing PDB/CIF."""

import biotite.structure as bts
import numpy as np
import pytest
import torch
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure as get_structure_cif

from src.structure_io import merge_waters, write_structure


def _make_protein(n=4, chain="A", b_factor=20.0, extra_occupancy=False):
    """A minimal protein AtomArray shaped like parse_asu_with_biotite's output
    (carries a b_factor field), optionally with an occupancy field too."""
    atoms = bts.AtomArray(n)
    atoms.coord = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    atoms.chain_id[:] = chain
    atoms.res_id[:] = np.arange(1, n + 1)
    atoms.ins_code[:] = ""
    atoms.res_name[:] = "ALA"
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    atoms.hetero[:] = False
    atoms.add_annotation("b_factor", dtype=float)
    atoms.b_factor[:] = b_factor
    if extra_occupancy:
        atoms.add_annotation("occupancy", dtype=float)
        atoms.occupancy[:] = 1.0
    return atoms


def _read_back(path):
    if str(path).endswith(".cif"):
        return get_structure_cif(CIFFile.read(str(path)), model=1)
    return PDBFile.read(str(path)).get_structure(model=1)


@pytest.mark.unit
class TestMergeWaters:
    def test_appends_waters_as_hoh_oxygens(self):
        prot = _make_protein(4)
        pos = np.array([[10.0, 0, 0], [11.0, 0, 0]], dtype=np.float32)

        merged = merge_waters(prot, pos)

        assert merged.array_length() == 4 + 2
        w = merged[merged.res_name == "HOH"]
        assert w.array_length() == 2
        assert (w.element == "O").all()
        assert (w.atom_name == "O").all()
        assert w.hetero.all()
        # positions written verbatim -- no re-centering in the IO layer
        assert np.allclose(w.coord, pos)
        # kept atoms are untouched and come first
        assert (merged[:4].res_name == "ALA").all()

    def test_water_chain_avoids_collision(self):
        prot = _make_protein(chain="A")
        merged = merge_waters(prot, np.zeros((1, 3), dtype=np.float32))
        assert merged[merged.res_name == "HOH"].chain_id[0] == "W"

        # when W is taken, the picker moves on
        prot_w = _make_protein(chain="W")
        merged_w = merge_waters(prot_w, np.zeros((1, 3), dtype=np.float32))
        assert merged_w[merged_w.res_name == "HOH"].chain_id[0] != "W"

    def test_water_res_ids_are_sequential(self):
        prot = _make_protein()
        merged = merge_waters(prot, np.zeros((3, 3), dtype=np.float32))
        assert merged[merged.res_name == "HOH"].res_id.tolist() == [1, 2, 3]

    def test_b_factor_defaults_to_local_mean_plus_offset(self):
        # Two atoms within 5 A of the water (B = 10, 30) and one far away (B = 80);
        # the water should take mean(10, 30) + 10.0.
        prot = bts.AtomArray(3)
        prot.coord = np.array([[0, 0, 0], [2, 0, 0], [100, 0, 0]], dtype=np.float32)
        prot.chain_id[:] = "A"
        prot.res_id[:] = [1, 2, 3]
        prot.ins_code[:] = ""
        prot.res_name[:] = "ALA"
        prot.atom_name[:] = "CA"
        prot.element[:] = "C"
        prot.hetero[:] = False
        prot.add_annotation("b_factor", dtype=float)
        prot.b_factor[:] = [10.0, 30.0, 80.0]

        merged = merge_waters(prot, np.array([[1.0, 0, 0]], dtype=np.float32))
        assert np.allclose(merged[merged.res_name == "HOH"].b_factor, 20.0 + 10.0)

    def test_b_factor_falls_back_to_global_mean_when_isolated(self):
        # A water with no atom within 5 A uses the overall mean + offset.
        prot = _make_protein(b_factor=42.0)
        merged = merge_waters(prot, np.array([[1000.0, 0, 0]], dtype=np.float32))
        assert np.allclose(merged[merged.res_name == "HOH"].b_factor, 42.0 + 10.0)

    def test_explicit_b_factor_overrides_default(self):
        prot = _make_protein(b_factor=42.0)
        merged = merge_waters(prot, np.zeros((1, 3), dtype=np.float32), b_factor=5.0)
        assert np.allclose(merged[merged.res_name == "HOH"].b_factor, 5.0)

    def test_accepts_torch_tensor_positions(self):
        prot = _make_protein()
        pos = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        merged = merge_waters(prot, pos)
        assert np.allclose(merged[merged.res_name == "HOH"].coord, pos.numpy())

    def test_empty_positions_returns_atoms_unchanged(self):
        prot = _make_protein(4)
        merged = merge_waters(prot, np.zeros((0, 3), dtype=np.float32))
        assert merged.array_length() == 4
        assert (merged.res_name == "HOH").sum() == 0

    def test_matches_extra_annotations_for_concat(self):
        # atoms carrying occupancy (an extra field) must still merge cleanly,
        # with waters mirroring the field.
        prot = _make_protein(extra_occupancy=True)
        merged = merge_waters(prot, np.zeros((2, 3), dtype=np.float32), occupancy=0.5)
        w = merged[merged.res_name == "HOH"]
        assert "occupancy" in merged.get_annotation_categories()
        assert np.allclose(w.occupancy, 0.5)


@pytest.mark.unit
class TestWriteStructure:
    @pytest.mark.parametrize("ext", [".pdb", ".cif"])
    def test_round_trip_preserves_waters(self, tmp_path, ext):
        prot = _make_protein(5)
        pos = np.array([[20.0, 1, 2], [21.0, 3, 4]], dtype=np.float32)
        merged = merge_waters(prot, pos)

        out = tmp_path / f"pred{ext}"
        write_structure(merged, str(out))
        assert out.exists()

        back = _read_back(out)
        assert back.array_length() == 7
        w = back[back.res_name == "HOH"]
        assert w.array_length() == 2
        assert np.allclose(np.sort(w.coord[:, 0]), [20.0, 21.0])

    def test_extension_selects_format(self, tmp_path):
        prot = _make_protein(2)
        merged = merge_waters(prot, np.zeros((1, 3), dtype=np.float32))

        pdb_path = tmp_path / "s.pdb"
        cif_path = tmp_path / "s.cif"
        write_structure(merged, str(pdb_path))
        write_structure(merged, str(cif_path))

        assert (
            pdb_path.read_text()
            .lstrip()
            .startswith(("HEADER", "ATOM", "HETATM", "CRYST", "MODEL"))
        )
        assert cif_path.read_text().lstrip().startswith(("data_", "#"))
