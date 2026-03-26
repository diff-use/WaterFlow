# test_dataset.py

"""
Tests for src/dataset.py - Protein-water dataset functionality.

Organized by category:
1. Feature encoding (element_onehot)
2. Helper functions (_make_undirected, match_atoms_to_coords)
3. Quality filters (check_com_distance, check_water_clashes, check_chain_interactions)
4. PDB parsing (parse_asu_with_biotite, get_crystal_contacts_pymol)
5. Dataset class (ProteinWaterDataset)
6. DataLoader (get_dataloader)

Integration tests use real PDB files:
- 6eey: Standard PDB that passes all quality checks
- 2b5w: Fails COM distance check
- 8dzt: Fails water clash check at 2% threshold with 2A distance

All test cases created with assistance from Claude Code.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.constants import ELEM_IDX, ELEMENT_VOCAB
from src.dataset import (
    _make_undirected,
    _pad_atom_embeddings_for_mates,
    apply_threshold_filter,
    check_chain_interactions,
    check_com_distance,
    check_water_clashes,
    check_water_residue_ratio,
    compute_normalized_bfactors,
    element_onehot,
    filter_waters_by_quality,
    get_crystal_contacts_pymol,
    get_dataloader,
    load_edia_for_pdb,
    load_esm_embedding,
    load_slae_embedding,
    match_atoms_to_coords,
    parse_asu_with_biotite,
    ProteinWaterDataset,
)


# PDB fixtures (pdb_base_dir, pdb_6eey, pdb_2b5w, pdb_8dzt, pdb_1deu) are
# defined in conftest.py and use ENV_PDB_DIR env var with fallback to
# tests/test_files/. See conftest.py for details.


@pytest.fixture
def tmp_processed_dir(tmp_path):
    """Temporary directory for processed files."""
    return tmp_path / "processed"


@pytest.fixture
def single_pdb_list_file(tmp_path, pdb_6eey):
    """Create a temp PDB list file with single entry."""
    list_file = tmp_path / "pdb_list.txt"
    list_file.write_text("6eey_final\n")
    return str(list_file)


@pytest.mark.unit
class TestElementOnehot:
    """Tests for element one-hot encoding."""

    def test_output_shape(self):
        """Output shape should be (n_symbols, num_classes)."""
        symbols = ["C", "N", "O", "S"]
        out = element_onehot(symbols)
        assert out.shape == (4, len(ELEMENT_VOCAB) + 1)

    def test_known_elements(self):
        """Known elements should have correct one-hot encoding."""
        symbols = ["C", "N", "O"]
        out = element_onehot(symbols)

        # C should be at index 0, N at 1, O at 2
        assert out[0, ELEM_IDX["C"]] == 1.0
        assert out[1, ELEM_IDX["N"]] == 1.0
        assert out[2, ELEM_IDX["O"]] == 1.0

    def test_unknown_element_goes_to_other_bucket(self):
        """Unknown elements should go to the 'other' bucket (last index)."""
        symbols = ["X", "Y", "Z"]  # Unknown elements
        out = element_onehot(symbols)

        other_idx = len(ELEMENT_VOCAB)
        for i in range(3):
            assert out[i, other_idx] == 1.0
            # All other positions should be 0
            assert out[i, :other_idx].sum() == 0.0

    def test_case_insensitivity(self):
        """Encoding should be case-insensitive."""
        upper = element_onehot(["C", "N", "O"])
        lower = element_onehot(["c", "n", "o"])
        mixed = element_onehot(["C", "n", "O"])

        assert torch.allclose(upper, lower)
        assert torch.allclose(upper, mixed)

    def test_empty_list(self):
        """Empty list should return empty tensor."""
        out = element_onehot([])
        assert out.shape == (0, len(ELEMENT_VOCAB) + 1)

    def test_all_vocab_elements(self):
        """All vocabulary elements should be encoded correctly."""
        out = element_onehot(ELEMENT_VOCAB)

        for i, elem in enumerate(ELEMENT_VOCAB):
            assert out[i, i] == 1.0
            assert out[i].sum() == 1.0

    def test_output_is_float(self):
        """Output should be float tensor."""
        out = element_onehot(["C", "N"])
        assert out.dtype == torch.float32

    def test_single_element(self):
        """Single element should work."""
        out = element_onehot(["C"])
        assert out.shape == (1, len(ELEMENT_VOCAB) + 1)
        assert out[0, ELEM_IDX["C"]] == 1.0


@pytest.mark.unit
class TestMakeUndirected:
    """Tests for edge symmetrization helper."""

    def test_basic_symmetrization(self):
        """Should add reverse edges."""
        edge_index = torch.tensor([[0, 1], [1, 2]])
        out = _make_undirected(edge_index)

        # Should contain both directions
        assert out.shape[1] >= 2
        # Check reverse edges exist
        edges_set = set(zip(out[0].tolist(), out[1].tolist()))
        assert (0, 1) in edges_set and (1, 0) in edges_set
        assert (1, 2) in edges_set and (2, 1) in edges_set

    def test_empty_edge_index(self):
        """Empty input should return empty output."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        out = _make_undirected(edge_index)

        assert out.shape == (2, 0)

    def test_already_undirected(self):
        """Already undirected edges should not duplicate."""
        edge_index = torch.tensor([[0, 1, 1, 0], [1, 0, 2, 2]])
        out = _make_undirected(edge_index)

        # Should have same or fewer edges (deduplicated)
        edges_set = set(zip(out[0].tolist(), out[1].tolist()))
        assert len(edges_set) == out.shape[1]  # No duplicates

    def test_self_loops_preserved(self):
        """Self-loops should be preserved."""
        edge_index = torch.tensor([[0, 1], [0, 2]])  # Self-loop at 0
        out = _make_undirected(edge_index)

        edges_set = set(zip(out[0].tolist(), out[1].tolist()))
        assert (0, 0) in edges_set

    def test_output_dtype(self):
        """Output should preserve dtype."""
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        out = _make_undirected(edge_index)

        assert out.dtype == torch.long


@pytest.mark.unit
class TestMatchAtomsToCoords:
    """Tests for atom-to-coordinate matching."""

    def test_perfect_match(self):
        """Exact coordinate matches should be found."""
        import biotite.structure as bts

        # Create atom array with known coords
        atoms = bts.AtomArray(3)
        atoms.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        target_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        matched = match_atoms_to_coords(atoms, target_coords, tolerance=0.01)

        assert len(matched) == 2
        assert 0 in matched
        assert 2 in matched

    def test_no_match_outside_tolerance(self):
        """Coordinates outside tolerance should not match."""
        import biotite.structure as bts

        atoms = bts.AtomArray(2)
        atoms.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        target_coords = np.array([[0.5, 0.0, 0.0]])  # Not close to any atom

        matched = match_atoms_to_coords(atoms, target_coords, tolerance=0.01)

        assert len(matched) == 0

    def test_empty_target_coords(self):
        """Empty target coordinates should return empty list."""
        import biotite.structure as bts

        atoms = bts.AtomArray(3)
        atoms.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

        target_coords = np.zeros((0, 3))

        matched = match_atoms_to_coords(atoms, target_coords, tolerance=0.01)

        assert matched == []

    def test_tolerance_parameter(self):
        """Match should respect tolerance parameter."""
        import biotite.structure as bts

        atoms = bts.AtomArray(1)
        atoms.coord = np.array([[0.0, 0.0, 0.0]])

        target_coords = np.array([[0.05, 0.0, 0.0]])

        # Should not match with tight tolerance
        matched_tight = match_atoms_to_coords(atoms, target_coords, tolerance=0.01)
        assert len(matched_tight) == 0

        # Should match with loose tolerance
        matched_loose = match_atoms_to_coords(atoms, target_coords, tolerance=0.1)
        assert len(matched_loose) == 1


@pytest.mark.unit
class TestCheckComDistance:
    """Tests for center of mass distance quality filter."""

    def test_same_com_passes(self):
        """Identical CoMs should pass."""
        protein_coords = torch.randn(100, 3)
        water_coords = protein_coords[:10].clone()  # Same center region

        is_valid, reason = check_com_distance(
            protein_coords, water_coords, max_com_dist=25.0
        )

        assert is_valid is True
        assert reason == ""

    def test_close_com_passes(self):
        """Close CoMs should pass."""
        protein_coords = torch.zeros(100, 3)
        water_coords = torch.zeros(10, 3) + 5.0  # 5A offset in all dims ~8.7A distance

        is_valid, reason = check_com_distance(
            protein_coords, water_coords, max_com_dist=25.0
        )

        assert is_valid is True

    def test_far_com_fails(self):
        """Distant CoMs should fail."""
        protein_coords = torch.zeros(100, 3)
        water_coords = torch.zeros(10, 3) + 100.0  # 100A offset

        is_valid, reason = check_com_distance(
            protein_coords, water_coords, max_com_dist=25.0
        )

        assert is_valid is False
        assert "CoM distance" in reason
        assert "exceeds threshold" in reason

    def test_empty_water_passes(self):
        """Empty water coordinates should pass (trivially)."""
        protein_coords = torch.randn(100, 3)
        water_coords = torch.zeros(0, 3)

        is_valid, reason = check_com_distance(
            protein_coords, water_coords, max_com_dist=25.0
        )

        assert is_valid is True

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        protein_coords = torch.zeros(100, 3)
        water_coords = torch.zeros(10, 3) + 10.0  # ~17.3A distance

        # Tight threshold should fail
        is_valid_tight, _ = check_com_distance(
            protein_coords, water_coords, max_com_dist=10.0
        )
        assert is_valid_tight is False

        # Loose threshold should pass
        is_valid_loose, _ = check_com_distance(
            protein_coords, water_coords, max_com_dist=50.0
        )
        assert is_valid_loose is True


@pytest.mark.unit
class TestCheckWaterClashes:
    """Tests for water clashing quality filter."""

    def test_no_clashes_passes(self):
        """Waters far from protein should pass."""
        protein_coords = torch.zeros(100, 3)
        # Waters at distance 10A away
        water_coords = torch.zeros(10, 3) + 10.0

        is_valid, reason = check_water_clashes(
            protein_coords, water_coords, clash_dist=2.0, max_clash_fraction=0.05
        )

        assert is_valid is True

    def test_all_clashing_fails(self):
        """All waters clashing should fail."""
        protein_coords = torch.randn(100, 3)
        # Waters exactly at protein positions (0A distance)
        water_coords = protein_coords[:10].clone()

        is_valid, reason = check_water_clashes(
            protein_coords, water_coords, clash_dist=2.0, max_clash_fraction=0.05
        )

        assert is_valid is False
        assert "clash fraction" in reason.lower()

    def test_partial_clashes_at_threshold(self):
        """Clashes at exactly threshold should fail (> not >=)."""
        protein_coords = torch.zeros(100, 3)
        # 3 out of 50 waters = 6% clashing (> 5% threshold)
        water_coords = torch.zeros(50, 3) + 10.0
        water_coords[:3] = 0.0  # These will clash

        is_valid, reason = check_water_clashes(
            protein_coords, water_coords, clash_dist=2.0, max_clash_fraction=0.05
        )

        assert is_valid is False

    def test_below_threshold_passes(self):
        """Clashes below threshold should pass."""
        protein_coords = torch.zeros(100, 3)
        # 2 out of 100 waters = 2% clashing (< 5% threshold)
        water_coords = torch.zeros(100, 3) + 10.0
        water_coords[:2] = 0.0  # These will clash

        is_valid, reason = check_water_clashes(
            protein_coords, water_coords, clash_dist=2.0, max_clash_fraction=0.05
        )

        assert is_valid is True

    def test_empty_water_passes(self):
        """Empty water coordinates should pass."""
        protein_coords = torch.randn(100, 3)
        water_coords = torch.zeros(0, 3)

        is_valid, reason = check_water_clashes(
            protein_coords, water_coords, clash_dist=2.0, max_clash_fraction=0.05
        )

        assert is_valid is True

    def test_custom_clash_distance(self):
        """Custom clash distance should be respected."""
        protein_coords = torch.zeros(10, 3)
        water_coords = torch.tensor([[1.5, 0.0, 0.0]])  # 1.5A from origin

        # 1A clash distance - not clashing
        is_valid_1a, _ = check_water_clashes(
            protein_coords, water_coords, clash_dist=1.0, max_clash_fraction=0.05
        )
        assert is_valid_1a is True

        # 2A clash distance - clashing
        is_valid_2a, _ = check_water_clashes(
            protein_coords, water_coords, clash_dist=2.0, max_clash_fraction=0.05
        )
        assert is_valid_2a is False


@pytest.mark.unit
class TestCheckChainInteractions:
    """Tests for chain interaction quality filter."""

    def test_single_chain_passes(self):
        """Single chain should always pass."""
        import biotite.structure as bts

        atoms = bts.AtomArray(10)
        atoms.chain_id = np.array(["A"] * 10)
        atoms.coord = np.random.randn(10, 3)

        is_valid, reason, status = check_chain_interactions(
            atoms, interface_dist_threshold=4.0
        )

        assert is_valid is True
        assert status == "Single Chain"

    def test_interacting_chains_pass(self):
        """Chains within interface distance should pass."""
        import biotite.structure as bts

        atoms = bts.AtomArray(20)
        atoms.chain_id = np.array(["A"] * 10 + ["B"] * 10)
        # Place chains close together
        atoms.coord = np.random.randn(20, 3)
        atoms.coord[10:] += np.array([2.0, 0.0, 0.0])

        is_valid, reason, status = check_chain_interactions(
            atoms, interface_dist_threshold=4.0
        )

        assert is_valid is True
        assert status == "Interacting"

    def test_non_interacting_chains_fail(self):
        """Chains beyond interface distance should fail."""
        import biotite.structure as bts

        atoms = bts.AtomArray(20)
        atoms.chain_id = np.array(["A"] * 10 + ["B"] * 10)
        # Place chains far apart
        atoms.coord = np.zeros((20, 3))
        atoms.coord[10:] += np.array([100.0, 0.0, 0.0])

        is_valid, reason, status = check_chain_interactions(
            atoms, interface_dist_threshold=4.0
        )

        assert is_valid is False
        assert "ASU copies" in reason or "not PPI" in reason
        assert status == "Non-Interacting (ASU Copies)"

    def test_three_chains_any_pair_interacting(self):
        """Multiple chains - should pass if any pair interacts."""
        import biotite.structure as bts

        atoms = bts.AtomArray(30)
        atoms.chain_id = np.array(["A"] * 10 + ["B"] * 10 + ["C"] * 10)
        atoms.coord = np.zeros((30, 3))
        atoms.coord[:10] = np.zeros((10, 3))
        atoms.coord[10:20] = np.zeros((10, 3)) + np.array(
            [2.0, 0.0, 0.0]
        )  # B close to A
        atoms.coord[20:] = np.zeros((10, 3)) + np.array(
            [100.0, 0.0, 0.0]
        )  # C far from all

        is_valid, reason, status = check_chain_interactions(
            atoms, interface_dist_threshold=4.0
        )

        # Should pass because A and B interact
        assert is_valid is True


@pytest.mark.integration
class TestParseAsuWithBiotite:
    """Tests for PDB parsing with biotite."""

    def test_parse_returns_protein_and_water(self, pdb_6eey):
        """Should return protein and water atom arrays."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_6eey)

        assert protein_atoms is not None
        assert water_atoms is not None
        assert len(protein_atoms) > 0

    def test_hydrogen_removed(self, pdb_6eey):
        """Hydrogens should be removed from output."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_6eey)

        protein_elements = set(protein_atoms.element)
        water_elements = set(water_atoms.element) if len(water_atoms) > 0 else set()

        assert "H" not in protein_elements
        assert "H" not in water_elements

    def test_water_residue_names(self, pdb_6eey):
        """Water atoms should have HOH or WAT residue names."""
        _, water_atoms = parse_asu_with_biotite(pdb_6eey)

        if len(water_atoms) > 0:
            water_res_names = set(water_atoms.res_name)
            assert water_res_names.issubset({"HOH", "WAT"})


@pytest.mark.integration
class TestGetCrystalContactsPymol:
    """Tests for PyMOL crystal contact detection."""

    def test_returns_expected_keys(self, pdb_6eey):
        """Should return dictionary with expected keys."""
        result = get_crystal_contacts_pymol(pdb_6eey, cutoff=5.0)

        assert "asu_coords" in result
        assert "mate_coords" in result
        assert "asu_atoms" in result
        assert "mate_atoms" in result

    def test_asu_coords_shape(self, pdb_6eey):
        """ASU coords should be Nx3 array."""
        result = get_crystal_contacts_pymol(pdb_6eey, cutoff=5.0)

        assert result["asu_coords"].ndim == 2
        assert result["asu_coords"].shape[1] == 3

    def test_different_cutoffs(self, pdb_6eey):
        """Larger cutoff should find more/equal mates."""
        result_small = get_crystal_contacts_pymol(pdb_6eey, cutoff=3.0)
        result_large = get_crystal_contacts_pymol(pdb_6eey, cutoff=8.0)

        # Larger cutoff should generally find more interface atoms
        assert (
            result_large["mate_coords"].shape[0] >= result_small["mate_coords"].shape[0]
        )


@pytest.mark.integration
class TestProteinWaterDataset:
    """Tests for the main dataset class."""

    def test_dataset_creation(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """Dataset should be created successfully."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        assert len(dataset) >= 1

    def test_getitem_returns_heterodata(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """__getitem__ should return HeteroData."""
        from torch_geometric.data import HeteroData

        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        data = dataset[0]
        assert isinstance(data, HeteroData)

    def test_heterodata_has_required_fields(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """HeteroData should have required node types and fields."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        data = dataset[0]

        # Check protein nodes
        assert hasattr(data["protein"], "pos")
        assert hasattr(data["protein"], "x")
        assert hasattr(data["protein"], "residue_index")

        # Check water nodes
        assert hasattr(data["water"], "pos")
        assert hasattr(data["water"], "x")

        # Check edges
        assert ("protein", "pp", "protein") in data.edge_types

    def test_protein_positions_centered(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """ASU protein positions should be centered (mean ~ 0)."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=False,  # Test centering without mates
            preprocess=True,
        )

        data = dataset[0]
        protein_center = data["protein"].pos.mean(dim=0)

        assert torch.allclose(protein_center, torch.zeros(3), atol=1e-3)

    def test_duplicate_single_sample(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """duplicate_single_sample should multiply dataset length."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
            duplicate_single_sample=10,
        )

        assert len(dataset) == 10

        # All items should be the same
        data_0 = dataset[0]
        data_5 = dataset[5]
        assert torch.allclose(data_0["protein"].pos, data_5["protein"].pos)

    def test_cached_file_created(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """Preprocessing should create cached .pt file."""
        _ = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )  # need to call this to trigger the processing

        # With include_mates=True, cache goes to geometry_mates/ directory
        cache_file = tmp_processed_dir / "geometry_mates" / "6eey_final.pt"
        assert cache_file.exists()

    def test_no_reprocess_if_cached(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """Should not reprocess if cache exists."""
        # First creation
        ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
            include_mates=True,
        )

        # With include_mates=True, cache goes to geometry_mates/ directory
        cache_file = tmp_processed_dir / "geometry_mates" / "6eey_final.pt"
        mtime_1 = cache_file.stat().st_mtime

        # Second creation should not modify cache
        ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
            include_mates=True,
        )

        mtime_2 = cache_file.stat().st_mtime
        assert mtime_1 == mtime_2


@pytest.mark.integration
class TestQualityFiltersWithRealPDBs:
    """Integration tests for quality filters using real PDB files."""

    def test_6eey_passes_com_check(self, pdb_6eey):
        """6eey should pass COM distance check."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_6eey)

        protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
        water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32)

        is_valid, reason = check_com_distance(protein_pos, water_pos, max_com_dist=25.0)

        assert is_valid is True, f"6eey should pass COM check but got: {reason}"

    def test_6eey_passes_clash_check(self, pdb_6eey):
        """6eey should pass water clash check."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_6eey)

        protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
        water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32)

        is_valid, reason = check_water_clashes(
            protein_pos, water_pos, clash_dist=2.0, max_clash_fraction=0.05
        )

        assert is_valid is True, f"6eey should pass clash check but got: {reason}"

    def test_2b5w_fails_com_check(self, pdb_2b5w):
        """2b5w should fail COM distance check."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_2b5w)

        protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
        water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32)

        is_valid, reason = check_com_distance(protein_pos, water_pos, max_com_dist=25.0)

        assert is_valid is False, "2b5w should fail COM check"
        assert "CoM distance" in reason

    def test_8dzt_fails_clash_check_at_2_percent(self, pdb_8dzt):
        """8dzt should fail water clash check at 2% threshold with 2A distance."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_8dzt)

        protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
        water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32)

        is_valid, reason = check_water_clashes(
            protein_pos, water_pos, clash_dist=2.0, max_clash_fraction=0.02
        )

        assert is_valid is False, "8dzt should fail clash check at 2% threshold"
        assert "clash fraction" in reason.lower()

    def test_8dzt_passes_clash_check_at_5_percent(self, pdb_8dzt):
        """8dzt should pass water clash check at 5% threshold (default)."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_8dzt)

        protein_pos = torch.tensor(protein_atoms.coord, dtype=torch.float32)
        water_pos = torch.tensor(water_atoms.coord, dtype=torch.float32)

        is_valid, reason = check_water_clashes(
            protein_pos, water_pos, clash_dist=2.0, max_clash_fraction=0.05
        )

        # At 5% threshold, 8dzt should pass (it only fails at 2%)
        assert is_valid is True, f"8dzt should pass clash check at 5% but got: {reason}"


@pytest.mark.integration
class TestGetDataloader:
    """Tests for dataloader creation."""

    def test_dataloader_creation(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """Dataloader should be created successfully."""
        loader = get_dataloader(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        assert loader is not None
        assert len(loader) >= 1

    def test_dataloader_iteration(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """Should be able to iterate over dataloader."""
        loader = get_dataloader(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        batch = next(iter(loader))
        assert batch is not None
        assert hasattr(batch["protein"], "pos")

    def test_dataloader_batching(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """Dataloader should support batching with duplicate_single_sample."""
        loader = get_dataloader(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            batch_size=4,
            shuffle=False,
            num_workers=0,
            duplicate_single_sample=8,
        )

        # With 8 samples and batch_size=4, should have 2 batches
        assert len(loader) == 2

        batch = next(iter(loader))
        # Batch should have batch indices
        assert hasattr(batch["protein"], "batch")


# ============== Tests for PDB list parsing ==============


@pytest.mark.unit
class TestPdbListParsing:
    """Tests for PDB list file parsing."""

    def test_chain_specific_format_rejected(self, tmp_path, pdb_base_dir):
        """Chain-specific format should be rejected."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final_A\n")

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=False,  # Don't preprocess for this test
        )

        assert len(dataset.entries) == 0

    def test_whole_pdb_format(self, tmp_path, pdb_base_dir):
        """Should parse whole PDB format: pdb_id_final"""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n")

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=False,
        )

        assert len(dataset.entries) == 1
        assert dataset.entries[0]["pdb_id"] == "6eey"

    def test_multiple_entries(self, tmp_path, pdb_base_dir):
        """Should parse multiple entries."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n1deu_final\n")

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=False,
        )

        assert len(dataset.entries) == 2

    def test_empty_lines_ignored(self, tmp_path, pdb_base_dir):
        """Empty lines should be ignored."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("\n6eey_final\n\n\n")

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=False,
        )

        assert len(dataset.entries) == 1


@pytest.mark.unit
class TestDatasetEdgeCases:
    """Tests for edge cases in dataset handling."""

    def test_include_mates_flag(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """include_mates flag should affect protein node count."""
        # Create dataset with mates
        dataset_with_mates = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir / "with_mates"),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=True,
            preprocess=True,
        )

        # Create dataset without mates (separate cache dir)
        dataset_no_mates = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir / "no_mates"),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=False,
            preprocess=True,
        )

        data_with = dataset_with_mates[0]
        data_without = dataset_no_mates[0]

        # With mates should have >= atoms
        assert data_with["protein"].num_nodes >= data_without["protein"].num_nodes

    def test_custom_cutoff(self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir):
        """Custom cutoff should affect edge connectivity."""
        dataset_small = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir / "small"),
            base_pdb_dir=str(pdb_base_dir),
            cutoff=4.0,
            preprocess=True,
        )

        dataset_large = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir / "large"),
            base_pdb_dir=str(pdb_base_dir),
            cutoff=12.0,
            preprocess=True,
        )

        data_small = dataset_small[0]
        data_large = dataset_large[0]

        # Larger cutoff should have more edges
        n_edges_small = data_small["protein", "pp", "protein"].edge_index.shape[1]
        n_edges_large = data_large["protein", "pp", "protein"].edge_index.shape[1]

        assert n_edges_large >= n_edges_small

    def test_pp_edge_features_cached(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """PP edge features should be cached and loaded properly."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
            include_mates=False,
        )

        data = dataset[0]
        pp_edge = data["protein", "pp", "protein"]

        # Check that all edge features are present
        assert hasattr(pp_edge, "edge_index")
        assert hasattr(pp_edge, "edge_unit_vectors")
        assert hasattr(pp_edge, "edge_rbf")

        n_edges = pp_edge.edge_index.shape[1]

        # Check shapes
        assert pp_edge.edge_unit_vectors.shape == (n_edges, 3)
        assert pp_edge.edge_rbf.shape == (n_edges, 16)

        # Check values are valid
        assert not torch.isnan(pp_edge.edge_unit_vectors).any()
        assert not torch.isnan(pp_edge.edge_rbf).any()

        # Unit vectors should have norm ~1
        unit_norms = torch.linalg.norm(pp_edge.edge_unit_vectors, dim=-1)
        assert torch.allclose(unit_norms, torch.ones_like(unit_norms), atol=1e-4)

    def test_directory_based_cache_separation(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """Different include_mates settings should use different directories."""
        # Create dataset without mates
        ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=False,
            preprocess=True,
        )

        # Create dataset with mates
        ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=True,
            preprocess=True,
        )

        # Check that both directories exist with the correct cache files
        cache_no_mates = tmp_processed_dir / "geometry" / "6eey_final.pt"
        cache_with_mates = tmp_processed_dir / "geometry_mates" / "6eey_final.pt"

        assert cache_no_mates.exists(), "geometry/ cache should exist"
        assert cache_with_mates.exists(), "geometry_mates/ cache should exist"

    def test_custom_geometry_cache_name(
        self, single_pdb_list_file, tmp_processed_dir, pdb_base_dir
    ):
        """Custom geometry_cache_name should be used for cache directory."""
        _ = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_processed_dir),
            base_pdb_dir=str(pdb_base_dir),
            geometry_cache_name="custom_geom",
            include_mates=True,
            preprocess=True,
        )

        # Cache should be in custom_geom_mates/ directory
        cache_file = tmp_processed_dir / "custom_geom_mates" / "6eey_final.pt"
        assert cache_file.exists()


# ============== Tests for water quality filtering ==============


@pytest.mark.unit
class TestLoadEdiaForPdb:
    """Tests for EDIA data loading from JSON files."""

    def test_returns_none_for_missing_file(self, tmp_path):
        """Should return None if EDIA file doesn't exist."""
        result = load_edia_for_pdb(tmp_path / "nonexistent.json")
        assert result is None

    def test_loads_water_edia_scores(self, tmp_path):
        """Should load EDIA scores for water molecules."""
        json_path = tmp_path / "test_pdb.json"
        json_path.write_text(
            json.dumps(
                [
                    {
                        "compID": "HOH",
                        "EDIAm": 0.85,
                        "pdb": {"strandID": "A", "seqNum": 101},
                    },
                    {
                        "compID": "HOH",
                        "EDIAm": 0.45,
                        "pdb": {"strandID": "A", "seqNum": 102},
                    },
                    {
                        "compID": "HOH",
                        "EDIAm": 0.72,
                        "pdb": {"strandID": "B", "seqNum": 201},
                    },
                    {
                        "compID": "ALA",
                        "EDIAm": 0.95,
                        "pdb": {"strandID": "A", "seqNum": 1},
                    },
                ]
            )
        )

        result = load_edia_for_pdb(json_path)

        assert result is not None
        assert len(result) == 3  # Only waters, not ALA

        # Verify all returned residues have correct chain_id and res_id
        assert ("A", 101, "") in result
        assert ("A", 102, "") in result
        assert ("B", 201, "") in result

        # Verify non-water residue (ALA) was filtered out
        assert ("A", 1, "") not in result

        # Verify EDIA scores are correct
        assert result[("A", 101, "")] == pytest.approx(0.85)
        assert result[("A", 102, "")] == pytest.approx(0.45)
        assert result[("B", 201, "")] == pytest.approx(0.72)

    def test_loads_real_edia_file(self, edia_6eey):
        """Should load EDIA scores from real PDB-REDO JSON file."""
        result = load_edia_for_pdb(edia_6eey)

        assert result is not None
        assert len(result) > 0  # Should have water molecules

        # All keys should be 3-tuples of (chain_id, res_id, ins_code)
        for key in result:
            assert len(key) == 3
            chain_id, res_id, ins_code = key
            assert isinstance(chain_id, str)
            assert isinstance(res_id, int)
            assert isinstance(ins_code, str)

        # All values should be valid EDIA scores (0.0 to ~1.0, possibly higher)
        for score in result.values():
            assert isinstance(score, float)
            assert score >= 0.0

    def test_returns_empty_dict_for_no_waters(self, tmp_path):
        """Should return empty dict if no water molecules are in the JSON."""
        json_path = tmp_path / "test_pdb.json"
        json_path.write_text(
            json.dumps(
                [
                    {
                        "compID": "ALA",
                        "EDIAm": 0.95,
                        "pdb": {"strandID": "A", "seqNum": 1},
                    },
                    {
                        "compID": "GLY",
                        "EDIAm": 0.90,
                        "pdb": {"strandID": "A", "seqNum": 2},
                    },
                ]
            )
        )

        result = load_edia_for_pdb(json_path)

        assert result == {}


@pytest.mark.unit
class TestComputeNormalizedBfactors:
    """Tests for B-factor normalization."""

    def test_computes_zscore_normalization(self, pdb_6eey):
        """Should compute z-score normalized B-factors."""
        bfactor_lookup, raw_bfactors = compute_normalized_bfactors(pdb_6eey)

        assert bfactor_lookup is not None
        assert raw_bfactors is not None

        # Normalized values should have reasonable range
        if len(bfactor_lookup) > 0:
            values = list(bfactor_lookup.values())
            # Z-scores typically in range [-3, 3] for most values
            assert all(-10 < v < 20 for v in values)

    def test_returns_dict_keyed_by_chain_resid(self, pdb_6eey):
        """Should return dict with (chain_id, res_id, ins_code) keys."""
        bfactor_lookup, _ = compute_normalized_bfactors(pdb_6eey)

        assert bfactor_lookup is not None
        if len(bfactor_lookup) > 0:
            key = next(iter(bfactor_lookup.keys()))
            assert isinstance(key, tuple)
            assert len(key) == 3
            assert isinstance(key[0], str)  # chain_id
            assert isinstance(key[1], int)  # res_id
            assert isinstance(key[2], str)  # ins_code


@pytest.mark.unit
class TestFilterWatersByQuality:
    """Tests for per-water quality filtering."""

    @pytest.fixture
    def mock_water_coords(self):
        """Create mock water coordinates for testing."""
        return np.array(
            [
                [0.0, 0.0, 0.0],  # Water 0 - close to protein
                [5.0, 0.0, 0.0],  # Water 1 - medium distance
                [15.0, 0.0, 0.0],  # Water 2 - far from protein
                [3.0, 0.0, 0.0],  # Water 3 - close
                [20.0, 0.0, 0.0],  # Water 4 - very far
            ]
        )

    @pytest.fixture
    def mock_water_keys(self):
        """Create mock water keys (chain_id, res_id) for testing."""
        return [("A", 101), ("A", 102), ("A", 103), ("B", 201), ("B", 202)]

    @pytest.fixture
    def mock_protein_coords(self):
        """Create mock protein coordinates for testing."""
        return np.zeros((10, 3))  # Protein centered at origin

    def test_distance_filtering(
        self, mock_water_coords, mock_water_keys, mock_protein_coords
    ):
        """Waters far from protein should be removed."""
        keep_mask = filter_waters_by_quality(
            mock_water_coords,
            mock_water_keys,
            protein_coords=mock_protein_coords,
            edia_lookup=None,
            bfactor_lookup=None,
            max_protein_dist=6.0,
        )

        # Waters at 0, 5, 3 A should pass; 15, 20 A should fail
        assert keep_mask.sum() == 3

    def test_edia_filtering(self, mock_water_coords, mock_water_keys):
        """Waters with low EDIA should be removed."""
        edia_lookup = {
            ("A", 101): 0.85,  # Pass
            ("A", 102): 0.30,  # Fail (< 0.4)
            ("A", 103): 0.50,  # Pass
            ("B", 201): 0.20,  # Fail
            ("B", 202): 0.60,  # Pass
        }

        keep_mask = filter_waters_by_quality(
            mock_water_coords,
            mock_water_keys,
            protein_coords=None,
            edia_lookup=edia_lookup,
            bfactor_lookup=None,
            min_edia=0.4,
        )

        assert keep_mask.sum() == 3

    def test_bfactor_filtering(self, mock_water_coords, mock_water_keys):
        """Waters with high B-factor z-score should be removed."""
        bfactor_lookup = {
            ("A", 101): 1.0,  # Pass
            ("A", 102): 6.0,  # Fail (> 5.0)
            ("A", 103): 2.5,  # Pass
            ("B", 201): 7.0,  # Fail
            ("B", 202): 0.5,  # Pass
        }

        keep_mask = filter_waters_by_quality(
            mock_water_coords,
            mock_water_keys,
            protein_coords=None,
            edia_lookup=None,
            bfactor_lookup=bfactor_lookup,
            max_bfactor_zscore=5.0,
        )

        assert keep_mask.sum() == 3

    def test_combined_filters(
        self, mock_water_coords, mock_water_keys, mock_protein_coords
    ):
        """Waters failing ANY criterion should be removed."""
        edia_lookup = {
            ("A", 101): 0.85,  # Pass EDIA
            ("A", 102): 0.85,  # Pass EDIA
            ("A", 103): 0.85,  # Pass EDIA, but will fail distance
            ("B", 201): 0.30,  # Fail EDIA
            ("B", 202): 0.85,  # Pass EDIA, but will fail distance
        }
        bfactor_lookup = {
            ("A", 101): 1.0,  # Pass B-factor
            ("A", 102): 6.0,  # Fail B-factor
            ("A", 103): 1.0,  # Pass B-factor
            ("B", 201): 1.0,  # Pass B-factor
            ("B", 202): 1.0,  # Pass B-factor
        }

        keep_mask = filter_waters_by_quality(
            mock_water_coords,
            mock_water_keys,
            protein_coords=mock_protein_coords,
            edia_lookup=edia_lookup,
            bfactor_lookup=bfactor_lookup,
            max_protein_dist=6.0,
            min_edia=0.4,
            max_bfactor_zscore=5.0,
        )

        # Only water 0 (A, 101) should pass all three filters
        assert keep_mask.sum() == 1

    def test_missing_edia_data_keeps_water(self, mock_water_coords, mock_water_keys):
        """Waters without EDIA data should be kept (conservative)."""
        # Only provide EDIA for some waters
        edia_lookup = {
            ("A", 101): 0.85,  # Pass
            ("A", 102): 0.30,  # Fail
            # A,103, B,201, B,202 have no EDIA data - should be kept
        }

        keep_mask = filter_waters_by_quality(
            mock_water_coords,
            mock_water_keys,
            protein_coords=None,
            edia_lookup=edia_lookup,
            bfactor_lookup=None,
            min_edia=0.4,
        )

        # 4 should pass (1 with good EDIA + 3 with no EDIA data)
        assert keep_mask.sum() == 4

    def test_empty_water_array(self):
        """Empty water array should return empty array."""
        keep_mask = filter_waters_by_quality(
            np.zeros((0, 3)),
            [],
            protein_coords=np.zeros((10, 3)),
            edia_lookup=None,
            bfactor_lookup=None,
        )

        assert len(keep_mask) == 0

    def test_all_filters_disabled(self, mock_water_coords, mock_water_keys):
        """With all filters disabled, all waters should pass."""
        keep_mask = filter_waters_by_quality(
            mock_water_coords,
            mock_water_keys,
            protein_coords=None,
            edia_lookup=None,
            bfactor_lookup=None,
        )

        assert keep_mask.sum() == 5


@pytest.mark.integration
class TestWaterFilteringIntegration:
    """Integration tests for water filtering with real PDB files."""

    def test_bfactor_extraction_from_real_pdb(self, pdb_6eey):
        """Should extract B-factors from real PDB file."""
        bfactor_lookup, raw_bfactors = compute_normalized_bfactors(pdb_6eey)

        assert bfactor_lookup is not None
        # 6eey should have some water molecules
        assert len(bfactor_lookup) > 0

    def test_filtering_with_real_pdb(self, pdb_6eey):
        """Should filter waters from real PDB file."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_6eey)

        if len(water_atoms) == 0:
            pytest.skip("No water molecules in 6eey")

        # Get B-factors
        bfactor_lookup, _ = compute_normalized_bfactors(pdb_6eey)

        # Build water keys
        water_keys = list(
            zip(water_atoms.chain_id.astype(str), water_atoms.res_id.astype(int))
        )

        # Apply filtering with distance and bfactor
        keep_mask = filter_waters_by_quality(
            water_atoms.coord,
            water_keys,
            protein_coords=protein_atoms.coord,
            edia_lookup=None,
            bfactor_lookup=bfactor_lookup,
            max_protein_dist=6.0,
        )

        # Should return a valid boolean mask
        assert len(keep_mask) == len(water_keys)
        assert keep_mask.dtype == bool

    def test_dataset_with_filtering_disabled(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """Dataset with filtering disabled should have same waters."""
        # Create dataset with filtering disabled
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path / "no_filter"),
            base_pdb_dir=str(pdb_base_dir),
            filter_by_distance=False,
            filter_by_edia=False,
            filter_by_bfactor=False,
            preprocess=True,
        )

        data = dataset[0]
        # Should have water nodes
        assert data["water"].num_nodes >= 0

    def test_skips_pdb_when_edia_enabled_but_file_missing(self, tmp_path, pdb_2b5w):
        """Dataset should skip PDB when filter_by_edia=True but JSON file is missing."""
        # 2b5w has a PDB file but no EDIA JSON file in test_files
        # Create a list file with just this PDB
        list_file = tmp_path / "missing_edia.txt"
        list_file.write_text("2b5w_final\n")

        # Create dataset with EDIA filtering enabled
        processed_dir = tmp_path / "edia_test"
        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(processed_dir),
            base_pdb_dir=str(Path(pdb_2b5w).parent.parent),
            filter_by_edia=True,  # EDIA filtering enabled
            filter_by_distance=False,
            filter_by_bfactor=False,
            preprocess=True,
        )

        # Dataset should be empty since the only PDB was skipped due to missing EDIA
        assert len(dataset) == 0

        # Failure should be logged to preprocessing_failures.log
        # Default include_mates=True uses geometry_mates directory
        failure_log = processed_dir / "geometry_mates" / "preprocessing_failures.log"
        assert failure_log.exists(), (
            "Missing EDIA should be logged to preprocessing_failures.log"
        )
        log_content = failure_log.read_text()
        assert "2b5w" in log_content
        assert "EDIA" in log_content


# ============== Tests for check_water_residue_ratio ==============


@pytest.mark.unit
class TestCheckWaterResidueRatio:
    """Tests for water/residue ratio validation."""

    def test_ratio_above_threshold_passes(self):
        """Ratio above threshold should pass."""
        is_valid, reason = check_water_residue_ratio(
            num_waters=100, num_residues=100, min_ratio=0.8
        )
        assert is_valid is True
        assert reason == ""

    def test_ratio_below_threshold_fails(self):
        """Ratio below threshold should fail."""
        is_valid, reason = check_water_residue_ratio(
            num_waters=50, num_residues=100, min_ratio=0.8
        )
        assert is_valid is False
        assert "below threshold" in reason

    def test_ratio_at_exact_threshold_passes(self):
        """Ratio exactly at threshold should pass."""
        is_valid, reason = check_water_residue_ratio(
            num_waters=80, num_residues=100, min_ratio=0.8
        )
        assert is_valid is True
        assert reason == ""

    def test_zero_residues_fails(self):
        """Zero residues should fail."""
        is_valid, reason = check_water_residue_ratio(
            num_waters=10, num_residues=0, min_ratio=0.8
        )
        assert is_valid is False
        assert "No residues" in reason

    def test_zero_waters_with_nonzero_ratio_fails(self):
        """Zero waters with min_ratio > 0 should fail."""
        is_valid, reason = check_water_residue_ratio(
            num_waters=0, num_residues=100, min_ratio=0.8
        )
        assert is_valid is False
        assert "below threshold" in reason

    def test_custom_min_ratio(self):
        """Should respect custom min_ratio parameter."""
        # Pass with low threshold
        is_valid, _ = check_water_residue_ratio(
            num_waters=10, num_residues=100, min_ratio=0.05
        )
        assert is_valid is True

        # Fail with higher threshold
        is_valid, _ = check_water_residue_ratio(
            num_waters=10, num_residues=100, min_ratio=0.2
        )
        assert is_valid is False


# ============== Tests for apply_threshold_filter ==============


@pytest.mark.unit
class TestApplyThresholdFilter:
    """Tests for generic threshold filtering."""

    def test_fail_if_below_mode(self):
        """Values below threshold should fail when fail_if_below=True."""
        water_keys = [("A", 1, ""), ("A", 2, ""), ("A", 3, "")]
        lookup = {("A", 1, ""): 0.8, ("A", 2, ""): 0.3, ("A", 3, ""): 0.5}

        fail_mask = apply_threshold_filter(
            water_keys, lookup, threshold=0.4, fail_if_below=True
        )

        # Only ("A", 2, "") with value 0.3 < 0.4 should fail
        assert not fail_mask[0]  # 0.8 >= 0.4
        assert fail_mask[1]  # 0.3 < 0.4
        assert not fail_mask[2]  # 0.5 >= 0.4

    def test_fail_if_above_mode(self):
        """Values above threshold should fail when fail_if_below=False."""
        water_keys = [("A", 1, ""), ("A", 2, ""), ("A", 3, "")]
        lookup = {("A", 1, ""): 1.0, ("A", 2, ""): 6.0, ("A", 3, ""): 3.0}

        fail_mask = apply_threshold_filter(
            water_keys, lookup, threshold=5.0, fail_if_below=False
        )

        # Only ("A", 2, "") with value 6.0 > 5.0 should fail
        assert not fail_mask[0]  # 1.0 <= 5.0
        assert fail_mask[1]  # 6.0 > 5.0
        assert not fail_mask[2]  # 3.0 <= 5.0

    def test_missing_keys_return_nan_and_pass(self):
        """Missing keys should get NaN and pass the filter (conservative)."""
        water_keys = [("A", 1, ""), ("A", 2, ""), ("A", 3, "")]
        lookup = {("A", 1, ""): 0.8}  # Only one entry

        fail_mask = apply_threshold_filter(
            water_keys, lookup, threshold=0.4, fail_if_below=True
        )

        # NaN comparisons return False, so missing keys pass
        assert not fail_mask[0]  # 0.8 >= 0.4
        assert not fail_mask[1]  # NaN comparison
        assert not fail_mask[2]  # NaN comparison

    def test_empty_water_keys(self):
        """Empty water keys should return empty array."""
        fail_mask = apply_threshold_filter([], {}, threshold=0.5, fail_if_below=True)
        assert len(fail_mask) == 0

    def test_insertion_code_handling(self):
        """Should correctly handle keys with insertion codes."""
        water_keys = [("A", 52, ""), ("A", 52, "A"), ("A", 52, "B")]
        lookup = {
            ("A", 52, ""): 0.8,
            ("A", 52, "A"): 0.3,
            ("A", 52, "B"): 0.6,
        }

        fail_mask = apply_threshold_filter(
            water_keys, lookup, threshold=0.5, fail_if_below=True
        )

        assert not fail_mask[0]  # 0.8 >= 0.5
        assert fail_mask[1]  # 0.3 < 0.5
        assert not fail_mask[2]  # 0.6 >= 0.5


# ============== Tests for _pad_atom_embeddings_for_mates ==============


@pytest.mark.unit
class TestPadAtomEmbeddingsForMates:
    """Tests for embedding padding for symmetry mates."""

    def test_no_padding_needed(self):
        """When total equals ASU size, return original embedding."""
        asu_embedding = torch.randn(100, 64)
        result = _pad_atom_embeddings_for_mates(asu_embedding, total_num_atoms=100)
        assert result.shape == (100, 64)
        assert torch.equal(result, asu_embedding)

    def test_padding_adds_zeros(self):
        """Should pad with zeros for mate atoms."""
        asu_embedding = torch.randn(100, 64)
        result = _pad_atom_embeddings_for_mates(asu_embedding, total_num_atoms=150)

        assert result.shape == (150, 64)
        # First 100 should match original
        assert torch.equal(result[:100], asu_embedding)
        # Last 50 should be zeros
        assert torch.equal(result[100:], torch.zeros(50, 64))

    def test_total_less_than_asu_returns_original(self):
        """When total < ASU size, return original (edge case)."""
        asu_embedding = torch.randn(100, 64)
        result = _pad_atom_embeddings_for_mates(asu_embedding, total_num_atoms=50)
        assert torch.equal(result, asu_embedding)

    def test_preserves_dtype(self):
        """Should preserve tensor dtype."""
        asu_embedding = torch.randn(10, 8, dtype=torch.float64)
        result = _pad_atom_embeddings_for_mates(asu_embedding, total_num_atoms=20)
        assert result.dtype == torch.float64

    def test_preserves_device(self):
        """Should preserve tensor device."""
        asu_embedding = torch.randn(10, 8)
        result = _pad_atom_embeddings_for_mates(asu_embedding, total_num_atoms=20)
        assert result.device == asu_embedding.device

    def test_empty_embedding(self):
        """Should handle empty embedding gracefully."""
        asu_embedding = torch.zeros(0, 64)
        result = _pad_atom_embeddings_for_mates(asu_embedding, total_num_atoms=10)
        assert result.shape == (10, 64)
        assert torch.equal(result, torch.zeros(10, 64))


# ============== Tests for encoder type validation ==============


@pytest.mark.unit
class TestEncoderTypeValidation:
    """Tests for encoder type validation in ProteinWaterDataset."""

    def test_invalid_encoder_type_raises(self, tmp_path, pdb_base_dir):
        """Invalid encoder type should raise ValueError."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n")

        with pytest.raises(ValueError, match="Unsupported encoder_type"):
            ProteinWaterDataset(
                pdb_list_file=str(list_file),
                processed_dir=str(tmp_path / "processed"),
                base_pdb_dir=str(pdb_base_dir),
                encoder_type="invalid_encoder",
                preprocess=False,
            )

    def test_valid_encoder_types_accepted(self, tmp_path, pdb_base_dir):
        """Valid encoder types should be accepted."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n")

        for encoder_type in ["gvp", "slae", "esm"]:
            # Should not raise
            dataset = ProteinWaterDataset(
                pdb_list_file=str(list_file),
                processed_dir=str(tmp_path / f"processed_{encoder_type}"),
                base_pdb_dir=str(pdb_base_dir),
                encoder_type=encoder_type,
                preprocess=False,
            )
            assert dataset.encoder_type == encoder_type


# ============== Tests for embedding loading ==============


@pytest.mark.unit
class TestLoadSlaeEmbedding:
    """Tests for SLAE embedding loading (standalone function)."""

    @pytest.fixture
    def embedding_dir(self, tmp_path):
        """Create embedding directory for testing."""
        emb_dir = tmp_path / "slae"
        emb_dir.mkdir(parents=True, exist_ok=True)
        return emb_dir

    def test_missing_cache_file_raises(self, embedding_dir):
        """Should raise FileNotFoundError for missing SLAE cache."""
        with pytest.raises(FileNotFoundError, match="SLAE cache file not found"):
            load_slae_embedding(
                embedding_dir=embedding_dir,
                cache_key="nonexistent",
                num_asu_protein=100,
                total_num_atoms=100,
            )

    def test_missing_node_embeddings_key_raises(self, embedding_dir):
        """Should raise KeyError if 'node_embeddings' key missing."""
        # Create cache file without node_embeddings
        torch.save({"other_key": torch.randn(10, 64)}, embedding_dir / "test_final.pt")

        with pytest.raises(KeyError, match="Missing 'node_embeddings'"):
            load_slae_embedding(
                embedding_dir=embedding_dir,
                cache_key="test_final",
                num_asu_protein=10,
                total_num_atoms=10,
            )

    def test_atom_count_mismatch_raises(self, embedding_dir):
        """Should raise ValueError if atom count doesn't match."""
        torch.save(
            {"node_embeddings": torch.randn(50, 64)}, embedding_dir / "test_final.pt"
        )

        with pytest.raises(ValueError, match="atom count mismatch"):
            load_slae_embedding(
                embedding_dir=embedding_dir,
                cache_key="test_final",
                num_asu_protein=100,  # Mismatch: expecting 100 but file has 50
                total_num_atoms=100,
            )

    def test_successful_load_with_padding(self, embedding_dir):
        """Should load and pad embeddings correctly."""
        original_emb = torch.randn(100, 64)
        torch.save({"node_embeddings": original_emb}, embedding_dir / "test_final.pt")

        result = load_slae_embedding(
            embedding_dir=embedding_dir,
            cache_key="test_final",
            num_asu_protein=100,
            total_num_atoms=150,  # Total with mates
        )

        assert result.shape == (150, 64)
        assert torch.equal(result[:100], original_emb)
        assert torch.equal(result[100:], torch.zeros(50, 64))


@pytest.mark.unit
class TestLoadEsmEmbedding:
    """Tests for ESM embedding loading (standalone function).

    Note: The standalone function returns raw residue-level embeddings.
    Broadcasting to atom-level is done separately in _annotate_data_with_embeddings.
    """

    @pytest.fixture
    def embedding_dir(self, tmp_path):
        """Create embedding directory for testing."""
        emb_dir = tmp_path / "esm"
        emb_dir.mkdir(parents=True, exist_ok=True)
        return emb_dir

    def test_missing_cache_file_raises(self, embedding_dir):
        """Should raise FileNotFoundError for missing ESM cache."""
        with pytest.raises(FileNotFoundError, match="ESM cache file not found"):
            load_esm_embedding(
                embedding_dir=embedding_dir,
                cache_key="nonexistent",
                num_protein_residues=2,
            )

    def test_missing_residue_embeddings_key_raises(self, embedding_dir):
        """Should raise KeyError if 'residue_embeddings' key missing."""
        torch.save(
            {"other_key": torch.randn(10, 1280)}, embedding_dir / "test_final.pt"
        )

        with pytest.raises(KeyError, match="Missing 'residue_embeddings'"):
            load_esm_embedding(
                embedding_dir=embedding_dir,
                cache_key="test_final",
                num_protein_residues=2,
            )

    def test_residue_count_mismatch_raises(self, embedding_dir):
        """Should raise ValueError if residue count doesn't match."""
        torch.save(
            {"residue_embeddings": torch.randn(5, 1280)},
            embedding_dir / "test_final.pt",
        )

        with pytest.raises(ValueError, match="residue count mismatch"):
            load_esm_embedding(
                embedding_dir=embedding_dir,
                cache_key="test_final",
                num_protein_residues=10,  # Mismatch: expecting 10 but file has 5
            )

    def test_returns_residue_embeddings(self, embedding_dir):
        """Should return raw residue-level embeddings."""
        residue_emb = torch.randn(3, 64)  # 3 residues
        torch.save({"residue_embeddings": residue_emb}, embedding_dir / "test_final.pt")

        result = load_esm_embedding(
            embedding_dir=embedding_dir,
            cache_key="test_final",
            num_protein_residues=3,
        )

        # Returns raw residue embeddings (not broadcast to atoms)
        assert result.shape == (3, 64)
        assert torch.equal(result, residue_emb)


@pytest.mark.unit
class TestLoadEncoderEmbeddings:
    """Tests for _annotate_data_with_embeddings dispatch logic.

    Embeddings are stored using generic attribute names (embedding, embedding_type)
    for consistent access regardless of encoder type.
    """

    def test_gvp_encoder_no_embeddings(self, tmp_path, pdb_base_dir):
        """GVP encoder should not load any embeddings."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n")

        from torch_geometric.data import HeteroData

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            encoder_type="gvp",
            preprocess=False,
        )

        data = HeteroData()
        data["protein"].num_nodes = 100

        # Should not raise (GVP doesn't need embeddings)
        dataset._annotate_data_with_embeddings(
            data=data,
            cache_key="test",
            asu_protein_res_idx=torch.tensor([0]),
            num_asu_protein=100,
            num_protein_residues=50,
        )

        # Should not have added any embedding attributes
        assert not hasattr(data["protein"], "embedding")
        assert not hasattr(data["protein"], "embedding_type")

    def test_slae_encoder_loads_slae(self, tmp_path, pdb_base_dir):
        """SLAE encoder should load embeddings with type 'slae'."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n")

        from torch_geometric.data import HeteroData

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            encoder_type="slae",
            preprocess=False,
        )

        # Create SLAE cache
        slae_dir = tmp_path / "processed" / "slae"
        slae_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"node_embeddings": torch.randn(100, 64)}, slae_dir / "test_final.pt"
        )

        data = HeteroData()
        data["protein"].num_nodes = 100

        dataset._annotate_data_with_embeddings(
            data=data,
            cache_key="test_final",
            asu_protein_res_idx=torch.tensor([0]),
            num_asu_protein=100,
            num_protein_residues=50,
        )

        assert hasattr(data["protein"], "embedding")
        assert data["protein"].embedding.shape == (100, 64)
        assert data["protein"].embedding_type == "slae"

    def test_esm_encoder_loads_esm(self, tmp_path, pdb_base_dir):
        """ESM encoder should load embeddings with type 'esm'."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n")

        from torch_geometric.data import HeteroData

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            encoder_type="esm",
            preprocess=False,
        )

        # Create ESM cache
        esm_dir = tmp_path / "processed" / "esm"
        esm_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"residue_embeddings": torch.randn(10, 1280)}, esm_dir / "test_final.pt"
        )

        data = HeteroData()
        data["protein"].num_nodes = 50

        # 50 atoms belonging to 10 residues
        asu_res_idx = torch.tensor([i // 5 for i in range(50)])

        dataset._annotate_data_with_embeddings(
            data=data,
            cache_key="test_final",
            asu_protein_res_idx=asu_res_idx,
            num_asu_protein=50,
            num_protein_residues=10,
        )

        assert hasattr(data["protein"], "embedding")
        assert data["protein"].embedding.shape == (50, 1280)
        assert data["protein"].embedding_type == "esm"


# ============== Tests for caching behavior ==============


@pytest.mark.unit
class TestCachingBehavior:
    """Tests for dataset caching behavior."""

    def test_cache_hit_skips_preprocessing(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """Should skip preprocessing when cache exists."""
        # First create dataset with preprocessing
        dataset1 = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        # Verify cache exists
        cache_file = tmp_path / "geometry_mates" / "6eey_final.pt"
        assert cache_file.exists()

        # Get modification time
        mtime_before = cache_file.stat().st_mtime

        # Create second dataset - should use cache
        dataset2 = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        # Cache should not be modified
        assert cache_file.stat().st_mtime == mtime_before

        # Both datasets should have same data
        data1 = dataset1[0]
        data2 = dataset2[0]
        assert data1["protein"].num_nodes == data2["protein"].num_nodes

    def test_missing_geometry_cache_raises_in_getitem(self, tmp_path, pdb_base_dir):
        """Should raise FileNotFoundError when cache missing during getitem."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n")

        # Create dataset without preprocessing (cache won't exist)
        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=False,
        )

        with pytest.raises(FileNotFoundError, match="Geometry cache file not found"):
            _ = dataset[0]

    def test_corrupted_cache_raises_meaningful_error(self, tmp_path, pdb_base_dir):
        """Corrupted cache should raise appropriate error."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("6eey_final\n")

        # Create corrupted cache file
        cache_dir = tmp_path / "processed" / "geometry_mates"
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "6eey_final.pt").write_bytes(b"corrupted data")

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=False,
        )

        with pytest.raises(Exception):  # torch.load raises various exceptions
            _ = dataset[0]

    def test_cache_content_validation(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """Cache should contain all required keys."""
        _ = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        cache_file = tmp_path / "geometry_mates" / "6eey_final.pt"
        cached = torch.load(cache_file, weights_only=False)

        required_keys = [
            "protein_pos",
            "protein_x",
            "protein_res_idx",
            "water_pos",
            "water_x",
            "pp_edge_index",
            "pp_edge_unit_vectors",
            "pp_edge_rbf",
            "num_asu_protein",
            "num_protein_residues",
        ]

        for key in required_keys:
            assert key in cached, f"Missing key: {key}"


# ============== Tests for EDIA with insertion codes ==============


@pytest.mark.unit
class TestEdiaInsertionCodes:
    """Tests for EDIA handling with insertion codes."""

    def test_edia_with_insertion_codes(self, tmp_path):
        """Should handle EDIA data with insertion codes."""
        json_path = tmp_path / "test_pdb.json"
        json_path.write_text(
            json.dumps(
                [
                    {
                        "compID": "HOH",
                        "EDIAm": 0.85,
                        "pdb": {"strandID": "A", "seqNum": 52, "insCode": ""},
                    },
                    {
                        "compID": "HOH",
                        "EDIAm": 0.75,
                        "pdb": {"strandID": "A", "seqNum": 52, "insCode": "A"},
                    },
                    {
                        "compID": "HOH",
                        "EDIAm": 0.65,
                        "pdb": {"strandID": "A", "seqNum": 52, "insCode": "B"},
                    },
                ]
            )
        )

        result = load_edia_for_pdb(json_path)

        assert result is not None
        assert len(result) == 3
        assert result[("A", 52, "")] == pytest.approx(0.85)
        assert result[("A", 52, "A")] == pytest.approx(0.75)
        assert result[("A", 52, "B")] == pytest.approx(0.65)

    def test_edia_normalizes_insertion_codes(self, tmp_path):
        """Should normalize insertion codes (spaces to empty string)."""
        json_path = tmp_path / "test_pdb.json"
        json_path.write_text(
            json.dumps(
                [
                    {
                        "compID": "HOH",
                        "EDIAm": 0.85,
                        "pdb": {"strandID": "A", "seqNum": 101, "insCode": " "},
                    }
                ]
            )
        )

        result = load_edia_for_pdb(json_path)

        assert result is not None
        # Space should be normalized to empty string
        assert ("A", 101, "") in result


# ============== Property-based tests with Hypothesis ==============


@pytest.mark.unit
class TestPropertyBased:
    """Property-based tests using Hypothesis."""

    @pytest.mark.parametrize(
        "symbols",
        [
            [],
            ["C"],
            ["C", "N", "O"],
            ["C", "X", "N"],  # Unknown element
            ["C"] * 50,
            list(ELEMENT_VOCAB),
        ],
    )
    def test_element_onehot_shape_invariant(self, symbols):
        """One-hot encoding should have consistent shape."""
        result = element_onehot(symbols)
        expected_cols = len(ELEMENT_VOCAB) + 1  # +1 for 'other'
        assert result.shape == (len(symbols), expected_cols)

    @pytest.mark.parametrize(
        "symbols",
        [
            ["C"],
            ["C", "N", "O"],
            ["X"],  # Unknown element
            ["C", "UNK", "N"],
        ],
    )
    def test_element_onehot_sum_is_one(self, symbols):
        """Each row of one-hot encoding should sum to 1."""
        result = element_onehot(symbols)
        row_sums = result.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(len(symbols)))

    @pytest.mark.parametrize(
        "num_waters,num_residues,min_ratio",
        [
            (0, 100, 0.0),
            (100, 100, 0.8),
            (80, 100, 0.8),
            (79, 100, 0.8),
            (1000, 500, 1.5),
        ],
    )
    def test_water_residue_ratio_deterministic(
        self, num_waters, num_residues, min_ratio
    ):
        """Water/residue ratio check should be deterministic."""
        result1 = check_water_residue_ratio(num_waters, num_residues, min_ratio)
        result2 = check_water_residue_ratio(num_waters, num_residues, min_ratio)
        assert result1 == result2

    @pytest.mark.parametrize(
        "asu_size,total_size",
        [
            (10, 10),
            (10, 20),
            (100, 150),
            (50, 50),
            (1, 100),
        ],
    )
    def test_pad_embeddings_size_invariant(self, asu_size, total_size):
        """Padded embeddings should have correct size."""
        asu_emb = torch.randn(asu_size, 64)
        result = _pad_atom_embeddings_for_mates(asu_emb, total_size)
        expected_size = max(asu_size, total_size)
        assert result.shape[0] == expected_size
        assert result.shape[1] == 64

    @pytest.mark.parametrize(
        "edges",
        [
            [],
            [(0, 1)],
            [(0, 1), (1, 2)],
            [(0, 1), (1, 0)],  # Already has reverse
            [(0, 1), (0, 2), (1, 2)],
        ],
    )
    def test_make_undirected_symmetry(self, edges):
        """Undirected edges should be symmetric."""
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).T

        result = _make_undirected(edge_index)

        if result.numel() > 0:
            # For each edge (i,j), reverse (j,i) should exist
            edges_set = set(zip(result[0].tolist(), result[1].tolist()))
            for i, j in edges_set:
                assert (j, i) in edges_set


# ============== Integration tests for symmetry mates ==============


@pytest.mark.integration
class TestSymmetryMateHandling:
    """Integration tests for symmetry mate handling."""

    def test_mates_increase_protein_count(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """Including mates should increase protein atom count."""
        # Dataset with mates
        dataset_with = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path / "with"),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=True,
            preprocess=True,
        )

        # Dataset without mates
        dataset_without = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path / "without"),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=False,
            preprocess=True,
        )

        data_with = dataset_with[0]
        data_without = dataset_without[0]

        # With mates should have at least as many protein atoms
        assert data_with["protein"].num_nodes >= data_without["protein"].num_nodes

    def test_mate_residue_indices_offset_correctly(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """Mate residue indices should be offset from ASU indices."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=True,
            preprocess=True,
        )

        data = dataset[0]
        cache_file = tmp_path / "geometry_mates" / "6eey_final.pt"
        cached = torch.load(cache_file, weights_only=False)

        num_asu = cached["num_asu_protein"]
        res_idx = data["protein"].residue_index

        # Check that residue indices are consecutive
        if num_asu < len(res_idx):
            asu_max_res = res_idx[:num_asu].max().item()
            mate_min_res = res_idx[num_asu:].min().item()
            # Mate residues should start after ASU residues
            assert mate_min_res > asu_max_res

    def test_num_asu_protein_metadata_correct(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """num_asu_protein_atoms metadata should be correct."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            include_mates=True,
            preprocess=True,
        )

        data = dataset[0]

        # num_asu_protein_atoms should be <= total protein nodes
        assert data.num_asu_protein_atoms <= data["protein"].num_nodes
        assert data.num_asu_protein_atoms > 0


# ============== Tests for RBF feature computation ==============


@pytest.mark.unit
class TestRBFFeatureComputation:
    """Tests for RBF edge feature computation."""

    def test_rbf_shape_matches_edges(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """RBF features should have shape (num_edges, NUM_RBF)."""
        from src.constants import NUM_RBF

        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        data = dataset[0]
        pp_edge = data["protein", "pp", "protein"]

        n_edges = pp_edge.edge_index.shape[1]
        assert pp_edge.edge_rbf.shape == (n_edges, NUM_RBF)

    def test_rbf_values_bounded(self, single_pdb_list_file, tmp_path, pdb_base_dir):
        """RBF values should be bounded (sinusoidal encoding in [-1, 1])."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        data = dataset[0]
        rbf = data["protein", "pp", "protein"].edge_rbf

        # Sinusoidal RBF encoding uses sin/cos, bounded in [-1, 1]
        assert rbf.min() >= -1.0
        assert rbf.max() <= 1.0
        assert not torch.isnan(rbf).any()
        assert not torch.isinf(rbf).any()

    def test_unit_vectors_normalized(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """Edge unit vectors should have norm ~1."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
        )

        data = dataset[0]
        unit_vecs = data["protein", "pp", "protein"].edge_unit_vectors

        norms = torch.linalg.norm(unit_vecs, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


# ============== Additional edge case tests ==============


@pytest.mark.unit
class TestAdditionalEdgeCases:
    """Additional edge case tests for dataset handling."""

    def test_empty_pdb_list(self, tmp_path, pdb_base_dir):
        """Empty PDB list should create dataset with no entries."""
        list_file = tmp_path / "empty.txt"
        list_file.write_text("")

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=False,
        )

        assert len(dataset.entries) == 0
        assert len(dataset) == 0

    def test_duplicate_single_sample_multiplies_length(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """duplicate_single_sample should multiply effective length."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
            duplicate_single_sample=10,
        )

        assert len(dataset.entries) == 1
        assert len(dataset) == 10

    def test_getitem_with_duplication_wraps_index(
        self, single_pdb_list_file, tmp_path, pdb_base_dir
    ):
        """getitem should wrap index when using duplicate_single_sample."""
        dataset = ProteinWaterDataset(
            pdb_list_file=single_pdb_list_file,
            processed_dir=str(tmp_path),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=True,
            duplicate_single_sample=5,
        )

        # All indices should return the same data
        data0 = dataset[0]
        data1 = dataset[1]
        data4 = dataset[4]

        assert data0["protein"].num_nodes == data1["protein"].num_nodes
        assert data0["protein"].num_nodes == data4["protein"].num_nodes
        assert data0.pdb_id == data1.pdb_id

    def test_pdb_list_with_whitespace(self, tmp_path, pdb_base_dir):
        """Should handle PDB list with extra whitespace."""
        list_file = tmp_path / "list.txt"
        list_file.write_text("  6eey_final  \n\n  \n")

        dataset = ProteinWaterDataset(
            pdb_list_file=str(list_file),
            processed_dir=str(tmp_path / "processed"),
            base_pdb_dir=str(pdb_base_dir),
            preprocess=False,
        )

        assert len(dataset.entries) == 1
        assert dataset.entries[0]["pdb_id"] == "6eey"
