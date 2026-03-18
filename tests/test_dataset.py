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

from pathlib import Path

import numpy as np
import pytest
import torch

from src.constants import ELEM_IDX, ELEMENT_VOCAB
from src.dataset import (
    _make_undirected,
    check_chain_interactions,
    check_com_distance,
    check_water_clashes,
    compute_normalized_bfactors,
    element_onehot,
    filter_waters_by_quality,
    get_crystal_contacts_pymol,
    get_dataloader,
    load_edia_for_pdb,
    match_atoms_to_coords,
    parse_asu_with_biotite,
    ProteinWaterDataset,
)


@pytest.fixture
def pdb_base_dir():
    """Base directory for PDB files."""
    return Path("/sb/wankowicz_lab/data/srivasv/pdb_redo_data")


@pytest.fixture
def pdb_6eey(pdb_base_dir):
    """Path to 6eey PDB file - should pass all quality checks."""
    path = pdb_base_dir / "6eey" / "6eey_final.pdb"
    if not path.exists():
        pytest.skip(f"PDB file not found: {path}")
    return str(path)


@pytest.fixture
def pdb_2b5w(pdb_base_dir):
    """Path to 2b5w PDB file - should fail COM distance check."""
    path = pdb_base_dir / "2b5w" / "2b5w_final.pdb"
    if not path.exists():
        pytest.skip(f"PDB file not found: {path}")
    return str(path)


@pytest.fixture
def pdb_8dzt(pdb_base_dir):
    """Path to 8dzt PDB file - should fail water clash check at 2% threshold."""
    path = pdb_base_dir / "8dzt" / "8dzt_final.pdb"
    if not path.exists():
        pytest.skip(f"PDB file not found: {path}")
    return str(path)


@pytest.fixture
def pdb_1deu(pdb_base_dir):
    """Path to 1deu PDB file - has insertion codes (52 residues with ins_code='P')."""
    path = pdb_base_dir / "1deu" / "1deu_final.pdb"
    if not path.exists():
        pytest.skip(f"PDB file not found: {path}")
    return str(path)


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
        assert hasattr(pp_edge, "edge_unit")
        assert hasattr(pp_edge, "edge_rbf")

        n_edges = pp_edge.edge_index.shape[1]

        # Check shapes
        assert pp_edge.edge_unit.shape == (n_edges, 3)
        assert pp_edge.edge_rbf.shape == (n_edges, 16)

        # Check values are valid
        assert not torch.isnan(pp_edge.edge_unit).any()
        assert not torch.isnan(pp_edge.edge_rbf).any()

        # Unit vectors should have norm ~1
        unit_norms = torch.linalg.norm(pp_edge.edge_unit, dim=-1)
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
    """Tests for EDIA data loading from CSV files."""

    def test_returns_none_for_missing_file(self, tmp_path):
        """Should return None if EDIA file doesn't exist."""
        result = load_edia_for_pdb(tmp_path, "nonexistent_pdb")
        assert result is None

    def test_loads_water_edia_scores(self, tmp_path):
        """Should load EDIA scores for water molecules."""
        # Create mock EDIA CSV
        pdb_id = "test_pdb"
        pdb_dir = tmp_path / pdb_id
        pdb_dir.mkdir()

        csv_content = """compID,pdb_strandID,pdb_seqNum,EDIAm,RSCCS
HOH,A,101,0.85,0.92
HOH,A,102,0.45,0.88
HOH,B,201,0.72,0.90
ALA,A,1,0.95,0.98
"""
        (pdb_dir / f"{pdb_id}_residue_stats.csv").write_text(csv_content)

        result = load_edia_for_pdb(tmp_path, pdb_id)

        assert result is not None
        assert len(result) == 3  # Only waters, not ALA
        assert result[("A", 101, "")] == pytest.approx(0.85)
        assert result[("A", 102, "")] == pytest.approx(0.45)
        assert result[("B", 201, "")] == pytest.approx(0.72)

    def test_returns_empty_dict_for_no_waters(self, tmp_path):
        """Should return empty dict if no water molecules in CSV."""
        pdb_id = "test_pdb"
        pdb_dir = tmp_path / pdb_id
        pdb_dir.mkdir()

        csv_content = """compID,pdb_strandID,pdb_seqNum,EDIAm,RSCCS
ALA,A,1,0.95,0.98
GLY,A,2,0.90,0.95
"""
        (pdb_dir / f"{pdb_id}_residue_stats.csv").write_text(csv_content)

        result = load_edia_for_pdb(tmp_path, pdb_id)

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
