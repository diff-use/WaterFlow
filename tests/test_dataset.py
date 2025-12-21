"""Unit tests for dataset.py"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from torch_geometric.data import HeteroData
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import *

TEST_PDB_PATH = Path(__file__).parent.parent / "test_files" / "6eey_final.pdb"


@pytest.fixture
def pdb_path():
    """Return path to test PDB, skip if not found."""
    if not TEST_PDB_PATH.exists():
        pytest.skip(f"Test PDB not found: {TEST_PDB_PATH}")
    return str(TEST_PDB_PATH)

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def temp_dir():
    """Create temporary directory for test cache files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_cached_data():
    """Create mock cached data matching preprocessing output."""
    return {
        'protein_pos': torch.randn(20, 3),
        'protein_x': torch.randn(20, 16),
        'protein_res_idx': torch.randint(0, 5, (20,)),
        'water_pos': torch.randn(5, 3),
        'water_x': torch.randn(5, 16),
        'mate_pos': torch.randn(8, 3),
        'mate_x': torch.randn(8, 16),
    }

@pytest.mark.unit
class TestEdgeFeatures:
    
    def test_basic_output(self):
        src_pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        dst_pos = src_pos.clone()
        edge_index = torch.tensor([[0, 1], [1, 2]])
        
        rbf, vec = edge_features(src_pos, dst_pos, edge_index, num_rbf=16, cutoff=8.0)
        
        assert rbf.shape == (2, 16)
        assert vec.shape == (2, 1, 3)
    
    def test_empty_edges(self):
        src_pos = torch.randn(5, 3)
        dst_pos = torch.randn(5, 3)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        rbf, vec = edge_features(src_pos, dst_pos, edge_index, num_rbf=16)
        
        assert rbf.shape == (0, 16)
        assert vec.shape == (0, 1, 3)
    
    def test_unit_vectors_normalized(self):
        src_pos = torch.tensor([[0., 0., 0.], [3., 4., 0.]])
        dst_pos = src_pos.clone()
        edge_index = torch.tensor([[0], [1]])
        
        _, vec = edge_features(src_pos, dst_pos, edge_index)
        
        norms = vec.squeeze(1).norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_different_num_rbf(self):
        src_pos = torch.randn(10, 3)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
        
        for num_rbf in [8, 16, 32]:
            rbf, _ = edge_features(src_pos, src_pos, edge_index, num_rbf=num_rbf)
            assert rbf.shape[1] == num_rbf

@pytest.mark.unit
class TestElementOnehot:
    
    def test_known_elements(self):
        symbols = ["C", "N", "O"]
        onehot = element_onehot(symbols)
        
        assert onehot.shape == (3, len(ELEMENT_VOCAB) + 1)
        assert onehot.sum(dim=-1).allclose(torch.ones(3))
        
        # Check correct indices
        assert onehot[0, ELEM_IDX["C"]] == 1.0
        assert onehot[1, ELEM_IDX["N"]] == 1.0
        assert onehot[2, ELEM_IDX["O"]] == 1.0
    
    def test_unknown_element(self):
        symbols = ["C", "XX", "N"]  # XX is unknown
        onehot = element_onehot(symbols)
        
        other_idx = len(ELEMENT_VOCAB)
        assert onehot[1, other_idx] == 1.0  # Unknown goes to "other" bucket
    
    def test_case_insensitive(self):
        symbols_upper = ["C", "N", "O"]
        symbols_lower = ["c", "n", "o"]
        
        onehot_upper = element_onehot(symbols_upper)
        onehot_lower = element_onehot(symbols_lower)
        
        assert torch.allclose(onehot_upper, onehot_lower)
    
    def test_empty_list(self):
        onehot = element_onehot([])
        assert onehot.shape == (0, len(ELEMENT_VOCAB) + 1)
    
    def test_all_vocab_elements(self):
        onehot = element_onehot(ELEMENT_VOCAB)
        
        # Each element should activate its own index
        for i, elem in enumerate(ELEMENT_VOCAB):
            assert onehot[i, i] == 1.0
            assert onehot[i].sum() == 1.0

@pytest.mark.unit
class TestMatchAtomsToCoords:
    
    def test_exact_match(self):
        # Mock biotite AtomArray
        atoms = Mock()
        atoms.coord = np.array([[0., 0., 0.], [1., 1., 1.], [2., 2., 2.]])
        
        target = np.array([[1., 1., 1.], [0., 0., 0.]])
        
        matched = match_atoms_to_coords(atoms, target, tolerance=0.01)
        
        assert len(matched) == 2
        assert 1 in matched  # [1,1,1] matches index 1
        assert 0 in matched  # [0,0,0] matches index 0
    
    def test_within_tolerance(self):
        atoms = Mock()
        atoms.coord = np.array([[0., 0., 0.], [1., 1., 1.]])
        
        # Slightly off coordinates
        target = np.array([[0.005, 0.005, 0.005]])
        
        matched = match_atoms_to_coords(atoms, target, tolerance=0.01)
        assert 0 in matched
    
    def test_outside_tolerance(self):
        """Test that matches outside tolerance are excluded."""
        atoms = Mock()
        atoms.coord = np.array([[0., 0., 0.], [1., 1., 1.]])
        
        # [0.1, 0.1, 0.1] is ~0.17 away from nearest atom [0,0,0]
        target = np.array([[0.1, 0.1, 0.1]])
        
        matched = match_atoms_to_coords(atoms, target, tolerance=0.01)
        assert len(matched) == 0 
    
    def test_empty_target(self):
        atoms = Mock()
        atoms.coord = np.array([[0., 0., 0.]])
        
        target = np.zeros((0, 3))
        matched = match_atoms_to_coords(atoms, target)
        
        assert matched == []


@pytest.mark.unit
class TestMakeUndirected:
    
    def test_adds_reverse_edges(self):
        edge_index = torch.tensor([[0, 1], [1, 2]])
        
        result = _make_undirected(edge_index)
        
        # Should contain both directions
        edges_set = set(tuple(e) for e in result.T.tolist())
        assert (0, 1) in edges_set
        assert (1, 0) in edges_set
        assert (1, 2) in edges_set
        assert (2, 1) in edges_set
    
    def test_removes_duplicates(self):
        # Already has some reverse edges
        edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
        
        result = _make_undirected(edge_index)
        
        # Count unique edges
        unique_edges = result.T.unique(dim=0)
        assert unique_edges.shape[0] == result.shape[1]
    
    def test_empty_edges(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        result = _make_undirected(edge_index)
        
        assert result.shape == (2, 0)
    
    def test_single_edge(self):
        edge_index = torch.tensor([[0], [1]])
        
        result = _make_undirected(edge_index)
        
        assert result.shape[1] == 2
        edges_set = set(tuple(e) for e in result.T.tolist())
        assert (0, 1) in edges_set
        assert (1, 0) in edges_set

@pytest.mark.unit
class TestProteinWaterDataset:
    
    @pytest.fixture
    def pdb_list_file(self, temp_dir):
        """Create a temporary PDB list file."""
        list_path = Path(temp_dir) / "pdb_list.txt"
        list_path.write_text("1abc_final_A\n2def_final_B\n")
        return str(list_path)
    
    @pytest.fixture
    def cached_dataset(self, temp_dir, pdb_list_file, mock_cached_data):
        """Create dataset with pre-cached data files."""
        processed_dir = Path(temp_dir) / "processed"
        processed_dir.mkdir()
        
        # Create cached files
        for key in ["1abc_final_A", "2def_final_B"]:
            torch.save(mock_cached_data, processed_dir / f"{key}.pt")
        
        return ProteinWaterDataset(
            pdb_list_file=pdb_list_file,
            processed_dir=str(processed_dir),
            base_pdb_dir=temp_dir,
            preprocess=True,
        )
    
    def test_parse_pdb_list(self, pdb_list_file, temp_dir):
        """Test PDB list parsing."""
        dataset = ProteinWaterDataset(
            pdb_list_file=pdb_list_file,
            processed_dir=temp_dir,
            preprocess=True,
        )
        
        assert len(dataset.entries) == 2
        assert dataset.entries[0]['pdb_id'] == '1abc'
        assert dataset.entries[0]['chain_id'] == 'A'
        assert dataset.entries[1]['pdb_id'] == '2def'
        assert dataset.entries[1]['chain_id'] == 'B'
    
    def test_len(self, cached_dataset):
        assert len(cached_dataset) == 2
    
    def test_getitem_returns_heterodata(self, cached_dataset):
        data = cached_dataset[0]
        
        assert isinstance(data, HeteroData)
        assert 'protein' in data.node_types
        assert 'water' in data.node_types
    
    def test_getitem_protein_attributes(self, cached_dataset):
        data = cached_dataset[0]
        
        assert hasattr(data['protein'], 'pos')
        assert hasattr(data['protein'], 'x')
        assert hasattr(data['protein'], 'residue_index')
        assert data['protein'].pos.shape[1] == 3
    
    def test_getitem_water_attributes(self, cached_dataset):
        data = cached_dataset[0]
        
        assert hasattr(data['water'], 'pos')
        assert hasattr(data['water'], 'x')
    
    def test_getitem_edges(self, cached_dataset):
        data = cached_dataset[0]
        
        assert ('protein', 'pp', 'protein') in data.edge_types
        edge_data = data['protein', 'pp', 'protein']
        assert hasattr(edge_data, 'edge_index')
        assert hasattr(edge_data, 'edge_rbf')
        assert hasattr(edge_data, 'edge_vec')
    
    def test_include_mates_true(self, cached_dataset, mock_cached_data):
        """Test that mates are included when flag is True."""
        data = cached_dataset[0]
        
        expected_protein_nodes = (
            mock_cached_data['protein_pos'].size(0) +
            mock_cached_data['mate_pos'].size(0)
        )
        assert data['protein'].num_nodes == expected_protein_nodes
    
    def test_include_mates_false(self, temp_dir, pdb_list_file, mock_cached_data):
        """Test that mates are excluded when flag is False."""
        processed_dir = Path(temp_dir) / "processed_no_mates"
        processed_dir.mkdir()
        
        torch.save(mock_cached_data, processed_dir / "1abc_final_A.pt")
        torch.save(mock_cached_data, processed_dir / "2def_final_B.pt")
        
        dataset = ProteinWaterDataset(
            pdb_list_file=pdb_list_file,
            processed_dir=str(processed_dir),
            preprocess=True,
            include_mates=False,
        )
        
        data = dataset[0]
        assert data['protein'].num_nodes == mock_cached_data['protein_pos'].size(0)
    
    def test_missing_cache_raises(self, temp_dir, pdb_list_file):
        """Test that missing cache file raises FileNotFoundError."""
        dataset = ProteinWaterDataset(
            pdb_list_file=pdb_list_file,
            processed_dir=temp_dir,
            preprocess=True,
        )
        
        with pytest.raises(FileNotFoundError):
            _ = dataset[0]
    
    def test_metadata_stored(self, cached_dataset):
        """Test that metadata is stored on HeteroData."""
        data = cached_dataset[0]
        
        assert hasattr(data, 'pdb_id')
        assert hasattr(data, 'num_asu_protein_atoms')

@pytest.mark.unit
class TestGetDataloader:
    
    @pytest.fixture
    def dataloader_setup(self, temp_dir, mock_cached_data):
        """Setup for dataloader tests."""
        processed_dir = Path(temp_dir) / "processed"
        processed_dir.mkdir()
        
        pdb_list = Path(temp_dir) / "list.txt"
        pdb_list.write_text("test1_final_A\ntest2_final_B\n")
        
        for key in ["test1_final_A", "test2_final_B"]:
            torch.save(mock_cached_data, processed_dir / f"{key}.pt")
        
        return str(pdb_list), str(processed_dir)
    
    def test_returns_dataloader(self, dataloader_setup):
        pdb_list, processed_dir = dataloader_setup
        
        loader = get_dataloader(
            pdb_list_file=pdb_list,
            processed_dir=processed_dir,
            batch_size=2,
            preprocess=True,
            num_workers=0,
        )
        
        from torch.utils.data import DataLoader
        assert isinstance(loader, DataLoader)
    
    def test_batch_size(self, dataloader_setup):
        pdb_list, processed_dir = dataloader_setup
        
        loader = get_dataloader(
            pdb_list_file=pdb_list,
            processed_dir=processed_dir,
            batch_size=2,
            preprocess=True,
            num_workers=0,
        )
        
        batch = next(iter(loader))
        # Batch should contain data from 2 graphs
        assert batch['protein'].batch.max().item() == 1  # 0 and 1 for 2 graphs
    
    def test_shuffle_option(self, dataloader_setup):
        pdb_list, processed_dir = dataloader_setup
        
        loader_shuffled = get_dataloader(
            pdb_list_file=pdb_list,
            processed_dir=processed_dir,
            batch_size=1,
            shuffle=True,
            preprocess=True,
            num_workers=0,
        )
        
        loader_unshuffled = get_dataloader(
            pdb_list_file=pdb_list,
            processed_dir=processed_dir,
            batch_size=1,
            shuffle=False,
            preprocess=True,
            num_workers=0,
        )
        
        # Both should work
        assert len(list(loader_shuffled)) == 2
        assert len(list(loader_unshuffled)) == 2

@pytest.mark.unit
class TestEdgeCases:
    
    def test_no_water_molecules(self, temp_dir):
        """Test dataset handles entries with no water."""
        processed_dir = Path(temp_dir) / "processed"
        processed_dir.mkdir()
        
        pdb_list = Path(temp_dir) / "list.txt"
        pdb_list.write_text("nowat_final_A\n")
        
        cached = {
            'protein_pos': torch.randn(20, 3),
            'protein_x': torch.randn(20, 16),
            'protein_res_idx': torch.randint(0, 5, (20,)),
            'water_pos': torch.zeros((0, 3)),
            'water_x': torch.zeros((0, 16)),
            'mate_pos': torch.zeros((0, 3)),
            'mate_x': torch.zeros((0, 16)),
        }
        torch.save(cached, processed_dir / "nowat_final_A.pt")
        
        dataset = ProteinWaterDataset(
            pdb_list_file=str(pdb_list),
            processed_dir=str(processed_dir),
            preprocess=True,
        )
        
        data = dataset[0]
        assert data['water'].num_nodes == 0
    
    def test_no_mates(self, temp_dir):
        """Test dataset handles entries with no symmetry mates."""
        processed_dir = Path(temp_dir) / "processed"
        processed_dir.mkdir()
        
        pdb_list = Path(temp_dir) / "list.txt"
        pdb_list.write_text("nomate_final_A\n")
        
        cached = {
            'protein_pos': torch.randn(15, 3),
            'protein_x': torch.randn(15, 16),
            'protein_res_idx': torch.randint(0, 3, (15,)),
            'water_pos': torch.randn(3, 3),
            'water_x': torch.randn(3, 16),
            'mate_pos': torch.zeros((0, 3)),
            'mate_x': torch.zeros((0, 16)),
        }
        torch.save(cached, processed_dir / "nomate_final_A.pt")
        
        dataset = ProteinWaterDataset(
            pdb_list_file=str(pdb_list),
            processed_dir=str(processed_dir),
            preprocess=True,
            include_mates=True,
        )
        
        data = dataset[0]
        # Should only have ASU protein atoms
        assert data['protein'].num_nodes == 15
    
    def test_malformed_pdb_list_line(self, temp_dir, capsys):
        """Test that malformed lines are skipped with warning."""
        pdb_list = Path(temp_dir) / "list.txt"
        pdb_list.write_text("valid_final_A\ninvalid\n")
        
        dataset = ProteinWaterDataset(
            pdb_list_file=str(pdb_list),
            processed_dir=temp_dir,
            preprocess=True,
        )
        
        assert len(dataset.entries) == 1
        captured = capsys.readouterr()
        assert "Skipping malformed" in captured.out

@pytest.mark.integration
class TestParseAsuWithBiotite:
    
    def test_returns_tuple(self, pdb_path):
        """Test function returns tuple of two AtomArrays."""
        result = parse_asu_with_biotite(pdb_path)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_protein_atoms_structure(self, pdb_path):
        """Test protein atoms have expected attributes."""
        protein_atoms, _ = parse_asu_with_biotite(pdb_path)
        
        assert hasattr(protein_atoms, 'coord')
        assert hasattr(protein_atoms, 'element')
        assert hasattr(protein_atoms, 'res_name')
        assert hasattr(protein_atoms, 'chain_id')
        assert hasattr(protein_atoms, 'res_id')
        
        assert protein_atoms.coord.shape[1] == 3
        assert len(protein_atoms) > 0
    
    def test_water_atoms_structure(self, pdb_path):
        """Test water atoms have expected attributes."""
        _, water_atoms = parse_asu_with_biotite(pdb_path)
        
        assert hasattr(water_atoms, 'coord')
        assert hasattr(water_atoms, 'element')
        assert hasattr(water_atoms, 'res_name')
        
        if len(water_atoms) > 0:
            assert water_atoms.coord.shape[1] == 3
            # All water should be HOH or WAT
            assert all(r in ("HOH", "WAT") for r in water_atoms.res_name)
    
    def test_no_hydrogens(self, pdb_path):
        """Test that hydrogen atoms are excluded."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_path)
        
        protein_elements = [str(e).upper() for e in protein_atoms.element]
        water_elements = [str(e).upper() for e in water_atoms.element]
        
        assert "H" not in protein_elements
        assert "H" not in water_elements
    
    def test_chain_filter(self, pdb_path):
        """Test chain filtering works."""
        # Get all chains first
        protein_all, _ = parse_asu_with_biotite(pdb_path, chain_filter=None)
        all_chains = set(protein_all.chain_id)
        
        if len(all_chains) > 0:
            # Filter to first chain only
            first_chain = list(all_chains)[0]
            protein_filtered, _ = parse_asu_with_biotite(pdb_path, chain_filter=[first_chain])
            
            filtered_chains = set(protein_filtered.chain_id)
            assert filtered_chains == {first_chain}
            assert len(protein_filtered) <= len(protein_all)
    
    def test_chain_filter_nonexistent(self, pdb_path):
        """Test filtering by nonexistent chain returns empty."""
        protein_atoms, water_atoms = parse_asu_with_biotite(pdb_path, chain_filter=["ZZZZZ"])
        
        assert len(protein_atoms) == 0
    
    def test_protein_is_amino_acids(self, pdb_path):
        """Test protein atoms are from canonical amino acids."""
        protein_atoms, _ = parse_asu_with_biotite(pdb_path)
        
        # Standard amino acid 3-letter codes
        standard_aa = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL"
        }
        
        protein_res_names = set(protein_atoms.res_name)
        assert protein_res_names.issubset(standard_aa)


@pytest.mark.integration
class TestGetCrystalContactsPymol:
    
    def test_returns_dict(self, pdb_path):
        """Test function returns dict with expected keys."""
        result = get_crystal_contacts_pymol(pdb_path, cutoff=5.0)
        
        assert isinstance(result, dict)
        assert "asu_coords" in result
        assert "mate_coords" in result
        assert "asu_atoms" in result
        assert "mate_atoms" in result
    
    def test_asu_coords_shape(self, pdb_path):
        """Test ASU coordinates have correct shape."""
        result = get_crystal_contacts_pymol(pdb_path)
        
        asu_coords = result["asu_coords"]
        assert isinstance(asu_coords, np.ndarray)
        assert asu_coords.ndim == 2
        assert asu_coords.shape[1] == 3
        assert len(asu_coords) > 0
    
    def test_mate_coords_shape(self, pdb_path):
        """Test mate coordinates have correct shape."""
        result = get_crystal_contacts_pymol(pdb_path)
        
        mate_coords = result["mate_coords"]
        assert isinstance(mate_coords, np.ndarray)
        assert mate_coords.ndim == 2
        if mate_coords.shape[0] > 0:
            assert mate_coords.shape[1] == 3
    
    def test_asu_atoms_list(self, pdb_path):
        """Test ASU atoms is a list of atom objects."""
        result = get_crystal_contacts_pymol(pdb_path)
        
        asu_atoms = result["asu_atoms"]
        assert isinstance(asu_atoms, list)
        assert len(asu_atoms) > 0
        
        # Check first atom has expected attributes
        atom = asu_atoms[0]
        assert hasattr(atom, 'symbol')
        assert hasattr(atom, 'coord')
    
    def test_cutoff_affects_mates(self, pdb_path):
        """Test that larger cutoff finds more or equal mate atoms."""
        result_small = get_crystal_contacts_pymol(pdb_path, cutoff=3.0)
        result_large = get_crystal_contacts_pymol(pdb_path, cutoff=8.0)
        
        # Larger cutoff should find >= mate atoms
        assert result_large["mate_coords"].shape[0] >= result_small["mate_coords"].shape[0]
    
    def test_coords_match_atoms_count(self, pdb_path):
        """Test coordinates array length matches atom list length."""
        result = get_crystal_contacts_pymol(pdb_path)
        
        assert result["asu_coords"].shape[0] == len(result["asu_atoms"])
        assert result["mate_coords"].shape[0] == len(result["mate_atoms"])


@pytest.mark.integration
class TestPreprocessingPipeline:
    
    @pytest.fixture
    def setup_pdb_structure(self, pdb_path, tmp_path):
        """Create directory structure matching dataset expectations."""
        # Dataset expects: base_dir/{pdb_id}/{pdb_id}_final.pdb
        # Our file: 6eey_final.pdb -> pdb_id = "6eey"
        pdb_id = "6eey"
        pdb_subdir = tmp_path / pdb_id
        pdb_subdir.mkdir()
        
        # Symlink or copy the file
        target_path = pdb_subdir / f"{pdb_id}_final.pdb"
        target_path.symlink_to(Path(pdb_path).resolve())
        
        return tmp_path  # This is the base_pdb_dir
    
    def test_preprocess_creates_cache_file(self, pdb_path, tmp_path, setup_pdb_structure):
        """Test that preprocessing creates the expected cache file."""
        base_pdb_dir = setup_pdb_structure
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        
        pdb_list = tmp_path / "list.txt"
        pdb_list.write_text("6eey_final_A\n")
        
        dataset = ProteinWaterDataset(
            pdb_list_file=str(pdb_list),
            processed_dir=str(processed_dir),
            base_pdb_dir=str(base_pdb_dir),
            preprocess=True,
            cutoff=5.0,
        )
        
        cache_file = processed_dir / "6eey_final_A.pt"
        assert cache_file.exists()
    
    def test_cached_data_structure(self, pdb_path, tmp_path, setup_pdb_structure):
        """Test cached data has all required keys."""
        base_pdb_dir = setup_pdb_structure
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        
        pdb_list = tmp_path / "list.txt"
        pdb_list.write_text("6eey_final_A\n")
        
        ProteinWaterDataset(
            pdb_list_file=str(pdb_list),
            processed_dir=str(processed_dir),
            base_pdb_dir=str(base_pdb_dir),
            preprocess=True,
        )
        
        cached = torch.load(processed_dir / "6eey_final_A.pt", weights_only=False)
        
        required_keys = [
            'protein_pos', 'protein_x', 'protein_res_idx',
            'water_pos', 'water_x',
            'mate_pos', 'mate_x'
        ]
        for key in required_keys:
            assert key in cached, f"Missing key: {key}"
    
    def test_cached_data_types(self, pdb_path, tmp_path, setup_pdb_structure):
        """Test cached tensors have correct types and shapes."""
        base_pdb_dir = setup_pdb_structure
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        
        pdb_list = tmp_path / "list.txt"
        pdb_list.write_text("6eey_final_A\n")
        
        ProteinWaterDataset(
            pdb_list_file=str(pdb_list),
            processed_dir=str(processed_dir),
            base_pdb_dir=str(base_pdb_dir),
            preprocess=True,
        )
        
        cached = torch.load(processed_dir / "6eey_final_A.pt", weights_only=False)
        
        assert cached['protein_pos'].shape[1] == 3
        assert cached['protein_pos'].dtype == torch.float32
        assert cached['protein_x'].shape[0] == cached['protein_pos'].shape[0]
        assert cached['protein_res_idx'].shape[0] == cached['protein_pos'].shape[0]
        assert cached['water_x'].shape[0] == cached['water_pos'].shape[0]
        assert cached['mate_x'].shape[0] == cached['mate_pos'].shape[0]
    
    def test_full_getitem_with_real_data(self, pdb_path, tmp_path, setup_pdb_structure):
        """Test __getitem__ works end-to-end with real preprocessed data."""
        base_pdb_dir = setup_pdb_structure
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        
        pdb_list = tmp_path / "list.txt"
        pdb_list.write_text("6eey_final_A\n")
        
        dataset = ProteinWaterDataset(
            pdb_list_file=str(pdb_list),
            processed_dir=str(processed_dir),
            base_pdb_dir=str(base_pdb_dir),
            preprocess=True,
        )
        
        data = dataset[0]
        
        assert 'protein' in data.node_types
        assert 'water' in data.node_types
        assert ('protein', 'pp', 'protein') in data.edge_types
        assert data['protein'].num_nodes > 0
        assert data['protein'].pos.shape == (data['protein'].num_nodes, 3)
        assert data['protein', 'pp', 'protein'].edge_index.shape[0] == 2
