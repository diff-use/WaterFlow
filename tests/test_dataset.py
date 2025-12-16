import pytest
import torch
import numpy as np

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    rbf, edge_features, element_onehot, 
    ELEMENT_VOCAB, ELEM_IDX, _make_undirected
)


class TestRBF:
    """Tests for radial basis function encoding."""
    
    def test_rbf_shape(self):
        """Test RBF output shape."""
        r = torch.tensor([0.5, 1.0, 2.0, 5.0])
        result = rbf(r, num_gaussians=16, cutoff=8.0)
        assert result.shape == (4, 16)
    
    def test_rbf_range(self):
        """Test RBF values are finite and reasonable."""
        # Avoid r=0 and exact cutoff boundary (numerical edge cases)
        r = torch.linspace(0.0, 7.9, 50)
        result = rbf(r, num_gaussians=16, cutoff=8.0)
        
        assert torch.isfinite(result).all().item()
        assert (result.abs().sum(dim=-1) > 0).all().item()

    def test_rbf_cutoff(self):
        """With cutoff=True, values beyond cutoff should be ~0 (up to fp tolerance)."""
        r = torch.tensor([0.0, 4.0, 8.0, 12.0])
        result = rbf(r, num_gaussians=16, cutoff=8.0)

        # Beyond cutoff should be (close to) zero
        assert result[-1].abs().max() < 1e-6

        # Well inside cutoff should have non-trivial magnitude
        assert (result[1] ** 2).sum() > 1e-6


class TestEdgeFeatures:
    """Tests for edge feature computation."""
    
    def test_edge_features_shape(self):
        """Test output shapes."""
        src_pos = torch.randn(10, 3)
        dst_pos = torch.randn(10, 3)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        
        e_rbf, e_vec = edge_features(src_pos, dst_pos, edge_index, num_rbf=16, cutoff=8.0)
        
        assert e_rbf.shape == (3, 16)  # 3 edges, 16 RBF bins
        assert e_vec.shape == (3, 1, 3)  # 3 edges, 1 vector channel, 3D
    
    def test_edge_features_unit_vectors(self):
        """Test that displacement vectors are unit normalized."""
        src_pos = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
        dst_pos = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
        edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        
        _, e_vec = edge_features(src_pos, dst_pos, edge_index, num_rbf=16)
        
        # Check unit normalization
        norms = torch.linalg.norm(e_vec.squeeze(1), dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_empty_edges(self):
        """Test with no edges."""
        src_pos = torch.randn(10, 3)
        dst_pos = torch.randn(10, 3)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        e_rbf, e_vec = edge_features(src_pos, dst_pos, edge_index, num_rbf=16)
        
        assert e_rbf.shape == (0, 16)
        assert e_vec.shape == (0, 1, 3)


class TestElementOnehot:
    """Tests for element one-hot encoding."""
    
    def test_known_elements(self):
        """Test encoding of known elements."""
        symbols = ["C", "N", "O"]
        result = element_onehot(symbols)
        
        assert result.shape == (3, len(ELEMENT_VOCAB) + 1)  # +1 for "other"
        assert result[0, ELEM_IDX["C"]] == 1.0
        assert result[1, ELEM_IDX["N"]] == 1.0
        assert result[2, ELEM_IDX["O"]] == 1.0
    
    def test_unknown_element(self):
        """Test encoding of unknown element goes to 'other' bucket."""
        symbols = ["C", "XYZ"]  # XYZ is not in vocab
        result = element_onehot(symbols)
        
        assert result[0, ELEM_IDX["C"]] == 1.0
        assert result[1, -1] == 1.0  # Last position is "other"
    
    def test_case_insensitive(self):
        """Test that encoding is case-insensitive."""
        symbols_lower = ["c", "n", "o"]
        symbols_upper = ["C", "N", "O"]
        
        result_lower = element_onehot(symbols_lower)
        result_upper = element_onehot(symbols_upper)
        
        assert torch.allclose(result_lower, result_upper)


class TestMakeUndirected:
    """Tests for edge undirecting."""
    
    def test_simple_edges(self):
        """Test making directed edges undirected."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        result = _make_undirected(edge_index)
        
        # Should have forward and reverse edges (deduplicated)
        assert result.shape[0] == 2
        assert result.shape[1] == 6  # 3 original + 3 reverse
    
    def test_empty_edges(self):
        """Test with empty edge index."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        result = _make_undirected(edge_index)
        
        assert result.shape == (2, 0)
    
    def test_no_duplicates(self):
        """Test that duplicates are removed."""
        # Include same edge twice
        edge_index = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long)
        result = _make_undirected(edge_index)
        
        # Should have unique edges only
        unique_edges = torch.unique(result, dim=1)
        assert result.shape[1] == unique_edges.shape[1]