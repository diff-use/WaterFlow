import pytest
import torch
from torch_geometric.data import HeteroData

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flow import rbf, edge_features, build_knn_edges, FlowMatcher


class TestFlowRBF:
    """Tests for RBF encoding in flow module."""
    
    def test_rbf_shape(self):
        """Test RBF output shape."""
        d = torch.tensor([1.0, 2.0, 5.0, 10.0])
        result = rbf(d, num_rbf=16, D_min=0.0, D_max=20.0)
        
        assert result.shape == (4, 16)
    
    def test_rbf_values(self):
        """Test RBF values are non-negative."""
        d = torch.linspace(0, 20, 50)
        result = rbf(d, num_rbf=16)
        
        assert (result >= 0).all()


class TestFlowEdgeFeatures:
    """Tests for edge feature computation in flow module."""
    
    def test_edge_features_shape(self):
        """Test output shapes."""
        src_pos = torch.randn(10, 3)
        dst_pos = torch.randn(10, 3)
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
        
        e_s, e_v = edge_features(src_pos, dst_pos, edge_index, num_rbf=16)
        
        assert e_s.shape == (3, 16)
        assert e_v.shape == (3, 1, 3)
    
    def test_unit_vectors(self):
        """Test displacement vectors are normalized."""
        src_pos = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
        dst_pos = torch.tensor([[2., 0., 0.], [0., 0., 0.]])
        edge_index = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        
        _, e_v = edge_features(src_pos, dst_pos, edge_index)
        
        norms = torch.linalg.norm(e_v.squeeze(1), dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_empty_edges(self):
        """Test with no edges."""
        src_pos = torch.randn(5, 3)
        dst_pos = torch.randn(5, 3)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
        e_s, e_v = edge_features(src_pos, dst_pos, edge_index)
        
        assert e_s.shape == (0, 16)
        assert e_v.shape == (0, 1, 3)


class TestBuildKNNEdges:
    """Tests for KNN edge construction."""
    
    def test_knn_basic(self):
        """Test basic KNN edge construction."""
        src_pos = torch.randn(10, 3)
        dst_pos = torch.randn(15, 3)
        k = 5
        
        edges = build_knn_edges(src_pos, dst_pos, k=k)
        
        assert edges.shape[0] == 2  # [src_idx, dst_idx]
        # Should have roughly k * num_src edges (after deduplication)
        assert edges.shape[1] > 0
        assert edges.shape[1] <= k * src_pos.size(0)
    
    def test_knn_empty_inputs(self):
        """Test with empty position tensors."""
        src_pos = torch.empty(0, 3)
        dst_pos = torch.randn(10, 3)
        
        edges = build_knn_edges(src_pos, dst_pos, k=5)
        
        assert edges.shape == (2, 0)
    
    def test_knn_no_self_edges(self):
        """Test that self edges are removed for homogeneous case."""
        pos = torch.randn(10, 3)
        
        edges = build_knn_edges(pos, pos, k=3)
        
        # Check no self edges
        assert not (edges[0] == edges[1]).any()


class TestFlowMatcher:
    """Tests for FlowMatcher high-level interface."""
    
    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        # Simple mock that returns zero velocity
        class DummyModel(torch.nn.Module):
            def forward(self, data, t, sc=None):
                n_water = data['water'].num_nodes
                return torch.zeros(n_water, 3, device=data['protein'].pos.device)
            
            def parameters(self):
                return []
            
            def eval(self):
                pass
            
            def train(self):
                pass
        
        return DummyModel()
    
    @pytest.fixture
    def sample_hetero_data(self):
        """Create sample HeteroData."""
        data = HeteroData()
        
        # Protein nodes
        data['protein'].pos = torch.randn(20, 3)
        data['protein'].x = torch.randn(20, 16)
        data['protein'].batch = torch.zeros(20, dtype=torch.long)
        
        # Water nodes
        data['water'].pos = torch.randn(10, 3)
        data['water'].x = torch.randn(10, 16)
        data['water'].batch = torch.zeros(10, dtype=torch.long)
        
        # Protein-protein edges
        data['protein', 'pp', 'protein'].edge_index = torch.randint(0, 20, (2, 40))
        
        return data
    
    def test_compute_sigma(self, sample_hetero_data):
        """Test sigma computation."""
        sigma = FlowMatcher.compute_sigma(sample_hetero_data)
        
        assert isinstance(sigma, float)
        assert sigma > 0
    
    def test_validation_step_no_crash(self, dummy_model, sample_hetero_data):
        """Test validation step runs without errors."""
        matcher = FlowMatcher(dummy_model)
        
        result = matcher.validation_step(sample_hetero_data)
        
        assert 'loss' in result
        assert 'rmsd' in result
        assert isinstance(result['loss'], float)
        assert isinstance(result['rmsd'], float)
    
    def test_sample_euler(self, dummy_model, sample_hetero_data):
        """Test Euler integration sampling."""
        matcher = FlowMatcher(dummy_model)
        
        water_pred = matcher.sample(
            sample_hetero_data, 
            num_steps=10, 
            method="euler",
            use_sc=False,
            device="cpu"
        )
        
        assert water_pred.shape == (sample_hetero_data['water'].num_nodes, 3)