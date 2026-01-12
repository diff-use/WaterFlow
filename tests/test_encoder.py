import pytest
import torch
from torch_geometric.data import Data

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import ProteinGVPEncoder, FlowEncoder


class TestProteinGVPEncoder:
    """Tests for ProteinGVPEncoder."""

    @pytest.fixture
    def simple_encoder(self):
        """Create a simple encoder for testing."""
        return ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            edge_scalar_in=16,
            n_layers=2,
            pooled_dim=32,
            pool_residue=True,
            num_edge_rbf=16,
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample protein graph data."""
        num_nodes = 20
        num_edges = 40
        
        data = Data(
            x=torch.randn(num_nodes, 16),
            pos=torch.randn(num_nodes, 3),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
            residue_index=torch.randint(0, 5, (num_nodes,)),
            num_residues=5,
        )
        return data
    
    def test_encoder_initialization(self, simple_encoder):
        """Test encoder initializes without errors."""
        assert simple_encoder is not None
        assert simple_encoder.n_layers == 2
    
    def test_encoder_forward(self, simple_encoder, sample_data):
        """Test forward pass produces correct output shape."""
        output = simple_encoder(sample_data)
        
        # Should pool to residue level
        assert output.shape == (sample_data.num_residues, simple_encoder.pooled_dim)
    
    def test_encoder_no_pooling(self, sample_data):
        """Test encoder without residue pooling."""
        encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            edge_scalar_in=16,
            n_layers=1,
            pool_residue=False,  # No pooling
            num_edge_rbf=16,
        )
        
        output = encoder(sample_data)
        
        # Should return tuple (s, V) at atom level
        assert isinstance(output, tuple)
        s, v = output
        assert s.shape == (sample_data.num_nodes, 64)
        assert v.shape == (sample_data.num_nodes, 16, 3)
    
    def test_encoder_empty_graph(self, simple_encoder):
        """Test encoder handles empty graphs gracefully."""
        data = Data(
            x=torch.zeros(0, 16),
            pos=torch.zeros(0, 3),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            residue_index=torch.zeros(0, dtype=torch.long),
            num_residues=0,
        )
        
        # Should not crash
        output = simple_encoder(data)
        assert output.shape == (0, simple_encoder.pooled_dim)


class TestFlowEncoder:
    """Tests for FlowEncoder wrapper."""
    
    @pytest.fixture
    def flow_encoder(self):
        """Create flow encoder (without checkpoint)."""
        return FlowEncoder(
            checkpoint_path="",  # No checkpoint, random init
            node_scalar_in=16,
            device="cpu",
            freeze=False,
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        num_nodes = 15
        num_edges = 30
        
        data = Data(
            x=torch.randn(num_nodes, 16),
            pos=torch.randn(num_nodes, 3),
            edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        )
        return data
    
    def test_flow_encoder_initialization(self, flow_encoder):
        """Test FlowEncoder initializes."""
        assert flow_encoder is not None
        assert hasattr(flow_encoder, 'encoder')
        assert hasattr(flow_encoder, 'pooled_dim')
    
    def test_flow_encoder_forward(self, flow_encoder, sample_data):
        """Test forward returns tuple."""
        output = flow_encoder(sample_data)
        
        assert isinstance(output, tuple)
        s, v = output
        assert s.shape[0] == sample_data.num_nodes
        assert v.shape[0] == sample_data.num_nodes
        assert v.shape[2] == 3  # 3D vectors
    
    def test_flow_encoder_freeze(self):
        """Test that freeze parameter works."""
        encoder_frozen = FlowEncoder(checkpoint_path="", node_scalar_in=16, device="cpu", freeze=True)
        encoder_unfrozen = FlowEncoder(checkpoint_path="", node_scalar_in=16, device="cpu", freeze=False)
        
        # Check frozen encoder has no gradients
        for p in encoder_frozen.encoder.parameters():
            assert not p.requires_grad
        
        # Check unfrozen encoder has gradients
        for p in encoder_unfrozen.encoder.parameters():
            assert p.requires_grad