"""
Unified encoder tests for the modular encoder architecture.

Tests:
1. Registry pattern (register, get, build)
2. Base encoder interface contract
3. GVP encoder (ProteinGVPEncoder + GVPEncoder wrapper)
4. SLAE encoder and projection
5. Encoder interoperability (both work with flow model)
"""

import pytest
import torch
from torch_geometric.data import Data, HeteroData
from torch_cluster import radius_graph

from src.encoder_base import build_encoder, get_encoder_class
from src.gvp_encoder import ProteinGVPEncoder, GVPEncoder, make_encoder_data
from src.slae import SLAEEncoder


# ============== Fixtures ==============

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_homogeneous_data():
    """Sample Data for testing ProteinGVPEncoder directly."""
    num_nodes = 20
    num_edges = 40

    return Data(
        x=torch.randn(num_nodes, 16),
        pos=torch.randn(num_nodes, 3),
        edge_index=torch.randint(0, num_nodes, (2, num_edges)),
        residue_index=torch.randint(0, 5, (num_nodes,)),
        num_residues=5,
    )


@pytest.fixture
def sample_hetero_data(device):
    """Sample HeteroData for testing encoder wrappers."""
    num_protein = 30
    num_water = 10

    data = HeteroData()

    # Protein nodes
    data['protein'].x = torch.randn(num_protein, 16, device=device)
    data['protein'].pos = torch.randn(num_protein, 3, device=device)
    data['protein'].batch = torch.zeros(num_protein, dtype=torch.long, device=device)
    data['protein'].num_nodes = num_protein

    # Water nodes
    data['water'].x = torch.randn(num_water, 16, device=device)
    data['water'].pos = torch.randn(num_water, 3, device=device)
    data['water'].batch = torch.zeros(num_water, dtype=torch.long, device=device)

    # Protein-protein edges
    pp_edges = radius_graph(data['protein'].pos, r=8.0, loop=False)
    data['protein', 'pp', 'protein'].edge_index = pp_edges

    return data


@pytest.fixture
def sample_hetero_data_with_slae(sample_hetero_data):
    """Sample HeteroData with SLAE embeddings."""
    data = sample_hetero_data
    num_protein = data['protein'].num_nodes
    data['protein'].slae_embedding = torch.randn(num_protein, 128, device=data['protein'].pos.device)
    return data


# ============== Registry Tests ==============

class TestEncoderRegistry:
    """Tests for encoder registry pattern."""

    def test_gvp_registered(self):
        """GVP encoder should be registered."""
        cls = get_encoder_class('gvp')
        assert cls is GVPEncoder

    def test_slae_registered(self):
        """SLAE encoder should be registered."""
        cls = get_encoder_class('slae')
        assert cls is SLAEEncoder

    def test_unknown_encoder_raises(self):
        """Unknown encoder type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown encoder type"):
            get_encoder_class('nonexistent_encoder')

    def test_build_encoder_gvp(self, device):
        """build_encoder should construct GVP encoder from config."""
        config = {
            'encoder_type': 'gvp',
            'node_scalar_in': 16,
            'hidden_s': 64,
            'hidden_v': 16,
        }
        encoder = build_encoder(config, device)

        assert isinstance(encoder, GVPEncoder)
        assert encoder.encoder_type == 'gvp'
        assert encoder.output_dims == (64, 16)

    def test_build_encoder_slae(self, device):
        """build_encoder should construct SLAE encoder from config."""
        config = {
            'encoder_type': 'slae',
            'slae_dim': 128,
        }
        encoder = build_encoder(config, device)

        assert isinstance(encoder, SLAEEncoder)
        assert encoder.encoder_type == 'slae'
        assert encoder.output_dims == (128, 0)

    def test_build_encoder_default_gvp(self, device):
        """build_encoder should default to GVP if encoder_type not specified."""
        config = {'node_scalar_in': 16}
        encoder = build_encoder(config, device)
        assert isinstance(encoder, GVPEncoder)


# ============== Base Interface Tests ==============

class TestBaseEncoderInterface:
    """Tests for BaseProteinEncoder interface contract."""

    def test_gvp_implements_interface(self, device, sample_hetero_data):
        """GVPEncoder should implement all required interface methods."""
        encoder = GVPEncoder(
            encoder=ProteinGVPEncoder(
                node_scalar_in=16,
                hidden_dims=(64, 16),
                edge_scalar_in=16,
                pool_residue=False,
            ).to(device),
            freeze=False,
        )

        # Check properties
        assert isinstance(encoder.output_dims, tuple)
        assert len(encoder.output_dims) == 2
        assert isinstance(encoder.encoder_type, str)

        # Check forward returns (s, V) tuple
        s, V = encoder(sample_hetero_data)
        assert s.shape[0] == sample_hetero_data['protein'].num_nodes
        assert V.shape[0] == sample_hetero_data['protein'].num_nodes
        assert V.shape[2] == 3

    def test_slae_implements_interface(self, device, sample_hetero_data_with_slae):
        """SLAEEncoder should implement all required interface methods."""
        encoder = SLAEEncoder(slae_dim=128).to(device)

        # Check properties
        assert isinstance(encoder.output_dims, tuple)
        assert len(encoder.output_dims) == 2
        assert encoder.output_dims == (128, 0)
        assert isinstance(encoder.encoder_type, str)

        # Check forward returns (s, V) tuple
        s, V = encoder(sample_hetero_data_with_slae)
        assert s.shape[0] == sample_hetero_data_with_slae['protein'].num_nodes
        assert s.shape[1] == 128
        assert V.shape == (sample_hetero_data_with_slae['protein'].num_nodes, 0, 3)

    def test_from_config_class_method(self, device):
        """Both encoders should have from_config class method."""
        gvp_config = {'encoder_type': 'gvp', 'node_scalar_in': 16, 'hidden_s': 64, 'hidden_v': 16}
        slae_config = {'encoder_type': 'slae', 'slae_dim': 128}

        gvp_encoder = GVPEncoder.from_config(gvp_config, device)
        slae_encoder = SLAEEncoder.from_config(slae_config, device)

        assert isinstance(gvp_encoder, GVPEncoder)
        assert isinstance(slae_encoder, SLAEEncoder)
        assert slae_encoder.output_dims == (128, 0)


# ============== GVP Encoder Tests ==============

class TestProteinGVPEncoder:
    """Tests for the core ProteinGVPEncoder."""

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

    def test_encoder_initialization(self, simple_encoder):
        """Test encoder initializes without errors."""
        assert simple_encoder is not None
        assert simple_encoder.n_layers == 2

    def test_encoder_forward_with_pooling(self, simple_encoder, sample_homogeneous_data):
        """Test forward pass with residue pooling."""
        output = simple_encoder(sample_homogeneous_data)
        assert output.shape == (sample_homogeneous_data.num_residues, simple_encoder.pooled_dim)

    def test_encoder_forward_no_pooling(self, sample_homogeneous_data):
        """Test encoder without residue pooling returns (s, V) tuple."""
        encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            edge_scalar_in=16,
            n_layers=1,
            pool_residue=False,
            num_edge_rbf=16,
        )

        output = encoder(sample_homogeneous_data)

        assert isinstance(output, tuple)
        s, v = output
        assert s.shape == (sample_homogeneous_data.num_nodes, 64)
        assert v.shape == (sample_homogeneous_data.num_nodes, 16, 3)

    def test_encoder_empty_graph(self, simple_encoder):
        """Test encoder handles empty graphs gracefully."""
        data = Data(
            x=torch.zeros(0, 16),
            pos=torch.zeros(0, 3),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            residue_index=torch.zeros(0, dtype=torch.long),
            num_residues=0,
        )

        output = simple_encoder(data)
        assert output.shape == (0, simple_encoder.pooled_dim)


class TestGVPEncoderWrapper:
    """Tests for GVPEncoder (BaseProteinEncoder wrapper)."""

    def test_wrapper_output_dims(self, device):
        """Wrapper should expose correct output_dims."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(128, 32),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        assert encoder.output_dims == (128, 32)

    def test_wrapper_encoder_type(self, device):
        """Wrapper should return 'gvp' as encoder_type."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        assert encoder.encoder_type == 'gvp'

    def test_wrapper_forward(self, device, sample_hetero_data):
        """Wrapper forward should return (s, V) from HeteroData."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        s, V = encoder(sample_hetero_data)

        assert s.shape == (sample_hetero_data['protein'].num_nodes, 64)
        assert V.shape == (sample_hetero_data['protein'].num_nodes, 16, 3)

    def test_wrapper_freeze(self, device):
        """Freeze parameter should disable gradients."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        encoder = GVPEncoder(encoder=base_encoder, freeze=True)

        for p in encoder.encoder.parameters():
            assert not p.requires_grad


class TestMakeEncoderData:
    """Tests for make_encoder_data helper function."""

    def test_converts_hetero_to_homo(self, sample_hetero_data):
        """Should convert HeteroData to homogeneous Data."""
        enc_data = make_encoder_data(sample_hetero_data)

        assert isinstance(enc_data, Data)
        assert hasattr(enc_data, 'x')
        assert hasattr(enc_data, 'pos')
        assert hasattr(enc_data, 'edge_index')

    def test_preserves_batch(self, sample_hetero_data):
        """Should preserve batch attribute."""
        enc_data = make_encoder_data(sample_hetero_data)

        assert hasattr(enc_data, 'batch')
        assert enc_data.batch.shape[0] == sample_hetero_data['protein'].num_nodes


# ============== SLAE Encoder Tests ==============

class TestSLAEEncoder:
    """Tests for SLAEEncoder (BaseProteinEncoder implementation)."""

    def test_encoder_output_dims(self, device):
        """Encoder should expose correct output_dims."""
        encoder = SLAEEncoder(slae_dim=128).to(device)
        assert encoder.output_dims == (128, 0)

    def test_encoder_type(self, device):
        """Encoder should return 'slae' as encoder_type."""
        encoder = SLAEEncoder(slae_dim=128).to(device)
        assert encoder.encoder_type == 'slae'

    def test_encoder_forward(self, device, sample_hetero_data_with_slae):
        """Forward pass should return (s, V) tuple with raw embeddings."""
        encoder = SLAEEncoder(slae_dim=128).to(device)

        s, V = encoder(sample_hetero_data_with_slae)

        n_atoms = sample_hetero_data_with_slae['protein'].num_nodes
        assert s.shape == (n_atoms, 128)
        assert V.shape == (n_atoms, 0, 3)
        # Raw embeddings should be identical to input
        assert torch.allclose(s, sample_hetero_data_with_slae['protein'].slae_embedding)

    def test_encoder_missing_embeddings_error(self, device, sample_hetero_data):
        """Should raise NotImplementedError when embeddings are missing."""
        encoder = SLAEEncoder(slae_dim=128).to(device)

        # sample_hetero_data does NOT have slae_embedding
        with pytest.raises(NotImplementedError, match="requires cached embeddings"):
            encoder(sample_hetero_data)

    def test_encoder_no_nans(self, device, sample_hetero_data_with_slae):
        """Output should not contain NaNs or Infs."""
        encoder = SLAEEncoder(slae_dim=128).to(device)

        s, V = encoder(sample_hetero_data_with_slae)

        assert not torch.isnan(s).any(), "Scalar output contains NaNs"
        assert not torch.isinf(s).any(), "Scalar output contains Infs"

    def test_encoder_no_learnable_params(self, device):
        """SLAE encoder should have no learnable parameters."""
        encoder = SLAEEncoder(slae_dim=128).to(device)
        assert sum(p.numel() for p in encoder.parameters()) == 0


# ============== Encoder Interoperability Tests ==============

class TestEncoderInteroperability:
    """Tests that both encoders work interchangeably with flow model."""

    def test_both_encoders_work_with_flow(self, device, sample_hetero_data_with_slae):
        """Both encoders should work with FlowWaterGVP."""
        from src.flow import FlowWaterGVP

        hidden_dims = (64, 16)

        # GVP encoder
        gvp_encoder = GVPEncoder(
            encoder=ProteinGVPEncoder(
                node_scalar_in=16,
                hidden_dims=hidden_dims,
                edge_scalar_in=16,
                pool_residue=False,
            ).to(device),
            freeze=False,
        )

        # SLAE encoder (output_dims = (128, 0), bridged by encoder_to_flow)
        slae_encoder = SLAEEncoder(slae_dim=128).to(device)

        # Create flow models with each encoder
        flow_gvp = FlowWaterGVP(
            encoder=gvp_encoder,
            hidden_dims=hidden_dims,
            layers=1,
        ).to(device)

        flow_slae = FlowWaterGVP(
            encoder=slae_encoder,
            hidden_dims=hidden_dims,
            layers=1,
        ).to(device)

        t = torch.tensor([0.5], device=device)

        # Both should run without errors
        v_gvp = flow_gvp(sample_hetero_data_with_slae, t)
        v_slae = flow_slae(sample_hetero_data_with_slae, t)

        # Both should have same output shape
        assert v_gvp.shape == v_slae.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
