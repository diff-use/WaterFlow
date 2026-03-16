"""
Unified encoder tests for the modular encoder architecture.

Tests:
1. Registry pattern (register, get, build)
2. Base encoder interface contract
3. GVP encoder (ProteinGVPEncoder + GVPEncoder wrapper)
4. Cached embedding encoder (SLAE, ESM)
5. Encoder interoperability (both work with flow model)
"""

import pytest
import torch
from torch_cluster import radius_graph
from torch_geometric.data import Data, HeteroData

from src.encoder_base import build_encoder, CachedEmbeddingEncoder, get_encoder_class
from src.gvp_encoder import GVPEncoder, ProteinGVPEncoder


# ============== Fixtures ==============


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
    data["protein"].x = torch.randn(num_protein, 16, device=device)
    data["protein"].pos = torch.randn(num_protein, 3, device=device)
    data["protein"].batch = torch.zeros(num_protein, dtype=torch.long, device=device)
    data["protein"].num_nodes = num_protein

    # Water nodes
    data["water"].x = torch.randn(num_water, 16, device=device)
    data["water"].pos = torch.randn(num_water, 3, device=device)
    data["water"].batch = torch.zeros(num_water, dtype=torch.long, device=device)

    # Protein-protein edges
    pp_edges = radius_graph(data["protein"].pos, r=8.0, loop=False)
    data["protein", "pp", "protein"].edge_index = pp_edges

    return data


@pytest.fixture
def sample_hetero_data_with_slae(sample_hetero_data):
    """Sample HeteroData with SLAE embeddings."""
    data = sample_hetero_data
    num_protein = data["protein"].num_nodes
    data["protein"].slae_embedding = torch.randn(
        num_protein, 128, device=data["protein"].pos.device
    )
    return data


# ============== Registry Tests ==============


class TestPackageLevelImport:
    """Tests that importing from src package triggers encoder registration."""

    def test_build_encoder_works_with_package_import(self):
        """
        Verify that importing build_encoder from src (not src.encoder_base)
        has encoders already registered. This tests that src/__init__.py
        correctly imports the encoder modules to trigger registration.
        """
        # Import fresh from package level - this is how scripts should import
        from src import build_encoder as pkg_build_encoder

        # Should work without any explicit encoder module imports
        device = torch.device("cpu")

        gvp_encoder = pkg_build_encoder(
            {"encoder_type": "gvp", "node_scalar_in": 16}, device
        )
        assert gvp_encoder.encoder_type == "gvp"

        slae_encoder = pkg_build_encoder({"encoder_type": "slae"}, device)
        assert slae_encoder.encoder_type == "slae"


class TestEncoderRegistry:
    """Tests for encoder registry pattern."""

    def test_gvp_registered(self):
        """GVP encoder should be registered."""
        cls = get_encoder_class("gvp")
        assert cls is GVPEncoder

    def test_slae_registered(self):
        """SLAE encoder should be registered."""
        cls = get_encoder_class("slae")
        assert cls is CachedEmbeddingEncoder

    def test_esm_registered(self):
        """ESM encoder should be registered."""
        cls = get_encoder_class("esm")
        assert cls is CachedEmbeddingEncoder

    def test_unknown_encoder_raises(self):
        """Unknown encoder type should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown encoder type"):
            get_encoder_class("nonexistent_encoder")

    def test_build_encoder_gvp(self, device):
        """build_encoder should construct GVP encoder from config."""
        config = {
            "encoder_type": "gvp",
            "node_scalar_in": 16,
            "hidden_s": 64,
            "hidden_v": 16,
        }
        encoder = build_encoder(config, device)

        assert isinstance(encoder, GVPEncoder)
        assert encoder.encoder_type == "gvp"
        assert encoder.output_dims == (64, 16)

    def test_build_encoder_slae(self, device):
        """build_encoder should construct SLAE encoder from config."""
        config = {
            "encoder_type": "slae",
        }
        encoder = build_encoder(config, device)

        assert isinstance(encoder, CachedEmbeddingEncoder)
        assert encoder.encoder_type == "slae"
        # output_dims not available until forward pass

    def test_build_encoder_esm(self, device):
        """build_encoder should construct ESM encoder from config."""
        config = {
            "encoder_type": "esm",
        }
        encoder = build_encoder(config, device)

        assert isinstance(encoder, CachedEmbeddingEncoder)
        assert encoder.encoder_type == "esm"
        # output_dims not available until forward pass


# ============== Base Interface Tests ==============


class TestBaseEncoderInterface:
    """Tests for BaseProteinEncoder interface contract."""

    def test_gvp_implements_interface(self, device, sample_hetero_data):
        """GVPEncoder should implement all required interface methods."""
        encoder = GVPEncoder(
            encoder=ProteinGVPEncoder(
                node_scalar_in=16,
                hidden_dims=(64, 16),
                n_edge_scalar_in=16,
                pool_residue=False,
            ).to(device),
            freeze=False,
        )

        # Check properties
        assert isinstance(encoder.output_dims, tuple)
        assert len(encoder.output_dims) == 2
        assert isinstance(encoder.encoder_type, str)

        # Check forward returns (s, V, pp_edge_attr) tuple
        s, V, pp_edge_attr = encoder(sample_hetero_data)
        assert s.shape[0] == sample_hetero_data["protein"].num_nodes
        assert V.shape[0] == sample_hetero_data["protein"].num_nodes
        assert V.shape[2] == 3
        # GVP encoder should return edge features
        assert pp_edge_attr is not None or encoder.encoder.edge_update is None

    def test_cached_embedding_implements_interface(
        self, device, sample_hetero_data_with_slae
    ):
        """CachedEmbeddingEncoder should implement all required interface methods."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae"
        ).to(device)

        assert isinstance(encoder.encoder_type, str)

        # Check forward returns (s, V, pp_edge_attr) tuple
        s, V, pp_edge_attr = encoder(sample_hetero_data_with_slae)
        assert s.shape[0] == sample_hetero_data_with_slae["protein"].num_nodes
        assert s.shape[1] == 128
        assert V.shape == (sample_hetero_data_with_slae["protein"].num_nodes, 0, 3)
        # Cached embedding encoder should return None for edge features
        assert pp_edge_attr is None

        # output_dims available after forward
        assert isinstance(encoder.output_dims, tuple)
        assert len(encoder.output_dims) == 2
        assert encoder.output_dims == (128, 0)

    def test_from_config_class_method(self, device):
        """Both encoders should have from_config class method."""
        gvp_config = {
            "encoder_type": "gvp",
            "node_scalar_in": 16,
            "hidden_s": 64,
            "hidden_v": 16,
        }
        slae_config = {"encoder_type": "slae"}

        gvp_encoder = GVPEncoder.from_config(gvp_config, device)
        slae_encoder = CachedEmbeddingEncoder.from_config(slae_config, device)

        assert isinstance(gvp_encoder, GVPEncoder)
        assert isinstance(slae_encoder, CachedEmbeddingEncoder)
        # output_dims not available until forward pass for cached encoders


# ============== GVP Encoder Tests ==============


class TestProteinGVPEncoder:
    """Tests for the core ProteinGVPEncoder."""

    @pytest.fixture
    def simple_encoder(self):
        """Create a simple encoder for testing."""
        return ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            n_edge_scalar_in=16,
            n_layers=2,
            pooled_dim=32,
            pool_residue=True,
            num_edge_rbf=16,
        )

    def test_encoder_initialization(self, simple_encoder):
        """Test encoder initializes without errors."""
        assert simple_encoder is not None
        assert simple_encoder.n_layers == 2

    def test_encoder_forward_with_pooling(
        self, simple_encoder, sample_homogeneous_data
    ):
        """Test forward pass with residue pooling."""
        output, edge_attr = simple_encoder(sample_homogeneous_data)
        assert output.shape == (
            sample_homogeneous_data.num_residues,
            simple_encoder.pooled_dim,
        )
        # Pooling mode returns None for edge features
        assert edge_attr is None

    def test_encoder_forward_no_pooling(self, sample_homogeneous_data):
        """Test encoder without residue pooling returns ((s, V), edge_attr) tuple."""
        encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            n_edge_scalar_in=16,
            n_layers=1,
            pool_residue=False,
            num_edge_rbf=16,
        )

        (s, v), edge_attr = encoder(sample_homogeneous_data)

        assert s.shape == (sample_homogeneous_data.num_nodes, 64)
        assert v.shape == (sample_homogeneous_data.num_nodes, 16, 3)
        # edge_attr should be a tuple (s_edge, V_edge)
        assert edge_attr is not None
        s_edge, V_edge = edge_attr
        assert s_edge.dim() == 2
        assert V_edge.dim() == 3

    def test_encoder_empty_graph(self, simple_encoder):
        """Test encoder handles empty graphs gracefully."""
        data = Data(
            x=torch.zeros(0, 16),
            pos=torch.zeros(0, 3),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            residue_index=torch.zeros(0, dtype=torch.long),
            num_residues=0,
        )

        output, edge_attr = simple_encoder(data)
        assert output.shape == (0, simple_encoder.pooled_dim)
        # Pooling mode returns None for edge features
        assert edge_attr is None


class TestGVPEncoderWrapper:
    """Tests for GVPEncoder (BaseProteinEncoder wrapper)."""

    def test_wrapper_output_dims(self, device):
        """Wrapper should expose correct output_dims."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(128, 32),
            n_edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        assert encoder.output_dims == (128, 32)

    def test_wrapper_encoder_type(self, device):
        """Wrapper should return 'gvp' as encoder_type."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            n_edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        assert encoder.encoder_type == "gvp"

    def test_wrapper_forward(self, device, sample_hetero_data):
        """Wrapper forward should return (s, V, edge_attr) from HeteroData."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            n_edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        encoder = GVPEncoder(encoder=base_encoder, freeze=False)

        s, V, edge_attr = encoder(sample_hetero_data)

        assert s.shape == (sample_hetero_data["protein"].num_nodes, 64)
        assert V.shape == (sample_hetero_data["protein"].num_nodes, 16, 3)
        # edge_attr should be a tuple (s_edge, V_edge) when edge_update is enabled
        assert edge_attr is not None
        s_edge, V_edge = edge_attr
        assert s_edge.dim() == 2  # (E, scalar_dim)
        assert V_edge.dim() == 3  # (E, 1, 3)

    def test_wrapper_freeze(self, device):
        """Freeze parameter should disable gradients."""
        base_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(64, 16),
            n_edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        encoder = GVPEncoder(encoder=base_encoder, freeze=True)

        for p in encoder.encoder.parameters():
            assert not p.requires_grad


# ============== Cached Embedding Encoder Tests ==============


class TestCachedEmbeddingEncoder:
    """Tests for CachedEmbeddingEncoder (handles both SLAE and ESM)."""

    def test_output_dims_before_forward_raises(self, device):
        """output_dims should raise RuntimeError before forward pass."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae"
        ).to(device)
        with pytest.raises(RuntimeError, match="dimension not yet known"):
            _ = encoder.output_dims

    def test_slae_output_dims_after_forward(self, device, sample_hetero_data_with_slae):
        """SLAE encoder should infer output_dims from data."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae"
        ).to(device)
        encoder(sample_hetero_data_with_slae)
        assert encoder.output_dims == (128, 0)

    def test_esm_output_dims_after_forward(self, device, sample_hetero_data):
        """ESM encoder should infer output_dims from data."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="esm_embedding", encoder_type="esm"
        ).to(device)
        n_atoms = sample_hetero_data["protein"].num_nodes
        sample_hetero_data["protein"].esm_embedding = torch.randn(
            n_atoms, 1536, device=device
        )
        encoder(sample_hetero_data)
        assert encoder.output_dims == (1536, 0)

    def test_slae_encoder_type(self, device):
        """SLAE encoder should return 'slae' as encoder_type."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae"
        ).to(device)
        assert encoder.encoder_type == "slae"

    def test_esm_encoder_type(self, device):
        """ESM encoder should return 'esm' as encoder_type."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="esm_embedding", encoder_type="esm"
        ).to(device)
        assert encoder.encoder_type == "esm"

    def test_slae_forward(self, device, sample_hetero_data_with_slae):
        """SLAE forward pass should return (s, V, None) tuple with raw embeddings."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae"
        ).to(device)

        s, V, pp_edge_attr = encoder(sample_hetero_data_with_slae)

        n_atoms = sample_hetero_data_with_slae["protein"].num_nodes
        assert s.shape == (n_atoms, 128)
        assert V.shape == (n_atoms, 0, 3)
        # Raw embeddings should be identical to input
        assert torch.allclose(s, sample_hetero_data_with_slae["protein"].slae_embedding)
        # Cached embedding encoder doesn't return edge features
        assert pp_edge_attr is None

    def test_esm_forward(self, device, sample_hetero_data):
        """ESM forward pass should return (s, V, None) tuple with raw embeddings."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="esm_embedding", encoder_type="esm"
        ).to(device)

        # Add mock ESM embeddings
        n_atoms = sample_hetero_data["protein"].num_nodes
        sample_hetero_data["protein"].esm_embedding = torch.randn(
            n_atoms, 1536, device=device
        )

        s, V, pp_edge_attr = encoder(sample_hetero_data)

        assert s.shape == (n_atoms, 1536)
        assert V.shape == (n_atoms, 0, 3)
        # Raw embeddings should be identical to input
        assert torch.allclose(s, sample_hetero_data["protein"].esm_embedding)
        # Cached embedding encoder doesn't return edge features
        assert pp_edge_attr is None

    def test_slae_missing_embeddings_error(self, device, sample_hetero_data):
        """Should raise KeyError when SLAE embeddings are missing."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae"
        ).to(device)

        # sample_hetero_data does NOT have slae_embedding
        with pytest.raises(KeyError, match="requires cached embeddings"):
            encoder(sample_hetero_data)

    def test_esm_missing_embeddings_error(self, device, sample_hetero_data):
        """Should raise KeyError when ESM embeddings are missing."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="esm_embedding", encoder_type="esm"
        ).to(device)

        # sample_hetero_data does NOT have esm_embedding
        with pytest.raises(KeyError, match="requires cached embeddings"):
            encoder(sample_hetero_data)

    def test_encoder_no_nans(self, device, sample_hetero_data_with_slae):
        """Output should not contain NaNs or Infs."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae"
        ).to(device)

        s, V, _ = encoder(sample_hetero_data_with_slae)

        assert not torch.isnan(s).any(), "Scalar output contains NaNs"
        assert not torch.isinf(s).any(), "Scalar output contains Infs"

    def test_encoder_no_learnable_params(self, device):
        """Cached embedding encoder should have no learnable parameters."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae"
        ).to(device)
        assert sum(p.numel() for p in encoder.parameters()) == 0

    def test_slae_from_config(self, device, sample_hetero_data_with_slae):
        """Should construct SLAE from config and infer dim from data."""
        config = {"encoder_type": "slae"}
        encoder = CachedEmbeddingEncoder.from_config(config, device)
        assert encoder.encoder_type == "slae"
        encoder(sample_hetero_data_with_slae)
        assert encoder.output_dims == (128, 0)

    def test_esm_from_config(self, device, sample_hetero_data):
        """Should construct ESM from config and infer dim from data."""
        config = {"encoder_type": "esm"}
        encoder = CachedEmbeddingEncoder.from_config(config, device)
        assert encoder.encoder_type == "esm"
        n_atoms = sample_hetero_data["protein"].num_nodes
        sample_hetero_data["protein"].esm_embedding = torch.randn(
            n_atoms, 2048, device=device
        )
        encoder(sample_hetero_data)
        assert encoder.output_dims == (2048, 0)

    def test_device_placement(self, device, sample_hetero_data):
        """Verify tensors are on the correct device."""
        encoder = CachedEmbeddingEncoder(
            embedding_key="esm_embedding", encoder_type="esm"
        ).to(device)

        # Add mock ESM embeddings on correct device
        n_atoms = sample_hetero_data["protein"].num_nodes
        sample_hetero_data["protein"].esm_embedding = torch.randn(
            n_atoms, 1536, device=device
        )

        s, V, _ = encoder(sample_hetero_data)

        # Compare device types (handles cuda vs cuda:0)
        assert s.device.type == device.type, (
            f"Expected device type {device.type}, got {s.device.type}"
        )
        assert V.device.type == device.type, (
            f"Expected device type {device.type}, got {V.device.type}"
        )


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
                n_edge_scalar_in=16,
                pool_residue=False,
            ).to(device),
            freeze=False,
        )

        # SLAE encoder via CachedEmbeddingEncoder
        # embedding_dim=128 matches the fixture's slae_embedding shape
        slae_encoder = CachedEmbeddingEncoder(
            embedding_key="slae_embedding", encoder_type="slae", embedding_dim=128
        ).to(device)

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
