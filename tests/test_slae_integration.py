"""
Integration tests for SLAE encoder integration.

Tests:
1. SLAEToGVPAdapter dimension correctness
2. Dataset loading with SLAE embeddings
3. FlowWaterGVP forward pass with SLAE
4. No NaNs in outputs
5. Gradient flow through adapter

All test cases created with assistance from Claude Code and refined.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder_adapters import SLAEToGVPAdapter
from src.flow import FlowWaterGVP
from src.encoder import ProteinGVPEncoder


class DummyEncoder(nn.Module):
    """Dummy encoder for testing (acts like SLAE encoder)."""

    def __init__(self, out_dim=128):
        super().__init__()
        self.out_dim = out_dim
        self.dummy_layer = nn.Linear(1, out_dim)

    def forward(self, data):
        # Return dummy embeddings
        return {'node_embedding': torch.randn(100, self.out_dim)}


class TestSLAEToGVPAdapter:
    """Test the SLAE to GVP adapter."""

    def test_adapter_dimensions(self):
        """Test adapter output dimensions."""
        adapter = SLAEToGVPAdapter(slae_dim=128, out_dims=(256, 32))

        # Create dummy SLAE embeddings
        embeddings = torch.randn(100, 128)

        # Forward pass
        s, V = adapter(embeddings)

        # Check dimensions
        assert s.shape == (100, 256), f"Scalar shape mismatch: {s.shape}"
        assert V.shape == (100, 32, 3), f"Vector shape mismatch: {V.shape}"

    def test_adapter_zero_vectors(self):
        """Test that adapter outputs zero vectors."""
        adapter = SLAEToGVPAdapter(slae_dim=128, out_dims=(256, 32))
        embeddings = torch.randn(50, 128)

        s, V = adapter(embeddings)

        # Vectors should be zero
        assert torch.all(V == 0), "Vectors should be all zeros"

    def test_adapter_gradient_flow(self):
        """Test that gradients flow through adapter."""
        adapter = SLAEToGVPAdapter(slae_dim=128, out_dims=(256, 32))
        embeddings = torch.randn(20, 128, requires_grad=False)

        # Forward pass
        s, V = adapter(embeddings)

        # Check gradients can flow
        loss = s.sum()
        loss.backward()

        # Check adapter has gradients
        for param in adapter.parameters():
            assert param.grad is not None, "Adapter should have gradients"


class TestDatasetSLAELoading:
    """Test dataset loading with SLAE embeddings."""

    def test_slae_embedding_loading(self):
        """Test that dataset loads SLAE embeddings if available."""
        from src.dataset import ProteinWaterDataset
        import tempfile
        import os

        # Create temporary cache directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy cache file with SLAE embeddings
            cache_data = {
                'protein_pos': torch.randn(50, 3),
                'protein_x': torch.randn(50, 16),
                'protein_res_idx': torch.arange(50),
                'protein_slae_embedding': torch.randn(50, 128),  # SLAE embeddings
                'water_pos': torch.randn(10, 3),
                'water_x': torch.randn(10, 16),
                'mate_pos': torch.zeros(0, 3),
                'mate_x': torch.zeros(0, 16),
            }

            cache_path = Path(tmpdir) / "test_final_A.pt"
            torch.save(cache_data, cache_path)

            # Create PDB list file
            list_file = Path(tmpdir) / "test_list.txt"
            list_file.write_text("test_final_A\n")

            # Create dataset (skip preprocessing)
            dataset = ProteinWaterDataset(
                pdb_list_file=str(list_file),
                processed_dir=tmpdir,
                preprocess=False,
            )

            # Load data
            data = dataset[0]

            # Check SLAE embeddings are loaded
            assert 'slae_embedding' in data['protein'], "SLAE embeddings should be loaded"
            assert data['protein'].slae_embedding.shape == (50, 128), \
                f"SLAE embedding shape mismatch: {data['protein'].slae_embedding.shape}"

    def test_slae_embedding_optional(self):
        """Test that dataset works without SLAE embeddings (backward compatibility)."""
        from src.dataset import ProteinWaterDataset
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache file WITHOUT SLAE embeddings
            cache_data = {
                'protein_pos': torch.randn(30, 3),
                'protein_x': torch.randn(30, 16),
                'protein_res_idx': torch.arange(30),
                'water_pos': torch.randn(5, 3),
                'water_x': torch.randn(5, 16),
                'mate_pos': torch.zeros(0, 3),
                'mate_x': torch.zeros(0, 16),
            }

            cache_path = Path(tmpdir) / "test_final_B.pt"
            torch.save(cache_data, cache_path)

            list_file = Path(tmpdir) / "test_list.txt"
            list_file.write_text("test_final_B\n")

            dataset = ProteinWaterDataset(
                pdb_list_file=str(list_file),
                processed_dir=tmpdir,
                preprocess=False,
            )

            data = dataset[0]

            # Should work without SLAE embeddings
            assert 'slae_embedding' not in data['protein'], \
                "SLAE embeddings should not be present when not in cache"


class TestFlowWaterGVPWithSLAE:
    """Test FlowWaterGVP with SLAE encoder."""

    def create_dummy_data_with_slae(self, num_protein=50, num_water=10):
        """Create dummy HeteroData with SLAE embeddings."""
        data = HeteroData()

        # Protein nodes
        data['protein'].x = torch.randn(num_protein, 16)
        data['protein'].pos = torch.randn(num_protein, 3)
        data['protein'].residue_index = torch.randint(0, 10, (num_protein,))
        data['protein'].batch = torch.zeros(num_protein, dtype=torch.long)
        data['protein'].slae_embedding = torch.randn(num_protein, 128)  # SLAE embeddings

        # Water nodes
        data['water'].x = torch.randn(num_water, 16)
        data['water'].pos = torch.randn(num_water, 3)
        data['water'].batch = torch.zeros(num_water, dtype=torch.long)

        # Protein-protein edges
        edge_index = torch.randint(0, num_protein, (2, 100))
        data['protein', 'pp', 'protein'].edge_index = edge_index
        data['protein', 'pp', 'protein'].edge_rbf = torch.randn(100, 16)
        data['protein', 'pp', 'protein'].edge_vec = torch.randn(100, 3)

        return data

    def test_slae_forward_pass(self):
        """Test FlowWaterGVP forward pass with SLAE encoder."""
        # Create dummy encoder and adapter
        encoder = DummyEncoder(out_dim=128)
        adapter = SLAEToGVPAdapter(slae_dim=128, out_dims=(256, 32))

        # Create model
        model = FlowWaterGVP(
            encoder=encoder,
            encoder_type="slae",
            use_cached_slae=True,
            slae_adapter=adapter,
            hidden_dims=(256, 32),
            freeze_encoder=True,
        )

        # Create dummy data
        data = self.create_dummy_data_with_slae(num_protein=50, num_water=10)

        # Time
        t = torch.zeros(1)

        # Forward pass
        v_pred = model(data, t)

        # Check output shape
        assert v_pred.shape == (10, 3), f"Output shape mismatch: {v_pred.shape}"

    def test_slae_no_nans(self):
        """Test that SLAE forward pass doesn't produce NaNs."""
        encoder = DummyEncoder(out_dim=128)
        adapter = SLAEToGVPAdapter(slae_dim=128, out_dims=(256, 32))

        model = FlowWaterGVP(
            encoder=encoder,
            encoder_type="slae",
            use_cached_slae=True,
            slae_adapter=adapter,
            hidden_dims=(256, 32),
        )

        data = self.create_dummy_data_with_slae(num_protein=100, num_water=20)
        t = torch.ones(1) * 0.5

        v_pred = model(data, t)

        # Check for NaNs
        assert not torch.isnan(v_pred).any(), "Output contains NaNs"
        assert not torch.isinf(v_pred).any(), "Output contains Infs"

    def test_slae_gradient_flow(self):
        """Test that gradients flow through adapter and flow model."""
        encoder = DummyEncoder(out_dim=128)
        adapter = SLAEToGVPAdapter(slae_dim=128, out_dims=(256, 32))

        model = FlowWaterGVP(
            encoder=encoder,
            encoder_type="slae",
            use_cached_slae=True,
            slae_adapter=adapter,
            hidden_dims=(256, 32),
            freeze_encoder=True,
        )

        data = self.create_dummy_data_with_slae(num_protein=30, num_water=5)
        t = torch.ones(1) * 0.5

        # Forward pass
        v_pred = model(data, t)

        # Compute loss
        loss = v_pred.sum()
        loss.backward()

        # Check adapter has gradients
        for name, param in adapter.named_parameters():
            assert param.grad is not None, f"Adapter parameter {name} has no gradient"

        # Check encoder is frozen (no gradients)
        for param in encoder.parameters():
            assert param.grad is None, "Encoder should be frozen"

    def test_slae_missing_embeddings_error(self):
        """Test that model raises error when embeddings are missing and cached is required."""
        encoder = DummyEncoder(out_dim=128)
        adapter = SLAEToGVPAdapter(slae_dim=128, out_dims=(256, 32))

        model = FlowWaterGVP(
            encoder=encoder,
            encoder_type="slae",
            use_cached_slae=True,  # Requires cached embeddings
            slae_adapter=adapter,
            hidden_dims=(256, 32),
        )

        # Create data WITHOUT SLAE embeddings
        data = HeteroData()
        data['protein'].x = torch.randn(20, 16)
        data['protein'].pos = torch.randn(20, 3)
        data['protein'].batch = torch.zeros(20, dtype=torch.long)
        # NO slae_embedding field

        data['water'].x = torch.randn(5, 16)
        data['water'].pos = torch.randn(5, 3)
        data['water'].batch = torch.zeros(5, dtype=torch.long)

        t = torch.ones(1) * 0.5

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            model(data, t)


class TestGVPVsSLAEComparison:
    """Compare GVP and SLAE encoder outputs."""

    def test_both_encoders_run(self):
        """Test that both encoder types can run on same data."""
        # Create GVP model
        gvp_encoder = ProteinGVPEncoder(
            node_scalar_in=16,
            hidden_dims=(256, 32),
            edge_scalar_in=16,
        )

        gvp_model = FlowWaterGVP(
            encoder=gvp_encoder,
            encoder_type="gvp",
            hidden_dims=(256, 32),
        )

        # Create SLAE model
        slae_encoder = DummyEncoder(out_dim=128)
        slae_adapter = SLAEToGVPAdapter(slae_dim=128, out_dims=(256, 32))

        slae_model = FlowWaterGVP(
            encoder=slae_encoder,
            encoder_type="slae",
            use_cached_slae=True,
            slae_adapter=slae_adapter,
            hidden_dims=(256, 32),
        )

        # Create data for GVP
        gvp_data = HeteroData()
        gvp_data['protein'].x = torch.randn(30, 16)
        gvp_data['protein'].pos = torch.randn(30, 3)
        gvp_data['protein'].batch = torch.zeros(30, dtype=torch.long)
        gvp_data['water'].x = torch.randn(8, 16)
        gvp_data['water'].pos = torch.randn(8, 3)
        gvp_data['water'].batch = torch.zeros(8, dtype=torch.long)

        edge_index = torch.randint(0, 30, (2, 50))
        gvp_data['protein', 'pp', 'protein'].edge_index = edge_index
        gvp_data['protein', 'pp', 'protein'].edge_rbf = torch.randn(50, 16)
        gvp_data['protein', 'pp', 'protein'].edge_vec = torch.randn(50, 3)

        # Create data for SLAE (same structure + embeddings)
        slae_data = gvp_data.clone()
        slae_data['protein'].slae_embedding = torch.randn(30, 128)

        t = torch.ones(1) * 0.5

        # Both should run without errors
        gvp_output = gvp_model(gvp_data, t)
        slae_output = slae_model(slae_data, t)

        # Both should have same output shape
        assert gvp_output.shape == slae_output.shape == (8, 3), \
            f"Output shape mismatch: GVP={gvp_output.shape}, SLAE={slae_output.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
