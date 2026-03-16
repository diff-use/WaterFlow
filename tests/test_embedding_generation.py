"""
Tests for embedding generation and loading.

Tests:
1. Dataset loading with pre-computed embeddings
2. Backward compatibility (dataset works without embeddings)

"""

import tempfile
from pathlib import Path

import pytest
import torch


class TestDatasetEmbeddingLoading:
    """Test dataset loading with pre-computed embeddings."""

    def test_slae_embedding_loading(self):
        """Test that dataset loads SLAE embeddings if available."""
        from src.dataset import ProteinWaterDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy cache file with SLAE embeddings
            cache_data = {
                "protein_pos": torch.randn(50, 3),
                "protein_x": torch.randn(50, 16),
                "protein_res_idx": torch.arange(50),
                "protein_slae_embedding": torch.randn(50, 128),  # SLAE embeddings
                "water_pos": torch.randn(10, 3),
                "water_x": torch.randn(10, 16),
                "mate_pos": torch.zeros(0, 3),
                "mate_x": torch.zeros(0, 16),
                "mate_res_idx": torch.zeros(0, dtype=torch.long),
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
            assert "slae_embedding" in data["protein"], (
                "SLAE embeddings should be loaded"
            )
            assert data["protein"].slae_embedding.shape == (50, 128), (
                f"SLAE embedding shape mismatch: {data['protein'].slae_embedding.shape}"
            )

    def test_embedding_optional_backward_compat(self):
        """Test that dataset works without SLAE embeddings (backward compatibility)."""
        from src.dataset import ProteinWaterDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache file WITHOUT SLAE embeddings
            cache_data = {
                "protein_pos": torch.randn(30, 3),
                "protein_x": torch.randn(30, 16),
                "protein_res_idx": torch.arange(30),
                "water_pos": torch.randn(5, 3),
                "water_x": torch.randn(5, 16),
                "mate_pos": torch.zeros(0, 3),
                "mate_x": torch.zeros(0, 16),
                "mate_res_idx": torch.zeros(0, dtype=torch.long),
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
            assert "slae_embedding" not in data["protein"], (
                "SLAE embeddings should not be present when not in cache"
            )

    def test_embedding_with_mates(self):
        """Test that embeddings are correctly concatenated with mate embeddings."""
        from src.dataset import ProteinWaterDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache file with protein and mate SLAE embeddings
            cache_data = {
                "protein_pos": torch.randn(30, 3),
                "protein_x": torch.randn(30, 16),
                "protein_res_idx": torch.arange(30),
                "protein_slae_embedding": torch.randn(30, 128),
                "water_pos": torch.randn(5, 3),
                "water_x": torch.randn(5, 16),
                "mate_pos": torch.randn(10, 3),
                "mate_x": torch.randn(10, 16),
                "mate_res_idx": torch.arange(10),
                "mate_slae_embedding": torch.randn(10, 128),
            }

            cache_path = Path(tmpdir) / "test_final_C.pt"
            torch.save(cache_data, cache_path)

            list_file = Path(tmpdir) / "test_list.txt"
            list_file.write_text("test_final_C\n")

            dataset = ProteinWaterDataset(
                pdb_list_file=str(list_file),
                processed_dir=tmpdir,
                preprocess=False,
                include_mates=True,
            )

            data = dataset[0]

            # Embeddings should be concatenated (protein + mate)
            assert "slae_embedding" in data["protein"]
            assert data["protein"].slae_embedding.shape == (40, 128), (
                f"Expected (40, 128), got {data['protein'].slae_embedding.shape}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
