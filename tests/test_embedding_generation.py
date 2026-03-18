"""
Tests for embedding generation and loading with modular cache layout.

Layout under cache root:
- geometry/<cache_key>.pt
- slae/<cache_key>.pt
- esm/<cache_key>.pt
"""

import tempfile
from pathlib import Path

import pytest
import torch


class TestDatasetEmbeddingLoading:
    """Unit tests for dataset embedding loading with mock cache data.

    These tests use synthetic cache files to verify embedding loading logic
    in isolation. For integration tests with real PDB files, see
    tests/test_dataset.py::TestEmbeddingLoading.
    """

    def test_slae_embedding_loading(self):
        """Dataset should load SLAE embeddings from cache_root/slae/."""
        from src.dataset import ProteinWaterDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            (cache_root / "geometry").mkdir(parents=True, exist_ok=True)
            (cache_root / "slae").mkdir(parents=True, exist_ok=True)

            # Geometry cache (new schema with pp_edge_index and metadata)
            protein_pos = torch.randn(50, 3)
            geom_data = {
                'protein_pos': protein_pos,
                'protein_x': torch.randn(50, 16),
                'protein_res_idx': torch.arange(50),
                'water_pos': torch.randn(10, 3),
                'water_x': torch.randn(10, 16),
                'pp_edge_index': torch.empty((2, 0), dtype=torch.long),
                'pp_edge_unit': torch.empty((0, 3)),
                'pp_edge_rbf': torch.empty((0, 16)),
                'num_asu_protein': 50,
                'num_protein_residues': 50,
                'include_mates': False,
            }
            torch.save(geom_data, cache_root / "geometry" / "test_final.pt")

            # SLAE embedding cache (new script schema)
            slae_data = {
                'node_embeddings': torch.randn(50, 128),
                'atom37_coords': torch.zeros(10, 37, 3),
                'pdb_id': 'test',
            }
            torch.save(slae_data, cache_root / "slae" / "test_final.pt")

            list_file = cache_root / "test_list.txt"
            list_file.write_text("test_final\n")

            dataset = ProteinWaterDataset(
                pdb_list_file=str(list_file),
                processed_dir=str(cache_root),
                encoder_type="slae",
                preprocess=False,
                include_mates=False,  # Use no-mates cache key
            )

            data = dataset[0]
            assert 'slae_embedding' in data['protein']
            assert data['protein'].slae_embedding.shape == (50, 128)

    def test_embedding_optional_backward_compat(self):
        """Dataset should work without embedding subdirectories present."""
        from src.dataset import ProteinWaterDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            (cache_root / "geometry").mkdir(parents=True, exist_ok=True)

            # Geometry cache (new schema with pp_edge_index and metadata)
            geom_data = {
                'protein_pos': torch.randn(30, 3),
                'protein_x': torch.randn(30, 16),
                'protein_res_idx': torch.arange(30),
                'water_pos': torch.randn(5, 3),
                'water_x': torch.randn(5, 16),
                'pp_edge_index': torch.empty((2, 0), dtype=torch.long),
                'pp_edge_unit': torch.empty((0, 3)),
                'pp_edge_rbf': torch.empty((0, 16)),
                'num_asu_protein': 30,
                'num_protein_residues': 30,
                'include_mates': False,
            }
            torch.save(geom_data, cache_root / "geometry" / "test_final.pt")

            list_file = cache_root / "test_list.txt"
            list_file.write_text("test_final\n")

            dataset = ProteinWaterDataset(
                pdb_list_file=str(list_file),
                processed_dir=str(cache_root),
                encoder_type="gvp",
                preprocess=False,
                include_mates=False,  # Use no-mates cache key
            )

            data = dataset[0]
            assert 'slae_embedding' not in data['protein']
            assert 'esm_embedding' not in data['protein']

    def test_embedding_with_mates(self):
        """SLAE embeddings should be zero-padded for mate atoms."""
        from src.dataset import ProteinWaterDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir)
            # When include_mates=True, dataset looks in geometry_mates/ subdirectory
            (cache_root / "geometry_mates").mkdir(parents=True, exist_ok=True)
            (cache_root / "slae").mkdir(parents=True, exist_ok=True)

            # Geometry cache includes mates (new schema with pp_edge_index)
            # protein_pos already includes mates when include_mates=True
            protein_pos = torch.randn(30, 3)
            mate_pos = torch.randn(10, 3)
            combined_pos = torch.cat([protein_pos, mate_pos], dim=0)
            protein_res_idx = torch.arange(30)
            mate_res_idx = torch.arange(10) + 30  # Offset by max protein residue

            geom_data = {
                'protein_pos': combined_pos,
                'protein_x': torch.randn(40, 16),
                'protein_res_idx': torch.cat([protein_res_idx, mate_res_idx]),
                'water_pos': torch.randn(5, 3),
                'water_x': torch.randn(5, 16),
                'pp_edge_index': torch.empty((2, 0), dtype=torch.long),
                'pp_edge_unit': torch.empty((0, 3)),
                'pp_edge_rbf': torch.empty((0, 16)),
                'num_asu_protein': 30,  # ASU atom count for embedding validation
                'num_protein_residues': 30,
                'include_mates': True,
            }
            torch.save(geom_data, cache_root / "geometry_mates" / "test_final.pt")

            slae_data = {
                'node_embeddings': torch.randn(30, 128),
                'atom37_coords': torch.zeros(10, 37, 3),
                'pdb_id': 'test',
            }
            torch.save(slae_data, cache_root / "slae" / "test_final.pt")

            list_file = cache_root / "test_list.txt"
            list_file.write_text("test_final\n")

            dataset = ProteinWaterDataset(
                pdb_list_file=str(list_file),
                processed_dir=str(cache_root),
                encoder_type="slae",
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
