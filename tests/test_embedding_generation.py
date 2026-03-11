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
            assert 'slae_embedding' in data['protein']
            assert data['protein'].slae_embedding.shape == (40, 128)
            assert torch.allclose(data['protein'].slae_embedding[30:], torch.zeros(10, 128))


class TestAlignSlaeToGeometry:
    """Test the align_slae_to_geometry function."""

    # Mock PROTEIN_ATOMS for testing (subset of actual 37 atom types)
    MOCK_PROTEIN_ATOMS = ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1"]

    @pytest.fixture
    def make_slae_test_data(self):
        """Factory fixture to create SLAE-like test data from atom specs.

        Args:
            atom_specs: List of (chain, res_id, ins_code, atom_name) tuples

        Returns:
            Dict with slae_emb, slae_residue_idx, slae_atom_type,
            slae_chains, slae_residue_ids, slae_ins_codes
        """
        import numpy as np

        def _make_data(atom_specs, embedding_dim=128):
            # Build residue info from unique (chain, res_id, ins_code) tuples
            residue_keys = []
            for chain, res_id, ins_code, _ in atom_specs:
                key = (chain, res_id, ins_code)
                if key not in residue_keys:
                    residue_keys.append(key)

            # Map residue key to index
            res_key_to_idx = {k: i for i, k in enumerate(residue_keys)}

            # Build arrays
            n_atoms = len(atom_specs)
            n_residues = len(residue_keys)

            slae_residue_idx = torch.zeros(n_atoms, dtype=torch.long)
            slae_atom_type = torch.zeros(n_atoms, dtype=torch.long)
            # Generate distinguishable embeddings: atom i gets value i+1 in first dim
            slae_emb = torch.zeros(n_atoms, embedding_dim)

            for i, (chain, res_id, ins_code, atom_name) in enumerate(atom_specs):
                res_key = (chain, res_id, ins_code)
                slae_residue_idx[i] = res_key_to_idx[res_key]
                # Map atom name to type index
                if atom_name in self.MOCK_PROTEIN_ATOMS:
                    slae_atom_type[i] = self.MOCK_PROTEIN_ATOMS.index(atom_name)
                else:
                    slae_atom_type[i] = 0  # Fallback
                # Use index-based value for easy verification
                slae_emb[i, 0] = float(i + 1)

            # Build per-residue arrays
            slae_chains = np.array([k[0] for k in residue_keys])
            slae_residue_ids = torch.tensor([k[1] for k in residue_keys])
            slae_ins_codes = np.array([k[2] for k in residue_keys])

            return {
                "slae_emb": slae_emb,
                "slae_residue_idx": slae_residue_idx,
                "slae_atom_type": slae_atom_type,
                "slae_chains": slae_chains,
                "slae_residue_ids": slae_residue_ids,
                "slae_ins_codes": slae_ins_codes,
            }

        return _make_data

    @pytest.mark.unit
    def test_perfect_alignment_single_residue(self, make_slae_test_data):
        """Same atoms in same order align exactly."""
        from unittest.mock import patch

        atom_specs = [
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
            ("A", 1, "", "C"),
            ("A", 1, "", "O"),
        ]
        data = make_slae_test_data(atom_specs)

        geometry_atom_info = [
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
            ("A", 1, "", "C"),
            ("A", 1, "", "O"),
        ]

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                data["slae_emb"],
                data["slae_residue_idx"],
                data["slae_atom_type"],
                data["slae_chains"],
                data["slae_residue_ids"],
                data["slae_ins_codes"],
                geometry_atom_info,
            )

        assert aligned.shape == (4, 128)
        # Check first dimension values match expected order
        assert torch.allclose(aligned[:, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    @pytest.mark.unit
    def test_reordered_atoms(self, make_slae_test_data):
        """Atoms reordered correctly when geometry order differs."""
        from unittest.mock import patch

        atom_specs = [
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
            ("A", 1, "", "C"),
            ("A", 1, "", "O"),
        ]
        data = make_slae_test_data(atom_specs)

        # Geometry has different order
        geometry_atom_info = [
            ("A", 1, "", "O"),
            ("A", 1, "", "C"),
            ("A", 1, "", "CA"),
            ("A", 1, "", "N"),
        ]

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                data["slae_emb"],
                data["slae_residue_idx"],
                data["slae_atom_type"],
                data["slae_chains"],
                data["slae_residue_ids"],
                data["slae_ins_codes"],
                geometry_atom_info,
            )

        assert aligned.shape == (4, 128)
        # O was index 3 (value 4), C was 2 (value 3), CA was 1 (value 2), N was 0 (value 1)
        assert torch.allclose(aligned[:, 0], torch.tensor([4.0, 3.0, 2.0, 1.0]))

    @pytest.mark.unit
    def test_noncanonical_atom_gets_zero_vector(self, make_slae_test_data):
        """Non-SLAE atoms (e.g., 'XX1') get zero embeddings."""
        from unittest.mock import patch

        atom_specs = [
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
        ]
        data = make_slae_test_data(atom_specs)

        # Geometry has a non-canonical atom
        geometry_atom_info = [
            ("A", 1, "", "N"),
            ("A", 1, "", "XX1"),  # Non-canonical
            ("A", 1, "", "CA"),
        ]

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                data["slae_emb"],
                data["slae_residue_idx"],
                data["slae_atom_type"],
                data["slae_chains"],
                data["slae_residue_ids"],
                data["slae_ins_codes"],
                geometry_atom_info,
            )

        assert aligned.shape == (3, 128)
        assert torch.allclose(aligned[0, 0], torch.tensor(1.0))  # N
        assert torch.allclose(aligned[1], torch.zeros(128))  # XX1 -> zero vector
        assert torch.allclose(aligned[2, 0], torch.tensor(2.0))  # CA

    @pytest.mark.unit
    def test_multiple_chains(self, make_slae_test_data):
        """Same res_id in different chains distinguished correctly."""
        from unittest.mock import patch

        atom_specs = [
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
            ("B", 1, "", "N"),
            ("B", 1, "", "CA"),
        ]
        data = make_slae_test_data(atom_specs)

        # Geometry requests atoms from both chains
        geometry_atom_info = [
            ("B", 1, "", "CA"),
            ("A", 1, "", "N"),
            ("B", 1, "", "N"),
            ("A", 1, "", "CA"),
        ]

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                data["slae_emb"],
                data["slae_residue_idx"],
                data["slae_atom_type"],
                data["slae_chains"],
                data["slae_residue_ids"],
                data["slae_ins_codes"],
                geometry_atom_info,
            )

        assert aligned.shape == (4, 128)
        # B:1:CA=4, A:1:N=1, B:1:N=3, A:1:CA=2
        assert torch.allclose(aligned[:, 0], torch.tensor([4.0, 1.0, 3.0, 2.0]))

    @pytest.mark.unit
    def test_with_insertion_codes(self, make_slae_test_data):
        """Insertion codes differentiate atoms properly."""
        from unittest.mock import patch

        atom_specs = [
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
            ("A", 1, "A", "N"),  # Same res_id but with insertion code
            ("A", 1, "A", "CA"),
        ]
        data = make_slae_test_data(atom_specs)

        geometry_atom_info = [
            ("A", 1, "A", "CA"),
            ("A", 1, "", "N"),
            ("A", 1, "A", "N"),
            ("A", 1, "", "CA"),
        ]

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                data["slae_emb"],
                data["slae_residue_idx"],
                data["slae_atom_type"],
                data["slae_chains"],
                data["slae_residue_ids"],
                data["slae_ins_codes"],
                geometry_atom_info,
            )

        assert aligned.shape == (4, 128)
        # A:1:A:CA=4, A:1::N=1, A:1:A:N=3, A:1::CA=2
        assert torch.allclose(aligned[:, 0], torch.tensor([4.0, 1.0, 3.0, 2.0]))

    @pytest.mark.unit
    def test_empty_geometry_list(self, make_slae_test_data):
        """Returns shape (0, embedding_dim) for empty geometry."""
        from unittest.mock import patch

        atom_specs = [
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
        ]
        data = make_slae_test_data(atom_specs)

        geometry_atom_info = []

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                data["slae_emb"],
                data["slae_residue_idx"],
                data["slae_atom_type"],
                data["slae_chains"],
                data["slae_residue_ids"],
                data["slae_ins_codes"],
                geometry_atom_info,
            )

        assert aligned.shape == (0, 128)

    @pytest.mark.unit
    def test_empty_slae_embeddings(self):
        """All geometry atoms get zero vectors when SLAE is empty."""
        import numpy as np
        from unittest.mock import patch

        # Empty SLAE data
        slae_emb = torch.zeros(0, 128)
        slae_residue_idx = torch.zeros(0, dtype=torch.long)
        slae_atom_type = torch.zeros(0, dtype=torch.long)
        slae_chains = np.array([])
        slae_residue_ids = torch.zeros(0, dtype=torch.long)
        slae_ins_codes = np.array([])

        geometry_atom_info = [
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
        ]

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                slae_emb,
                slae_residue_idx,
                slae_atom_type,
                slae_chains,
                slae_residue_ids,
                slae_ins_codes,
                geometry_atom_info,
            )

        assert aligned.shape == (2, 128)
        assert torch.allclose(aligned, torch.zeros(2, 128))

    @pytest.mark.unit
    def test_negative_residue_ids(self, make_slae_test_data):
        """Handles negative res_ids (like HIS A -1 in 6eey)."""
        from unittest.mock import patch

        atom_specs = [
            ("A", -1, "", "N"),
            ("A", -1, "", "CA"),
            ("A", 0, "", "N"),
            ("A", 1, "", "N"),
        ]
        data = make_slae_test_data(atom_specs)

        geometry_atom_info = [
            ("A", 1, "", "N"),
            ("A", -1, "", "CA"),
            ("A", 0, "", "N"),
            ("A", -1, "", "N"),
        ]

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                data["slae_emb"],
                data["slae_residue_idx"],
                data["slae_atom_type"],
                data["slae_chains"],
                data["slae_residue_ids"],
                data["slae_ins_codes"],
                geometry_atom_info,
            )

        assert aligned.shape == (4, 128)
        # A:1:N=4, A:-1:CA=2, A:0:N=3, A:-1:N=1
        assert torch.allclose(aligned[:, 0], torch.tensor([4.0, 2.0, 3.0, 1.0]))

    @pytest.mark.unit
    def test_with_6eey_pdb_subset(self, make_slae_test_data):
        """Integration test with real PDB atom patterns (6eey style)."""
        from unittest.mock import patch

        # Simulate 6eey-like structure: multiple chains, insertion codes, gaps
        atom_specs = [
            # Chain A residue 1
            ("A", 1, "", "N"),
            ("A", 1, "", "CA"),
            ("A", 1, "", "C"),
            ("A", 1, "", "O"),
            ("A", 1, "", "CB"),
            # Chain A residue 2 with insertion code
            ("A", 2, "A", "N"),
            ("A", 2, "A", "CA"),
            # Chain B residue 1 (same res_id as chain A)
            ("B", 1, "", "N"),
            ("B", 1, "", "CA"),
        ]
        data = make_slae_test_data(atom_specs)

        # Geometry with different order and some missing atoms
        geometry_atom_info = [
            ("B", 1, "", "CA"),  # Chain B first
            ("B", 1, "", "N"),
            ("A", 2, "A", "N"),  # Insertion code residue
            ("A", 1, "", "CB"),
            ("A", 1, "", "O"),
            ("A", 1, "", "C"),
            ("A", 1, "", "CA"),
            ("A", 1, "", "N"),
            ("A", 2, "A", "CA"),
            ("A", 1, "", "OXT"),  # Non-canonical - not in SLAE
        ]

        with patch(
            "scripts.generate_slae_embeddings.PROTEIN_ATOMS", self.MOCK_PROTEIN_ATOMS
        ):
            from scripts.generate_slae_embeddings import align_slae_to_geometry

            aligned = align_slae_to_geometry(
                data["slae_emb"],
                data["slae_residue_idx"],
                data["slae_atom_type"],
                data["slae_chains"],
                data["slae_residue_ids"],
                data["slae_ins_codes"],
                geometry_atom_info,
            )

        assert aligned.shape == (10, 128)
        # Verify specific mappings:
        # B:1:CA=9, B:1:N=8, A:2A:N=6, A:1:CB=5, A:1:O=4, A:1:C=3, A:1:CA=2, A:1:N=1, A:2A:CA=7, OXT=0
        expected = torch.tensor([9.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 7.0, 0.0])
        assert torch.allclose(aligned[:, 0], expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
