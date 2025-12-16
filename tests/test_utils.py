import pytest
import numpy as np
import torch

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import compute_rmsd, condot_pair_hard_hungarian, cov_prec_at_threshold


class TestComputeRMSD:
    """Tests for RMSD computation with Hungarian matching."""
    
    def test_identical_positions(self):
        """RMSD should be 0 for identical positions."""
        pred = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        target = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(0.0, abs=1e-6)
    
    def test_permuted_positions(self):
        """RMSD should be 0 for permuted identical positions (Hungarian handles this)."""
        pred = np.array([[2, 2, 2], [0, 0, 0], [1, 1, 1]])
        target = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(0.0, abs=1e-6)
    
    def test_known_rmsd(self):
        """Test RMSD with known value."""
        pred = np.array([[0, 0, 0], [1, 0, 0]])
        target = np.array([[0, 0, 0], [0, 1, 0]])
        # After optimal matching, RMSD = sqrt(mean([0, 1])) = sqrt(0.5) ≈ 0.707
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(np.sqrt(0.5), abs=1e-3)
    
    def test_torch_tensors(self):
        """Should work with torch tensors."""
        pred = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        target = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(0.0, abs=1e-6)


class TestCondotPairHardHungarian:
    """Tests for conditional OT with Hungarian algorithm."""
    
    def test_single_graph(self):
        """Test pairing for a single graph."""
        x1 = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        batch = torch.tensor([0, 0, 0])
        x0 = torch.randn(3, 3)
        
        x0_star, x1_star = condot_pair_hard_hungarian(x1, batch, x0)
        
        assert x0_star.shape == x1_star.shape == (3, 3)
        assert torch.allclose(x0_star, x0)  # x0 unchanged
        # x1_star should be a permutation of x1
        assert set(x1_star.flatten().tolist()) == pytest.approx(set(x1.flatten().tolist()), abs=1e-5)
    
    def test_multiple_graphs(self):
        """Test pairing with multiple graphs in batch."""
        x1 = torch.tensor([
            [0, 0, 0], [1, 1, 1],  # graph 0
            [2, 2, 2], [3, 3, 3],  # graph 1
        ], dtype=torch.float32)
        batch = torch.tensor([0, 0, 1, 1])
        x0 = torch.randn(4, 3)
        
        x0_star, x1_star = condot_pair_hard_hungarian(x1, batch, x0)
        
        assert x0_star.shape == x1_star.shape == (4, 3)


class TestCovPrecAtThreshold:
    """Tests for coverage and precision metrics."""
    
    def test_perfect_match(self):
        """Perfect match should give coverage=1, precision=1."""
        pred = np.array([[0, 0, 0], [1, 1, 1]])
        true = np.array([[0, 0, 0], [1, 1, 1]])
        cov, prec = cov_prec_at_threshold(pred, true, thresh=0.1)
        assert cov == 1.0
        assert prec == 1.0
    
    def test_partial_coverage(self):
        """Test partial coverage."""
        pred = np.array([[0, 0, 0], [1, 1, 1]])
        true = np.array([[0, 0, 0], [10, 10, 10]])  # second point far away
        cov, prec = cov_prec_at_threshold(pred, true, thresh=1.0)
        assert cov == 0.5  # only 1 of 2 true points covered
        assert prec == 0.5  # only 1 of 2 pred points has nearby true
    
    def test_empty_inputs(self):
        """Should handle empty inputs."""
        pred = np.array([]).reshape(0, 3)
        true = np.array([]).reshape(0, 3)
        cov, prec = cov_prec_at_threshold(pred, true, thresh=1.0)
        assert cov == 0.0
        assert prec == 0.0
    
    def test_torch_tensors(self):
        """Should work with torch tensors."""
        pred = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        true = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float32)
        cov, prec = cov_prec_at_threshold(pred, true, thresh=0.1)
        assert cov == 1.0
        assert prec == 1.0