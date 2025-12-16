# test_utils.py

import pytest
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    rbf,
    compute_rmsd,
    condot_pair_hard_hungarian,
    cov_prec_at_threshold,
    plot_3d_frame,
    save_protein_plot,
)

@pytest.mark.unit
class TestRBF:
    def test_output_shape(self):
        r = torch.rand(10) * 8.0  # values within cutoff
        out = rbf(r, num_gaussians=16, cutoff=8.0)
        assert out.shape == (10, 16)

    def test_different_num_basis(self):
        r = torch.rand(5) * 8.0
        out = rbf(r, num_gaussians=32, cutoff=8.0)
        assert out.shape == (5, 32)

    def test_positive_input(self):
        r = torch.tensor([0.1, 1.0, 4.0, 7.9])  # avoid exact 0
        out = rbf(r, num_gaussians=16, cutoff=8.0)
        assert out.shape == (4, 16)
        assert torch.isfinite(out).all()


@pytest.mark.unit
class TestComputeRMSD:
    def test_identical_points(self):
        pts = torch.rand(5, 3)
        rmsd = compute_rmsd(pts, pts.clone())
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_known_displacement(self):
        pred = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(1.0, abs=1e-6)

    def test_permutation_invariance(self):
        pred = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        target = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
class TestCondotPairHardHungarian:
    def test_output_shapes(self):
        x1 = torch.rand(10, 3)
        x0 = torch.rand(10, 3)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        x0_star, x1_star = condot_pair_hard_hungarian(x1, batch, x0)
        assert x0_star.shape == (10, 3)
        assert x1_star.shape == (10, 3)

    def test_single_graph(self):
        x1 = torch.rand(5, 3)
        x0 = torch.rand(5, 3)
        batch = torch.zeros(5, dtype=torch.long)
        x0_star, x1_star = condot_pair_hard_hungarian(x1, batch, x0)
        assert x0_star.shape == x1.shape
        assert x1_star.shape == x1.shape

    def test_x0_unchanged(self):
        x1 = torch.rand(4, 3)
        x0 = torch.rand(4, 3)
        batch = torch.tensor([0, 0, 1, 1])
        x0_star, _ = condot_pair_hard_hungarian(x1, batch, x0)
        assert torch.allclose(x0_star, x0)


@pytest.mark.unit
class TestCovPrecAtThreshold:
    def test_perfect_overlap(self):
        pts = torch.rand(5, 3)
        cov, prec = cov_prec_at_threshold(pts, pts.clone(), thresh=0.1)
        assert cov == pytest.approx(1.0)
        assert prec == pytest.approx(1.0)

    def test_no_overlap(self):
        pred = torch.tensor([[0.0, 0.0, 0.0]])
        true = torch.tensor([[100.0, 100.0, 100.0]])
        cov, prec = cov_prec_at_threshold(pred, true, thresh=1.0)
        assert cov == pytest.approx(0.0)
        assert prec == pytest.approx(0.0)

    def test_empty_arrays(self):
        cov, prec = cov_prec_at_threshold(np.zeros((0, 3)), np.array([[1, 2, 3]]))
        assert cov == 0.0
        assert prec == 0.0

    def test_partial_coverage(self):
        pred = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        true = torch.tensor([[0.0, 0.0, 0.0]])
        cov, prec = cov_prec_at_threshold(pred, true, thresh=1.0)
        assert cov == pytest.approx(1.0)
        assert prec == pytest.approx(0.5)


@pytest.mark.unit
class TestPlot3DFrame:
    def test_runs_without_error(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_frame(ax, np.random.rand(10, 3), np.random.rand(3, 3),
                      np.random.rand(5, 3), np.random.rand(5, 3), title="Test")
        plt.close(fig)

    def test_no_mates(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_frame(ax, np.random.rand(10, 3), None,
                      np.random.rand(5, 3), np.random.rand(5, 3))
        plt.close(fig)


@pytest.mark.unit
class TestSaveProteinPlot:
    def test_saves_file(self, tmp_path):
        save_protein_plot(torch.rand(20, 3), torch.rand(20, 3), step=1, save_dir=str(tmp_path))
        assert (tmp_path / "step_1.png").exists()