# test_utils.py

"""
Tests for src/utils.py utility functions.

Organized by category to match utils.py structure:
1. Feature encoding (rbf, atom37_to_atoms)
2. Optimal transport (ot_coupling)
3. Metrics (recall_precision, compute_rmsd, compute_placement_metrics)
4. Visualization (plot_3d_frame, save_protein_plot, create_trajectory_gif)

All test cases created with assistance from Claude Code and refined.
"""

import matplotlib
import numpy as np
import pytest
import torch


matplotlib.use("Agg")


from src.utils import (
    ATOM37_FILL,
    atom37_to_atoms,
    # Edge geometry
    compute_placement_metrics,
    compute_rmsd,
    ot_coupling,
    # Visualization
    plot_3d_frame,
    # Feature encoding
    rbf,
    # Metrics
    recall_precision,
    save_protein_plot,
)


@pytest.mark.unit
class TestRBF:
    """Tests for radial basis function encoding."""

    def test_output_shape(self):
        """RBF output should have shape (n_distances, num_gaussians)."""
        r = torch.rand(10) * 8.0
        out = rbf(r, num_gaussians=16, cutoff=8.0)
        assert out.shape == (10, 16)

    def test_different_num_basis(self):
        """RBF should respect num_gaussians parameter."""
        r = torch.rand(5) * 8.0
        for num_gaussians in [8, 16, 32, 64]:
            out = rbf(r, num_gaussians=num_gaussians, cutoff=8.0)
            assert out.shape == (5, num_gaussians)

    def test_zero_distance_finite(self):
        """RBF at zero distance should be finite (clamped)."""
        r = torch.zeros(10)
        out = rbf(r, num_gaussians=16, cutoff=8.0)
        assert torch.isfinite(out).all()

    def test_near_zero_finite(self):
        """RBF at very small distances should be finite."""
        r = torch.tensor([1e-10, 1e-8, 1e-6, 1e-4])
        out = rbf(r, num_gaussians=16, cutoff=8.0)
        assert torch.isfinite(out).all()

    def test_at_cutoff(self):
        """RBF at cutoff distance should be finite."""
        r = torch.tensor([7.9, 8.0, 8.1])
        out = rbf(r, num_gaussians=16, cutoff=8.0)
        assert torch.isfinite(out).all()

    def test_batched_input(self):
        """RBF should handle batched inputs."""
        r = torch.rand(100, device="cpu") * 10.0
        out = rbf(r, num_gaussians=16, cutoff=8.0)
        assert out.shape == (100, 16)
        assert torch.isfinite(out).all()

    def test_different_cutoffs(self):
        """RBF should work with different cutoff values."""
        r = torch.tensor([1.0, 2.0, 3.0])
        for cutoff in [4.0, 8.0, 12.0, 20.0]:
            out = rbf(r, num_gaussians=16, cutoff=cutoff)
            assert torch.isfinite(out).all()


@pytest.mark.unit
class TestAtom37ToAtoms:
    """Tests for atom37 representation conversion."""

    def test_basic_conversion(self):
        """Basic conversion from atom37 to flat atoms."""
        # Create atom37 tensor with some present atoms
        atom_tensor = torch.full((3, 37, 3), ATOM37_FILL)
        # Place atoms at specific positions
        atom_tensor[0, 0, :] = torch.tensor([1.0, 2.0, 3.0])  # CA of residue 0
        atom_tensor[0, 1, :] = torch.tensor([1.5, 2.5, 3.5])  # C of residue 0
        atom_tensor[1, 0, :] = torch.tensor([4.0, 5.0, 6.0])  # CA of residue 1

        coords, residue_idx, atom_type = atom37_to_atoms(atom_tensor)

        assert coords.shape == (3, 3)
        assert residue_idx.shape == (3,)
        assert atom_type.shape == (3,)

    def test_residue_indices_correct(self):
        """Residue indices should match the original residue."""
        atom_tensor = torch.full((2, 37, 3), ATOM37_FILL)
        atom_tensor[0, 0, :] = torch.tensor([1.0, 0.0, 0.0])
        atom_tensor[0, 1, :] = torch.tensor([2.0, 0.0, 0.0])
        atom_tensor[1, 5, :] = torch.tensor([3.0, 0.0, 0.0])

        _, residue_idx, atom_type = atom37_to_atoms(atom_tensor)

        assert residue_idx[0] == 0
        assert residue_idx[1] == 0
        assert residue_idx[2] == 1

    def test_atom_types_correct(self):
        """Atom types should match the slot index."""
        atom_tensor = torch.full((1, 37, 3), ATOM37_FILL)
        atom_tensor[0, 0, :] = torch.tensor([1.0, 0.0, 0.0])  # slot 0
        atom_tensor[0, 5, :] = torch.tensor([2.0, 0.0, 0.0])  # slot 5
        atom_tensor[0, 10, :] = torch.tensor([3.0, 0.0, 0.0])  # slot 10

        _, _, atom_type = atom37_to_atoms(atom_tensor)

        assert atom_type[0] == 0
        assert atom_type[1] == 5
        assert atom_type[2] == 10

    def test_empty_residues(self):
        """Empty residues should not contribute atoms."""
        atom_tensor = torch.full((3, 37, 3), ATOM37_FILL)
        # Only residue 0 has atoms
        atom_tensor[0, 0, :] = torch.tensor([1.0, 0.0, 0.0])

        coords, residue_idx, _ = atom37_to_atoms(atom_tensor)

        assert coords.shape == (1, 3)
        assert residue_idx[0] == 0

    def test_all_empty(self):
        """All-empty tensor should return empty outputs."""
        atom_tensor = torch.full((5, 37, 3), ATOM37_FILL)

        coords, residue_idx, atom_type = atom37_to_atoms(atom_tensor)

        assert coords.shape == (0, 3)
        assert residue_idx.shape == (0,)
        assert atom_type.shape == (0,)

    def test_coordinates_preserved(self):
        """Coordinates should be preserved exactly."""
        atom_tensor = torch.full((1, 37, 3), ATOM37_FILL)
        expected_coord = torch.tensor([1.234, 5.678, 9.012])
        atom_tensor[0, 0, :] = expected_coord

        coords, _, _ = atom37_to_atoms(atom_tensor)

        assert torch.allclose(coords[0], expected_coord)


@pytest.mark.unit
class TestOTCoupling:
    """Tests for OT coupling in flow matching."""

    def test_output_shapes(self):
        """Output shapes should match input shapes."""
        x1 = torch.rand(10, 3)
        x0 = torch.rand(10, 3)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])

        x0_star, x1_star = ot_coupling(x1, batch, x0)

        assert x0_star.shape == (10, 3)
        assert x1_star.shape == (10, 3)

    def test_single_graph(self):
        """Single graph should work correctly."""
        x1 = torch.rand(5, 3)
        x0 = torch.rand(5, 3)
        batch = torch.zeros(5, dtype=torch.long)

        x0_star, x1_star = ot_coupling(x1, batch, x0)

        assert x0_star.shape == x1.shape
        assert x1_star.shape == x1.shape

    def test_x0_unchanged(self):
        """x0 should be unchanged (only x1 is permuted)."""
        x1 = torch.rand(4, 3)
        x0 = torch.rand(4, 3)
        batch = torch.tensor([0, 0, 1, 1])

        x0_star, _ = ot_coupling(x1, batch, x0)

        assert torch.allclose(x0_star, x0)

    def test_deterministic(self):
        """Matching should be deterministic for same inputs."""
        x1 = torch.rand(10, 3)
        x0 = torch.rand(10, 3)
        batch = torch.zeros(10, dtype=torch.long)

        x0_star_1, x1_star_1 = ot_coupling(x1, batch, x0)
        x0_star_2, x1_star_2 = ot_coupling(x1, batch, x0)

        assert torch.allclose(x0_star_1, x0_star_2)
        assert torch.allclose(x1_star_1, x1_star_2)

    def test_is_permutation(self):
        """x1_star should be a permutation of x1 (per graph)."""
        x1 = torch.rand(5, 3)
        x0 = torch.rand(5, 3)
        batch = torch.zeros(5, dtype=torch.long)

        _, x1_star = ot_coupling(x1, batch, x0)

        # Every point in x1_star should exist in x1
        for i in range(len(x1_star)):
            min_dist = (x1 - x1_star[i]).norm(dim=-1).min()
            assert min_dist < 1e-5

    def test_no_cross_batch_matching(self):
        """Matching should not cross batch boundaries."""
        # Two graphs, far apart
        x1 = torch.cat([torch.randn(5, 3), torch.randn(5, 3) + 100.0])
        x0 = torch.cat([torch.randn(5, 3), torch.randn(5, 3) + 100.0])
        batch = torch.cat(
            [torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long)]
        )

        _, x1_star = ot_coupling(x1, batch, x0)

        # Points from graph 0 shouldn't match to graph 1
        x1_star_g0 = x1_star[:5]
        x1_g1 = x1[5:]

        for i in range(5):
            min_dist_to_g1 = (x1_g1 - x1_star_g0[i]).norm(dim=-1).min()
            assert min_dist_to_g1 > 50.0

    def test_optimal_matching(self):
        """Matching should minimize total cost."""
        # Simple case: x0 = [0,0,0], x1 = [[0,0,0], [1,0,0]]
        x0 = torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
        x1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        batch = torch.zeros(2, dtype=torch.long)

        _, x1_star = ot_coupling(x1, batch, x0)

        # Optimal: [0,0,0] -> [0,0,0], [0.9,0,0] -> [1,0,0]
        # x1_star[0] should be [0,0,0]
        # x1_star[1] should be [1,0,0]
        assert torch.allclose(x1_star[0], torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(x1_star[1], torch.tensor([1.0, 0.0, 0.0]))


@pytest.mark.unit
class TestRecallPrecision:
    """Tests for recall_precision metric."""

    def test_perfect_overlap(self):
        """Perfect overlap should give recall=1, precision=1."""
        pts = torch.rand(5, 3)
        recall, precision = recall_precision(pts, pts.clone(), thresh=0.1)
        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(1.0)

    def test_no_overlap(self):
        """No overlap should give recall=0, precision=0."""
        pred = torch.tensor([[0.0, 0.0, 0.0]])
        true = torch.tensor([[100.0, 100.0, 100.0]])
        recall, precision = recall_precision(pred, true, thresh=1.0)
        assert recall == pytest.approx(0.0)
        assert precision == pytest.approx(0.0)

    def test_empty_pred(self):
        """Empty predictions should return 0, 0."""
        pred = torch.zeros((0, 3))
        true = torch.rand(5, 3)
        recall, precision = recall_precision(pred, true, thresh=1.0)
        assert recall == 0.0
        assert precision == 0.0

    def test_empty_true(self):
        """Empty ground truth should return 0, 0."""
        pred = torch.rand(5, 3)
        true = torch.zeros((0, 3))
        recall, precision = recall_precision(pred, true, thresh=1.0)
        assert recall == 0.0
        assert precision == 0.0

    def test_partial_coverage(self):
        """Partial coverage should give expected values."""
        pred = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        true = torch.tensor([[0.0, 0.0, 0.0]])
        recall, precision = recall_precision(pred, true, thresh=1.0)
        assert recall == pytest.approx(1.0)  # All true covered
        assert precision == pytest.approx(0.5)  # Half of pred correct

    def test_numpy_input(self):
        """Should work with numpy arrays."""
        pred = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        true = np.array([[0.0, 0.0, 0.0]])
        recall, precision = recall_precision(pred, true, thresh=0.5)
        assert recall == pytest.approx(1.0)
        assert precision == pytest.approx(0.5)

    def test_different_thresholds(self):
        """Different thresholds should give different results."""
        pred = torch.tensor([[0.0, 0.0, 0.0]])
        true = torch.tensor([[0.5, 0.0, 0.0]])

        # Within threshold
        recall_close, _ = recall_precision(pred, true, thresh=1.0)
        assert recall_close == 1.0

        # Outside threshold
        recall_far, _ = recall_precision(pred, true, thresh=0.3)
        assert recall_far == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_tensors(self):
        """Should work with GPU tensors."""
        pred = torch.rand(10, 3, device="cuda")
        true = torch.rand(10, 3, device="cuda")
        recall, precision = recall_precision(pred, true, thresh=5.0)
        assert isinstance(recall, float)
        assert isinstance(precision, float)


@pytest.mark.unit
class TestComputeRMSD:
    """Tests for RMSD computation with Hungarian matching."""

    def test_identical_points(self):
        """RMSD of identical points should be 0."""
        pts = torch.rand(5, 3)
        rmsd = compute_rmsd(pts, pts.clone())
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_known_displacement(self):
        """RMSD of known displacement should match expected."""
        pred = torch.tensor([[0.0, 0.0, 0.0]])
        target = torch.tensor([[1.0, 0.0, 0.0]])
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(1.0, abs=1e-6)

    def test_permutation_invariance(self):
        """RMSD should be invariant to permutation (Hungarian)."""
        pred = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        target = torch.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_numpy_input(self):
        """Should work with numpy arrays."""
        pred = np.array([[0.0, 0.0, 0.0]])
        target = np.array([[1.0, 0.0, 0.0]])
        rmsd = compute_rmsd(pred, target)
        assert rmsd == pytest.approx(1.0, abs=1e-6)

    def test_batched(self):
        """Batched RMSD should average over graphs."""
        pred = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # graph 0
                [1.0, 0.0, 0.0],  # graph 0
                [0.0, 0.0, 0.0],  # graph 1
            ]
        )
        target = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],  # 2.0 away
            ]
        )
        batch = torch.tensor([0, 0, 1])

        rmsd = compute_rmsd(pred, target, batch=batch)
        # Graph 0: RMSD = 0, Graph 1: RMSD = 2.0
        expected = (0.0 + 2.0) / 2
        assert rmsd == pytest.approx(expected, abs=1e-6)


@pytest.mark.unit
class TestComputePlacementMetrics:
    """Tests for compute_placement_metrics function."""

    def test_perfect_placement(self):
        """Perfect placement should give high metrics."""
        pts = torch.rand(10, 3)
        metrics = compute_placement_metrics(pts, pts.clone(), threshold=0.1)

        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_no_overlap(self):
        """No overlap should give zero metrics."""
        pred = torch.zeros(5, 3)
        true = torch.ones(5, 3) * 100

        metrics = compute_placement_metrics(pred, true, threshold=1.0)

        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["f1"] == 0.0

    def test_empty_inputs(self):
        """Empty inputs should return zero metrics."""
        metrics = compute_placement_metrics(
            np.zeros((0, 3)), np.zeros((5, 3)), threshold=1.0
        )
        assert metrics["recall"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["f1"] == 0.0
        assert metrics["auc_pr"] == 0.0

    def test_auc_pr_range(self):
        """AUC-PR should be in [0, 1]."""
        pred = torch.rand(20, 3) * 10
        true = torch.rand(15, 3) * 10

        metrics = compute_placement_metrics(pred, true, threshold=1.0)

        assert 0.0 <= metrics["auc_pr"] <= 1.0

    def test_f1_formula(self):
        """F1 should be harmonic mean of precision and recall."""
        pred = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        true = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

        metrics = compute_placement_metrics(pred, true, threshold=1.0)

        expected_f1 = (
            2
            * metrics["precision"]
            * metrics["recall"]
            / (metrics["precision"] + metrics["recall"] + 1e-8)
        )
        assert metrics["f1"] == pytest.approx(expected_f1, abs=1e-6)


@pytest.mark.unit
class TestPlot3DFrame:
    """Tests for 3D plotting function."""

    def test_runs_without_error(self):
        """Basic plot should not raise."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_frame(
            ax,
            np.random.rand(10, 3),
            np.random.rand(3, 3),
            np.random.rand(5, 3),
            np.random.rand(5, 3),
            title="Test",
        )
        plt.close(fig)

    def test_no_mates(self):
        """Plot with no mates should work."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_frame(
            ax, np.random.rand(10, 3), None, np.random.rand(5, 3), np.random.rand(5, 3)
        )
        plt.close(fig)

    def test_empty_mates(self):
        """Plot with empty mates array should work."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_frame(
            ax,
            np.random.rand(10, 3),
            np.zeros((0, 3)),  # Empty mates
            np.random.rand(5, 3),
            np.random.rand(5, 3),
        )
        plt.close(fig)

    def test_with_axis_limits(self):
        """Plot with axis limits should work."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_3d_frame(
            ax,
            np.random.rand(10, 3),
            None,
            np.random.rand(5, 3),
            np.random.rand(5, 3),
            xlim=(-1, 1),
            ylim=(-1, 1),
            zlim=(-1, 1),
        )
        plt.close(fig)


@pytest.mark.unit
class TestSaveProteinPlot:
    """Tests for protein CA trace plotting."""

    def test_saves_file(self, tmp_path):
        """Plot should be saved to disk."""
        save_protein_plot(
            torch.rand(20, 3), torch.rand(20, 3), step=1, save_dir=str(tmp_path)
        )
        assert (tmp_path / "step_1.png").exists()

    def test_different_sizes(self, tmp_path):
        """Should work with different protein sizes."""
        for n in [5, 20, 100]:
            save_protein_plot(
                torch.rand(n, 3), torch.rand(n, 3), step=n, save_dir=str(tmp_path)
            )
            assert (tmp_path / f"step_{n}.png").exists()


# commenting the test below out as gif creation is just a viz tool and this test takes too long to run
# @pytest.mark.unit
# class TestCreateTrajectoryGif:
#     """Tests for GIF creation from trajectory."""

#     def test_creates_gif(self, tmp_path):
#         """GIF should be created from trajectory."""
#         trajectory = [np.random.rand(5, 3) for _ in range(10)]
#         protein_pos = np.random.rand(20, 3)
#         water_true = np.random.rand(5, 3)

#         gif_path = str(tmp_path / "test.gif")
#         create_trajectory_gif(
#             trajectory=trajectory,
#             protein_pos=protein_pos,
#             water_true=water_true,
#             save_path=gif_path,
#             fps=5,
#         )

#         assert Path(gif_path).exists()

#     def test_with_pdb_id(self, tmp_path):
#         """GIF should work with pdb_id parameter."""
#         trajectory = [np.random.rand(3, 3) for _ in range(5)]
#         protein_pos = np.random.rand(10, 3)
#         water_true = np.random.rand(3, 3)

#         gif_path = str(tmp_path / "test_pdb.gif")
#         create_trajectory_gif(
#             trajectory=trajectory,
#             protein_pos=protein_pos,
#             water_true=water_true,
#             save_path=gif_path,
#             pdb_id="1ABC",
#         )

#         assert Path(gif_path).exists()

#     def test_long_trajectory_sampled(self, tmp_path):
#         """Long trajectories should be sampled to max 100 frames."""
#         trajectory = [np.random.rand(3, 3) for _ in range(200)]
#         protein_pos = np.random.rand(10, 3)
#         water_true = np.random.rand(3, 3)

#         gif_path = str(tmp_path / "long.gif")
#         create_trajectory_gif(
#             trajectory=trajectory,
#             protein_pos=protein_pos,
#             water_true=water_true,
#             save_path=gif_path,
#         )

#         assert Path(gif_path).exists()
