"""Unit tests for src/confidence.py -- targets, clustering, and ConfidenceGVP."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from src.confidence import (
    cluster_waters_vdw,
    ConfidenceGVP,
    smootherstep_confidence,
    smootherstep_target,
)
from src.constants import EDGE_PP, EDGE_PW, NUM_RBF
from src.utils import compute_edge_features


# ============== smootherstep target ==============


@pytest.mark.unit
class TestSmootherstepTarget:
    def test_plateau_and_floor(self, device):
        gt = torch.zeros(1, 3, device=device)
        ds = torch.tensor([0.0, 0.4, 2.0, 5.0], device=device)
        cand = torch.stack([ds, torch.zeros_like(ds), torch.zeros_like(ds)], dim=1)
        out = smootherstep_target(cand, gt, r_in=0.4, r_out=2.0)
        assert out[0].item() == pytest.approx(1.0, abs=1e-6)  # d < r_in -> 1
        assert out[1].item() == pytest.approx(1.0, abs=1e-6)  # d == r_in -> 1
        assert out[2].item() == pytest.approx(0.0, abs=1e-6)  # d == r_out -> 0
        assert out[3].item() == pytest.approx(0.0, abs=1e-6)  # far -> 0

    def test_midpoint_is_half(self, device):
        gt = torch.zeros(1, 3, device=device)
        cand = torch.tensor([[1.2, 0.0, 0.0]], device=device)  # midpoint of [0.4, 2.0]
        out = smootherstep_target(cand, gt, r_in=0.4, r_out=2.0)
        assert out.item() == pytest.approx(0.5, abs=1e-6)

    def test_monotone_decreasing(self, device):
        gt = torch.zeros(1, 3, device=device)
        ds = torch.linspace(0.0, 2.5, 12, device=device)
        cand = torch.stack([ds, torch.zeros_like(ds), torch.zeros_like(ds)], dim=1)
        out = smootherstep_target(cand, gt, r_in=0.4, r_out=2.0)
        diffs = out[1:] - out[:-1]
        assert (diffs <= 1e-7).all()  # non-increasing in distance

    def test_values_in_unit_interval(self, device):
        gt = torch.randn(3, 3, device=device)
        cand = torch.randn(20, 3, device=device) * 3.0
        out = smootherstep_target(cand, gt, r_in=0.4, r_out=2.0)
        assert (out >= 0).all() and (out <= 1.0).all()

    def test_nearest_gt_is_used(self, device):
        # The nearest GT sets the target; the far one must not dilute it.
        gt = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], device=device)
        cand = torch.tensor([[1.2, 0.0, 0.0]], device=device)
        out = smootherstep_target(cand, gt, r_in=0.4, r_out=2.0)
        assert out.item() == pytest.approx(0.5, abs=1e-6)

    def test_confidence_matches_hand_values(self, device):
        d = torch.tensor([0.5, 0.8, 1.0, 1.5], device=device)
        out = smootherstep_confidence(d, r_in=0.4, r_out=2.0)
        expected = torch.tensor([0.998, 0.896, 0.725, 0.179], device=device)
        assert torch.allclose(out, expected, atol=2e-3)

    def test_default_radii_are_half_and_one_and_a_half(self, device):
        """The 0.5-crossing of the shipped defaults sits at 1.0 A."""
        gt = torch.zeros(1, 3, device=device)
        cand = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        assert smootherstep_target(cand, gt).item() == pytest.approx(0.5, abs=1e-6)

    def test_empty_candidates_ok(self, device):
        gt = torch.randn(3, 3, device=device)
        cand = torch.empty(0, 3, device=device)
        out = smootherstep_target(cand, gt)
        assert out.shape == (0,)

    def test_empty_gt_raises(self, device):
        cand = torch.randn(3, 3, device=device)
        gt = torch.empty(0, 3, device=device)
        with pytest.raises(ValueError, match="at least one GT"):
            smootherstep_target(cand, gt)

    def test_bad_radii_raises(self, device):
        gt = torch.zeros(1, 3, device=device)
        cand = torch.zeros(1, 3, device=device)
        with pytest.raises(ValueError, match="r_out"):
            smootherstep_target(cand, gt, r_in=2.0, r_out=1.0)


# ============== vdW clustering ==============


@pytest.mark.unit
class TestClusterWatersVdw:
    def test_two_close_waters_collapse(self, device):
        # 1 A apart with r=1.52 => absorbed.
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
        conf = torch.tensor([0.9, 0.8], device=device)
        out_pos, out_conf = cluster_waters_vdw(pos, conf, radius=1.52)
        assert out_pos.size(0) == 1
        expected_x = (0.9 * 0.0 + 0.8 * 1.0) / (0.9 + 0.8)
        assert out_pos[0, 0].item() == pytest.approx(expected_x, abs=1e-5)
        assert out_conf[0].item() == pytest.approx(0.9, abs=1e-6)  # cluster max

    def test_two_far_waters_preserved(self, device):
        # 2 A apart with r=1.52 => kept separate.
        pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=device)
        conf = torch.tensor([0.9, 0.8], device=device)
        out_pos, out_conf = cluster_waters_vdw(pos, conf, radius=1.52)
        assert out_pos.size(0) == 2
        # Each water is its own singleton cluster, so its centroid is its own
        # position and confidence, emitted highest-confidence first.
        assert torch.allclose(out_pos, pos)
        assert out_conf[0].item() == pytest.approx(0.9)
        assert out_conf[1].item() == pytest.approx(0.8)

    def test_threshold_filters_pre_cluster(self, device):
        # 0.5 sits exactly on the threshold and must survive: the cut is conf >= threshold.
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [15.0, 0.0, 0.0]],
            device=device,
        )
        conf = torch.tensor([0.1, 0.5, 0.6, 0.9], device=device)
        out_pos, out_conf = cluster_waters_vdw(pos, conf, radius=1.52, threshold=0.5)
        assert out_pos.size(0) == 3  # only 0.1 dropped; the 0.5 boundary is kept
        assert (out_conf >= 0.5).all()

    def test_threshold_runs_before_clustering(self, device):
        """
        A sub-threshold neighbour must not pull the centroid.

        Filtering after clustering would let it vote in the weighted mean, so
        the surviving centroid would land off its own position.
        """
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
        conf = torch.tensor([0.9, 0.1], device=device)
        out_pos, _ = cluster_waters_vdw(pos, conf, radius=1.52, threshold=0.5)
        assert out_pos.size(0) == 1
        assert out_pos[0, 0].item() == pytest.approx(0.0, abs=1e-6)

    def test_output_sorted_descending_confidence(self, device):
        torch.manual_seed(0)
        pos = torch.randn(30, 3, device=device) * 5.0
        conf = torch.rand(30, device=device)
        _, out_conf = cluster_waters_vdw(pos, conf, radius=1.52)
        assert (out_conf[:-1] >= out_conf[1:]).all()

    def test_no_pair_within_radius_after_clustering(self, device):
        torch.manual_seed(1)
        pos = torch.randn(50, 3, device=device) * 3.0
        conf = torch.rand(50, device=device)
        out_pos, _ = cluster_waters_vdw(pos, conf, radius=1.52)
        if out_pos.size(0) > 1:
            diffs = out_pos.unsqueeze(0) - out_pos.unsqueeze(1)
            dists = diffs.norm(dim=-1)
            mask = ~torch.eye(out_pos.size(0), dtype=torch.bool, device=device)
            assert (dists[mask] > 1.52 - 1e-5).all()

    def test_nms_operates_on_centroids_not_seeds(self, device):
        """
        Round 2 must compare centroids, not the seeds they came from.

        Seeds at 0.0 and 1.6 are 1.6 A apart, so round 1 keeps them separate.
        Absorbing the 1.5 A neighbour drags the first centroid to ~0.54, which
        brings the pair inside the radius -- a collision that only exists after
        the weighting.
        """
        pos = torch.tensor(
            [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [10.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            device=device,
        )
        conf = torch.tensor([0.9, 0.8, 0.7, 0.5], device=device)
        out_pos, out_conf = cluster_waters_vdw(pos, conf, radius=1.52)
        assert out_pos.size(0) == 2
        assert out_conf.tolist() == pytest.approx([0.9, 0.7], abs=1e-6)
        assert out_pos[0, 0].item() == pytest.approx(0.75 / 1.4, abs=1e-5)
        assert out_pos[1, 0].item() == pytest.approx(10.0, abs=1e-6)

    def test_cluster_confidence_is_the_max_not_the_mean(self, device):
        pos = torch.zeros(3, 3, device=device)
        conf = torch.tensor([0.9, 0.5, 0.1], device=device)
        _, out_conf = cluster_waters_vdw(pos, conf, radius=1.52)
        assert out_conf.shape == (1,)
        assert out_conf[0].item() == pytest.approx(0.9, abs=1e-6)

    def test_all_zero_confidence_falls_back_to_unweighted_mean(self, device):
        """Weights summing to zero must yield a position, not NaN."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device)
        conf = torch.zeros(2, device=device)
        out_pos, out_conf = cluster_waters_vdw(pos, conf, radius=1.52)
        assert out_pos.size(0) == 1
        assert torch.isfinite(out_pos).all()
        assert out_pos[0, 0].item() == pytest.approx(0.5, abs=1e-6)
        assert out_conf[0].item() == pytest.approx(0.0, abs=1e-6)

    def test_empty_input(self, device):
        pos = torch.empty(0, 3, device=device)
        conf = torch.empty(0, device=device)
        out_pos, out_conf = cluster_waters_vdw(pos, conf, radius=1.52)
        assert out_pos.shape == (0, 3)
        assert out_conf.shape == (0,)

    def test_threshold_drops_everything(self, device):
        pos = torch.randn(5, 3, device=device)
        conf = torch.zeros(5, device=device)
        out_pos, out_conf = cluster_waters_vdw(pos, conf, radius=1.52, threshold=0.5)
        assert out_pos.shape == (0, 3)
        assert out_conf.shape == (0,)

    def test_shape_validation(self, device):
        pos = torch.randn(5, 3, device=device)
        conf = torch.randn(4, device=device)  # wrong size
        with pytest.raises(ValueError):
            cluster_waters_vdw(pos, conf, radius=1.52)

    def test_positions_shape_validation(self, device):
        pos = torch.randn(5, 2, device=device)
        conf = torch.randn(5, device=device)
        with pytest.raises(ValueError, match=r"positions must be \(N, 3\)"):
            cluster_waters_vdw(pos, conf, radius=1.52)


# ============== ConfidenceGVP ==============


def _make_hetero(device, n_prot=10, n_wat=5, cached_pw=False):
    """
    A minimal single-graph HeteroData with cached PP edges.

    Args:
        device: Device to build on.
        n_prot: Number of protein atoms.
        n_wat: Number of candidate waters.
        cached_pw: Also attach a PW edge_index, standing in for a dataset that
            supplies pre-built protein->water edges.
    """
    data = HeteroData()
    data["protein"].pos = torch.randn(n_prot, 3, device=device)
    idx = torch.randint(0, 16, (n_prot,), device=device)
    data["protein"].x = F.one_hot(idx, num_classes=16).float()
    data["protein"].batch = torch.zeros(n_prot, dtype=torch.long, device=device)

    data["water"].pos = torch.randn(n_wat, 3, device=device)
    wat_idx = torch.full((n_wat,), 2, dtype=torch.long, device=device)
    data["water"].x = F.one_hot(wat_idx, num_classes=16).float()
    data["water"].batch = torch.zeros(n_wat, dtype=torch.long, device=device)

    pp_edge_index = torch.tensor(
        [[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long, device=device
    )
    edge_unit_vectors, edge_rbf = compute_edge_features(
        data["protein"].pos, pp_edge_index, num_gaussians=NUM_RBF, cutoff=8.0
    )
    data[EDGE_PP].edge_index = pp_edge_index
    data[EDGE_PP].edge_unit_vectors = edge_unit_vectors
    data[EDGE_PP].edge_rbf = edge_rbf

    if cached_pw:
        # Every candidate takes protein atom 0 as its sole source.
        data[EDGE_PW].edge_index = torch.stack(
            [
                torch.zeros(n_wat, dtype=torch.long, device=device),
                torch.arange(n_wat, dtype=torch.long, device=device),
            ]
        )
    return data


@pytest.mark.unit
class TestConfidenceGVP:
    def test_auto_edge_policy_is_rejected(self, gvp_encoder):
        # "auto" needs a sampling strategy the confidence model lacks; it must
        # raise rather than silently resolve to "radius".
        with pytest.raises(ValueError, match="auto"):
            ConfidenceGVP(encoder=gvp_encoder, dynamic_edge_policy="auto")

    def test_forward_output_shape(self, device, gvp_encoder):
        data = _make_hetero(device, n_prot=10, n_wat=5)
        model = ConfidenceGVP(
            encoder=gvp_encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        scores = model(data)
        assert scores.shape == (5,)
        assert (scores >= 0).all()
        assert (scores <= 1.0).all()

    def test_forward_no_water(self, device, gvp_encoder):
        data = HeteroData()
        data["protein"].pos = torch.randn(10, 3, device=device)
        data["protein"].x = torch.randn(10, 16, device=device)
        data["protein"].batch = torch.zeros(10, dtype=torch.long, device=device)
        data[EDGE_PP].edge_index = torch.tensor(
            [[0, 1], [1, 2]], dtype=torch.long, device=device
        )

        model = ConfidenceGVP(
            encoder=gvp_encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)
        scores = model(data)
        assert scores.shape == (0,)

    def test_return_logits(self, device, gvp_encoder):
        data = _make_hetero(device, n_prot=10, n_wat=5)
        model = ConfidenceGVP(
            encoder=gvp_encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)
        logits = model(data, return_logits=True)
        assert logits.shape == (5,)

        # Dropout makes two training-mode passes disagree; compare under eval.
        model.eval()
        with torch.no_grad():
            logits2 = model(data, return_logits=True)
            probs2 = model(data, return_logits=False)
        assert torch.allclose(probs2, torch.sigmoid(logits2), atol=1e-6)

    def test_no_time_conditioning_in_scalar_encoders(self, device, gvp_encoder):
        """
        The scalar encoders take the bare hidden width, not width+1.

        FlowWaterGVP appends a time channel; a confidence model that inherited
        it would silently consume a feature slot that is never populated.
        """
        model = ConfidenceGVP(
            encoder=gvp_encoder, hidden_dims=(64, 8), layers=1, water_input_dim=16
        ).to(device)
        assert model.protein_scalar_encoder[0].in_features == 64
        assert model.water_scalar_encoder[0].in_features == 16

    def test_only_pw_and_pp_edge_types_are_active(self, device, gvp_encoder):
        """Candidates are not refined here, so WW and WP carry nothing."""
        model = ConfidenceGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(
            device
        )
        assert set(model.updater.etypes) == {EDGE_PW, EDGE_PP}

    def test_gradients_reach_the_encoder(self, device, gvp_encoder):
        """Gradient must reach the warm-started backbone, not just the head."""
        data = _make_hetero(device, n_prot=10, n_wat=5)
        model = ConfidenceGVP(
            encoder=gvp_encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        scores = model(data)
        F.mse_loss(scores, torch.rand_like(scores)).backward()

        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.encoder.parameters()
        )

    def test_score_head_receives_gradient(self, device, gvp_encoder):
        """A head that never learns would leave the backbone doing all the work."""
        data = _make_hetero(device, n_prot=10, n_wat=5)
        model = ConfidenceGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(
            device
        )
        F.mse_loss(model(data), torch.rand(5, device=device)).backward()
        assert any(
            p.grad is not None and p.grad.abs().sum().item() > 0
            for p in model.score_head.parameters()
        )


@pytest.mark.unit
class TestConfidenceGVPCachedEdges:
    def test_cached_edges_skip_dynamic_construction(self, device, gvp_encoder):
        """
        With PP and PW both supplied, forward must build no edges at all.

        ConfidenceGVP runs on those two types only, so any call into
        `build_dynamic_edges` means a cached tensor was ignored and the graph
        was silently rebuilt.
        """
        data = _make_hetero(device, n_prot=8, n_wat=4, cached_pw=True)
        model = ConfidenceGVP(
            encoder=gvp_encoder,
            hidden_dims=(64, 8),
            layers=1,
        ).to(device)

        with patch("src.flow.build_dynamic_edges") as build_mock:
            scores = model(data)

        assert build_mock.call_count == 0, (
            f"build_dynamic_edges was called {build_mock.call_count} times -- "
            "cached PW/PP edges should make edge construction unnecessary."
        )
        assert scores.shape == (4,)
        assert (scores >= 0).all() and (scores <= 1.0).all()

    def test_cached_pw_edges_are_used_verbatim(self, device, gvp_encoder):
        data = _make_hetero(device, n_prot=8, n_wat=4, cached_pw=True)
        model = ConfidenceGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(
            device
        )
        edges = model.updater.build_edges(data)
        assert torch.equal(edges[EDGE_PW], data[EDGE_PW].edge_index)

    def test_uncached_pw_edges_are_built_by_radius(self, device, gvp_encoder):
        """Without cached PW, the radius query runs and every candidate is reached."""
        data = _make_hetero(device, n_prot=8, n_wat=4, cached_pw=False)
        model = ConfidenceGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(
            device
        )
        edges = model.updater.build_edges(data)
        assert EDGE_PW in edges
        assert set(edges[EDGE_PW][1].tolist()) == set(range(4))

    def test_candidate_beyond_the_cutoff_still_gets_edges(self, device, gvp_encoder):
        """Candidates can land in sparse regions; unreached ones would go unscored."""
        data = _make_hetero(device, n_prot=8, n_wat=4, cached_pw=False)
        data["water"].pos[3] = torch.tensor([500.0, 500.0, 500.0], device=device)
        model = ConfidenceGVP(encoder=gvp_encoder, hidden_dims=(64, 8), layers=1).to(
            device
        )
        edges = model.updater.build_edges(data)
        assert 3 in set(edges[EDGE_PW][1].tolist())
