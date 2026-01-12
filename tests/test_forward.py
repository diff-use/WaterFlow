"""
Integration-ish tests to catch NaNs/Infs during forward/training.

All test cases created with assistance from Claude Code and refined.
"""

import os
import pytest
import torch
from torch_geometric.data import HeteroData

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import ProteinGVPEncoder
from src.flow import FlowWaterGVP, FlowMatcher, make_protein_encoder_data


def _iter_tensors(obj):
    """Yield tensors from nested structures (tuple/list/dict)."""
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_tensors(x)


def _tensor_stats(t: torch.Tensor) -> str:
    nan = torch.isnan(t).sum().item()
    inf = torch.isinf(t).sum().item()
    finite_mask = torch.isfinite(t)
    if finite_mask.any():
        tf = t[finite_mask]
        return (
            f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"nan={nan} inf={inf} "
            f"finite_min={tf.min().item():.3e} finite_max={tf.max().item():.3e} finite_mean={tf.mean().item():.3e}"
        )
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
        f"nan={nan} inf={inf} (no finite values)"
    )


class FiniteHookManager:
    """
    Registers forward hooks on modules.
    Raises AssertionError immediately when a module outputs NaN/Inf.
    """

    def __init__(self):
        self._handles = []

    def watch(self, module: torch.nn.Module, name: str):
        def hook(_mod, _inp, out):
            for t in _iter_tensors(out):
                if not torch.is_tensor(t) or t.numel() == 0:
                    continue
                if not torch.isfinite(t).all():
                    raise AssertionError(
                        f"[NaN/Inf detected] module={name} | {_tensor_stats(t)}"
                    )

        self._handles.append(module.register_forward_hook(hook))

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False  # don't suppress exceptions


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batched_hetero(
    device,
    n_graphs=2,
    n_protein_per=24,
    n_water_per=12,
    duplicate_protein_coords=False,
):
    """
    Build a batched HeteroData with protein+water and simple pp edges.
    Keeps sizes >= default k_pw/k_ww to avoid knn(k>n) issues.
    """
    data = HeteroData()

    Np = n_graphs * n_protein_per
    Nw = n_graphs * n_water_per

    # protein nodes
    data["protein"].pos = torch.randn(Np, 3, device=device)
    data["protein"].x = torch.randn(Np, 16, device=device)
    data["protein"].batch = torch.repeat_interleave(
        torch.arange(n_graphs, device=device), n_protein_per
    )

    # water nodes
    data["water"].pos = torch.randn(Nw, 3, device=device)
    data["water"].x = torch.randn(Nw, 16, device=device)
    data["water"].batch = torch.repeat_interleave(
        torch.arange(n_graphs, device=device), n_water_per
    )

    # optional: force a zero-distance pair (within graph 0)
    if duplicate_protein_coords and Np >= 2:
        data["protein"].pos[1] = data["protein"].pos[0]

    # simple pp edges: chain inside each graph
    src_list = []
    dst_list = []
    for g in range(n_graphs):
        base = g * n_protein_per
        for i in range(n_protein_per - 1):
            src_list.append(base + i)
            dst_list.append(base + i + 1)
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)

    # add reverse to ensure nontrivial distances in both directions
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    data["protein", "pp", "protein"].edge_index = edge_index

    return data


def assert_edge_index_in_range(edge_index: torch.Tensor, n_src: int, n_dst: int, name: str):
    if edge_index.numel() == 0:
        return
    smax = int(edge_index[0].max().item())
    dmax = int(edge_index[1].max().item())
    assert smax < n_src, f"{name}: src index out of range (max={smax}, n_src={n_src})"
    assert dmax < n_dst, f"{name}: dst index out of range (max={dmax}, n_dst={n_dst})"

def test_rbf_zero_is_finite(device):
    from src.utils import rbf
    r = torch.zeros(128, device=device)
    out = rbf(r, num_gaussians=16, cutoff=8.0)
    assert torch.isfinite(out).all()

@pytest.mark.slow
def test_forward_pass_no_nan_with_module_hooks(device):
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    stress = os.getenv("FLOW_NAN_STRESS", "0") == "1"
    n_graphs = 2 if not stress else 4
    n_protein_per = 24 if not stress else 48
    n_water_per = 12 if not stress else 24

    data = make_batched_hetero(
        device,
        n_graphs=n_graphs,
        n_protein_per=n_protein_per,
        n_water_per=n_water_per,
        duplicate_protein_coords=False,
    )

    encoder = ProteinGVPEncoder(
        node_scalar_in=16,
        hidden_dims=(64, 8),
        edge_scalar_in=16,
        n_layers=2,
        pool_residue=False,
        num_edge_rbf=16,
        radius=8.0,
    ).to(device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(64, 8),
        layers=2,
        k_pw=8,   # keep <= n_water_per
        k_ww=8,   # keep <= n_water_per
        freeze_encoder=False,
    ).to(device)

    # Quick pre-check: protein encoder input features created from pp edges
    enc_data = make_protein_encoder_data(data, num_rbf=16, rbf_dmax=20.0)
    assert torch.isfinite(enc_data.edge_rbf).all(), "enc_data.edge_rbf has NaN/Inf"
    assert torch.isfinite(enc_data.edge_unit_vec).all(), "enc_data.edge_unit_vec has NaN/Inf"
    assert_edge_index_in_range(enc_data.edge_index, enc_data.x.size(0), enc_data.x.size(0), "pp edge_index")

    # Also validate knn edges are sane (catches orientation / k issues)
    edge_dict = model.updater.build_edges(data, k_pw=model.k_pw, k_ww=model.k_ww, include_pp=False)
    assert_edge_index_in_range(edge_dict[("protein", "pw", "water")], data["protein"].pos.size(0), data["water"].pos.size(0), "pw edge_index")
    assert_edge_index_in_range(edge_dict[("water", "ww", "water")], data["water"].pos.size(0), data["water"].pos.size(0), "ww edge_index")

    t = torch.linspace(0.05, 0.95, steps=n_graphs, device=device)

    with FiniteHookManager() as hm:
        # Encoder internals
        hm.watch(model.encoder.input_scalar_encoder, "encoder.input_scalar_encoder")
        hm.watch(model.encoder.input_gvp, "encoder.input_gvp")
        for i, layer in enumerate(model.encoder.layers):
            hm.watch(layer, f"encoder.layers[{i}]")
        hm.watch(model.encoder.edge_update, "encoder.edge_update")

        # Flow parts
        hm.watch(model.encoder_to_flow, "flow.encoder_to_flow")
        hm.watch(model.protein_scalar_encoder, "flow.protein_scalar_encoder")
        hm.watch(model.water_scalar_encoder, "flow.water_scalar_encoder")
        for i, blk in enumerate(model.updater.blocks):
            hm.watch(blk, f"flow.updater.blocks[{i}]")
        hm.watch(model.vfield_head, "flow.vfield_head")

        v_pred = model(data, t)

    assert v_pred.shape == (data["water"].pos.size(0), 3)
    assert torch.isfinite(v_pred).all(), "v_pred has NaN/Inf"


@pytest.mark.slow
def test_training_step_no_nan_tripwire(device):
    torch.manual_seed(1)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1)

    data = make_batched_hetero(
        device,
        n_graphs=2,
        n_protein_per=32,
        n_water_per=16,
        duplicate_protein_coords=False,
    )

    encoder = ProteinGVPEncoder(
        node_scalar_in=16,
        hidden_dims=(64, 8),
        edge_scalar_in=16,
        n_layers=2,
        pool_residue=False,
        num_edge_rbf=16,
        radius=8.0,
    ).to(device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(64, 8),
        layers=2,
        k_pw=8,
        k_ww=8,
        freeze_encoder=False,
    ).to(device)

    fm = FlowMatcher(
        model=model,
        p_self_cond=0.0,         # simpler/cleaner for tripwire
        use_distortion=False,
        loss_eps=1e-3,
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # Hook only a few big modules to keep overhead reasonable
    with FiniteHookManager() as hm:
        hm.watch(model.encoder, "encoder (top)")
        hm.watch(model.updater, "updater (top)")
        hm.watch(model.vfield_head, "vfield_head")

        for step in range(5):
            out = fm.training_step(data, opt, grad_clip=1.0, use_self_conditioning=False)
            loss = out["loss"]
            assert isinstance(loss, float)
            assert loss == loss, "loss is NaN"

            # ensure params stayed finite
            for name, p in model.named_parameters():
                if p.requires_grad:
                    assert torch.isfinite(p).all(), f"param {name} has NaN/Inf"


@pytest.mark.slow
def test_forward_with_duplicate_protein_coords_catches_nan(device):
    """
    Stress case: force a zero-distance pair in protein.pos and ensure we still
    produce finite outputs. If you have an RBF-at-0 issue, this often trips it.
    """
    torch.manual_seed(2)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2)

    data = make_batched_hetero(
        device,
        n_graphs=1,
        n_protein_per=32,
        n_water_per=16,
        duplicate_protein_coords=True,
    )

    encoder = ProteinGVPEncoder(
        node_scalar_in=16,
        hidden_dims=(64, 8),
        edge_scalar_in=16,
        n_layers=2,
        pool_residue=False,
        num_edge_rbf=16,
        radius=8.0,
    ).to(device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(64, 8),
        layers=2,
        k_pw=8,
        k_ww=8,
        freeze_encoder=False,
    ).to(device)

    t = torch.tensor([0.5], device=device)

    with FiniteHookManager() as hm:
        hm.watch(model.encoder, "encoder (top)")
        hm.watch(model.updater, "updater (top)")
        hm.watch(model.vfield_head, "vfield_head")

        v_pred = model(data, t)

    assert torch.isfinite(v_pred).all(), "v_pred has NaN/Inf under duplicate coords"

@pytest.mark.slow
def test_forward_with_duplicate_protein_coords_localizes_nan(device):
    torch.manual_seed(2)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(2)

    data = make_batched_hetero(
        device,
        n_graphs=1,
        n_protein_per=32,
        n_water_per=16,
        duplicate_protein_coords=True,   # force a zero-distance pair
    )

    encoder = ProteinGVPEncoder(
        node_scalar_in=16,
        hidden_dims=(64, 8),
        edge_scalar_in=16,
        n_layers=2,
        pool_residue=False,
        num_edge_rbf=16,
        radius=8.0,
    ).to(device)

    model = FlowWaterGVP(
        encoder=encoder,
        hidden_dims=(64, 8),
        layers=2,
        k_pw=8,
        k_ww=8,
        freeze_encoder=False,
    ).to(device)

    # ---- Pre-check: encoder input edges/RBFs ----
    enc_data = make_protein_encoder_data(data, num_rbf=16, rbf_dmax=20.0)
    assert torch.isfinite(enc_data.edge_rbf).all(), "enc_data.edge_rbf has NaN/Inf"
    assert torch.isfinite(enc_data.edge_unit_vec).all(), "enc_data.edge_unit_vec has NaN/Inf"

    t = torch.tensor([0.5], device=device)

    with FiniteHookManager() as hm:
        # encoder internals
        hm.watch(model.encoder.input_scalar_encoder, "encoder.input_scalar_encoder")
        hm.watch(model.encoder.input_gvp, "encoder.input_gvp")
        for i, layer in enumerate(model.encoder.layers):
            hm.watch(layer, f"encoder.layers[{i}]")
        hm.watch(model.encoder.edge_update, "encoder.edge_update")

        # flow top-level stages
        hm.watch(model.encoder_to_flow, "flow.encoder_to_flow")
        for i, blk in enumerate(model.updater.blocks):
            hm.watch(blk, f"flow.updater.blocks[{i}]")
        hm.watch(model.vfield_head, "flow.vfield_head")

        v_pred = model(data, t)

    assert torch.isfinite(v_pred).all()


# ============== Tests for Flow Matching Fundamentals ==============

@pytest.mark.unit
class TestHungarianMatchingConsistency:
    """Tests for Hungarian matching correctness."""

    def test_hungarian_is_deterministic(self, device):
        """Hungarian matching should be deterministic for same x0, x1."""
        from src.utils import ot_coupling

        x1 = torch.randn(10, 3, device=device)
        x0 = torch.randn(10, 3, device=device)
        batch = torch.zeros(10, dtype=torch.long, device=device)

        # Run matching twice
        x0_star_1, x1_star_1 = ot_coupling(x1, batch, x0)
        x0_star_2, x1_star_2 = ot_coupling(x1, batch, x0)

        # Should be identical
        assert torch.allclose(x0_star_1, x0_star_2, atol=1e-6), "Hungarian matching is non-deterministic!"
        assert torch.allclose(x1_star_1, x1_star_2, atol=1e-6), "Hungarian matching is non-deterministic!"

    def test_hungarian_is_permutation(self, device):
        """Hungarian matching is a permutation (reordering)."""
        from src.utils import ot_coupling

        x1 = torch.randn(10, 3, device=device)
        x0 = torch.randn(10, 3, device=device)
        batch = torch.zeros(10, dtype=torch.long, device=device)

        x0_star, x1_star = ot_coupling(x1, batch, x0)

        # x1_star should be a permutation of x1 (same set of points)
        for i in range(len(x1_star)):
            dists = torch.norm(x1 - x1_star[i], dim=-1)
            min_dist = dists.min().item()
            assert min_dist < 1e-5, f"x1_star[{i}] not found in x1 (min_dist={min_dist})"

    def test_hungarian_batched_no_cross_contamination(self, device):
        """Hungarian matching doesn't cross batch boundaries."""
        from src.utils import ot_coupling

        # Two separate graphs
        x1 = torch.cat([
            torch.randn(5, 3, device=device),
            torch.randn(7, 3, device=device) + 100.0  # Far away
        ])
        x0 = torch.cat([
            torch.randn(5, 3, device=device),
            torch.randn(7, 3, device=device) + 100.0
        ])
        batch = torch.cat([
            torch.zeros(5, dtype=torch.long),
            torch.ones(7, dtype=torch.long)
        ]).to(device)

        x0_star, x1_star = ot_coupling(x1, batch, x0)

        # Check graph 1 points don't match to graph 2
        x1_star_g1 = x1_star[:5]
        x1_g2 = x1[5:]

        for i in range(5):
            # Distance to any graph 2 point should be large
            min_dist_to_g2 = torch.norm(x1_g2 - x1_star_g1[i], dim=-1).min().item()
            assert min_dist_to_g2 > 50.0, "Hungarian crossed batch boundary!"


@pytest.mark.unit
class TestNoiseSamplingScale:
    """Test that noise scale matches protein coordinate std."""

    def test_compute_sigma_matches_protein_std(self, device):
        """Sigma should equal std of protein coordinates."""
        data = make_batched_hetero(device, n_graphs=1, n_protein_per=100, n_water_per=20)

        sigma = FlowMatcher.compute_sigma(data)
        expected_sigma = data['protein'].pos.std().item()

        assert abs(sigma - expected_sigma) < 1e-5, \
            f"Sigma {sigma:.6f} doesn't match protein std {expected_sigma:.6f}"

    def test_sigma_consistent_across_calls(self, device):
        """compute_sigma should be deterministic."""
        data = make_batched_hetero(device, n_graphs=1, n_protein_per=50, n_water_per=10)

        sigma1 = FlowMatcher.compute_sigma(data)
        sigma2 = FlowMatcher.compute_sigma(data)

        assert sigma1 == sigma2, "compute_sigma is non-deterministic!"

    def test_noise_scale_reasonable(self, device):
        """Initial noise x0 should have similar scale to protein."""
        data = make_batched_hetero(device, n_graphs=1, n_protein_per=50, n_water_per=20)

        sigma = FlowMatcher.compute_sigma(data)
        n_water = data['water'].num_nodes
        x0 = torch.randn(n_water, 3, device=device) * sigma

        x0_std = x0.std().item()

        # Allow some variation due to random sampling
        assert abs(x0_std - sigma) / sigma < 0.3, \
            f"x0 std {x0_std:.3f} too different from sigma {sigma:.3f}"


@pytest.mark.unit
class TestFlowIntegrationCorrectness:
    """Test that flow integration behaves correctly."""

    def test_integration_trajectory_length(self, device):
        """Integration trajectory should have num_steps entries."""
        data = make_batched_hetero(device, n_graphs=1, n_protein_per=24, n_water_per=12)

        encoder = ProteinGVPEncoder(
            node_scalar_in=16, hidden_dims=(64, 8), edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        model = FlowWaterGVP(
            encoder=encoder, hidden_dims=(64, 8), layers=1,
            k_pw=8, k_ww=8,
        ).to(device)

        fm = FlowMatcher(model, p_self_cond=0.0)

        num_steps = 20
        result = fm.rk4_integrate(
            data, num_steps=num_steps, use_sc=False,
            device=str(device), return_trajectory=True
        )

        assert len(result['trajectory']) == num_steps, \
            f"Expected {num_steps} trajectory steps, got {len(result['trajectory'])}"
        assert len(result['rmsd']) == num_steps, \
            f"Expected {num_steps} RMSD values, got {len(result['rmsd'])}"

    def test_interpolation_at_boundaries(self, device):
        """Interpolation x_t = (1-t)*x0 + t*x1 gives correct values at boundaries."""
        from src.utils import ot_coupling

        x1 = torch.randn(10, 3, device=device)
        batch = torch.zeros(10, dtype=torch.long, device=device)

        x0 = torch.randn(10, 3, device=device)
        x0_star, x1_star = ot_coupling(x1, batch, x0)

        # At t=0: x_t should equal x0_star
        t0 = torch.zeros(1, device=device)
        t_per_atom_0 = t0[batch].unsqueeze(-1)
        x_t_0 = (1.0 - t_per_atom_0) * x0_star + t_per_atom_0 * x1_star

        assert torch.allclose(x_t_0, x0_star, atol=1e-6), \
            "Interpolation at t=0 doesn't match x0"

        # At t=1: x_t should equal x1_star
        t1 = torch.ones(1, device=device)
        t_per_atom_1 = t1[batch].unsqueeze(-1)
        x_t_1 = (1.0 - t_per_atom_1) * x0_star + t_per_atom_1 * x1_star

        assert torch.allclose(x_t_1, x1_star, atol=1e-6), \
            "Interpolation at t=1 doesn't match x1"


@pytest.mark.unit
class TestVelocityFieldProperties:
    """Test velocity field sanity checks."""

    def test_velocity_field_finite(self, device):
        """Velocity predictions should be finite (no NaN or Inf)."""
        data = make_batched_hetero(device, n_graphs=1, n_protein_per=24, n_water_per=12)

        encoder = ProteinGVPEncoder(
            node_scalar_in=16, hidden_dims=(64, 8), edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        model = FlowWaterGVP(
            encoder=encoder, hidden_dims=(64, 8), layers=1,
            k_pw=8, k_ww=8,
        ).to(device)

        model.eval()

        # Test at multiple t values
        for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            t = torch.tensor([t_val], device=device)
            v_pred = model(data, t, sc=None)

            assert torch.isfinite(v_pred).all(), \
                f"Velocity has NaN/Inf at t={t_val}"

            # Check magnitude is not absurdly large
            max_mag = torch.norm(v_pred, dim=-1).max().item()
            assert max_mag < 1e6, \
                f"Velocity magnitude too large at t={t_val}: {max_mag:.3e}"

    def test_velocity_field_changes_with_t(self, device):
        """Velocity field should depend on t (different outputs for different times)."""
        data = make_batched_hetero(device, n_graphs=1, n_protein_per=24, n_water_per=12)

        encoder = ProteinGVPEncoder(
            node_scalar_in=16, hidden_dims=(64, 8), edge_scalar_in=16,
            pool_residue=False,
        ).to(device)

        model = FlowWaterGVP(
            encoder=encoder, hidden_dims=(64, 8), layers=1,
            k_pw=8, k_ww=8,
        ).to(device)

        model.eval()

        t0 = torch.tensor([0.1], device=device)
        t1 = torch.tensor([0.9], device=device)

        v0 = model(data, t0, sc=None)
        v1 = model(data, t1, sc=None)

        # Velocities should be different
        diff = torch.norm(v0 - v1, dim=-1).mean().item()

        assert diff > 1e-4, \
            f"Velocity field doesn't change with t (diff={diff:.6f})"


@pytest.mark.unit
class TestTargetVelocityScale:
    """Test that target velocity has expected scale."""

    def test_velocity_target_scale(self, device):
        """
        Check that target velocity v_target = x1 - x0 has expected scale.

        For random noise x0 ~ N(0, sigma²) and target x1, we expect:
        ||x1 - x0|| to be on order of sigma
        """
        from src.utils import ot_coupling

        data = make_batched_hetero(device, n_graphs=1, n_protein_per=50, n_water_per=20)

        x1 = data['water'].pos
        batch = data['water'].batch

        sigma = FlowMatcher.compute_sigma(data)
        x0 = torch.randn_like(x1) * sigma

        x0_star, x1_star = ot_coupling(x1, batch, x0)
        v_target = x1_star - x0_star

        # Average magnitude
        target_mag = torch.norm(v_target, dim=-1).mean().item()

        # Should be on order of sigma (could be sigma to 3*sigma depending on x1 spread)
        assert 0.5 * sigma < target_mag < 5 * sigma, \
            f"Target velocity magnitude {target_mag:.3f} seems off (sigma={sigma:.3f})"


