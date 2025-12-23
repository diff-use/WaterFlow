"""
Integration-ish tests to catch NaNs/Infs during forward/training.
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



