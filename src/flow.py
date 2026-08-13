"""
Flow matching model components for water placement prediction.

This module provides:
- ProteinWaterUpdate: Heterogeneous GVP message passing across 4 edge types
- FlowWaterGVP: End-to-end flow model combining encoder + GVP updates + vector field head
- FlowMatcher: High-level training, validation, and numerical integration interface
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn, Tensor
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import knn
from torch_scatter import scatter, scatter_mean
from tqdm.auto import tqdm

from src.constants import (
    ALL_EDGE_TYPES,
    EDGE_PP,
    EDGE_PW,
    EDGE_WP,
    EDGE_WW,
    ELEM_IDX,
    ELEMENT_VOCAB,
    NUM_RBF,
)
from src.encoder_base import BaseProteinEncoder
from src.gvp import GVP, GVPMultiEdgeConv
from src.utils import ot_coupling


def _batch_from_counts(num_waters: Tensor, device: torch.device) -> Tensor:
    """
    Build a graph-grouped batch vector from per-graph counts.

    Args:
        num_waters: (num_graphs,) water count per graph
        device: Output device

    Returns:
        (sum(num_waters),) graph index per water, non-decreasing
    """
    return torch.repeat_interleave(
        torch.arange(num_waters.numel(), device=device), num_waters.to(device)
    )


def sample_waters_uniform_ball(
    protein_pos: Tensor,
    batch_p: Tensor,
    batch_w: Tensor,
    cutoff: float = 8.0,
    device: torch.device | None = None,
    anchor_mask: Tensor | None = None,
) -> Tensor:
    """
    Sample water positions uniformly inside balls of radius *cutoff* centred
    on randomly chosen protein atoms.

    Every sample is guaranteed within *cutoff* of at least one protein atom.
    No rejection sampling — runs in O(1) rounds, fully vectorised.

    Args:
        protein_pos: (N_protein, 3) protein coordinates for all graphs
        batch_p: (N_protein,) graph indices for protein atoms
        batch_w: (N_water,) graph index per water to sample. Positions are returned
            in this order, so callers with existing water nodes can pass their own
            batch vector and get samples aligned to it.
        cutoff: Ball radius in Angstroms
        device: Optional output device (defaults to protein_pos.device)
        anchor_mask: Optional (N_protein,) bool selecting eligible anchors. Used to
            anchor on ASU atoms only, so the prior spawns where the targets live
            instead of dispersing onto symmetry mates that OT must then transport
            back. Every structure keeps >=1 ASU atom, so the mask never starves a
            graph; if it somehow does, the batch anchors on all atoms and warns.

    Returns:
        water_pos: (N_water, 3) sampled positions, one per entry of batch_w
    """
    if device is None:
        device = protein_pos.device

    batch_w = batch_w.to(device)
    total_waters = batch_w.numel()

    if total_waters == 0:
        return torch.empty(0, 3, dtype=protein_pos.dtype, device=device)

    # offsets below assume protein atoms are grouped contiguously by graph;
    # interleaved batch_p would pick anchors from the wrong graph.
    batch_p = batch_p.to(device)
    if batch_p.numel() > 1 and (batch_p[1:] < batch_p[:-1]).any():
        raise ValueError("batch_p must be sorted (non-decreasing) by graph index.")

    # cover every graph named by either side so the guard below can see empty ones
    num_graphs = int(batch_w.max().item()) + 1
    if batch_p.numel() > 0:
        num_graphs = max(num_graphs, int(batch_p.max().item()) + 1)

    protein_pos = protein_pos.to(device)

    # Drop to the eligible anchors (ASU-only). Skip the mask, rather than starve a
    # graph, if it would leave a water-requesting graph with no anchor -- an
    # invariant violation the dataset should never produce, so warn if it happens.
    if anchor_mask is not None:
        eligible = anchor_mask.to(device).bool()
        counts = torch.bincount(batch_p[eligible], minlength=num_graphs)
        if (counts[batch_w] == 0).any():
            logger.warning(
                "sample_waters_uniform_ball: anchor mask leaves a water-requesting "
                "graph with no anchor; anchoring the batch on all protein atoms."
            )
        else:
            protein_pos = protein_pos[eligible]
            batch_p = batch_p[eligible]

    # per-graph protein atom counts and cumulative offsets
    num_p_per_graph = scatter(
        torch.ones(batch_p.size(0), device=device, dtype=torch.long),
        batch_p,
        dim=0,
        dim_size=num_graphs,
        reduce="sum",
    )

    # fail fast: a graph that requests waters must have at least one protein atom.
    # Otherwise graph_sizes is 0 and graph_offsets + local_idx would index into a
    # neighbouring graph's atoms (or out of bounds) when picking anchors below.
    graph_sizes = num_p_per_graph[batch_w]
    if (graph_sizes == 0).any():
        bad = batch_w[graph_sizes == 0].unique().tolist()
        raise ValueError(
            f"Cannot sample waters for graph(s) {bad}: they request waters "
            "but have zero protein atoms."
        )

    offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    offsets[1:] = num_p_per_graph.cumsum(dim=0)

    # pick a random protein atom per water (uniform with replacement)
    graph_offsets = offsets[batch_w]
    local_idx = (torch.rand(total_waters, device=device) * graph_sizes.float()).long()
    anchors = protein_pos[graph_offsets + local_idx]

    # uniform direction on the unit sphere
    direction = torch.randn(total_waters, 3, device=device, dtype=protein_pos.dtype)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # uniform radius inside the ball: r = R * U^(1/3)
    r = cutoff * torch.rand(
        total_waters, 1, device=device, dtype=protein_pos.dtype
    ).pow(1.0 / 3.0)

    return anchors + r * direction


def sample_waters_scaled_gaussian(
    batch_w: Tensor,
    sigma_per_graph: Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Sample water positions from N(0, sigma^2 * I) with no rejection.

    Args:
        batch_w: (N_water,) graph index per water to sample. Positions are returned
            in this order, so callers with existing water nodes can pass their own
            batch vector and get samples aligned to it.
        sigma_per_graph: (num_graphs,) Gaussian scale per graph
        device: Output device
        dtype: Output dtype

    Returns:
        water_pos: (N_water, 3) sampled positions, one per entry of batch_w
    """
    batch_w = batch_w.to(device)
    total_waters = batch_w.numel()

    if total_waters == 0:
        return torch.empty(0, 3, dtype=dtype, device=device)

    sigma = sigma_per_graph.to(device=device, dtype=dtype)[batch_w].unsqueeze(-1)

    return torch.randn(total_waters, 3, device=device, dtype=dtype) * sigma


def build_knn_edges(
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    k: int,
    batch_src: torch.Tensor | None = None,
    batch_dst: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build KNN edges from src -> dst (source indices in row 0, dest in row 1).

    The KNN query is performed *per destination*: for each point in ``dst_pos``
    we look up its ``k`` nearest neighbors in ``src_pos`` (``knn(x=src_pos,
    y=dst_pos, ...)``) and emit them as incoming edges. As a consequence every
    destination node is guaranteed to have up to ``k`` incoming edges (and so
    appears in row 1), whereas a source node that is no destination's nearest
    neighbor may not appear in row 0 at all. Coverage checks ("every node has an
    edge") must therefore be made against the destination row (row 1).

    For a homogeneous graph (``src_pos is dst_pos``) self-edges are dropped.

    Args:
        src_pos: (N_src, 3) source node positions.
        dst_pos: (N_dst, 3) destination node positions.
        k: Number of nearest source neighbors to find per destination node.
        batch_src: (N_src,) batch assignment for source nodes, or None.
        batch_dst: (N_dst,) batch assignment for destination nodes, or None.

    Returns:
        (2, E) edge index tensor with source indices in row 0, destination in
        row 1.
    """
    if src_pos.numel() == 0 or dst_pos.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=src_pos.device)

    # knn(x, y) returns row 0 = y (query), row 1 = x (neighbor); swap for this
    # repo's src(row 0)->dst(row 1) convention. That row order is undocumented,
    # so it is pinned by tests/test_flow.py::TestBuildKnnEdgesDirection.
    idx = knn(x=src_pos, y=dst_pos, k=k, batch_x=batch_src, batch_y=batch_dst)
    idx = torch.stack((idx[1], idx[0]), dim=0)

    # remove self-edges if homogeneous
    if src_pos.data_ptr() == dst_pos.data_ptr():
        mask = idx[0] != idx[1]
        idx = idx[:, mask]

    return idx.unique(dim=1)


class ProteinWaterUpdate(nn.Module):
    """
    Heterogeneous GVP message passing with all four edge types:
      - protein -> water  (pw)
      - water   -> water  (ww)
      - protein -> protein (pp)
      - water   -> protein (wp)
    """

    def __init__(
        self,
        hidden_dims=(512, 64),
        rbf_dim=16,
        layers=3,
        drop_rate=0.0,
        n_message_gvps=2,
        n_update_gvps=2,
        vector_gate=True,
        aggr_edges="sum",
        use_dst_feats=True,
    ):
        """
        Initialize heterogeneous protein-water message passing module.

        Args:
            hidden_dims: (scalar_dim, vector_dim) hidden dimensions for GVP layers
            rbf_dim: Number of radial basis functions for distance encoding
            layers: Number of GVP message passing layers
            drop_rate: Dropout rate for regularization
            n_message_gvps: Number of GVP modules in each edge-type's message function
                (distinct from `layers` which controls message-passing iterations)
            n_update_gvps: Number of GVP modules in the node update function
                (applied after aggregating messages from all edge types)
            vector_gate: Whether to use vector gating in GVP layers
            aggr_edges: Edge aggregation method ('sum' or 'mean')
            use_dst_feats: Whether to include destination features in messages
        """
        super().__init__()
        # Unpack hidden dimensions: s_h = scalar hidden dim, v_h = vector hidden dim
        s_h, v_h = hidden_dims

        etypes = ALL_EDGE_TYPES

        self.blocks = nn.ModuleList(
            [
                GVPMultiEdgeConv(
                    etypes=etypes,
                    s_dim=s_h,
                    v_dim=v_h,
                    rbf_dim=rbf_dim,
                    n_message_gvps=n_message_gvps,
                    n_update_gvps=n_update_gvps,
                    use_dst_feats=use_dst_feats,
                    drop_rate=drop_rate,
                    aggr_edges=aggr_edges,
                    activations=(F.relu, torch.sigmoid),
                    vector_gate=vector_gate,
                )
                for _ in range(layers)
            ]
        )
        self.etypes = etypes

    def build_edges(
        self, data: HeteroData, k_pw: int = 12, k_ww: int = 8, k_wp: int = 8
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """
        Build KNN edges for protein-water interactions.

        For protein->water edges, we take the union of:
        - KNN(protein -> water): k nearest waters per protein
        - KNN(water -> protein) reversed: k nearest proteins per water
        This ensures every water has at least k_pw protein neighbors.

        PP edges are read from the dataset (cached at preprocessing time).

        Args:
            data: HeteroData with 'protein' and 'water' node types containing positions
            k_pw: Number of nearest neighbors for protein-water edges
            k_ww: Number of nearest neighbors for water-water edges
            k_wp: Number of nearest neighbors for water-protein edges

        Returns:
            Dict mapping edge type tuples to (2, E) edge index tensors
        """
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        device = data["protein"].pos.device

        batch_p = data["protein"].batch if "batch" in data["protein"] else None
        batch_w = data["water"].batch if "batch" in data["water"] else None

        pos_p = data["protein"].pos
        pos_w = data["water"].pos

        # protein -> water
        if pos_p.numel() > 0 and pos_w.numel() > 0:
            # p->w
            ei_pw = build_knn_edges(
                pos_p, pos_w, k=k_pw, batch_src=batch_p, batch_dst=batch_w
            )
            # w->p then reverse
            ei_wp = build_knn_edges(
                pos_w, pos_p, k=k_pw, batch_src=batch_w, batch_dst=batch_p
            )
            ei_wp_reversed = ei_wp.flip(0)
            # union
            ei_pw_union = torch.cat([ei_pw, ei_wp_reversed], dim=1).unique(dim=1)
            edge_index_dict[EDGE_PW] = ei_pw_union
        else:
            edge_index_dict[EDGE_PW] = torch.empty(
                2, 0, dtype=torch.long, device=device
            )

        # water -> water
        if pos_w.numel() > 0:
            edge_index_dict[EDGE_WW] = build_knn_edges(
                pos_w, pos_w, k=k_ww, batch_src=batch_w, batch_dst=batch_w
            )
        else:
            edge_index_dict[EDGE_WW] = torch.empty(
                2, 0, dtype=torch.long, device=device
            )

        # protein-protein edges (cached from dataset)
        if EDGE_PP in data.edge_types:
            edge_index_dict[EDGE_PP] = data[EDGE_PP].edge_index
        else:
            edge_index_dict[EDGE_PP] = build_knn_edges(
                pos_p, pos_p, k=k_pw, batch_src=batch_p, batch_dst=batch_p
            )

        # water -> protein
        if pos_w.numel() > 0 and pos_p.numel() > 0:
            edge_index_dict[EDGE_WP] = build_knn_edges(
                pos_w, pos_p, k=k_wp, batch_src=batch_w, batch_dst=batch_p
            )
        else:
            edge_index_dict[EDGE_WP] = torch.empty(
                2, 0, dtype=torch.long, device=device
            )

        for et in self.etypes:
            if et not in edge_index_dict:
                edge_index_dict[et] = torch.empty(2, 0, dtype=torch.long, device=device)

        return edge_index_dict

    def forward(
        self,
        x_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        data: HeteroData,
        k_pw: int = 12,
        k_ww: int = 8,
        k_wp: int = 8,
        pp_edge_attr: tuple | None = None,
    ):
        """
        Run heterogeneous message passing across protein and water nodes.

        Args:
            x_dict: Node features dict with:
                - 'protein': (s_p, v_p) where s_p is (N_p, scalar_dim), v_p is (N_p, vector_dim, 3)
                - 'water': (s_w, v_w) where s_w is (N_w, scalar_dim), v_w is (N_w, vector_dim, 3)
            data: HeteroData with 'protein' and 'water' node positions
            k_pw: Number of nearest neighbors for protein-water edges
            k_ww: Number of nearest neighbors for water-water edges
            k_wp: Number of nearest neighbors for water-protein edges
            pp_edge_attr: Optional encoder-learned edge features (s_edge, V_edge) for PP edges.
                If provided, uses encoder-learned scalar features (s_edge) combined with
                cached edge direction unit vectors (edge_unit_vectors, pre-normalized at preprocessing).
                If None, uses cached geometric edge features (edge_rbf, edge_unit_vectors) from the dataset.

        Returns:
            Updated x_dict with same structure as input
        """
        pos_dict = {nt: data[nt].pos for nt in data.node_types if "pos" in data[nt]}

        edge_index_dict = self.build_edges(
            data,
            k_pw=k_pw,
            k_ww=k_ww,
            k_wp=k_wp,
        )

        # PP edge features: encoder-provided take priority over cached geometric features
        cached_edge_attr_dict = {}
        if EDGE_PP in data.edge_types:
            pp_edge = data[EDGE_PP]

            # A given model sees one source or the other, never both.
            if pp_edge_attr is not None:
                # Use encoder-learned scalar features (s_edge) with unit vectors
                s_edge, V_edge = pp_edge_attr
                if hasattr(pp_edge, "edge_unit_vectors"):
                    cached_edge_attr_dict[EDGE_PP] = (s_edge, pp_edge.edge_unit_vectors)
                else:
                    # Graphs built outside the dataset carry vectors on the encoder side
                    cached_edge_attr_dict[EDGE_PP] = (s_edge, V_edge.squeeze(1))
            elif hasattr(pp_edge, "edge_rbf") and hasattr(pp_edge, "edge_unit_vectors"):
                # No encoder edge features (e.g., SLAE/ESM) - use cached geometric features
                cached_edge_attr_dict[EDGE_PP] = (
                    pp_edge.edge_rbf,
                    pp_edge.edge_unit_vectors,
                )

        for block in self.blocks:
            x_dict = block(x_dict, edge_index_dict, pos_dict, cached_edge_attr_dict)

        return x_dict


class FlowWaterGVP(nn.Module):
    """
    End-to-end:
      1. Encode protein (which may include mate atoms).
      2. Time-condition protein and water.
      3. Build protein->water edges.
      4. Run hetero multi-edge GVP update.
      5. Predict water vector field.
    """

    def __init__(
        self,
        encoder: BaseProteinEncoder,
        hidden_dims: tuple[int, int] = (256, 32),
        edge_scalar_dim: int = NUM_RBF,
        layers: int = 4,
        drop_rate: float = 0.1,
        n_message_gvps: int = 2,
        n_update_gvps: int = 2,
        vector_gate: bool = True,
        k_pw: int = 12,
        k_ww: int = 8,
        k_wp: int = 8,
        water_input_dim: int = 16,  # 1 hot with oxygen, same as encoder
    ):
        """
        Initialize end-to-end flow model for water placement.

        Args:
            encoder: Protein encoder implementing BaseProteinEncoder interface
            hidden_dims: (scalar_dim, vector_dim) hidden dimensions. Default: (256, 32)
            edge_scalar_dim: Dimension of edge scalar features. Default: NUM_RBF (32)
            layers: Number of heterogeneous GVP message passing layers. Default: 4
            drop_rate: Dropout rate for regularization. Default: 0.1
            n_message_gvps: Number of GVP modules in each edge-type's message function
                (distinct from `layers` which controls message-passing iterations). Default: 2
            n_update_gvps: Number of GVP modules in the node update function
                (applied after aggregating messages from all edge types). Default: 2
            vector_gate: Whether to use vector gating in GVP layers. Default: True
            k_pw: K nearest neighbors for protein-water edges. Default: 12
            k_ww: K nearest neighbors for water-water edges. Default: 8
            k_wp: K nearest neighbors for water-protein edges. Default: 8
            water_input_dim: Input dimension for water node features. Default: 16
        """
        super().__init__()
        self.encoder = encoder
        self.hidden_dims = hidden_dims
        self.edge_scalar_dim = edge_scalar_dim
        self.layers = layers
        self.drop_rate = drop_rate
        self.n_message_gvps = n_message_gvps
        self.n_update_gvps = n_update_gvps
        self.vector_gate = vector_gate
        self.k_pw = k_pw
        self.k_ww = k_ww
        self.k_wp = k_wp

        s_h, v_h = hidden_dims

        # Bridge encoder output dims -> flow dims (works for ANY encoder)
        self.encoder_to_flow = GVP(
            in_dims=encoder.output_dims,
            out_dims=hidden_dims,
            activations=(F.relu, torch.sigmoid),
            vector_gate=True,
        )

        # time-conditioning for protein
        self.protein_scalar_encoder = nn.Sequential(
            nn.Linear(s_h + 1, s_h),
            nn.GELU(),
            nn.LayerNorm(s_h),
        )

        # water scalar encoder (oxygen element one-hot etc.)
        self.water_scalar_encoder = nn.Sequential(
            nn.Linear(water_input_dim + 1, s_h),
            nn.GELU(),
            nn.LayerNorm(s_h),
        )

        # hetero updater: protein+water (always includes pp and wp edges)
        self.updater = ProteinWaterUpdate(
            hidden_dims=hidden_dims,
            rbf_dim=edge_scalar_dim,
            layers=layers,
            drop_rate=drop_rate,
            n_message_gvps=n_message_gvps,
            n_update_gvps=n_update_gvps,
            vector_gate=vector_gate,
            aggr_edges="sum",
            use_dst_feats=True,
        )

        self.sc_vec_encoder = GVP(
            in_dims=(0, 1),
            out_dims=(0, v_h),
            activations=(None, None),
            vector_gate=True,
        )

        self.sc_sca_encoder = nn.Sequential(
            nn.Linear(1, s_h),
            nn.GELU(),
            nn.LayerNorm(s_h),
        )

        # Water vector field head: project (s_h, v_h) -> (s_h // 4, 1) -> single vector channel
        # NOTE: vector_gate=True requires scalar input features. GVP gating works by
        # computing gate values from scalars via a learned linear map, then applying
        # sigmoid-gated element-wise multiplication to the output vectors.
        self.vfield_head = GVP(
            in_dims=hidden_dims,
            out_dims=(s_h // 4, 1),
            vector_gate=True,
        )

    def forward(
        self,
        data: HeteroData,
        t: torch.Tensor,
        self_cond: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Predict velocity field for water nodes given protein context and time.

        Args:
            data: HeteroData with:
                - 'protein' nodes (may include symmetry mates):
                    positions: (N_p, 3) Cartesian coordinates
                    features: (N_p, feat_dim) element one-hot or encoder embeddings
                - 'water' nodes:
                    positions: (N_w, 3) Cartesian coordinates
                    features: (N_w, 16) element one-hot encoding
            t: (B,) flow time per complex in batch, values in [0, 1]
            self_cond: Optional self-conditioning dict with 'x1_pred' key containing
                previous prediction (N_w, 3) for iterative refinement

        Returns:
            (N_w, 3) predicted velocity vector field at each water node
        """
        device = data["protein"].pos.device

        # all encoders return (s, V, pp_edge_attr) where pp_edge_attr is None for SLAE/ESM
        s_all, v_all, pp_edge_attr = self.encoder(data)

        # pass tuple when encoder has vector outputs, tensor when scalar-only
        encoder_input = (s_all, v_all) if self.encoder.output_dims[1] > 0 else s_all
        s_p_latent, v_p_latent = self.encoder_to_flow(encoder_input)

        if "water" not in data.node_types or data["water"].num_nodes == 0:
            return torch.zeros(0, 3, device=device)

        batch_p = data["protein"].batch
        batch_w = data["water"].batch

        t_p = t[batch_p].unsqueeze(-1)
        t_w = t[batch_w].unsqueeze(-1)

        s_p = self.protein_scalar_encoder(torch.cat([s_p_latent, t_p], dim=-1))
        s_w = self.water_scalar_encoder(torch.cat([data["water"].x, t_w], dim=-1))

        # initial water vectors (all zeros to start)
        v_w = torch.zeros(
            data["water"].num_nodes,
            self.hidden_dims[1],
            3,
            device=device,
        )

        # self conditioning
        if (
            self_cond is not None
            and ("x1_pred" in self_cond)
            and self_cond["x1_pred"] is not None
        ):
            delta = self_cond["x1_pred"] - data["water"].pos
            delta_vec = delta.unsqueeze(1)

            # vector conditioning (equivariant)
            s_empty = torch.empty(data["water"].num_nodes, 0, device=device)
            _, v_sc = self.sc_vec_encoder((s_empty, delta_vec))
            v_w = v_w + v_sc

            # scalar conditioning (invariant) on ||delta||
            d_mag = delta.norm(dim=-1, keepdim=True)
            s_sc = self.sc_sca_encoder(d_mag)
            s_w = s_w + s_sc

        # build hetero feature dict for GVP multi-edge updates
        x_dict = {
            "protein": (s_p, v_p_latent),
            "water": (s_w, v_w),
        }

        # hetero update (protein+water graph)
        # Pass encoder edge features (None for SLAE/ESM, tuple for GVP)
        x_dict = self.updater(
            x_dict,
            data,
            k_pw=self.k_pw,
            k_ww=self.k_ww,
            k_wp=self.k_wp,
            pp_edge_attr=pp_edge_attr,
        )

        # water vector field head
        _, v_pred = self.vfield_head(x_dict["water"])
        return v_pred.squeeze(1)


class FlowMatcher:
    """
    High level class for flow matching training, validation, and numerical integration
    """

    SAMPLING_STRATEGIES = ("uniform_ball", "scaled_gaussian")
    DYNAMIC_EDGE_POLICIES = ("auto", "radius", "knn_if_isolated")

    def __init__(
        self,
        model,
        p_self_cond: float = 0.5,
        use_distortion: bool = False,
        p_distort: float = 0.2,
        t_distort: float = 0.5,
        sigma_distort: float = 0.5,
        loss_eps: float = 1e-3,
        sampling_strategy: str = "uniform_ball",
        dynamic_edge_policy: str = "auto",
    ):
        """
        Initialize flow matcher for training and inference.

        Args:
            model: FlowWaterGVP model instance
            p_self_cond: Probability of using self-conditioning during training
            use_distortion: Whether to apply late-stage path distortion
            p_distort: Probability of applying distortion per sample
            t_distort: Time threshold after which distortion may be applied
            sigma_distort: Standard deviation of distortion noise
            loss_eps: Small constant for numerical stability in loss weighting
            sampling_strategy: Source distribution for flow matching noise.
                "uniform_ball" samples uniformly in balls around protein atoms.
                "scaled_gaussian" samples from N(0, sigma^2*I).
            dynamic_edge_policy: Runtime policy for dynamic water-edge building.
        """
        if sampling_strategy not in self.SAMPLING_STRATEGIES:
            raise ValueError(
                f"sampling_strategy must be one of {self.SAMPLING_STRATEGIES}, "
                f"got '{sampling_strategy}'"
            )
        if dynamic_edge_policy not in self.DYNAMIC_EDGE_POLICIES:
            raise ValueError(
                f"dynamic_edge_policy must be one of {self.DYNAMIC_EDGE_POLICIES}, "
                f"got '{dynamic_edge_policy}'"
            )
        self.model = model
        self.p_self_cond = p_self_cond
        self.use_distortion = use_distortion
        self.p_distort = p_distort
        self.t_distort = t_distort
        self.sigma_distort = sigma_distort
        self.loss_eps = loss_eps
        self.graph_cutoff = getattr(model, "cutoff", 8.0)
        self.sampling_strategy = sampling_strategy
        self.dynamic_edge_policy = dynamic_edge_policy

    @staticmethod
    def _num_graphs(data: HeteroData | Batch) -> int:
        """Infer graph count without forcing a device sync when batch metadata exists."""
        num_graphs = getattr(data, "num_graphs", None)
        if num_graphs is not None:
            return int(num_graphs)
        batch_p = data["protein"].batch
        if batch_p.numel() == 0:
            return 0
        return int(batch_p.max().item()) + 1

    @staticmethod
    def _asu_mask(data: HeteroData | Batch) -> Tensor | None:
        """
        Mask over protein nodes selecting ASU (non-mate) atoms, or None when the
        batch carries no mate annotation.

        No ``.any()`` short-circuit: it would force a host sync every step, and both
        consumers treat an all-True mask as a no-op.
        """
        is_mate = getattr(data["protein"], "is_mate", None)
        if is_mate is None:
            return None
        return ~is_mate.bool()

    def _sample_waters(
        self,
        batch_data: HeteroData | Batch,
        batch_w: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Dispatch to the configured sampling strategy, sampling one water per
        entry of batch_w and returning them in that order."""
        # Targets are ASU-only, so mate atoms disperse the prior (uniform_ball) and
        # inflate its scale (scaled_gaussian). No mates -> no mask -> unchanged.
        asu_mask = self._asu_mask(batch_data)

        if self.sampling_strategy == "uniform_ball":
            return sample_waters_uniform_ball(
                protein_pos=batch_data["protein"].pos,
                batch_p=batch_data["protein"].batch,
                batch_w=batch_w,
                cutoff=self.graph_cutoff,
                device=device,
                anchor_mask=asu_mask,
            )
        # scaled_gaussian
        sigma_per_graph = self.compute_sigma_per_graph(
            batch_data, device, node_mask=asu_mask
        )
        return sample_waters_scaled_gaussian(
            batch_w=batch_w,
            sigma_per_graph=sigma_per_graph,
            device=device,
            dtype=batch_data["protein"].pos.dtype,
        )

    def _effective_dynamic_edge_policy(self) -> str:
        """Resolve the dynamic edge policy for the current sampling strategy."""
        if self.dynamic_edge_policy == "auto":
            if self.sampling_strategy == "scaled_gaussian":
                return "knn_if_isolated"
            return "radius"
        return self.dynamic_edge_policy

    @staticmethod
    def compute_sigma(data: HeteroData) -> float:
        """
        Compute noise scale sigma as standard deviation of protein coordinates.

        Args:
            data: HeteroData with protein node positions

        Returns:
            Scalar sigma value (standard deviation across all protein coordinates)

        Note:
            Diagnostic only, no production caller, and does not exclude mates.
            Training and inference use compute_sigma_per_graph, which does.
        """
        pos = data["protein"].pos
        return float(pos.std().item())

    @staticmethod
    def compute_sigma_per_graph(
        data: HeteroData | Batch,
        device: torch.device,
        node_mask: Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute sigma (std of protein coordinates) per graph in a batch.

        Args:
            data: Batch carrying protein positions and a protein batch vector
            device: Unused for the computation; kept for call-site symmetry
            node_mask: Optional (N_protein,) bool selecting the atoms that define
                the scale. Mates sit far from the ASU, so counting them inflates
                sigma and pushes the prior out past the targets. Skipped for the
                whole batch (with a warning) if it would empty any graph.

        Returns:
            sigma: (num_graphs,) tensor of sigma values per graph
        """
        pos = data["protein"].pos  # (N_total, 3)
        batch_p = data["protein"].batch  # (N_total,)
        num_graphs = FlowMatcher._num_graphs(data)

        # an empty graph would otherwise shorten the output or yield a degenerate
        # sigma that silently places its waters at the origin. Checked against the
        # unmasked atoms so the error keeps its original meaning.
        empty = torch.bincount(batch_p, minlength=num_graphs) == 0
        if empty.any():
            raise ValueError(
                f"Cannot compute sigma for graph(s) {empty.nonzero().flatten().tolist()}: "
                "they have zero protein atoms."
            )

        # Restrict to the masked atoms (ASU-only). Skip the mask, rather than yield
        # a degenerate sigma, if it would leave a graph with no atoms -- which the
        # dataset should never produce, so warn if it happens.
        if node_mask is not None:
            eligible = node_mask.to(pos.device).bool()
            counts = torch.bincount(batch_p[eligible], minlength=num_graphs)
            if (counts == 0).any():
                logger.warning(
                    "compute_sigma_per_graph: node mask leaves a graph with no "
                    "atoms; computing sigma over all protein atoms."
                )
            else:
                pos = pos[eligible]
                batch_p = batch_p[eligible]

        # Var(X) = E[X^2] - E[X]^2
        mean_pos = scatter_mean(pos, batch_p, dim=0, dim_size=num_graphs)
        mean_sq = scatter_mean(pos**2, batch_p, dim=0, dim_size=num_graphs)
        var_per_dim = mean_sq - mean_pos**2  # (num_graphs, 3)
        sigma = torch.sqrt(var_per_dim.mean(dim=-1).clamp(min=1e-8))  # (num_graphs,)

        return sigma

    def training_step(
        self,
        batch: HeteroData,
        use_self_conditioning: bool = True,
        accumulation_steps: int = 1,
    ) -> dict[str, object]:
        """
        Single flow matching training step (forward + backward only).

        The optimizer step is handled by the caller to support gradient accumulation.

        Args:
            batch: HeteroData batch
            use_self_conditioning: Whether to use self-conditioning
            accumulation_steps: Number of gradient accumulation steps (loss is scaled by 1/accumulation_steps)

        Returns:
            Dict with 'loss', 'rmsd', 'sigma', and optionally 'per_sample_info'.

        Note:
            This method only computes forward pass, loss, and backward(). The caller
            is responsible for:
            1. optimizer.zero_grad() before calling
            2. Gradient clipping (e.g., torch.nn.utils.clip_grad_norm_)
            3. optimizer.step() after accumulating gradients

            For gradient accumulation, call this method N times, then step once.
            The loss is automatically scaled by 1/accumulation_steps for correct
            gradient magnitude. See scripts/train.py for reference implementation.
        """
        if accumulation_steps < 1:
            raise ValueError(
                f"accumulation_steps must be >= 1, got {accumulation_steps}"
            )

        self.model.train()
        device = batch["protein"].pos.device
        batch.dynamic_edge_policy = self._effective_dynamic_edge_policy()

        x1 = batch["water"].pos
        batch_w = batch["water"].batch
        num_graphs = self._num_graphs(batch)

        # same restriction the sampler uses, so the logged sigma matches it
        sigma_per_graph = self.compute_sigma_per_graph(
            batch, device, node_mask=self._asu_mask(batch)
        )
        # sampling against the batch's own water order keeps x0 aligned with x1, so
        # ot_coupling's per-graph mask selects the same nodes from both
        x0 = self._sample_waters(batch, batch_w, device)
        x0_star, x1_star = ot_coupling(x1=x1, batch=batch_w, x0=x0)

        t = torch.rand(num_graphs, device=device)
        t_per_atom = t[batch_w].unsqueeze(-1)

        x_t = (1.0 - t_per_atom) * x0_star + t_per_atom * x1_star

        # late stage path distortion
        if self.use_distortion:
            indicator = (t_per_atom >= self.t_distort).float()
            if indicator.any():
                mask = (torch.rand_like(t_per_atom) < self.p_distort).float()
                eps = torch.randn_like(x_t) * self.sigma_distort
                x_t = x_t + indicator * mask * eps

        # self-conditioning
        self_cond = None
        if use_self_conditioning and torch.rand(1).item() < self.p_self_cond:
            with torch.no_grad():
                batch["water"].pos = x_t
                v_pred_sc = self.model(batch, t, self_cond=None)
                x1_pred_sc = x_t + (1.0 - t_per_atom) * v_pred_sc
            self_cond = {"x1_pred": x1_pred_sc}

        # forward pass
        batch["water"].pos = x_t
        v_pred = self.model(batch, t, self_cond=self_cond)

        # target velocity
        v_target = x1_star - x0_star

        # weighted MSE loss (upweight near t=1)
        w = 1.0 / (self.loss_eps + (1.0 - t_per_atom))
        per_atom_mse = (v_pred - v_target).pow(2).mean(dim=-1, keepdim=True)
        loss = (w * per_atom_mse).sum() / w.sum()

        # training RMSD
        with torch.no_grad():
            x1_hat = x_t + (1.0 - t_per_atom) * v_pred
            diff2 = ((x1_hat - x1_star) ** 2).sum(-1)  # (Nw,)
            rmsd = torch.sqrt(scatter_mean(diff2, batch_w, dim=0)).mean()

        # check for high loss and compute per-sample losses for debugging
        per_sample_info = None
        if loss.item() > 100.0:
            with torch.no_grad():
                from torch_scatter import scatter_add

                weighted_mse = (w * per_atom_mse).squeeze(-1)
                numerator = scatter_add(
                    weighted_mse, batch_w, dim=0, dim_size=num_graphs
                )
                denominator = scatter_add(
                    w.squeeze(-1), batch_w, dim=0, dim_size=num_graphs
                )
                per_sample_loss = numerator / (denominator + 1e-8)
                per_sample_info = {"losses": per_sample_loss, "num_graphs": num_graphs}

        (loss / accumulation_steps).backward()

        return {
            "loss": loss.item(),
            "rmsd": rmsd.item(),
            "sigma": sigma_per_graph,
            "per_sample_info": per_sample_info,
        }

    @torch.inference_mode()
    def validation_step(self, batch: HeteroData) -> dict[str, float]:
        """
        Run single validation step without gradients.

        Args:
            batch: HeteroData batch with protein and water nodes

        Returns:
            Dict with 'loss' and 'rmsd' metrics

        Note:
            This method is for inference only. It sets model.eval(), disables
            gradients, and returns metrics. Training uses training_step() which
            handles gradient computation and loss calculation.
        """
        self.model.eval()
        device = batch["protein"].pos.device
        batch.dynamic_edge_policy = self._effective_dynamic_edge_policy()

        x1 = batch["water"].pos
        batch_w = batch["water"].batch
        num_graphs = self._num_graphs(batch)

        # sampling against the batch's own water order keeps x0 aligned with x1, so
        # ot_coupling's per-graph mask selects the same nodes from both
        x0 = self._sample_waters(batch, batch_w, device)
        x0_star, x1_star = ot_coupling(x1=x1, batch=batch_w, x0=x0)

        t = torch.rand(num_graphs, device=device)
        t_per_atom = t[batch_w].unsqueeze(-1)
        x_t = (1.0 - t_per_atom) * x0_star + t_per_atom * x1_star

        batch["water"].pos = x_t
        v_pred = self.model(batch, t, self_cond=None)

        v_target = x1_star - x0_star

        w = 1.0 / (self.loss_eps + (1.0 - t_per_atom))
        per_atom_mse = (v_pred - v_target).pow(2).mean(dim=-1, keepdim=True)
        loss = (w * per_atom_mse).sum() / w.sum()

        # GPU RMSD
        x1_hat = x_t + (1.0 - t_per_atom) * v_pred
        diff2 = ((x1_hat - x1_star) ** 2).sum(-1)  # (Nw,)
        rmsd = torch.sqrt(scatter_mean(diff2, batch_w, dim=0)).mean()

        return {
            "loss": loss.item(),
            "rmsd": rmsd.item(),
        }

    def _setup_water_nodes_from_ratio(
        self,
        g: Batch,
        water_ratio: float,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """
        Create water node positions and batch indices based on protein residue count.

        Args:
            g: Batched HeteroData graph (modified in-place)
            water_ratio: Ratio of waters to protein residues
            device: Device to create tensors on

        Returns:
            x: (N_water_total, 3) initial noise positions
            batch_w: (N_water_total,) batch indices
        """
        num_residues = g["protein"].num_residues  # (num_graphs,)

        # compute waters per graph: num_residues * ratio, minimum 1
        num_waters = (num_residues.float() * water_ratio).long().clamp(min=1)

        batch_w = _batch_from_counts(num_waters, device)
        x = self._sample_waters(g, batch_w, device)
        total_waters = batch_w.size(0)

        # create water features (oxygen one-hot; +1 for the trailing 'other' bucket)
        water_x = torch.zeros(
            total_waters, len(ELEMENT_VOCAB) + 1, dtype=torch.float32, device=device
        )
        water_x[:, ELEM_IDX["O"]] = 1.0

        # update graph with new water nodes
        g["water"].pos = x
        g["water"].x = water_x
        g["water"].batch = batch_w
        g["water"].num_nodes = total_waters

        return x, batch_w

    def _setup_water_nodes_from_count(
        self,
        g: Batch,
        water_count: int,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """
        Create water node positions and batch indices using a fixed count per protein.

        Args:
            g: Batched HeteroData graph (modified in-place)
            water_count: Exact number of waters to sample per protein
            device: Device to create tensors on

        Returns:
            x: (N_water_total, 3) initial noise positions
            batch_w: (N_water_total,) batch indices
        """
        if water_count < 0:
            raise ValueError(f"water_count must be >= 0, got {water_count}")

        num_graphs = self._num_graphs(g)

        num_waters = torch.full(
            (num_graphs,),
            water_count,
            dtype=torch.long,
            device=device,
        )

        batch_w = _batch_from_counts(num_waters, device)
        x = self._sample_waters(g, batch_w, device)
        total_waters = batch_w.size(0)

        # create water features (oxygen one-hot; +1 for the trailing 'other' bucket)
        water_x = torch.zeros(
            total_waters, len(ELEMENT_VOCAB) + 1, dtype=torch.float32, device=device
        )
        water_x[:, ELEM_IDX["O"]] = 1.0

        # update graph with new water nodes
        g["water"].pos = x
        g["water"].x = water_x
        g["water"].batch = batch_w
        g["water"].num_nodes = total_waters

        return x, batch_w

    def _setup_water_nodes(
        self,
        g: Batch,
        water_ratio: float | None,
        water_count: int | None,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """
        Create the initial water nodes to integrate from.

        Args:
            g: Batched HeteroData graph (modified in-place)
            water_ratio: If provided, sample num_residues * water_ratio waters.
                        Ignored when water_count is also given.
            water_count: If provided, sample exactly this many waters per protein.
                        Takes precedence over water_ratio. When neither is given,
                        the ground-truth water count is resampled from the prior.
            device: Device to create tensors on

        Returns:
            x: (N_water_total, 3) initial noise positions
            batch_w: (N_water_total,) batch indices
        """
        if water_count is not None:
            # sample fixed number of waters per protein
            return self._setup_water_nodes_from_count(g, water_count, device)

        if water_ratio is not None:
            # sample waters based on residue count
            return self._setup_water_nodes_from_ratio(g, water_ratio, device)

        # resample the existing water nodes in place; their batch is unchanged
        batch_w = g["water"].batch
        x = self._sample_waters(g, batch_w, device)

        return x, batch_w

    @torch.inference_mode()
    def euler_integrate(
        self,
        graphs: HeteroData | list[HeteroData],
        num_steps: int = 100,
        use_sc: bool = True,
        sc_ema_alpha: float = 0.2,
        device: str | torch.device = "cuda",
        water_ratio: float | None = None,
        water_count: int | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """
        Euler integration from noise to final positions.

        Args:
            graphs: Single HeteroData or list of HeteroData graphs to process
            num_steps: Number of integration steps
            use_sc: Whether to use self-conditioning
            sc_ema_alpha: EMA decay for self-conditioning
            device: Device to run on
            water_ratio: If provided, sample num_residues * water_ratio waters
                        instead of using ground truth water count. Ignored when
                        water_count is also given.
            water_count: If provided, sample exactly this many waters per protein.
                        Takes precedence over water_ratio. When neither is given,
                        the ground-truth water count is resampled from the prior.

        Returns:
            List of dicts, one per input graph, each with keys:
                'protein_pos': (Np, 3) - includes both ASU and mate atoms
                'water_true': (Nw, 3) ground-truth waters (always returned; when
                        water_ratio/water_count is set its count may differ from water_pred)
                'water_pred': (Nw, 3) final prediction
                'pdb_id': PDB identifier
        """
        self.model.eval()
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        # handle single graph input
        if isinstance(graphs, HeteroData):
            graphs = [graphs]

        # store original pdb_ids before batching
        pdb_ids = [getattr(g, "pdb_id", None) for g in graphs]

        # batch graphs together
        g = Batch.from_data_list([copy.deepcopy(graph) for graph in graphs]).to(device)

        batch_p = g["protein"].batch
        num_graphs = self._num_graphs(g)

        # store ground truth water positions and batch indices before modifying
        x1_true = g["water"].pos.clone()
        batch_w_true = g["water"].batch.clone()

        x, batch_w = self._setup_water_nodes(g, water_ratio, water_count, device)

        x1_pred_ema = x.clone()

        ts = torch.linspace(0, 1, num_steps, device=device)
        dt = ts[1] - ts[0]

        for i in range(num_steps - 1):
            t_scalar = ts[i]
            t = t_scalar.expand(num_graphs)  # (num_graphs,) all same value

            g["water"].pos = x
            self_cond = {"x1_pred": x1_pred_ema} if use_sc else None
            v = self.model(g, t, self_cond=self_cond)
            x = x + dt * v

            if use_sc:
                t_next_scalar = ts[i + 1]
                t_next = t_next_scalar.expand(num_graphs)
                g["water"].pos = x
                v_next = self.model(g, t_next, self_cond={"x1_pred": x1_pred_ema})
                x1_pred_now = x + (1.0 - t_next_scalar) * v_next
                x1_pred_ema = (
                    1.0 - sc_ema_alpha
                ) * x1_pred_ema + sc_ema_alpha * x1_pred_now

        # split results by graph
        x_cpu = x.detach().cpu()
        protein_pos_cpu = g["protein"].pos.detach().cpu()
        x1_true_cpu = x1_true.detach().cpu()
        batch_w_cpu = batch_w.cpu()
        batch_w_true_cpu = batch_w_true.cpu()
        batch_p_cpu = batch_p.cpu()

        results = []
        for i in range(num_graphs):
            mask_w = batch_w_cpu == i
            mask_w_true = batch_w_true_cpu == i
            mask_p = batch_p_cpu == i

            result = {
                "protein_pos": protein_pos_cpu[mask_p].numpy(),
                "water_true": x1_true_cpu[mask_w_true].numpy(),
                "water_pred": x_cpu[mask_w].numpy(),
                "pdb_id": pdb_ids[i],
            }
            results.append(result)

        return results

    @torch.inference_mode()
    def rk4_integrate(
        self,
        graphs: HeteroData | list[HeteroData],
        num_steps: int = 500,
        use_sc: bool = True,
        sc_ema_alpha: float = 0.2,
        device: str | torch.device = "cuda",
        return_trajectory: bool = True,
        water_ratio: float | None = None,
        water_count: int | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """
        RK4 integration from noise to final positions.

        Args:
            graphs: Single HeteroData or list of HeteroData graphs to process
            num_steps: Number of integration steps
            use_sc: Whether to use self-conditioning
            sc_ema_alpha: EMA decay for self-conditioning
            device: Device to run on
            return_trajectory: Whether to return full trajectory and metrics
            water_ratio: If provided, sample num_residues * water_ratio waters
                        instead of using ground truth water count. Ignored when
                        water_count is also given.
            water_count: If provided, sample exactly this many waters per protein.
                        Takes precedence over water_ratio. When neither is given,
                        the ground-truth water count is resampled from the prior.

        Returns:
            List of dicts, one per input graph, each with keys:
                'protein_pos': (Np, 3) - includes both ASU and mate atoms
                'water_true': (Nw, 3) ground-truth waters (always returned; when
                        water_ratio/water_count is set its count may differ from water_pred)
                'water_pred': (Nw, 3) final prediction
                'trajectory': list of (Nw, 3) at each step (if return_trajectory=True)
        """
        self.model.eval()
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        # handle single graph input
        if isinstance(graphs, HeteroData):
            graphs = [graphs]

        # store original pdb_ids before batching
        pdb_ids = [getattr(g, "pdb_id", None) for g in graphs]

        # batch graphs together
        g = Batch.from_data_list([copy.deepcopy(graph) for graph in graphs]).to(device)

        batch_p = g["protein"].batch
        num_graphs = self._num_graphs(g)

        # store ground truth water positions and batch indices before modifying
        x1_true = g["water"].pos.clone()
        batch_w_true = g["water"].batch.clone()

        x, batch_w = self._setup_water_nodes(g, water_ratio, water_count, device)

        x1_pred_ema = x.clone()

        ts = torch.linspace(0, 1, num_steps, device=device)
        dt = ts[1] - ts[0]

        if return_trajectory:
            # store trajectory per-graph
            trajectories = [[] for _ in range(num_graphs)]
            x_cpu = x.detach().cpu()
            batch_w_cpu = batch_w.cpu()
            for i in range(num_graphs):
                mask = batch_w_cpu == i
                trajectories[i].append(x_cpu[mask].numpy().copy())

        # rK4 integration
        for step in tqdm(range(num_steps - 1), desc="RK4 integration", leave=False):
            t0_scalar = ts[step]
            t0 = t0_scalar.expand(num_graphs)  # (num_graphs,) all same value

            def f(xpos, t_tensor):
                g["water"].pos = xpos
                self_cond = {"x1_pred": x1_pred_ema} if use_sc else None
                return self.model(g, t_tensor, self_cond=self_cond)

            k1 = f(x, t0)
            k2 = f(x + 0.5 * dt * k1, (t0_scalar + 0.5 * dt).expand(num_graphs))
            k3 = f(x + 0.5 * dt * k2, (t0_scalar + 0.5 * dt).expand(num_graphs))
            k4 = f(x + dt * k3, (t0_scalar + dt).expand(num_graphs))

            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if use_sc:
                t1_scalar = ts[step + 1]
                t1 = t1_scalar.expand(num_graphs)
                g["water"].pos = x
                v_next = self.model(g, t1, self_cond={"x1_pred": x1_pred_ema})
                x1_pred_now = x + (1.0 - t1_scalar) * v_next
                x1_pred_ema = (
                    1.0 - sc_ema_alpha
                ) * x1_pred_ema + sc_ema_alpha * x1_pred_now

            if return_trajectory:
                x_cpu = x.detach().cpu()
                for i in range(num_graphs):
                    mask = batch_w_cpu == i
                    trajectories[i].append(x_cpu[mask].numpy().copy())

        # split results by graph
        x_cpu = x.detach().cpu()
        protein_pos_cpu = g["protein"].pos.detach().cpu()
        x1_true_cpu = x1_true.detach().cpu()
        batch_w_cpu = batch_w.cpu()
        batch_w_true_cpu = batch_w_true.cpu()
        batch_p_cpu = batch_p.cpu()

        results = []
        for i in range(num_graphs):
            mask_w = batch_w_cpu == i
            mask_w_true = batch_w_true_cpu == i
            mask_p = batch_p_cpu == i

            result = {
                "protein_pos": protein_pos_cpu[mask_p].numpy(),
                "water_true": x1_true_cpu[mask_w_true].numpy(),
                "water_pred": x_cpu[mask_w].numpy(),
                "pdb_id": pdb_ids[i],
            }

            if return_trajectory:
                result["trajectory"] = trajectories[i]

            results.append(result)

        return results

    def sample(
        self,
        graphs: HeteroData | list[HeteroData],
        num_steps: int = 100,
        method: str = "euler",
        use_sc: bool = True,
        device: str = "cuda",
    ) -> np.ndarray | list[np.ndarray]:
        """
        Sample water positions for one or more graphs.

        Args:
            graphs: Single HeteroData or list of HeteroData graphs
            num_steps: Number of integration steps
            method: 'euler' or 'rk4'
            use_sc: Whether to use self-conditioning
            device: Device to run on

        Returns:
            If single graph input: (Nw, 3) predicted water positions
            If list input: List of (Nw_i, 3) predicted water positions
        """
        single_input = isinstance(graphs, HeteroData)

        if method == "euler":
            results = self.euler_integrate(graphs, num_steps, use_sc, device=device)
            results = [r["water_pred"] for r in results]
        elif method == "rk4":
            results = self.rk4_integrate(
                graphs, num_steps, use_sc, device=device, return_trajectory=False
            )
            results = [r["water_pred"] for r in results]
        else:
            raise ValueError(f"Unknown method: {method}")

        # return single array if single graph was provided
        if single_input:
            return results[0]
        return results
