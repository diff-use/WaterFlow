"""
Flow matching model components for water placement prediction.

This module provides:
- ProteinWaterUpdate: Heterogeneous GVP message passing across the active edge types
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
from torch_geometric.nn import knn, radius, radius_graph
from torch_scatter import scatter, scatter_mean
from tqdm.auto import tqdm

from src.constants import (
    ALL_EDGE_TYPES,
    DEFAULT_EDGE_CUTOFF,
    EDGE_PP,
    EDGE_PW,
    EDGE_WP,
    EDGE_WW,
    ELEM_IDX,
    ELEMENT_VOCAB,
    get_active_edge_types,
    NUM_RBF,
)
from src.encoder_base import BaseProteinEncoder
from src.gvp import GVP, GVPMultiEdgeConv
from src.utils import ot_coupling


# "knn_if_isolated" = radius edges, plus nearest-neighbour edges for nodes left with none.
DYNAMIC_EDGE_POLICIES = ("radius", "knn", "knn_if_isolated")


def resolve_edge_policy(policy: str, sampling_strategy: str = "uniform_ball") -> str:
    """
    Resolve a recorded `dynamic_edge_policy`, including the "auto" setting.

    "auto" is what every recorded run carries. It reads off the prior: uniform
    ball samples land within `cutoff` of a protein atom by construction, so
    nothing is stranded and the rescue is not wanted; Gaussian samples carry no
    such guarantee, so they get it.

    Args:
        policy: Value from a config file or CLI flag.
        sampling_strategy: Prior the run uses, consulted only for "auto".

    Returns:
        One of DYNAMIC_EDGE_POLICIES.

    Raises:
        ValueError: If the value is not "auto" or a member of
            DYNAMIC_EDGE_POLICIES.
    """
    if policy == "auto":
        return "knn_if_isolated" if sampling_strategy == "scaled_gaussian" else "radius"
    if policy not in DYNAMIC_EDGE_POLICIES:
        raise ValueError(
            f"dynamic_edge_policy must be 'auto' or one of {DYNAMIC_EDGE_POLICIES}, "
            f"got '{policy}'"
        )
    return policy


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


def _eligible_mask_or_skip(
    batch_p: Tensor,
    mask: Tensor,
    num_graphs: int,
    warn_msg: str,
    requesting: Tensor | None = None,
) -> Tensor | None:
    """Restrict protein atoms to *mask*, unless doing so would empty a graph.

    An ASU-only mask should always leave every graph with at least one atom. If
    it somehow does not, applying it would starve a graph (no anchor to sample
    from / a degenerate sigma at the origin), so we skip the mask entirely and
    warn rather than silently corrupt that graph.

    Args:
        batch_p: (N_protein,) graph index per protein atom.
        mask: (N_protein,) bool eligibility mask over the same atoms.
        num_graphs: Total graph count, so empty graphs are counted.
        warn_msg: Message logged when the mask would empty a graph.
        requesting: Optional graph indices that must stay non-empty; when None,
            every graph must. `sample_waters_uniform_ball` only cares about
            water-requesting graphs, so it passes `batch_w`.

    Returns:
        The bool mask to apply, or None if it would empty a graph (skip it).
    """
    eligible = mask.to(batch_p.device).bool()
    counts = torch.bincount(batch_p[eligible], minlength=num_graphs)
    empty = counts == 0 if requesting is None else counts[requesting] == 0
    if empty.any():
        logger.warning(warn_msg)
        return None
    return eligible


def sample_waters_uniform_ball(
    protein_pos: Tensor,
    batch_p: Tensor,
    batch_w: Tensor,
    cutoff: float = DEFAULT_EDGE_CUTOFF,
    device: torch.device | None = None,
    anchor_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
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

    # Drop to the eligible anchors (ASU-only), keeping only water-requesting graphs
    # non-empty; the helper skips the mask (with a warning) if it would leave such
    # a graph with no anchor -- an invariant the dataset should never violate.
    if anchor_mask is not None:
        eligible = _eligible_mask_or_skip(
            batch_p,
            anchor_mask,
            num_graphs,
            "sample_waters_uniform_ball: anchor mask leaves a water-requesting "
            "graph with no anchor; anchoring the batch on all protein atoms.",
            requesting=batch_w,
        )
        if eligible is not None:
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
    local_idx = (
        torch.rand(total_waters, device=device, generator=generator)
        * graph_sizes.float()
    ).long()
    anchors = protein_pos[graph_offsets + local_idx]

    # uniform direction on the unit sphere
    direction = torch.randn(
        total_waters, 3, device=device, dtype=protein_pos.dtype, generator=generator
    )
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    # uniform radius inside the ball: r = R * U^(1/3)
    r = cutoff * torch.rand(
        total_waters, 1, device=device, dtype=protein_pos.dtype, generator=generator
    ).pow(1.0 / 3.0)

    return anchors + r * direction


def sample_waters_scaled_gaussian(
    batch_w: Tensor,
    sigma_per_graph: Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
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

    return (
        torch.randn(total_waters, 3, device=device, dtype=dtype, generator=generator)
        * sigma
    )


def build_dynamic_edges(
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    *,
    policy: str,
    k: int,
    r: float,
    max_neighbors: int = 256,
    batch_src: torch.Tensor | None = None,
    batch_dst: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build edges from src -> dst (source indices in row 0, dest in row 1).

    The two policies differ in which side the neighbour budget applies to, which
    matters when reading coverage guarantees off the result:

    - ``"knn"`` queries *per destination*: each point in ``dst_pos`` takes its
      ``k`` nearest neighbours in ``src_pos``. Every destination is therefore
      guaranteed incoming edges and appears in row 1, while a source that is no
      destination's nearest neighbour may not appear in row 0 at all. Coverage
      checks must be made against row 1.
    - ``"radius"`` connects every pair within ``r``, capped at ``max_neighbors``
      *per source*. Nothing is guaranteed: a node with an empty neighbourhood
      gets no edges, which is what :meth:`ProteinWaterUpdate._add_knn_fallback`
      exists to repair.

    For a homogeneous graph (``src_pos is dst_pos``) self-edges are dropped.

    Args:
        src_pos: (N_src, 3) source node positions.
        dst_pos: (N_dst, 3) destination node positions.
        policy: "radius" or "knn". ("knn_if_isolated" is not handled here;
            ProteinWaterUpdate splits it into a "radius" call plus a rescue pass.)
        k: Nearest neighbours per destination, used when policy is "knn".
        r: Distance cutoff in Angstroms, used when policy is "radius".
        max_neighbors: Per-source cap on radius results.
        batch_src: (N_src,) batch assignment for source nodes, or None.
        batch_dst: (N_dst,) batch assignment for destination nodes, or None.

    Returns:
        (2, E) edge index tensor with source indices in row 0, destination in
        row 1.
    """
    if src_pos.numel() == 0 or dst_pos.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=src_pos.device)

    # Same object
    homogeneous = src_pos is dst_pos

    if policy == "knn":
        # Asked for each destination's nearest sources, so sources come back second.
        dst_idx, src_idx = knn(
            x=src_pos, y=dst_pos, k=k, batch_x=batch_src, batch_y=batch_dst
        )
        edge_index = torch.stack((src_idx, dst_idx), dim=0)
        if homogeneous:
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        return edge_index.unique(dim=1)

    # Cap against the number of reachable counterparts, minus the self-edge that
    # a homogeneous query would otherwise spend a slot on.
    num_candidates = dst_pos.size(0) - 1 if homogeneous else dst_pos.size(0)
    cap = max(1, min(num_candidates, max_neighbors))

    if homogeneous:
        return radius_graph(
            src_pos, r=r, batch=batch_src, loop=False, max_num_neighbors=cap
        )

    # Asked for each source's neighbours within r, so sources already come back first.
    return radius(
        x=dst_pos,
        y=src_pos,
        r=r,
        batch_x=batch_dst,
        batch_y=batch_src,
        max_num_neighbors=cap,
    )


class ProteinWaterUpdate(nn.Module):
    """
    Heterogeneous GVP message passing over the active edge types:
      - protein -> water  (pw)   always active
      - protein -> protein (pp)  always active
      - water   -> water  (ww)   ablatable
      - water   -> protein (wp)  ablatable

    Which are active is fixed at construction via `etypes`; see
    `constants.get_active_edge_types`.
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
        etypes: list[tuple[str, str, str]] | None = None,
        cutoff: float = DEFAULT_EDGE_CUTOFF,
        max_neighbors: int = 256,
        dynamic_edge_policy: str = "radius",
        sampling_strategy: str = "uniform_ball",
        knn_fallback_k: int = 8,
        k_pw: int = 12,
        k_ww: int = 8,
        k_wp: int = 8,
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
            etypes: Active edge types. Defaults to ALL_EDGE_TYPES. Build with
                `constants.get_active_edge_types` to ablate WW/WP.
            cutoff: Distance cutoff in Angstroms for radius edges.
            max_neighbors: Cap on neighbours per source node in radius queries,
                bounding edge count and runtime on dense structures.
            dynamic_edge_policy: "auto" or one of DYNAMIC_EDGE_POLICIES. "radius"
                connects everything within `cutoff`; "knn" connects a fixed
                number of nearest neighbours using k_pw/k_ww/k_wp.
            sampling_strategy: Prior the run uses, consulted only to resolve
                "auto"; see `resolve_edge_policy`.
            knn_fallback_k: Under "knn_if_isolated", attach this many nearest
                neighbours to any water the radius query left with no edges. 0
                disables the rescue. Ignored under the other policies.
            k_pw: Nearest neighbours for protein -> water edges under "knn".
            k_ww: Nearest neighbours for water -> water edges under "knn".
            k_wp: Nearest neighbours for water -> protein edges under "knn".

        Raises:
            ValueError: If `dynamic_edge_policy` is not a known value, or if
                `knn_fallback_k` is negative.
        """
        super().__init__()
        # Unpack hidden dimensions: s_h = scalar hidden dim, v_h = vector hidden dim
        s_h, v_h = hidden_dims

        if knn_fallback_k < 0:
            raise ValueError(f"knn_fallback_k must be >= 0, got {knn_fallback_k}")

        # build_edges only knows how to construct the four known relations, and
        # HeteroConv would KeyError on any other, so reject it up front.
        unknown = [et for et in (etypes or []) if et not in ALL_EDGE_TYPES]
        if unknown:
            raise ValueError(
                f"etypes must be a subset of {ALL_EDGE_TYPES}, got unknown {unknown}"
            )

        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        resolved = resolve_edge_policy(dynamic_edge_policy, sampling_strategy)
        # Same edges as "radius"; the difference is the extra pass for nodes left with none.
        self.rescue_isolated = resolved == "knn_if_isolated" and knn_fallback_k > 0
        self.dynamic_edge_policy = (
            "radius" if resolved == "knn_if_isolated" else resolved
        )
        self.knn_fallback_k = knn_fallback_k
        self.k_pw = k_pw
        self.k_ww = k_ww
        self.k_wp = k_wp

        etypes = ALL_EDGE_TYPES if etypes is None else etypes

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

    def _add_knn_fallback(
        self,
        edge_index: torch.Tensor,
        src_pos: torch.Tensor,
        dst_pos: torch.Tensor,
        batch_src: torch.Tensor | None,
        batch_dst: torch.Tensor | None,
        isolate_axis: int,
    ) -> torch.Tensor:
        """
        Attach KNN edges for nodes the radius query left with no edges.

        A radius query strands any node with nothing inside `cutoff`. Those nodes
        would reach the GVP blocks with no incoming messages, so they are
        reconnected to their `knn_fallback_k` nearest counterparts regardless of
        distance.

        Args:
            edge_index: (2, E) radius edges, source row 0, destination row 1.
            src_pos: (N_src, 3) source positions.
            dst_pos: (N_dst, 3) destination positions.
            batch_src: (N_src,) batch assignment for sources, or None.
            batch_dst: (N_dst,) batch assignment for destinations, or None.
            isolate_axis: Row to check for stranded nodes -- 0 for sources,
                1 for destinations.

        Returns:
            (2, E') edge index with fallback edges merged in and deduplicated.
        """
        device = src_pos.device
        num_nodes = dst_pos.size(0) if isolate_axis == 1 else src_pos.size(0)
        if num_nodes == 0:
            return edge_index

        connected = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if edge_index.numel() > 0:
            connected[edge_index[isolate_axis].unique()] = True
        isolated = (~connected).nonzero(as_tuple=False).flatten()
        if isolated.numel() == 0:
            return edge_index

        # Query only the stranded nodes, then lift the returned local indices
        # back into the full node set via `isolated`.
        if isolate_axis == 1:
            fallback = build_dynamic_edges(
                src_pos,
                dst_pos[isolated],
                policy="knn",
                k=max(1, min(self.knn_fallback_k, src_pos.size(0))),
                r=self.cutoff,
                batch_src=batch_src,
                batch_dst=batch_dst[isolated] if batch_dst is not None else None,
            )
            fallback = torch.stack((fallback[0], isolated[fallback[1]]), dim=0)
        else:
            fallback = build_dynamic_edges(
                dst_pos,
                src_pos[isolated],
                policy="knn",
                k=max(1, min(self.knn_fallback_k, dst_pos.size(0))),
                r=self.cutoff,
                batch_src=batch_dst,
                batch_dst=batch_src[isolated] if batch_src is not None else None,
            )
            fallback = torch.stack((isolated[fallback[1]], fallback[0]), dim=0)

        if edge_index.numel() == 0:
            return fallback
        return torch.cat((edge_index, fallback), dim=1).unique(dim=1)

    def build_edges(self, data: HeteroData) -> dict[tuple[str, str, str], torch.Tensor]:
        """
        Build the edge set for one batch under the active policy.

        PP and PW edges are read from the dataset when cached at preprocessing
        time and built on the fly otherwise. WW and WP are always dynamic, and
        are skipped entirely when ablated out of `etypes`.

        Args:
            data: HeteroData with 'protein' and 'water' node types containing
                positions, optionally carrying cached PP/PW edges.

        Returns:
            Dict mapping each active edge type to a (2, E) edge index tensor.
        """
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}

        batch_p = data["protein"].batch if "batch" in data["protein"] else None
        batch_w = data["water"].batch if "batch" in data["water"] else None

        pos_p = data["protein"].pos
        pos_w = data["water"].pos

        # The rescue only ever targets water nodes, so it runs on the two edge
        # types that connect water to protein: protein->water and water->protein.
        # The `isolate_axis` differs because water sits on opposite ends of them --
        # it is the destination of PW (axis 1) and the source of WP (axis 0) -- so
        # each pass checks the row where the water lives. Water-water is skipped: a
        # water with no WW neighbour still exchanges messages with protein through
        # PW/WP, so it is never truly stranded, and WW is context, not a lifeline.
        rescue = self.rescue_isolated

        # protein -> water (water is the destination, so it is row 1)
        if EDGE_PW in self.etypes:
            if EDGE_PW in data.edge_types:
                edge_index_dict[EDGE_PW] = data[EDGE_PW].edge_index
            else:
                ei = build_dynamic_edges(
                    pos_p,
                    pos_w,
                    policy=self.dynamic_edge_policy,
                    k=self.k_pw,
                    r=self.cutoff,
                    max_neighbors=self.max_neighbors,
                    batch_src=batch_p,
                    batch_dst=batch_w,
                )
                if rescue:
                    ei = self._add_knn_fallback(
                        ei, pos_p, pos_w, batch_p, batch_w, isolate_axis=1
                    )
                edge_index_dict[EDGE_PW] = ei

        # protein -> protein (cached from the dataset in every normal run)
        if EDGE_PP in self.etypes:
            edge_index_dict[EDGE_PP] = (
                data[EDGE_PP].edge_index
                if EDGE_PP in data.edge_types
                else build_dynamic_edges(
                    pos_p,
                    pos_p,
                    policy=self.dynamic_edge_policy,
                    k=self.k_pw,
                    r=self.cutoff,
                    max_neighbors=self.max_neighbors,
                    batch_src=batch_p,
                    batch_dst=batch_p,
                )
            )

        # water -> water
        if EDGE_WW in self.etypes:
            edge_index_dict[EDGE_WW] = build_dynamic_edges(
                pos_w,
                pos_w,
                policy=self.dynamic_edge_policy,
                k=self.k_ww,
                r=self.cutoff,
                max_neighbors=self.max_neighbors,
                batch_src=batch_w,
                batch_dst=batch_w,
            )

        # water -> protein (water is the source, so it is row 0)
        if EDGE_WP in self.etypes:
            ei = build_dynamic_edges(
                pos_w,
                pos_p,
                policy=self.dynamic_edge_policy,
                k=self.k_wp,
                r=self.cutoff,
                max_neighbors=self.max_neighbors,
                batch_src=batch_w,
                batch_dst=batch_p,
            )
            if rescue:
                ei = self._add_knn_fallback(
                    ei, pos_w, pos_p, batch_w, batch_p, isolate_axis=0
                )
            edge_index_dict[EDGE_WP] = ei

        return edge_index_dict

    def forward(
        self,
        x_dict: dict[str, tuple[torch.Tensor, torch.Tensor]],
        data: HeteroData,
        pp_edge_attr: tuple | None = None,
    ):
        """
        Run heterogeneous message passing across protein and water nodes.

        Args:
            x_dict: Node features dict with:
                - 'protein': (s_p, v_p) where s_p is (N_p, scalar_dim), v_p is (N_p, vector_dim, 3)
                - 'water': (s_w, v_w) where s_w is (N_w, scalar_dim), v_w is (N_w, vector_dim, 3)
            data: HeteroData with 'protein' and 'water' node positions
            pp_edge_attr: Optional encoder-learned edge features (s_edge, V_edge) for PP edges.
                If provided, uses encoder-learned scalar features (s_edge) combined with
                cached edge direction unit vectors (edge_unit_vectors, pre-normalized at preprocessing).
                If None, uses cached geometric edge features (edge_rbf, edge_unit_vectors) from the dataset.

        Returns:
            Updated x_dict with same structure as input
        """
        pos_dict = {nt: data[nt].pos for nt in data.node_types if "pos" in data[nt]}

        edge_index_dict = self.build_edges(data)

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
        water_input_dim: int = 16,  # 1 hot with oxygen, same as encoder
        cutoff: float = DEFAULT_EDGE_CUTOFF,
        max_neighbors: int = 256,
        dynamic_edge_policy: str = "radius",
        sampling_strategy: str = "uniform_ball",
        knn_fallback_k: int = 8,
        disable_ww: bool = False,
        disable_wp: bool = False,
        k_pw: int = 12,
        k_ww: int = 8,
        k_wp: int = 8,
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
            water_input_dim: Input dimension for water node features. Default: 16
            cutoff: Distance cutoff in Angstroms for radius edges. Default: 8.0
            max_neighbors: Per-source cap on radius results. Default: 256
            dynamic_edge_policy: How water-touching edges are built, one of
                DYNAMIC_EDGE_POLICIES. Default: "radius"
            knn_fallback_k: Nearest neighbours attached to waters the radius
                query stranded; 0 disables the rescue. Default: 8
            disable_ww: Ablate water -> water edges. Default: False
            disable_wp: Ablate water -> protein edges. Default: False
            k_pw: K nearest neighbors for protein-water edges under the "knn"
                policy. Default: 12
            k_ww: K nearest neighbors for water-water edges under "knn". Default: 8
            k_wp: K nearest neighbors for water-protein edges under "knn". Default: 8
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
        # Read back by FlowMatcher for the water prior's sampling radius. The rest
        # of the edge configuration is owned by `self.updater` and deliberately not
        # mirrored here -- two copies would be two things to keep in sync.
        self.cutoff = cutoff

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
            etypes=get_active_edge_types(disable_ww=disable_ww, disable_wp=disable_wp),
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            dynamic_edge_policy=dynamic_edge_policy,
            sampling_strategy=sampling_strategy,
            knn_fallback_k=knn_fallback_k,
            k_pw=k_pw,
            k_ww=k_ww,
            k_wp=k_wp,
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

    def __init__(
        self,
        model,
        sampling_strategy: str = "uniform_ball",
        use_amp: bool = False,
    ):
        """
        Initialize flow matcher for training and inference.

        Args:
            model: FlowWaterGVP model instance
            sampling_strategy: Source distribution for flow matching noise.
                "uniform_ball" samples uniformly in balls around protein atoms.
                "scaled_gaussian" samples from N(0, sigma^2*I).
            use_amp: Run forward passes under bfloat16 autocast (CUDA only). The
                loss is still reduced in fp32, so it is a no-op off CUDA.

        Note:
            Edge construction is configured on the model, not here, so training
            and integration always build edges the same way.
        """
        if sampling_strategy not in self.SAMPLING_STRATEGIES:
            raise ValueError(
                f"sampling_strategy must be one of {self.SAMPLING_STRATEGIES}, "
                f"got '{sampling_strategy}'"
            )
        self.model = model
        self.use_amp = use_amp
        # Read the graph cutoff off the flow model. Under DDP the attribute lives
        # on the wrapped `.module` (DDP does not forward attribute lookups), so
        # fall through to it; otherwise a DDP run would silently use the default
        # radius and change the water prior only under DDP.
        if hasattr(model, "cutoff"):
            self.graph_cutoff = model.cutoff
        elif hasattr(model, "module") and hasattr(model.module, "cutoff"):
            self.graph_cutoff = model.module.cutoff
        else:
            self.graph_cutoff = DEFAULT_EDGE_CUTOFF
        self.sampling_strategy = sampling_strategy

    def _autocast_context(self, device: torch.device):
        """bf16 autocast when use_amp is set and the device is a bf16-capable CUDA
        GPU; a no-op otherwise. Guards every caller, including direct construction.
        """
        enabled = (
            self.use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
        )
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled)

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
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Dispatch to the configured sampling strategy, sampling one water per
        entry of batch_w and returning them in that order. `generator`, if given,
        replaces the global torch RNG."""
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
                generator=generator,
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
            generator=generator,
        )

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

        # Restrict to the masked atoms (ASU-only). Every graph must stay non-empty
        # here (each yields a sigma), so no `requesting`; the helper skips the mask
        # (with a warning) rather than yield a degenerate origin-sigma for a graph
        # the mask would empty -- which the dataset should never produce.
        if node_mask is not None:
            eligible = _eligible_mask_or_skip(
                batch_p,
                node_mask,
                num_graphs,
                "compute_sigma_per_graph: node mask leaves a graph with no "
                "atoms; computing sigma over all protein atoms.",
            )
            if eligible is not None:
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
        accumulation_steps: int = 1,
    ) -> dict[str, object]:
        """
        Single flow matching training step (forward + backward only).

        The optimizer step is handled by the caller to support gradient accumulation.

        Args:
            batch: HeteroData batch
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

        # forward pass under bf16 autocast (CUDA only); cast back to fp32 so the
        # loss and RMSD below run in full precision
        batch["water"].pos = x_t
        with self._autocast_context(device):
            v_pred = self.model(batch, t)
        v_pred = v_pred.float()

        # target velocity
        v_target = x1_star - x0_star

        # MSE over the velocity field, reduced in fp32
        per_atom_mse = (v_pred - v_target).pow(2).mean(dim=-1)  # (Nw,)
        loss = per_atom_mse.mean()

        # training RMSD
        with torch.no_grad():
            x1_hat = x_t + (1.0 - t_per_atom) * v_pred
            diff2 = ((x1_hat - x1_star) ** 2).sum(-1)  # (Nw,)
            rmsd = torch.sqrt(scatter_mean(diff2, batch_w, dim=0)).mean()

        # check for high loss and compute per-sample losses for debugging
        per_sample_info = None
        if loss.item() > 100.0:
            with torch.no_grad():
                per_sample_loss = scatter_mean(
                    per_atom_mse, batch_w, dim=0, dim_size=num_graphs
                )
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
        with self._autocast_context(device):
            v_pred = self.model(batch, t)
        v_pred = v_pred.float()

        v_target = x1_star - x0_star

        # MSE over the velocity field, reduced in fp32
        loss = (v_pred - v_target).pow(2).mean()

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
        generator: torch.Generator | None = None,
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
        x = self._sample_waters(g, batch_w, device, generator)
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
        generator: torch.Generator | None = None,
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
        x = self._sample_waters(g, batch_w, device, generator)
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
        generator: torch.Generator | None = None,
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
            generator: RNG for the prior noise; None uses the global torch RNG.

        Returns:
            x: (N_water_total, 3) initial noise positions
            batch_w: (N_water_total,) batch indices
        """
        if water_count is not None:
            # sample fixed number of waters per protein
            return self._setup_water_nodes_from_count(g, water_count, device, generator)

        if water_ratio is not None:
            # sample waters based on residue count
            return self._setup_water_nodes_from_ratio(g, water_ratio, device, generator)

        # resample the existing water nodes in place; their batch is unchanged
        batch_w = g["water"].batch
        x = self._sample_waters(g, batch_w, device, generator)

        return x, batch_w

    @staticmethod
    def _split_waters(
        x: Tensor, batch_w_cpu: Tensor, num_graphs: int
    ) -> list[np.ndarray]:
        """Per-graph copies of water positions x, split by batch index."""
        x_cpu = x.detach().cpu()
        return [x_cpu[batch_w_cpu == i].numpy().copy() for i in range(num_graphs)]

    @torch.inference_mode()
    def euler_integrate(
        self,
        graphs: HeteroData | list[HeteroData],
        num_steps: int = 100,
        device: str | torch.device = "cuda",
        return_trajectory: bool = False,
        water_ratio: float | None = None,
        water_count: int | None = None,
        generator: torch.Generator | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """
        Euler integration from noise to final positions.

        Args:
            graphs: Single HeteroData or list of HeteroData graphs to process
            num_steps: Number of integration steps
            device: Device to run on
            return_trajectory: Whether to also return the per-step trajectory
            water_ratio: If provided, sample num_residues * water_ratio waters
                        instead of using ground truth water count. Ignored when
                        water_count is also given.
            water_count: If provided, sample exactly this many waters per protein.
                        Takes precedence over water_ratio. When neither is given,
                        the ground-truth water count is resampled from the prior.
            generator: RNG for the prior noise, the integration's only randomness.
                        None uses the global torch RNG.

        Returns:
            List of dicts, one per input graph, each with keys:
                'protein_pos': (Np, 3) - includes both ASU and mate atoms
                'water_true': (Nw, 3) ground-truth waters (always returned; when
                        water_ratio/water_count is set its count may differ from water_pred)
                'water_pred': (Nw, 3) final prediction
                'pdb_id': PDB identifier
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

        x, batch_w = self._setup_water_nodes(
            g, water_ratio, water_count, device, generator
        )

        ts = torch.linspace(0, 1, num_steps, device=device)
        dt = ts[1] - ts[0]

        batch_w_cpu = batch_w.cpu()
        if return_trajectory:
            trajectories = [
                [pos] for pos in self._split_waters(x, batch_w_cpu, num_graphs)
            ]

        for i in range(num_steps - 1):
            t_scalar = ts[i]
            t = t_scalar.expand(num_graphs)  # (num_graphs,) all same value

            g["water"].pos = x
            v = self.model(g, t)
            x = x + dt * v

            if return_trajectory:
                for traj, pos in zip(
                    trajectories, self._split_waters(x, batch_w_cpu, num_graphs)
                ):
                    traj.append(pos)

        # split results by graph
        x_cpu = x.detach().cpu()
        protein_pos_cpu = g["protein"].pos.detach().cpu()
        x1_true_cpu = x1_true.detach().cpu()
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

    @torch.inference_mode()
    def rk4_integrate(
        self,
        graphs: HeteroData | list[HeteroData],
        num_steps: int = 500,
        device: str | torch.device = "cuda",
        return_trajectory: bool = True,
        water_ratio: float | None = None,
        water_count: int | None = None,
        generator: torch.Generator | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """
        RK4 integration from noise to final positions.

        Args:
            graphs: Single HeteroData or list of HeteroData graphs to process
            num_steps: Number of integration steps
            device: Device to run on
            return_trajectory: Whether to return full trajectory and metrics
            water_ratio: If provided, sample num_residues * water_ratio waters
                        instead of using ground truth water count. Ignored when
                        water_count is also given.
            water_count: If provided, sample exactly this many waters per protein.
                        Takes precedence over water_ratio. When neither is given,
                        the ground-truth water count is resampled from the prior.
            generator: RNG for the prior noise, the integration's only randomness.
                        None uses the global torch RNG.

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

        x, batch_w = self._setup_water_nodes(
            g, water_ratio, water_count, device, generator
        )

        ts = torch.linspace(0, 1, num_steps, device=device)
        dt = ts[1] - ts[0]

        batch_w_cpu = batch_w.cpu()
        if return_trajectory:
            trajectories = [
                [pos] for pos in self._split_waters(x, batch_w_cpu, num_graphs)
            ]

        # rK4 integration
        for step in tqdm(range(num_steps - 1), desc="RK4 integration", leave=False):
            t0_scalar = ts[step]
            t0 = t0_scalar.expand(num_graphs)  # (num_graphs,) all same value

            def f(xpos, t_tensor):
                g["water"].pos = xpos
                return self.model(g, t_tensor)

            k1 = f(x, t0)
            k2 = f(x + 0.5 * dt * k1, (t0_scalar + 0.5 * dt).expand(num_graphs))
            k3 = f(x + 0.5 * dt * k2, (t0_scalar + 0.5 * dt).expand(num_graphs))
            k4 = f(x + dt * k3, (t0_scalar + dt).expand(num_graphs))

            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            if return_trajectory:
                for traj, pos in zip(
                    trajectories, self._split_waters(x, batch_w_cpu, num_graphs)
                ):
                    traj.append(pos)

        # split results by graph
        x_cpu = x.detach().cpu()
        protein_pos_cpu = g["protein"].pos.detach().cpu()
        x1_true_cpu = x1_true.detach().cpu()
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
        device: str = "cuda",
    ) -> np.ndarray | list[np.ndarray]:
        """
        Sample water positions for one or more graphs.

        Args:
            graphs: Single HeteroData or list of HeteroData graphs
            num_steps: Number of integration steps
            method: 'euler' or 'rk4'
            device: Device to run on

        Returns:
            If single graph input: (Nw, 3) predicted water positions
            If list input: List of (Nw_i, 3) predicted water positions
        """
        single_input = isinstance(graphs, HeteroData)

        if method == "euler":
            results = self.euler_integrate(graphs, num_steps, device=device)
            results = [r["water_pred"] for r in results]
        elif method == "rk4":
            results = self.rk4_integrate(
                graphs, num_steps, device=device, return_trajectory=False
            )
            results = [r["water_pred"] for r in results]
        else:
            raise ValueError(f"Unknown method: {method}")

        # return single array if single graph was provided
        if single_input:
            return results[0]
        return results
