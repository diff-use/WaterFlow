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
from torch import nn, Tensor
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import knn
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from src.constants import ALL_EDGE_TYPES, EDGE_PP, EDGE_PW, EDGE_WP, EDGE_WW, NUM_RBF
from src.encoder_base import BaseProteinEncoder
from src.gvp import GVP, GVPMultiEdgeConv
from src.utils import ot_coupling


def build_knn_edges(
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
    k: int,
    batch_src: torch.Tensor | None = None,
    batch_dst: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    KNN edges from src -> dst (source indices in row 0, dest in row 1).
    """
    if src_pos.numel() == 0 or dst_pos.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=src_pos.device)

    idx = knn(x=dst_pos, y=src_pos, k=k, batch_x=batch_dst, batch_y=batch_src)

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
                cached edge direction unit vectors (edge_unit, pre-normalized at preprocessing).
                If None, uses cached geometric edge features (edge_rbf, edge_unit) from the dataset.

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

            if pp_edge_attr is not None:
                # Use encoder-learned scalar features with original cached unit vectors
                # This preserves gradient flow through learned edge representations
                s_edge, V_edge = pp_edge_attr
                if hasattr(pp_edge, "edge_unit"):
                    cached_edge_attr_dict[EDGE_PP] = (s_edge, pp_edge.edge_unit)
                else:
                    # Fall back to full encoder output if no cached unit vectors
                    cached_edge_attr_dict[EDGE_PP] = (s_edge, V_edge.squeeze(1))
            elif hasattr(pp_edge, "edge_rbf") and hasattr(pp_edge, "edge_unit"):
                # Use cached geometric edge features
                cached_edge_attr_dict[EDGE_PP] = (pp_edge.edge_rbf, pp_edge.edge_unit)

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
            hidden_dims: (scalar_dim, vector_dim) hidden dimensions for flow model
            edge_scalar_dim: Dimension of edge scalar features (typically NUM_RBF)
            layers: Number of heterogeneous GVP message passing layers
            drop_rate: Dropout rate for regularization
            n_message_gvps: Number of GVP modules in each edge-type's message function
                (distinct from `layers` which controls message-passing iterations)
            n_update_gvps: Number of GVP modules in the node update function
                (applied after aggregating messages from all edge types)
            vector_gate: Whether to use vector gating in GVP layers
            k_pw: K nearest neighbors for protein-water edges
            k_ww: K nearest neighbors for water-water edges
            k_wp: K nearest neighbors for water-protein edges
            water_input_dim: Input dimension for water node features (element one-hot)
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
        if self_cond is not None and ("x1_pred" in self_cond) and self_cond["x1_pred"] is not None:
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

    def __init__(
        self,
        model,
        p_self_cond: float = 0.5,
        use_distortion: bool = False,
        p_distort: float = 0.2,
        t_distort: float = 0.5,
        sigma_distort: float = 0.5,
        loss_eps: float = 1e-3,
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
        """
        self.model = model
        self.p_self_cond = p_self_cond
        self.use_distortion = use_distortion
        self.p_distort = p_distort
        self.t_distort = t_distort
        self.sigma_distort = sigma_distort
        self.loss_eps = loss_eps

    @staticmethod
    def compute_sigma(data: HeteroData) -> float:
        """
        Compute noise scale sigma as standard deviation of protein coordinates.

        Args:
            data: HeteroData with protein node positions

        Returns:
            Scalar sigma value (standard deviation across all protein coordinates)
        """
        pos = data["protein"].pos
        return float(pos.std().item())

    @staticmethod
    def compute_sigma_per_graph(
        data: HeteroData | Batch, device: torch.device
    ) -> torch.Tensor:
        """
        Compute sigma (std of protein coordinates) per graph in a batch.

        Returns:
            sigma: (num_graphs,) tensor of sigma values per graph
        """
        pos = data["protein"].pos  # (N_total, 3)
        batch_p = data["protein"].batch  # (N_total,)

        # Var(X) = E[X^2] - E[X]^2
        mean_pos = scatter_mean(pos, batch_p, dim=0)  # (num_graphs, 3)
        mean_sq = scatter_mean(pos**2, batch_p, dim=0)  # (num_graphs, 3)
        var_per_dim = mean_sq - mean_pos**2  # (num_graphs, 3)
        sigma = torch.sqrt(var_per_dim.mean(dim=-1).clamp(min=1e-8))  # (num_graphs,)

        return sigma

    def training_step(
        self,
        batch: HeteroData,
        use_self_conditioning: bool = True,
        accumulation_steps: int = 1,
    ) -> dict[str, float]:
        """
        Single flow matching training step (forward + backward only).

        The optimizer step is handled by the caller to support gradient accumulation.

        Args:
            batch: HeteroData batch
            use_self_conditioning: Whether to use self-conditioning
            accumulation_steps: Number of gradient accumulation steps (loss is scaled by 1/accumulation_steps)

        Returns dict with 'loss', 'rmsd', 'sigma'.
        """
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")

        self.model.train()
        device = batch["protein"].pos.device

        x1 = batch["water"].pos
        batch_w = batch["water"].batch
        batch_p = batch["protein"].batch
        num_graphs = int(batch_p.max().item()) + 1

        sigma = self.compute_sigma(batch)

        x0 = torch.randn_like(x1) * sigma
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

        # check for high loss and compute per-sample losses for debugging
        per_sample_info = None
        if loss.item() > 100.0:
            with torch.no_grad():
                from torch_scatter import scatter_add

                weighted_mse = (w * per_atom_mse).squeeze(-1)
                # compute per-graph loss: sum(weighted_mse) / sum(w) for each graph
                numerator = scatter_add(weighted_mse, batch_w, dim=0)
                denominator = scatter_add(w.squeeze(-1), batch_w, dim=0)
                per_sample_loss = numerator / (denominator + 1e-8)
                per_sample_info = {"losses": per_sample_loss, "num_graphs": num_graphs}

        # backward (scale loss for gradient accumulation)
        (loss / accumulation_steps).backward()

        # training RMSD
        with torch.no_grad():
            x1_hat = x_t + (1.0 - t_per_atom) * v_pred
            # rmsd = compute_rmsd(x1_hat, x1_star)

            # on-gpu version of rmsd
            diff2 = ((x1_hat - x1_star) ** 2).sum(-1)  # (Nw,)
            rmsd = torch.sqrt(scatter_mean(diff2, batch_w, dim=0)).mean().item()

        return {
            "loss": loss.item(),
            "rmsd": rmsd,
            "sigma": sigma,
            "per_sample_info": per_sample_info,
        }

    @torch.no_grad()
    def validation_step(self, batch: HeteroData) -> dict[str, float]:
        """
        Run single validation step without gradients.

        Args:
            batch: HeteroData batch with protein and water nodes

        Returns:
            Dict with 'loss' and 'rmsd' metrics
        """
        self.model.eval()
        device = batch["protein"].pos.device

        x1 = batch["water"].pos
        batch_w = batch["water"].batch
        batch_p = batch["protein"].batch
        num_graphs = int(batch_p.max().item()) + 1

        sigma = self.compute_sigma(batch)
        x0 = torch.randn_like(x1) * sigma
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
        rmsd = torch.sqrt(scatter_mean(diff2, batch_w, dim=0)).mean().item()

        return {"loss": loss.item(), "rmsd": rmsd}

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
        num_graphs = num_residues.size(0)

        # compute waters per graph: num_residues * ratio, minimum 1
        num_waters = (num_residues.float() * water_ratio).long().clamp(min=1)

        # create batch indices (vectorized)
        batch_w = torch.repeat_interleave(
            torch.arange(num_graphs, device=device), num_waters
        )
        total_waters = batch_w.size(0)

        # compute sigma per graph and expand to per-water
        sigma_per_graph = self.compute_sigma_per_graph(g, device)
        sigma_per_water = sigma_per_graph[batch_w]

        # sample noise
        x = torch.randn(total_waters, 3, device=device) * sigma_per_water.unsqueeze(-1)

        # create water features (oxygen one-hot, index 2 for 'O' in ELEMENT_VOCAB)
        water_x = torch.zeros(total_waters, 16, device=device)
        water_x[:, 2] = 1.0  # oxygen is index 2 in ELEMENT_VOCAB

        # update graph with new water nodes
        g["water"].pos = x
        g["water"].x = water_x
        g["water"].batch = batch_w
        g["water"].num_nodes = total_waters

        return x, batch_w

    @torch.no_grad()
    def euler_integrate(
        self,
        graphs: HeteroData | list[HeteroData],
        num_steps: int = 100,
        use_sc: bool = True,
        sc_ema_alpha: float = 0.2,
        device: str | torch.device = "cuda",
        water_ratio: float | None = None,
    ) -> list[np.ndarray]:
        """
        Euler integration from noise to final positions.

        Args:
            graphs: Single HeteroData or list of HeteroData graphs to process
            num_steps: Number of integration steps
            use_sc: Whether to use self-conditioning
            sc_ema_alpha: EMA decay for self-conditioning
            device: Device to run on
            water_ratio: If provided, sample num_residues * water_ratio waters
                        instead of using ground truth water count

        Returns:
            List of (Nw_i, 3) predicted water positions for each input graph
        """
        self.model.eval()
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        # handle single graph input
        if isinstance(graphs, HeteroData):
            graphs = [graphs]

        # batch graphs together
        g = Batch.from_data_list([copy.deepcopy(graph) for graph in graphs]).to(device)

        if water_ratio is not None:
            # sample waters based on residue count
            x, batch_w = self._setup_water_nodes_from_ratio(g, water_ratio, device)
            num_graphs = g["protein"].num_residues.size(0)
        else:
            # use existing water nodes
            batch_w = g["water"].batch
            num_graphs = int(batch_w.max().item()) + 1
            sigma_per_graph = self.compute_sigma_per_graph(g, device)
            sigma_per_water = sigma_per_graph[batch_w]
            x = torch.randn(
                g["water"].num_nodes, 3, device=device
            ) * sigma_per_water.unsqueeze(-1)

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
        results = []
        for i in range(num_graphs):
            mask = batch_w.cpu() == i
            results.append(x_cpu[mask].numpy())

        return results

    @torch.no_grad()
    def rk4_integrate(
        self,
        graphs: HeteroData | list[HeteroData],
        num_steps: int = 500,
        use_sc: bool = True,
        sc_ema_alpha: float = 0.2,
        device: str | torch.device = "cuda",
        return_trajectory: bool = True,
        water_ratio: float | None = None,
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
                        instead of using ground truth water count

        Returns:
            List of dicts, one per input graph, each with keys:
                'protein_pos': (Np, 3) - includes both ASU and mate atoms
                'water_true': (Nw, 3) - None if water_ratio is used
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

        # store ground truth water positions and batch indices before modifying
        x1_true = g["water"].pos.clone()
        batch_w_true = g["water"].batch.clone()

        if water_ratio is not None:
            # sample waters based on residue count
            x, batch_w = self._setup_water_nodes_from_ratio(g, water_ratio, device)
            num_graphs = g["protein"].num_residues.size(0)
        else:
            # use existing water nodes
            batch_w = g["water"].batch
            num_graphs = int(batch_w.max().item()) + 1
            sigma_per_graph = self.compute_sigma_per_graph(g, device)
            sigma_per_water = sigma_per_graph[batch_w]
            x = torch.randn_like(x1_true) * sigma_per_water.unsqueeze(-1)

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
        
