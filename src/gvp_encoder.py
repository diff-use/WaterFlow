"""
GVP (Geometric Vector Perceptron) encoder implementation.

This encoder processes protein structure directly using GVP layers
to produce geometric features for the flow model.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_geometric.data import Batch, Data, HeteroData
from torch_scatter import scatter_add, scatter_max, scatter_mean

from src.constants import EDGE_PP
from src.encoder_base import BaseProteinEncoder, register_encoder
from src.gvp import GVP, EdgeUpdate, GVPConvLayer
from src.utils import compute_edge_geometry, rbf as _rbf


def make_encoder_data(data: HeteroData) -> Data:
    """
    Build a homogeneous Data with protein nodes for GVP encoder.

    Extracts protein subgraph from HeteroData for use with GVP encoder.
    Edge features are computed by the encoder itself.

    Args:
        data: HeteroData with protein nodes

    Returns:
        enc_data: Data with x, pos, edge_index
    """
    device = data['protein'].pos.device
    prot = data['protein']

    x = prot.x
    pos = prot.pos

    # protein-protein edges (topology only - features computed by encoder)
    if EDGE_PP in data.edge_types:
        edge_index = data[EDGE_PP].edge_index
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

    enc_data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
    )

    # batch for multi-complex batches
    if hasattr(prot, "batch"):
        enc_data.batch = prot.batch

    return enc_data


class ProteinGVPEncoder(nn.Module):
    """
    Core GVP encoder architecture for protein structures.

    This is the underlying encoder that processes protein graphs
    using GVP message passing layers.
    """

    def __init__(
        self,
        node_scalar_in: int = 17,
        node_vec_in: int = 1,
        hidden_dims: tuple[int, int] = (256, 32),
        edge_scalar_in: int = 16,
        edge_vec_in: int = 1,
        edge_scalar_out: int = 16,
        n_layers: int = 3,
        n_message: int = 2,
        n_feedforward: int = 2,
        drop_rate: float = 0.1,
        vector_gate: bool = True,
        scalar_activation: Callable = F.relu,
        vector_activation: Callable = torch.sigmoid,
        init_vec_zero: bool = True,
        pooled_dim: int = 128,
        pool_residue: bool = False,
        pool_aggr: Literal["mean", "sum", "max"] = "mean",
        update_w_distance: bool = True,
        distance_dim: int | None = None,
        radius: float = 8.0,
        num_edge_rbf: int = 16,
    ):
        super().__init__()
        self.node_scalar_in = node_scalar_in
        self.node_vec_in = node_vec_in
        self.hidden_dims = hidden_dims
        self.edge_scalar_in = edge_scalar_in
        self.edge_vec_in = edge_vec_in
        self.edge_scalar_out = edge_scalar_out
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.vector_gate = vector_gate
        self.init_vec_zero = init_vec_zero
        self.pool_residue = pool_residue
        self.pool_aggr = pool_aggr
        self.update_w_distance = update_w_distance
        self.radius = radius
        self.num_edge_rbf = num_edge_rbf
        self.pooled_dim = pooled_dim

        distance_dim = distance_dim or edge_scalar_in
        self.distance_dim = distance_dim

        activations = (scalar_activation, vector_activation)
        S_hid, V_hid = hidden_dims

        self.input_scalar_encoder = nn.Sequential(
            nn.Linear(node_scalar_in, node_scalar_in * 2),
            nn.LayerNorm(node_scalar_in * 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(node_scalar_in * 2, node_scalar_in),
        )

        self.input_gvp = GVP(
            in_dims=(node_scalar_in, node_vec_in),
            out_dims=hidden_dims,
            activations=activations,
            vector_gate=vector_gate,
        )

        self.s_edge_width = edge_scalar_out
        if edge_scalar_in != edge_scalar_out:
            self.edge_in_proj = nn.Linear(edge_scalar_in, edge_scalar_out, bias=False)
        else:
            self.edge_in_proj = nn.Identity()

        edge_dims = (self.s_edge_width, edge_vec_in)
        self.layers = nn.ModuleList([
            GVPConvLayer(
                node_dims=hidden_dims,
                edge_dims=edge_dims,
                n_message=n_message,
                n_feedforward=n_feedforward,
                drop_rate=drop_rate,
                activations=activations,
                vector_gate=vector_gate,
            )
            for _ in range(n_layers)
        ])

        self.edge_update = EdgeUpdate(
            n_node_scalars=S_hid,
            s_edge_width=self.s_edge_width,
            update_w_distance=update_w_distance,
            distance_dim=distance_dim,
        )

        self.atom_readout = nn.Sequential(
            nn.Linear(S_hid + V_hid, pooled_dim),
            nn.LayerNorm(pooled_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(pooled_dim, pooled_dim),
        )

    @staticmethod
    def _tuple_to_scalar_dense(x_tuple: tuple) -> torch.Tensor:
        s, V = x_tuple
        vnorm = torch.linalg.norm(V, dim=-1)
        return torch.cat([s, vnorm], dim=-1)

    def _pool_by_residue(
        self, atom_embed: torch.Tensor, residue_index: torch.Tensor, num_residues: int
    ) -> torch.Tensor:
        aggr = self.pool_aggr
        if aggr == "mean":
            return scatter_mean(atom_embed, residue_index, dim=0, dim_size=num_residues)
        if aggr == "sum":
            return scatter_add(atom_embed, residue_index, dim=0, dim_size=num_residues)
        if aggr == "max":
            out, _ = scatter_max(atom_embed, residue_index, dim=0, dim_size=num_residues)
            return out
        raise ValueError(f"Unknown pool_aggr={aggr!r}")

    @staticmethod
    def _initial_node_tuple(x_scalar: torch.Tensor, device=None) -> tuple:
        zeros = torch.zeros(x_scalar.size(0), 1, 3, device=x_scalar.device if device is None else device)
        return (x_scalar, zeros)

    def _compute_edge_attr(self, pos: torch.Tensor, edge_index: torch.Tensor):
        d, u = compute_edge_geometry(pos, edge_index)
        s_edge_raw = _rbf(d, num_gaussians=self.num_edge_rbf, cutoff=self.radius)
        s_edge = self.edge_in_proj(s_edge_raw)
        V_edge = u.unsqueeze(1)
        return (s_edge, V_edge), s_edge_raw

    def forward(self, data: Batch):
        x_scalar = self.input_scalar_encoder(data.x)

        if self.init_vec_zero or not hasattr(data, "node_v"):
            node_features = self._initial_node_tuple(x_scalar)
        else:
            node_features = (x_scalar, data.node_v.unsqueeze(1))

        x = self.input_gvp(node_features)
        edge_attr, dist_feat = self._compute_edge_attr(data.pos, data.edge_index)

        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr)
            edge_attr = self.edge_update(
                node_tuple=x,
                edge_index=data.edge_index,
                edge_attr=edge_attr,
                distance_feat=(dist_feat if self.update_w_distance else None),
            )

        if self.pool_residue:
            assert hasattr(data, "residue_index") and hasattr(data, "num_residues"), \
                "Pooling requires data.residue_index (N,) and data.num_residues (int)."
            atom_dense = self._tuple_to_scalar_dense(x)
            atom_embed = self.atom_readout(atom_dense)
            res_embed = self._pool_by_residue(atom_embed, data.residue_index, int(data.num_residues))
            return res_embed

        return x


def load_encoder_from_checkpoint(
    checkpoint_path: str,
    node_scalar_in: int | None = None,
    device: str = "cuda",
    default_hidden_dims: tuple[int, int] = (256, 64),
    default_pooled_dim: int = 128,
    default_num_edge_rbf: int = 16,
    default_radius: float = 8.0,
) -> tuple[ProteinGVPEncoder, dict[str, Any]]:
    """
    Load pretrained ProteinGVPEncoder from checkpoint.
    Falls back to blank encoder if checkpoint doesn't exist or fails to load.

    Args:
        checkpoint_path: Path to checkpoint file
        node_scalar_in: Input feature dimension. Read from checkpoint if not provided.
            Required if checkpoint doesn't exist (for fallback blank encoder).

    Returns:
        Tuple of (encoder, args_dict)
    """
    args = {
        "hidden_dims": list(default_hidden_dims),
        "pooled_dim": default_pooled_dim,
        "num_edge_rbf": default_num_edge_rbf,
        "radius": default_radius,
    }

    loaded = False
    state_dict = None

    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            args = ckpt.get("args", args)
            state_dict = ckpt.get("encoder", ckpt)
            loaded = True
            logger.info(f"Loaded encoder checkpoint from {checkpoint_path}")

            # use node_scalar_in from checkpoint if not specified
            if node_scalar_in is None:
                if "node_scalar_in" not in args:
                    raise ValueError("node_scalar_in not in checkpoint and not provided")
                node_scalar_in = args["node_scalar_in"]
            elif "node_scalar_in" in args and args["node_scalar_in"] != node_scalar_in:
                raise ValueError(
                    f"node_scalar_in mismatch: checkpoint has {args['node_scalar_in']}, "
                    f"but {node_scalar_in} was specified"
                )
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
            logger.info("Initializing blank encoder instead.")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}, initializing blank encoder.")

    if node_scalar_in is None:
        raise ValueError("node_scalar_in required when checkpoint doesn't exist")

    hidden_dims = tuple(args.get("hidden_dims", list(default_hidden_dims)))

    encoder = ProteinGVPEncoder(
        node_scalar_in=node_scalar_in,
        hidden_dims=hidden_dims,
        edge_scalar_in=args.get("num_edge_rbf", default_num_edge_rbf),
        edge_vec_in=1,
        edge_scalar_out=16,
        update_w_distance=True,
        pooled_dim=args.get("pooled_dim", default_pooled_dim),
        radius=args.get("radius", default_radius),
        num_edge_rbf=args.get("num_edge_rbf", default_num_edge_rbf),
    ).to(device)

    if loaded and state_dict is not None:
        try:
            encoder.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Failed to load state dict: {e}")
            logger.info("Using randomly initialized weights.")

    return encoder, args


@register_encoder('gvp')
class GVPEncoder(BaseProteinEncoder):
    """
    GVP encoder implementing the BaseProteinEncoder interface.

    This encoder wraps ProteinGVPEncoder to provide the standard interface
    expected by the flow model.
    """

    def __init__(
        self,
        encoder: ProteinGVPEncoder,
        freeze: bool = False,
    ):
        """
        Args:
            encoder: Underlying ProteinGVPEncoder instance
            freeze: If True, freeze encoder parameters
        """
        super().__init__()
        self.encoder = encoder
        self._freeze = freeze

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

    @property
    def output_dims(self) -> tuple[int, int]:
        """Return (scalar_dim, vector_dim)."""
        return self.encoder.hidden_dims

    @property
    def encoder_type(self) -> str:
        """Return encoder type identifier."""
        return 'gvp'

    def forward(self, data: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein data.

        Args:
            data: HeteroData with protein nodes

        Returns:
            Tuple of (s, V) features
        """
        # Convert HeteroData to homogeneous Data for GVP encoder
        enc_data = make_encoder_data(data)

        with torch.set_grad_enabled(not self._freeze):
            s, V = self.encoder(enc_data)

        return s, V

    @classmethod
    def from_config(cls, config: dict, device: torch.device) -> GVPEncoder:
        """
        Construct GVPEncoder from config dict.

        Args:
            config: Configuration dictionary with:
                - encoder_ckpt: Path to checkpoint (optional)
                - node_scalar_in: Input feature dimension (default: 16)
                - hidden_s, hidden_v: Hidden dimensions
                - freeze_encoder: Whether to freeze encoder
            device: Device to place the encoder on

        Returns:
            Instantiated GVPEncoder
        """
        encoder_ckpt = config.get('encoder_ckpt')
        node_scalar_in = config.get('node_scalar_in', 16)
        hidden_s = config.get('hidden_s', 256)
        hidden_v = config.get('hidden_v', 32)
        freeze = config.get('freeze_encoder', False)

        if encoder_ckpt:
            encoder, _ = load_encoder_from_checkpoint(
                encoder_ckpt,
                node_scalar_in=node_scalar_in,
                device=str(device),
            )
        else:
            encoder = ProteinGVPEncoder(
                node_scalar_in=node_scalar_in,
                hidden_dims=(hidden_s, hidden_v),
                edge_scalar_in=16,
            ).to(device)

        return cls(encoder=encoder, freeze=freeze)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        freeze: bool = True,
    ) -> GVPEncoder:
        """
        Load GVPEncoder from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to place encoder on
            freeze: Whether to freeze encoder parameters

        Returns:
            Instantiated GVPEncoder
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        args = ckpt["args"]
        state_dict = ckpt.get("encoder", ckpt)

        encoder = ProteinGVPEncoder(
            node_scalar_in=args["node_scalar_in"],
            hidden_dims=tuple(args["hidden_dims"]),
            edge_scalar_in=args.get("num_edge_rbf", 16),
            edge_vec_in=1,
            edge_scalar_out=16,
            update_w_distance=True,
            pooled_dim=args.get("pooled_dim", 128),
            radius=args.get("radius", 8.0),
            num_edge_rbf=args.get("num_edge_rbf", 16),
        ).to(device)

        encoder.load_state_dict(state_dict)
        return cls(encoder=encoder, freeze=freeze)
