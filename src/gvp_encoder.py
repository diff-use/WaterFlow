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

from src.constants import EDGE_PP, NODE_FEATURE_DIM, NUM_RBF, RBF_CUTOFF
from src.encoder_base import BaseProteinEncoder, register_encoder
from src.gvp import EdgeUpdate, GVP, GVPConvLayer
from src.utils import rbf


# Type aliases for GVP feature tuples
type GVPTuple = tuple[torch.Tensor, torch.Tensor]
"""(scalar, vector) feature pair. Scalar: (N, dim), Vector: (N, dim, 3)."""

type NodeFeatures = GVPTuple
"""Node (scalar, vector) features from GVP layers."""

type EdgeAttr = GVPTuple
"""Edge (scalar, vector) attributes."""


def edge_vectors(
    pos: torch.Tensor, edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute edge distances and unit vectors from node positions.

    Args:
        pos: (N, 3) node position coordinates
        edge_index: (2, E) edge indices with source in row 0, destination in row 1

    Returns:
        rij: (E,) edge distances, clamped to minimum 1e-4 to avoid division by zero
        r_hat: (E, 3) unit vectors pointing from source to destination,
               computed as vec / rij (using clamped distances)
    """
    src, dst = edge_index[0], edge_index[1]
    vec = pos[dst] - pos[src]
    rij = torch.linalg.norm(vec, dim=-1).clamp(min=1e-4)
    r_hat = vec / rij[:, None]
    return rij, r_hat


def make_gvp_encoder_data(data: HeteroData) -> Data:
    """
    Build homogeneous Data from HeteroData for GVP encoder.

    Extracts protein subgraph. Can be used independently of model instantiation.

    Args:
        data: HeteroData with protein nodes

    Returns:
        enc_data: Data with x, pos, edge_index, and optionally cached edge features
    """
    device = data["protein"].pos.device
    prot = data["protein"]

    x = prot.x
    pos = prot.pos

    # protein-protein edges
    if EDGE_PP in data.edge_types:
        edge_index = data[EDGE_PP].edge_index
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

    enc_data = Data(x=x, pos=pos, edge_index=edge_index)

    # Copy cached edge features if available
    if EDGE_PP in data.edge_types:
        pp_edge = data[EDGE_PP]
        if hasattr(pp_edge, "edge_rbf"):
            enc_data.edge_rbf = pp_edge.edge_rbf
        if hasattr(pp_edge, "edge_unit_vectors"):
            enc_data.edge_unit_vectors = pp_edge.edge_unit_vectors

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
        node_scalar_in: int = NODE_FEATURE_DIM,
        node_vec_in: int = 1,
        hidden_dims: tuple[int, int] = (256, 32),
        n_edge_scalar_in: int = NUM_RBF,
        n_edge_vec_in: int = 1,
        n_edge_scalar_out: int = NUM_RBF,
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
        update_w_distance_features: bool = True,
        distance_dim: int | None = None,
        radius: float = RBF_CUTOFF,
        num_edge_rbf: int = NUM_RBF,
        use_edge_update: bool = True,
    ):
        """
        Initialize GVP encoder for protein structure processing.

        Args:
            node_scalar_in: Number of input scalar features (e.g., 16 for element one-hot)
            node_vec_in: Number of input vector channels (typically 1 for orientation)
            hidden_dims: (scalar_dim, vector_dim) hidden layer dimensions
            n_edge_scalar_in: Number of input edge scalar features (RBF features)
            n_edge_vec_in: Number of input edge vector channels (e.g., 1 for unit displacement vectors)
            n_edge_scalar_out: Number of output edge scalar features
            n_layers: Number of GVP convolution layers
            n_message: Number of GVPs in message function
            n_feedforward: Number of GVPs in feedforward function
            drop_rate: Dropout rate for regularization
            vector_gate: Whether to use vector gating in GVP layers
            scalar_activation: Activation function for scalar channels
            vector_activation: Activation function for vector gating
            init_vec_zero: If True, initialize input vectors as zeros
            pooled_dim: Output dimension when pooling by residue
            pool_residue: If True, pool atom features to residue level
            pool_aggr: Aggregation method for residue pooling ('mean', 'sum', 'max')
            update_w_distance_features: Include distance features in edge updates
            distance_dim: Dimension for distance conditioning, defaults to edge_scalar_in
            radius: Distance cutoff in Angstroms for RBF encoding
            num_edge_rbf: Number of RBF basis functions
            use_edge_update: Whether to update edge features between layers
        """
        super().__init__()
        self.node_scalar_in = node_scalar_in
        self.node_vec_in = node_vec_in
        self.hidden_dims = hidden_dims
        self.n_edge_scalar_in = n_edge_scalar_in
        self.n_edge_vec_in = n_edge_vec_in
        self.n_edge_scalar_out = n_edge_scalar_out
        self.n_layers = n_layers
        self.drop_rate = drop_rate
        self.vector_gate = vector_gate
        self.init_vec_zero = init_vec_zero
        self.pool_residue = pool_residue
        self.pool_aggr = pool_aggr
        self.update_w_distance_features = update_w_distance_features
        self.radius = radius
        self.num_edge_rbf = num_edge_rbf
        self.pooled_dim = pooled_dim
        self.use_edge_update = use_edge_update

        distance_dim = distance_dim or n_edge_scalar_in
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

        self.s_edge_width = n_edge_scalar_out
        if n_edge_scalar_in != n_edge_scalar_out:
            self.edge_in_proj = nn.Linear(
                n_edge_scalar_in, n_edge_scalar_out, bias=False
            )
        else:
            self.edge_in_proj = nn.Identity()

        edge_dims = (self.s_edge_width, n_edge_vec_in)
        self.layers = nn.ModuleList(
            [
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
            ]
        )

        if use_edge_update:
            self.edge_update = EdgeUpdate(
                n_node_scalars=S_hid,
                s_edge_width=self.s_edge_width,
                update_w_distance_features=update_w_distance_features,
                distance_dim=distance_dim,
            )
        else:
            self.edge_update = None

        self.atom_readout = nn.Sequential(
            nn.Linear(S_hid + V_hid, pooled_dim),
            nn.LayerNorm(pooled_dim),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(pooled_dim, pooled_dim),
        )

    @staticmethod
    def _tuple_to_scalar_dense(x_tuple: tuple) -> torch.Tensor:
        """
        Convert GVP tuple to dense scalar representation.

        Computes L2 norms of vector features and concatenates with scalars.

        Args:
            x_tuple: (s, V) where s is (N, scalar_dim) and V is (N, vector_dim, 3)

        Returns:
            (N, scalar_dim + vector_dim) concatenation of scalars and vector L2 norms
        """
        s, V = x_tuple
        vnorm = torch.linalg.norm(V, dim=-1)
        return torch.cat([s, vnorm], dim=-1)

    def _pool_by_residue(
        self, atom_embed: torch.Tensor, residue_index: torch.Tensor, num_residues: int
    ) -> torch.Tensor:
        """
        Pool atom-level embeddings to residue level.

        Args:
            atom_embed: (N_atoms, embed_dim) atom embeddings
            residue_index: (N_atoms,) residue index per atom
            num_residues: Total number of residues

        Returns:
            (num_residues, embed_dim) pooled residue embeddings
        """
        aggr = self.pool_aggr
        if aggr == "mean":
            return scatter_mean(atom_embed, residue_index, dim=0, dim_size=num_residues)
        elif aggr == "sum":
            return scatter_add(atom_embed, residue_index, dim=0, dim_size=num_residues)
        elif aggr == "max":
            out, _ = scatter_max(
                atom_embed, residue_index, dim=0, dim_size=num_residues
            )
            return out
        else:
            raise ValueError(f"Unknown pool_aggr={aggr}")

    @staticmethod
    def _initial_node_tuple(
        x_scalar: torch.Tensor, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros(
            x_scalar.size(0), 1, 3, device=x_scalar.device if device is None else device
        )
        return (x_scalar, zeros)

    def _compute_edge_attr(self, data: Batch):
        """
        Build edge attributes from positions or cached features.

        If cached edge features (edge_rbf, edge_unit_vectors) are both available in data,
        use them directly. Otherwise, compute from positions.

        Args:
            data: Batch with pos, edge_index, and optionally edge_rbf, edge_unit_vectors

        Returns:
            (s_edge, V_edge): Tuple of scalar and vector edge features
            s_edge_raw: Raw RBF features (for distance conditioning)
        """
        # Use cached features if available
        if hasattr(data, "edge_rbf") and hasattr(data, "edge_unit_vectors"):
            s_edge_raw = data.edge_rbf
            u = data.edge_unit_vectors
        else:
            # Fallback: compute from positions
            rij, u = edge_vectors(data.pos, data.edge_index)
            s_edge_raw = rbf(rij, num_gaussians=self.num_edge_rbf, cutoff=self.radius)

        s_edge = self.edge_in_proj(s_edge_raw)
        V_edge = u.unsqueeze(1)
        return (s_edge, V_edge), s_edge_raw

    def forward(self, data: Batch) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """
        Forward pass through the GVP encoder.

        Args:
            data: PyG Batch/Data object with required attributes:
                - x: (N, node_scalar_in) node scalar features
                - pos: (N, 3) node position coordinates
                - edge_index: (2, E) edge indices
                Optional cached edge features (if absent, computed from pos):
                - edge_rbf: (E, num_rbf) RBF distance features
                - edge_unit_vectors: (E, 3) unit edge vectors

        Returns:
            s: (N, scalar_dim) scalar node features
            V: (N, vector_dim, 3) vector node features (empty tensor if pooling)
            edge_attr: tuple (s_edge, V_edge) of edge features, or None if pooling or edge updates disabled
        """
        x_scalar = self.input_scalar_encoder(data.x)

        if self.init_vec_zero or not hasattr(data, "node_v"):
            node_features = self._initial_node_tuple(x_scalar)
        else:
            node_features = (x_scalar, data.node_v.unsqueeze(1))

        x = self.input_gvp(node_features)
        edge_attr, dist_feat = self._compute_edge_attr(data)

        for layer in self.layers:
            x = layer(x, data.edge_index, edge_attr)
            if self.edge_update is not None:
                edge_attr = self.edge_update(
                    node_tuple=x,
                    edge_index=data.edge_index,
                    edge_attr=edge_attr,
                    distance_feat=(
                        dist_feat if self.update_w_distance_features else None
                    ),
                )

        if self.pool_residue:
            if not (hasattr(data, "residue_index") and hasattr(data, "num_residues")):
                raise ValueError(
                    "Pooling requires data.residue_index and data.num_residues"
                )
            atom_dense = self._tuple_to_scalar_dense(x)
            atom_embed = self.atom_readout(atom_dense)
            res_embed = self._pool_by_residue(
                atom_embed, data.residue_index, int(data.num_residues)
            )
            # Return empty V tensor (N, 0, 3) to match 3-tuple signature
            return res_embed, res_embed.new_empty(res_embed.size(0), 0, 3), None

        # Return edge_attr only if edge_update was used
        final_edge_attr = edge_attr if self.edge_update is not None else None
        return x[0], x[1], final_edge_attr


def load_encoder_from_checkpoint(
    checkpoint_path: str,
    node_scalar_in: int | None = None,
    device: str = "cuda",
    default_hidden_dims: tuple[int, int] = (256, 64),
    default_pooled_dim: int = 128,
    default_num_edge_rbf: int = NUM_RBF,
    default_radius: float = RBF_CUTOFF,
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
                    raise ValueError(
                        "node_scalar_in not in checkpoint and not provided"
                    )
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
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}, initializing blank encoder."
        )

    if node_scalar_in is None:
        raise ValueError("node_scalar_in required when checkpoint doesn't exist")

    hidden_dims = tuple(args.get("hidden_dims", list(default_hidden_dims)))

    encoder = ProteinGVPEncoder(
        node_scalar_in=node_scalar_in,
        hidden_dims=hidden_dims,
        n_edge_scalar_in=args.get("num_edge_rbf", default_num_edge_rbf),
        n_edge_vec_in=1,
        n_edge_scalar_out=NUM_RBF,
        update_w_distance_features=True,
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


@register_encoder("gvp")
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
        return "gvp"

    def forward(
        self, data: HeteroData
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | None]:
        """
        Encode protein data.

        Args:
            data: HeteroData with protein nodes

        Returns:
            s: (N, scalar_dim) scalar features
            V: (N, vector_dim, 3) vector features
            pp_edge_attr: tuple (s_edge, V_edge) for PP edges, or None if edge updates disabled
        """
        # Convert HeteroData to homogeneous Data for GVP encoder
        enc_data = make_gvp_encoder_data(data)

        with torch.set_grad_enabled(not self._freeze):
            s, V, edge_attr = self.encoder(enc_data)

        return s, V, edge_attr

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
                - use_edge_update: Whether to use edge updates (default: True)
            device: Device to place the encoder on

        Returns:
            Instantiated GVPEncoder
        """
        encoder_ckpt = config.get("encoder_ckpt")
        node_scalar_in = config.get("node_scalar_in", 16)
        hidden_s = config.get("hidden_s", 256)
        hidden_v = config.get("hidden_v", 32)
        freeze = config.get("freeze_encoder", False)
        use_edge_update = config.get("use_edge_update", True)

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
                n_edge_scalar_in=NUM_RBF,
                use_edge_update=use_edge_update,
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
            n_edge_scalar_in=args.get("num_edge_rbf", NUM_RBF),
            n_edge_vec_in=1,
            n_edge_scalar_out=NUM_RBF,
            update_w_distance_features=True,
            pooled_dim=args.get("pooled_dim", 128),
            radius=args.get("radius", RBF_CUTOFF),
            num_edge_rbf=args.get("num_edge_rbf", NUM_RBF),
        ).to(device)

        encoder.load_state_dict(state_dict)
        return cls(encoder=encoder, freeze=freeze)
