# loading in encoder model from either OG SLAE or self implementation

"""
For OG SLAE implemenatation:
- if weights_frozen == True, load in the encoder from the checkpoint once, and cache the protein embeddings as a protein atom node feature -- one forward pass per protein for the whole training run
- if weights_frozen == False, load in the encoder from the checkpoint before first forward pass, and fine-tune the encoder weights during training -- one forward pass per protein per training step

Currently, the learned embeddings will be paired with GVP based flow classes. Idea to try would be to make the flow model also based on SLAE architecture, so that the entire model is SLAE based.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 

from typing import Callable, Literal, Optional, Tuple

from torch_geometric.data import Batch, Data
from torch_scatter import scatter_add, scatter_mean, scatter_max

import math
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import yaml

from gvp import EdgeUpdate, GVP, GVPConvLayer
import e3nn

def _edge_vectors(pos: torch.Tensor, edge_index: torch.Tensor):
    src, dst = edge_index[0], edge_index[1]
    vec = pos[dst] - pos[src]
    rij = torch.linalg.norm(vec, dim=-1)
    r_hat = torch.where(rij[:, None] > 0, vec / rij[:, None], torch.zeros_like(vec))
    return rij, r_hat

def _rbf(d: torch.Tensor, number: int, cutoff: float) -> torch.Tensor:
    return e3nn.math.soft_one_hot_linspace(
        d, start=0.0, end=cutoff, number=number, basis="bessel", cutoff=True
    )

def load_yaml_dict(path: Union[str, Path]) -> dict:
    """Load YAML config, removing hydra _target_ if present."""
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"YAML at {path} did not parse to a dict.")
    d.pop("_target_", None)
    return d

class EncoderType(Enum):
    GVP = "gvp"
    SLAE = "slae"

class ProteinGVPEncoder(nn.Module):
    def __init__(
        self,
        node_scalar_in: int = 16,
        node_vec_in: int = 1,
        hidden_dims: Tuple[int, int] = (256, 32),
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
        pool_residue: bool = True,
        pool_aggr: Literal["mean", "sum", "max"] = "mean",
        update_w_distance: bool = True,
        distance_dim: Optional[int] = None,
        radius: float = 8.0,
        max_neighbors: Optional[int] = 256,
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
        d, u = _edge_vectors(pos, edge_index)
        s_edge_raw = _rbf(d, number=self.num_edge_rbf, cutoff=self.radius)
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

class BaseProteinEncoder(ABC, nn.Module):
    """Abstract encoder interface for FlowWaterGVP."""

    @property
    @abstractmethod
    def output_dims(self) -> Tuple[int, int]:
        """Return (scalar_dim, vector_dim) of encoder output."""
        pass

    @abstractmethod
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode protein+mate graph.
        
        Args:
            data: PyG Data with x, pos, edge_index, edge_rbf, edge_unit_vec
            
        Returns:
            (s, v): scalar features (N, S), vector features (N, V, 3)
        """
        pass

class GVPEncoderWrapper(BaseProteinEncoder):
    """Wraps ProteinGVPEncoder to conform to BaseProteinEncoder interface."""

    def __init__(self, encoder: ProteinGVPEncoder):
        super().__init__()
        self.encoder = encoder
        # Ensure atom-level output
        self.encoder.pool_residue = False

    @property
    def output_dims(self) -> Tuple[int, int]:
        return self.encoder.hidden_dims

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(data)
    
# get scalar and vector features from SLAE?


