# encoder.py

from __future__ import annotations

from pathlib import Path
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F 

from typing import Callable, Literal, Optional, Tuple, Any, Dict

from torch_geometric.data import Batch, Data
from torch_scatter import scatter_add, scatter_mean, scatter_max

from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import yaml

from src.gvp import EdgeUpdate, GVP, GVPConvLayer
import e3nn
from e3nn.math import soft_one_hot_linspace

from src.utils import rbf as _rbf

def _edge_vectors(pos: torch.Tensor, edge_index: torch.Tensor):
    src, dst = edge_index[0], edge_index[1]
    vec = pos[dst] - pos[src]
    rij = torch.linalg.norm(vec, dim=-1).clamp(min=1e-4) 
    r_hat = vec / rij[:, None]
    return rij, r_hat


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
        pool_residue: bool = False,
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
        d, u = _edge_vectors(pos, edge_index)
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
    device: str = "cuda",
    freeze: bool = True,
    default_hidden_dims: Tuple[int, int] = (256, 64),
    default_pooled_dim: int = 128,
    default_num_edge_rbf: int = 16,
    default_radius: float = 8.0,
    default_max_neighbors: int = 64,
) -> Tuple[ProteinGVPEncoder, Dict[str, Any]]:
    """
    Load pretrained ProteinGVPEncoder from SLAE checkpoint.
    Falls back to blank encoder if checkpoint doesn't exist or fails to load.
    """
    args = {
        "hidden_dims": list(default_hidden_dims),
        "pooled_dim": default_pooled_dim,
        "num_edge_rbf": default_num_edge_rbf,
        "radius": default_radius,
        "max_neighbors": default_max_neighbors,
    }
    
    loaded = False
    state_dict = None
    
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            args = ckpt.get("args", args)
            state_dict = ckpt.get("encoder", ckpt)
            loaded = True
            print(f"Loaded encoder checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint {checkpoint_path}: {e}")
            print("Initializing blank encoder instead.")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, initializing blank encoder.")

    hidden_dims = tuple(args.get("hidden_dims", list(default_hidden_dims)))
    
    encoder = ProteinGVPEncoder(
        node_scalar_in=16,
        hidden_dims=hidden_dims,
        edge_scalar_in=args.get("num_edge_rbf", default_num_edge_rbf),
        edge_vec_in=1,
        edge_scalar_out=16,
        update_w_distance=True,
        pooled_dim=args.get("pooled_dim", default_pooled_dim),
        radius=args.get("radius", default_radius),
        max_neighbors=args.get("max_neighbors", default_max_neighbors),
        num_edge_rbf=args.get("num_edge_rbf", default_num_edge_rbf),
    ).to(device)

    if loaded and state_dict is not None:
        try:
            encoder.load_state_dict(state_dict)
        except Exception as e:
            print(f"Failed to load state dict: {e}")
            print("Using randomly initialized weights.")

    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()

    return encoder, args


class FlowEncoder(nn.Module):
    """
    Wrapper that loads pretrained encoder and provides forward pass
    for combined protein+mate homogeneous graphs.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder, self.args = load_encoder_from_checkpoint(
            checkpoint_path, device, freeze
        )
        self.freeze = freeze
        self.hidden_dims = self.encoder.hidden_dims
        self.pooled_dim = self.args.get("pooled_dim", 128)
        
    @property
    def device(self):
        return next(self.encoder.parameters()).device
        
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (scalar, vector) tuple before pooling.
        
        Args:
            data: PyG Data with pos, x, edge_index
            
        Returns:
            Tuple of (scalar_features, vector_features) at atom level
        """
        with torch.set_grad_enabled(not self.freeze):
            x_scalar = self.encoder.input_scalar_encoder(data.x)
            node_features = self.encoder._initial_node_tuple(x_scalar)
            x = self.encoder.input_gvp(node_features)
            edge_attr, dist_feat = self.encoder._compute_edge_attr(
                data.pos, data.edge_index
            )
            
            for layer in self.encoder.layers:
                x = layer(x, data.edge_index, edge_attr)
                edge_attr = self.encoder.edge_update(
                    node_tuple=x,
                    edge_index=data.edge_index,
                    edge_attr=edge_attr,
                    distance_feat=(dist_feat if self.encoder.update_w_distance else None),
                )
            
            return x  # (s, V) tuple
    
    def forward_pooled(self, data: Data) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze):
            return self.encoder(data)


