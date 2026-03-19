"""
Geometric Vector Perceptron (GVP) layers adapted from Jing et al. (2021).

This module provides:
- Core GVP operations: GVP, LayerNorm, Dropout
- Graph convolution: GVPConv, GVPConvLayer
- Edge updates: EdgeUpdate
- Multi-edge heterogeneous convolution: GVPMultiEdge, GVPMultiEdgeConv
- Utility functions: tuple_sum, tuple_cat, tuple_index, randn
"""

import functools

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_scatter import scatter_add

from src.constants import NUM_RBF, RBF_CUTOFF
from src.utils import compute_edge_features


def tuple_sum(*args):
    """
    Sum any number of GVP tuples (s, V) elementwise.

    Args:
        *args: Variable number of (s, V) tuples where s is scalar tensor
            and V is vector tensor

    Returns:
        (s_sum, V_sum) tuple with elementwise sums
    """
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    """
    Concatenates any number of tuples (s, V) elementwise.

    :param dim: dimension along which to concatenate when viewed
                as the `dim` index for the scalar-channel tensors.
                This means that `dim=-1` will be applied as
                `dim=-2` for the vector-channel tensors.
    """
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def tuple_index(x, idx):
    """
    Indexes into a tuple (s, V) along the first dimension.

    :param idx: any object which can be used to index into a `torch.Tensor`
    """
    return x[0][idx], x[1][idx]


def randn(n, dims, device="cpu"):
    """
    Returns random tuples (s, V) drawn elementwise from a normal distribution.

    :param n: number of data points
    :param dims: tuple of dimensions (n_scalar, n_vector)

    :return: (s, V) with s.shape = (n, n_scalar) and
             V.shape = (n, n_vector, 3)
    """
    return torch.randn(n, dims[0], device=device), torch.randn(
        n, dims[1], 3, device=device
    )


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-6, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.

    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


def _split(x, nv):
    """
    Splits a merged representation of (s, V) back into a tuple.
    Should be used only with `_merge(s, V)` and only if the tuple
    representation cannot be used.

    :param x: the `torch.Tensor` returned from `_merge`
    :param nv: the number of vector channels in the input to `_merge`
    """
    v = torch.reshape(x[..., -3 * nv :], x.shape[:-1] + (nv, 3))
    s = x[..., : -3 * nv]
    return s, v


def _merge(s, v):
    """
    Merges a tuple (s, V) into a single `torch.Tensor`, where the
    vector channels are flattened and appended to the scalar channels.
    Should be used only if the tuple representation cannot be used.
    Use `_split(x, nv)` to reverse.
    """
    v = torch.reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    return torch.cat([s, v], -1)


class GVP(nn.Module):
    """
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim=None,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        """
        Initialize Geometric Vector Perceptron layer.

        Args:
            in_dims: (n_scalar_in, n_vector_in) input dimensions, where n_vector_in
                is the number of 3D vector channels (each vector has 3 components)
            out_dims: (n_scalar_out, n_vector_out) output dimensions, where n_vector_out
                is the number of 3D vector channels. E.g., n_vector_out=1 means
                output shape is (batch, 1, 3) - one vector channel with 3D coordinates.
            h_dim: Intermediate vector channel dimension, defaults to max(vi, vo)
            activations: (scalar_act, vector_act) activation functions
            vector_gate: If True, use vector gating; vector_act becomes sigma^+ gate
        """
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        """
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    """
    Vector channel dropout where the elements of each
    vector channel are dropped together.
    """

    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: `torch.Tensor` corresponding to vector channels
        """
        device = self.dummy_param.device
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=device)
        ).unsqueeze(-1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Module):
    """
    Combined dropout for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if type(x) is torch.Tensor:
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    """
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    """

    def __init__(self, dims):
        """
        Initialize combined layer normalization for GVP tuples.

        Args:
            dims: (n_scalar, n_vector) feature dimensions
        """
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor`
                  (will be assumed to be scalar channels)
        """
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / (vn + 1e-5)


class GVPConv(MessagePassing):
    """
    Graph convolution / message passing with Geometric Vector Perceptrons.
    Takes in a graph with node and edge embeddings,
    and returns new node embeddings.

    This does NOT do residual updates and pointwise feedforward layers
    ---see `GVPConvLayer`.

    :param in_dims: input node embedding dimensions (n_scalar, n_vector)
    :param out_dims: output node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_layers: number of GVPs in the message function
    :param module_list: preconstructed message function, overrides n_layers
    :param aggr: should be "add" if some incoming edges are masked, as in
                 a masked autoregressive decoder architecture, otherwise "mean"
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        edge_dims,
        n_layers=3,
        module_list=None,
        aggr="mean",
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        """
        Initialize GVP graph convolution layer.

        Args:
            in_dims: (n_scalar_in, n_vector_in) input node dimensions
            out_dims: (n_scalar_out, n_vector_out) output node dimensions
            edge_dims: (n_scalar_edge, n_vector_edge) edge feature dimensions
            n_layers: Number of GVPs in message function
            module_list: Pre-constructed message function, overrides n_layers
            aggr: Aggregation method ('mean' or 'add')
            activations: (scalar_act, vector_act) activation functions
            vector_gate: Whether to use vector gating in GVP layers
        """
        super().__init__(aggr=aggr)
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims

        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP_(
                        (2 * self.si + self.se, 2 * self.vi + self.ve),
                        (self.so, self.vo),
                        activations=(None, None),
                    )
                )
            else:
                module_list.append(
                    GVP_((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims)
                )
                for i in range(n_layers - 2):
                    module_list.append(GVP_(out_dims, out_dims))
                module_list.append(GVP_(out_dims, out_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*module_list)

    def forward(self, x, edge_index, edge_attr):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        """
        x_s, x_v = x
        message = self.propagate(
            edge_index,
            s=x_s,
            v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
            edge_attr=edge_attr,
        )
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        """
        Construct message from source to destination node.

        Args:
            s_i: (E,) destination scalar features (flattened for PyG)
            v_i: (E, 3*n_vector) destination vector features (flattened)
            s_j: (E,) source scalar features
            v_j: (E, 3*n_vector) source vector features (flattened)
            edge_attr: (s_edge, V_edge) edge feature tuple

        Returns:
            (E, s_out + 3*v_out) merged message tensor
        """
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Module):
    """
    Full graph convolution / message passing layer with
    Geometric Vector Perceptrons. Residually updates node embeddings with
    aggregated incoming messages, applies a pointwise feedforward
    network to node embeddings, and returns updated node embeddings.

    To only compute the aggregated messages, see `GVPConv`.

    :param node_dims: node embedding dimensions (n_scalar, n_vector)
    :param edge_dims: input edge embedding dimensions (n_scalar, n_vector)
    :param n_message: number of GVPs to use in message function
    :param n_feedforward: number of GVPs to use in feedforward function
    :param drop_rate: drop probability in all dropout layers
    :param autoregressive: if `True`, this `GVPConvLayer` will be used
           with a different set of input node embeddings for messages
           where src >= dst
    :param activations: tuple of functions (scalar_act, vector_act) to use in GVPs
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1,
        autoregressive=False,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        """
        Initialize full GVP convolution layer with residual and feedforward.

        Args:
            node_dims: (n_scalar, n_vector) node feature dimensions
            edge_dims: (n_scalar_edge, n_vector_edge) edge feature dimensions
            n_message: Number of GVPs in message function
            n_feedforward: Number of GVPs in feedforward function
            drop_rate: Dropout probability
            autoregressive: If True, use 'add' aggregation for masked decoding where
                future nodes are masked out. Default 'mean' aggregation is used for
                standard bidirectional message passing.
            activations: (scalar_act, vector_act) activation functions
            vector_gate: Whether to use vector gating
        """
        super().__init__()
        self.conv = GVPConv(
            node_dims,
            node_dims,
            edge_dims,
            n_message,
            aggr="add" if autoregressive else "mean",
            activations=activations,
            vector_gate=vector_gate,
        )
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP_(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP_(node_dims, hid_dims))
            for i in range(n_feedforward - 2):
                ff_func.append(GVP_(hid_dims, hid_dims))
            ff_func.append(GVP_(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_func)

    def forward(self, x, edge_index, edge_attr, autoregressive_x=None, node_mask=None):
        """
        :param x: tuple (s, V) of `torch.Tensor`
        :param edge_index: array of shape [2, n_edges]
        :param edge_attr: tuple (s, V) of `torch.Tensor`
        :param autoregressive_x: tuple (s, V) of `torch.Tensor`.
                If not `None`, will be used as src node embeddings
                for forming messages where src >= dst. The corrent node
                embeddings `x` will still be the base of the update and the
                pointwise feedforward.
        :param node_mask: array of type `bool` to index into the first
                dim of node embeddings (s, V). If not `None`, only
                these nodes will be updated.
        """

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward),
            )

            count = (
                scatter_add(torch.ones_like(dst), dst, dim_size=dh[0].size(0))
                .clamp(min=1)
                .unsqueeze(-1)
            )

            dh = dh[0] / count, dh[1] / count.unsqueeze(-1)

        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_
        return x


class EdgeUpdate(nn.Module):
    """
    Residual edge update that keeps a fixed scalar edge width across layers.
    Input to the MLP is [s_src, s_dst, s_edge( fixed width ), (optional) distance/RBF].
    Output shape == input edge width. No width drift.
    """

    def __init__(
        self,
        n_node_scalars: int,  # S_node (e.g., 256)
        s_edge_width: int,  # fixed model edge width used everywhere
        update_w_distance_features: bool = False,
        distance_dim: int = 0,  # e.g., RBF size
    ):
        """
        Initialize residual edge update module.

        Args:
            n_node_scalars: Dimension of node scalar features
            s_edge_width: Fixed edge scalar width maintained across layers
            update_w_distance: If True, include distance features in update
            distance_dim: Dimension of distance features (e.g., RBF size)
        """
        super().__init__()
        self.update_w_distance_features = update_w_distance_features
        self.s_edge_width = s_edge_width

        in_dim = (
            (2 * n_node_scalars)
            + s_edge_width
            + (distance_dim if update_w_distance_features else 0)
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim, s_edge_width),
            nn.SiLU(),
            nn.Linear(s_edge_width, s_edge_width),
            nn.SiLU(),
        )
        self.edge_norm = nn.LayerNorm(s_edge_width)

    def forward(
        self,
        node_tuple: tuple,  # (s_node, V_node) with s_node: (N, S_node)
        edge_index: torch.Tensor,  # (2, E)
        edge_attr: tuple,  # (s_edge, V_edge) with s_edge: (E, s_edge_width)
        distance_feat: torch.Tensor | None = None,  # (E, D) if enabled
    ) -> tuple:
        """
        Compute residual edge feature update.

        Args:
            node_tuple: (s_node, V_node) where s_node is (N, n_node_scalars)
            edge_index: (2, E) edge indices
            edge_attr: (s_edge, V_edge) current edge features
            distance_feat: (E, distance_dim) optional distance features

        Returns:
            (s_edge_updated, V_edge) tuple with updated scalar features;
            vector features pass through unchanged
        """
        s_node, _ = node_tuple
        s_edge, V_edge = edge_attr

        if s_edge.shape[-1] != self.s_edge_width:
            raise ValueError(
                f"EdgeUpdate expected width {self.s_edge_width}, got {s_edge.shape[-1]}"
            )

        src, dst = edge_index[0], edge_index[1]
        parts = [s_node[src], s_node[dst], s_edge]

        if self.update_w_distance_features:
            parts.append(distance_feat)

        h = torch.cat(parts, dim=-1)  # (E, 2*S_node + s_edge_width (+D))
        upd = self.edge_mlp(h)  # (E, s_edge_width)
        s_edge = self.edge_norm(s_edge + upd)  # residual, fixed width
        return (s_edge, V_edge)  # vectors unchanged


# multi edge gvp


class GVPMultiEdge(MessagePassing):
    """
    Per-edge-type GVP message function (messages only, no residual/update).
    Outputs a merged tensor for PyG aggregation: _merge(s_msg, v_msg).
    """

    def __init__(
        self,
        src_type: str,
        dst_type: str,
        s_dim: int,
        v_dim: int,
        rbf_dim: int = NUM_RBF,
        use_dst_feats: bool = False,
        n_message_gvps: int = 1,
        activations=(F.relu, torch.sigmoid),
        vector_gate=True,
        aggr="sum",
        rbf_dmax: float = RBF_CUTOFF,
    ):
        """
        Initialize per-edge-type GVP message passing layer.

        Args:
            src_type: Source node type identifier
            dst_type: Destination node type identifier
            s_dim: Scalar feature dimension
            v_dim: Vector feature dimension (number of 3D channels)
            rbf_dim: Number of radial basis functions
            use_dst_feats: If True, include destination features in messages
            n_message_gvps: Number of GVP layers in message function
            activations: (scalar_act, vector_act) activation functions
            vector_gate: Whether to use vector gating
            aggr: Message aggregation method ('sum' or 'mean')
            rbf_dmax: Maximum distance in Angstroms for RBF encoding
        """
        super().__init__(aggr=aggr)
        self.src_type, self.dst_type = src_type, dst_type
        self.s_dim, self.v_dim = s_dim, v_dim
        self.use_dst_feats = use_dst_feats
        self.rbf_dim, self.rbf_dmax = rbf_dim, rbf_dmax
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        # message GVP stack; first layer takes [unit_vec] and [rbf] extras
        msg_layers = []
        for i in range(n_message_gvps):
            vector_input_dim = (
                v_dim
                + (1 if i == 0 else 0)  # +1 for unit displacement vector on first layer
                + (v_dim if (i == 0 and use_dst_feats) else 0)
            )
            scalar_input_dim = (
                s_dim
                + (rbf_dim if i == 0 else 0)
                + (s_dim if (i == 0 and use_dst_feats) else 0)
            )
            msg_layers.append(
                GVP_(
                    in_dims=(scalar_input_dim, vector_input_dim),
                    out_dims=(s_dim, v_dim),
                    vector_gate=True,
                    activations=activations,
                )
            )
        self.edge_message = nn.Sequential(*msg_layers)

    def forward(self, x, edge_index, pos_pair, cached_edge_attr=None):
        """
        x: ((s_src, v_src), (s_dst, v_dst)) for bipartite, or (s,v) for homogeneous
        pos_pair: (pos_src, pos_dst) positions
        cached_edge_attr: optional tuple (cached_rbf, cached_unit) for pre-computed edge features
        returns: merged messages at dst nodes: shape [N_dst, s_dim + 3*v_dim]
        """
        # Unpack by relation type:
        if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], tuple):
            (s_src, v_src), (s_dst, v_dst) = x
        else:
            # homogeneous: src==dst and x is (s,v)
            s_src = s_dst = x[0]
            v_src = v_dst = x[1]

        # edge case for no edges being made for weird reason (debugs in flow)
        if edge_index.numel() == 0:
            N_dst = s_dst.size(0)
            return s_dst.new_zeros(N_dst, self.s_dim + 3 * self.v_dim)

        pos_src, pos_dst = pos_pair

        # flatten vectors for passing; we'll unflatten in message()
        v_src_f = v_src.reshape(v_src.size(0), -1)  # (N_src, 3*v)
        v_dst_f = v_dst.reshape(v_dst.size(0), -1)  # (N_dst, 3*v)

        # use cached edge features if provided, otherwise compute on the fly
        if cached_edge_attr is not None:
            rbf_e, unit = cached_edge_attr
        else:
            unit, rbf_e = compute_edge_features(
                pos=pos_src,
                edge_index=edge_index,
                pos_dst=pos_dst,
                num_gaussians=self.rbf_dim,
                cutoff=self.rbf_dmax,
            )

        # pack as simple tensors for message()
        # edge_attr = (rbf_e, unit)  -- pass separately to keep shapes explicit
        out = self.propagate(
            edge_index,
            size=(s_src.size(0), s_dst.size(0)),
            s=(s_src, s_dst),
            vf=(v_src_f, v_dst_f),
            rbf_e=rbf_e,
            unit=unit,
        )
        return out  # merged messages (s_msg + flattened v_msg)

    def message(self, s_i, s_j, vf_i, vf_j, rbf_e, unit):
        """
        Construct GVP message from source to destination.

        Args:
            s_i: (E, s_dim) destination scalar features
            s_j: (E, s_dim) source scalar features
            vf_i: (E, 3*v_dim) destination vector features (flattened)
            vf_j: (E, 3*v_dim) source vector features (flattened)
            rbf_e: (E, rbf_dim) RBF distance features
            unit: (E, 3) unit displacement vectors

        Returns:
            (E, s_dim + 3*v_dim) merged message tensor
        """
        # Unflatten vectors
        v_j = vf_j.view(vf_j.size(0), -1, 3)  # (E, v, 3)
        v_i = vf_i.view(vf_i.size(0), -1, 3)  # (E, v, 3)

        # Build vector features: [unit_vec] (+ src v) (+ dst v if use_dst_feats)
        v_list = [unit.unsqueeze(1), v_j]  # (E, 1,3) + (E, v,3)
        s_list = [s_j, rbf_e]  # (E, s_dim) + (E, rbf_dim)
        if self.use_dst_feats:
            v_list.append(v_i)
            s_list.append(s_i)

        v_in = torch.cat(v_list, dim=1)
        s_in = torch.cat(s_list, dim=1)

        s_msg, v_msg = self.edge_message((s_in, v_in))  # (E, s_dim), (E, v, 3)
        return _merge(s_msg, v_msg)  # (E, s_dim + 3*v)


class GVPMultiEdgeConv(nn.Module):
    """
    Hetero multi-edge message passing – messages only per relation,
    then single residual+FFN+layernorm per destination node type.
    """

    def __init__(
        self,
        etypes: list[tuple[str, str, str]],  # (src_type, edge_type, dst_type)
        s_dim: int,
        v_dim: int,
        rbf_dim: int = 16,
        n_message_gvps: int = 1,
        n_update_gvps: int = 1,
        use_dst_feats: bool = False,
        drop_rate: float = 0.1,
        aggr_edges: str = "sum",  # 'mean' or 'add' per edge aggregation
        activations=(F.relu, torch.sigmoid),
        vector_gate=True,
    ):
        """
        Initialize heterogeneous multi-edge GVP convolution layer.

        Args:
            etypes: List of (src_type, relation, dst_type) edge type tuples
            s_dim: Scalar feature dimension
            v_dim: Vector feature dimension
            rbf_dim: Number of radial basis functions
            n_message_gvps: Number of GVPs in per-edge-type message function
            n_update_gvps: Number of GVPs in per-node-type update function
            use_dst_feats: If True, include destination features in messages
            drop_rate: Dropout probability
            aggr_edges: Per-edge aggregation method ('sum' or 'mean')
            activations: (scalar_act, vector_act) activation functions
            vector_gate: Whether to use vector gating
        """
        super().__init__()
        self.s_dim, self.v_dim = s_dim, v_dim
        self.drop = Dropout(drop_rate)
        GVP_ = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        # per-dst-type norms and update stacks
        dst_ntypes = sorted({dst for (_, _, dst) in etypes})
        self.msg_norms = nn.ModuleDict(
            {nt: LayerNorm((s_dim, v_dim)) for nt in dst_ntypes}
        )
        self.upd_norms = nn.ModuleDict(
            {nt: LayerNorm((s_dim, v_dim)) for nt in dst_ntypes}
        )

        self.node_updates = nn.ModuleDict()
        for nt in dst_ntypes:
            upd_layers = []
            for _ in range(n_update_gvps):
                upd_layers.append(GVP_((s_dim, v_dim), (s_dim, v_dim)))
            self.node_updates[nt] = nn.Sequential(*upd_layers)

        # per-edge-type message convs feeding a HeteroConv(aggr='sum' across relations)
        rel_convs = {}
        for src, rel, dst in etypes:
            rel_convs[(src, rel, dst)] = GVPMultiEdge(
                src,
                dst,
                s_dim,
                v_dim,
                rbf_dim=rbf_dim,
                use_dst_feats=use_dst_feats,
                n_message_gvps=n_message_gvps,
                activations=activations,
                vector_gate=vector_gate,
                aggr=("mean" if aggr_edges == "mean" else "sum"),
            )
        self.hconv = HeteroConv(rel_convs, aggr="sum")  # sum messages across edge types

    def forward(self, x_dict, edge_index_dict, pos_dict, cached_edge_attr_dict=None):
        """
        Run heterogeneous message passing across all edge types.

        Args:
            x_dict: Dict mapping node type to (s, V) feature tuples
                - s: (N, s_dim) scalar features
                - V: (N, v_dim, 3) vector features
            edge_index_dict: Dict mapping (src, rel, dst) to (2, E) edge indices
            pos_dict: Dict mapping node type to (N, 3) position tensors
            cached_edge_attr_dict: Optional dict mapping edge types to
                (rbf_features, unit_vectors) tuples for pre-computed edge features,
                where rbf_features are (E, rbf_dim) radial basis function distance
                encodings and unit_vectors are (E, 3) normalized displacement vectors.

        Returns:
            Updated x_dict with same structure as input
        """
        # build per-edge-type (pos_src, pos_dst) tuples
        pos_pair_dict = {
            et: (pos_dict[et[0]], pos_dict[et[2]]) for et in edge_index_dict.keys()
        }

        # build cached edge attr dict for HeteroConv (None for edges without cache)
        if cached_edge_attr_dict is None:
            cached_edge_attr_dict = {}
        edge_attr_for_hconv = {
            et: cached_edge_attr_dict.get(et, None) for et in edge_index_dict.keys()
        }

        # heteroConv will forward kwarg name without '_dict' into each conv
        merged_msgs = self.hconv(
            x_dict,
            edge_index_dict,
            pos_pair_dict=pos_pair_dict,
            cached_edge_attr_dict=edge_attr_for_hconv,
        )
        # merged_msgs[ntype] is a merged tensor of total messages: [N, s_dim + 3*v_dim]
        for ntype, merged in merged_msgs.items():
            s_msg, v_msg = _split(merged, self.v_dim)

            # apply dropout, residual add, layernorm (message stage)
            s_old, v_old = x_dict[ntype]
            s_msg, v_msg = self.drop((s_msg, v_msg))
            s_mid, v_mid = self.msg_norms[ntype](
                tuple_sum((s_old, v_old), (s_msg, v_msg))
            )

            # per-node update stack (GVP x N), then dropout + residual + layernorm
            s_res, v_res = self.node_updates[ntype]((s_mid, v_mid))
            s_res, v_res = self.drop((s_res, v_res))
            x_dict[ntype] = self.upd_norms[ntype](
                tuple_sum((s_mid, v_mid), (s_res, v_res))
            )

        return x_dict
