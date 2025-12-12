# flow.py

import math
from typing import Dict, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import knn
from torch_geometric.data import HeteroData, Data
import e3nn

from gvp import GVP, GVPMultiEdgeConv

import copy

import numpy as np
from torch import Tensor
from tqdm.auto import tqdm

from utils import condot_pair_hard_hungarian, compute_rmsd, cov_prec_at_threshold

def rbf(d: torch.Tensor, num_rbf: int, D_min: float = 0.0, D_max: float = 20.0) -> torch.Tensor:
    return e3nn.math.soft_one_hot_linspace(
        d, start=D_min, end=D_max, number=num_rbf, basis="bessel", cutoff=True
    )

def edge_features(src_pos: torch.Tensor,
                  dst_pos: torch.Tensor,
                  edge_index: torch.Tensor,
                  num_rbf: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute RBF(dist) and unit displacement vectors for given edges.
    Returns:
        e_s: (E, num_rbf)
        e_v: (E, 1, 3)
    """
    if edge_index.numel() == 0:
        return (torch.empty(0, num_rbf, device=src_pos.device),
                torch.empty(0, 1, 3, device=src_pos.device))

    s_idx, d_idx = edge_index
    assert s_idx.max() < src_pos.size(0), "src index out of bounds"
    assert d_idx.max() < dst_pos.size(0), "dst index out of bounds"

    disp = src_pos[s_idx] - dst_pos[d_idx]
    dist = torch.clamp(disp.norm(dim=-1, keepdim=True), min=1e-8)
    e_s = rbf(dist.squeeze(-1), num_rbf=num_rbf)
    e_v = (disp / dist).unsqueeze(1)
    return e_s, e_v


def build_knn_edges(src_pos: torch.Tensor,
                    dst_pos: torch.Tensor,
                    k: int,
                    batch_src: Optional[torch.Tensor] = None,
                    batch_dst: Optional[torch.Tensor] = None) -> torch.Tensor:
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

def make_protein_mate_encoder_data(
    data: HeteroData,
    num_rbf: int,
    rbf_dmin: float = 0.0,
    rbf_dmax: float = 20.0,
) -> Tuple[Data, int, int]:
    """
    Build a homogeneous Data with nodes = protein + mate, and edges:
      - protein-protein from ('protein', 'pp', 'protein')
      - mate-mate from ('mate', 'mm', 'mate')
      - protein-mate (both directions) from ('protein','pm','mate')

    Returns:
        enc_data: Data with x, pos, edge_index, edge_rbf, edge_unit_vec
        num_prot: # protein nodes
        num_mate: # mate nodes
    """
    device = data['protein'].pos.device

    prot = data['protein']
    mate = data['mate']

    num_prot = prot.num_nodes
    num_mate = mate.num_nodes

    x = torch.cat([prot.x, mate.x], dim=0)
    pos = torch.cat([prot.pos, mate.pos], dim=0)

    edge_index_list = []

    # protein-protein
    if ('protein', 'pp', 'protein') in data.edge_types:
        pp = data['protein', 'pp', 'protein'].edge_index
        edge_index_list.append(pp)

    # mate-mate (offset indices by num_prot)
    if num_mate > 0 and ('mate', 'mm', 'mate') in data.edge_types:
        mm = data['mate', 'mm', 'mate'].edge_index.clone()
        mm = mm + num_prot
        edge_index_list.append(mm)

    # protein-mate contacts (both directions)
    if num_mate > 0 and ('protein', 'pm', 'mate') in data.edge_types:
        pm_local = data['protein', 'pm', 'mate'].edge_index
        pm_comb = torch.stack(
            [pm_local[0], pm_local[1] + num_prot], dim=0
        )
        mp_comb = pm_comb.flip(0)
        edge_index_list.extend([pm_comb, mp_comb])

    if edge_index_list:
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long, device=device)

    # geometric edge features for encoder
    if edge_index.numel() > 0:
        s_idx, d_idx = edge_index
        disp = pos[s_idx] - pos[d_idx]
        dist = torch.clamp(disp.norm(dim=-1, keepdim=True), 1e-8)
        r = dist.squeeze(-1)
        edge_rbf = rbf(r, num_rbf=num_rbf, D_min=rbf_dmin, D_max=rbf_dmax)
        edge_unit_vec = (disp / dist)
    else:
        edge_rbf = torch.empty(0, num_rbf, device=device)
        edge_unit_vec = torch.empty(0, 3, device=device)

    enc_data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_rbf=edge_rbf,
        edge_unit_vec=edge_unit_vec,
    )

    # batch for multi-complex batches
    if hasattr(prot, "batch") and hasattr(mate, "batch"):
        enc_data.batch = torch.cat([prot.batch, mate.batch], dim=0)

    return enc_data, num_prot, num_mate

class HeteroToDataWrapper:
    """Wrap a single node type of HeteroData to mimic a Data object."""

    def __init__(self, hetero_data: HeteroData, node_type: str = 'protein'):
        self.hetero_data = hetero_data
        self.node_type = node_type

    def __getattr__(self, name):
        if name in self.hetero_data[self.node_type]:
            return self.hetero_data[self.node_type][name]

        edge_key = (self.node_type, 'pp', self.node_type)
        if edge_key in self.hetero_data.edge_types:
            edge_storage = self.hetero_data[edge_key]
            if name in edge_storage:
                return edge_storage[name]

        if name == 'edge_index':
            return self.hetero_data[self.node_type, 'pp', self.node_type].edge_index
        elif name == 'edge_rbf':
            return self.hetero_data[self.node_type, 'pp', self.node_type].edge_rbf
        elif name == 'edge_unit_vec':
            return self.hetero_data[self.node_type, 'pp', self.node_type].edge_unit_vec

        raise AttributeError(f"{self.__class__.__name__} has no attribute {name!r}")

class ProteinWaterUpdate(nn.Module):
    """
    Heterogeneous GVP message passing:
      - protein -> water  (pw)
      - mate    -> water  (mw)
      - water   -> water  (ww)
      - (optional) protein<->protein (pp), water->protein (wp)
      - (optional) mate<->mate (mm), water->mate (wm)
    """

    def __init__(
        self,
        hidden_dims=(512, 64),
        rbf_dim=16,
        layers=3,
        drop_rate=0.0,
        vector_gate=True,
        aggr_edges="sum",
        update_protein=False,
        update_mate=False,
        use_dst_feats=True,
    ):
        super().__init__()
        s_h, v_h = hidden_dims

        etypes = [
            ('protein', 'pw', 'water'),
            ('mate',    'mw', 'water'),
            ('water',   'ww', 'water'),
        ]
        if update_protein:
            etypes += [
                ('protein', 'pp', 'protein'),
                ('water',   'wp', 'protein'),
            ]
        if update_mate:
            etypes += [
                ('mate',  'mm', 'mate'),
                ('water', 'wm', 'mate'),
            ]

        self.blocks = nn.ModuleList([
            GVPMultiEdgeConv(
                etypes=etypes,
                s_dim=s_h, v_dim=v_h,
                rbf_dim=rbf_dim,
                n_message_gvps=3,
                n_update_gvps=3,
                use_dst_feats=use_dst_feats,
                drop_rate=drop_rate,
                aggr_edges=aggr_edges,
                activations=(F.relu, torch.sigmoid),
                vector_gate=vector_gate,
            )
            for _ in range(layers)
        ])
        self.etypes = etypes

    def build_edges(self,
                    data: HeteroData,
                    k_pw: int = 12,
                    k_ww: int = 8,
                    k_wp: int = 8,
                    include_pp: bool = False,
                    include_mm: bool = False) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """
        Build KNN edges for:
          - protein->water, water->water,
          - optional protein->protein, water->protein,
          - mate->water, optional mate->mate, water->mate.
        Always returns entries for all self.etypes (possibly empty).
        """
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor] = {}
        device = data['protein'].pos.device

        def knn_edges(src_pos, dst_pos, k, b_src=None, b_dst=None):
            if src_pos.numel() == 0 or dst_pos.numel() == 0:
                return torch.empty(2, 0, dtype=torch.long, device=src_pos.device)
            ei = knn(x=dst_pos, y=src_pos, k=k, batch_x=b_dst, batch_y=b_src)
            if src_pos.data_ptr() == dst_pos.data_ptr():
                mask = ei[0] != ei[1]
                ei = ei[:, mask]
            return ei.unique(dim=1)

        batch_p = data['protein'].batch if 'batch' in data['protein'] else None
        batch_w = data['water'].batch if 'batch' in data['water'] else None
        batch_m = data['mate'].batch if 'batch' in data['mate'] else None

        pos_p = data['protein'].pos
        pos_w = data['water'].pos
        pos_m = data['mate'].pos

        # protein -> water
        if pos_p.numel() > 0 and pos_w.numel() > 0:
            edge_index_dict[('protein', 'pw', 'water')] = knn_edges(
                pos_p, pos_w, k=k_pw, b_src=batch_p, b_dst=batch_w
            )
        else:
            edge_index_dict[('protein', 'pw', 'water')] = torch.empty(
                2, 0, dtype=torch.long, device=device
            )

        # water -> water
        if pos_w.numel() > 0:
            edge_index_dict[('water', 'ww', 'water')] = knn_edges(
                pos_w, pos_w, k=k_ww, b_src=batch_w, b_dst=batch_w
            )
        else:
            edge_index_dict[('water', 'ww', 'water')] = torch.empty(
                2, 0, dtype=torch.long, device=device
            )

        # optional protein-protein and water->protein
        if include_pp:
            if ('protein', 'pp', 'protein') in data.edge_types:
                edge_index_dict[('protein', 'pp', 'protein')] = \
                    data['protein', 'pp', 'protein'].edge_index
            else:
                edge_index_dict[('protein', 'pp', 'protein')] = knn_edges(
                    pos_p, pos_p, k=k_pw, b_src=batch_p, b_dst=batch_p
                )

            if pos_w.numel() > 0 and pos_p.numel() > 0:
                edge_index_dict[('water', 'wp', 'protein')] = knn_edges(
                    pos_w, pos_p, k=k_wp, b_src=batch_w, b_dst=batch_p
                )
            else:
                edge_index_dict[('water', 'wp', 'protein')] = torch.empty(
                    2, 0, dtype=torch.long, device=device
                )

        # mate -> water
        if pos_m.numel() > 0 and pos_w.numel() > 0:
            edge_index_dict[('mate', 'mw', 'water')] = knn_edges(
                pos_m, pos_w, k=k_pw, b_src=batch_m, b_dst=batch_w
            )
        else:
            edge_index_dict[('mate', 'mw', 'water')] = torch.empty(
                2, 0, dtype=torch.long, device=device
            )

        # optional mate-mate and water->mate
        if include_mm:
            if ('mate', 'mm', 'mate') in data.edge_types:
                edge_index_dict[('mate', 'mm', 'mate')] = \
                    data['mate', 'mm', 'mate'].edge_index
            else:
                edge_index_dict[('mate', 'mm', 'mate')] = knn_edges(
                    pos_m, pos_m, k=k_pw, b_src=batch_m, b_dst=batch_m
                )

            if pos_w.numel() > 0 and pos_m.numel() > 0:
                edge_index_dict[('water', 'wm', 'mate')] = knn_edges(
                    pos_w, pos_m, k=k_wp, b_src=batch_w, b_dst=batch_m
                )
            else:
                edge_index_dict[('water', 'wm', 'mate')] = torch.empty(
                    2, 0, dtype=torch.long, device=device
                )

        # ensure all etypes are present, even if empty
        for et in self.etypes:
            if et not in edge_index_dict:
                edge_index_dict[et] = torch.empty(
                    2, 0, dtype=torch.long, device=device
                )

        return edge_index_dict

    def forward(self,
                x_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                data: HeteroData,
                k_pw: int = 12,
                k_ww: int = 8,
                k_wp: int = 8,
                update_protein: bool = False,
                update_mate: bool = False):
        """
        x_dict: {
            'protein': (s_p, v_p),
            'mate':    (s_m, v_m),
            'water':   (s_w, v_w)
        }
        """
        pos_dict = {nt: data[nt].pos for nt in data.node_types if 'pos' in data[nt]}

        edge_index_dict = self.build_edges(
            data,
            k_pw=k_pw, k_ww=k_ww, k_wp=k_wp,
            include_pp=update_protein,
            include_mm=update_mate,
        )

        for block in self.blocks:
            x_dict = block(x_dict, edge_index_dict, pos_dict)

        return x_dict


class FlowWaterGVP(nn.Module):
    """
    End-to-end:
      1. Encode protein+mate as homogeneous graph.
      2. Split latents back into protein/mate nodes.
      3. Time-condition protein, mate, and water.
      4. Build protein->water and mate->water edges.
      5. Run hetero multi-edge GVP update.
      6. Predict water vector field.
    """

    def __init__(
        self,
        encoder: ProteinGVPEncoder,
        hidden_dims: Tuple[int, int] = (256, 32),
        edge_scalar_dim: int = 32,
        layers: int = 4,
        drop_rate: float = 0.0,
        vector_gate: bool = True,
        k_pw: int = 12,
        k_ww: int = 8,
        k_wp: int = 8,
        freeze_encoder: bool = False,
        water_input_dim = 16 # 1 hot with oxygen, same as encoder
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_dims = hidden_dims
        self.edge_scalar_dim = edge_scalar_dim
        self.layers = layers
        self.drop_rate = drop_rate
        self.vector_gate = vector_gate
        self.k_pw = k_pw
        self.k_ww = k_ww
        self.k_wp = k_wp
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        # map encoder hidden dims -> flow hidden dims
        self.encoder_hidden_dims = encoder.hidden_dims
        s_h, v_h = hidden_dims

        self.encoder_to_flow = GVP(
            in_dims=self.encoder_hidden_dims,
            out_dims=hidden_dims,
            activations=(F.relu, torch.sigmoid),
            vector_gate=True,
        )

        # time-conditioning for protein/mate
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

        # hetero updater: protein+mate+water
        self.updater = ProteinWaterUpdate(
            hidden_dims=hidden_dims,
            rbf_dim=edge_scalar_dim,
            layers=layers,
            drop_rate=drop_rate,
            vector_gate=vector_gate,
            aggr_edges="sum",
            update_protein=not freeze_encoder,
            update_mate=True,
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

        # water vector field head: (s,v) -> (0,1) => single vector channel
        self.vfield_head = GVP(
            in_dims=hidden_dims,
            out_dims=(0, 1),
            vector_gate=True,
        )

    def forward(self,
                data: HeteroData,
                t: torch.Tensor,
                sc: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        data: HeteroData with node types 'protein', 'mate', 'water'
        t: (B,) diffusion time per complex
        sc: optional dict with 'x1_pred' for score conditioning (same shape as water.pos)

        Returns:
            v_pred: (N_water, 3) vector field at each water node.
        """
        device = data['protein'].pos.device

        # encoder over protein + mate as a single homogeneous graph
        enc_data, num_prot, num_mate = make_protein_mate_encoder_data(
            data,
            num_rbf=self.encoder.edge_scalar_in,
            rbf_dmin=0.0,
            rbf_dmax=20.0,
        )

        with torch.set_grad_enabled(not self.freeze_encoder):
            s_all, v_all = self.encoder(enc_data)

        # bridge encoder dims -> flow dims
        s_all, v_all = self.encoder_to_flow((s_all, v_all))

        # split back into protein / mate
        s_p_latent = s_all[:num_prot]
        v_p_latent = v_all[:num_prot]
        s_m_latent = s_all[num_prot:num_prot + num_mate]
        v_m_latent = v_all[num_prot:num_prot + num_mate]

        if 'water' not in data.node_types or data['water'].num_nodes == 0:
            return torch.zeros(0, 3, device=device)

        batch_p = data['protein'].batch
        batch_w = data['water'].batch
        batch_m = data['mate'].batch

        t_p = t[batch_p].unsqueeze(-1)
        t_w = t[batch_w].unsqueeze(-1)
        t_m = t[batch_m].unsqueeze(-1)

        s_p = self.protein_scalar_encoder(torch.cat([s_p_latent, t_p], dim=-1))
        if num_mate > 0:
            s_m = self.protein_scalar_encoder(torch.cat([s_m_latent, t_m], dim=-1))
        else:
            s_m = s_m_latent

        s_w = self.water_scalar_encoder(torch.cat([data['water'].x, t_w], dim=-1))

        # initial water vectors (all zeros to start)
        v_w = torch.zeros(
            data['water'].num_nodes,
            self.hidden_dims[1],
            3,
            device=device,
        )

        # self conditioning
        if sc is not None and ('x1_pred' in sc) and sc['x1_pred'] is not None:
            delta = (sc['x1_pred'] - data['water'].pos)
            delta_vec = delta.unsqueeze(1)

            # vector conditioning (equivariant)
            _, v_sc = self.sc_vec_encoder(
                (torch.empty(0, device=device), delta_vec)
            )
            v_w = v_w + v_sc

            # scalar conditioning (invariant) on ||delta||
            d_mag = delta.norm(dim=-1, keepdim=True)
            s_sc = self.sc_sca_encoder(d_mag)
            s_w = s_w + s_sc

        # build hetero feature dict for GVP multi-edge updates
        x_dict = {
            'protein': (s_p, v_p_latent),
            'mate':    (s_m, v_m_latent),
            'water':   (s_w, v_w),
        }

        # hetero update (protein+mate+water graph)
        x_dict = self.updater(
            x_dict,
            data,
            k_pw=self.k_pw,
            k_ww=self.k_ww,
            k_wp=self.k_wp,
            update_protein=not self.freeze_encoder,
            update_mate=False,
        )

        # water vector field head
        _, v_pred = self.vfield_head(x_dict['water'])
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
        self.model = model
        self.p_self_cond = p_self_cond
        self.use_distortion = use_distortion
        self.p_distort = p_distort
        self.t_distort = t_distort
        self.sigma_distort = sigma_distort
        self.loss_eps = loss_eps

    @staticmethod
    def compute_sigma(data: HeteroData) -> float:
        """Compute sigma as std of protein coordinates."""
        pos = data['protein'].pos
        return float(pos.std().item())

    def training_step(
        self,
        batch: HeteroData,
        optimizer: torch.optim.Optimizer,
        grad_clip: float = 1.0,
        use_self_conditioning: bool = True,
    ) -> Dict[str, float]:
        """
        Single flow matching training step.
        
        Returns dict with 'loss', 'rmsd', 'sigma'.
        """
        self.model.train()
        device = batch['protein'].pos.device

        x1 = batch['water'].pos
        batch_w = batch['water'].batch
        batch_p = batch['protein'].batch
        num_graphs = int(batch_p.max().item()) + 1

        sigma = self.compute_sigma(batch)

        x0 = torch.randn_like(x1) * sigma
        x0_star, x1_star = condot_pair_hard_hungarian(x1=x1, batch=batch_w, x0=x0)

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
        sc = None
        if use_self_conditioning and torch.rand(1).item() < self.p_self_cond:
            with torch.no_grad():
                batch['water'].pos = x_t
                v_sc = self.model(batch, t, sc=None)
                x1_sc = x_t + (1.0 - t_per_atom) * v_sc
            sc = {'x1_pred': x1_sc}

        # forward pass
        batch['water'].pos = x_t
        v_pred = self.model(batch, t, sc=sc)

        # target velocity
        v_target = x1_star - x0_star

        # weighted MSE loss (upweight near t=1)
        w = 1.0 / (self.loss_eps + (1.0 - t_per_atom))
        per_atom_mse = (v_pred - v_target).pow(2).mean(dim=-1, keepdim=True)
        loss = (w * per_atom_mse).sum() / w.sum()

        # backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=grad_clip
            )
        optimizer.step()

        # training RMSD
        with torch.no_grad():
            x1_hat = x_t + (1.0 - t_per_atom) * v_pred
            rmsd = compute_rmsd(x1_hat, x1_star, batch_w)

        return {'loss': loss.item(), 'rmsd': rmsd, 'sigma': sigma}

    @torch.no_grad()
    def validation_step(self, batch: HeteroData) -> Dict[str, float]:
        """Single validation step (no gradient, no optimizer)."""
        self.model.eval()
        device = batch['protein'].pos.device

        x1 = batch['water'].pos
        batch_w = batch['water'].batch
        batch_p = batch['protein'].batch
        num_graphs = int(batch_p.max().item()) + 1

        sigma = self.compute_sigma(batch)
        x0 = torch.randn_like(x1) * sigma
        x0_star, x1_star = condot_pair_hard_hungarian(x1=x1, batch=batch_w, x0=x0)

        t = torch.rand(num_graphs, device=device)
        t_per_atom = t[batch_w].unsqueeze(-1)
        x_t = (1.0 - t_per_atom) * x0_star + t_per_atom * x1_star

        batch['water'].pos = x_t
        v_pred = self.model(batch, t, sc=None)
        v_target = x1_star - x0_star

        w = 1.0 / (self.loss_eps + (1.0 - t_per_atom))
        per_atom_mse = (v_pred - v_target).pow(2).mean(dim=-1, keepdim=True)
        loss = (w * per_atom_mse).sum() / w.sum()

        x1_hat = x_t + (1.0 - t_per_atom) * v_pred
        rmsd = compute_rmsd(x1_hat, x1_star, batch_w)

        return {'loss': loss.item(), 'rmsd': rmsd}

    @torch.no_grad()
    def rk4_integrate(
        self,
        graph: HeteroData,
        num_steps: int = 500,
        use_sc: bool = True,
        sc_ema_alpha: float = 0.2,
        device: str = "cuda",
    ) -> Dict[str, np.ndarray]:
        """
        RK4 integration from noise to final positions.
        
        Returns:
            dict with keys:
                'protein_pos': (Np, 3)
                'mate_pos': (Nm, 3) or empty
                'water_true': (Nw, 3)
                'water_pred': (Nw, 3) final prediction
                'trajectory': list of (Nw, 3) at each step
                'rmsd': list of RMSD values
                'coverage': list of coverage values
                'precision': list of precision values
        """
        self.model.eval()
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        g = copy.deepcopy(graph).to(device)
        assert 'water' in g.node_types and g['water'].num_nodes > 0

        # Extract positions
        protein_pos = g['protein'].pos.detach().cpu().numpy()
        mate_pos = g['mate'].pos.detach().cpu().numpy() if g['mate'].num_nodes > 0 else np.zeros((0, 3))
        x1_true = g['water'].pos.clone()
        water_true = x1_true.detach().cpu().numpy()

        # Sigma from protein
        sigma = self.compute_sigma(g)

        # Initial noise
        x = torch.randn_like(x1_true) * sigma
        x1_pred_ema = x.clone()

        # Ensure batch tensors exist
        if 'batch' not in g['protein']:
            g['protein'].batch = torch.zeros(g['protein'].num_nodes, dtype=torch.long, device=device)
        if 'batch' not in g['water']:
            g['water'].batch = torch.zeros(g['water'].num_nodes, dtype=torch.long, device=device)
        if 'mate' in g.node_types and 'batch' not in g['mate']:
            g['mate'].batch = torch.zeros(g['mate'].num_nodes, dtype=torch.long, device=device)

        # Time grid
        ts = torch.linspace(0, 1, num_steps, device=device)
        dt = ts[1] - ts[0]

        # Storage
        trajectory = [x.detach().cpu().numpy().copy()]
        rmsd_values = [compute_rmsd(x, x1_true)]
        cov0, prec0 = cov_prec_at_threshold(x, x1_true, thresh=1.0)
        cov_values, prec_values = [cov0], [prec0]

        # RK4 integration
        for i in tqdm(range(num_steps - 1), desc="RK4 integration", leave=False):
            t0 = ts[i]

            def f(xpos, t_scalar):
                g['water'].pos = xpos
                sc = {'x1_pred': x1_pred_ema} if use_sc else None
                return self.model(g, t_scalar.view(1), sc=sc)

            k1 = f(x, t0)
            k2 = f(x + 0.5 * dt * k1, t0 + 0.5 * dt)
            k3 = f(x + 0.5 * dt * k2, t0 + 0.5 * dt)
            k4 = f(x + dt * k3, t0 + dt)

            x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

            # Update self-conditioning EMA
            if use_sc:
                t1 = ts[i + 1]
                g['water'].pos = x
                v_next = self.model(g, t1.view(1), sc={'x1_pred': x1_pred_ema})
                x1_pred_now = x + (1.0 - t1) * v_next
                x1_pred_ema = (1.0 - sc_ema_alpha) * x1_pred_ema + sc_ema_alpha * x1_pred_now

            trajectory.append(x.detach().cpu().numpy().copy())
            rmsd_values.append(compute_rmsd(x, x1_true))
            cov_i, prec_i = cov_prec_at_threshold(x, x1_true, thresh=1.0)
            cov_values.append(cov_i)
            prec_values.append(prec_i)

        return {
            'protein_pos': protein_pos,
            'mate_pos': mate_pos,
            'water_true': water_true,
            'water_pred': x.detach().cpu().numpy(),
            'trajectory': trajectory,
            'rmsd': rmsd_values,
            'coverage': cov_values,
            'precision': prec_values,
        }

    @torch.no_grad()
    def euler_integrate(
        self,
        graph: HeteroData,
        num_steps: int = 100,
        use_sc: bool = True,
        device: str = "cuda",
    ) -> np.ndarray:
        """Simple Euler integration (faster than RK4, less accurate)."""
        self.model.eval()
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        g = copy.deepcopy(graph).to(device)
        sigma = self.compute_sigma(g)
        x = torch.randn(g['water'].num_nodes, 3, device=device) * sigma

        if 'batch' not in g['protein']:
            g['protein'].batch = torch.zeros(g['protein'].num_nodes, dtype=torch.long, device=device)
        if 'batch' not in g['water']:
            g['water'].batch = torch.zeros(g['water'].num_nodes, dtype=torch.long, device=device)
        if 'mate' in g.node_types and 'batch' not in g['mate']:
            g['mate'].batch = torch.zeros(g['mate'].num_nodes, dtype=torch.long, device=device)

        ts = torch.linspace(0, 1, num_steps, device=device)
        dt = ts[1] - ts[0]
        x1_pred_ema = x.clone()

        for i in range(num_steps - 1):
            t = ts[i]
            g['water'].pos = x
            sc = {'x1_pred': x1_pred_ema} if use_sc else None
            v = self.model(g, t.view(1), sc=sc)
            x = x + dt * v

            if use_sc:
                t_next = ts[i + 1]
                g['water'].pos = x
                v_next = self.model(g, t_next.view(1), sc={'x1_pred': x1_pred_ema})
                x1_pred_now = x + (1.0 - t_next) * v_next
                x1_pred_ema = 0.8 * x1_pred_ema + 0.2 * x1_pred_now

        return x.detach().cpu().numpy()

    @torch.no_grad()
    def sample_batch(
        self,
        batch: HeteroData,
        num_steps: int = 100,
        use_sc: bool = True,
        method: str = "euler",
    ) -> Tensor:
        """
        Sample water positions for a batched HeteroData.
        Returns: (M, 3) predicted water positions.
        """
        self.model.eval()
        device = batch['protein'].pos.device

        sigma = self.compute_sigma(batch)
        x = torch.randn(batch['water'].num_nodes, 3, device=device) * sigma

        batch_w = batch['water'].batch
        ts = torch.linspace(0, 1, num_steps, device=device)
        dt = ts[1] - ts[0]
        x1_pred_ema = x.clone()

        for i in range(num_steps - 1):
            t_scalar = ts[i]
            num_graphs = int(batch_w.max().item()) + 1
            t = t_scalar.expand(num_graphs)

            batch['water'].pos = x
            sc = {'x1_pred': x1_pred_ema} if use_sc else None
            v = self.model(batch, t, sc=sc)

            if method == "euler":
                x = x + dt * v
            else:  # midpoint
                x_mid = x + 0.5 * dt * v
                batch['water'].pos = x_mid
                t_mid = t_scalar + 0.5 * dt
                v_mid = self.model(batch, t_mid.expand(num_graphs), sc=sc)
                x = x + dt * v_mid

            if use_sc:
                t_next = ts[i + 1]
                batch['water'].pos = x
                t_vec = t_next.expand(num_graphs)
                v_next = self.model(batch, t_vec, sc={'x1_pred': x1_pred_ema})
                t_per_atom = t_next.unsqueeze(0).expand(x.size(0), 1)
                x1_pred_now = x + (1.0 - t_per_atom) * v_next
                x1_pred_ema = 0.8 * x1_pred_ema + 0.2 * x1_pred_now

        return x
