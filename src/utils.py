# utils.py

import torch
from torch import cdist
import numpy as np
from scipy.optimize import linear_sum_assignment
import scipy.spatial.distance as spdist
import matplotlib.pyplot as plt

from typing import Tuple
from torch import Tensor

from e3nn.math import soft_one_hot_linspace

def rbf(r: Tensor, num_gaussians: int = 16, cutoff: float = 8.0) -> Tensor:
    """Radial basis function encoding of distances."""
    r = r.clamp(min=1e-4)
    return soft_one_hot_linspace(
        r, 
        start=0.0, 
        end=cutoff, 
        number=num_gaussians,
        basis='bessel',
        cutoff=True
    )


#slow and do not use when training
@torch.no_grad
def compute_rmsd(pred, target, batch=None):
    """Compute RMSD with optimal assignment using Hungarian algorithm."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    if batch is not None:
        if isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().numpy()
        rmsds = []
        for g in np.unique(batch):
            m = batch == g
            rmsds.append(compute_rmsd(pred[m], target[m], batch=None))
        return float(np.mean(rmsds))
    
    dist_matrix = spdist.cdist(pred, target)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    diff = pred[row_ind] - target[col_ind]
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))

@torch.no_grad()
def condot_pair_hard_hungarian(
    x1: torch.Tensor,   # (M,3) true waters concatenated across graphs
    batch: torch.Tensor,# (M,) graph id per water
    x0: torch.Tensor,   # (M,3) Gaussian samples, same batch layout
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hard OT (Hungarian) pairing per graph.

    Returns:
        x0_star: (M,3) noise positions (just x0 reordered, but here it's unchanged)
        x1_star: (M,3) x1 permuted so it's matched to x0 per graph.
    """
    device = x1.device
    x0_star = torch.empty_like(x1)
    x1_star = torch.empty_like(x1)

    unique = torch.unique(batch)
    for g in unique:
        m = (batch == g)
        if m.sum() == 0:
            continue

        X1 = x1[m]   # (Ng,3)
        X0 = x0[m]   # (Ng,3)
        Ng = X1.size(0)

        # cost = L2^2
        C = cdist(X0, X1, p=2).pow(2).cpu().numpy()
        r, c = linear_sum_assignment(C)   # r is 0..Ng-1
        X1_match = X1[c]

        x0_star[m] = X0
        x1_star[m] = X1_match

    return x0_star, x1_star

def cov_prec_at_threshold(pred, true, thresh=1.0):
    """
    coverage: fraction of true points with >=1 prediction within thresh
    precision: fraction of predicted points with >=1 true within thresh
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if pred.size == 0 or true.size == 0:
        return 0.0, 0.0
    D = spdist.cdist(true, pred)   # [N_true, N_pred]
    coverage  = float((D.min(axis=1) <= thresh).mean())
    precision = float((D.min(axis=0) <= thresh).mean())
    return coverage, precision

def water_metrics(pred, true, thresholds=(0.5, 1.0, 1.5, 2.0)):
    """Compute coverage, precision, F1 at multiple distance thresholds."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    if pred.size == 0 or true.size == 0:
        return {t: {'cov': 0., 'prec': 0., 'f1': 0.} for t in thresholds}
    
    D = spdist.cdist(true, pred)
    results = {}
    
    for t in thresholds:
        cov = float((D.min(axis=1) <= t).mean())   # true -> nearest pred
        prec = float((D.min(axis=0) <= t).mean())  # pred -> nearest true
        f1 = 2 * cov * prec / (cov + prec + 1e-8)
        results[t] = {'cov': cov, 'prec': prec, 'f1': f1}
    
    return results

def plot_3d_frame(ax, protein_pos, mate_pos, water_pred, water_true, title=""):
    """Plot a single 3D frame (protein, mates, true vs predicted waters)."""
    ax.clear()

    # Protein (fixed)
    if protein_pos.size > 0:
        ax.scatter(
            protein_pos[:, 0], protein_pos[:, 1], protein_pos[:, 2],
            c='gray', alpha=0.3, s=8, label='Protein'
        )

    # Mates (optional)
    if mate_pos is not None and mate_pos.size > 0:
        ax.scatter(
            mate_pos[:, 0], mate_pos[:, 1], mate_pos[:, 2],
            c='dimgrey', alpha=0.6, s=10, label='Mates'
        )

    # Ground truth water
    ax.scatter(
        water_true[:, 0], water_true[:, 1], water_true[:, 2],
        c='red', marker='*', alpha=0.9, s=16, label='True Water'
    )

    # Predicted water
    ax.scatter(
        water_pred[:, 0], water_pred[:, 1], water_pred[:, 2],
        c='blue', alpha=0.9, s=14, label='Predicted Water'
    )

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    ax.legend(loc='upper right')

    # consistent limits based on protein + mates + true waters
    all_arrays = [water_true]
    if protein_pos.size > 0:
        all_arrays.append(protein_pos)
    if mate_pos is not None and mate_pos.size > 0:
        all_arrays.append(mate_pos)
    all_pos = np.vstack(all_arrays)
    lims = [(all_pos[:, i].min() - 2, all_pos[:, i].max() + 2) for i in range(3)]
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    ax.set_zlim(lims[2])

def save_protein_plot(pred_ca, true_ca, step, save_dir):
    """Aligns and plots CA traces."""
    # Convert to numpy
    P = pred_ca.detach().float().cpu().numpy()
    Q = true_ca.detach().float().cpu().numpy()
    
    # Center
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    
    # Kabsch Alignment (Numpy) for visualization
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    P_aligned = np.dot(P, R)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], color='black', linewidth=2, label='Ground Truth', alpha=0.6)
    ax.plot(P_aligned[:, 0], P_aligned[:, 1], P_aligned[:, 2], color='red', linewidth=2, label='Prediction')
    ax.legend()
    ax.set_title(f"Step {step}")
    plt.savefig(f"{save_dir}/step_{step}.png")
    plt.close()