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

# Constant used in atom37 representation (copied from SLAE)
ATOM37_FILL = 1e-5

def atom37_to_atoms(atom_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert atom37 representation to flat atom list.

    Given an atom tensor of shape (N_res, 37, 3), return:
    - coords: (N_atoms, 3) - coordinates of present atoms
    - residue_index: (N_atoms,) - which residue each atom belongs to
    - atom_type: (N_atoms,) - atom type index (0-36)

    This function is copied from SLAE to avoid import dependency.
    """
    # A site is present if ANY of its 3 coords differs from ATOM37_FILL
    present = (atom_tensor != ATOM37_FILL).any(dim=-1)  # (N_res, 37)
    nz = present.nonzero(as_tuple=False)  # (N_atoms, 2)
    residue_index = nz[:, 0]
    atom_type = nz[:, 1].long()

    flat = atom_tensor.reshape(-1, 3)
    flat_mask = present.reshape(-1)  # (N_res * 37,)
    coords = flat[flat_mask]  # (N_atoms, 3)

    return coords, residue_index, atom_type

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


@torch.no_grad()
def recall_precision_gpu(pred: torch.Tensor, true: torch.Tensor, thresh: float = 0.5):
    """
    Fast GPU version of recall and precision computation.

    Args:
        pred: (N_pred, 3) predicted positions
        true: (N_true, 3) ground truth positions
        thresh: distance threshold in Angstroms

    Returns:
        recall: fraction of true waters with a prediction within thresh
        precision: fraction of predictions within thresh of a true water
    """
    if pred.numel() == 0 or true.numel() == 0:
        return 0.0, 0.0

    # Compute pairwise distances: [N_true, N_pred]
    D = torch.cdist(true, pred, p=2)

    # Recall: fraction of true waters covered
    recall = (D.min(dim=1)[0] <= thresh).float().mean().item()

    # Precision: fraction of predictions that are correct
    precision = (D.min(dim=0)[0] <= thresh).float().mean().item()

    return recall, precision

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

def plot_3d_frame(ax, protein_pos, mate_pos, water_pred, water_true, title="", xlim=None, ylim=None, zlim=None):
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

    # Set fixed axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)


def compute_placement_metrics(pred, true, threshold=1.0):
    """
    Compute standard ML metrics for water placement.

    Args:
        pred: (N_pred, 3) predicted water positions
        true: (N_true, 3) ground truth water positions
        threshold: Distance threshold in Angstroms for considering a match

    Returns:
        dict with keys: 'precision', 'recall', 'f1', 'auc_pr'
    """
    from sklearn.metrics import auc, precision_recall_curve

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    if pred.size == 0 or true.size == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc_pr': 0.0
        }

    # Compute pairwise distances
    D = spdist.cdist(true, pred)  # [N_true, N_pred]

    # For fixed threshold
    recall = float((D.min(axis=1) <= threshold).mean())  # fraction of true covered
    precision = float((D.min(axis=0) <= threshold).mean())  # fraction of pred with nearby true
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # For AUC-PR: sweep over thresholds
    thresholds = np.linspace(0.1, 3.0, 50)
    precisions = []
    recalls = []

    for t in thresholds:
        r = float((D.min(axis=1) <= t).mean())
        p = float((D.min(axis=0) <= t).mean())
        recalls.append(r)
        precisions.append(p)

    # Sort by recall for proper AUC computation
    sorted_indices = np.argsort(recalls)
    recalls_sorted = np.array(recalls)[sorted_indices]
    precisions_sorted = np.array(precisions)[sorted_indices]

    # Compute AUC using trapezoidal rule
    auc_pr = auc(recalls_sorted, precisions_sorted)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_pr': auc_pr
    }


def create_trajectory_gif(trajectory, protein_pos, water_true, save_path, title="", fps=10, pdb_id=None):
    """
    Create a GIF from a trajectory of water positions.

    Args:
        trajectory: List of (N_water, 3) arrays at each timestep
        protein_pos: (N_protein, 3) protein positions
        water_true: (N_water, 3) ground truth water positions
        save_path: Path to save the GIF
        title: Title prefix for frames
        fps: Frames per second in the GIF
        pdb_id: PDB ID to include in title
    """
    from PIL import Image

    frames = []

    # Compute fixed axis limits based on all data
    all_coords = [protein_pos, water_true]
    all_coords.extend(trajectory)
    all_points = np.vstack([c for c in all_coords if c.size > 0])

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    # Add 10% padding
    ranges = maxs - mins
    xlim = (mins[0] - 0.1 * ranges[0], maxs[0] + 0.1 * ranges[0])
    ylim = (mins[1] - 0.1 * ranges[1], maxs[1] + 0.1 * ranges[1])
    zlim = (mins[2] - 0.1 * ranges[2], maxs[2] + 0.1 * ranges[2])

    # Sample frames to keep GIF size reasonable (max 50 frames)
    step = max(1, len(trajectory) // 50)

    for i in range(0, len(trajectory), step):
        water_pred = trajectory[i]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Create title with PDB ID if available
        frame_title = f"{title} Step {i}/{len(trajectory)-1}"
        if pdb_id:
            frame_title = f"{pdb_id} | {frame_title}"

        plot_3d_frame(
            ax,
            protein_pos,
            None,
            water_pred,
            water_true,
            title=frame_title,
            xlim=xlim,
            ylim=ylim,
            zlim=zlim
        )

        # Convert figure to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Drop alpha channel to get RGB
        frames.append(Image.fromarray(img))
        plt.close(fig)

    # Save as GIF
    if frames:
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0
        )

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