# utils.py
"""
Utility functions organized by category:
1. Feature encoding (rbf, atom37_to_atoms)
2. Optimal transport (ot_coupling)
3. Metrics (recall_precision, compute_rmsd, compute_placement_metrics)
4. Visualization (plot_3d_frame, create_trajectory_gif, save_protein_plot)
"""

from __future__ import annotations

from collections.abc import Sequence
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as spdist
import torch
from e3nn.math import soft_one_hot_linspace
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from tqdm import tqdm

from src.constants import NUM_RBF, RBF_CUTOFF

ATOM37_FILL = 1e-5


def setup_logging_for_tqdm(
    level: str = "INFO",
    log_file: str | None = None,
):
    """
    Configure loguru to work with tqdm progress bars.

    Redirects log output through tqdm.write() so log messages don't
    break progress bar rendering. Call this once at the start of
    scripts that use both logging and tqdm.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file for persistent logging
    """
    from pathlib import Path

    from loguru import logger

    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        level=level.upper(),
        colorize=True,
        format="<level>{level: <8}</level> | <level>{message}</level>",
    )
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(str(log_path), level=level.upper(), enqueue=True)


def rbf(r: Tensor, num_gaussians: int = NUM_RBF, cutoff: float = RBF_CUTOFF) -> Tensor:
    """
    Compute radial basis function encoding of distances.

    Uses Bessel basis functions with smooth cutoff for distance encoding.

    Args:
        r: (N,) distance tensor in Angstroms
        num_gaussians: Number of RBF basis functions
        cutoff: Maximum distance in Angstroms (values beyond are zeroed)

    Returns:
        (N, num_gaussians) RBF feature tensor
    """
    r = r.clamp(min=1e-4)
    return soft_one_hot_linspace(
        r, start=0.0, end=cutoff, number=num_gaussians, basis="bessel", cutoff=True
    )


def atom37_to_atoms(
    atom_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert atom37 representation to flat atom list.

    Args:
        atom_tensor: (N_res, 37, 3) atom37 coordinates

    Returns:
        coords: (N_atoms, 3) coordinates of present atoms
        residue_index: (N_atoms,) which residue each atom belongs to
        atom_type: (N_atoms,) atom type index (0-36)
    """
    present = (atom_tensor != ATOM37_FILL).any(dim=-1)  # (N_res, 37)
    nz = present.nonzero(as_tuple=False)  # (N_atoms, 2)
    residue_index = nz[:, 0]
    atom_type = nz[:, 1].long()

    flat = atom_tensor.reshape(-1, 3)
    flat_mask = present.reshape(-1)
    coords = flat[flat_mask]

    return coords, residue_index, atom_type


@torch.no_grad()
def ot_coupling(
    x1: torch.Tensor,
    batch: torch.Tensor,
    x0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hard OT (Hungarian) pairing per graph for flow matching.

    Args:
        x1: (M, 3) target positions (e.g., true water positions)
        batch: (M,) graph ID per point
        x0: (M, 3) source positions (e.g., Gaussian samples)

    Returns:
        x0_star: (M, 3) source positions (unchanged)
        x1_star: (M, 3) x1 permuted to match x0 per graph
    """
    x0_star = torch.empty_like(x1)
    x1_star = torch.empty_like(x1)

    for g in torch.unique(batch):
        m = batch == g
        if m.sum() == 0:
            continue

        X1 = x1[m]
        X0 = x0[m]

        C = torch.cdist(X0, X1, p=2).pow(2).cpu().numpy()
        _, c = linear_sum_assignment(C)
        X1_match = X1[c]

        x0_star[m] = X0
        x1_star[m] = X1_match

    return x0_star, x1_star


# eval metric functions


@torch.no_grad()
def recall_precision(
    pred: torch.Tensor | np.ndarray,
    true: torch.Tensor | np.ndarray,
    thresh: float = 1.0,
) -> tuple[float, float]:
    """
    Compute recall and precision for point set matching.

    GPU-native when given torch tensors on GPU, falls back to CPU for numpy.

    Args:
        pred: (N_pred, 3) predicted positions
        true: (N_true, 3) ground truth positions
        thresh: distance threshold in Angstroms

    Returns:
        recall: fraction of true points with a prediction within thresh
        precision: fraction of predictions within thresh of a true point
    """
    # handle empty inputs
    if isinstance(pred, np.ndarray):
        if pred.size == 0 or true.size == 0:
            return 0.0, 0.0
        pred = torch.from_numpy(pred)
        true = torch.from_numpy(true)
    else:
        if pred.numel() == 0 or true.numel() == 0:
            return 0.0, 0.0

    # ensure same device
    if pred.device != true.device:
        true = true.to(pred.device)

    D = torch.cdist(true.float(), pred.float(), p=2)

    recall = (D.min(dim=1)[0] <= thresh).float().mean().item()
    precision = (D.min(dim=0)[0] <= thresh).float().mean().item()

    return recall, precision


@torch.no_grad()
def compute_rmsd(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    batch: torch.Tensor | np.ndarray | None = None,
) -> float:
    """
    Compute RMSD with optimal assignment using Hungarian algorithm.

    This is the CPU version for evaluation. For training, use scatter_mean
    directly on GPU (see flow.py training_step).

    Args:
        pred: (N, 3) predicted positions
        target: (N, 3) target positions
        batch: optional (N,) batch indices for per-graph computation

    Returns:
        RMSD value (averaged over graphs if batch provided)
    """
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


def compute_placement_metrics(
    pred: torch.Tensor | np.ndarray,
    true: torch.Tensor | np.ndarray,
    threshold: float = 1.0,
) -> dict[str, float]:
    """
    Compute standard metrics for water placement evaluation.

    Args:
        pred: (N_pred, 3) predicted water positions
        true: (N_true, 3) ground truth water positions
        threshold: distance threshold in Angstroms for point matching

    Returns:
        dict with 'precision', 'recall', 'f1', 'auc_pr'
    """
    from sklearn.metrics import auc

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    if pred.size == 0 or true.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auc_pr": 0.0}

    D = spdist.cdist(true, pred)

    # metrics at fixed threshold
    recall = float((D.min(axis=1) <= threshold).mean())
    precision = float((D.min(axis=0) <= threshold).mean())
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # sweep over thresholds
    thresholds = np.linspace(0.1, 3.0, 50)
    recalls, precisions = [], []

    for t in thresholds:
        r = float((D.min(axis=1) <= t).mean())
        p = float((D.min(axis=0) <= t).mean())
        recalls.append(r)
        precisions.append(p)

    sorted_idx = np.argsort(recalls)
    auc_pr = auc(np.array(recalls)[sorted_idx], np.array(precisions)[sorted_idx])

    return {"precision": precision, "recall": recall, "f1": f1, "auc_pr": auc_pr}


# viz functions
def plot_3d_frame(
    ax,
    protein_pos: np.ndarray,
    mate_pos: np.ndarray,
    water_pred: np.ndarray,
    water_true: np.ndarray,
    title: str = "",
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None,
):
    """
    Plot a single 3D frame showing protein structure and water positions.

    Args:
        ax: Matplotlib 3D axes object
        protein_pos: (N_p, 3) protein atom coordinates
        mate_pos: (N_m, 3) symmetry mate coordinates, or None
        water_pred: (N_w, 3) predicted water positions
        water_true: (N_w, 3) ground truth water positions
        title: Plot title string
        xlim: Optional (min, max) x-axis limits in Angstroms
        ylim: Optional (min, max) y-axis limits in Angstroms
        zlim: Optional (min, max) z-axis limits in Angstroms
    """
    ax.clear()

    if protein_pos.size > 0:
        ax.scatter(
            protein_pos[:, 0],
            protein_pos[:, 1],
            protein_pos[:, 2],
            c="gray",
            alpha=0.3,
            s=8,
            label="Protein",
        )

    if mate_pos is not None and mate_pos.size > 0:
        ax.scatter(
            mate_pos[:, 0],
            mate_pos[:, 1],
            mate_pos[:, 2],
            c="dimgrey",
            alpha=0.6,
            s=10,
            label="Mates",
        )

    ax.scatter(
        water_true[:, 0],
        water_true[:, 1],
        water_true[:, 2],
        c="red",
        marker="*",
        alpha=0.9,
        s=16,
        label="True Water",
    )

    ax.scatter(
        water_pred[:, 0],
        water_pred[:, 1],
        water_pred[:, 2],
        c="blue",
        alpha=0.9,
        s=14,
        label="Predicted Water",
    )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)


def create_trajectory_gif(
    trajectory: Sequence[np.ndarray],
    protein_pos: np.ndarray,
    water_true: np.ndarray,
    save_path: str,
    title: str = "",
    fps: int = 10,
    pdb_id: str = None,
):
    """
    Create a GIF from a trajectory of water positions.

    Args:
        trajectory: list of (N_water, 3) arrays at each timestep
        protein_pos: (N_protein, 3) protein positions
        water_true: (N_water, 3) ground truth water positions
        save_path: path to save the GIF
        title: title prefix for frames
        fps: frames per second
        pdb_id: PDB ID to include in title
    """
    frames = []

    # compute fixed axis limits
    all_coords = [protein_pos, water_true] + list(trajectory)
    all_points = np.vstack([c for c in all_coords if c.size > 0])

    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    ranges = maxs - mins
    xlim = (mins[0] - 0.1 * ranges[0], maxs[0] + 0.1 * ranges[0])
    ylim = (mins[1] - 0.1 * ranges[1], maxs[1] + 0.1 * ranges[1])
    zlim = (mins[2] - 0.1 * ranges[2], maxs[2] + 0.1 * ranges[2])

    # sample frames (max 100)
    step = max(1, len(trajectory) // 100)
    frame_indices = list(range(0, len(trajectory), step))
    if frame_indices and frame_indices[-1] != len(trajectory) - 1:
        frame_indices.append(len(trajectory) - 1)

    for i in frame_indices:
        water_pred = trajectory[i]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        frame_title = f"{title} Step {i}/{len(trajectory) - 1}"
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
            zlim=zlim,
        )

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]
        frames.append(Image.fromarray(img))
        plt.close(fig)

    if frames:
        last_frame_copies = [frames[-1]] * 30
        all_frames = frames + last_frame_copies
        all_frames[0].save(
            save_path,
            save_all=True,
            append_images=all_frames[1:],
            duration=1000 // fps,
            loop=0,
        )


def save_protein_plot(
    pred_ca: torch.Tensor,
    true_ca: torch.Tensor,
    step: int,
    save_dir: str,
):
    """
    Save 3D plot of predicted vs true CA traces with Kabsch alignment.

    Args:
        pred_ca: (N_res, 3) predicted CA coordinates
        true_ca: (N_res, 3) ground truth CA coordinates
        step: Step number for filename
        save_dir: Directory to save plot image
    """

    P = pred_ca.detach().float().cpu().numpy()
    Q = true_ca.detach().float().cpu().numpy()

    # center
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)

    # kabsch alignment
    H = np.dot(P.T, Q)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    P_aligned = np.dot(P, R)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        Q[:, 0],
        Q[:, 1],
        Q[:, 2],
        color="black",
        linewidth=2,
        label="Ground Truth",
        alpha=0.6,
    )
    ax.plot(
        P_aligned[:, 0],
        P_aligned[:, 1],
        P_aligned[:, 2],
        color="red",
        linewidth=2,
        label="Prediction",
    )
    ax.legend()
    ax.set_title(f"Step {step}")
    plt.savefig(f"{save_dir}/step_{step}.png")
    plt.close()
