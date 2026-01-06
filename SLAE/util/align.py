import torch

def aligned_mse_loss(
    self,
    coords,         # [B, N, 3]
    coords_true, 
    mask=None, 
    weight=None,    # [B, N] for per-atom weighting
    compute_rmsd=False
):
    """
    Computes MSE (or RMSD) for two batched sets of coordinates `coords` and `coords_true`
    after aligning them via an optimal rotation (Kabsch). Skips any batch item that
    has zero valid residues in `mask`.
    """
    B = coords.shape[0]
    c1 = coords.reshape(B, -1, 3)
    c2 = coords_true.reshape(B, -1, 3)

    # Prepare mask
    if mask is None:
        mask = torch.ones_like(c1, dtype=torch.bool, device=c1.device)
    else:
        # Expand [B, N] to [B, N*3], then to [B, N*3, 3]
        mask = mask.unsqueeze(-1).repeat_interleave(3, dim=1).expand(-1, -1, 3).bool().to(c1.device)
        assert mask.shape == c1.shape, f"mask shape: {mask.shape}, c1 shape: {c1.shape}"

    device = c1.device
    # [B, N*3, 3]
    P = c1.transpose(1,2)  # => [B, 3, N*3]
    Q = c2.transpose(1,2)  # => [B, 3, N*3]

    # Sum of valid atoms per batch
    mask_sum = mask.sum(dim=(1,2))  # [B], total number of valid atoms

    # Pre-allocate the final MSE or RMSD
    # We'll store a [B, *] shape so each batch item gets a value
    if compute_rmsd:
        # One RMSD scalar per batch
        output = torch.zeros((B,), device=device)
    else:
        # We'll store the full [B, 3, N*3] of squared diffs
        output = torch.zeros_like(P)

    for b in range(B):
        if mask_sum[b] < 1e-8:
            # Completely empty mask for this batch item -> skip alignment
            # Could set output[b] = 0 or NaN or just leave it zero.
            # We'll just leave it as zeros.
            continue

        # Extract the valid piece for this item
        Mb = mask[b]         # shape [N*3, 3]
        Pb = P[b]            # shape [3, N*3]
        Qb = Q[b]

        # Expand Mb to [3, N*3] for easy multiplication
        Mb_t = Mb.transpose(0,1).float()  # [3, N*3]
        mask_len = Mb.sum()

        # Compute centroid of the valid coords
        Pb_mean = (Pb * Mb_t).sum(dim=1, keepdim=True) / mask_len
        Qb_mean = (Qb * Mb_t).sum(dim=1, keepdim=True) / mask_len

        # Center
        Pb_centered = (Pb - Pb_mean) * Mb_t
        Qb_centered = (Qb - Qb_mean) * Mb_t

        # Cov
        cov = torch.matmul(Pb_centered, Qb_centered.transpose(0,1))  # [3,3]

        with torch.no_grad():
            # SVD for rotation
            U, S, V = torch.svd(cov)
            d = torch.eye(3, device=device)
            d[-1, -1] = torch.det(torch.matmul(V, U.transpose(0,1)))
            rot = torch.matmul(torch.matmul(V, d), U.transpose(0,1))

        # Rotate Pb_centered
        rot_Pb = torch.matmul(rot, Pb_centered)

        # MSE or RMSD
        diff = Qb_centered - rot_Pb  # [3, N*3]
        sq_diff = diff.pow(2) * Mb_t
        if compute_rmsd:
            # sum over dims -> scalar
            # add small value to avoid NaN in sqrt(0)
            val = (sq_diff.sum() + 1e-6).sqrt()
            output[b] = val
        else:
            output[b] = sq_diff  # store the full [3, N*3] difference

    return output


def aligned_rotation(
    reconstructed_bb: torch.Tensor,
    true_bb: torch.Tensor,
    predicted_sidechain: torch.Tensor,
    mask: torch.Tensor = None
):
    """
    Align reconstructed backbone coords to true backbone by Kabsch, 
    skipping items where the mask is completely empty. 
    Those items get an identity transformation (no rotation).
    """
    device = reconstructed_bb.device
    B, N_res, _, _ = reconstructed_bb.shape
    assert true_bb.shape == reconstructed_bb.shape, f"true_bb shape: {true_bb.shape}, reconstructed_bb shape: {reconstructed_bb.shape}"
    assert predicted_sidechain.shape == (B, N_res, 34, 3), f"predicted_sidechain shape: {predicted_sidechain.shape}"
    assert mask is None or mask.shape == (B, N_res), f"mask shape: {mask.shape}"

    # Flatten backbone coords
    c1 = reconstructed_bb.reshape(B, -1, 3)  # [B, N_res*3, 3]
    c2 = true_bb.reshape(B, -1, 3)           # [B, N_res*3, 3]

    # If mask is None, treat all as valid
    if mask is None:
        mask = torch.ones_like(c1, dtype=torch.bool)

    else:
        # Expand from [B, N_res] to [B, N_res*3, 3]
        mask = mask.unsqueeze(-1).repeat_interleave(3, dim=1).expand(-1, -1, 3).bool().to(device)
        assert mask.shape == c1.shape, f"mask shape: {mask.shape}, c1 shape: {c1.shape}"

    # We'll store the final rotation for each batch
    rot_mats = torch.empty((B, 3, 3), device=device)
    # We'll store the "rotated backbone" and "rotated sidechain"
    rotated_bb = torch.empty_like(reconstructed_bb)
    rotated_sc = torch.empty_like(predicted_sidechain)

    # Summation of valid atoms per item
    mask_sum = mask.sum(dim=(1,2))  # [B]

    for b in range(B):
        if mask_sum[b] < 1e-8:
            # No valid residue in this batch item => identity transform
            rot_mats[b] = torch.eye(3, device=device)
            # Keep them as is
            rotated_bb[b] = reconstructed_bb[b]
            rotated_sc[b] = predicted_sidechain[b]
            continue

        c1_b = c1[b]  # shape [N_res*3, 3]
        c2_b = c2[b]
        m_b  = mask[b]  # shape [N_res*3, 3]

        # Sum up valid
        m_count = m_b.sum()

        # Compute centroids
        # transpose(0,1) => [3, N_res*3]
        c1_b_t = c1_b.transpose(0,1)
        c2_b_t = c2_b.transpose(0,1)
        m_b_t  = m_b.transpose(0,1).float()

        P_mean = (c1_b_t * m_b_t).sum(dim=1, keepdim=True) / m_count
        Q_mean = (c2_b_t * m_b_t).sum(dim=1, keepdim=True) / m_count

        # Center
        P = (c1_b_t - P_mean) * m_b_t
        Q = (c2_b_t - Q_mean) * m_b_t

        # Covariance
        cov = torch.matmul(P, Q.transpose(0,1))  # [3,3]

        with torch.no_grad():
            U, S, V = torch.svd(cov)
            d = torch.eye(3, device=device)
            d[-1, -1] = torch.det(torch.matmul(V, U.transpose(0,1)))
            rot = torch.matmul(torch.matmul(V, d), U.transpose(0,1))
        rot_mats[b] = rot

        # Rotate the predicted backbone
        # shape => [3, N_res*3]
        c1_centered = c1_b_t - P_mean
        rotated_bb_flat = torch.matmul(rot, c1_centered) + Q_mean
        rotated_bb[b] = rotated_bb_flat.transpose(0,1).view(N_res, 3, 3)

        # Rotate predicted sidechain
        sc_flat = predicted_sidechain[b].reshape(-1, 3).transpose(0,1)  # [3, N_res*34]
        sc_centered = sc_flat - P_mean  # same shift as backbone
        rotated_sc_flat = torch.matmul(rot, sc_centered) + Q_mean
        rotated_sc[b] = rotated_sc_flat.transpose(0,1).view(N_res, 34, 3)

    return rotated_bb, rotated_sc