import torch
import torch.nn.functional as F
from loguru import logger

def debug_residue_embedding_differences(embeddings: torch.Tensor, mask: torch.Tensor = None):
    """
    Monitors the diversity of residue embeddings by computing pairwise similarities and distances.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape [B, L, D] where B is batch size, 
            L is the sequence length (number of residues), and D is the embedding dimension.
        mask (torch.Tensor, optional): Boolean tensor of shape [B, L] indicating which residues
            are valid (True for valid residues, False for padded/invalid ones). If None, all
            residues are considered valid.
            
    Logs:
        For each batch item, prints:
          - Mean off-diagonal cosine similarity.
          - Mean off-diagonal Euclidean distance.
          - Mean variance per embedding dimension.
    """
    B, L, D = embeddings.shape

    for b in range(B):
        # Select only valid embeddings if a mask is provided
        if mask is not None:
            valid_idx = mask[b].nonzero(as_tuple=False).squeeze(1)
            if valid_idx.numel() < 2:
                logger.info(f"Batch item {b} has less than 2 valid residues. Skipping pairwise computation.")
                continue
            emb = embeddings[b, valid_idx, :]  # shape [n_valid, D]
        else:
            emb = embeddings[b]  # shape [L, D]
        
        n = emb.size(0)
        
        # Normalize embeddings for cosine similarity computation
        emb_norm = F.normalize(emb, p=2, dim=-1)
        # Compute cosine similarity matrix: shape [n, n]
        cos_sim_matrix = emb_norm @ emb_norm.transpose(0, 1)
        
        # Create a mask to select off-diagonal elements
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
        mean_cos_sim = cos_sim_matrix[off_diag_mask].mean().item()
        
        # Compute pairwise Euclidean distances
        # Compute differences: shape [n, n, D]
        diff = emb.unsqueeze(1) - emb.unsqueeze(0)
        # Compute Euclidean distances (add a small epsilon to avoid sqrt(0))
        euclid_dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        mean_euclid = euclid_dist[off_diag_mask].mean().item()
        
        # Compute the variance per embedding dimension and average over dimensions
        variance_per_dim = emb.var(dim=0)  # shape [D]
        mean_variance = variance_per_dim.mean().item()
        
        logger.info(
            f"Batch item {b}: Mean off-diagonal cosine similarity = {mean_cos_sim:.4f}, "
            f"Mean off-diagonal Euclidean distance = {mean_euclid:.4f}, "
            f"Mean variance per dimension = {mean_variance:.4f}"
        )
def debug_edge_embedding_differences(edge_embeddings: torch.Tensor):
    """
    Monitors the diversity of edge embeddings by computing pairwise similarities and distances.

    Args:
        edge_embeddings (torch.Tensor): Tensor of shape [E, D] where E is the number of edges 
            and D is the embedding dimension.
        edge_index (torch.Tensor): Tensor of shape [2, E] containing the indices of nodes 
            connected by each edge.

    Logs:
        - Mean off-diagonal cosine similarity.
        - Mean off-diagonal Euclidean distance.
        - Mean variance per embedding dimension.
        - Per-edge standard deviation to detect collapsed representations.
    """
    E, D = edge_embeddings.shape

    if E < 2:
        logger.info("Not enough edges for pairwise analysis. Skipping.")
        return

    # Normalize embeddings for cosine similarity computation
    emb_norm = F.normalize(edge_embeddings, p=2, dim=-1)

    # Compute cosine similarity matrix: shape [E, E]
    cos_sim_matrix = emb_norm @ emb_norm.T

    # Create a mask to select off-diagonal elements
    off_diag_mask = ~torch.eye(E, dtype=torch.bool, device=edge_embeddings.device)
    mean_cos_sim = cos_sim_matrix[off_diag_mask].mean().item()

    # Compute pairwise Euclidean distances
    diff = edge_embeddings.unsqueeze(1) - edge_embeddings.unsqueeze(0)  # shape [E, E, D]
    euclid_dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # shape [E, E]
    mean_euclid = euclid_dist[off_diag_mask].mean().item()

    # Compute the variance per embedding dimension and average over dimensions
    variance_per_dim = edge_embeddings.var(dim=0)  # shape [D]
    mean_variance = variance_per_dim.mean().item()

    # Per-edge standard deviation (detecting collapsed representations)
    per_edge_std = edge_embeddings.std(dim=-1)  # shape [E]
    mean_std_per_edge = per_edge_std.mean().item()
    min_std, max_std = per_edge_std.min().item(), per_edge_std.max().item()

    logger.info(
        f"Node embeddings: Mean off-diagonal cosine similarity = {mean_cos_sim:.4f}, "
        f"Mean off-diagonal Euclidean distance = {mean_euclid:.4f}, "
        f"Mean variance per dimension = {mean_variance:.4f}, "
        f"Mean per-edge std = {mean_std_per_edge:.4f} (min: {min_std:.4f}, max: {max_std:.4f})"
    )
# Example usage:
if __name__ == "__main__":
    # Simulate a batch of residue embeddings: batch size = 2, sequence length = 10, embedding dim = 128
    B, L, D = 2, 10, 128
    embeddings = torch.randn(B, L, D)
    # For demonstration, assume all residues are valid
    mask = torch.ones(B, L, dtype=torch.bool)
    debug_residue_embedding_differences(embeddings, mask)