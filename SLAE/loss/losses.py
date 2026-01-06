import torch
import torch.nn.functional as F
from loguru import logger
from SLAE.util.constants import fill_in_cbeta_for_atom3_all
import einx
from SLAE.util.rigid_utils import Rigid
from SLAE.util.constants import (
    ATOM_ORDER,
    RESTYPES,
    RESTYPE_1TO3,
    CHI_ANGLES_ATOMS,
    CHI_ANGLES_MASK,
    CHI_PI_PERIODIC,
)

def backbone_distance_loss(
    pred: torch.Tensor,      # [Res*3, 3]
    true: torch.Tensor,      # [Res*3, 3]
    batch_indices: torch.Tensor,  # [Res*3], batch index for each atom
    B: int,                  # number of batch items
    clamp: float = 25.0      # squared distance clamp => (5 Å)^2 = 25
) -> torch.Tensor:
    assert not torch.isnan(pred).any(), "NaN detected in pred"
    assert not torch.isnan(true).any(), "NaN detected in true"
    assert not torch.isnan(batch_indices).any(), "NaN detected in batch_indices"
    device = pred.device
 
   
    # If no atoms remain, return 0.
    n_atoms_total = pred.size(0)
    assert n_atoms_total == batch_indices.size(0), f"{n_atoms_total} != {batch_indices.size(0)}"
    if n_atoms_total < 2:
        return torch.tensor(0.0, device=device)

    # 3) Pairwise squared distances among all valid atoms
    D_pred = torch.cdist(pred, pred, p=2).square()  # [S, S]
    D_true = torch.cdist(true, true, p=2).square()

    # 4) We only keep pairs (i,j) if they belong to the same batch item.
    #    same_sample_mask is [S,S] with True if batch_indices[i] == batch_indices[j].
    same_sample_mask = (batch_indices.unsqueeze(1) == batch_indices.unsqueeze(0))

    # (D_pred - D_true)^2 => clamp => zero out cross-batch pairs
    E = (D_pred - D_true).pow(2).clamp(max=clamp)
    E[~same_sample_mask] = 0.0

    # 5) Sum up all valid pairs, then normalize by sum_{b}( n_b^2 )
    #    where n_b is the # of valid atoms for batch b.
    # n_b = count of batch_indices == b
    n_atoms_per_batch = torch.bincount(batch_indices, minlength=B)  # shape [B]
    # Each batch's pair count includes i=j => n_b^2 total pairs
    pair_counts = (n_atoms_per_batch * n_atoms_per_batch).sum().float()
    assert pair_counts > 0, f"Invalid pair_counts: {pair_counts}"
    if pair_counts < 1.0:
        return torch.tensor(0.0, device=device)

    E_sum = E.sum()  # sum of error across all valid pairs
    return E_sum / pair_counts

        
def backbone_pairwise_direction_loss(gt_bb_coords: torch.Tensor | None=None,
                                    pred_bb_coords: torch.Tensor | None=None, 
                                    seq_mask: torch.Tensor | None=None,
                                    loss_clamp: int = 20,
                                    ce_loss: bool = False,
                                    norm_dir_loss: bool = True):    
    assert gt_bb_coords.shape[2] == 3 and pred_bb_coords.shape[2] == 3, "Input only require N, CA, C"
    
    bsz, seq_len, _, _ = gt_bb_coords.shape
        
    gt_NCA = gt_bb_coords[:, :, 1] - gt_bb_coords[:, :, 0]        
    gt_CAC = gt_bb_coords[:, :, 2] - gt_bb_coords[:, :, 1]
    gt_CN = gt_bb_coords[:, 1:, 0] - gt_bb_coords[:, :-1, 2]        
    
    #! Originally, gt_Cprev_N was gt_C_Nnext and gt_C_Nnext was gt_Cprev_N.
    #! So gt_normal_N and gt_normal_C were calculated wrong.
    gt_Cprev_N = torch.cat([torch.zeros(bsz, 1, 3, device=gt_bb_coords.device), gt_CN], dim=1)
    gt_C_Nnext = torch.cat([gt_CN, torch.zeros(bsz, 1, 3, device=gt_bb_coords.device)], dim=1)                
    
    # Normalizing the vectors
    if norm_dir_loss:
        gt_NCA = gt_NCA / (gt_NCA.norm(dim=-1, keepdim=True) + 1e-6)
        gt_CAC = gt_CAC / (gt_CAC.norm(dim=-1, keepdim=True) + 1e-6)
        gt_CN = gt_CN / (gt_CN.norm(dim=-1, keepdim=True) + 1e-6)
        gt_C_Nnext = gt_C_Nnext / (gt_C_Nnext.norm(dim=-1, keepdim=True) + 1e-6)
        gt_Cprev_N = gt_Cprev_N / (gt_Cprev_N.norm(dim=-1, keepdim=True) + 1e-6)        
    
    # normal vectors for the planes at each atom
    gt_normal_CA = torch.linalg.cross(-gt_NCA, gt_CAC, dim=-1)
    gt_normal_N = torch.linalg.cross(gt_Cprev_N, gt_NCA, dim=-1)
    gt_normal_C = torch.linalg.cross(gt_CAC, gt_C_Nnext, dim=-1)
    
    # Normalizing the vectors
    if norm_dir_loss:
        gt_normal_CA = gt_normal_CA / (gt_normal_CA.norm(dim=-1, keepdim=True) + 1e-6)
        gt_normal_N = gt_normal_N / (gt_normal_N.norm(dim=-1, keepdim=True) + 1e-6)
        gt_normal_C = gt_normal_C / (gt_normal_C.norm(dim=-1, keepdim=True) + 1e-6)
    
    gt_dirc_vectors = torch.cat([gt_NCA, gt_CAC, gt_C_Nnext, gt_normal_CA, gt_normal_N, gt_normal_C], dim=1) #! changed
    
    # direction vectors for predicted coordinates                         
    pred_NCA = pred_bb_coords[:, :, 1] - pred_bb_coords[:, :, 0]
    pred_CAC = pred_bb_coords[:, :, 2] - pred_bb_coords[:, :, 1]
    pred_CN = pred_bb_coords[:, 1:, 0] - pred_bb_coords[:, :-1, 2]
    pred_Cprev_N = torch.cat([torch.zeros(bsz, 1, 3, device=pred_bb_coords.device), pred_CN], dim=1)
    pred_C_Nnext = torch.cat([pred_CN, torch.zeros(bsz, 1, 3, device=pred_bb_coords.device)], dim=1)
            
    # Normalizing the vectors
    if norm_dir_loss:
        pred_NCA = pred_NCA / (pred_NCA.norm(dim=-1, keepdim=True) + 1e-6)
        pred_CAC = pred_CAC / (pred_CAC.norm(dim=-1, keepdim=True) + 1e-6)
        pred_CN = pred_CN / (pred_CN.norm(dim=-1, keepdim=True) + 1e-6)
        pred_C_Nnext = pred_C_Nnext / (pred_C_Nnext.norm(dim=-1, keepdim=True) + 1e-6)
        pred_Cprev_N = pred_Cprev_N / (pred_Cprev_N.norm(dim=-1, keepdim=True) + 1e-6)
    
    # normal vectors for the planes at each atom
    pred_normal_CA = torch.linalg.cross(-pred_NCA, pred_CAC, dim=-1)
    pred_normal_N = torch.linalg.cross(pred_Cprev_N, pred_NCA, dim=-1)
    pred_normal_C = torch.linalg.cross(pred_CAC, pred_C_Nnext, dim=-1)
    
    # Normalizing the vectors
    if norm_dir_loss:
        pred_normal_CA = pred_normal_CA / (pred_normal_CA.norm(dim=-1, keepdim=True) + 1e-6)
        pred_normal_N = pred_normal_N / (pred_normal_N.norm(dim=-1, keepdim=True) + 1e-6)
        pred_normal_C = pred_normal_C / (pred_normal_C.norm(dim=-1, keepdim=True) + 1e-6)
    
    pred_dirc_vectors = torch.cat([pred_NCA, pred_CAC, pred_C_Nnext, pred_normal_CA, pred_normal_N, pred_normal_C], dim=1)
    
    gt_dot_matrix = torch.einsum('bik,bjk->bij', gt_dirc_vectors, gt_dirc_vectors)
    pred_dot_matrix = torch.einsum('bik,bjk->bij', pred_dirc_vectors, pred_dirc_vectors)
    
    diff_direction = (gt_dot_matrix - pred_dot_matrix) ** 2
    diff_direction = torch.clamp(diff_direction, max=loss_clamp)
    
    mask_dir = seq_mask.repeat(1, 6)[:, :, None] * seq_mask.repeat(1, 6)[:, None, :]
    
    #! masking out the padded parts
    diff_direction *= mask_dir
            
    bb_dir_loss = diff_direction.sum(dim=(1, 2)) / mask_dir.sum(dim=(1, 2))
    #AVOID NAN
    #denom = mask_dir.sum(dim=(1,2))
    #numer = diff_direction.sum(dim=(1,2))
    # Replace zero denominators with 1 so we don’t do 0/0
    #denom_clamped = torch.where(denom == 0, torch.ones_like(denom), denom)
    #sample_loss = numer / denom_clamped
    # For truly empty samples, we want the fraction to be 0
    #sample_loss = torch.where(denom == 0, torch.zeros_like(sample_loss), sample_loss)
    #bb_dir_loss = sample_loss
    #logger.info(f"seq_mask sum = {seq_mask.sum()}")
    #logger.info(f"mask_dir sum = {mask_dir.sum()}")


    if not ce_loss:
        mean_bb_dir_loss = bb_dir_loss.mean()
        #logger.info(f"mean_bb_dir_loss is {mean_bb_dir_loss}")
        return mean_bb_dir_loss # TODO average over all proteins
    else:
        return bb_dir_loss, {"gt_NCA": gt_NCA, "gt_CAC": gt_CAC, "gt_normal_CA": gt_normal_CA}




def backbone_pairwise_direction_classification_loss(vector_dicts: dict = None,
                                                    seq_mask = None,
                                                    binned_dir_logits = None, # c = direction_loss_bins * # of dot products
                                                    num_bins: int = None,
                                                    ):
        

    gt_NCA = vector_dicts["gt_NCA"] # b n 3
    gt_CAC = vector_dicts["gt_CAC"] # b n 3
    gt_normal_CA = vector_dicts["gt_normal_CA"] # b n 3, 
        
    # Compute pairwise dot products between each pair of vectors            
    # NCA, CAC
    dot_NCA_CAC = torch.einsum('bik, bjk->bij', gt_NCA, gt_CAC) # b n n    
    dot_NCA_CAC = dot_NCA_CAC[..., None]
    # NCA, normal_CA
    dot_NCA_normal_CA = torch.einsum('bik, bjk->bij', gt_NCA, gt_normal_CA) # b n n #! changed
    dot_NCA_normal_CA = dot_NCA_normal_CA[..., None]
    # CAC, normal_CA
    dot_CAC_normal_CA = torch.einsum('bik, bjk->bij', gt_CAC, gt_normal_CA) # b n n #! changed
    dot_CAC_normal_CA = dot_CAC_normal_CA[..., None]

    # NCA, NCA
    dot_NCA_NCA = torch.einsum('bik, bjk->bij', gt_NCA, gt_NCA)  # b n n
    dot_NCA_NCA = dot_NCA_NCA[..., None]

    # CAC, CAC
    dot_CAC_CAC = torch.einsum('bik, bjk->bij', gt_CAC, gt_CAC)  # b n n
    dot_CAC_CAC = dot_CAC_CAC[..., None]

    # normal_CA, normal_CA
    dot_normal_CA_normal_CA = torch.einsum('bik, bjk->bij', gt_normal_CA, gt_normal_CA)  # b n n #! changed
    dot_normal_CA_normal_CA = dot_normal_CA_normal_CA[..., None]

    #! caution: if want to consider other dot products as well, 
    #! need to change line 525 of __init__ of Decoder class.

    gt_logits = torch.cat([
			    dot_NCA_NCA,
			    dot_NCA_CAC,
			    dot_NCA_normal_CA,
			    dot_CAC_CAC,
			    dot_CAC_normal_CA,
			    dot_normal_CA_normal_CA
			], dim=-1)  # b n n 6
	
    # binning the dot products
    bin_edges = torch.linspace(-1.0, 1.0, steps=17, device=seq_mask.device)    
    gt_logits = torch.clamp(gt_logits, min=-1.0, max=1.0)    
    gt_labels = torch.bucketize(gt_logits, bin_edges[1:-1], right=False) # b n n 6

    b, n, _, c = gt_labels.shape
	    
    # 2D mask
    mask = seq_mask[:, :, None] * seq_mask[:, None, :] # b n n 
    mask = mask.bool()
    mask_repeated = mask[:, :, :, None].repeat(1, 1, 1, c) # b n n 6
	
    # reshape binned_dir_logits
    binned_dir_logits = binned_dir_logits.reshape(-1, num_bins) # b*n*n*6 16
    gt_labels = gt_labels.view(-1) # b*n*n*6

    # Compute the cross entropy loss
    direction_ce_loss = F.cross_entropy(binned_dir_logits, gt_labels, reduction='none') # b*n*n*6

    # change the shape of loss to (b, n, n, 6)
    direction_ce_loss = direction_ce_loss.reshape(b, n, n, c) # b n n 6
	
    # masking out the padded parts
    direction_ce_loss = (direction_ce_loss * mask_repeated).sum(dim=(1, 2, 3)) / mask_repeated.sum(dim=(1, 2, 3)) # b
	
    return direction_ce_loss # b
"""
def backbone_pairwise_distance_classification_loss(gt_bb_coords: torch.Tensor | None=None,
				    binned_dist_logits: torch.Tensor | None=None,
				    seq_mask: torch.Tensor | None=None):
    
    
    # Get the distogram using the ground truth cbeta coordinates
    gt_coords_with_cb = fill_in_cbeta_for_atom3_all(gt_bb_coords) # b n 4 3
    gt_coords_cb = gt_coords_with_cb[:, :, 3] # b n 3
    
    dvecs = gt_coords_cb[:, :, None, :] - gt_coords_cb[:, None, :, :] # b n n 3
    gt_distogram_cb = dvecs.pow(2).sum(-1) # b n n 
	
    # calculate bin_edges
    starting_edges = torch.tensor([0.0, 2.3125], device=seq_mask.device)
    remaining_edges = torch.linspace(1, 63, steps=63, device=seq_mask.device) * (19.375/63) + 2.3125
    bin_edges = torch.cat((starting_edges, remaining_edges))
    bin_edges = bin_edges ** 2
    
    # binning the distogram    
    gt_distogram_cb = torch.clamp(gt_distogram_cb, max=21.6875**2)
    gt_labels = torch.bucketize(gt_distogram_cb, bin_edges[1:-1], right=False) # b n n
	
    # 2D mask
    mask = seq_mask[:, :, None ] * seq_mask[:, None, :] # b n n    
    mask = mask.bool()
    
    # Flatten the logits and the labels, compute cross entropy loss
    b, n, _, c = binned_dist_logits.shape
    binned_dist_logits = binned_dist_logits.view(-1, c) # b*n*n 64
    gt_labels = gt_labels.view(-1) # b*n*n
    mask_flat = mask.view(-1) # b*n*n
    
    distance_ce_loss = F.cross_entropy(binned_dist_logits, gt_labels, reduction='none') # b*n*n
    
    # change the shape of loss to (b, n, n) and masking out the padded parts
    distance_ce_loss = distance_ce_loss.view(b, n, n) # b n n
    distance_ce_loss = (distance_ce_loss * mask).sum(dim=(0, 1, 2)) / mask.sum(dim=(0, 1, 2)) # b
    	
    return distance_ce_loss
"""


def contact_map_loss(gt_bb_coords: torch.Tensor,
                    pred_contact_logits: torch.Tensor, 
                    seq_mask: torch.Tensor,
                    cutoff: float = 8.0,
                    seq_sep: int = 4) :
    """
    Compute the contact map loss between the predicted and true contact maps
    Args:
        pred_contact_logits: The predicted contact map. Shape [B, L, L, 2] (2 channels for binary cross entropy)
        true: The true contact map. Shape [B, L, L, 2] (2 channels for binary cross entropy)
        seq_mask: The sequence mask. Shape [B, L]"
    """
    # compute the true contact map from the ground truth cbeta coordinates [B, L, L, 2]
    gt_coords_with_cb = fill_in_cbeta_for_atom3_all(gt_bb_coords)
    gt_coords_cb = gt_coords_with_cb[:, :, 3] # b n 3
    dvecs = gt_coords_cb[:, :, None, :] - gt_coords_cb[:, None, :, :] # b n n 3
    gt_distogram_cb = dvecs.pow(2).sum(-1) 

    # assume that the contact map is 1 if the distance is less than the cutoff
    gt_contact_map = (gt_distogram_cb < cutoff ** 2).long()  # b n n
    b, n, _,  c = pred_contact_logits.shape
    #assert gt_contact_map.shape == pred_contact_logits.shape, "The shapes of the predicted and true contact maps should be the same"
    assert c == 2, "The contact map should have 2 channels"

    mask = seq_mask[:, :, None ] * seq_mask[:, None, :] # b n n    
    #mask = mask.bool()

    seq_idx = torch.arange(n, device=seq_mask.device)
    seq_distances = torch.abs(seq_idx[None, :, None] - seq_idx[None, None, :])  # [1, L, L]
    seq_sep_mask = (seq_distances > seq_sep).to(mask.dtype)

    final_mask = (mask * seq_sep_mask).bool()
    gt_flat = gt_contact_map[final_mask]
    pred_flat = pred_contact_logits[final_mask]
    with torch.no_grad():
        pos = (gt_flat == 1).sum()
        neg = (gt_flat == 0).sum()
        weights = torch.tensor([1.0, neg.float() / (pos.float() + 1e-6)],
                            device=pred_flat.device)
    contact_ce_loss = F.cross_entropy(pred_flat, gt_flat, reduction='mean', weight = weights)
    return contact_ce_loss


def contact_consistency_loss(pred_coords: torch.Tensor,
                             pred_contact_logits: torch.Tensor,  # [B,L,L,2]
                             seq_mask: torch.Tensor,
                             cutoff = 8.0,
                             seq_sep = 4,
                             pos_weight = None):
    """
    Penalise disagreement between predicted coordinates and contact logits.
    """
    B, L, _, _ = pred_contact_logits.shape
    CA_IDX = 1  # adjust if your atom order differs
    coords_ca = pred_coords[:, :, CA_IDX, :]                 # [B,L,3]
    pairwise_d = torch.cdist(coords_ca, coords_ca, p=2)      # [B,L,L]

    # --- build contact label from predicted coords -------------------------
    contact_label = (pairwise_d < cutoff).float()            # [B,L,L]

    # --- same masking as in contact_map_loss -------------------------------
    mask = seq_mask[:, :, None] * seq_mask[:, None, :]       # valid residues
    seq_idx = torch.arange(L, device=seq_mask.device)
    seq_sep_mask = (seq_idx[None,:,None] - seq_idx[None,None,:]).abs() > seq_sep
    final_mask = (mask & seq_sep_mask)                       # bool

    # --- extract the **contact logit** from the 2‑channel head -------------
    # logits_contact = log P(contact) - log P(noncontact)
    logits_contact = (pred_contact_logits[..., 1] - pred_contact_logits[..., 0])

    logits_flat  = logits_contact[final_mask]                # [N]
    target_flat  = contact_label[final_mask]                 # [N]

    # --- class weight (optional but recommended) ---------------------------
    if pos_weight is None:
        with torch.no_grad():
            pos = target_flat.sum()
            neg = target_flat.numel() - pos
            pos_weight = neg / (pos + 1e-6)
    loss = F.binary_cross_entropy_with_logits(
                logits_flat, target_flat,
                pos_weight=pos_weight, reduction='mean')
    return loss

def backbone_pairwise_distance_classification_loss(gt_bb_coords: torch.Tensor | None=None,
				    binned_dist_logits: torch.Tensor | None=None,
				    seq_mask: torch.Tensor | None=None):
    
    
    # Get the distogram using the ground truth cbeta coordinates
    gt_coords_with_cb = fill_in_cbeta_for_atom3_all(gt_bb_coords) # b n 4 3
    gt_coords_cb = gt_coords_with_cb[:, :, 3] # b n 3
    
    dvecs = gt_coords_cb[:, :, None, :] - gt_coords_cb[:, None, :, :] # b n n 3
    gt_distogram_cb = dvecs.norm(dim=-1).clamp(max=21.6875) # b n n 
	
    # calculate bin_edges
    starting_edges = torch.tensor([0.0, 2.3125], device=gt_coords_cb.device)
    remaining_edges = torch.linspace(1, 63, steps=63, device=gt_coords_cb.device) * (19.375/63) + 2.3125
    bin_edges = torch.cat((starting_edges, remaining_edges))
    

    gt_labels = torch.bucketize(gt_distogram_cb, bin_edges[1:-1], right=False) # b n n
	
    # 2D mask
    mask = seq_mask[:, :, None ] * seq_mask[:, None, :] # b n n    
    mask = mask.bool()
    
    # Flatten the logits and the labels, compute cross entropy loss
    b, n, _, c = binned_dist_logits.shape
    binned_dist_logits = binned_dist_logits.view(-1, c) # b*n*n 64
    gt_labels = gt_labels.view(-1) # b*n*n
    mask_flat = mask.view(-1) # b*n*n
    distance_ce_loss = F.cross_entropy(binned_dist_logits[mask_flat], gt_labels[mask_flat], reduction='mean') # b*n*n
    
    return distance_ce_loss



def chainbreak_loss(gt_bb_coords: torch.Tensor, 
                    pred_chainbreak_logits: torch.Tensor,
                    seq_mask: torch.Tensor,
                    break_weight = 5.0,
):
    """
    Compute the chainbreak loss between the predicted and true chainbreaks
    Args:
        pred_bb_coords: The predicted backbone coordinates. Shape [B, L, 3, 3]
        seq_mask: The sequence mask. Shape [B, L]
        chainbreak_logits: The chain mask. Shape [B, L, 2]
    """
    # get ground truth chainbreaks : consider there's a chain break if the distance between two consecutive C atoms is greater than 4.2
    max_ca_ca_dist = 4.2


    # first ca 
    first_ca_coords = gt_bb_coords[:, :-1, 1, :] 
    second_ca_coords = gt_bb_coords[:, 1:, 1, :]

    ca_ca_dist = torch.norm(first_ca_coords - second_ca_coords, dim=-1) # b n-1
    gt_chainbreak = (ca_ca_dist > max_ca_ca_dist).long() # b n-1

    # add a zero to the tail of the chainbreak tensor to match the shape of the predicted chainbreaks
    gt_chainbreak = torch.cat([gt_chainbreak, torch.zeros_like(gt_chainbreak[:, -1:])], dim=1) # b n

    gt_chainbreak_flat = gt_chainbreak.view(-1) 
    pred_logits_flat = pred_chainbreak_logits.view(-1, 2) 
    # compute the cross entropy loss then mask out the padded parts
    weights = torch.tensor([1.0, break_weight], device=pred_logits_flat.device)
    chainbreak_ce_loss = F.cross_entropy(pred_logits_flat, gt_chainbreak_flat, reduction='none', weight = weights) # b n
    mask = seq_mask.bool().view(-1)
    chainbreak_ce_loss = (chainbreak_ce_loss * mask).sum() / mask.sum() # b

    return chainbreak_ce_loss




def to_pairwise_mask(
    mask_i: torch.Tensor, #Bool['... n'],
    mask_j: torch.Tensor = None): # -> Bool['... n n']:

    mask_j = mask_i if mask_j is None else mask_j
    assert mask_i.shape == mask_j.shape
    return einx.logical_and('... i, ... j -> ... i j', mask_i, mask_j)


def masked_average(
    t: torch.Tensor,  #Shaped['...'],
    mask: torch.Tensor, #Shaped['...'],
    *,
    dim, #int | Tuple[int, ...],
    eps = 1.): #-> Float['...']:

    num = (t * mask).sum(dim = dim)
    den = mask.sum(dim = dim)
    return num / den.clamp(min = eps)


class SmoothLDDTLoss(torch.nn.Module):
    def __init__(
        self,
        cutoff: float = 15.0,
        schedule: list = [0.5, 1.0, 2.0, 4.0],
        scheduled: bool = False,
        cb: bool = False,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.schedule = schedule
        self.scheduled = scheduled
        self.register_buffer('lddt_thresholds', torch.tensor(schedule))
        self.cb = cb
    def set_schedule(self, epoch: int):
        if epoch < 2:
            #logger.info("set schedule LDDT to [2.0, 4.0, 8.0, 16.0]")
            self.lddt_thresholds = torch.tensor([2.0, 4.0, 8.0, 16.0])
        elif epoch < 4:
            self.lddt_thresholds = torch.tensor([1.0, 2.0, 4.0, 8.0])
        elif epoch < 6:
            self.lddt_thresholds = torch.tensor([0.5, 1.0, 2.0, 4.0])
        else:
            self.lddt_thresholds = torch.tensor([0.25, 0.5, 1.0, 2.0])
    
    def pad_coords(
        self,
        pred_coords: torch.Tensor,  # [B, N, 3]
        true_coords: torch.Tensor,  # [B, N, 3]
        coords_mask: torch.Tensor   # [B, N] bool
    ):
        """
        Gathers valid coords from each batch item, and pads them to shape [B, N_max, 3].
        Also returns a padded mask [B, N_max].
        """
        B, N, _ = pred_coords.shape
        device = pred_coords.device

        # 1) Count how many valid atoms each batch item has
        #    n_valids[b] = sum(coords_mask[b])
        n_valids = coords_mask.sum(dim=1)  # shape [B]
        N_max = n_valids.max().item()      # largest valid-atom count in the batch

        # 2) Allocate padded tensors
        padded_pred = pred_coords.new_zeros((B, N_max, 3))
        padded_true = true_coords.new_zeros((B, N_max, 3))
        padded_mask = coords_mask.new_zeros((B, N_max), dtype=torch.bool)

        # 3) For each item, gather valid coords, copy them into the padded slice
        for b in range(B):
            n_b = n_valids[b]
            if n_b == 0:
                continue  # no valid coords in this item
            valid_b = coords_mask[b]              # shape [N]
            coords_b_pred = pred_coords[b][valid_b]  # [n_b, 3]
            coords_b_true = true_coords[b][valid_b]  # [n_b, 3]

            padded_pred[b, :n_b] = coords_b_pred
            padded_true[b, :n_b] = coords_b_true
            padded_mask[b, :n_b] = True

        return padded_pred, padded_true, padded_mask  # shapes: [B, N_max, 3], [B, N_max]

    def forward(
        self,
        pred_coords: torch.Tensor, #Float['b n 3'],
        true_coords: torch.Tensor, #Float['b n 3'],
        coords_mask: torch.Tensor, #Bool['b n'] | None = None,
        save_mem: bool = True,
        epoch: int = None,
        ): #-> Float['']:
        """
        pred_coords: predicted coordinates, For backbone flattened from [b n 3 3] to [b n*3 3]
        true_coords: true coordinates
        coords_mask: mask for the coordinates. For backbone use residue mask [b n] and expand each residue to 3 atoms [b n*3]
        """
        if self.scheduled:
            assert epoch != None, "Epoch cannot be NONE for scheduled LDDT loss"
            self.set_schedule(epoch)
        # Compute distances between all pairs of atoms
        self.lddt_thresholds = self.lddt_thresholds.to(pred_coords.device)
        if save_mem is False:
            true_dists = torch.cdist(true_coords, true_coords)
            # Compute distance difference for all pairs of atoms
            dist_diff = torch.abs(true_dists - pred_dists)

        else:
            padded_pred, padded_true, padded_mask = self.pad_coords(pred_coords, true_coords, coords_mask)
            true_dists = torch.cdist(padded_true, padded_true)
            pred_dists = torch.cdist(padded_pred, padded_pred)
            coords_mask = padded_mask
            pred_coords = padded_pred
            true_coords = padded_true

        # Compute distance difference for all pairs of atoms
        dist_diff = torch.abs(true_dists - pred_dists)
    
                # Compute epsilon values

        eps = einx.subtract('thresholds, ... -> ... thresholds', self.lddt_thresholds, dist_diff)
        eps = eps.sigmoid().mean(dim = -1)


        inclusion_radius = true_dists < self.cutoff

        # Compute mean, avoiding self term
        mask = inclusion_radius & ~torch.eye(pred_coords.shape[1], dtype=torch.bool, device=pred_coords.device)

        # Take into account variable lengthed atoms in batch
        if coords_mask is not None:
            paired_coords_mask = to_pairwise_mask(coords_mask)
            mask = mask & paired_coords_mask

        # Calculate masked averaging
        lddt = masked_average(eps, mask = mask, dim = (-1, -2), eps = 1)

        return 1. - lddt.mean()  
 
def contact_mse_loss(gt_bb_coords: torch.Tensor,
                    pred_bb_coords: torch.Tensor, 
                    seq_mask: torch.Tensor,
                    cutoff: float = 8.0,
                    temperature: float = 1.0,
                    contact_weight=5.0) :
    """
    Compute the contact map loss between the predicted and true contacts maps computed from backbone coordinates
    """
    # compute the true contact map from the ground truth cbeta coordinates [B, L, L, 2]
    gt_coords_with_cb = fill_in_cbeta_for_atom3_all(gt_bb_coords)
    gt_coords_cb = gt_coords_with_cb[:, :, 3] # b n 3
    dvecs = gt_coords_cb[:, :, None, :] - gt_coords_cb[:, None, :, :] # b n n 3
    gt_distogram_cb = dvecs.pow(2).sum(-1) 

    # compute the predicted contact map from the predicted backbone cbeta coordinates [B, L, L, 2]
    pred_coords_with_cb = fill_in_cbeta_for_atom3_all(pred_bb_coords)
    pred_coords_cb = pred_coords_with_cb[:, :, 3] # b n 3
    dvecs = pred_coords_cb[:, :, None, :] - pred_coords_cb[:, None, :, :] # b n n 3
    pred_distogram_cb = dvecs.pow(2).sum(-1)

    soft_gt_contact_map = torch.sigmoid((cutoff ** 2 - gt_distogram_cb) / temperature)
    soft_pred_contact_map = torch.sigmoid((cutoff ** 2 - pred_distogram_cb) / temperature)

    # compute the loss
    weight =  1 + (contact_weight - 1) * soft_gt_contact_map
    loss = (soft_pred_contact_map - soft_gt_contact_map).pow(2) * weight

    mask = seq_mask[:, :, None] * seq_mask[:, None, :]

    loss = loss * mask
    loss = loss.sum() / mask.sum().clamp(min=1e-6)
    return loss


def masked_mse_loss(gt_bb_coords: torch.Tensor,
                    pred_bb_coords: torch.Tensor, 
                    seq_mask: torch.Tensor,
                    atom_mask: torch.Tensor = None):
                
    """
    Compute the MSE loss between the predicted and true backbone coordinates
    """

    loss = (pred_bb_coords - gt_bb_coords).pow(2)  # Shape: [B, L, 3, 3]
    #logger.info(f"Pred shape {pred_bb_coords.shape} seq mask shape {seq_mask.shape} loss shape {loss.shape}")
    # Expand seq_mask to match loss shape
    mask = seq_mask[:, :, None, None]  # Shape: [B, L, 1]
    if atom_mask is not None:
        mask = mask * atom_mask[:, :, :, None]
    masked_loss = loss * mask  # Apply mask
   
    final_loss = masked_loss.sum() / mask.sum().clamp(min=1e-6) 

    return final_loss

def interaction_mse_loss(gt_bb_coords: torch.Tensor,
                                    pred_bb_coords: torch.Tensor, 
                                    seq_mask: torch.Tensor,
                                    d_covalent_min: float = 3.8,  
                                    d_covalent_max: float = 4.2,  
                                    d_interaction_min: float = 4.5,  
                                    d_interaction_max: float = 10.0,  
                                    clash_threshold: float =  3.5,  # Clash penalty threshold
                                    clash_penalty_scale: float = 10.0,  # Scale for clash penalty
                                    temperature: float = 2.0,
                                    covalent_weight: float = 8.0,  
                                    interaction_weight: float = 4.0,  
                                    non_interacting_weight: float = 1.0):
    """
    Compute a differentiable loss based on pairwise residue distances using MSE loss.

    Covalent bonds are based on Cα-Cα distances in [3.8, 4.0] Å.
    Noncovalent interactions are based on Cβ-Cβ distances in a soft range [4.5, 10.0] Å.
    A clash penalty is applied when residue distances are < 4.0 Å.

    Residue pairs are softly categorized as:
      - Covalent bonded (3.8 Å ≤ Cα-Cα ≤ 4.0 Å)
      - Non-covalent interaction (4.5 Å ≤ Cβ-Cβ ≤ 12.0 Å)
      - Non-interacting (otherwise)
      - **Clashing Residues (`< 4.0 Å`) receive a penalty**

    Args:
        gt_bb_coords (torch.Tensor): Ground truth backbone coordinates of shape [B, N, 3, 3].
        pred_bb_coords (torch.Tensor): Predicted backbone coordinates of shape [B, N, 3, 3].
        seq_mask (torch.Tensor): Sequence mask of shape [B, N] (1 for valid residues, 0 for padding).
        d_covalent_min (float): Minimum Cα-Cα distance for covalent bonds.
        d_covalent_max (float): Maximum Cα-Cα distance for covalent bonds.
        d_interaction_min (float): Minimum Cβ-Cβ distance for noncovalent interactions.
        d_interaction_max (float): Maximum Cβ-Cβ distance for noncovalent interactions.
        clash_threshold (float): Distance below which a penalty is applied.
        clash_penalty_scale (float): Controls how much to penalize clashes.
        temperature (float): Sigmoid temperature for smooth classification.
        covalent_weight (float): Weight for covalent bonds.
        interaction_weight (float): Weight for non-covalent interactions.
        non_interacting_weight (float): Weight for non-interacting residues.

    Returns:
        torch.Tensor: MSE-based soft pairwise distance loss with contact weighting and clash penalty.
    """

    B, N, _, _ = pred_bb_coords.shape

    # **Compute Cβ Coordinates Using `fill_in_cbeta_for_atom3_all`**
    gt_coords_with_cb = fill_in_cbeta_for_atom3_all(gt_bb_coords)
    pred_coords_with_cb = fill_in_cbeta_for_atom3_all(pred_bb_coords)

    gt_cb_coords = gt_coords_with_cb[:, :, 3]  # [B, N, 3]
    pred_cb_coords = pred_coords_with_cb[:, :, 3]  # [B, N, 3]

    # **Extract Cα Coordinates (for covalent bonds)**
    gt_ca_coords = gt_bb_coords[:, :, 1, :]  # [B, N, 3]
    pred_ca_coords = pred_bb_coords[:, :, 1, :]  # [B, N, 3]

    # **Compute Pairwise Distances**
    pred_ca_dists = torch.cdist(pred_ca_coords, pred_ca_coords)  # [B, N, N]
    gt_ca_dists = torch.cdist(gt_ca_coords, gt_ca_coords)  # [B, N, N]

    pred_cb_dists = torch.cdist(pred_cb_coords, pred_cb_coords)  # [B, N, N]
    gt_cb_dists = torch.cdist(gt_cb_coords, gt_cb_coords)  # [B, N, N]

    # **Soft classification of covalent bonds (Cα-Cα distance)**
    soft_covalent_gt = torch.sigmoid((d_covalent_max - gt_ca_dists) / temperature) * \
                       torch.sigmoid((gt_ca_dists - d_covalent_min) / temperature)
    
    soft_covalent_pred = torch.sigmoid((d_covalent_max - pred_ca_dists) / temperature) * \
                         torch.sigmoid((pred_ca_dists - d_covalent_min) / temperature)

    # **Soft classification of noncovalent interactions (Cβ-Cβ distance)**
    soft_interacting_gt = torch.sigmoid((d_interaction_max - gt_cb_dists) / temperature) * \
                          torch.sigmoid((gt_cb_dists - d_interaction_min) / temperature)
    
    soft_interacting_pred = torch.sigmoid((d_interaction_max - pred_cb_dists) / temperature) * \
                            torch.sigmoid((pred_cb_dists - d_interaction_min) / temperature)

    # **Soft classification of non-interacting residues**, 
    soft_non_interacting_gt = torch.clamp(1 - soft_covalent_gt - soft_interacting_gt, min=0.0, max=1.0)
    soft_non_interacting_pred = torch.clamp(1 - soft_covalent_pred - soft_interacting_pred, min=0.0, max=1.0)

    # **Clash Penalty for distances < 4.0 Å (Exponential Growth)**
    clash_penalty = clash_penalty_scale * torch.exp(clash_threshold - pred_ca_dists / 1.0)  # Strong penalty for small distances
    clash_mask = (pred_ca_dists < clash_threshold).float()
    clash_penalty = clash_penalty * clash_mask  # Only apply penalty where clashes exist

    # **Stack into probability tensors: [B, N, N, 3]**
    gt_probs = torch.stack([soft_covalent_gt, soft_interacting_gt, soft_non_interacting_gt], dim=-1)
    pred_probs = torch.stack([soft_covalent_pred, soft_interacting_pred, soft_non_interacting_pred], dim=-1)

    # **Define category-specific weights**
    weights = torch.stack([
        covalent_weight * soft_covalent_gt,  
        interaction_weight * soft_interacting_gt,  
        non_interacting_weight * soft_non_interacting_gt  
    ], dim=-1)  # Shape: [B, N, N, 3]

    # **Compute MSE loss for soft categories**
    loss = (pred_probs - gt_probs).pow(2)  # Squared difference [B, N, N, 3]

    # **Apply category-specific weighting**
    weighted_loss = loss * weights  

    # **Sum over class dimension**
    weighted_loss = weighted_loss.sum(dim=-1)  # [B, N, N]

    # **Apply sequence mask to exclude invalid positions**
    mask = seq_mask[:, :, None] * seq_mask[:, None, :]  # Shape: [B, N, N]

    # Create a [B, N, N] diagonal mask of zeros for each batch

    diag_mask = torch.eye(N, dtype=mask.dtype, device=mask.device).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

    # Convert that diagonal of 1s to 0 and everything else 1
    # (i.e., this mask is 0 on the diagonal, 1 off-diagonal)
    offdiag_mask =  ~diag_mask

    # Combine with your original mask
    # Now mask[i, i] = 0 so it zeroes out the diagonal
    mask = mask & offdiag_mask


    weighted_loss = weighted_loss * mask  

    clash_penalty = clash_penalty * mask

    # **Add clash penalty to the final loss**
    final_loss = weighted_loss.sum() / mask.sum().clamp(min=1e-6)  
    final_loss += clash_penalty.sum() / mask.sum().clamp(min=1e-6)  

    return final_loss


def energy_mse_loss(
    pred_eng: torch.Tensor, # [B, N, N, 3]
    gt_eng: dict, # {pdb_id: [N, N, 3]}
    pdb_ids, # batch["id"]
    energy_dir: str = "/data/eng_calc/rosetta_fa_hb",
    energy_type: list[str] = ["hbond", "sol", "elec"], # 2, 3, 4
    energy_weight: list[float] = [1.0, 1.0, 1.0],
    seq_mask: torch.Tensor = None, # [B, N]
):
    """
    Compute the energy loss between the predicted and true energies
    """
    # read the energy pt files named as pdb_ids.pt in the energy_dir
    device = pred_eng.device
    total_loss = torch.tensor(0.0, device=device)
    total_pairs = 0
    # check which pdb_ids has a non-zero seq_mask
    non_zero_mask = seq_mask.sum(dim=-1) > 0
    #device = pred_eng.device

    for i, pdb_id in enumerate(pdb_ids):
        if not non_zero_mask[i]:
            continue
        #energy_file = os.path.join(energy_dir, f"{pdb_id}.pt")
        #if not os.path.exists(energy_file):
            #raise FileNotFoundError(f"Energy file {energy_file} does not exist")
        #    logger.info(f"Energy file {energy_file} does not exist")
         #   continue
        #energy = torch.load(energy_file, weights_only=True).to(device)[:, :, 2:]  # [N, N, 5] -> [N, N, 3]

        if pdb_id not in gt_eng:
            logger.warning(f"Missing ground-truth energy for {pdb_id}, skipping.")
            continue
        energy = gt_eng[pdb_id]
        seq_len = energy.shape[0]
        pred_eng_i = pred_eng[i, :seq_len, :seq_len, :] # [N, N, 3]
        #assert list(pred_eng_i.shape) == list(energy.shape), f"Pred eng shape {pred_eng_i.shape} does not match gt eng shape {energy.shape}"
        # if length of pred_eng_i is not equal to length of energy, crop either to the min length
        min_len = min(pred_eng_i.shape[0], energy.shape[0])
        pred_eng_i = pred_eng_i[:min_len, :min_len, :]
        energy = energy[:min_len, :min_len, :]

        # compute regression loss with a mse loss
        sq_error = (pred_eng_i - energy).pow(2).sum()
        total_loss += sq_error
        num_elements = energy.numel()
        total_pairs += num_elements
        # clear the memory
        del energy

    mean_loss = total_loss / (total_pairs + 1e-6)

    return mean_loss
 

def weighted_energy_mse_loss(
    pred_eng: torch.Tensor,  # [B, N, N, 3]
    gt_eng: dict, # {pdb_id: [N, N, 3]}
    pdb_ids,
    energy_dir: str = "/data/rosetta_fa_hb",
    energy_type: list[str] = ["hbond", "sol", "elec"],
    energy_weight: list[float] = [1.0, 1.0, 1.0],
    seq_mask: torch.Tensor = None,  # [B, N]
    zero_weight: float = 1.0,
    nonzero_weight: float = 3.0,
    zero_threshold: float = 1e-3,
    sparsity_loss: bool = True,
    sparsity_weight: float = 0.01,

):
    device = pred_eng.device
    total_loss = torch.tensor(0.0, device=device)
    total_pairs = 0
    non_padded_mask = seq_mask.sum(dim=-1) > 0
    #logger.info(f"Mean pred energy: {pred_eng.mean().item()}, Min pred energy: {pred_eng.min().item()}, Max pred energy: {pred_eng.max().item()}")
    total_zero = 0
    correctly_predicted_zero = 0

    for i, pdb_id in enumerate(pdb_ids):
        if not non_padded_mask[i]:
            continue

        #energy_file = os.path.join(energy_dir, f"{pdb_id}.pt")
        #if not os.path.exists(energy_file):
        #    logger.info(f"Energy file {energy_file} does not exist")
        #    continue

        #energy = torch.load(energy_file, weights_only=True).to(device)[:, :, 2:]  # [N, N, 3]

        if pdb_id not in gt_eng:
            logger.warning(f"Missing ground-truth energy for {pdb_id}, skipping.")
            continue
        energy = gt_eng[pdb_id]
        seq_len = energy.shape[0]
        pred_eng_i = pred_eng[i, :seq_len, :seq_len, :]
        min_len = min(pred_eng_i.shape[0], energy.shape[0])
        pred_eng_i = pred_eng_i[:min_len, :min_len, :]
        energy = energy[:min_len, :min_len, :]

        # Create masks
        nonzero_mask = abs(energy) > zero_threshold
        zero_mask = ~nonzero_mask

        # Apply different weights to errors
        error = pred_eng_i - energy
        sq_error = (
            zero_weight * (error * zero_mask).pow(2).sum() +
            nonzero_weight * (error * nonzero_mask).pow(2).sum()
        )

        total_loss += sq_error
        total_pairs += energy.numel()

        # Evaluate percentage of correctly predicted zero entries
        with torch.no_grad():
            abs_pred = pred_eng_i.abs()
            correct_zeros = (abs_pred < zero_threshold) & zero_mask
            correctly_predicted_zero += correct_zeros.sum().item()
            total_zero += zero_mask.sum().item()

        del energy

    mean_loss = total_loss / (total_pairs + 1e-6)

    if sparsity_loss:
        avg_magnitude = pred_eng.abs().mean()
        mean_loss += sparsity_weight * avg_magnitude

    zero_percentage = correctly_predicted_zero / (total_zero + 1e-6)
    # convert to tensor
    zero_percentage = torch.tensor(zero_percentage, device=device)
    return mean_loss, zero_percentage

from sklearn.metrics import average_precision_score
MAX_N = 512

TERTIARY_MASK_FULL = (torch.abs(
    torch.arange(MAX_N).view(-1, 1) - torch.arange(MAX_N).view(1, -1)
) >= 5)  # shape: [512, 512]

def energy_prediction_pr_auc(
    pred_eng: torch.Tensor,  # [B, N, N, 3]
    gt_eng: dict,  # {pdb_id: [N, N, 3]}
    pdb_ids,
    energy_dir: str = "/data/eng_calc/rosetta_fa_hb",
    seq_mask: torch.Tensor = None,  # [B, N]
    zero_threshold: float = 1e-3,
):
    device = pred_eng.device
    non_padded_mask = seq_mask.sum(dim=-1) > 0

    all_pred_scores = []
    all_gt_labels = []

    tertiary_pred_scores = []
    tertiary_gt_labels = []

    with torch.no_grad():
        for i, pdb_id in enumerate(pdb_ids):
            if not non_padded_mask[i]:
                continue
            if pdb_id not in gt_eng:
                logger.warning(f"Missing ground-truth energy for {pdb_id}, skipping.")
                continue

            energy = gt_eng[pdb_id]
            seq_len = energy.shape[0]
            pred_eng_i = pred_eng[i, :seq_len, :seq_len, :]
            min_len = min(pred_eng_i.shape[0], energy.shape[0])
            pred_eng_i = pred_eng_i[:min_len, :min_len, :]
            energy = energy[:min_len, :min_len, :]

            pred_score = pred_eng_i.norm(dim=-1)  # [N, N]
            gt_label = (energy.abs() > zero_threshold).any(dim=-1).long()  # [N, N]

            # Flatten for all-pair PR-AUC
            all_pred_scores.append(pred_score.flatten())
            all_gt_labels.append(gt_label.flatten())

            # Tertiary interactions: |i - j| >= 5
            tertiary_mask = TERTIARY_MASK_FULL[:min_len, :min_len].to(device)
            tertiary_pred_scores.append(pred_score[tertiary_mask])
            tertiary_gt_labels.append(gt_label[tertiary_mask])

            del energy
    # if no valid entries, return zeros
    if len(all_pred_scores) == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # Prepare all and tertiary PR-AUC inputs
    all_pred_scores = torch.cat(all_pred_scores).detach().cpu().numpy()
    all_gt_labels = torch.cat(all_gt_labels).cpu().numpy()
    tertiary_pred_scores = torch.cat(tertiary_pred_scores).detach().cpu().numpy()
    tertiary_gt_labels = torch.cat(tertiary_gt_labels).cpu().numpy()

    pr_auc_all = average_precision_score(all_gt_labels, all_pred_scores)
    pr_auc_tertiary = average_precision_score(tertiary_gt_labels, tertiary_pred_scores)

    return pr_auc_all, pr_auc_tertiary



CA_CA = 3.80209737096
def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def extreme_ca_ca_distance_violations(
    gt_bb_coords: torch.Tensor,   # [B, N, 3, 3]
    pred_bb_coords: torch.Tensor, # [B, N, 3, 3]
    seq_mask: torch.Tensor,       # [B, N]
    max_angstrom_tolerance: float = 1.5,
    min_clash_distance: float = 3.5,
    eps: float = 1e-6,
):
    """
    1) Fraction of consecutive CA-CA pairs that deviate by more than `max_angstrom_tolerance`
       from the canonical ~3.802 Å, ignoring chain breaks and padded residues.

    2) Fraction of non-neighbor predicted CA-CA pairs that are too close
       (< min_clash_distance), ignoring padded residues.

    Returns:
      - mean_neighbor_violations: [B], fraction of invalid consecutive pairs
      - clash_fraction:           [B], fraction of non-neighbor pairs that clash
    """
    ################################################################
    # Part 1: Consecutive residue violations
    ################################################################
    B, N, _, _ = gt_bb_coords.shape

    # Extract consecutive CA coordinates in ground truth
    gt_first_ca  = gt_bb_coords[:, :-1, 1, :]  # [B, N-1, 3]
    gt_second_ca = gt_bb_coords[:,  1:, 1, :]  # [B, N-1, 3]

    # Extract consecutive CA coordinates in prediction
    pred_first_ca  = pred_bb_coords[:, :-1, 1, :]
    pred_second_ca = pred_bb_coords[:,  1:, 1, :]

    # Mask for valid neighboring residues (exclude padding)
    this_ca_mask = seq_mask[:, :-1]  # [B, N-1]
    next_ca_mask = seq_mask[:, 1:]   # [B, N-1]

    # Ground-truth consecutive distances
    gt_ca_ca_distance = torch.sqrt(
        eps + torch.sum((gt_first_ca - gt_second_ca) ** 2, dim=-1)
    )  # [B, N-1]

    # Predicted consecutive distances
    pred_ca_ca_distance = torch.sqrt(
        eps + torch.sum((pred_first_ca - pred_second_ca) ** 2, dim=-1)
    )  # [B, N-1]

    # Chain breaks in ground truth
    max_ca_ca_dist = CA_CA + max_angstrom_tolerance
    chainbreak_mask = (gt_ca_ca_distance > max_ca_ca_dist)  # [B, N-1]

    # Which consecutive pairs are actually valid (no break, not padded)?
    neighbor_mask = (this_ca_mask & next_ca_mask & (~chainbreak_mask))  # [B, N-1]

    # Check for large deviation from ~3.802 Å
    violations = (pred_ca_ca_distance - CA_CA).abs() > max_angstrom_tolerance

    # Fraction of neighbor violations
    mean_neighbor_violations = masked_mean(neighbor_mask, violations, dim=[0, 1])

    ################################################################
    # Part 2: Non-neighbor Cα–Cα Clashes (Memory-Optimized)
    ################################################################
    # Extract all predicted CA
    pred_ca_all = pred_bb_coords[:, :, 1, :]  # [B, N, 3]

    # We'll compute squared distances via (x-y)^2 = x^2 + y^2 - 2 x·y
    # 1) squares = sum(pred_ca_all^2, dim=-1) -> [B, N]
    # 2) dot = pred_ca_all @ pred_ca_all^T    -> [B, N, N]
    # 3) dist_sq = squares[:, :, None] + squares[:, None, :] - 2*dot
    # Then we check dist_sq < (min_clash_distance^2)

    squares = torch.sum(pred_ca_all * pred_ca_all, dim=-1)  # [B, N]
    # matrix multiply -> x·y
    # pred_ca_all: [B, N, 3]
    # we want dot: [B, N, N]
    # => for each b in [0..B-1], dot[b] = pred_ca_all[b] @ pred_ca_all[b].T
    dot = torch.bmm(pred_ca_all, pred_ca_all.transpose(1, 2))  # [B, N, N]

    dist_sq = squares[:, :, None] + squares[:, None, :] - 2.0 * dot  # [B, N, N]
    # We can clamp small negative values due to numerical errors
    dist_sq = torch.clamp(dist_sq, min=0.0)

    # Build a mask for valid non-neighbor pairs
    # (1) Must not be adjacent: |i-j| > 1
    idx = torch.arange(N, device=pred_ca_all.device)
    i_idx = idx[:, None]
    j_idx = idx[None, :]
    non_neighbor_mask_2d = (i_idx - j_idx).abs() > 1  # [N, N]
    # Expand to [B, N, N]
    non_neighbor_mask = non_neighbor_mask_2d.unsqueeze(0).expand(B, -1, -1)

    # (2) Each residue must not be padded
    valid_ca_mask = seq_mask[:, :, None] & seq_mask[:, None, :]  # [B, N, N]

    # Combine
    valid_pair_mask = valid_ca_mask & non_neighbor_mask

    # A clash if dist_sq < min_clash_distance^2
    min_clash_dist_sq = min_clash_distance * min_clash_distance
    clashes = (dist_sq < min_clash_dist_sq) & valid_pair_mask

    clash_fraction = masked_mean(valid_pair_mask, clashes, dim=[0, 1, 2])

    return mean_neighbor_violations, clash_fraction




N_IDX, CA_IDX, C_IDX = 0, 1, 2      # backbone atoms we use to build frames

def gram_schmidt(v1, v2, eps=1e-8):
    """Given two non‑parallel vectors, return an orthonormal 3x3 rotation matrix."""
    v1 = F.normalize(v1, dim=-1, eps=eps)
    v2 = v2 - (v1 * v2).sum(dim=-1, keepdim=True) * v1
    v2 = F.normalize(v2, dim=-1, eps=eps)
    v3 = torch.cross(v1, v2, dim=-1)
    return torch.stack((v1, v2, v3), dim=-2)          # [..., 3, 3]

def make_frames(xyz):
    """
    Build residue frames T = [R|t] from backbone atoms.
    xyz: [B, L, 37, 3]
    Returns R: [B, L, 3, 3], t: [B, L, 3]
    """
    n = xyz[..., N_IDX, :]          # N
    ca = xyz[..., CA_IDX, :]        # Cα
    c = xyz[..., C_IDX, :]          # C

    e1 = c - ca                     # x‑axis → Cα→C
    e2 = n - ca                     # helper to build y‑axis
    R = gram_schmidt(e1, e2)        # [..., 3, 3]
    t = ca                          # translation
    return R, t

def to_local(xyz, R, t):
    """Transform xyz to local frame given R,t. Shapes broadcast."""
    return torch.matmul((xyz - t.unsqueeze(-2)), R.transpose(-2,-1))     # last dims 37×3 ⋅ 3×3

"""
def fape_loss(
    xyz_pred: torch.Tensor, # [B, L, 37, 3]
    xyz_true: torch.Tensor, # [B, L, 37, 3]
    atom_mask: torch.Tensor, # [B, L, 37]
    d_clamp: float = 10.0,
    eps: float = 1e-6,
):
    
    #Frame‑Aligned Point Error (single‑frame, per‑residue).
    #Returns scalar loss (Å²).
    
    # Build frames
    R_pred, t_pred = make_frames(xyz_pred)   # [B, L, 3, 3], [B, L, 3]
    R_true, t_true = make_frames(xyz_true)
    # 2) single‑frame losses in *true* frame
    pred_in_true = to_local(xyz_pred, R_true, t_true)
    true_in_true = to_local(xyz_true, R_true, t_true)
    d_true = torch.norm(pred_in_true - true_in_true, dim=-1)

    # 3) single‑frame losses in *predicted* frame
    pred_in_pred = to_local(xyz_pred, R_pred, t_pred)
    true_in_pred = to_local(xyz_true, R_pred, t_pred)
    d_pred = torch.norm(pred_in_pred - true_in_pred, dim=-1)

    # 4) clamp, mask, average dual‑frame
    d_true = torch.clamp(d_true, max=d_clamp)
    d_pred = torch.clamp(d_pred, max=d_clamp)
    dual_d = (d_true + d_pred) * 0.5

    loss = (dual_d * atom_mask).sum() / (atom_mask.sum() + eps)

    return loss
"""

def fape_loss(
    xyz_pred: torch.Tensor,   # [B, L, 37, 3]
    xyz_true: torch.Tensor,   # [B, L, 37, 3]
    atom_mask: torch.Tensor,  # [B, L, 37]  (1 = present)
    res_mask:  torch.Tensor,  # [B, L]      (1 = residue present)
    d_clamp: float = 10.0,
    eps: float = 1e-6,
):
    """
    Frame‑Aligned Point Error with proper masking of *both* atoms and frames.
    """
    # ----- 1. build frames only for valid residues -----------------------
    R_pred, t_pred = make_frames(xyz_pred)   # [B,L,3,3]  , [B,L,3]
    R_true, t_true = make_frames(xyz_true)

    # replace invalid‑residue frames with identity so they contribute 0 error
    I = xyz_pred.new_zeros(3, 3); I[..., :] = torch.eye(3, device=xyz_pred.device)
    R_pred = torch.where(res_mask[..., None, None], R_pred, I)      # [B,L,3,3]
    t_pred = torch.where(res_mask[..., None],      t_pred, 0.0)

    R_true = torch.where(res_mask[..., None, None], R_true, I)
    t_true = torch.where(res_mask[..., None],      t_true, 0.0)

    # ----- 2. points in the *true* frame --------------------------------
    pred_T = to_local(xyz_pred, R_true, t_true)      # [B,L,37,3]
    true_T = to_local(xyz_true, R_true, t_true)
    d_T = torch.norm(pred_T - true_T, dim=-1).clamp(max=d_clamp)

    # ----- 3. points in the *predicted* frame ---------------------------
    pred_P = to_local(xyz_pred, R_pred, t_pred)
    true_P = to_local(xyz_true, R_pred, t_pred)
    d_P = torch.norm(pred_P - true_P, dim=-1).clamp(max=d_clamp)

    # ----- 4. dual‑frame mean, mask, final average ----------------------
    dual_d = 0.5 * (d_T + d_P)                           # [B,L,37]
    masked_sum = (dual_d * atom_mask).sum()
    norm       = atom_mask.sum().clamp_min(eps)
    return masked_sum / norm




def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in RESTYPES:
        residue_name = RESTYPE_1TO3[residue_name]
        residue_chi_angles = CHI_ANGLES_ATOMS[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([ATOM_ORDER[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices

def atom37_to_torsion_angles(
    coords: torch.Tensor, # [B, L, 37, 3]
    atom_mask: torch.Tensor, # [B, L, 37]
    aatype: torch.Tensor, # [B, L]
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    all_atom_positions = coords
    all_atom_mask = atom_mask
    # convert coords to float32
    #all_atom_positions = all_atom_positions.to(torch.float32)
    #all_atom_mask = all_atom_mask.to(torch.float32)

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, 37, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(
        prev_all_atom_mask[..., 1:3], dim=-1
    ) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
        torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
        * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(), device=aatype.device
    )

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = list(CHI_ANGLES_MASK)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = torsion_frames.invert().apply(
        torsions_atom_pos[..., 3, :]
    )

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        CHI_PI_PERIODIC,
    )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    )

    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask


def torsion_angle_loss(
    xyz_pred: torch.Tensor, # [B, L, 37, 3]
    xyz_true: torch.Tensor, # [B, L, 37, 3]
    atom_mask: torch.Tensor, # [B, L, 37]
    aatype: torch.Tensor, # [B, L]
):
    
    epsilon = 1e-8
    
    
    a_gt, a_alt_gt, _ =  atom37_to_torsion_angles(xyz_true, atom_mask, aatype)
    a, _, _ =  atom37_to_torsion_angles(xyz_pred, atom_mask, aatype)
    assert not torch.isnan(a).any(), "NaN in torsion angles (a)"
    assert not torch.isnan(a_gt).any(), "NaN in torsion angles (a_gt)"

    norm = torch.norm(a, dim=-1)

    # [*, N, 7, 2]
    a = a / (norm.unsqueeze(-1) + epsilon)
    assert not torch.isnan(a).any(), "NaN in torsion angles (a) after normalization"

    # [*, N, 7]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    diff_norm_alt_gt = torch.norm(a - a_alt_gt, dim=-1)
    min_diff = torch.minimum(diff_norm_gt ** 2, diff_norm_alt_gt ** 2)

    # [*]
    l_torsion = torch.mean(min_diff, dim=(-1, -2))
    l_angle_norm = torch.mean(torch.abs(norm - 1), dim=(-1, -2))

    an_weight = 0.02

    torsion_loss = torch.mean(l_torsion + an_weight * l_angle_norm)


    dihedral_true = a_gt[:, :, :2, :]
    dihedral_pred = a[:, :, :2, :]
    dihedral_norm = torch.norm(dihedral_pred, dim=-1)
    dihedral_pred = dihedral_pred/(dihedral_norm.unsqueeze(-1) + epsilon)
    diff_norm = torch.norm(dihedral_pred - dihedral_true, dim=-1)**2
    l_dihedral = torch.mean(diff_norm, dim=(-1, -2))
    l_dihedral_norm = torch.mean(torch.abs(dihedral_norm - 1), dim=(-1, -2))
    dihedral_loss = torch.mean(l_dihedral + an_weight * l_dihedral_norm)

    logger.info(f"torsion_loss: {torsion_loss}, dihedral_loss: {dihedral_loss}")

    return dihedral_loss, torsion_loss



def torsion_l2_loss(
    xyz_pred: torch.Tensor,  # [B, L, 37, 3]
    xyz_true: torch.Tensor,  # [B, L, 37, 3]
    atom_mask: torch.Tensor,  # [B, L, 37]
    aatype: torch.Tensor,  # [B, L]
):
    epsilon = 1e-6
    # Extract torsion angle sin/cos vectors and their validity mask
    torsion_pred, _, _ = atom37_to_torsion_angles(xyz_pred, atom_mask, aatype)  # [B, L, 7, 2], [B, L, 7]
    torsion_true, torsion_alt_true, torsion_mask = atom37_to_torsion_angles(xyz_true, atom_mask, aatype)  # [B, L, 7, 2], [B, L, 7], [B, L, 7]
    # Compute L2 squared distances to both canonical and alternate torsion targets
    diff_true = torch.sum((torsion_pred - torsion_true) ** 2, dim=-1)        # [B, L, 7]
    diff_alt = torch.sum((torsion_pred - torsion_alt_true) ** 2, dim=-1)     # [B, L, 7]
    min_diff = torch.min(diff_true, diff_alt)                                # [B, L, 7]

    # Apply torsion mask (which already accounts for atom validity internally)
    masked_loss = min_diff * torsion_mask                                    # [B, L, 7]
    torsion_loss = torch.sum(masked_loss) / (torch.sum(torsion_mask) + epsilon)         # scalar


    dihedral_pred = torsion_pred[:, :, :2, :]
    dihedral_true = torsion_true[:, :, :2, :]
    dihedral_mask = torsion_mask[:, :, :2]
    diff_dihedral = torch.sum((dihedral_pred - dihedral_true) ** 2, dim=-1)
    masked_dihedral_loss = diff_dihedral * dihedral_mask
    dihedral_loss = torch.sum(masked_dihedral_loss) / (torch.sum(dihedral_mask) + epsilon)

    return dihedral_loss, torsion_loss


def backbone_distance_loss_l2(
    pred: torch.Tensor,      # [Res*3, 3]
    true: torch.Tensor,      # [Res*3, 3]
    batch_indices: torch.Tensor,  # [Res*3], batch index for each atom
    B: int,                  # number of batch items
    epoch: int,   
) -> torch.Tensor:
    if epoch < 1:
        #clamp = 5.0
        clamp = 10.0
    elif epoch < 2:
        clamp = 5.0
    elif epoch < 3:
        clamp = 3.0
    elif epoch < 5:
        clamp = 1.5
    else:
        clamp = 1
    #logger.info(f"start computing bb_dist_l2 with clamp {clamp}")
    assert pred.shape == true.shape
    assert pred.shape[0] == batch_indices.shape[0], f"pred shape {pred.shape} and batch_idx shape {batch_indices.shape}"
    if pred.size(0) < 2:                     # nothing to compare
        return pred.new_tensor(0.0)
    
    d_pred = torch.cdist(pred, pred, p=2)  # [S, S]
    d_true = torch.cdist(true, true, p=2)  # [S, S]

    same_sample_mask = batch_indices.unsqueeze(1).eq(batch_indices.unsqueeze(0))

    err2 = (d_pred - d_true).pow(2)                  # Å²
    err2_clamped = torch.clamp(err2, max=clamp ** 2)

    err2_clamped = err2_clamped * same_sample_mask

    n_atoms_per_chain = torch.bincount(batch_indices, minlength=B)
    pair_norm = (n_atoms_per_chain.float() ** 2).sum() 

    if pair_norm == 0:
            return pred.new_tensor(0.0)
    #logger.info(f"Error sum {err2_clamped.sum()} with pairnorm {pair_norm}")
    return err2_clamped.sum() / pair_norm
        


"""
def backbone_distance_loss_l2mean(
    pred: torch.Tensor,      # [Res*3, 3]
    true: torch.Tensor,      # [Res*3, 3]
    batch_indices: torch.Tensor,  # [Res*3], batch index for each atom
    B: int,                  # number of batch items
    epoch: int,   
) -> torch.Tensor:
    if epoch < 2:
        clamp = 10.0
    elif epoch < 4:
        clamp = 5.0
    elif epoch < 6:
        clamp = 3.0
    else:
        clamp = 1.5
"""
def backbone_distance_loss_l2mean(
    pred: torch.Tensor,      # [Res*3, 3]
    true: torch.Tensor,      # [Res*3, 3]
    batch_indices: torch.Tensor,  # [Res*3], batch index for each atom
    B: int,                  # number of batch items
    epoch: int,   
    schedule: list[float] = [10.0, 5.0, 3.0, 1.5],
) -> torch.Tensor:
    if epoch < 2:
        clamp = schedule[0]
    elif epoch < 4:
        clamp = schedule[1]
    elif epoch < 6:
        clamp = schedule[2]
    else:
        clamp = schedule[3]
    assert pred.shape == true.shape
    assert pred.shape[0] == batch_indices.shape[0]
    if pred.size(0) < 2:                     # nothing to compare
        return pred.new_tensor(0.0)
    
    d_pred = torch.cdist(pred, pred, p=2)  # [S, S]
    d_true = torch.cdist(true, true, p=2)  # [S, S]

    same_sample_mask = batch_indices.unsqueeze(1).eq(batch_indices.unsqueeze(0))

    err2 = (d_pred - d_true).pow(2)                  # Å²
    err2_clamped = torch.clamp(err2, max=clamp ** 2)


    return err2_clamped[same_sample_mask].mean()


def mask_l2pairwise_distance_loss(
    pred: torch.Tensor,                # [B,L,3|37,3]
    true: torch.Tensor,                # same as pred
    seq_mask: torch.Tensor,            # [B,L] 1 = residue present
    epoch: int,
    atom_mask: torch.Tensor | None = None,  # [B,L,37] optional
    schedule: tuple[float, ...] = (10.0, 5.0, 3.0, 1.5),
) -> torch.Tensor:

    clamp = schedule[min(epoch // 2, len(schedule) - 1)]
    B, L, A, _ = pred.shape            # A = 3 or 37

    # ---------- 1. build atom-level mask ---------------------------------
    if A == 3:
        pred = fill_in_cbeta_for_atom3_all(pred)      # -> [B,L,4,3]
        true = fill_in_cbeta_for_atom3_all(true)
        atom_mask_full = seq_mask.bool().unsqueeze(-1).expand(-1, -1, 4)
    elif A == 5:
        atom_mask_full = seq_mask.bool().unsqueeze(-1).expand(-1, -1, 5)
    else:
        if atom_mask is None:
            raise ValueError("atom_mask must be provided for 37‑atom inputs.")
        atom_mask_full = (
            atom_mask.to(dtype=torch.bool, device=pred.device)
            & seq_mask.bool().unsqueeze(-1)
        )

    if atom_mask_full.sum() < 2:                      # nothing to compare
        return pred.new_tensor(0.0)

    # ---------- 2. flatten valid atoms -----------------------------------
    flat_pred = pred[atom_mask_full].view(-1, 3)
    flat_true = true[atom_mask_full].view(-1, 3)

    # ---------- 3. per‑atom batch indices --------------------------------
    n_atoms = atom_mask_full.shape[2]                 # 4 or 37
    batch_idx = (
        torch.arange(B, device=pred.device)
        .view(B, 1, 1)
        .expand(-1, L, n_atoms)[atom_mask_full]       # [N]
    )

    # ---------- 4. pairwise distances ------------------------------------
    d_pred = torch.cdist(flat_pred, flat_pred, p=2)
    d_true = torch.cdist(flat_true, flat_true, p=2)

    same_sample = batch_idx[:, None].eq(batch_idx[None, :])
    not_self    = ~torch.eye(d_pred.size(0), dtype=torch.bool,
                             device=pred.device)
    pair_mask   = same_sample & not_self

    # ---------- 5. clamped mean‑squared error ----------------------------
    err2 = (d_pred - d_true).pow(2).clamp(max=clamp ** 2)
    pair_mask_f = pair_mask.float()
    return (err2 * pair_mask_f).sum() / pair_mask_f.sum().clamp(min=1.0)



def build_frames(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Make an orthonormal frame from three positions a‑b‑c.
    Axes (column‑major):
        e0 – plane normal  (b→c) × (b→a)
        e2 – bisector of (b→a) and (b→c)
        e1 – e2 × e0       (completes RH basis)
    Returns  [..., 3, 3]  (unit columns, det ≈ 1)
    """
    x = b - a                      # b→a
    y = c - b                      # b→c
    e2 = x - y                     # bisector  (will be z‑axis)
    e0 = torch.cross(y, x, dim=-1) # plane normal
    e1 = torch.cross(e2, e0, dim=-1)
    u = torch.stack((e0, e1, e2), dim=-1)   # [..., 3, 3]
    u = F.normalize(u, dim=-2)              # normalise the 3 columns
    return u


def huber(d: torch.Tensor, delta: float, cut: float) -> torch.Tensor:
    """Huber loss with hard clamp."""
    loss = torch.where(
        d < delta,
        0.5 * d * d,
        delta * (d - 0.5 * delta),
    ) / delta
    return torch.clamp(loss, max=cut)


def allatom_FAPE(
    pred_pos: torch.Tensor,          # [N_atoms, 3]
    true_pos: torch.Tensor,          # [N_atoms, 3]
    batch_idx: torch.Tensor,         # [N_atoms]
    frames: torch.Tensor,            # [N_frames, 3]
    delta: float = 1.0,
    cut: float = 10.0,
) -> torch.Tensor:                   # single scalar
    if frames.numel() == 0:
        return pred_pos.new_tensor(0.0)

    # -------- sort frames by batch once --------------------------------
    i, j, k = frames.T
    f_batch = batch_idx[i]                           # [F]
    order   = torch.argsort(f_batch)
    i, j, k, f_batch = i[order], j[order], k[order], f_batch[order]

    u_pred = build_frames(pred_pos[i], pred_pos[j], pred_pos[k])
    u_true = build_frames(true_pos[i], true_pos[j], true_pos[k])
    t_pred, t_true = pred_pos[j], true_pos[j]

    # -------- unique batch IDs -----------------------------------------
    uniq, counts = torch.unique_consecutive(f_batch, return_counts=True)
    f_splits = counts.cumsum(0)

    total_loss, total_atoms = 0.0, 0
    start = 0
    for b, end in zip(uniq.tolist(), f_splits.tolist()):
        f_slice = slice(start, end)              # frames of this batch
        a_mask  = batch_idx == b                 # atoms of this batch

        # local tensors
        uP, uT   = u_pred[f_slice], u_true[f_slice]         # [F_b,3,3]
        tP, tT   = t_pred[f_slice], t_true[f_slice]         # [F_b,3]
        P,  T    = pred_pos[a_mask], true_pos[a_mask]       # [A,3]

        # broadcast & rotate
        diffP = P.unsqueeze(0) - tP.unsqueeze(1)            # [F_b,A,3]
        diffT = T.unsqueeze(0) - tT.unsqueeze(1)
        rP    = torch.einsum('fij,faj->fai', uP, diffP)
        rT    = torch.einsum('fij,faj->fai', uT, diffT)

        #d      = (rP - rT).square().sum(-1).sqrt() + 1e-4   # [F_b,A]
        sq = (rP - rT).square().sum(-1).clamp(min=1e-6, max=cut*cut)
        d  = sq.sqrt()
        loss_b = huber(d, delta, cut).mean()                # scalar

        total_loss  += loss_b * P.size(0)   # weight by atom count
        total_atoms += P.size(0)

        start = end                          # move slice window

    return total_loss / total_atoms          # single scalar 
