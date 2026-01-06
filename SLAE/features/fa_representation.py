import torch
from torch_geometric.data import Batch

from SLAE.util.constants import FILL
from SLAE.io.atom_tensor import atom37_to_atoms



def custom_batch_from_data_list(data_list):
    cumsum_num_residues = 0
    batch_list_adjusted = []
    all_res_idx_lists = []
    max_len = 0
    slice_idxs = []
    for data in data_list:
        # FIX #1: Use precomputed pos, residue_index, atom_type if available
        if hasattr(data, 'pos') and hasattr(data, 'residue_index') and hasattr(data, 'atom_type'):
            # Already precomputed during preprocessing
            data.residue_index = data.residue_index + cumsum_num_residues
        else:
            # Fallback to computing on-the-fly (for backward compatibility)
            coords, residue_index, atom_type = atom37_to_atoms(data.coords)
            data.pos = coords
            data.residue_index = residue_index + cumsum_num_residues
            data.atom_type = atom_type.long()

        data.num_nodes = data.pos.shape[0]

        residue_int_list = data.residue_id

        # Keep track of length so we can pad properly
        max_len = max(max_len, len(residue_int_list))
        all_res_idx_lists.append(residue_int_list)

        if hasattr(data, 'slice_idx'):
            slice_idxs.append(data.slice_idx)

        batch_list_adjusted.append(data)

        # Update cumulative sum of residues
        cumsum_num_residues += data.coords.shape[0]
        #logger.info(f"Cumulative sum of residues: {cumsum_num_residues}")

    batch = Batch.from_data_list(batch_list_adjusted, exclude_keys=['residue_index', 'slice_idx'])
    batch.residue_index = torch.cat([data.residue_index for data in batch_list_adjusted])
    #logger.info(f"Residue index range after batching: {torch.min(batch.residue_index)} - {torch.max(batch.residue_index)}")
    if len(slice_idxs) > 0:
        #logger.info(f"Setting slice_idx to {torch.stack(slice_idxs)}")
        batch.slice_idx = torch.stack(slice_idxs)
  
    # Pad residue_index to max_lenB = len(batch_list_adjusted)
    B = len(batch_list_adjusted)
    padded_res_idx = -1 * torch.ones((B, max_len), dtype=torch.long)
    for i, res_idx_list in enumerate(all_res_idx_lists):
        padded_res_idx[i, :len(res_idx_list)] = torch.tensor(res_idx_list)

    batch.residue_id = padded_res_idx
    #logger.info(f"Res id is now {batch.residue_id}")
    return batch




def transform_representation_fa(
    x: Batch
) -> Batch:
    """
    Factory method to transform a batch into a specified representation.

    :param x: A minibatch of data
    :type x: Batch

    """
    batch = x
    residue_batch = batch.batch
    batch = custom_batch_from_data_list(batch.to_data_list())
    batch.residue_batch = residue_batch
    assert hasattr(batch, "atom_type")

    return batch


