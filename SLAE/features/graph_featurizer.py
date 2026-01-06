from typing import List, Union
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data


from SLAE.features.fa_representation import transform_representation_fa
from SLAE.util.constants import ATOM37_TYPE
from torch_geometric.nn.pool import radius_graph


class ProteinGraphFeaturizer(nn.Module):
    """
    Featuriser for full-atom protein graph building
    """

    def __init__(
        self,
        radius: float = 8.0,
        use_atom37: bool = True,
        true_coords_copy: bool = False,
    ):
        super().__init__()
        self.radius = radius
        self.use_atom37 = use_atom37
        self.true_coords_copy = true_coords_copy

    def forward(
        self, batch: Batch
    ) -> Batch:
        
        if self.true_coords_copy:
            assert hasattr(batch, "coords_true"), "Batch does not have coords_true attribute"
            # make a copy of the batch with true_coords as coords
            batch_copy = batch.clone()
            batch_copy.coords = batch.coords_true
            # process the same way as batch
            batch_copy = transform_representation_fa(batch_copy)
            batch_copy.atom37_type = batch_copy.atom_type.clone()
            local_r = self.radius
            batch_copy.edge_index = radius_graph(x = batch_copy.pos,
                                        r=local_r, # 8.0
                                        batch=batch_copy.batch,
                                        loop=False,
                                        max_num_neighbors=1000,
                                        flow="source_to_target",
                                        num_workers=1)

        # Representation
        batch = transform_representation_fa(batch)

        # Edges
        batch.atom37_type = batch.atom_type.clone()
        # convert atom_type to atom37_type
        if not self.use_atom37:
            batch.atom_type = torch.tensor(ATOM37_TYPE.to(batch.atom_type.device)[batch.atom_type]).to(batch.atom_type.device)

        # Edges
            
        local_r = self.radius
        batch.edge_index = radius_graph(x = batch.pos,
                                    r=local_r, # 8.0
                                    batch=batch.batch,
                                    loop=False,
                                    max_num_neighbors=1000,
                                    flow="source_to_target",
                                    num_workers=1)
        
    
        if self.true_coords_copy:
            # then copy over all attributes to original batch
            
            #print("Copying key with true coords:", key)
            # append _true to indicate true coords derived
            #batch[key + "_true"] = batch_copy[key]
            # set these

            batch.residue_type_true = batch_copy.residue_type
            batch.residue_index_true = batch_copy.residue_index
            batch.batch_true = batch_copy.batch
            batch.atom37_type_true = batch_copy.atom37_type
            batch.edge_index_true = batch_copy.edge_index
            batch.residue_batch_true = batch_copy.residue_batch
        #print("After featurizer", batch)
        
        return batch

