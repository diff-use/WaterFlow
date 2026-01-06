from typing import Optional, Union
from torch_geometric.data import Batch

import numpy as np

import torch





def get_avg_num_neighbors(
    dataset: Union[Batch],
) -> Optional[float]:
    # Compute avg_num_neighbors
    all_counts = []
    # The batch object must have been created via from_data_list()
    #  in order to be able reconstruct the initial objects.
    #  This is necessary to access the edge_index attribute.
    # DONOT use the from_data_list() method in the dataset class

    all_edge_index = dataset.edge_index
    centers = all_edge_index[0]
    counts = torch.unique(centers, sorted=True, return_counts=True)[1]
    # in case the cutoff is small and some nodes have no neighbors,
    # we need to pad `counts` up to the right length
    counts = torch.nn.functional.pad(
        #counts, pad=(0, len(data[AtomicDataDict.POSITIONS_KEY]) - len(counts))
        counts, pad=(0, len(dataset.pos) - len(counts))
    )


    
    # take the mean and variance of the counts
    ann  = torch.mean(counts.float())
    var_nn = torch.var(counts.float(), unbiased=True)

    return ann, var_nn
