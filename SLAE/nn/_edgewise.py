from typing import Optional, Callable, Set, Union
import math

import torch
from torch_geometric.data import Batch

from SLAE.nn.graph_mixin import GraphModuleMixin
from torch_scatter import scatter



class EdgewiseEnergySum(GraphModuleMixin, torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    _factor: Optional[float]

    def __init__(
        self,
        field: str,
        out_field: Optional[str] = None,
        reduce: str = "mean",
        normalize_edge_energy_sum: bool = True,
        embed_dim: int = 128, # specify the embedding dimension after allegro module
        irreps_in={},
    ):
        """Sum edges into nodes."""
        super().__init__()
        assert reduce in ("sum", "mean", "min", "max")
        self.reduce = reduce
        self.field = field
        self.out_field = f"{reduce}_{field}" if out_field is None else out_field
        self.normalize_edge_energy_sum = normalize_edge_energy_sum

        self.embed_dim = embed_dim

        self._init_irreps(
            irreps_in=irreps_in,
            my_irreps_in={self.field: f"{embed_dim}x0e"},
            irreps_out={self.out_field: f"{embed_dim}x0e"},
        )


    def forward(self, batch: Batch) -> Batch:
        edge_center = batch.edge_index[0]


        self._factor = None
        avg_num_neighbors = batch.avg_num_neighbors
        if self.normalize_edge_energy_sum:
            self._factor = 1.0 / math.sqrt(avg_num_neighbors)

        #edge_eng = batch.edge_embed
        edge_eng = getattr(batch, self.field)
        assert edge_eng is not None, "Edge embedding is required for EdgewiseEnergySum"
        assert edge_eng.shape[-1] == self.embed_dim, "Edge embedding dimension mismatch"
        species = batch.atom_type.squeeze(-1) # shape (num_atoms, 1)

        atom_eng = scatter(src = edge_eng, index = edge_center, dim=0, dim_size=len(species), reduce=self.reduce)
        factor: Optional[float] = self._factor  # torchscript hack for typing
        if factor is not None:
            atom_eng = atom_eng * factor

        #
        setattr(batch, self.out_field, atom_eng)

        return batch
    


