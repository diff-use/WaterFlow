from typing import Optional, Callable, Set, Union
import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from torch_geometric.data import Batch
from SLAE.nn.graph_mixin import (
    GraphModuleMixin,
)

@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """Compute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = False,
        in_field: str = "atom_type",
        out_field: str = "node_features",
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        self.node_invariant_field = out_field
        self.in_field = in_field
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {self.node_invariant_field: Irreps([(self.num_types, (0, 1))])}
        """ # Not used in allegro
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        """
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, batch: Batch) -> Batch:
        atom_type = getattr(batch, self.in_field)
        type_numbers = atom_type.squeeze(-1)
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=batch.pos.dtype)

        # after we are done with the one-hot encoding, we can remove the original atom type field
        # TODO check this
        #delattr(batch, self.in_field)

        setattr(batch, self.node_invariant_field, one_hot)

        ''' # Not used in allegro
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        '''
        return batch