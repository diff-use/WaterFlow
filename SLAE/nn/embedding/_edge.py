from typing import Union

import torch
from torch_geometric.data import Batch

from e3nn import o3
from e3nn.util.jit import compile_mode


from SLAE.nn.radial_basis import BesselBasis
from SLAE.nn.cutoffs import PolynomialCutoff

from SLAE.nn.graph_mixin import (
    GraphModuleMixin,
)

from loguru import logger

@compile_mode("script")
class SphericalHarmonicEdgeAttrs(GraphModuleMixin, torch.nn.Module):
    """Construct edge attrs as spherical harmonic projections of edge vectors.

    Parameters follow ``e3nn.o3.spherical_harmonics``.

    Args:
        irreps_edge_sh (int, str, or o3.Irreps): if int, will be treated as lmax for o3.Irreps.spherical_harmonics(lmax)
        edge_sh_normalization (str): the normalization scheme to use
        edge_sh_normalize (bool, default: True): whether to normalize the spherical harmonics
        out_field (str, default: AtomicDataDict.EDGE_ATTRS_KEY: data/irreps field
    """

    out_field: str

    def __init__(
        self,
        irreps_edge_sh: Union[int, str, o3.Irreps],
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
        irreps_in=None,
        out_field: str = "edge_attr", #AtomicDataDict.EDGE_ATTRS_KEY,
    ):
        super().__init__()
        self.out_field = out_field

        if isinstance(irreps_edge_sh, int):
            self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        else:
            self.irreps_edge_sh = o3.Irreps(irreps_edge_sh)
        
        # TODO
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={out_field: self.irreps_edge_sh},
        )
        
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )

    def forward(self, data: Batch) -> Batch:
        # Compute the edge displacement vectors for a graph
        #edge_vectors = data.pos[data.edge_index[0]] - data.pos[data.edge_index[1]]
        edge_vectors = torch.index_select(data.pos, 0, data.edge_index[0]) - torch.index_select(
            data.pos, 0, data.edge_index[1])
        edge_sh = self.sh(edge_vectors)
        #data[self.out_field] = edge_sh #TODO check this
        setattr(data, self.out_field, edge_sh)
        return data


@compile_mode("script")
class RadialBasisEdgeEncoding(GraphModuleMixin, torch.nn.Module):
    out_field: str

    def __init__(
        self,
        basis=BesselBasis,
        cutoff=PolynomialCutoff,
        basis_kwargs={},
        cutoff_kwargs={},
        out_field: str = "edge_embedding", #AtomicDataDict.EDGE_EMBEDDING_KEY,
        irreps_in=None,
    ):
        super().__init__()
        self.basis = basis(**basis_kwargs)
        self.cutoff = cutoff(**cutoff_kwargs)
        self.out_field = out_field
        self._init_irreps(
            irreps_in=irreps_in,
            irreps_out={
                self.out_field: o3.Irreps([(self.basis.num_basis, (0, 1))]),
                "edge_cutoff": "0e",  # # [n_edge, 1] invariant of the radial cutoff envelope for each edge, allows reuse of cutoff envelopes
            },
        )

    def forward(self, data: Batch) -> Batch:
        
        edge_length =  torch.pairwise_distance(data.pos[data.edge_index[0, :]], data.pos[data.edge_index[1, :]])
        safe_min = 1e-6  # Small positive value
        edge_length = torch.clamp(edge_length, min=safe_min)

        setattr(data, "edge_distance", edge_length) #TODO 

        #edge_length = getattr(data, "edge_distance")
        cutoff = self.cutoff(edge_length).unsqueeze(-1)
        if torch.isnan(cutoff).any():
            logger.warning("NaN in Radial Basis Cutoff!")
        cutoff = torch.nan_to_num(cutoff, nan=0.0)
        setattr(data, "edge_cutoff", cutoff)
        edge_length_embedded = self.basis(edge_length) * cutoff
        #data[self.out_field] = edge_length_embedded
        setattr(data, self.out_field, edge_length_embedded)

        return data

