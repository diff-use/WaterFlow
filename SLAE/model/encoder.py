from typing import List, Dict

import torch
import torch_scatter
from e3nn import o3

from torch_geometric.data import Batch



from SLAE.nn.cutoffs import PolynomialCutoff
from SLAE.nn.radial_basis import BesselBasis, NormalizedBasis

from SLAE.nn.embedding import (
    OneHotAtomEncoding,
    SphericalHarmonicEdgeAttrs,
    RadialBasisEdgeEncoding,
)

from SLAE.nn import (
    ScalarMLP,
    Allegro_Module, 
    EdgewiseEnergySum, 
    AtomwiseReduce,
)

from SLAE.util.avg_neighbor import get_avg_num_neighbors

class ProteinEncoder(torch.nn.Module):
    def __init__(
        self,
        atom_representation: str = "atom37",
        edge_attr: str = "edge_attr",
        edge_invariant_field: str = "edge_embedding",
        node_invariant_field: str = "node_features",
        edge_energy_field: str = "edge_energy",
        atom_energy_field: str = "node_energy",
        residue_energy_field: str = "residue_energy",
        l_max: int = 2,
        parity: str = "o3_full",
        r_max: float = 6.0,
        num_basis: int = 8,
        BesselBasis_trainable: bool = True,
        PolynomialCutoff_p: int = 6,
        normalize_basis: bool = True,
        num_layers: int = 2,
        env_embed_multiplicity: int =  64,
        embed_initial_edge: bool = True,
        two_body_latent_mlp_latent_dimensions: List[int] = [128, 256, 512, 1024],
        two_body_latent_mlp_nonlinearity: str = "silu",
        two_body_latent_mlp_initialization: str = "uniform",
        latent_mlp_latent_dimensions: List[int] = [1024, 1024, 1024],
        latent_mlp_nonlinearity: str = "silu",
        latent_mlp_initialization: str = "uniform",
        latent_resnet: bool = True,
        env_embed_mlp_latent_dimensions: List = [], # TODO
        env_embed_mlp_nonlinearity: str = None,
        env_embed_mlp_initialization: str = "uniform",
        edge_eng_mlp_latent_dimensions: List = [1024], # TODO
        edgewise_reduce: str = "sum",
        atomwise_reduce: str = "mean",
        out_dim: int = 128, # TODO
        readout: str = "mean",
    ):
        super().__init__()
        self.atom_energy_field = atom_energy_field
        self.residue_energy_field = residue_energy_field
        assert parity in ("o3_full", "o3_restricted", "so3")
        if atom_representation == "atom37":
            num_types = 37
        elif atom_representation == "element":
            num_types = 4
        else: 
            raise ValueError(f"Unsupported atom representation: {atom_representation}")


        irreps_edge_sh, nonscalars_include_parity = self._configure_irreps(
            l_max, parity
        )


        # define layers
        self.one_hot = OneHotAtomEncoding(
            num_types = num_types,
            out_field = node_invariant_field,
        )

        '''
        self.besselbasis = BesselBasis(
            r_max=r_max, 
            num_basis = num_basis, 
            trainable=BesselBasis_trainable)
        self.normalizedbasis = NormalizedBasis(
            r_max = r_max,
            original_basis = BesselBasis,
            original_basis_kwargs={"r_max": r_max, "num_basis": num_basis, "trainable": BesselBasis_trainable},)
        '''
        self.radial_basis = RadialBasisEdgeEncoding(
            basis=(
                NormalizedBasis if normalize_basis 
                else BesselBasis
            ),
            basis_kwargs={"r_max": r_max, 
                          "original_basis_kwargs": {"r_max": r_max, "num_basis": num_basis, "trainable": BesselBasis_trainable}
                          },
            cutoff = PolynomialCutoff, 
            cutoff_kwargs = {"r_max" : r_max, "p": PolynomialCutoff_p},
            out_field= edge_invariant_field,# "edge_embedding",  #AtomicDataDict.EDGE_EMBEDDING_KEY,
            irreps_in = self.one_hot.irreps_out, # TODO
        )
        #self.one_hot._add_independent_irreps(self.one_hot.irreps_out)
        

        self.spharm = SphericalHarmonicEdgeAttrs(
            irreps_edge_sh = irreps_edge_sh,
            out_field = edge_attr, #AtomicDataDict.EDGE_ATTRS_KEY,
            irreps_in=self.radial_basis.irreps_out,
        )
        #self.spharm._add_independent_irreps(self.radial_basis.irreps_out)

        two_body_latent_kwargs = {
            #"mlp_input_dimension" :  None, # TODO check this
            #"mlp_output_dimension" : None,
            "mlp_latent_dimensions" : two_body_latent_mlp_latent_dimensions,
            "mlp_nonlinearity" : two_body_latent_mlp_nonlinearity,
            "mlp_initialization" : two_body_latent_mlp_initialization,
        }

        latent_kwargs = {
            #"mlp_input_dimension" : None, # TODO check this
            #"mlp_output_dimension" : None,
            "mlp_latent_dimensions" : latent_mlp_latent_dimensions,
            "mlp_nonlinearity" : latent_mlp_nonlinearity,
            "mlp_initialization" : latent_mlp_initialization,
        }

        env_embed_kwargs = {
            #"mlp_input_dimension": None, # TODO check this
            #"mlp_output_dimension" : None,
            "mlp_latent_dimensions" : env_embed_mlp_latent_dimensions,
            "mlp_nonlinearity" :  env_embed_mlp_nonlinearity,
            "mlp_initialization" : env_embed_mlp_initialization,
        }

        
        # Core Allegro model
        self.allegro = Allegro_Module(
            field = edge_attr, #"edge_attr",#AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is edge spherical harmonics (SH)
            edge_invariant_field = edge_invariant_field, #"edge_embedding",  #AtomicDataDict.EDGE_EMBEDDING_KEY,
            node_invariant_field = node_invariant_field, #"node_features", #AtomicDataDict.NODE_ATTRS_KEY,
            latent_out_field = edge_energy_field, #"edge_energy", #EDGE_ENERGY,
            num_layers = num_layers,
            num_types = num_types,
            r_max = r_max,
            #avg_num_neighbors = avg_num_neighbors,
            PolynomialCutoff_p = PolynomialCutoff_p,
            env_embed_multiplicity = env_embed_multiplicity,
            env_embed_kwargs = env_embed_kwargs,
            embed_initial_edge = embed_initial_edge,
            nonscalars_include_parity = nonscalars_include_parity,
            two_body_latent_kwargs=two_body_latent_kwargs,
            latent_kwargs = latent_kwargs,
            latent_resnet = latent_resnet,
            irreps_in = self.spharm.irreps_out, #TODO
        )

        #elf.allegro._add_independent_irreps(self.spharm.irreps_out)

        # Edge-wise energy layer
        self.edge_eng = ScalarMLP(
            field = edge_energy_field, 
            out_field = edge_energy_field, #EDGE_ENERGY,  
            mlp_latent_dimensions = edge_eng_mlp_latent_dimensions,
            mlp_output_dimension = out_dim,
            irreps_in = self.allegro.irreps_out,
        )
        #self.edge_eng._add_independent_irreps(self.allegro.irreps_out)

        # Sum edge-wise energies to per-atom energies
        self.edge_eng_sum = EdgewiseEnergySum(
            field = edge_energy_field,#"edge_energy", 
            out_field = atom_energy_field, #"node_features", 
            reduce = edgewise_reduce, 
            embed_dim = out_dim,
            irreps_in  = self.edge_eng.irreps_out,
        )
        #self.edge_eng_sum._add_independent_irreps(self.edge_eng.irreps_out)
    
        # Final energy readout: sum per-atom energies for total energy
        self.total_energy_sum = AtomwiseReduce(
            field = atom_energy_field,#"node_features",
            out_field = residue_energy_field,#"residue_features",
            feature_dim = out_dim,
            pooling_type = atomwise_reduce,
            irreps_in = self.edge_eng_sum.irreps_out,
        )
        #self.total_energy_sum._add_independent_irreps(self.edge_eng_sum.irreps_out)
        
        # Set readout option
        self.readout = readout
        self.embedding_dim = out_dim

        self.required_batch_attributes = {"pos", "atom_type", "edge_index", "batch", "residue_index"}
    

    def _configure_irreps(self, l_max, parity_setting):
        """Configure irreducible representations (irreps) based on l_max and parity settings."""
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(l_max, p=(1 if parity_setting == "so3" else -1))
        )
        nonscalars_include_parity = parity_setting == "o3_full"
        return irreps_edge_sh, nonscalars_include_parity
    """
    def required_batch_attributes(self) -> Set[str]:
        #TODO check this
        return {
            "pos",
            "atom_type", # [n_atom, 1] long tensor, (``[0-36]``),
            "edge_index",
            "batch",
            "residue_index", #residue_index (``N_atoms``),
    
        }
    """
    def forward(self, batch: Batch) -> Dict:
        """Defines the forward pass with Allegro layers and customizable readout."""
        avg_nn, _ = get_avg_num_neighbors(batch)
        #print("#########################################")
        #print(avg_nn)  # eg. 31.87
        setattr(batch, "avg_num_neighbors", avg_nn)
        # string together the layers
        batch = self.one_hot(batch)
        batch = self.radial_basis(batch)
        batch = self.spharm(batch)
        #print(batch)
        batch = self.allegro(batch)
        batch = self.edge_eng(batch)
        batch = self.edge_eng_sum(batch)
        atomic_outputs = self.total_energy_sum(batch)

        # Node embeddings
        node_embeddings = getattr(atomic_outputs, self.atom_energy_field)
        residue_embeddings = getattr(atomic_outputs, self.residue_energy_field)

        # Apply readout function
        if self.readout == "sum":
            graph_embeddings = torch_scatter.scatter(node_embeddings, batch.batch, dim=0, reduce="sum")
        elif self.readout == "mean":
            graph_embeddings = torch_scatter.scatter(node_embeddings, batch.batch, dim=0, reduce="mean")
        else:
            raise ValueError(f"Unsupported readout type: {self.readout}")

        return {
            "node_embedding": node_embeddings,
            "residue_embedding": residue_embeddings,
            "graph_embedding": graph_embeddings,
        }