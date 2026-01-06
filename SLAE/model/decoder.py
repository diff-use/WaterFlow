import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from loguru import logger




from torch_geometric.utils import add_self_loops

from SLAE.nn.mlp_decoder import MLPDecoder, PositionDecoder
from SLAE.nn import (
    TransformerStack,
    Dim6RotStructureHead,
)

from SLAE.util.constants import VALID_ATOM37_MASK



FILL = 1e-5


def atom_mask_from_seq_pred(seq_pred: torch.Tensor, mask: torch.Tensor = None):
    """
    Generates an Atom37 mask from a residue type prediction.
    aa_pred is [N_valid_residues, 23]
    mask is [B, N] that indicates valid residues, for invalid residues

    return [B, N, 37] mask 
    """
    B, N = mask.shape
    device = seq_pred.device
    atom_mask = torch.zeros((B, N, 37), device=device)
    valid_mask37  = VALID_ATOM37_MASK.to(device=device)

    seq_pred_copy = seq_pred.clone()

    residue_types = torch.argmax(seq_pred_copy, dim=-1)  # [N_valid_residues]
    atom_mask[mask] = valid_mask37[residue_types]

    return atom_mask


def generate_atom37_mask(
    aa_pred: torch.Tensor,
    precomputed_mask: torch.Tensor = VALID_ATOM37_MASK,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Generates an Atom37 mask using a precomputed mask tensor, considering padding.

    :param aa_pred: Residue type predictions of shape [batch_size, seq_len, num_residues].
    :param precomputed_mask: Precomputed Atom37 mask tensor of shape [num_residues, num_atoms].
    :param mask: Binary mask indicating valid residues [batch_size, seq_len].
    :returns: Atom37 mask of shape [batch_size, seq_len, num_atoms].
    """
    # TODO: if max is more than 20, bound down to 20 
    if aa_pred.size(-1) > 20:
        # Force out-of-range classes to never be argmax by setting their logits to -∞
        aa_pred[..., 20:] = -1e9

    #logger.info(f"aa_pred shape: {aa_pred.shape}")
    residue_types = torch.argmax(aa_pred, dim=-1)  # [batch_size, seq_len]
    
    #logger.info(f"maximum residue type: {residue_types.max()}")
    precomputed_mask = precomputed_mask.to(residue_types.device)
    assert residue_types.max().item() < precomputed_mask.shape[0], f"more residue types {residue_types.max().item()} than mask {precomputed_mask.shape[0]} "
    atom_mask = precomputed_mask[residue_types]  # [batch_size, seq_len, num_atoms]

    # Apply mask to zero out invalid residues
    if mask is not None:
        atom_mask = atom_mask * mask.unsqueeze(-1)

    return atom_mask


def reconstruct_edges(
    bb_pred: torch.Tensor, # [B, L, 3, 3]
    atom_mask: torch.Tensor, # [B, L, 37]
    mask: torch.Tensor, # [B, L]
    residue_batch: torch.Tensor,
    noise : float = None,
    move_backbone: bool = False) -> (torch.Tensor, torch.Tensor):

    B, L, _, _ = bb_pred.shape
    assert atom_mask.shape[-1] == 37, "Atom mask must have 37 atom types"

    # expand batch to [B, L, 37]
    device = bb_pred.device
    res_batch_2d = torch.zeros(B, L, dtype=residue_batch.dtype).to(device)  # or -1, if you prefer a sentinel
   
    b_inds, l_inds = mask.nonzero(as_tuple=True)  # each is 1D of length N_residues_valid
    res_batch_2d[b_inds, l_inds] = residue_batch.to(device)
    if move_backbone:
        res_batch_3d = res_batch_2d.unsqueeze(-1).expand(-1,-1,37)
    else:
        res_batch_3d = res_batch_2d.unsqueeze(-1).expand(-1,-1,34)


    # 1) Select Cα (index 1) as reference positions
    ca_positions = bb_pred[:, :, 1, :] * mask.unsqueeze(-1)  
    assert list(ca_positions.shape) == [B, L, 3], f"ca_positions shape {ca_positions.shape} is not [B, L, 3]"

    # 2) Identify valid side-chain atoms (exclude backbone atoms: N, Cα, C)
    if move_backbone == False:
        side_chain_mask = atom_mask[:, :, 3:]  # [B, L, 34]
        valid_mask = (side_chain_mask * mask.unsqueeze(-1)).bool()  # [B, L, 34]
        assert list(valid_mask.shape) == [B, L, 34], f"valid_mask shape {valid_mask.shape} is not [B, L, 34]"
    else:
        valid_mask = atom_mask * mask.unsqueeze(-1).bool() # [B, L, 37]
        assert list(valid_mask.shape) == [B, L, 37], f"valid_mask shape {valid_mask.shape} is not [B, L, 37]"

    # 3) Gather side-chain coordinates
    ca_positions_repeated = ca_positions.unsqueeze(2).expand(-1, -1, 34, -1) 
    assert list(ca_positions_repeated.shape) == [B, L, 34, 3], f"ca_positions_repeated shape {ca_positions_repeated.shape} is not [B, L, 34, 3]"
    if move_backbone:
        # append bb_pred[:, :, :3, :] to ca_positions_repeated
        bb_positions = bb_pred[:, :, :3, :] * mask.unsqueeze(-1).unsqueeze(-1)
        ca_positions_repeated = torch.cat([ca_positions_repeated, bb_positions], dim=2) # [B, L, 37, 3]
        assert ca_positions_repeated.shape[-2] == 37, f"Number of atoms in ca_positions_repeated with shape {ca_positions_repeated.shape} is not 37"
        assert ca_positions_repeated.shape[-1] == 3, f"Number of coordinates in ca_positions_repeated  with shape {ca_positions_repeated.shape} is not 3"


    # NEW
    if noise is not None:
        # add gaussian noise with std = noise to the positions
        ca_positions_repeated = ca_positions_repeated + torch.randn_like(ca_positions_repeated) * noise
        #assert list(ca_positions_repeated.shape) == [B, L, 37, 3], f"ca_positions_repeated shape {ca_positions_repeated.shape} is not [B, L, 37, 3]"

    
    
    pos = ca_positions_repeated[valid_mask.bool()]  # [N_valid_sc_atoms, 3]

    batch = res_batch_3d[valid_mask.bool()]


    edge_fn = functools.partial(gp.compute_edges, batch=batch)
    #edges = []
    edges = edge_fn(pos, "eps_8")

    assert edges.max().item() < pos.shape[0], f"Max pos index in edges is {edges.max().item()} which is greater than the number of positions {pos.shape[0]}"


    return edges, pos

class AtomTypeEmbedder(nn.Module):
    def __init__(self, num_atom_types=37, embedding_dim=128):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, embedding_dim)

    def forward(self, atom_mask):
        """
        Args:
            atom_mask: Tensor of shape [batch_size, seq_len, num_atom_types]
                       (binary mask indicating valid atoms for each residue).

        Returns:
            atom_embeddings: Tensor of shape [batch_size, seq_len, num_atom_types, embedding_dim].
        """
        atom_indices = torch.arange(atom_mask.size(-1), device=atom_mask.device)  # [num_atom_types]
        atom_indices = atom_indices.unsqueeze(0).unsqueeze(0).expand_as(atom_mask)  # [B, L, 37]

        atom_embeddings = self.atom_embedding(atom_indices)  # [B, L, 37, embedding_dim]
        atom_embeddings = atom_embeddings * atom_mask.unsqueeze(-1)  # Zero invalid atoms
        return atom_embeddings

class ResidueAndAtomEmbedder(nn.Module):
    def __init__(self, atom_dim=128, num_atom_types=37):
        super().__init__()
        self.atom_type_embedder = AtomTypeEmbedder(num_atom_types=num_atom_types, embedding_dim=atom_dim)

    def forward(self,
                residue_embeddings, 
                atom_mask):
        """
        Args:
            residue_embeddings: Residue embeddings [batch_size, seq_len, residue_dim].
            atom_mask: Binary mask for Atom37 [batch_size, seq_len, num_atom_types].

        Returns:
            combined_sc_atom_embeddings: Tensor of shape [num_total_sc_atom, dim_embed], containing embeddings
                                          for valid side chain atoms only.
        """
        batch_size, seq_len, residue_dim = residue_embeddings.shape
        num_atom_types = atom_mask.shape[-1]

        # Generate atom type embeddings
        atom_type_embeddings = self.atom_type_embedder(atom_mask)  # [B, L, 37, atom_dim]

        # Expand residue embeddings to match atom embeddings
        residue_embeddings = residue_embeddings.unsqueeze(2)  # [B, L, 1, residue_dim]
        residue_embeddings = residue_embeddings.expand(-1, -1, num_atom_types, -1)  # [B, L, 37, residue_dim]

        # Combine residue and atom embeddings
        combined_embeddings = residue_embeddings + atom_type_embeddings  # [B, L, 37, residue_dim]

        # Exclude the first 3 columns (N, CA, C)
        combined_sc_embeddings = combined_embeddings[:, :, 3:, :]  # [B, L, 37-3, dim_embed]
        atom_sc_mask = atom_mask[:, :, 3:]  # [B, L, 37-3]

        # Select only valid side chain atoms
        valid_indices = atom_mask.bool()  # Convert mask to boolean
        valid_sc_indices = atom_sc_mask.bool()
        
        combined_atom_embeddings = combined_embeddings[valid_indices]
        combined_sc_atom_embeddings = combined_sc_embeddings[valid_sc_indices]  # [num_total_sc_atom, dim_embed]

        assert combined_atom_embeddings.shape[0] >= combined_sc_atom_embeddings.shape[0], f"combined_atom_embeddings and combined_sc_atom_embeddings have incorrect number of atoms {combined_atom_embeddings.shape[0]} and {combined_sc_atom_embeddings.shape[0]}"
        return combined_atom_embeddings, combined_sc_atom_embeddings

class PositionPredictor(nn.Module):
    def __init__(
        self,
        position_decoder: PositionDecoder,
        residue_dim=128,
        atom_dim=128,
        num_atom_types=37,
        noise=0.5,
        move_backbone=False,
    ):
        super().__init__()
        self.position_decoder = position_decoder
        self.residue_atom_embedder = ResidueAndAtomEmbedder(
            atom_dim=atom_dim, num_atom_types=num_atom_types
        )

        self.noise = noise
        self.move_backbone = move_backbone

    def forward(self, bb_pred, scalar_features, aa_pred, residue_batch, mask: torch.Tensor = None):
        """
        Args:
            bb_pred: Backbone coordinates [batch_size, max_seq_len, 3, 3].
            scalar_features: Residue embeddings [batch_size, max_seq_len, residue_dim].
            aa_pred: Residue type predictions [batch_size, max_seq_len, 20].
            mask: Padding mask [batch_size, max_seq_len].

        Returns:
            atom_coords: Predicted Atom37 positions [batch_size, max_seq_len, 37, 3].
            atom_mask: Valid Atom37 mask [batch_size, max_seq_len, 37].
        """
        # Generate Atom37 mask
        atom_mask = generate_atom37_mask(aa_pred, mask=mask)  # [B, L, 37]
        assert len(atom_mask.shape) == 3, "Atom mask is not 3D"

        # Combine residue and atom embeddings
        combined_atom_embeddings, combined_sc_atom_embeddings = self.residue_atom_embedder(scalar_features, atom_mask)

        # Reconstruct edges
        edge_index, pos = reconstruct_edges(bb_pred = bb_pred, atom_mask = atom_mask, mask = mask, residue_batch = residue_batch, noise=self.noise, move_backbone=self.move_backbone)  # [2, num_edges], [N_sc_atoms, 3]
        assert not torch.isnan(edge_index).any(), "Edge index contains NaN"

        #logger.info(f"edge_index shape: {edge_index.shape}")
        #logger.info(f"pos & side_chain_positions_ca shape: {side_chain_positions_ca.shape}")
        #logger.info(f"combined_sc_atom_embeddings shape: {combined_sc_atom_embeddings.shape}")
        # Predict displacements
        # important ADD SELF LOOP to AVOID SHAPE MISMATCH
        #logger.info(f"combined_sc_atom_embeddings.shape = {combined_sc_atom_embeddings.shape}")
        #logger.info(f"side_chain_positions_ca.shape = {side_chain_positions_ca.shape}")
        #logger.info(f"edge_index max: {edge_index.max().item()}, # nodes = {side_chain_positions_ca.size(0)}")

        edge_index_with_loops, _ = add_self_loops(
                    edge_index, num_nodes=pos.size(0)
                    )
        
        displacement = self.position_decoder(
            edge_index=edge_index_with_loops,
            scalar_features=combined_atom_embeddings if self.move_backbone else combined_sc_atom_embeddings,  # [N_valid_atoms, residue_dim]
            pos=pos,  # [N_valid_sc_atoms, 3]
        )
        #logger.info(f"displacement shape: {displacement.shape}")
        # Add displacements to CA positions
        updated_pos = pos + displacement  # [N_sc_atoms, 3] or [N_atoms, 3]

        return atom_mask, updated_pos


class PairwisePredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        downproject_dim: int,
        hidden_dim: int,
        n_bins: int,
        bias: bool = True,
        pairwise_state_dim: int = 0,
    ):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=bias)
        self.linear1 = nn.Linear(
            downproject_dim + pairwise_state_dim, hidden_dim, bias=bias
        )
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_bins, bias=bias)

    def forward(self, x, pairwise: torch.Tensor | None = None):
        """
        Args:
            x: [B x L x D]

        Output:
            [B x L x L x K]
        """
        x = self.downproject(x)
        # Let x_i be a vector of size (B, D).
        # Input is {x_1, ..., x_L} of size (B, L, D)
        # Output is 2D where x_ij = cat([x_i * x_j, x_i - x_j])
        q, k = x.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x_2d = [prod, diff]
        if pairwise is not None:
            x_2d.append(pairwise)
        x = torch.cat(x_2d, dim=-1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x

class PairwiseFeatureHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        downproject_dim: int,
        hidden_dim: int,
        output_dim: int,
        bias: bool = True,
        num_layers: int = 2,
        sigmoid: bool = True,
    ):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=bias)

        self.mlp = nn.ModuleList()
        in_dim = downproject_dim
        for _ in range(num_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            self.mlp.append(nn.GELU())
            self.mlp.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        
        self.output_layer = nn.Linear(in_dim, output_dim, bias=bias)
        self.sigmoid = sigmoid



    def forward(self, x):
        """
        Args:
            x: [B x L x D]

        Output:
            [B x L x L x K]
        """
        x = self.downproject(x)
        # Let x_i be a vector of size (B, D).
        # Input is {x_1, ..., x_L} of size (B, L, D)
        # Output is 2D where x_ij = cat([x_i * x_j, x_i - x_j])
        q, k = x.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x_2d = [prod, diff]
        x = torch.cat(x_2d, dim=-1)
         
        for layer in self.mlp:
            x = layer(x)

        x = self.output_layer(x)                

        # Apply sigmoid to constrain individual values between [0, 1]
        if self.sigmoid:
            x = torch.sigmoid(x)

        return x
    
class EnergyHead(nn.Module):
    def __init__(self,
                 d_pairwise: int,
                 energy_type: list[str] = ["hbond", "sol", "elec"], # "hbond", "sol", "elec"
                 bias: bool = False,
                 clamp: bool = True,
                 clamp_abs_min: float = 1e-3,

                 ):
        super().__init__()
        self.energy_type = energy_type
        self.clamp = clamp
        self.clamp_abs_min = clamp_abs_min
        # initialize a dictionary to store the linear layer heads
        self.energy_heads = nn.ModuleDict()
        for energy_type in self.energy_type:
            self.energy_heads[energy_type] = nn.Linear(d_pairwise, 1, bias=bias)
            

    def forward(self, x):
        # x is a B, N, N, d_pairwise tensor
        # apply a linear layer head to each energy type
        pred_eng_dict = {}
        for energy_type in self.energy_type:
            e = self.energy_heads[energy_type](x)
            if self.clamp:
                # if abs value of e is smaller than clamp_abs_min, set e to 0
                e = torch.where(torch.abs(e) < self.clamp_abs_min, torch.zeros_like(e), e)
            pred_eng_dict[energy_type] = e
        # stack energy types along the last dimension  to get B, N, N, 3
        pred_eng = torch.cat(list(pred_eng_dict.values()), dim=-1)
        return pred_eng, pred_eng_dict
    


        
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class AllAtomDecoder(nn.Module):
    def __init__(self, 
                 res_idx_chainbreak: bool = True, 
                 decoder_proj: bool = False,
                 struct_out: str = "frame", # "linear"
                 eng_after_tfm: bool = True,
                 d_in: int = 128,
                 d_model: int = 128, 
                 n_heads: int =  16, 
                 n_layers: int = 8,
                 eng_mlp_layers: int = 3, 
                 eng_type: list[str] = ["hbond", "sol", "elec"],
                 eng_clamp: bool = True,
                 eng_clamp_abs_min: float = 1e-3,
                 pos_enc: bool = True, 
                 pos_enc_type: str = "rotary", # "rel"
                 dim_pairwise: int | None = None,
                 seq_mlp_hidden_dim: list[int] = [128, 128],
                 seq_mlp_dropout: float = 0.0,
                 seq_mlp_activation: list[str] = ["relu", "relu", "none"],
                 seq_mlp_skip: str = "concat",
                 seq_mlp_out_dim: int = 20,
                 sc_noise: float = 0.5,
                 sc_move_backbone: bool = True,
                 sc_num_message_layers: int =  2,
                 sc_message_hidden_dim: int =  128,
                 sc_message_activation: str = "relu",
                 sc_message_dropout: float = 0.0,
                 sc_message_skip: bool = False,
                 sc_num_distance_layers: int =  1,
                 sc_distance_hidden_dim: int = 128,
                 sc_distance_activation: int = "relu",
                 sc_distance_dropout: float = 0.0,
                 sc_distance_skip: bool = False,
            
                 ):
        super().__init__()
        self.res_idx_chainbreak = res_idx_chainbreak
        self.decoder_proj = decoder_proj
        self.eng_after_tfm = eng_after_tfm
        self.proj = nn.Linear(d_in, d_model)
        self.decoder_channels = d_model
        self.pairwise_dim = dim_pairwise
        self.decoder_stack = TransformerStack(
            d_model, 
            n_heads, 
            1, 
            n_layers, 
            scale_residue=False, 
            n_layers_geom=0, 
            pos_enc = pos_enc,
            pos_enc_type = pos_enc_type,
            dim_pairwise = dim_pairwise
        )

        self.struct_out = struct_out
        if self.struct_out == "frame":
            self.affine_output_projection = Dim6RotStructureHead(
                self.decoder_channels, 10, predict_torsion_angles=True
            )
        elif self.struct_out == "linear":
            # [batch_size, max_seq_len, d_model] -> [batch_size, max_seq_len, 9] -> [batch_size, max_seq_len, 3, 3]
            self.linear_output_projection = nn.Sequential(
                LayerNorm(d_model),
                nn.Linear(d_model, 9),
                # reshape the last dimension to 3, 3
                nn.Unflatten(dim=-1, unflattened_size=(3, 3))
            )
        elif self.struct_out == "linear_aa":
            self.linear_output_projection = nn.Sequential(
                LayerNorm(d_model),
                nn.Linear(d_model, 37*3),
                # reshape the last dimension to 37, 3
                nn.Unflatten(dim=-1, unflattened_size=(37, 3))
            )


        if self.decoder_proj:
            # replace all entries in seq_mlp_hidden_dim with d_model
            seq_mlp_hidden_dim = [d_model for _ in seq_mlp_hidden_dim]
            sc_message_hidden_dim = d_model
            sc_distance_hidden_dim = d_model

        self.seq_mlp = MLPDecoder(hidden_dim=seq_mlp_hidden_dim,
                                        dropout=seq_mlp_dropout,
                                        activations=seq_mlp_activation,
                                        skip=seq_mlp_skip,
                                        out_dim=seq_mlp_out_dim, 
                                        input = "node_embedding")
        self.num_class = seq_mlp_out_dim

        self.sidechain_position_predictor = PositionPredictor(
            position_decoder=PositionDecoder(num_message_layers = sc_num_message_layers,
                                            message_hidden_dim = sc_message_hidden_dim,
                                            message_activation = sc_message_activation,
                                            message_dropout = sc_message_dropout,
                                            message_skip = sc_message_skip,
                                            num_distance_layers = sc_num_distance_layers,
                                            distance_hidden_dim = sc_distance_hidden_dim,
                                            distance_activation = sc_distance_activation,
                                            distance_dropout = sc_distance_dropout,
                                            distance_skip = sc_distance_skip),
            residue_dim=d_model,
            atom_dim=d_model,
            num_atom_types=37,
            noise = sc_noise,
            move_backbone = sc_move_backbone,
        )

        self.pairwise_bins = [2,  # contact map
                              64] # distogram
        self.pairwise_classification_head = PairwisePredictionHead(
            self.decoder_channels,
            downproject_dim=64,
            hidden_dim=64,
            n_bins=sum(self.pairwise_bins),
            bias=False,
        )

        self.pairwise_energy_feat =  PairwiseFeatureHead(
            input_dim = self.decoder_channels,
            downproject_dim = 64,
            hidden_dim = 64,
            output_dim = self.pairwise_dim ,
            num_layers = eng_mlp_layers,
            bias = False,
            sigmoid = True,
        )


        self.energy_head = EnergyHead(
            d_pairwise = self.pairwise_dim,
            energy_type = eng_type,
            bias = False,
            clamp = eng_clamp,
            clamp_abs_min = eng_clamp_abs_min,
        )
        
      

    def forward(self, 
                x, # [batch_size, max_seq_len, d_model]
                mask: torch.Tensor = None, # [batch_size, max_seq_len]
                batch: torch.Tensor = None,
                res_idx: torch.Tensor = None, # [batch_size, max_seq_len]
                ):
        # Basic input validation
        assert not torch.isnan(x).any(), "decoder input x contains NaN"
        assert not torch.isnan(mask).any(), "decoder input mask contains NaN"
        assert not torch.isnan(batch).any(), "decoder input batch contains NaN"
        
        # Filter out batches with no valid tokens
        valid_lengths = mask.sum(dim=1)  # shape [batch_size]
        if (valid_lengths < 1).any():
            logger.warning(
                f"Found at least one sequence with 0 valid tokens. "
                f"Per-sample valid lengths = {valid_lengths.tolist()}"
            )
        
        valid_batch_mask = valid_lengths >= 1
        valid_batch_indices = valid_batch_mask.nonzero(as_tuple=True)[0] 
        x = x[valid_batch_indices] 
        valid_residues_mask = torch.isin(batch, valid_batch_indices)
        batch = batch[valid_residues_mask]
        mask_old = mask
        mask = mask[valid_batch_indices]

        B, L = mask.shape
       
        res_idx = res_idx[valid_batch_indices.cpu()]

  
        # Project input if needed
        if self.decoder_proj:
           x = self.proj(x)



        if self.res_idx_chainbreak and res_idx is not None:
            assert res_idx is not None, "res_idx is required for chainbreak prediction"
            # constuct chain mask using residue indices
            # res_idx is [batch_size, max_seq_len], consecutive residues are in the same chain
            #TODO
            chainbreaks = torch.zeros((B, L), dtype=torch.bool, device=x.device)
            chainbreaks[:, 1:] = (res_idx[:, 1:] - res_idx[:, :-1]) != 1
            # turn that into chainbreak logits
            chain_ids = torch.cumsum(chainbreaks, dim=-1)  # shape (B, L)
            chain_mask = chain_ids.unsqueeze(-1) == chain_ids.unsqueeze(-2)  # (B, L, L)
            assert chain_mask.shape[1] == L and chain_mask.shape[2] == L, f"chain_mask shape {chain_mask.shape} does not match L {L}"


        if not self.eng_after_tfm:
            logger.info("Predicting energy BEFORE transformer stack!")
            pred_eng_feat = self.pairwise_energy_feat(x)
            assert pred_eng_feat.shape[-1] == self.pairwise_dim, f"pred_eng_feat shape {pred_eng_feat.shape} does not match pairwise_dim {self.pairwise_dim}"
            pred_eng, _ = self.energy_head(pred_eng_feat)

        # Run through transformer stack
        x, _, _ = self.decoder_stack.forward(
            x, 
            affine=None, 
            affine_mask=None, 
            sequence_id=torch.zeros_like(x[:, :, 0:1], dtype=torch.int64), 
            chain_id=torch.zeros_like(x[:, :, 0:1], dtype=torch.int64), 
            mask=mask,
            pairwise_mask=chain_mask if self.res_idx_chainbreak else None,
            pairwise_repr=None,
            res_idx=res_idx,
        ) # x [batch_size, max_seq_len, d_model]

        pairwise_logits = self.pairwise_classification_head(x)

        if self.eng_after_tfm:
            #logger.info("Predicting energy AFTER transformer stack!")
            pred_eng_feat = self.pairwise_energy_feat(x)
            assert pred_eng_feat.shape[-1] == self.pairwise_dim, f"pred_eng_feat shape {pred_eng_feat.shape} does not match pairwise_dim {self.pairwise_dim}"
            pred_eng, _ = self.energy_head(pred_eng_feat)


        x_valid_res_flat = x[mask.bool()]
        seq_pred = self.seq_mlp(x_valid_res_flat) 
        
        # Generate backbone coordinates
        if self.struct_out == "frame":
            _, bb_pred, _ = self.affine_output_projection(
                x, 
                affine=None, 
                affine_mask=torch.zeros_like(x[:, :, 0:1], dtype=torch.bool)
            ) # bb_pred [batch_size, seq_len, 3, 3]
        elif self.struct_out == "linear":
            bb_pred = self.linear_output_projection(x)
        elif self.struct_out == "linear_aa":
            aa_pred = self.linear_output_projection(x)
            bb_pred = aa_pred[:, :, :3, :]
            sidechain_coords = aa_pred[:, :, 3:, :]
        else:
            raise ValueError(f"Invalid struct_out: {self.struct_out}")

        # Zero out backbone prediction for invalid residues
        bb_pred = bb_pred * mask.unsqueeze(-1).unsqueeze(-1)
        
        """    
        # Pad sequence predictions
        seq_pred_padded = torch.zeros((B, L, self.num_class), device=seq_pred.device)
        seq_pred_padded[mask.bool()] = seq_pred

        # Predict sidechain positions and atom mask
        atom_mask, updated_pos = self.sidechain_position_predictor(
            bb_pred, x, seq_pred_padded, batch, mask
        ) # atom_mask [batch_size, seq_len, 37], sidechain_pos [num_total_sc_atom, 3]
        """

        atom_mask = atom_mask_from_seq_pred(seq_pred, mask)

        sc_atom_mask = atom_mask[:, :, 3:]  # [batch_size, seq_len, 34]
       

        # Get valid indices for sidechain atoms
        valid_indices = sc_atom_mask.nonzero(as_tuple=False)  # [num_total_sc_atom, 3]
        batch_indices, residue_indices, atom_indices = (
            valid_indices[:, 0],
            valid_indices[:, 1],
            valid_indices[:, 2],
        )

        updated_pos = None
        # Assign sidechain positions
        if self.struct_out != "linear_aa":
            sidechain_coords = torch.full(
                (bb_pred.shape[0], bb_pred.shape[1], 34, 3),
                fill_value=FILL,
                device=bb_pred.device,
            )
            sidechain_coords[
                batch_indices, residue_indices, atom_indices
            ] = updated_pos

        # Apply mask to fill invalid residues
        """
        sidechain_coords = sidechain_coords * sc_atom_mask.unsqueeze(-1) + (
            1 - sc_atom_mask.unsqueeze(-1)
        ) * FILL
        """

        # Reassemble outputs back to full batch size
        device = x.device
        B_old, max_len = mask_old.shape
        if B_old > B:
            logger.info(f"Padding decoder output from {B} seq to {B_old}")
        
        # Initialize output tensors
        bb_pred_out = torch.zeros((B_old, max_len, 3, 3), device=device)
        atom_mask_out = torch.zeros((B_old, max_len, 37), device=device)
        sidechain_coords_out = torch.zeros((B_old, max_len, 34, 3), device=device)
        pred_eng_out = torch.zeros((B_old, max_len, max_len, 3), device=device)
        dist_pair_logits_out = torch.zeros((B_old, max_len, max_len, 64), device=device)
        contact_logits_out = torch.zeros((B_old, max_len, max_len, 2), device=device)
         
        # Fill in the valid batch indices
        bb_pred_out[valid_batch_indices] = bb_pred
        atom_mask_out[valid_batch_indices] = atom_mask
        sidechain_coords_out[valid_batch_indices] = sidechain_coords
        pred_eng_out[valid_batch_indices] = pred_eng
        dist_pair_logits_out[valid_batch_indices] = pairwise_logits[:, :, :, 2:]
        contact_logits_out[valid_batch_indices] = pairwise_logits[:, :, :, :2]

        # Return only the essential outputs
        return {
            "backbone_coords": bb_pred_out,  # [batch_size, seq_len, 3, 3]
            "sidechain_coords": sidechain_coords_out,  # [batch_size, seq_len, 34, 3]
            "seq_pred": seq_pred,  # [N_res_total, 20]
            "atom_mask": atom_mask_out,  # [batch_size, seq_len, 37]
            "dist_pair_logits": dist_pair_logits_out, # [B, N, N, 64]
            "contact_logits": contact_logits_out,
            "chainbreak_logits": None,
            "torsion_angles": None,
            "pred_eng": pred_eng_out, # [B, N, N, 3]
        }
