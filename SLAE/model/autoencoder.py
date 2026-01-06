import os
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Union

import hydra
import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from SLAE.util.types import ModelOutput
from SLAE.util.memory_utils import clean_up_torch_gpu_memory
from SLAE.nn.base_model import BaseModel

from SLAE.util.align import aligned_rotation

from SLAE.io.write_pdb import to_pdb


from SLAE.loss.losses import (
    backbone_distance_loss,
    backbone_pairwise_distance_classification_loss,
    backbone_pairwise_direction_loss,
    contact_map_loss,
    chainbreak_loss,
    SmoothLDDTLoss,
    contact_mse_loss,
    masked_mse_loss,
    interaction_mse_loss,
    energy_mse_loss,
    weighted_energy_mse_loss,
    energy_prediction_pr_auc,
    extreme_ca_ca_distance_violations,
    fape_loss,
    torsion_angle_loss,
    torsion_l2_loss,
    backbone_distance_loss_l2,
    backbone_distance_loss_l2mean,
    contact_consistency_loss,
    mask_l2pairwise_distance_loss,
    allatom_FAPE
)
from SLAE.util.constants import fill_in_cbeta_for_atom3_all
from SLAE.util.fape_utils import make_filtered_frames

from torch.nn.parameter import UninitializedParameter
import math
PAD_VALUE = 1e-5




def initialize_weights_transformer(model: nn.Module) -> nn.Module:
    """
    Re‑initialise a (decoder)‑style transformer in place.

    * FFN / MLP Linear layers  :  N(0, sqrt(2 / fan_in))
    * Other Linear layers      :  N(0, sqrt(2 / (5 * fan_in)))
    * Norm layers              :  weight = 1, bias = 0
    * Any other weight tensor  :  Kaiming‑like N(0, sqrt(2 / fan_in))

    Returns
    -------
    nn.Module
        The same module, modified in place.
    """
    for name, module in model.named_modules():

        # ------------------------------------------------------------------ #
        # Linear layers
        # ------------------------------------------------------------------ #
        if isinstance(module, nn.Linear):
            # Skip lazy / meta tensors
            if isinstance(module.weight, UninitializedParameter):
                logger.info(f"[init] skipping lazy Linear {name}")
                continue

            fan_in = module.in_features
            is_ffn = any(tag in name.lower() for tag in ("ffn", "mlp", "linearskipblock"))

            if is_ffn:
                std = math.sqrt(2.0 / fan_in)            # ★ fixed formula
            else:
                std = math.sqrt(2.0 / (5 * fan_in))      # ★ uses fan_in now

            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # ------------------------------------------------------------------ #
        # Normalisation layers
        # ------------------------------------------------------------------ #
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.GroupNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

        # ------------------------------------------------------------------ #
        # Fallback for any other tensor called "weight"
        # ------------------------------------------------------------------ #
        elif hasattr(module, "weight") and module.weight is not None:
            fan_in = module.weight.shape[1] if module.weight.ndim >= 2 else 1
            std = math.sqrt(2.0 / fan_in) if module.weight.ndim >= 2 else 0.02
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    return model

class AutoEncoderModel(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.config = cfg
        self.inference_only = self.config.task.get("inference_only", False)
        self.frozen_encoder = False
        self.frozen_decoder = False
   
        # self.encoder = get_protein_encoder(cfg)
        logger.info("Instantiating encoder...")
        self.encoder: nn.Module = hydra.utils.instantiate(cfg.encoder)
        logger.info(self.encoder)

        # INIT VQ    
        if hasattr(self.config, "vq") and self.config.vq is not None:
            logger.info("Instantiating quantizer...")
            self.quantizer: nn.Module = hydra.utils.instantiate(self.config.vq)
            logger.info(self.quantizer)
        else:
            self.quantizer = None

        logger.info("Instantiating decoders...")
        self.decoder: nn.ModuleDict = self._build_output_decoders()
        logger.info(self.decoder)

        if self.config.task.pretrained_ckpt is not None:
            logger.info(f"Loading pretrained weights from {self.config.task.pretrained_ckpt}")
            self._load_pretrained_weights()
            logger.info("--------------------------------Finished loading pretrained weights--------------------------------")

        if self.config.task.init_decoder:
            logger.info("Reinitializing decoder weights")
            self.decoder.allatom_decoder = initialize_weights_transformer(self.decoder.allatom_decoder)

        # Freeze encoder if specified
        if self.config.task.get("freeze_encoder", True):
            self._freeze_encoder()
            
        # Freeze decoder if specified
        if self.config.task.get("freeze_decoder", True):
            self._freeze_decoder()

            if self.config.task.get("unfreeze_decoder_parts"):
                # parts is a list of strings, each string is a part of the decoder
                # eg. ["proj", "chain_break_prediction_head", "pairwise_energy_feat", "energy_head"]
                for part in self.config.task.unfreeze_decoder_parts:
                    self._unfreeze_decoder_part(part)
    
        if self.inference_only:
            self._freeze_all_parameters()
            logger.info("Running in inference-only mode. All parameters are frozen.")


        logger.info("Instantiating losses...")
        self.losses = self.configure_losses(cfg.task.losses)
        logger.info(f"Using losses: {self.losses}")

        if self.config.get("task.aux_loss_coefficient"):
            logger.info(
                f"Using aux loss coefficient: {self.config.task.aux_loss_coefficient}"
            )
        else:
            logger.info("Not using aux loss scaling")

        logger.info("Configuring metrics...")
        self.metrics = self.configure_metrics()
        logger.info(self.metric_names)

        logger.info("Instantiating featuriser...")
        self.featuriser: nn.Module = hydra.utils.instantiate(cfg.features)
        logger.info(self.featuriser)

        logger.info("Instantiating task transform...")
        self.task_transform = hydra.utils.instantiate(
            cfg.get("task.transform")
        )
        logger.info(self.task_transform)


        self.save_hyperparameters()
        
        self.clip_decoder_grad = self.config.task.clip_decoder_grad

        logger.info("--------------------------------FINISHED INITIALIZING MODEL--------------------------------")

       
    def _load_pretrained_weights(self) -> None:
        """Load pretrained weights from checkpoint."""
        ckpt_path = self.config.task.pretrained_ckpt
        logger.info(f"Loading pretrained weights from: {ckpt_path}")
        
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Extract state dict
        state_dict = checkpoint["state_dict"]
        
        # Remove 'encoder.' prefix from keys for encoder weights
        encoder_dict = {
            k.replace("encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        
        # Load encoder weights
        missing, unexpected = self.encoder.load_state_dict(encoder_dict, strict=False)
        logger.info(f"Loaded encoder weights. Missing: {missing}, Unexpected: {unexpected}")
        
        # Load decoder weights if they exist
        decoder_dict = {
            k.replace("decoder.allatom_decoder.", "allatom_decoder."): v
            for k, v in state_dict.items()
            if k.startswith("decoder.")
        }

        if hasattr(self.config.task, "reinit_decoder_modules"):
            for module in self.config.task.reinit_decoder_modules:
                logger.info(f"Ignoring checkpoint weights for module {module}")
                remove_list = []
                for k, v in decoder_dict.items():
                    if module in k:
                        logger.info(f"Reinitializing {k} weights")
                        # remove the module name from dict
                        remove_list.append(k)
                for k in remove_list:
                    decoder_dict.pop(k)

        if decoder_dict:
            missing, unexpected = self.decoder.load_state_dict(decoder_dict, strict=False)
            logger.info(f"Loaded decoder weights. Missing: {missing}, Unexpected: {unexpected}")
            
        # Move model to correct device
        self.to(self.device)

    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder = True
        logger.info("Froze all encoder parameters; only decoder will be trainable.")
    
    def _freeze_decoder(self) -> None:
        """Freeze all decoder parameters."""
        for param in self.decoder.parameters():
            param.requires_grad = False
        self.frozen_decoder = True
        logger.info("Froze all decoder parameters; only encoder will be trainable.")

    def _unfreeze_decoder_part(self, part: str) -> None:
        """Unfreeze a specific part of the decoder."""
        for param in getattr(self.decoder["allatom_decoder"], part).parameters():
            param.requires_grad = True
        logger.info(f"Unfroze decoder part {part}")
    
    def _freeze_all_parameters(self) -> None:
        """Freeze all model parameters for inference-only mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.frozen_encoder = True
        self.frozen_decoder = True
        logger.info("Froze all model parameters for inference-only mode.")

    def get_labels(self, batch: Batch) -> Dict:
        """
        Computes or retrieves labels from a batch of data.

        Labels are returned as a dictionary of tensors indexed by output name.

        :param batch: Batch of data to compute labels for
        :type batch: Batch
        :return: Dictionary of labels indexed by output name
        :rtype: Label
        """
        labels: Dict[str, torch.Tensor] = {}
        for output in self.config.task.supervise_on:
            
            if output == "residue_type":

                
                labels["residue_type"] = batch.residue_type
                mask = torch.ones_like(labels["residue_type"], dtype=torch.bool)

            elif output == "energy":
                    labels["pdb_id"] = batch.id
                    # TODO hard code for now
                    energy_dir = self.config.task.energy_dir
                    device = batch.coords.device
                    if hasattr(batch, 'slice_idx'):
                        slice_idxs = batch.slice_idx
                    else:
                        slice_idxs = None
                    labels["energy"] = self._preload_energy_batch(labels["pdb_id"], energy_dir, device, slice_idxs)

            elif output == "backbone_coords" or output == "allatom_coords":
                logger.info(f"Debugging batch index for {batch} with residue_batch shape {batch.residue_batch.shape} and range {batch.residue_batch.min()} to {batch.residue_batch.max()}")
                mask = torch.ones_like(batch.residue_batch, dtype=torch.bool)
                #logger.info(f"batch is {batch}")
                # TODO
                if hasattr(batch, "coords_true"):
                    coords = batch.coords_true[mask, : , :]
                    # if have other true attributes, copy them over
                    """
                    print("!Using true coords for labels!")
                    if hasattr(batch, "residue_type_true"):
                        batch.residue_type = batch.residue_type_true
                        batch.residue_index = batch.residue_index_true
                        batch.batch = batch.batch_true
                        batch.atom37_type = batch.atom37_type_true
                        batch.edge_index = batch.edge_index_true
                    """
                    residue_batch = batch.residue_batch[mask]
        
                else:
                    coords = batch.coords[mask, : , :]
                    residue_batch = batch.residue_batch[mask]
                #coords = batch.coords[mask, : , :]
                #logger.info(f"How many different proteins in atom_batch {len(torch.unique(batch.batch))}")
                #logger.info(f"How many different proteins in residue_batch {len(torch.unique(batch.residue_batch))}")
                #assert self.config.dataset.datamodule.batch_size == len(torch.unique(batch.residue_batch)), f"batch_size {self.config.dataset.datamodule.batch_size} != len(torch.unique(batch.residue_batch)) {len(torch.unique(batch.residue_batch))}"
                

                batch_size = len(batch.id)#self.config.dataset.datamodule.batch_size
                lengths = torch.bincount(residue_batch, minlength=batch_size)  # [batch_size]
                    

                if output == "backbone_coords":
                    backbone_coords = coords[:,:3,:] # no oxygen # (N_res, 3, 3)
            

                    # padding to make the shape [B, N_res, 3, 3] with start_end list
                    backbone_coords_padded = []

                    start = 0
                    for length in lengths:
                        seq_len = length.item()
                        backbone_coords_padded.append(backbone_coords[start: start + seq_len])  # [seq_len_i, 3]
                        start += seq_len
            
                    backbone_coords = torch.nn.utils.rnn.pad_sequence(backbone_coords_padded, batch_first=True, padding_value=PAD_VALUE)

                    labels["backbone_coords"] = backbone_coords
                    #logger.info(f"true backbone_coords has the shape {backbone_coords.shape}")
                elif output == "allatom_coords":
                    sc_coords = coords[:, 3:, :] # (N_res, 34, 3)
                    if hasattr(batch, "residue_type_true"):
                        aatype = batch.residue_type_true
                    else:
                        aatype = batch.residue_type
                    #aatype = batch.residue_type_true # (N_res)
                    sc_coords_padded = []
                    aatype_padded = []
                    assert sc_coords.shape[0] == aatype.shape[0], f"sc_coords.shape[0] {sc_coords.shape[0]} != aatype.shape[0] {aatype.shape[0]}"

                    start = 0
                    for length in lengths:
                        seq_len = length.item()
                        sc_coords_padded.append(sc_coords[start: start + seq_len])
                        aatype_padded.append(aatype[start: start + seq_len])
                        start += seq_len
                    
                    sc_coords = torch.nn.utils.rnn.pad_sequence(sc_coords_padded, batch_first=True, padding_value=PAD_VALUE)
                    aatype = torch.nn.utils.rnn.pad_sequence(aatype_padded, batch_first=True, padding_value=0)
                    labels["sidechain_coords"] = sc_coords
                    labels["residue_type_batched"] = aatype
                    #logger.info(f"true sidechain_coords has the shape {sc_coords.shape}")

                    # combine backbone and sidechain coords
                    all_coords = torch.cat([backbone_coords, sc_coords], dim=2) 
                    if hasattr(batch, "coords_true") and not hasattr(batch, "atom37_type_true"):
                        # remove side chain coords if backbone only
                        all_coords[:, 4:, :] = PAD_VALUE
                    valid_atom_mask = all_coords[:, :, :, 0] != PAD_VALUE
                    labels["atom_mask"] = valid_atom_mask
                    #logger.info(f"true atom_mask has the shape {valid_atom_mask.shape}")

                    if hasattr(batch, "atom37_type_true"):
                        labels["atom37_type"] = batch.atom37_type_true
                    labels["atom_type"] = batch.atom37_type     
        
        if hasattr(batch, "batch_true"):
            labels["batch_idx"] = batch.batch_true
            labels["edge_index"] = batch.edge_index_true
            labels["residue_index"] = batch.residue_index_true         
        else:
            labels["batch_idx"] = batch.batch
            labels["edge_index"] = batch.edge_index
            labels["residue_index"] = batch.residue_index
            

        return labels
    
    def _preload_energy_batch(self, 
                              pdb_ids, 
                              energy_dir, 
                              device, 
                              slice_idxs = None):
        energy_dict = {}
        i = 0
        for pdb_id in pdb_ids:
            path = os.path.join(energy_dir, f"{pdb_id}.pt")
            if os.path.exists(path):
                energy_dict[pdb_id] = torch.load(path, weights_only=False, map_location=device)[:, :, 2:]
                if slice_idxs is not None:
                    slice_idx = slice_idxs[i]
                    energy_dict[pdb_id] = energy_dict[pdb_id][slice_idx[0]:slice_idx[1]]
            i +=1
        return energy_dict
    
    def get_loss(self, name: str) -> nn.Module:
        """Get a loss function by name.
        
        Args:
            name: Name of the loss function
            
        Returns:
            Loss function instance
        """
        if name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif name == "bce":
            return nn.BCEWithLogitsLoss()
        elif name == "nll_loss":
            return F.nll_loss
        elif name == "mse_loss":
            return F.mse_loss
        elif name == "l1_loss":
            return F.l1_loss
        elif name == "backbone_distance_loss":
            #return backbone_distance_loss
            if hasattr(self.config.task, "loss_clamp") and "backbone_distance" in self.config.task.loss_clamp:
                self.dist_clamp= self.config.task.loss_clamp.backbone_distance
            else:
                self.dist_clamp= 25.0
            return backbone_distance_loss
        elif name == "backbone_pairwise_direction_loss":
            return backbone_pairwise_direction_loss
        elif name == "backbone_pairwise_distance_classification_loss":
            return backbone_pairwise_distance_classification_loss
        elif name == "contact_map_loss":
            return contact_map_loss
        elif name == "chainbreak_loss":
            return chainbreak_loss
        elif name == "smooth_lddt_loss":
            #return SmoothLDDTLoss(cutoff=15.0)
            if hasattr(self.config.task, "loss_clamp") and "lddt" in self.config.task.loss_clamp:
                #return SmoothLDDTLoss(cutoff=10.0, schedule=[0.5, 1.0, 2.0, 4.0], clamp=self.config.task.loss_clamp.get("lddt"))
                logger.info(f"CLAMPING LDDT at {self.config.task.loss_clamp.get('lddt')} A")
                return SmoothLDDTLoss(cutoff=self.config.task.loss_clamp.get("lddt"), schedule=[0.25, 0.5, 1.0, 2.0])
            else:
                return SmoothLDDTLoss(cutoff=15.0)
        elif name == "contact_mse_loss":
            return contact_mse_loss
        elif name == "masked_mse_loss":
            return masked_mse_loss
        elif name == "interaction_mse_loss":
            return interaction_mse_loss
        elif name == "energy_mse_loss":
            return energy_mse_loss
        elif name == "weighted_energy_mse_loss":
            return weighted_energy_mse_loss
        elif name == "extreme_ca_ca_distance_violations":
            return extreme_ca_ca_distance_violations
        elif name == "fape_loss":
            #return fape_loss
            if hasattr(self.config.task, "loss_clamp") and "fape" in self.config.task.loss_clamp:
                logger.info(f"CLAMPING FAPE at {self.config.task.loss_clamp.get('fape')} A")
                self.fape_clamp = self.config.task.loss_clamp.get("fape")    
            else:
                self.fape_clamp= 10.0
            return fape_loss
        elif name == "torsion_angle_loss":
            return torsion_angle_loss
        elif name == "torsion_l2_loss":
            return torsion_l2_loss
 
        elif name == "smooth_lddt_large_loss":
            return SmoothLDDTLoss(cutoff=25.0, schedule=[2.0, 4.0, 8.0, 15.0])
        elif name == "backbone_distance_loss_l2":
            return backbone_distance_loss_l2       
        elif name == "backbone_distance_loss_l2mean":
            if hasattr(self.config.task, "loss_clamp") and "dist" in self.config.task.loss_clamp:
                logger.info(f"CLAMPING L2DIST at {self.config.task.loss_clamp.get('dist')} A")
                self.dist_clamp = self.config.task.loss_clamp.get("dist")
            else:
                self.dist_clamp= [10.0, 5.0, 3.0, 1.5]

            return backbone_distance_loss_l2mean
        elif name == "smooth_lddt_loss_scheduled" or name == "smooth_lddt_loss_scheduled_cb":
            cb = True if "cb" in name else False
            return SmoothLDDTLoss(scheduled=True, cb=cb)
        elif name == "contact_consistency_loss":
            return contact_consistency_loss
        elif name == "backbone_pairwisel2_cb_loss" or name == "allatom_pairwisel2_loss":
            if hasattr(self.config.task, "loss_clamp") and "dist" in self.config.task.loss_clamp:
                logger.info(f"CLAMPING L2DIST at {self.config.task.loss_clamp.get('dist')} A")
                self.dist_clamp = self.config.task.loss_clamp.get("dist")
            else:
                self.dist_clamp= [10.0, 5.0, 3.0, 1.5]
            return mask_l2pairwise_distance_loss
        elif name == "allatom_FAPE":
            return allatom_FAPE
        else:
            raise ValueError(f"Unknown loss function: {name}")
        

    def configure_losses(
        self, loss_dict: Dict[str, str]
    ) -> Dict[str, Callable]:
        """
        Configures losses from config. Returns a dictionary of losses mapping
        each output name to its respective loss function.

        :param loss_dict: Config dictionary of losses indexed by output name
        :type loss_dict: Dict[str, str]
        :return: Dictionary of losses indexed by output name
        :rtype: Dict[str, Callable]
        """

        # return a dictionary of output name : list of losses callable
        loss_map = {}
        for k, v in loss_dict.items():
            if isinstance(v, str):
                loss_map[k] = [self.get_loss(v)]   
            else:
                logger.info(f"Multiple losses detected for {k}")
                loss_map[k] = [self.get_loss(loss_name) for loss_name in v]
        return loss_map
    
    def compute_loss(
        self,
        y_hat: ModelOutput, 
        y: Dict
    ) -> Dict[str, torch.Tensor]:


        device = y_hat["backbone_coords"].device
        B, N, _, _ = y_hat["backbone_coords"].shape
        # turn backbone_coords from [batch_size, seq_len, 3, 3] to [valid_res * 3, 3]
        batch_indices = torch.arange(B, device=device).view(B,1,1).expand(B,N,3)

        # get valid residue mask
        valid_residue_mask = y["atom_mask"][:, :, 0:3].all(dim=-1)
        valid_batch_indices, valid_residue_indices = torch.nonzero(valid_residue_mask, as_tuple=True)

        true_backbone_coords = y["backbone_coords"][valid_batch_indices, valid_residue_indices]
        pred_backbone_coords = y_hat["backbone_coords"][valid_batch_indices, valid_residue_indices]

        batch_indices = batch_indices[valid_batch_indices, valid_residue_indices]
        # flatten the coords to [N_atoms, 3]
        true_backbone_coords_flat = true_backbone_coords.view(-1, 3)
        pred_backbone_coords_flat = pred_backbone_coords.view(-1, 3)
        batch_indices_flat = batch_indices.view(-1)
        # compute the loss
        assert true_backbone_coords_flat.shape == pred_backbone_coords_flat.shape
        assert batch_indices.shape[0] == true_backbone_coords.shape[0], f"batch_indices.shape {batch_indices.shape} != true_backbone_coords.shape[0] {true_backbone_coords.shape[0]}"

        
        #loss = {} # k: v(y_hat[k], y[k]) for k, v in self.losses.items()
        loss = {k: torch.tensor(0.0, device=pred_backbone_coords.device, requires_grad=True) for k, _ in self.losses.items()}
        for k, vs in self.losses.items():
            for v in vs:
                # if backbone coords and MSE loss, compute loss for only valid residues
                if k == "backbone_coords":
                    #loss[k] = torch.tensor(0.0, device=pred_backbone_coords.device)
                    coefficients = self.config.task.get("backbone_loss_coefficient")
                    if hasattr(self.config.task, "backbone_atoms"):
                        num_atoms = self.config.task.get("backbone_atoms")
                    else:
                        num_atoms = 3
                    assert num_atoms in [3, 5], f"Number of backbone atoms {num_atoms} not supported"
                    bb_true_coords = torch.cat([y["backbone_coords"], y["sidechain_coords"]], dim=2)[:, :, :num_atoms]
                    bb_pred_coords = torch.cat([y_hat["backbone_coords"], y_hat["sidechain_coords"]], dim=2)[:, :, :num_atoms]

                    if v == backbone_distance_loss:
                        c = coefficients.get("backbone_distance_loss")
                        loss_before_scale = v(pred_backbone_coords_flat, true_backbone_coords_flat, batch_indices_flat, B = B, clamp=self.config.task.dist_loss_clamp)
                        loss["bb_dist"] = loss_before_scale    
                        scaled_loss = c * loss_before_scale #v(pred_backbone_coords, true_backbone_coords, batch_indices, B = B)
                        loss["bb_dist_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_distance_loss is {scaled_loss}")
                        loss[k] += scaled_loss
                    elif v == backbone_pairwise_direction_loss:
                        c = coefficients.get("backbone_pairwise_direction_loss")
                        loss_before_scale =  v(gt_bb_coords = y["backbone_coords"],
                                    pred_bb_coords = y_hat["backbone_coords"], 
                                    seq_mask = valid_residue_mask)
                        loss["bb_dir"] = loss_before_scale                     
                        scaled_loss = c * loss_before_scale
                        loss["bb_dir_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_pairwise_direction_loss is {scaled_loss}")
                        loss[k] += scaled_loss
                    elif v == backbone_pairwise_distance_classification_loss:
                        #assert y_hat["dist_pair_logits"].shape
                        c = coefficients.get("backbone_pairwise_distance_classification_loss")
                        
                        loss_before_scale =  v(gt_bb_coords = y["backbone_coords"],
                                    binned_dist_logits= y_hat["dist_pair_logits"],
                                    seq_mask = valid_residue_mask)
                        loss["bb_bin_dist"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["bb_bin_dist_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_pairwise_direction_loss is {scaled_loss}")
                        loss[k] += scaled_loss
                    elif v == chainbreak_loss:
                        #assert y_hat["dist_pair_logits"].shape
                        c = coefficients.get("chainbreak_loss")

                        loss_before_scale =  v(gt_bb_coords = y["backbone_coords"],
                                    pred_chainbreak_logits= y_hat["chainbreak_logits"],
                                    seq_mask = valid_residue_mask)
                        loss["chainbreak"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["chainbreak_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_pairwise_direction_loss is {scaled_loss}")
                        loss[k] += scaled_loss
                    elif v == contact_mse_loss:
                        #assert y_hat["dist_pair_logits"].shape
                        c = coefficients.get("contact_mse_loss")

                        loss_before_scale =  v(gt_bb_coords = y["backbone_coords"],
                                               pred_bb_coords = y_hat["backbone_coords"],
                                               seq_mask = valid_residue_mask)
                        loss["contact_mse"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["contact_mse_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_pairwise_direction_loss is {scaled_loss}")
                        loss[k] += scaled_loss
                    elif v == F.mse_loss:
                        c = coefficients.get("mse_loss")
                        loss_before_scale =  v(pred_backbone_coords_flat, true_backbone_coords_flat)
                        loss["mse"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["mse_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_pairwise_direction_loss is {scaled_loss}")
                        loss[k] += scaled_loss
                    elif isinstance(v, SmoothLDDTLoss) and v.scheduled == False:
                        c = coefficients.get("smooth_lddt_loss")
                        # flatten pred and true coords
                        pred_bb_flat = y_hat["backbone_coords"].reshape(B, N*3, 3)
                        true_bb_flat = y["backbone_coords"].reshape(B, N*3, 3)
                        bb_coords_mask = valid_residue_mask.repeat_interleave(3, dim=1)
                        assert true_bb_flat.shape[1] == bb_coords_mask.shape[1]
                        loss_before_scale =  v(pred_coords = pred_bb_flat, true_coords = true_bb_flat, coords_mask=bb_coords_mask)
                        loss["smooth_lddt_loss"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["smooth_lddt_loss_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_pairwise_direction_loss is {scaled_loss}")
                        loss[k] += scaled_loss
                    elif v == masked_mse_loss:
                        c = coefficients.get("masked_mse_loss")

                        loss_before_scale =  v(gt_bb_coords = bb_true_coords,
                                               pred_bb_coords = bb_pred_coords,
                                               seq_mask = valid_residue_mask,
                                               atom_mask = y["atom_mask"][:, :, 0:num_atoms])
                        loss["masked_mse"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["masked_mse_scaled"] = scaled_loss
                        loss[k] += scaled_loss           

                    elif v == interaction_mse_loss:
                        c = coefficients.get("interaction_mse_loss")

                        loss_before_scale =  v(gt_bb_coords = bb_true_coords,
                                               pred_bb_coords = bb_pred_coords,
                                               seq_mask = valid_residue_mask)
                        loss["interaction_mse"] = loss_before_scale
                        scale_loss = c * loss_before_scale
                        loss["interaction_mse_scaled"] = scale_loss
                        loss[k] += scale_loss
                    
                    elif v == contact_map_loss:
                        #assert y_hat["dist_pair_logits"].shape
                        c = coefficients.get("contact_map_loss")

                        loss_before_scale =  v(gt_bb_coords = bb_true_coords,
                                    pred_contact_logits= bb_pred_coords,
                                    seq_mask = valid_residue_mask)
                        loss["contact"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["contact_scaled"] = scaled_loss
                        loss[k] += scaled_loss
                    elif v == extreme_ca_ca_distance_violations:
                        c = coefficients.get("extreme_ca_ca_distance_violations")
                        mean_neighbor_violations, clash_fraction = v(gt_bb_coords = bb_true_coords,
                                                                     pred_bb_coords = bb_pred_coords,
                                                                     seq_mask = valid_residue_mask)
                        loss["ca_neighbor_violations"] = mean_neighbor_violations
                        loss["ca_clash_fraction"] = clash_fraction
                        loss["ca_neighbor_violations_scaled"] = c * mean_neighbor_violations
                        loss["ca_clash_fraction_scaled"] = c * clash_fraction
                        loss[k] += loss["ca_neighbor_violations_scaled"] + loss["ca_clash_fraction_scaled"]
                       
                    elif v == fape_loss:
                        c = coefficients.get("backbone_fape_loss")
                        if num_atoms == 3:
                            xyz_pred = fill_in_cbeta_for_atom3_all(bb_pred_coords)
                            xyz_true = fill_in_cbeta_for_atom3_all(bb_true_coords)
                            num_atoms_total = 4
                        else:
                            xyz_pred = bb_pred_coords
                            xyz_true = bb_true_coords
                            num_atoms_total = 5
                        loss_before_scale = v(xyz_pred = xyz_pred,
                                               xyz_true = xyz_true,
                                               atom_mask = y["atom_mask"][:, :, 0:num_atoms_total], 
                                               res_mask = valid_residue_mask,
                                               d_clamp = self.fape_clamp)
                        loss["backbone_fape_loss"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["backbone_fape_loss_scaled"] = scaled_loss
                        #loss[k] += scaled_loss 
                    
                    elif v == backbone_distance_loss_l2:
                        true_backbone_coords_flattened = true_backbone_coords.view(-1, 3)
                        pred_backbone_coords_flattened = pred_backbone_coords.view(-1, 3)
                        batch_indices = batch_indices.view(-1)
                        c = coefficients.get("backbone_distance_loss_l2")
                        loss_before_scale = v(pred = pred_backbone_coords_flattened,
                                               true = true_backbone_coords_flattened,
                                               batch_indices = batch_indices,
                                               B = B,
                                               epoch = self.current_epoch)
                        loss["backbone_distance_loss_l2"] = loss_before_scale
                        #logger.info(f"Before scaled l2 loss {loss_before_scale}")
                        scaled_loss = c * loss_before_scale
                        loss["backbone_distance_loss_l2_scaled"] = scaled_loss
                        loss[k] += scaled_loss

                    elif v == backbone_distance_loss_l2mean:
                        true_backbone_coords_flattened = true_backbone_coords.view(-1, 3)
                        pred_backbone_coords_flattened = pred_backbone_coords.view(-1, 3)
                        batch_indices = batch_indices.view(-1)
                        c = coefficients.get("backbone_distance_loss_l2mean")
                        loss_before_scale = v(pred = pred_backbone_coords_flattened,
                                               true = true_backbone_coords_flattened,
                                               batch_indices = batch_indices,
                                               B = B,
                                               epoch = self.current_epoch,
                                               schedule = self.dist_clamp,)
                        loss["backbone_distance_loss_l2mean"] = loss_before_scale
                        #logger.info(f"Before scaled l2 loss {loss_before_scale}")
                        scaled_loss = c * loss_before_scale
                        loss["backbone_distance_loss_l2mean_scaled"] = scaled_loss
                        loss[k] += scaled_loss
                   
                    elif isinstance(v, SmoothLDDTLoss) and v.scheduled:
                        c = coefficients.get("smooth_lddt_loss_scheduled")
                        if v.cb and num_atoms == 3:
                            # add Cb to the pred and true
                            pred_bb = fill_in_cbeta_for_atom3_all(bb_pred_coords)
                            true_bb = fill_in_cbeta_for_atom3_all(bb_true_coords)
                            pred_bb_flat = pred_bb.reshape(B, N*4, 3)
                            true_bb_flat = true_bb.reshape(B, N*4, 3)
                            bb_coords_mask = valid_residue_mask.repeat_interleave(4, dim=1)
                            #logger.info("Added cb for LDDT")
                        else:
                            # flatten pred and true coords
                            pred_bb_flat =bb_pred_coords.reshape(B, N*num_atoms, 3)
                            true_bb_flat = bb_true_coords.reshape(B, N*num_atoms, 3)
                            bb_coords_mask = valid_residue_mask.repeat_interleave(num_atoms, dim=1)
                        assert true_bb_flat.shape[1] == bb_coords_mask.shape[1]
                        loss_before_scale =  v(pred_coords = pred_bb_flat, true_coords = true_bb_flat, coords_mask=bb_coords_mask, epoch = self.current_epoch)
                        loss["smooth_lddt_loss_scheduled"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["smooth_lddt_loss_scheduled_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_pairwise_direction_loss is {scaled_loss}")
                        #loss[k] += scaled_loss
                    elif v == contact_consistency_loss:
                        c = coefficients.get("contact_consistency_loss")
                        loss_before_scale = v(pred_coords = bb_pred_coords,
                                               pred_contact_logits = bb_true_coords,
                                               seq_mask = valid_residue_mask)
                        loss["contact_consistency_loss"] = loss_before_scale
                        scale_loss = c * loss_before_scale
                        loss["contact_consistency_loss_scaled"] = scale_loss
                        loss[k] += scale_loss
                    elif v == mask_l2pairwise_distance_loss:
                        c = coefficients.get("backbone_pairwisel2_cb_loss")
                        loss_before_scale = v(pred = bb_pred_coords,
                                               true = bb_true_coords,
                                               seq_mask = valid_residue_mask,
                                               epoch = self.current_epoch,
                                               schedule = self.dist_clamp,)
                        loss["backbone_pairwisel2_cb_loss"] = loss_before_scale
                        #logger.info(f"Before scaled l2 loss {loss_before_scale}")
                        scaled_loss = c * loss_before_scale
                        loss["backbone_pairwisel2_cb_loss_scaled"] = scaled_loss
                        loss[k] += scaled_loss 
                    else:
                        raise NotImplementedError(f"Loss {v} not implemented for backbone_coords")

                
                elif k == "allatom_coords":
                    coefficients = self.config.task.get("allatom_loss_coefficient")
                    # concatenate backbone and sidechain coords
                    allatom_true_coords = torch.cat([y["backbone_coords"], y["sidechain_coords"]], dim=2)
                    allatom_pred_coords = torch.cat([y_hat["backbone_coords"], y_hat["sidechain_coords"]], dim=2)

                    if v == masked_mse_loss:
                        c = coefficients.get("allatom_masked_mse_loss")
                        loss_before_scale =  v(gt_bb_coords = allatom_true_coords,
                                               pred_bb_coords = allatom_pred_coords,
                                               seq_mask = valid_residue_mask,
                                               atom_mask = y["atom_mask"])
                        loss["allatom_mse"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["allatom_mse_scaled"] = scaled_loss
                        loss[k] += scaled_loss
                    
                    elif isinstance(v, SmoothLDDTLoss) and v.schedule != [2.0, 4.0, 8.0, 15.0]:
                        c = coefficients.get("allatom_lddt_loss")
                        # flatten pred and true coords
                        pred_flat = allatom_pred_coords.reshape(B, N*37, 3)
                        true_flat = allatom_true_coords.reshape(B, N*37, 3)
                        # The coords mask need to filter out only atom that exists in the allatom_true_coords
                        coords_mask = y["atom_mask"].reshape(B, N*37)
                        assert coords_mask.shape[0] == pred_flat.shape[0], f"Coords mask shape {coords_mask.shape} does not match pred_flat shape {pred_flat.shape}"
                        assert coords_mask.shape[1] == pred_flat.shape[1], f"Coords mask shape {coords_mask.shape} does not match pred_flat shape {pred_flat.shape}"
        

                        loss_before_scale =  v(pred_coords = pred_flat, 
                                               true_coords = true_flat, 
                                               coords_mask=coords_mask)
                        loss["allatom_lddt_loss"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["allatom_lddt_loss_scaled"] = scaled_loss
                        #logger.info(f"Scaled loss for backbone_pairwise_direction_loss is {scaled_loss}")
                        #loss[k] += scaled_loss
                    elif isinstance(v, SmoothLDDTLoss) and v.schedule == [2.0, 4.0, 8.0, 15.0]:
                        c = coefficients.get("allatom_lddt_large_loss")
                        # flatten pred and true coords
                        pred_flat = allatom_pred_coords.reshape(B, N*37, 3)
                        true_flat = allatom_true_coords.reshape(B, N*37, 3)
                        coords_mask = y["atom_mask"].reshape(B, N*37)
                        assert coords_mask.shape[0] == pred_flat.shape[0], f"Coords mask shape {coords_mask.shape} does not match pred_flat shape {pred_flat.shape}"
                        assert coords_mask.shape[1] == pred_flat.shape[1], f"Coords mask shape {coords_mask.shape} does not match pred_flat shape {pred_flat.shape}"
        

                        loss_before_scale =  v(pred_coords = pred_flat, 
                                               true_coords = true_flat, 
                                               coords_mask=coords_mask)
                        loss["allatom_lddt_large_loss"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["allatom_lddt_large_loss_scaled"] = scaled_loss
                        loss[k] += scaled_loss       
                    elif v == fape_loss:
                        c = coefficients.get("allatom_fape_loss")
                        loss_before_scale = v(xyz_pred = allatom_pred_coords,
                                               xyz_true = allatom_true_coords,
                                               atom_mask = y["atom_mask"],
                                               res_mask = valid_residue_mask,
                                               d_clamp = self.fape_clamp)
                        loss["allatom_fape_loss"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["allatom_fape_loss_scaled"] = scaled_loss
                        #loss[k] += scaled_loss


                    elif v == torsion_angle_loss or v == torsion_l2_loss:
                        c_dihedral = coefficients.get("dihedral_angle_loss")
                        c_torsion = coefficients.get("torsion_angle_loss")
                        dihedral_loss, torsion_loss = v(xyz_pred = allatom_pred_coords,
                                                        xyz_true = allatom_true_coords,
                                                        atom_mask = y["atom_mask"],
                                                        aatype = y["residue_type_batched"])
                        loss["allatom_dihedral_angle_loss"] = dihedral_loss
                        loss["allatom_dihedral_angle_loss_scaled"] = c_dihedral * dihedral_loss
                        loss["allatom_torsion_angle_loss"] = torsion_loss
                        loss["allatom_torsion_angle_loss_scaled"] = c_torsion * torsion_loss
                        loss[k] += c_dihedral * dihedral_loss + c_torsion * torsion_loss
                    elif v == mask_l2pairwise_distance_loss:
                        c = coefficients.get("allatom_pairwisel2_loss")

                        loss_before_scale = v(pred = allatom_pred_coords,
                                               true = allatom_true_coords,
                                               epoch = self.current_epoch,
                                               seq_mask = valid_residue_mask,
                                               atom_mask = y["atom_mask"],
                                               schedule = self.dist_clamp,)
                        loss["allatom_pairwisel2_loss"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["allatom_pairwisel2_loss_scaled"] = scaled_loss
                        loss[k] += scaled_loss
                    elif v == allatom_FAPE:
                        c = coefficients.get("allatom_FAPE")
                        atom_mask = y["atom_mask"]
                        flat_pred_coords = allatom_pred_coords[atom_mask]
                        flat_true_coords = allatom_true_coords[atom_mask]
                        assert flat_pred_coords.shape[0] == y["atom_type"].shape[0], f"Flat pred coords shape {flat_pred_coords.shape} does not match atom type shape {y['atom_type'].shape}"
                        frames = make_filtered_frames(
                            pos = flat_true_coords,
                            atom_type = y["atom_type"], # [N_atoms]
                            edge_index = y["edge_index"],   # [2, N_edges]
                            residue_index = y["residue_index"],   # [N_atoms]
                            residue_type = y["residue_type"]
                        )
                        loss_before_scale = v(pred_pos = flat_pred_coords,
                                               true_pos = flat_true_coords,
                                               batch_idx = y["batch_idx"],
                                               frames = frames)
                        loss["allatom_FAPE"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["allatom_FAPE_scaled"] = scaled_loss
                        #loss[k] += scaled_loss
                    else:
                        raise NotImplementedError(f"Other finetune loss {v} not implemented!")
                
                elif k == "energy":
                    coefficients = self.config.task.get("energy_loss_coefficient")
                    if v == energy_mse_loss:
                        c = coefficients.get("energy_mse_loss")
                        loss_before_scale = v(pred_eng = y_hat["energy"], # [B, N, N, 3]
                                            gt_eng = y["energy"], # {pdb_id: [N, N, 3]}
                                            pdb_ids = y["pdb_id"], # batch["id"]
                                            energy_dir = self.config.task.energy_dir,
                                            seq_mask = valid_residue_mask)
                        loss["energy_mse"] = loss_before_scale
                        scaled_loss = c * loss_before_scale
                        loss["energy_mse_scaled"] = scaled_loss
                        #loss[k] += scaled_loss
                    elif v == weighted_energy_mse_loss:
                        c = coefficients.get("weighted_energy_mse_loss")
                        loss_before_scale, zero_percentage = v(pred_eng = y_hat["energy"], # [B, N, N, 3]
                                            gt_eng = y["energy"], # {pdb_id: [N, N, 3]}
                                            pdb_ids = y["pdb_id"], # batch["id"]
                                            energy_dir = self.config.task.energy_dir,
                                            seq_mask = valid_residue_mask)
                        loss["weighted_energy_mse"] = loss_before_scale
                        loss["correct_no_energy_percentage"] = zero_percentage
                        scaled_loss = c * loss_before_scale
                        loss["weighted_energy_mse_scaled"] = scaled_loss
                        #loss[k] += scaled_loss
                    else:
                        raise NotImplementedError(f"Other loss {v} not implemented! for energy")
                    
                else:
                    #logger.info(f"Computing loss for {k} using loss {v}, with y_hat[k].shape {y_hat[k].shape} and y[k].shape {y[k].shape}")
                    loss[k] = v(y_hat[k], y[k])
           
        if self.quantizer is not None:
            coefficients = self.config.task.get("vq_loss_coefficient")
            c = coefficients.get("vq_loss")
            loss["vq_loss"] = y_hat["vq_loss"]
            loss["vq_loss_scaled"] = c * loss["vq_loss"] 
        
        # Scale loss terms by coefficient
        if self.config.task.get("aux_loss_coefficient") is not None:
            for (
                output,
                coefficient,
            ) in self.config.task.aux_loss_coefficient.items():
                #logger.info(f"Scaling {output} value {loss[output]} with {coefficient}")
                loss[output] = coefficient * loss[output]

        #loss["total"] = sum(loss.values())
        #loss["total"] = sum(torch.nan_to_num(v, nan=0.0) for v in loss.values())
        """
        logger.info(f"{loss}")
        for name, loss_term in loss.items():
            logger.info(f"Loss term '{name}' requires_grad: {loss_term.requires_grad}")
        """
        loss["total"] = sum(
            torch.nan_to_num(loss, nan=0.0)
            for name, loss in loss.items()
            if name.endswith("scaled") or "residue_type" in name
        )
        return loss


    def featurise(
        self, batch: Batch
        ) -> Batch:
        """Applies the featuriser (``self.featuriser``) to a batch of data.

    
        :param batch: Batch of data
        :type batch: Batch
        :return: Featurised batch
        :rtype: Batch
        """
        out = self.featuriser(batch)
        if self.task_transform is not None:
            out = self.task_transform(out)
        return out




    def on_after_batch_transfer(
        self, batch: Batch, dataloader_idx: int
        ) -> Batch:
        """
        Featurise batch **after** it has been transferred to the correct device.

        :param batch: Batch of data
        :type batch: Batch
        :param dataloader_idx: Index of dataloader
        :type dataloader_idx: int
        :return: Featurised batch
        :rtype: Batch
        """
        return self.featurise(batch)
    

    def forward(self, batch: Batch  ) -> ModelOutput:
        """
        Implements the forward pass of the model.


        1. Apply the model encoder (``self.encoder``) to the batch of data.
        2. (Optionally) apply any transformations to the encoder output
        (:py:meth:`BaseModel.transform_encoder_output`)
        3. Iterate over the decoder heads (``self.decoder``) and apply each
        decoder to the relevant part of the encoder output.
        4. (Optionally) apply any post-processing to the model output.
        (:py:meth:`BaseModel.compute_output`)

        :param batch: Mini-batch of data.
        :type batch: Batch
        :return: Model output.
        :rtype: ModelOutput
        """
        #logger.info(f"Batch: {batch}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), "NaN in input key {key}" 
        if self.frozen_encoder:
            with torch.no_grad():
                output: Dict = self.encoder(batch)
        else:
            #logger.info("Start encoding")
            output: Dict = self.encoder(batch)


        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), f"NaN in encoder output key {key}"

        #logger.info(f"Output of encoder is {output}")

        if self.decoder is not None:
            for output_head in self.config.decoder.keys():

                if output_head == "allatom_decoder":
                    emb_type = "residue_embedding"
                    assert emb_type in output, f"emb_type {emb_type} not in output {output.keys()}"
                    res_embedding = output[emb_type]

                    logger.info(f"res_embedding shape: {res_embedding.shape}")

                    
                    output["vq_loss"] = None

                    residue_batch = batch.residue_batch

                    # NOISE
                    if hasattr(self.config.task, "residue_embedding_noise") and self.config.task.residue_embedding_noise != 0.0:
                        #logger.info(f"Latent norm: {res_embedding.norm(dim=-1).mean()}, Latent norm std: {res_embedding.norm(dim=-1).std()}") 
                        #logger.info(f"Latent mean: {res_embedding.mean()}, Latent std: {res_embedding.std()}")
                        # gaussian noise with mean 0 and std = config.task.residue_embedding_noise
                        noise = torch.randn_like(res_embedding) * self.config.task.residue_embedding_noise
                        res_embedding = res_embedding + noise
                        #logger.info(f"Latent norm after noise: {res_embedding.norm(dim=-1).mean()}, Latent norm std after noise: {res_embedding.norm(dim=-1).std()}") 
                        #logger.info(f"Latent mean after noise: {res_embedding.mean()}, Latent std after noise: {res_embedding.std()}")                    

                    batch_size = len(batch.id)
                    assert batch_size == self.config.dataset.datamodule.batch_size or batch_size == 1

                   
                    lengths = torch.bincount(residue_batch, minlength=batch_size)  # [batch_size]
                    # Split the embeddings into a list of [seq_len_i, d_model] based on lengths
                    seq_list = []
                    start = 0
                    for length in lengths:
                        seq_len = length.item()
                        seq_list.append(res_embedding[start : start + seq_len])  # [seq_len_i, d_model]
                        start += seq_len

                    # Pad sequences to form [batch_size, max_seq_len, d_model]
                    res_embedding_padded = torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=PAD_VALUE)
                    #logger.info(f"res_embedding_padded {res_embedding_padded.shape}")
                    max_seq_len = res_embedding_padded.size(1)

                    # Create a mask indicating which positions are valid (True) vs. padded (False)
                    mask = torch.arange(max_seq_len, device=res_embedding_padded.device).unsqueeze(0) < lengths.unsqueeze(1)

                    #logger.info(f"Decoding with allatom_decoder, res_embedding_padded shape: {res_embedding_padded.shape}, mask shape: {mask.shape}, res_idx range: {batch.residue_id.min()} to {batch.residue_id.max()}")
                    #logger.info(f"residue_batch range: {batch.residue_batch.min()} to {batch.residue_batch.max()}; residue_batch.shape: {batch.residue_batch.shape}: {batch.residue_batch}")
                    decoder_output = self.decoder[output_head](
                        res_embedding_padded, # [batch_size, max_seq_len, d_model]
                        mask,  # mask: [batch_size, max_seq_len]
                        batch = batch.residue_batch,
                        res_idx = batch.residue_id,
                    )
                    #logger.info(f"Output of allatom_decoder is {decoder_output}")      
                    output["torsion_angles"] = decoder_output["torsion_angles"]              
                    output["residue_type"] = decoder_output["seq_pred"] #.reshape(-1, 23) # TODO best not to hardcode this
                    output["backbone_coords"] = decoder_output["backbone_coords"]
                    output["sidechain_coords"] = decoder_output["sidechain_coords"]
                    output["atom_mask"] = decoder_output["atom_mask"]
                    output["mask"] = mask
                    output["dist_pair_logits"] = decoder_output["dist_pair_logits"]
                    output["contact_logits"] = decoder_output["contact_logits"]
                    output["chainbreak_logits"] = decoder_output["chainbreak_logits"]


                    # also add energy to the output
                    output["energy"] = decoder_output["pred_eng"]
                # sequence decoding
                else:
                    raise ValueError(f"Output head {output_head} not recognized")
        #logger.info(f"Decoder output is {output}")         

        return self.compute_output(output, batch)



    def compute_output(self, output: ModelOutput, batch: Batch) -> ModelOutput:
        """
        Computes output from model output.

        - For dihedral angle prediction, this involves normalising the
        'sin'/'cos' pairs for each angle such that the have norm 1.
        - For sequence denoising, this masks the output such that we only
        supervise on the corrupted residues.

        :param output: Model output (dictionary mapping output name to the
            output tensor)
        :type: ModelOutput
        :param batch: Batch of data
        :type batch: Batch
        :return: Model output (dictionary mapping output name to the
            transformed output)
        :rtype: ModelOutput
        """
        if "dihedrals" in output.keys():
            # Normalize output so each pair of sin(ang) and cos(ang) sum to 1.
            output["dihedrals"] = F.normalize(
                output["dihedrals"].view(-1, 3, 2), dim=-1
            ).view(-1, 6)
        # If we have a mask, apply it
        """
        if hasattr(batch, "sequence_corruption_mask"):
            output["residue_type"] = output["residue_type"][
                batch.sequence_corruption_mask
            ]
        if hasattr(batch, "center_residue_mask"):
            output["residue_type"] = output["residue_type"][
                batch.center_residue_mask
            ]
        """
        if "residue_type" in output.keys():
            if hasattr(batch, "sequence_corruption_mask"):
                mask = batch.sequence_corruption_mask
            else:
                mask = torch.ones_like(output["residue_type"], dtype=torch.bool)
            
            output["residue_type"] = output["residue_type"][mask].reshape(-1, 23) # best not to hard code this

        return output

    def _do_step(
        self,
        batch: Batch,
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        return self._do_step_catch_oom(batch, batch_idx, stage)

    def _do_step_catch_oom(
        self,
        batch: Batch,
        batch_idx: int,
        stage: Literal["train", "val"],
    ) -> Optional[torch.Tensor]:
        """Performs a training/validation step
        while catching out of memory errors.
        Note that this should not be used for
        test steps for proper benchmarking.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :param stage: Stage of training (``"train"``, ``"val"``)
        :type stage: Literal["train", "val"]
        :return: Loss
        :rtype: torch.Tensor
        """
               # by default, do not skip the current batch
        skip_flag = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )  # NOTE: for skipping batches in a multi-device setting
        loss = None 
        try:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    assert not torch.isnan(v).any(), f"NaN detected in the input batch's {k}" 
            y = self.get_labels(batch)
            y_hat = self(batch)
            #logger.info(f"Computing loss for batch {batch_idx}") 
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    assert not torch.isnan(v).any(), f"NaN detected in decoder output {k} at batch {batch_idx}"
            
            if self.config.task.align_backbone: 
            # given reconstructed backbone coords [B, N, 3, 3], true backbone coords [B, N, 3, 3], predicted sidechain coords [B, N, 34, 3],
            # align the backbone coords, and then rotate the predicted sidechain coords base on the alignement. 
                #logger.info(f"reconstructed_bb {y_hat['backbone_coords'].shape}")
                #logger.info(f"true_bb {y['backbone_coords'].shape}")
                #logger.info(f"predicted_sidechain {y_hat['sidechain_coords'].shape}")
                #logger.info(f"backbone mask {y_hat['mask'].shape}")
                
                rotated_bb, rotated_sidechain  = aligned_rotation(reconstructed_bb = y_hat["backbone_coords"], 
                                                                true_bb = y["backbone_coords"], 
                                                                predicted_sidechain = y_hat["sidechain_coords"],
                                                                mask = y_hat['mask'])
                y_hat["sidechain_coords"] = rotated_sidechain
                y_hat["backbone_coords"] = rotated_bb
            #logger.info("Start computing loss")
            
            loss = self.compute_loss(y_hat, y)

            

            #loss = self.compute_loss(y_hat, y)
            assert not torch.isnan(loss["total"]), f"NaN detected in total loss, skipping batch {batch_idx}"

            log_dict = self.log_metrics(loss, y_hat, y, stage, batch=batch)

            if stage == "val":  #and batch_idx % 2 == 0:
                # convert rotated_bb to pdb
                # convert true_bb to pdb
                # save both to given directory
                pdb = batch.id[0]
                logger.info(f"Saving pdb files for {pdb} to {self.config.monitor.output_dir}")
                output_dir = self.config.monitor.output_dir
                # mkdir if not exists
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # convert first item in the batch to pdb 
                epoch = self.current_epoch
                if f"{stage}/backbone_coords/rmse" in log_dict.keys():
                    rmse = log_dict[f"{stage}/backbone_coords/rmse"]
                else:
                    rmse = 0
                if self.config.monitor.pdb_out == "first_backbone":
                    atom_tensor_pred = y_hat["backbone_coords"][0].detach().clone().cpu()
                    atom_tensor_true = y["backbone_coords"][0].detach().clone().cpu()
                elif self.config.monitor.pdb_out == "first_aa":
                    allatom_true_coords = torch.cat([y["backbone_coords"], y["sidechain_coords"]], dim=2)
                    allatom_pred_coords = torch.cat([y_hat["backbone_coords"], y_hat["sidechain_coords"]], dim=2)
                    # use y_hat["atom_mask"] to mask out atoms with FILL value in allatom_pred_coords
                    allatom_pred_coords[~y_hat["atom_mask"].bool()] = 1e-5
                    atom_tensor_pred = allatom_pred_coords[0].detach().clone().cpu()
                    atom_tensor_true = allatom_true_coords[0].detach().clone().cpu()
                    # find argmax of y_hat["residue_type"] and compare with y["residue_type"]
                    
                    residue_types = torch.argmax(y_hat["residue_type"], dim=-1)
                    # print out an accuracy of residue_types vs y["residue_type"]
                    accuracy = (residue_types == y["residue_type"]).float().mean()
                    #logger.info(y["residue_type"].tolist())
                    logger.info(f"Accuracy of residue_types vs y['residue_type'] is {accuracy:.3f}")
                    """ 
                    from allatom.util.constants import VALID_ATOM37_MASK
                    B, N = y_hat["mask"].shape
                    true_atom_mask = torch.zeros((B, N, 37), device=y_hat["mask"].device)
                    true_res = torch.clamp(y["residue_type"], max=19)
                    logger.info(f"TRUE RES: {true_res}")
                    valid_mask37  = VALID_ATOM37_MASK.to(device=y_hat["mask"].device)
                    true_atom_mask[y_hat["mask"]] = valid_mask37[true_res]
                    # check if true_atom_mask is the same as y["atom_mask"]
                    assert (true_atom_mask.bool() == y["atom_mask"].bool()).all(), "atom_mask does not match"
                    """
                path_pred = os.path.join(output_dir, f"pred_e{epoch}_{pdb}_{rmse:.1f}.pdb")
                path_true = os.path.join(output_dir, f"true_e{epoch}_{pdb}_{rmse:.1f}.pdb")
                to_pdb(atom_tensor_pred, path_pred)
                to_pdb(atom_tensor_true, path_true)

        except RuntimeError as e:
                        
                        logger.warning(
                            f"[LOSS FAIL] RuntimeError during loss on {stage} batch {batch_idx}: {e}. "
                            f"Skipping this batch."
                        )

                        skip_flag = torch.ones((), device=self.device, dtype=torch.bool)

                        # clean gradients
                        if not torch_dist.is_initialized() and self.training:
                            for p in self.trainer.model.parameters():
                                if p.grad is not None:
                                    del p.grad
                        # free memory to avoid cascading failures
                        torch.cuda.empty_cache()
                        return None

        except AssertionError as e:
            skip_flag = torch.ones((), device=self.device, dtype=torch.bool)
            logger.warning(
                f"Ran into {str(e)} in the forward pass. \n Skipping current {stage} batch with index {batch_idx}."
            )
            if not torch_dist.is_initialized():
                if self.training:
                    for p in self.trainer.model.parameters():
                        if p.grad is not None:
                            del p.grad  # Free some memory
                    return None

        except Exception as e:
            skip_flag = torch.ones((), device=self.device, dtype=torch.bool)

            if "out of memory" in str(e) or "Runtime" in str(e): # TODO check this
                logger.warning(
                    f"Ran out of memory in the forward pass. Skipping current {stage} batch with index {batch_idx}."
                )
                #logger.warning(
                #    f"Ran into {str(e)} in the forward pass. \n Skipping current {stage} batch with index {batch_idx}."
                #)
                if not torch_dist.is_initialized():
                    # NOTE: for skipping batches in a single-device setting
                    if self.training:
                        for p in self.trainer.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                    return None
            else:
                if not torch_dist.is_initialized():
                    raise e
                
        # NOTE: for skipping batches in a multi-device setting
        # credit: https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1553404417
        if torch_dist.is_initialized():
            # if any rank skips a batch, then all other ranks need to skip
            # their batches as well so DDP can properly keep all ranks synced
            world_size = torch_dist.get_world_size()
            torch_dist.barrier()
            result = [torch.zeros_like(skip_flag) for _ in range(world_size)]
            torch_dist.all_gather(result, skip_flag)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            if any_skipped:
                if self.training:
                    for p in self.trainer.model.parameters():
                        if p.grad is not None:
                            del p.grad
                logger.warning(
                    f"Failed to perform the forward pass for at least one rank. Skipping {stage} batches for all ranks."
                )
                return None

        return loss["total"]


    def training_step(
        self, batch: Batch, batch_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Perform training step.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :return: Loss
        :rtype: Optional[torch.Tensor]
        """
        if self.inference_only:
            with torch.no_grad():
                return self._do_step_catch_oom(batch, batch_idx, "train")
        else:
            return self._do_step_catch_oom(batch, batch_idx, "train")

    @torch.no_grad()
    def validation_step(
        self, batch: Batch, batch_idx: int
    ) -> Optional[torch.Tensor]:
        """
        Perform validation step.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :return: Loss
        :rtype: Optional[torch.Tensor]
        """
        #return self._do_step_catch_oom(batch, batch_idx, "val")
        try:
            loss = self._do_step_catch_oom(batch, batch_idx, "val")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"OOM error at validation batch {batch_idx}, attempting to recover...")
                torch.cuda.empty_cache()  # Clear unused memory
                return None
            else:
                raise e  # Raise other errors
    
        # Explicitly free memory
        del batch
        torch.cuda.empty_cache()

        return loss

    def test_step(
        self, batch: Batch, batch_idx: int
    ) -> torch.Tensor:
        """Perform test step.

        1. Obtains labels from :py:meth:`get_labels`
        2. Computes model output :py:meth:`forward`
        3. Computes loss :py:meth:`compute_loss`
        4. Logs metrics :py:meth:`log_metrics`

        Returns the total loss.

        :param batch: Mini-batch of data.
        :type batch: Batch
        :param batch_idx: Index of batch.
        :type batch_idx: int
        :return: Loss
        :rtype: torch.Tensor
        """
        return self._do_step(batch, batch_idx, "test")

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Dict[str, Any]):
        """Overrides Lightning's `backward` hook to add an out-of-memory (OOM) check.

        :param loss: The loss value to backpropagate.
        :param args: Additional positional arguments to pass to `torch.Tensor.backward`.
        :param kwargs: Additional keyword arguments to pass to `torch.Tensor.backward`.
        """
        # by default, do not skip the current batch
        #logger.info(f"Starting Backward")
        skip_flag = torch.zeros(
            (), device=self.device, dtype=torch.bool
        )  # NOTE: for skipping batches in a multi-device setting

        try:
            loss.backward(*args, **kwargs)
            for name, param in self.encoder.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"Backward: NaN detected in gradient of {name}"
                    #logger.warning(f"Backward: NaN detected in gradient of {name}")
            for name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"Backward: NaN detected in gradient of {name}"
                    #logger.warning(f"Backward: NaN detected in gradient of {name}")
            global_grad_norm_sq = 0.0
            for name, param in self.decoder.named_parameters():
                if param.grad is not None:
                    global_grad_norm_sq += param.grad.data.norm() ** 2
            global_grad_norm = global_grad_norm_sq ** 0.5
            grad_threshold = 100.0  # Adjust this threshold as needed
            if global_grad_norm > grad_threshold:
                logger.warning(f"Exploding gradients detected: global gradient norm = {global_grad_norm:.3f}")

            # Optionally, check for exploding parameter values too.
            global_param_norm_sq = 0.0
            for name, param in self.decoder.named_parameters():
                global_param_norm_sq += param.data.norm() ** 2
            global_param_norm = global_param_norm_sq ** 0.5
            param_threshold = 1e3  # Example threshold; adjust as needed
            if global_param_norm > param_threshold:
                logger.warning(f"Exploding parameters detected: global parameter norm = {global_param_norm:.3f}")

            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
            if self.clip_decoder_grad:
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
        except AssertionError as e:
            skip_flag = torch.ones((), device=self.device, dtype=torch.bool)
            logger.warning(
                f"Ran into {str(e)} in the backward pass. \n Skipping current batch."
            )
            for p in self.trainer.model.parameters():
                if p.grad is not None:
                    del p.grad
            logger.warning("Finished cleaning up all gradients following the failed backward pass.")

        except Exception as e:
            skip_flag = torch.ones((), device=self.device, dtype=torch.bool)
            logger.warning(f"Failed the backward pass. Skipping it for the current rank due to: {e}")
            for p in self.trainer.model.parameters():
                if p.grad is not None:
                    del p.grad
            logger.warning("Finished cleaning up all gradients following the failed backward pass.")
            if "out of memory" not in str(e) and not torch_dist.is_initialized():
                raise e

        # NOTE: for skipping batches in a multi-device setting
        # credit: https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1553404417
        if torch_dist.is_initialized():
            # if any rank skips a batch, then all other ranks need to skip
            # their batches as well so DDP can properly keep all ranks synced
            world_size = torch_dist.get_world_size()
            torch_dist.barrier()
            result = [torch.zeros_like(skip_flag) for _ in range(world_size)]
            torch_dist.all_gather(result, skip_flag)
            any_skipped = torch.sum(torch.stack(result)).bool().item()
            if any_skipped:
                logger.warning(
                    "Skipping backward for all ranks after detecting a failed backward pass."
                )
                del loss  # delete the computation graph
                logger.warning(
                    "Finished cleaning up the computation graph following one of the rank's failed backward pass."
                )
                for p in self.trainer.model.parameters():
                    if p.grad is not None:
                        del p.grad
                logger.warning(
                    "Finished cleaning up all gradients following one of the rank's failed backward pass."
                )
                clean_up_torch_gpu_memory()
                logger.warning(
                    "Finished manually freeing up memory following one of the rank's failed backward pass."
                )

    def log_metrics(
        self, loss, y_hat: ModelOutput, y: Dict, stage: str, batch: Batch
    ):
        """
        Logs metrics to logger.

        :param loss: Dictionary of losses indexed by output name (str)
        :type loss: Dict[str, torch.Tensor]
        :param y_hat: Output of model. This should be a dictionary of outputs
            indexed by the output name (str)
        :type y_hat: ModelOutput
        :param y: Labels. This should be a dictionary of labels (torch.Tensor)
            indexed by the output name (str)
        :type y: Label
        :param stage: Stage of training (``"train"``, ``"val"``, ``"test"``)
        :type stage: str
        :param batch: Batch of data
        :type batch: Batch
        """
        # Log losses
        loss = {k: torch.nan_to_num(v, nan=0.0) for k, v in loss.items()}  # Prevent NaN
        log_dict = {f"{stage}/loss/{k}": v for k, v in loss.items()}
        
        # Log metric
        compute_metrics = True
        if stage == "train" and (self.global_step % 200 != 0):
            compute_metrics = False

        if compute_metrics:
            for m in self.metric_names:
                for output in self.config.task.output:
                    if output == "energy":
                        valid_residue_mask = y["atom_mask"][:, :, 0:3].all(dim=-1)
                        # use energy_prediction_pr_auc to compute auc of energy prediction
                        auc_score, tertiary_auc_score = energy_prediction_pr_auc(pred_eng = y_hat["energy"], # [B, N, N, 3]
                                                gt_eng = y["energy"], # {pdb_id: [N, N, 3]}
                                                pdb_ids = y["pdb_id"], # batch["id"]
                                                energy_dir = self.config.task.energy_dir,
                                                seq_mask = valid_residue_mask)
                        log_dict[f"{stage}/energy/pr_auc"] = auc_score
                        log_dict[f"{stage}/energy/tertiary_pr_auc"] = tertiary_auc_score
                        
                        continue
                    # log VQ metrics if quantizer is not None
                    
                    if hasattr(self, f"{stage}_{output}_{m}") and output != "allatom_coords":
                        try:
                            metric = getattr(self, f"{stage}_{output}_{m}")
                            pred = y_hat[output]
                            target = y[output]

                            if m == "perplexity":
                                pred = to_dense_batch(pred, batch.batch)[0]
                                target = to_dense_batch(
                                    target, batch.batch, fill_value=-100
                                )[0]
                            # This is a hack for MSE-type metrics which fail on e.g. [4,1] & [4]
                            try:
                                if m == "rmse":
                                    mask = y_hat["mask"]
                                    pred = pred[mask]
                                    target = target[mask]
                                val = metric(pred, target)
                            except RuntimeError:
                                val = metric(pred, target.unsqueeze(-1))
                            log_dict[f"{stage}/{output}/{m}"] = val

                        except (ValueError, RuntimeError) as e:
                            logger.warning(
                                f"Failed to compute metric {m} for output {output} in stage {stage}."
                            )
                            logger.warning(e)
                            continue
        #self.log_dict(log_dict, prog_bar=True)
        if stage == "train":
            self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        else:
            # For val/test, usually on_step=False, on_epoch=True
            self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)

        return log_dict

    def configure_metrics(self):
        """
        Instantiates metrics from config.

        Metrics are Torchmetrics Objects :py:class:`torchmetrics.Metric`
        (see `torchmetrics <https://torchmetrics.readthedocs.io/en/latest/>`_)

        Metrics are set as model attributes as:

        ``{stage}_{output}_{metric_name}`` (e.g. ``train_residue_type_f1_score``)
        """

        CLASSIFICATION_METRICS: Set[str] = {
            "f1_score",
            "auprc",
            "accuracy",
            "f1_max",
            "rocauc",
            "perplexity",
        }
        REGRESSION_METRICS: Set[str] = {"mse", "mae", "r2", "rmse"}
        CONTINUOUS_OUTPUTS: Set[str] = {
            "b_factor",
            "plddt",
            "pos",
            "dihedrals",
            "torsional_noise",
            "backbone_coords",
            "sidechain_coords",
        }
        CATEGORICAL_OUTPUTS: Set[str] = {"residue_type"}

        metric_names = []
        for metric_name, metric_conf in self.config.metrics.items():
            for output in self.config.task.output:
                stages = {"train", "val", "test"}
                for stage in stages:
                    metric = hydra.utils.instantiate(metric_conf)
                    if output == "residue_type":
                        if metric_name not in {"accuracy", "perplexity"}:
                            continue
                        metric.num_classes = 23 # TODO check this #23
                        metric.task = "multiclass"

                    # Skip incompatible metrics
                    if (
                        output in CONTINUOUS_OUTPUTS
                        and metric_name in CLASSIFICATION_METRICS
                    ):
                        logger.info(
                            f"Skipping classification metric {metric_name} for output {output} as output is continuous"
                        )
                        continue
                    if (
                        output in CATEGORICAL_OUTPUTS
                        and metric_name in REGRESSION_METRICS
                    ):
                        logger.info(
                            f"Skipping regression metric {metric_name} for output {output} as output is categorical"
                        )
                        continue
                    setattr(self, f"{stage}_{output}_{metric_name}", metric)
            metric_names.append(f"{metric_name}")
        setattr(self, "metric_names", metric_names)
