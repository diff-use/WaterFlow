#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a per-residue 15N chemical-shift regressor (encoder edition)
=================================================================
• Encoder cfg :  /scratch/users/yilinc5/allatomAE/allatomAE/configs/encoder/SLAE.yaml
• Dataset  cfg :  /scratch/users/yilinc5/allatomAE/allatomAE/configs/dataset/pdb.yaml
• Embeddings & shifts are read on-the-fly by the PDBDataModule.
• Checkpoint :  same .ckpt file supplies encoder weights (incl. lazy).

Run:
    python mbmrb.py --ckpt <slae_or_ae.ckpt> --epochs 20 --batch 8
"""

# ------------------------------------------------------------------ #
#  Imports                                                           #
# ------------------------------------------------------------------ #
import argparse, os, math, json, warnings
from pathlib import Path
from typing import Any, Mapping, List, Tuple, Dict

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import callbacks as plcb
from torch.utils.data import Subset
from scipy.stats import spearmanr, pearsonr as pearson
import numpy as np
from omegaconf import OmegaConf
from flash.core.optimizers import LinearWarmupCosineAnnealingLR

# ---------- SLAE specific ------------------------------------ #
from SLAE.model.encoder import ProteinEncoder
from SLAE.model.cs import ShiftNet
from SLAE.datasets.datamodule import PDBDataModule
from SLAE.features.graph_featurizer import ProteinGraphFeaturizer
from SLAE.model.decoder import AllAtomDecoder
from SLAE.model.autoencoder import initialize_weights_transformer

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ------------------------------------------------------------------ #
#  Paths & constants                                                 #
# ------------------------------------------------------------------ #
ENCODER_CFG  = Path("/path/SLAE/configs/encoder/protein_encoder.yaml")
DECODER_CFG  = Path("/path/SLAE/configs/decoder/allatom_decoder.yaml")
EMB_PKL      = Path("/path/cs/mbmrb_res.pkl")
TRAIN_JSON   = Path("/path/cs/mbmrb_stringent_shiftx_f_train.json")
VAL_JSON     = Path("/path/cs/mbmrb_stringent_shiftx_f_val.json")
TEST_JSON    = Path("/path/cs/shiftx_test_shifts_filtered_mbmrb.json")
PAD_VALUE    = 1e-5

# ------------------------------------------------------------------ #
#  Global residue-embedding dict (keys already lower-case)           #
# ------------------------------------------------------------------ #
def _load_pickle(path: Path) -> Dict[str, torch.Tensor]:
    import pickle
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {k.lower(): torch.as_tensor(v, dtype=torch.float32, device="cpu")
            for k, v in raw.items()}

EMB_DICT = _load_pickle(EMB_PKL)      # loaded once – reused everywhere

# ------------------------------------------------------------------ #
#  Shift-JSON helpers                                                #
# ------------------------------------------------------------------ #
"""
def _parse_shift_list(lst: List) -> Tuple[torch.Tensor, torch.Tensor]:
    vals, mask = [], []
    for x in lst:
        if x is None or (isinstance(x, str) and x.lower() == "none") or (isinstance(x, str) and x.lower() == "null"):
            vals.append(0.0);  mask.append(False)
        else:
            vals.append(float(x));  mask.append(True)
    return (torch.tensor(vals, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool))
"""
def _parse_shift_list(lst: List) -> Tuple[torch.Tensor, torch.Tensor]:
    vals, mask = [], []
    for x in lst:
        bad = False
        if x is None:
            bad = True
        elif isinstance(x, str):
            xs = x.strip().lower()
            if xs in {"none", "null", "nan", "inf", "+inf", "-inf"}:
                bad = True
            else:
                try:
                    x = float(x)
                except Exception:
                    bad = True
        # if it's already a number, check finiteness
        if not bad and isinstance(x, (int, float)):
            if not math.isfinite(float(x)):
                bad = True

        if bad:
            vals.append(0.0); mask.append(False)
        else:
            vals.append(float(x)); mask.append(True)

    return (torch.tensor(vals, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool))

def _json_to_dict(json_path: Path, atom: str = "N") -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    with open(json_path) as f:
        refdb = json.load(f)

    out = {}
    for ref_id, shift_lst in refdb[atom].items():
        if all(x is None or (isinstance(x, str) and x.lower() == "none") or (isinstance(x, str) and x.lower() == "null") for x in shift_lst):
            continue                                          # skip all-None rows
        pdb_id = ref_id#f"{refdb['entry_ID'][ref_id]}".lower()
        shift, mask = _parse_shift_list(shift_lst)

        # keep only if embeddings exist **and** lengths match
        if pdb_id in EMB_DICT and EMB_DICT[pdb_id].shape[0] == len(shift):
            out[pdb_id] = (shift, mask)
    return out


def build_shift_and_id_sets(atom: str = "N"):
    shift_train = _json_to_dict(TRAIN_JSON, atom=atom)
    shift_val   = _json_to_dict(VAL_JSON, atom=atom)
    shift_test  = _json_to_dict(TEST_JSON, atom=atom)
    # union used at batch-time for all splits
    shift_dict  = {**shift_train, **shift_val, **shift_test}
    # print out the min and max shift for the given atom
    # only consider where the mask is True don't consider 0.0 entries  the dict is out[pdb_id] = (shift, mask) so we need to check the mask
    all_shifts = [v[0] for v in shift_dict.values()]
    all_shifts = np.concatenate(all_shifts)
    all_shifts = all_shifts[all_shifts != 0.0]



    print(f"Min shift for {atom}: {min(all_shifts)}")
    print(f"Max shift for {atom}: {max(all_shifts)}")
    print(f"Mean shift for {atom}: {np.mean(all_shifts)}")
    return shift_dict, set(shift_train.keys()), set(shift_val.keys()), set(shift_test.keys())

# ------------------------------------------------------------------ #
#  Misc utilities                                                    #
# ------------------------------------------------------------------ #
def _to_plain_dict(cfg_node) -> dict:
    """Flatten an OmegaConf node to a plain Python dict (recursive)."""
    return OmegaConf.to_container(cfg_node, resolve=True, structured_config_mode=False)


def init_weights_slae(m: nn.Module):
    """
    • Linear / Conv layers  →  Kaiming-Uniform (fan-in)
    • LayerNorm / BatchNorm →  weight=1, bias=0
    • Embedding             →  Normal(0, 1/√d)
    """
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1.)
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0., std=1./math.sqrt(m.embedding_dim))
    return m

# ------------------------------------------------------------------ #
#  Encoder / decoder builders                                        #
# ------------------------------------------------------------------ #
def build_encoder_from_files(cfg_path: Path, ckpt_path: Path,
                             map_location="cpu", batch=None,
                             reinit=False) -> ProteinEncoder:
    """Instantiate ProteinEncoder and (optionally) load weights from ckpt."""
    full_cfg  = OmegaConf.load(cfg_path)
    enc_cfg   = _to_plain_dict(full_cfg)
    enc_cfg.pop("_target_", None)
    encoder = ProteinEncoder(**enc_cfg).to("cuda")

    if reinit:
        print("+++ Re-initialising ProteinEncoder weights +++")
        return init_weights_slae(encoder)

    ckpt      = torch.load(ckpt_path, map_location=map_location)
    enc_state = {k.replace("encoder.", ""): v
                 for k, v in ckpt["state_dict"].items()
                 if k.startswith("encoder.")}
    # materialise lazy layers if a dummy batch is provided
    strict_load = False
    if batch is not None:
        with torch.no_grad():
            _ = encoder(batch)
        strict_load = True

    missing, unexpected = encoder.load_state_dict(enc_state, strict=strict_load)
    if missing or unexpected:
        print("State-dict mismatch: missing=", missing, "  unexpected=", unexpected)
    encoder.eval()
    return encoder


def load_decoder_from_ae_ckpt(ckpt_path: Path,
                              decoder_kwargs: Mapping[str, Any],
                              map_location="cpu") -> AllAtomDecoder:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    dec_state = {k.replace("decoder.allatom_decoder.", ""): v
                 for k, v in ckpt["state_dict"].items()
                 if k.startswith("decoder.allatom_decoder.")}
    decoder = AllAtomDecoder(**decoder_kwargs)
    decoder.load_state_dict(dec_state, strict=False)
    decoder.eval()
    return decoder


def build_decoder_from_files(cfg_path: Path, ckpt_path: Path,
                             map_location="cpu", reinit_weights="none"):
    full_cfg = OmegaConf.load(cfg_path)
    dec_cfg  = _to_plain_dict(full_cfg["allatom_decoder"])
    dec_cfg.pop("_target_", None)
    dec_cfg["pos_enc_type"] = "rotary"

    if reinit_weights == "all":
        decoder = AllAtomDecoder(**dec_cfg)
        return initialize_weights_transformer(decoder)
    elif reinit_weights == "linear":
        # re-initialise only linear_output_projection layers
        module = "linear_output_projection"
        for k in list(dec_cfg.keys()):
            if module in k:
                dec_cfg.pop(k)
    return load_decoder_from_ae_ckpt(ckpt_path, dec_cfg, map_location)

# ------------------------------------------------------------------ #
#  Shift-lookup (batch-time)                                         #
# ------------------------------------------------------------------ #
def get_shift_and_mask(shift_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                       pdb_ids: List[str],
                       pad_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (shift, mask) tensors aligned with padded residue batch.
    """
    device      = pad_masks.device
    B, L_max    = pad_masks.shape
    shift_pad   = torch.zeros((B, L_max), dtype=torch.float32, device=device)
    shift_mask  = torch.zeros((B, L_max), dtype=torch.bool,  device=device)

    for i, pdb in enumerate(pdb_ids):
        entry = shift_dict.get(pdb.lower())
        if entry is None:
            continue
        shift_1d, mask_1d = entry
        n = min(len(shift_1d), L_max)
        shift_pad[i, :n]  = shift_1d[:n].to(device)
        shift_mask[i, :n] = mask_1d[:n].to(device)

    total_mask = pad_masks & shift_mask
    return shift_pad, total_mask

# ------------------------------------------------------------------ #
#  LightningModule                                                   #
# ------------------------------------------------------------------ #
class CSLightning(pl.LightningModule):
    def __init__(self, ckpt: Path, shift_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                 lr: float = 1e-3, lr_last: float = 3e-5, loss_type: str = "huber",
                 transformer: bool = False, reinit_decoder_weights="none",
                 reinit_encoder_weights=False, embedding="residue_embedding", atom: str = "N"):
        super().__init__()
        self.save_hyperparameters(ignore=["shift_dict"])     # for ckpt reproducibility

        encoder  = build_encoder_from_files(ENCODER_CFG, ckpt,
                                            reinit=reinit_encoder_weights)
        self.embedding = embedding
        self.featuriser = ProteinGraphFeaturizer(
            radius=8.0,
            use_atom37=True)

        self.atom = atom

        
        print("Using ShiftNet")
        if atom == "N":
            min_shift = 100.0
            max_shift = 140.0
            bias = 119.0
        elif atom == "C":
            min_shift = 163.0
            max_shift = 186.0
            bias = 175.0
        elif atom == "H":
            min_shift = 4
            max_shift = 12
            bias = 8
        elif atom == "CA":
            min_shift = 40
            max_shift = 72
            bias =  56
        elif atom == "CB":
            min_shift = 14
            max_shift = 80
            bias = 38
        elif atom == "HA":
            min_shift = 2
            max_shift = 7
            bias = 4.5
        print("Bias", bias)
        self.model = ShiftNet(emb_dim=128, width=256,
                                ppm_window=(min_shift, max_shift),  
                                bias = bias,#drop=0.3,
                                #conv_kernel_size=3,
                                #transformer_heads = 4,
                                #transformer_layers = 4, 
                                analytic_centering=False, depth=4)
        print("ShiftNet initialized with bias", self.model.bias)
            
        self.encoder = encoder
        self.shift_dict = shift_dict
        self.loss_type = loss_type.lower()
        self.lr, self.lr_last = lr, lr_last

        # buffers for validation logging
        self._val_pred, self._val_true, self._val_res_idx, self._val_pdb = ([] for _ in range(4))
        self._buf = {
            "val":  {"pred": [], "true": [], "res_idx": [], "pdb": []},
            "test": {"pred": [], "true": [], "res_idx": [], "pdb": []},
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        split = "val" if dataloader_idx == 0 else "test"
        pred, pad_mask = self.forward(batch)
        shift, total_mask = get_shift_and_mask(self.shift_dict, batch.id, pad_mask)

        if self.atom == "N":
            in_window = (shift >= 100.0) & (shift <= 140.0)
        elif self.atom == "C":
            in_window = (shift >= 163) & (shift <= 186)
        elif self.atom == "H":
            in_window = (shift >= 4) & (shift <= 12)
        elif self.atom == "CA":
            in_window = (shift >= 40) & (shift <= 72)
        elif self.atom == "CB":
            in_window = (shift >= 14) & (shift <= 80)
        elif self.atom == "HA":
            in_window = (shift >= 2) & (shift <= 7)
        total_mask &= in_window

        if total_mask.sum().item() == 0:
            return

        err  = (pred - shift)[total_mask]
        rmse = torch.sqrt((err ** 2).mean())
        mae  = err.abs().mean()
        x = pred[total_mask].detach().cpu().numpy()
        y = shift[total_mask].detach().cpu().numpy()
        rho = spearmanr(x, y).correlation
        if np.isnan(rho):
            rho = 0.0

        self.log_dict({f"{split}_RMSE": rmse,
                    f"{split}_MAE": mae,
                    f"{split}_Spearman": rho},
                    prog_bar=True, on_step=False, on_epoch=True,
                    batch_size=shift.size(0), add_dataloader_idx=False,)

        b = self._buf[split]
        b["pred"].append(pred[total_mask].detach().cpu())
        b["true"].append(shift[total_mask].detach().cpu())
        for i, pdb in enumerate(batch.id):
            idx = total_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if idx.numel():
                b["res_idx"].append(idx.cpu())
                b["pdb"].extend([pdb] * idx.numel())

    # --- on_validation_epoch_end ---
    @torch.no_grad()
    def on_validation_epoch_end(self):
        log_dir = getattr(self.logger, "save_dir", None) or getattr(self.trainer, "default_root_dir", ".")
        out_dir = Path(log_dir) / "shift_history"
        out_dir.mkdir(parents=True, exist_ok=True)


        for split in ("val", "test"):
            b = self._buf[split]
            if not b["pred"]:
                continue
            pred = torch.cat(b["pred"], 0).numpy()
            true = torch.cat(b["true"], 0).numpy()
            idx  = torch.cat(b["res_idx"], 0).numpy()
            pdb  = np.array(b["pdb"])
            corr = pearson(pred, true)[0]
            # compute rmse
            rmse = np.sqrt(((pred - true) ** 2).mean())
            print(f"split: {split}, rmse: {rmse}, corr: {corr}")

            out_path = out_dir / f"{self.atom}_{split}_epoch{self.current_epoch:03d}_corr{corr:.3f}_rmse{rmse:.3f}.pt"
            torch.save({"pred": pred, "true": true, "res_idx": idx, "pdb_id": pdb}, out_path)
            print(f"Saved {split} shift history to {out_path}")

            # clear split buffers
            for k in b.keys():
                b[k].clear()

    # ---------------- forward -------------------------------------- #
    def forward(self, batch):
        batch = self.featuriser(batch)
      

        if self.embedding == "residue_embedding":
            res_embedding = self.encoder(batch)[self.embedding]
        else:  # node_embedding – use N atoms only
            node_embedding = self.encoder(batch)[self.embedding]   # [N_atom, 128]
            n_mask         = batch.atom37_type.eq(0)
            res_embedding  = node_embedding[n_mask]
            assert res_embedding.size(0) == batch.residue_batch.size(0)

        residue_batch = batch.residue_batch
        batch_size = len(batch.id)
        lengths = torch.bincount(residue_batch, minlength=batch_size)  # [B]
        seq_list, start = [], 0
        for length in lengths:
            seq_len = length.item()
            seq_list.append(res_embedding[start:start + seq_len])
            start += seq_len

        res_emb_padded = nn.utils.rnn.pad_sequence(seq_list, batch_first=True,
                                                   padding_value=PAD_VALUE)
        max_len = res_emb_padded.size(1)
        mask = torch.arange(max_len, device=res_emb_padded.device).unsqueeze(0) < lengths.unsqueeze(1)
        pred = self.model(res_emb_padded, mask)
        return pred, mask

    # ---------------- training step -------------------------------- #
    def training_step(self, batch, _ ):
        pred, pad_mask = self.forward(batch)
        shift, total_mask = get_shift_and_mask(self.shift_dict, batch.id, pad_mask)
        if self.atom == "N":
            in_window = (shift >= 100.0) & (shift <= 140.0)
        elif self.atom == "C":
            in_window = (shift >= 163) & (shift <= 186)
        elif self.atom == "H":
            in_window = (shift >= 4) & (shift <= 12)
        elif self.atom == "CA":
            in_window = (shift >= 40) & (shift <= 72)
        elif self.atom == "CB":
            in_window = (shift >= 14) & (shift <= 80)
        elif self.atom == "HA":
            in_window = (shift >= 2) & (shift <= 7) 
        total_mask &= in_window

        if self.loss_type == "mse":
            if self.model.analytic_centering:
                denom = total_mask.sum(-1, keepdim=True).clamp(min=1)
                mean_err = ((pred - shift) * total_mask).sum(-1, keepdim=True) / denom
                pred_adj = pred - mean_err
            else:
                pred_adj = pred
            loss = ((pred_adj - shift) ** 2)[total_mask].mean()
        else:                   # huber
            loss  = self.model.loss(pred, shift, total_mask)


        err = (pred - shift)[total_mask]
        rmse = torch.sqrt((err ** 2).mean())
        mae  = err.abs().mean()

        self.log_dict({"train_loss": loss, "train_RMSE": rmse, "train_MAE": mae},
                      prog_bar=True, on_step=True, on_epoch=True,
                      batch_size=shift.size(0))
        return loss


    # ---------------- optimiser + scheduler ------------------------ #
    def configure_optimizers(self):
        opt   = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = LinearWarmupCosineAnnealingLR(opt,
                                              warmup_epochs=100,
                                              max_epochs=30000,
                                              warmup_start_lr=1e-6)
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sched,
                                 "interval": "step",
                                 "name": "lr"}}

# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",    required=True, type=Path,
                   help="Checkpoint containing encoder weights (.ckpt)")
    p.add_argument("--batch",   type=int, default=2)
    p.add_argument("--epochs",  type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--loss",    choices=["huber", "mse"], default="huber")
    p.add_argument("--transformer", action="store_true")
    p.add_argument("--reinit",  choices=["all", "linear", "none"], default="none")
    p.add_argument("--reinitenc", action="store_true")
    p.add_argument("--embed",   choices=["residue_embedding", "node_embedding"],
                   default="residue_embedding")
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--atom", type=str, default="N", choices=["N", "C", "CA", "CB", "H", "HA"])
    args = p.parse_args()

    # ---------- load JSONs & build ID sets ------------------------ #
    shift_dict, train_ids, val_ids, test_ids = build_shift_and_id_sets(atom=args.atom)

    print(len(train_ids))
    print(len(val_ids))
    print(len(test_ids))
    # ---------- data: PDBDataModule ------------------------------- #
    dm = PDBDataModule(
        pdb_dir="/scratch/users/yilinc5/cs/mbmrb_shiftxf_structures_af2",
        processed_dir="/scratch/users/yilinc5/cs/mbmrb_shiftxf_structures_af2_processed",
        batch_size=args.batch,
        dataset_fraction=args.fraction,
        overwrite=False,
        in_memory=False,
        crop=False,
        rand_slice=False,
        train_list="mbmrb_stringent_shiftx_f_train.list",
        val_list="mbmrb_stringent_shiftx_f_val.list",
        test_list="shiftx_test_shifts_filtered_mbmrb.list", # "C" or "H"
    )
    dm.setup()        # build the raw datasets first



    # ---------- build raw dataloaders from the DM ------------------ #
    raw_train_loader = dm.train_dataloader()
    raw_val_loader   = dm.val_dataloader()
    raw_test_loader  = dm.test_dataloader()

    from torch.utils.data import Subset, DataLoader
    def _filtered_loader(raw_loader, allowed_ids, shuffle):
        raw_ds = raw_loader.dataset
        keep_idx = [i for i in range(len(raw_ds))
                    if getattr(raw_ds[i], "id", "").lower() in allowed_ids]
        filt_ds = Subset(raw_ds, keep_idx)
        return DataLoader(
            filt_ds,
            batch_size=args.batch,
            shuffle=shuffle,
            num_workers=args.workers,
            collate_fn=raw_loader.collate_fn,
            pin_memory=getattr(raw_loader, "pin_memory", False),
        )

    train_loader = _filtered_loader(raw_train_loader, train_ids, shuffle=True)
    val_loader   = _filtered_loader(raw_val_loader,   val_ids,   shuffle=False)
    test_loader  = _filtered_loader(raw_test_loader,  test_ids,  shuffle=False)

    # overlap checks (optional but recommended)
    def ids_from_loader(loader):
        subset, raw_ds = loader.dataset, loader.dataset.dataset
        return {raw_ds[i].id.lower() for i in subset.indices}

    train_seen = ids_from_loader(train_loader)
    val_seen   = ids_from_loader(val_loader)
    test_seen  = ids_from_loader(test_loader)
    for name, a, b in [("train-val", train_seen, val_seen),
                    ("train-test", train_seen, test_seen),
                    ("val-test", val_seen, test_seen)]:
        o = a & b
        print(f"ID overlap ({name}):", len(o))
        assert not o, f"Leakage {name}! examples: {list(o)[:5]}"

    # ---------- model --------------------------------------------- #
    model = CSLightning(ckpt=args.ckpt,
                        shift_dict=shift_dict,
                        loss_type=args.loss,
                        transformer=args.transformer,
                        reinit_decoder_weights=args.reinit,
                        reinit_encoder_weights=args.reinitenc,
                        embedding=args.embed,
                        lr = 5e-4,
                        atom=args.atom).to("cuda")

    # ---------- lightning trainer --------------------------------- #
    wandb_logger = pl.loggers.WandbLogger(project="Mapper",
                                          save_dir="/path/cs/logger")
    ckpt_cb = plcb.ModelCheckpoint(monitor="val_RMSE", mode="min",
                                   filename="best-{atom}_{val_RMSE:.3f}", save_top_k=3)
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision=32,
                         max_epochs=args.epochs,
                         callbacks=[ckpt_cb,
                                    plcb.LearningRateMonitor(logging_interval="step")],
                         log_every_n_steps=10,
                         logger=wandb_logger,
                         gradient_clip_val=1.0)

    # ---------- sanity-check encoder lazy layers ------------------ #
    dm.setup(stage="lazy_init")
    batch = next(iter(dm.val_dataloader())).to("cuda")
    batch_feat = model.featuriser(batch).to("cuda")
    model.encoder = build_encoder_from_files(ENCODER_CFG, args.ckpt,
                                             batch=batch_feat).to("cuda")
    del batch, batch_feat

    # ---------- train --------------------------------------------- #
    #trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
    trainer.fit(model, train_loader, [val_loader, test_loader])
    wandb_logger.experiment.save(ckpt_cb.best_model_path)
    print("Best checkpoint:", ckpt_cb.best_model_path)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    cli()