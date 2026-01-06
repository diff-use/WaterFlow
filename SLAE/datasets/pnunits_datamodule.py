from pathlib import Path
from typing import Optional, List, Any

import lightning as L
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from loguru import logger

from atomworks.ml.datasets.datasets import PandasDataset
from atomworks.ml.samplers import calculate_weights_for_pdb_dataset_df
from atomworks.ml.datasets.loaders import create_loader_with_query_pn_units

from SLAE.features.transform import build_transform_pipeline
from SLAE.datasets.dataloader import ProteinDataLoader  # <- uses Collater under the hood


PDB_MIRROR_PATH = "/scratch/users/yilinc5/pdb/"


def worker_init_fn(worker_id: int):
    # Same worker seeding style as your existing DataModule
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


class GraphPandasDataset(PandasDataset):
    """
    PandasDataset variant whose __getitem__ returns only the transformed
    graph object (data["graph"]) instead of the full dict.
    """

    def __getitem__(self, idx: int) -> Any:
        # This mirrors the original PandasDataset.__getitem__,
        # but returns data["graph"] after transform.
        raw_data = self.data.iloc[idx]
        example_id = self._get_example_id(idx)
        try:
            data = self._apply_loader(raw_data)
            data = self._apply_transform(data, example_id=example_id, idx=idx)
        except Exception as e:
            logger.warning(
                f"[DATASET] Failed at idx={idx}, example_id={example_id}: {e}"
            )
            # Return None → your Collater will drop this sample
            return None
        #data = self._apply_loader(raw_data)
        #data = self._apply_transform(data, example_id=example_id, idx=idx)
        # also check that data["graph"].coords has first dimension > 5
        if data["graph"].coords.size(0) <= 5:
            logger.warning(
                f"[DATASET] Dropping idx={idx}, example_id={example_id} "
                f"with too few residues: {data['graph'].coords.size(0)}"
            )
            return None

        # Assumes transform returns a mapping with key "graph"
        return data["graph"]

class PNUnitsDataModule(L.LightningDataModule):
    """
    DataModule for pn_unit-based training/validation using PandasDataset,
    AtomWorks loaders, and WeightedRandomSampler, but with the same
    ProteinDataLoader/Collater stack as the original PDBDataModule.
    """

    def __init__(
        self,
        train_parquet: str = "",
        val_parquet: str = "",
        batch_size: int = 8,
        train_samples_per_epoch: int = 10_000,
        val_samples_per_epoch: int = 2_000,
        pdb_mirror_path: str = PDB_MIRROR_PATH,
        crop_size: int = 512,
        num_workers: int = 3,
        pin_memory: bool = True,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ) -> None:
        super().__init__()

        self.train_parquet = train_parquet
        self.val_parquet = val_parquet
        self.batch_size = batch_size
        self.train_samples_per_epoch = train_samples_per_epoch
        self.val_samples_per_epoch = val_samples_per_epoch

        self.pdb_mirror_path = Path(pdb_mirror_path)
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.follow_batch = follow_batch or []
        self.exclude_keys = exclude_keys or []

        # These will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.train_sampler = None
        self.val_sampler = None

        self.crop_size = crop_size

    def _build_pn_loader(self):
        """Create AtomWorks loader for pn_units."""
        pn_loader = create_loader_with_query_pn_units(
            pn_unit_iid_colnames=["q_pn_unit_iid"],
            base_path=str(self.pdb_mirror_path),
            parser_args={"hydrogen_policy": "remove"},
        )
        return pn_loader

    def _build_transform(self):
        """SLAE graph transform / crop pipeline."""
        graph_pipeline = build_transform_pipeline(
            is_inference=False,
            crop_size=self.crop_size,
            crop_center_cutoff_distance=15.0,
            crop_contiguous_probability=1.0,
            crop_spatial_probability=0.0,
            max_atoms_in_crop=None,
        )
        return graph_pipeline

    def setup(self, stage: Optional[str] = None):
        """
        Build datasets and samplers. Lightning will call this before training/validation.
        """
        # Only build once (Lightning may call setup multiple times)
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        logger.info("Setting up PNUnitsDataModule (stage={!r})", stage)

        pn_loader = self._build_pn_loader()
        graph_pipeline = self._build_transform()

        # -------------------- Train dataset -------------------- #
        train_filters = [
            "deposition_date < '2022-01-01'",
            "resolution < 5.0 and ~method.str.contains('NMR')",
            "num_polymer_pn_units <= 20",
            "cluster.notnull()",
            "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
            # Train only on D/L polypeptides:
            "q_pn_unit_type in [5, 6]",  # 5 = POLYPEPTIDE_D, 6 = POLYPEPTIDE_L
            # Exclude ligands from AF3 excluded set:
            "~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('HEM|FAD|NAP|FMN|ADP|GDP|GTP|ATP|COA|NAD|NADP'))",
        ]

        self.train_dataset = GraphPandasDataset(
            name="pn_units_train",
            id_column="example_id",
            data=self.train_parquet,
            loader=pn_loader,
            transform=graph_pipeline,
            filters=train_filters,
        )

        logger.info(
            f"Train dataset initialized with {len(self.train_dataset.data)} rows "
            f"after filters."
        )

        # -------------------- Val dataset -------------------- #
        val_filters = [
            "deposition_date > '2022-05-01'",
            "deposition_date < '2023-01-12'",
            "resolution < 5.0 and ~method.str.contains('NMR')",
            "num_polymer_pn_units <= 20",
            "cluster.notnull()",
            "method in ['X-RAY_DIFFRACTION', 'ELECTRON_MICROSCOPY']",
            "q_pn_unit_type in [5, 6]",
            "~(q_pn_unit_non_polymer_res_names.notnull() and q_pn_unit_non_polymer_res_names.str.contains('HEM|FAD|NAP|FMN|ADP|GDP|GTP|ATP|COA|NAD|NADP'))",
        ]

        self.val_dataset = GraphPandasDataset(
            name="pn_units_val",
            id_column="example_id",
            data=self.val_parquet,
            loader=pn_loader,
            transform=graph_pipeline,
            filters=val_filters,
        )

        logger.info(
            f"Val dataset initialized with {len(self.val_dataset.data)} rows "
            f"after filters."
        )

        # -------------------- Samplers -------------------- #
        alphas = {
            "a_prot": 1.0,
            # "a_nuc": 3.0,
            # "a_ligand": 1.0,
        }
        b_pn_unit = 1.0

        train_dataset_weights = calculate_weights_for_pdb_dataset_df(
            dataset_df=self.train_dataset.data, alphas=alphas, beta=b_pn_unit
        )
        val_dataset_weights = calculate_weights_for_pdb_dataset_df(
            dataset_df=self.val_dataset.data, alphas=alphas, beta=b_pn_unit
        )

        self.train_sampler = WeightedRandomSampler(
            weights=train_dataset_weights,
            num_samples=self.train_samples_per_epoch,
            replacement=True,
        )
        self.val_sampler = WeightedRandomSampler(
            weights=val_dataset_weights,
            num_samples=self.val_samples_per_epoch,
            replacement=True,
        )

        logger.info(
            f"Train sampler: {self.train_samples_per_epoch} samples/epoch; "
            f"Val sampler: {self.val_samples_per_epoch} samples/epoch."
        )

    # ------------------------------------------------------------------ #
    #  Dataloaders                                                       #
    # ------------------------------------------------------------------ #
    def train_dataloader(self) -> ProteinDataLoader:
        logger.info("Initializing PNUnits train dataloader")
        return ProteinDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # must be False when using sampler
            sampler=self.train_sampler,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        logger.info("Initializing PNUnits val dataloader")
        return ProteinDataLoader(
            self.val_dataset,
            batch_size=1, #self.batch_size,
            shuffle=False,  # must be False when using sampler
            sampler=self.val_sampler,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            follow_batch=self.follow_batch,
            exclude_keys=self.exclude_keys,
        )
