import functools
import os
import random
from pathlib import Path
from typing import Dict, List, Optional


from loguru import logger
import lightning as L

import torch
from SLAE.datasets.dataset import ProteinDataset
from SLAE.datasets.dataloader import ProteinDataLoader
import numpy as np


def worker_init_fn(worker_id):
    # Seed the random number generators for the worker processes
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

class PDBDataModule(L.LightningDataModule):
    """Data module for CATH dataset.

    :param path: Path to store data.
    :type path: str   
    :param batch_size: Batch size for dataloaders.
    :type batch_size: int
    :param format: Format to load PDB files in.
    :type format: Literal["mmtf", "pdb"]
    :param pdb_dir: Path to directory containing PDB files.
    :type pdb_dir: str
    :param pin_memory: Whether to pin memory for dataloaders.
    :type pin_memory: bool
    :param in_memory: Whether to load the entire dataset into memory.
    :type in_memory: bool
    :param num_workers: Number of workers for dataloaders.
    :type num_workers: int
    :param dataset_fraction: Fraction of dataset to use.
    :type dataset_fraction: float
    :param overwrite: Whether to overwrite existing data.
        Defaults to ``False``.
    :type overwrite: bool
    """

    def __init__(
        self, 
        batch_size: int,
        format: str = "pdb",
        pdb_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        pin_memory: bool = True,
        in_memory: bool = False,
        num_workers: int = 16,
        dataset_fraction: float = 1.0,
        overwrite: bool = False,
        full_graph_eval: bool = True,
        crop: bool = False, 
        crop_len: int = None, 
        rand_slice: bool = False,
        slice_range: list = None,
        inference_only: bool = False,
        train_list: str = None, 
        val_list: str = None,
        test_list: str = None,
        remove_sidechains: bool = False,
    ) -> None:
        super().__init__()

       
        self.processed_dir = Path(processed_dir)
        # check if processed dir exists
        if not self.processed_dir.exists():
            logger.info(f"Creating processed directory at {self.processed_dir}")
            self.processed_dir.mkdir(parents=True, exist_ok=True)


        self.in_memory = in_memory
        self.overwrite = overwrite

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.format = format
        self.pdb_dir = pdb_dir

        self.dataset_fraction = dataset_fraction

        self.prepare_data_per_node = False

        self.full_graph_eval = full_graph_eval

        self.crop = crop
        self.crop_len = crop_len
        self.rand_slice = rand_slice
        self.slice_range = slice_range
        self.inference_only = inference_only

        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list

        self.remove_sidechains = remove_sidechains

    @functools.lru_cache
    def parse_dataset(self) -> Dict[str, List[str]]:
        """Parses dataset index file

        Returns a dictionary with keys "train", "validation", and "test" and
        values as lists of PDB IDs.

        :return: Dictionary of PDB IDs
        :rtype: Dict[str, List[str]]
        """

        
    

        if self.inference_only:
            inference_pdbs = [x.strip() for x in [os.path.basename(pdb) for pdb in os.listdir(self.pdb_dir)]]

            self.inference_pdbs = inference_pdbs
            logger.info(f"Found {len(inference_pdbs)} chains in inference set")
            return {
                "inference": inference_pdbs,
            }

        train_list = self.train_list

        with open(train_list, "r") as f:
            train_pdbs = f.readlines()
        train_pdbs = [x.strip() for x in train_pdbs]
        self.train_pdbs = train_pdbs
        logger.info(f"Found {len(self.train_pdbs)} chains in training set")
        
        logger.info(
            f"Sampling fraction {self.dataset_fraction} of training set"
        )
        fraction = int(self.dataset_fraction * len(self.train_pdbs))
        self.train_pdbs = random.sample(self.train_pdbs, fraction)


       
        # list all pdb files in the data_dir
        val_list = self.val_list
        with open(val_list, "r") as f:
            val_pdbs = f.readlines()
        val_pdbs = [x.strip() for x in val_pdbs]


        self.val_pdbs = val_pdbs
        logger.info(f"Found {len(self.val_pdbs)} chains in validation set")

        if self.test_list:
            test_list = self.test_list
            with open(test_list, "r") as f:
                test_pdbs = f.readlines()
            test_pdbs = [x.strip() for x in test_pdbs]
            self.test_pdbs = test_pdbs
            logger.info(f"Found {len(self.test_pdbs)} chains in test set")
        else:
            self.test_pdbs = []
        data = {
            "train": self.train_pdbs,
            "validation": self.val_pdbs,
            "test": self.test_pdbs,
        }
        return data
    
    def inference_dataset(self) -> ProteinDataset:
        """Returns the inference dataset.

        :return: Inference dataset
        :rtype: ProteinDataset
        """
        logger.info("Initializing inference dataset")

        if not hasattr(self, "inference_pdbs"):
            self.parse_dataset()
        pdb_codes = [pdb.split(".")[0] for pdb in self.inference_pdbs]

        return ProteinDataset(
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            processed_dir=self.processed_dir,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
            crop=False,
            rand_slice=False,
            remove_sidechains=self.remove_sidechains,
        )

    def train_dataset(self) -> ProteinDataset:
        """Returns the training dataset.

        :return: Training dataset
        :rtype: CleanedProteinDataset
        """
        logger.info("Initializing training dataset")

        if not hasattr(self, "train_pdbs"):
            self.parse_dataset()
        pdb_codes = [pdb.split(".")[0] for pdb in self.train_pdbs]

        return ProteinDataset(
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            processed_dir=self.processed_dir,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
            crop=self.crop,
            crop_len=self.crop_len,
            rand_slice=self.rand_slice,
            slice_range=self.slice_range,
            remove_sidechains=self.remove_sidechains,
        )

    def val_dataset(self) -> ProteinDataset:
        """Returns the validation dataset.

        :return: Validation dataset
        :rtype: CleanedProteinDataset
        """
        if not hasattr(self, "val_pdbs"):
            self.parse_dataset()

        pdb_codes = [pdb.split(".")[0] for pdb in self.val_pdbs]

        
        logger.info("Full graph evaluation: getting val dataset")
        return ProteinDataset(
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            processed_dir=self.processed_dir,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
            crop=self.crop,
            crop_len=self.crop_len,
            rand_slice=False,
            remove_sidechains=self.remove_sidechains,   

        )
       

    def test_dataset(self) -> ProteinDataset:
        """Returns the test dataset.

        :return: Test dataset
        :rtype: ProteinDatasetSubgraph
        """
        if not hasattr(self, "test_pdbs"):
            self.parse_dataset()
        pdb_codes = [pdb.split(".")[0] for pdb in self.test_pdbs]

        return ProteinDataset(
            pdb_dir=self.pdb_dir,
            pdb_codes=pdb_codes,
            processed_dir=self.processed_dir,
            format=self.format,
            in_memory=self.in_memory,
            overwrite=self.overwrite,
            crop=self.crop,
            crop_len=self.crop_len,
            remove_sidechains=self.remove_sidechains,
        )


    def train_dataloader(self) -> ProteinDataLoader:
        """Returns the training dataloader.

        :return: Training dataloader
        :rtype: ProteinDataLoader
        """
        logger.info("Initalizing training dataloader")

        if not hasattr(self, "train_ds"):
            self.train_ds = self.train_dataset()
        
        return ProteinDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self) -> ProteinDataLoader:
        logger.info("Initalizing validation dataloader")
        if not hasattr(self, "val_ds"):
            self.val_ds = self.val_dataset()
        if self.full_graph_eval:
            return ProteinDataLoader(
                self.val_ds,
                batch_size=1,
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=worker_init_fn,
            )
        return ProteinDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )

    def test_dataloader(self) -> ProteinDataLoader:
        """Returns the test dataloader.

        :return: Test dataloader
        :rtype: ProteinDataLoader
        """
        logger.info("Initalizing test dataloader")
        if not hasattr(self, "test_ds"):
            self.test_ds = self.test_dataset()
        if self.full_graph_eval:
            return ProteinDataLoader(
                self.test_ds,
                batch_size=1,
                shuffle=False,
                drop_last=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                worker_init_fn=worker_init_fn,
            )   
        return ProteinDataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )


    def inference_dataloader(self) -> ProteinDataLoader:
        """Returns the inference dataloader.

        :return: Inference dataloader
        :rtype: ProteinDataLoader
        """
        logger.info("Initalizing inference dataloader")

        if not hasattr(self, "inference_ds"):
            self.inference_ds = self.inference_dataset()
        
        return ProteinDataLoader(
            self.inference_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
        )
    

