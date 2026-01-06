"""Base model class.

"""

import lightning as L
import torch
import torch.nn as nn
from typing import Optional, Any
from loguru import logger
import hydra
from omegaconf import OmegaConf



class BaseModel(L.LightningModule):
    """Base class for protein models.

    Provides common functionality for training, validation, and testing.
    """

    def __init__(self):
        super().__init__()
        self.featuriser = None

    def featurise(self, batch):
        """Featurise a batch of data.

        Args:
            batch: Input batch

        Returns:
            Featurised batch
        """
        if self.featuriser is not None:
            return self.featuriser(batch)
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Called after batch is transferred to device.

        Args:
            batch: The batch
            dataloader_idx: Index of the dataloader

        Returns:
            The batch (possibly modified)
        """
        return self.featurise(batch)

    def configure_optimizers(self):
        """Configure optimizers.

        Override this method to configure custom optimizers.

        Returns:
            Optimizer configuration
        """
        if hasattr(self, 'optimizer'):
            return self.optimizer
        return torch.optim.Adam(self.parameters(), lr=1e-4)
    
    def _build_output_decoders(self) -> nn.ModuleDict:
        """
        Instantiate output decoders.

        Decoders are instantiated from their respective config files.

        Decoders are stored in :py:class:`nn.ModuleDict`, indexed by output
        name.

        :return: ModuleDict of decoders indexed by output name
        :rtype: nn.ModuleDict
        """
        decoders = nn.ModuleDict()
        for output_head in self.config.decoder.keys():
            cfg = self.config.decoder.get(output_head)
            logger.info(
                f"Building {output_head} decoder. Output dim {cfg.get('out_dim')}"
            )
            logger.info(cfg)
            decoders[output_head] = hydra.utils.instantiate(cfg)
        return decoders

    def configure_optimizers(self):  # sourcery skip: extract-method
        logger.info("Instantiating optimiser...")
        optimiser = hydra.utils.instantiate(self.config.optimiser)["optimizer"]
        logger.info(optimiser)
        optimiser = optimiser(self.parameters())

        if self.config.get("scheduler"):
            logger.info("Instantiating scheduler...")
            scheduler = hydra.utils.instantiate(
                self.config.scheduler, optimiser
            )
            scheduler = OmegaConf.to_container(scheduler)
            scheduler["scheduler"] = scheduler["scheduler"](
                optimizer=optimiser
            )
            optimiser_config = {
                "optimizer": optimiser,
                "lr_scheduler": scheduler,
            }
            logger.info(f"Optimiser configuration: {optimiser_config}")
            return optimiser_config
        return optimiser