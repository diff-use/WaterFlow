import os
import warnings
import sys
import types
from typing import List, Optional
import pathlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Hydra tools
import hydra
from hydra.compose import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.core.plugins import Plugins


import lightning as L
import lovely_tensors as lt
import torch
import torch.nn as nn
import torch_geometric

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from loguru import logger as log
from omegaconf import DictConfig


from SLAE.model.autoencoder import AutoEncoderModel

# Set config path
HYDRA_CONFIG_PATH = "/home/srivasv/SLAE/configs"

try:
    from lightning_utilities.core.apply_func import apply_to_collection
except ImportError:
    from pytorch_lightning.utilities.apply_func import apply_to_collection


def _num_training_steps(
    train_dataset, trainer: L.Trainer
) -> int:
    """
    Returns total training steps inferred from datamodule and devices.

    :param train_dataset: Training dataloader
    :param trainer: Lightning trainer
    :type trainer: L.Trainer
    :return: Total number of training steps
    :rtype: int
    """
    if trainer.max_steps != -1:
        return trainer.max_steps

    dataset_size = (
        trainer.limit_train_batches
        if trainer.limit_train_batches not in {0, 1}
        else len(train_dataset) * train_dataset.batch_size
    )

    log.info(f"Dataset size: {dataset_size}")

    num_devices = max(1, trainer.num_devices)
    effective_batch_size = (
        train_dataset.batch_size
        * trainer.accumulate_grad_batches
        * num_devices
    )
    return (dataset_size // effective_batch_size) * trainer.max_epochs




# Create a fake module and insert it into sys.modules
fake_module = types.ModuleType("pytorch_lightning.utilities.apply_func")
fake_module.apply_to_collection = apply_to_collection
sys.modules["pytorch_lightning.utilities.apply_func"] = fake_module


warnings.filterwarnings("ignore", category=UserWarning)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "56226"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
version_base = "1.3"  # Note: Need to update whenever Hydra is upgraded



def init_hydra_singleton(
    path: os.PathLike = HYDRA_CONFIG_PATH,
    reload: bool = False,
    version_base: str = "1.3",
) -> None:
    """Initialises the hydra singleton.

    .. seealso::
        https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main

    :param path: Path to hydra config, defaults to ``constants.HYDRA_CONFIG_PATH``
    :type path: os.PathLike, optional
    :param reload: Whether to reload the hydra config if it has already been
        initialised, defaults to ``False``
    :type reload: bool, optional
    :raises ValueError: If hydra has already been initialised and ``reload`` is
        ``False``
    """
    # See: https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
    if reload:
        #clear_hydra_singleton()
        if hydra.core.global_hydra.GlobalHydra not in hydra.core.singleton.Singleton._instances:  # type: ignore
            return
        hydra_singleton = hydra.core.singleton.Singleton._instances[hydra.core.global_hydra.GlobalHydra]  # type: ignore
        hydra_singleton.clear()
        log.info("Hydra singleton cleared and ready to re-initialise.")
    try:
        path = pathlib.Path(path)
        # Note: hydra needs to be initialised with a relative path. Since the hydra
        #  singleton is first created here, it needs to be created relative to this
        #  file. The `rel_path` below takes care of that.
        rel_path = os.path.relpath(path, start=pathlib.Path(__file__).parent)
        hydra.initialize(rel_path, version_base=version_base)
        log.info(f"Hydra initialised at {path.absolute()}.")
    except ValueError:
        log.info("Hydra already initialised.")


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiates callbacks from Hydra config.

    :param callbacks_cfg: Hydra config for callbacks
    :type callbacks_cfg: DictConfig
    :raises TypeError: If callbacks config is not a DictConfig
    :return: List of instantiated callbacks
    :rtype: List[Callback]
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("Callbacks config is empty.")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks

def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: Hydra config for loggers
    :type logger_cfg: DictConfig
    :raises TypeError: If logger config is not a DictConfig
    :return: List of instantiated loggers
    :rtype: List[Logger]
    """

    logger: List[Logger] = []

    if not logger_cfg:
        log.warning("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.info(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    return logger


init_hydra_singleton(reload=True, version_base=version_base)

# IMPORTANT
path = HYDRA_CONFIG_PATH
rel_path = os.path.relpath(path, start=".")

GlobalHydra.instance().clear()
#hydra.initialize(rel_path, version_base=version_base)

with hydra.initialize(config_path=rel_path, version_base=version_base):
    cfg = hydra.compose(
    config_name="train_autoencoder",
    overrides=[
        "name=my_training_run",
        "task=eng_struct_pretrain",
        "task.pretrained_ckpt=null",  # or path to pretrained weights if you have them
    ],
    return_hydra_config=True
    )

    cfg.callbacks.early_stopping = None

    cfg.hydra.job.num = 0
    cfg.hydra.job.id = 0
    cfg.hydra.hydra_help.hydra_help = False
    # where to save checkpoints
    cfg.hydra.runtime.output_dir = "/home/srivasv/slae_cps"
    os.makedirs(cfg.hydra.runtime.output_dir, exist_ok=True)
    HydraConfig.instance().set_config(cfg)


lt.monkey_patch()
def train_model(
    cfg: DictConfig,
):  # sourcery skip: extract-method
    """
    Trains a model from a config.


    1. The datamodule is instantiated from ``cfg.dataset.datamodule``.
    2. The callbacks are instantiated from ``cfg.callbacks``.
    3. The logger is instantiated from ``cfg.logger``.
    4. The trainer is instantiated from ``cfg.trainer``.
    5. (Optional) If the config contains a scheduler, the number of training steps is
         inferred from the datamodule and devices and set in the scheduler.
    6. The model is instantiated from ``cfg.model``.
    7. The datamodule is setup and a dummy forward pass is run to initialise
    lazy layers for accurate parameter counts.
    8. Hyperparameters are logged to wandb if a logger is present.
    9. The model is compiled if ``cfg.compile`` is True.
    10. The model is trained if ``cfg.task_name`` is ``"train"``.
    11. The model is tested if ``cfg.test`` is ``True``.

    :param cfg: DictConfig containing the config for the experiment
    :type cfg: DictConfig
    :param encoder: Optional encoder to use instead of the one specified in
        the config
    :type encoder: Optional[nn.Module]
    """
    # set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(cfg.seed)

    log.info(
        f"Instantiating datamodule: <{cfg.dataset.datamodule._target_}..."
    )
    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule
    )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(
        cfg.get("callbacks")
    )

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating trainer...")
    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    if cfg.get("scheduler"):
        if (
            cfg.scheduler.scheduler._target_
            == "flash.core.optimizers.LinearWarmupCosineAnnealingLR"
            and cfg.scheduler.interval == "step"
        ):
            datamodule.setup(stage="fit")  # type: ignore
            num_steps = _num_training_steps(
                datamodule.train_dataloader(), trainer
            )
            log.info(
                f"Setting number of training steps in scheduler to: {num_steps}"
            )
            cfg.scheduler.scheduler.warmup_epochs = (
                num_steps * cfg.scheduler.scheduler.warmup_epochs / trainer.max_epochs
            )
            cfg.scheduler.scheduler.max_epochs = num_steps
            log.info(cfg.scheduler)

    log.info("Instantiating model...")
    device = torch.device("cuda")
    log.info(f"Moving model to device  {device}")
    model: L.LightningModule = AutoEncoderModel(cfg).to(device)


    log.info("Initializing lazy layers...")
    with torch.no_grad():
        datamodule.setup(stage="fit")  # type: ignore
        batch = next(iter(datamodule.val_dataloader()))
        log.info(f"Unfeaturized batch: {batch}")
        batch = model.featurise(batch).to("cuda")
        log.info(f"Featurized batch: {batch}")
        log.info(f"Batch id: {batch.id}")
        log.info(f"Example labels: {model.get_labels(batch)}")
        # Check batch has required attributes
        for attr in model.encoder.required_batch_attributes:  # type: ignore
            if not hasattr(batch, attr):
                raise AttributeError(
                    f"Batch {batch} does not have required attribute: {attr} ({model.encoder.required_batch_attributes})"
                )
        out = model(batch)
        log.info(f"Model output: {out}")
        del batch, out

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch_geometric.compile(model, dynamic=True)

    if cfg.get("task_name") == "train":
        log.info("Starting training!")
        trainer.fit(
            model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
        )

train_model(cfg)