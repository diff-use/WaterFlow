"""Elegant embedding extraction utilities.

This module provides clean, reusable utilities for extracting embeddings
from trained models, designed to work seamlessly with the AllatomAE package.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Union, Literal, List
from tqdm.auto import tqdm
from loguru import logger
import pickle

from SLAE.datasets.datamodule import PDBDataModule
from SLAE.model.autoencoder import AutoEncoderModel


class EmbeddingExtractor:
    """Extract embeddings from trained AllatomAE models.

    This class provides an elegant interface for extracting residue-level
    and graph-level embeddings from protein structures.

    Example:
        >>> extractor = EmbeddingExtractor.from_checkpoint(
        ...     "path/to/checkpoint.ckpt",
        ...     config_path="configs/train_autoencoder.yaml"
        ... )
        >>> embeddings = extractor.extract_from_datamodule(datamodule)
        >>> extractor.save_embeddings(embeddings, "output/embeddings.pkl")
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        use_amp: bool = False,
    ):
        """Initialize the embedding extractor.

        Args:
            model: Trained model (typically AutoEncoderModel)
            device: Device to run extraction on
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.use_amp = use_amp

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        use_amp: bool = False,
    ) -> "EmbeddingExtractor":
        """Create an extractor from a checkpoint file.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Optional path to config file (if needed for model init)
            device: Device to run extraction on
            use_amp: Whether to use automatic mixed precision

        Returns:
            EmbeddingExtractor instance
        """
        checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Try to reconstruct model from checkpoint
        # This assumes the checkpoint contains the full model state
        if config_path is not None:
            from omegaconf import OmegaConf
            import hydra
            cfg = OmegaConf.load(config_path)
            model = AutoEncoderModel(cfg)
        else:
            # Try to infer from checkpoint
            # This may need adjustment based on how models are saved
            raise NotImplementedError(
                "Please provide config_path for model initialization"
            )

        # Load weights
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        return cls(model, device=device, use_amp=use_amp)

    @torch.no_grad()
    def extract_from_batch(
        self,
        batch,
        embedding_type: Literal["residue", "graph", "both"] = "both",
    ) -> Dict[str, torch.Tensor]:
        """Extract embeddings from a single batch.

        Args:
            batch: Input batch (will be featurized automatically)
            embedding_type: Type of embedding to extract

        Returns:
            Dictionary with embeddings and metadata
        """
        # Move batch to device
        batch = batch.to(self.device)

        # Featurize if needed
        if hasattr(self.model, 'featurise'):
            batch = self.model.featurise(batch)

        # Forward pass with optional AMP
        if self.use_amp:
            with torch.cuda.amp.autocast():
                output = self.model(batch)
        else:
            output = self.model(batch)

        # Extract requested embeddings
        result = {}

        if embedding_type in ("residue", "both"):
            if "residue_embedding" in output:
                result["residue_embedding"] = output["residue_embedding"].cpu()

        if embedding_type in ("graph", "both"):
            if "graph_embedding" in output:
                result["graph_embedding"] = output["graph_embedding"].cpu()

        # Add metadata
        if hasattr(batch, 'id'):
            result["pdb_id"] = batch.id[0] if len(batch.id) == 1 else batch.id

        return result

    @torch.no_grad()
    def extract_from_datamodule(
        self,
        datamodule: PDBDataModule,
        split: Literal["train", "val", "test"] = "val",
        embedding_type: Literal["residue", "graph", "both"] = "both",
        max_samples: Optional[int] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Extract embeddings from an entire datamodule.

        Args:
            datamodule: PDBDataModule instance
            split: Which split to extract from
            embedding_type: Type of embedding to extract
            max_samples: Maximum number of samples to process (None = all)

        Returns:
            Dictionary mapping PDB IDs to embeddings
        """
        # Setup datamodule
        datamodule.setup(stage="validate" if split == "val" else split)

        # Get appropriate dataloader
        if split == "train":
            dataloader = datamodule.train_dataloader()
        elif split == "val":
            dataloader = datamodule.val_dataloader()
        else:
            dataloader = datamodule.test_dataloader()

        # Extract embeddings
        residue_embeddings = {}
        graph_embeddings = {}

        for i, batch in enumerate(tqdm(dataloader, desc=f"Extracting {split} embeddings")):
            if max_samples is not None and i >= max_samples:
                break

            try:
                result = self.extract_from_batch(batch, embedding_type)

                # Store embeddings by PDB ID
                pdb_id = result.get("pdb_id")
                if pdb_id is None:
                    logger.warning(f"Batch {i} has no PDB ID, skipping")
                    continue

                if "residue_embedding" in result:
                    residue_embeddings[pdb_id] = result["residue_embedding"]

                if "graph_embedding" in result:
                    graph_embeddings[pdb_id] = result["graph_embedding"]

            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                continue

        # Return organized embeddings
        output = {}
        if residue_embeddings:
            output["residue"] = residue_embeddings
        if graph_embeddings:
            output["graph"] = graph_embeddings

        logger.info(f"Extracted embeddings for {len(residue_embeddings)} structures")
        return output

    @staticmethod
    def save_embeddings(
        embeddings: Dict[str, Dict[str, torch.Tensor]],
        output_path: Union[str, Path],
        separate_files: bool = False,
    ):
        """Save embeddings to disk.

        Args:
            embeddings: Dictionary of embeddings from extract_from_datamodule
            output_path: Path to save embeddings
            separate_files: If True, save residue and graph embeddings separately
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if separate_files:
            # Save residue and graph embeddings separately
            if "residue" in embeddings:
                res_path = output_path.parent / f"{output_path.stem}_residue.pkl"
                with open(res_path, "wb") as f:
                    pickle.dump(embeddings["residue"], f)
                logger.info(f"Saved residue embeddings to {res_path}")

            if "graph" in embeddings:
                graph_path = output_path.parent / f"{output_path.stem}_graph.pkl"
                with open(graph_path, "wb") as f:
                    pickle.dump(embeddings["graph"], f)
                logger.info(f"Saved graph embeddings to {graph_path}")
        else:
            # Save all embeddings together
            with open(output_path, "wb") as f:
                pickle.dump(embeddings, f)
            logger.info(f"Saved embeddings to {output_path}")

    @staticmethod
    def load_embeddings(
        input_path: Union[str, Path]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load embeddings from disk.

        Args:
            input_path: Path to saved embeddings

        Returns:
            Dictionary of embeddings
        """
        with open(input_path, "rb") as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings from {input_path}")
        return embeddings


def extract_embeddings_simple(
    checkpoint_path: Union[str, Path],
    datamodule: PDBDataModule,
    output_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    split: Literal["train", "val", "test"] = "val",
    embedding_type: Literal["residue", "graph", "both"] = "both",
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Simple one-line function to extract and save embeddings.

    This is a convenience function that wraps the EmbeddingExtractor class
    for simple use cases.

    Args:
        checkpoint_path: Path to model checkpoint
        datamodule: PDBDataModule instance
        output_path: Path to save embeddings
        config_path: Optional path to config file
        split: Which split to extract from
        embedding_type: Type of embedding to extract
        device: Device to run on

    Returns:
        Dictionary of extracted embeddings

    Example:
        >>> from SLAE.datasets.datamodule import PDBDataModule
        >>> from SLAE.util.embedding_extractor import extract_embeddings_simple
        >>>
        >>> dm = PDBDataModule(pdb_dir="data/pdbs", ...)
        >>> embeddings = extract_embeddings_simple(
        ...     "checkpoints/model.ckpt",
        ...     dm,
        ...     "output/embeddings.pkl"
        ... )
    """
    # Create extractor
    extractor = EmbeddingExtractor.from_checkpoint(
        checkpoint_path,
        config_path=config_path,
        device=device
    )

    # Extract embeddings
    embeddings = extractor.extract_from_datamodule(
        datamodule,
        split=split,
        embedding_type=embedding_type
    )

    # Save embeddings
    EmbeddingExtractor.save_embeddings(embeddings, output_path)

    return embeddings
