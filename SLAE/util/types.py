"""Type definitions for the project.
"""

from typing import Dict, Any, Optional, Literal
from dataclasses import dataclass
import torch
from torch import Tensor

ActivationType = Literal[
    "relu", "elu", "leaky_relu", "tanh", "sigmoid", "none", "silu", "swish"
]




@dataclass
class ModelOutput:
    """Output from models.

    Attributes:
        batch: The processed batch
        encoder_output: Output from encoder
        decoder_output: Output from decoder
        metrics: Optional metrics dictionary
    """
    batch: Any
    encoder_output: Dict
    decoder_output: Optional[Dict[str, Tensor]] = None
    metrics: Optional[Dict[str, float]] = None


