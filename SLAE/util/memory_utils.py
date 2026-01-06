"""Memory utilities.

This module provides memory management utilities.
"""

import torch
import gc


def clean_up_torch_gpu_memory():
    """Clean up GPU memory by clearing cache and running garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
