"""Deterministic seed management for reproducible experiments."""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Set random seed across all libraries for deterministic behavior.

    Must be called as the very first action in any training or evaluation run,
    before model construction, data loading, or shuffling.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %d", seed)
