"""BaseModel abstract class, model registry, and factory function."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from torch import nn

from c5_snn.utils.exceptions import ConfigError

logger = logging.getLogger("c5_snn")

MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


class BaseModel(ABC, nn.Module):
    """Abstract base class for all models in the c5_SNN pipeline.

    All models MUST implement forward() with the contract:
        Input:  (batch, W, 39) — windowed multi-hot event sequences
        Output: (batch, 39) — logits/scores for the 39 parts
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, W, 39) -> (batch, 39)."""
        ...


def get_model(config: dict) -> BaseModel:
    """Instantiate a model from config.

    Args:
        config: Experiment config dict. Must have config["model"]["type"].

    Returns:
        Instantiated model.

    Raises:
        ConfigError: If model type not found in registry.
    """
    model_type = config.get("model", {}).get("type", "")
    if model_type not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ConfigError(
            f"Unknown model type '{model_type}'. "
            f"Available: {available}"
        )
    logger.info("Creating model: %s", model_type)
    return MODEL_REGISTRY[model_type](config)
