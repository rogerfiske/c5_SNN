"""Spike encoding layer for SNN model front-ends (STORY-4.1).

Provides a configurable SpikeEncoder that converts windowed multi-hot
tensors into spike trains using snnTorch's time-first convention.

Modes:
    direct:     Unsqueeze T=1 dim. Binary values passed through as spikes.
    rate_coded: Expand temporal dimension via snntorch.spikegen.rate().

Input:  (batch, W, 39) — windowed multi-hot binary tensor
Output: (T, batch, W, 39) — spike train in snnTorch time-first format
"""

import logging

import torch
from snntorch import spikegen
from torch import nn

from c5_snn.utils.exceptions import ConfigError

logger = logging.getLogger("c5_snn")

VALID_ENCODINGS = ("direct", "rate_coded")


class SpikeEncoder(nn.Module):
    """Configurable spike encoding layer for SNN model front-ends.

    All SNN models should compose this as their input layer so that
    encoding mode can be swapped via config without changing model code.

    Args:
        config: Experiment config dict. Reads from config["model"]:
            encoding (str): "direct" or "rate_coded". Default "direct".
            timesteps (int): Number of time steps for rate_coded. Default 10.
                Ignored when encoding="direct".

    Raises:
        ConfigError: If encoding mode is not one of VALID_ENCODINGS.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        self.encoding = model_cfg.get("encoding", "direct")
        self.timesteps = int(model_cfg.get("timesteps", 10))

        if self.encoding not in VALID_ENCODINGS:
            raise ConfigError(
                f"Unknown encoding mode: '{self.encoding}'. "
                f"Expected one of {VALID_ENCODINGS}."
            )

        logger.info(
            "SpikeEncoder: mode=%s, timesteps=%d",
            self.encoding,
            self.timesteps if self.encoding == "rate_coded" else 1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor into spike trains.

        Args:
            x: Input tensor of shape (batch, W, 39).

        Returns:
            Spike tensor of shape (T, batch, W, 39) where:
                T=1 for direct mode,
                T=self.timesteps for rate_coded mode.
        """
        if self.encoding == "direct":
            # Unsqueeze time dimension: (batch, W, 39) -> (1, batch, W, 39)
            return x.unsqueeze(0)
        # rate_coded: uses snntorch.spikegen.rate
        # spikegen.rate prepends T dimension: (batch, W, 39) -> (T, batch, W, 39)
        return spikegen.rate(x, num_steps=self.timesteps)

    @property
    def num_steps(self) -> int:
        """Return the number of time steps this encoder produces."""
        if self.encoding == "direct":
            return 1
        return self.timesteps
