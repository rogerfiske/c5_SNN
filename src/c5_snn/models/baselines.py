"""Heuristic baseline models for the CA5 task."""

import logging

import torch

from c5_snn.models.base import MODEL_REGISTRY, BaseModel

logger = logging.getLogger("c5_snn")


class FrequencyBaseline(BaseModel):
    """Non-learned baseline ranking parts by frequency + recency.

    For each sample in the batch, scores each of the 39 parts as:
        score = freq_weight * frequency + recency_weight * recency

    where:
    - frequency = count of active steps per part across the window
    - recency = exponentially-decayed sum, weighting recent steps higher

    This model has NO learnable parameters.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        self.freq_weight = float(model_cfg.get("freq_weight", 1.0))
        self.recency_weight = float(
            model_cfg.get("recency_weight", 1.0)
        )
        self.decay = float(model_cfg.get("decay", 0.9))
        logger.info(
            "FrequencyBaseline: freq_weight=%.2f, "
            "recency_weight=%.2f, decay=%.2f",
            self.freq_weight,
            self.recency_weight,
            self.decay,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score parts by frequency + exponentially-decayed recency.

        Args:
            x: (batch, W, 39) windowed multi-hot input.

        Returns:
            (batch, 39) scores for each part.
        """
        W = x.shape[1]

        # Frequency: count of active steps per part
        freq = x.sum(dim=1)  # (batch, 39)

        # Recency weights: decay^(W-1-t) for t=0..W-1
        # t=0 (oldest) -> decay^(W-1), t=W-1 (newest) -> decay^0 = 1.0
        exponents = torch.arange(
            W - 1, -1, -1, dtype=x.dtype, device=x.device
        )
        weights = self.decay ** exponents  # (W,)

        # Weighted sum: (batch, W, 39) * (W, 1) -> sum over W -> (batch, 39)
        recency = (x * weights.unsqueeze(1)).sum(dim=1)  # (batch, 39)

        return self.freq_weight * freq + self.recency_weight * recency


# Register in model registry
MODEL_REGISTRY["frequency_baseline"] = FrequencyBaseline
