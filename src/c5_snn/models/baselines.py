"""Baseline models for the CA5 task (heuristic + ANN)."""

import logging

import torch
from torch import nn

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


class GRUBaseline(BaseModel):
    """Conventional GRU-based neural network baseline.

    Architecture:
        nn.GRU encoder -> final hidden state -> nn.Linear -> 39 logits

    This is the first trainable model in the pipeline, providing a strong
    learned-model benchmark for SNN comparison.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})
        self.hidden_size = int(model_cfg.get("hidden_size", 128))
        self.num_layers = int(model_cfg.get("num_layers", 1))
        self.dropout = float(model_cfg.get("dropout", 0.0))

        self.gru = nn.GRU(
            input_size=39,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(self.hidden_size, 39)

        logger.info(
            "GRUBaseline: hidden_size=%d, num_layers=%d, dropout=%.2f",
            self.hidden_size,
            self.num_layers,
            self.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """GRU encoder -> final hidden -> linear projection.

        Args:
            x: (batch, W, 39) windowed multi-hot input.

        Returns:
            (batch, 39) logits for each part.
        """
        # x: (batch, W, 39)
        _output, h_n = self.gru(x)  # h_n: (num_layers, batch, hidden_size)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        logits = self.fc(last_hidden)  # (batch, 39)
        return logits


MODEL_REGISTRY["gru_baseline"] = GRUBaseline
