"""Phase B Spiking Neural Network models (STORY-5.1+).

SpikeGRU: Recurrent spiking model that processes the window event-by-event
using snnTorch RLeaky (recurrent LIF) neurons, combining recurrence over
the window dimension W with temporal dynamics over encoding timesteps T.
"""

import logging

import snntorch
import torch
from snntorch import surrogate
from torch import nn

from c5_snn.models.base import MODEL_REGISTRY, BaseModel
from c5_snn.models.encoding import SpikeEncoder

logger = logging.getLogger("c5_snn")

N_FEATURES = 39


class SpikeGRU(BaseModel):
    """Spiking GRU with recurrent LIF neurons for CA5 prediction.

    Architecture:
        SpikeEncoder → for t in T: for w in W: RLeaky(input, spk, mem)
        → mean(T) → Linear → (batch, 39) logits

    Processes the window dimension W event-by-event through recurrent
    spiking neurons (snnTorch RLeaky), accumulating membrane state across
    events within each encoding timestep. This mirrors GRU's sequential
    window processing but with spiking dynamics.

    Two temporal dimensions:
        T: Encoding timesteps (T=1 for direct, T=timesteps for rate_coded)
        W: Window length processed recurrently event-by-event

    Args:
        config: Experiment config dict. Reads from config["model"]:
            hidden_size (int): Recurrent hidden dimension. Default 128.
            num_layers (int): Number of stacked RLeaky layers. Default 1.
            beta (float): LIF membrane decay factor. Default 0.95.
            dropout (float): Dropout between layers (0.0 for single). Default 0.0.
            encoding (str): Spike encoding mode. Default "direct".
            timesteps (int): Time steps for rate_coded encoding. Default 10.
            surrogate (str): Surrogate gradient function. Default "fast_sigmoid".
        Also reads config["data"]["window_size"] (default 21).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})

        # Encoding front-end (shared with all SNN models)
        self.encoder = SpikeEncoder(config)

        # Architecture params
        self.hidden_size = int(model_cfg.get("hidden_size", 128))
        self.num_layers = int(model_cfg.get("num_layers", 1))
        self.beta = float(model_cfg.get("beta", 0.95))
        self.dropout_rate = float(model_cfg.get("dropout", 0.0))
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Input projection: 39 features -> hidden_size
        self.fc_input = nn.Linear(N_FEATURES, self.hidden_size)

        # Stacked RLeaky layers with dense recurrence
        self.rlif_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.rlif_layers.append(
                snntorch.RLeaky(
                    beta=self.beta,
                    linear_features=self.hidden_size,
                    spike_grad=spike_grad,
                    all_to_all=True,
                    init_hidden=False,
                )
            )

        # Dropout between layers (only when num_layers > 1)
        self.dropout = (
            nn.Dropout(self.dropout_rate)
            if self.num_layers > 1 and self.dropout_rate > 0
            else None
        )

        # Readout layer (non-spiking)
        self.readout = nn.Linear(self.hidden_size, N_FEATURES)

        logger.info(
            "SpikeGRU: hidden_size=%d, num_layers=%d, beta=%.3f, "
            "dropout=%.2f, encoding=%s",
            self.hidden_size,
            self.num_layers,
            self.beta,
            self.dropout_rate,
            self.encoder.encoding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking GRU.

        Processes W events recurrently at each encoding timestep T,
        then aggregates over T for the final readout.

        Args:
            x: (batch, W, 39) windowed multi-hot input.

        Returns:
            (batch, 39) logits for each part.
        """
        # Encode input into spike trains: (T, batch, W, 39)
        spikes = self.encoder(x)
        T, B, W = spikes.size(0), spikes.size(1), spikes.size(2)

        # Collect final hidden spike from each encoding timestep
        t_records = []

        for t in range(T):
            # Initialize membrane and spike states for all layers
            spks = [
                torch.zeros(B, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
            mems = [
                torch.zeros(B, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]

            # Process W events recurrently
            for w in range(W):
                cur = self.fc_input(spikes[t, :, w, :])  # (B, hidden_size)

                for i, rlif in enumerate(self.rlif_layers):
                    cur, mems[i] = rlif(cur, spks[i], mems[i])
                    spks[i] = cur  # Output spikes feed into next layer

                    # Apply dropout between layers (not after last)
                    if (
                        self.dropout is not None
                        and i < self.num_layers - 1
                    ):
                        cur = self.dropout(cur)

            # Record final spike state after processing all W events
            t_records.append(spks[-1])  # (B, hidden_size)

        # Aggregate over encoding timesteps T
        stacked = torch.stack(t_records)  # (T, B, hidden_size)
        agg = stacked.mean(dim=0)  # (B, hidden_size)

        # Linear readout
        return self.readout(agg)  # (B, 39) logits


MODEL_REGISTRY["spike_gru"] = SpikeGRU
