"""Spiking Neural Network models for the CA5 task (STORY-4.2+)."""

import logging

import snntorch
import torch
from snntorch import surrogate
from torch import nn

from c5_snn.models.base import MODEL_REGISTRY, BaseModel
from c5_snn.models.encoding import SpikeEncoder

logger = logging.getLogger("c5_snn")

N_FEATURES = 39


class SpikingMLP(BaseModel):
    """Spiking MLP with LIF neurons for CA5 prediction.

    Architecture:
        SpikeEncoder → flatten(T, batch, W*39) → [FC + LIF]×N → mean(T) → Linear → (batch, 39)

    Uses snntorch.Leaky LIF neurons with configurable beta and
    surrogate.fast_sigmoid for gradient computation through spikes.
    Membrane potentials are reset to zero at the start of each forward
    pass to ensure batch independence.

    Args:
        config: Experiment config dict. Reads from config["model"]:
            hidden_sizes (list[int]): Hidden layer sizes. Default [256, 128].
            beta (float): LIF membrane decay factor. Default 0.95.
            encoding (str): Spike encoding mode. Default "direct".
            timesteps (int): Time steps for rate_coded encoding. Default 10.
        Also reads config["data"]["window_size"] (default 21).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})

        # Encoding front-end (shared with all SNN models)
        self.encoder = SpikeEncoder(config)

        # Architecture params
        self.hidden_sizes = list(model_cfg.get("hidden_sizes", [256, 128]))
        beta = float(model_cfg.get("beta", 0.95))
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Input size: W * 39 (flattened spatial dims)
        window_size = config.get("data", {}).get("window_size", 21)
        self.input_size = window_size * N_FEATURES

        # Build spiking layers: pairs of (Linear, Leaky)
        self.fc_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        in_size = self.input_size
        for h_size in self.hidden_sizes:
            self.fc_layers.append(nn.Linear(in_size, h_size))
            self.lif_layers.append(
                snntorch.Leaky(beta=beta, spike_grad=spike_grad)
            )
            in_size = h_size

        # Readout layer (non-spiking)
        self.readout = nn.Linear(in_size, N_FEATURES)

        logger.info(
            "SpikingMLP: hidden_sizes=%s, beta=%.3f, encoding=%s, "
            "input_size=%d",
            self.hidden_sizes,
            beta,
            self.encoder.encoding,
            self.input_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking MLP.

        Args:
            x: (batch, W, 39) windowed multi-hot input.

        Returns:
            (batch, 39) logits for each part.
        """
        # Encode input into spike trains: (T, batch, W, 39)
        spikes = self.encoder(x)
        T, B = spikes.size(0), spikes.size(1)

        # Flatten spatial dims: (T, batch, W*39)
        spikes = spikes.view(T, B, -1)

        # Initialize membrane potentials (reset each forward call)
        mems = [
            torch.zeros(B, h, device=x.device)
            for h in self.hidden_sizes
        ]

        # Temporal loop through spiking layers
        spike_rec = []
        for t in range(T):
            cur = spikes[t]
            for i, (fc, lif) in enumerate(
                zip(self.fc_layers, self.lif_layers)
            ):
                cur = fc(cur)
                cur, mems[i] = lif(cur, mems[i])
            spike_rec.append(cur)

        # Aggregate over time: mean of output spikes
        spike_out = torch.stack(spike_rec)  # (T, B, last_hidden)
        agg = spike_out.mean(dim=0)  # (B, last_hidden)

        # Linear readout
        return self.readout(agg)  # (B, 39) logits


MODEL_REGISTRY["spiking_mlp"] = SpikingMLP


class SpikingCNN1D(BaseModel):
    """Spiking 1D-CNN with temporal convolutions for CA5 prediction.

    Architecture:
        SpikeEncoder → permute(B, 39, W) → [Conv1d + LIF]×N → pool(W) → mean(T) → Linear → (B, 39)

    Applies Conv1d along the window dimension (W) to learn local temporal
    patterns before LIF spiking neurons. Global average pooling over W
    collapses spatial information before the readout layer.

    Args:
        config: Experiment config dict. Reads from config["model"]:
            channels (list[int]): Output channels per conv layer. Default [64, 64].
            kernel_sizes (list[int]): Kernel size per conv layer. Default [3, 3].
            beta (float): LIF membrane decay factor. Default 0.95.
            encoding (str): Spike encoding mode. Default "direct".
            timesteps (int): Time steps for rate_coded encoding. Default 10.
        Also reads config["data"]["window_size"] (default 21).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})

        # Encoding front-end (shared with all SNN models)
        self.encoder = SpikeEncoder(config)

        # Architecture params
        self.channels = list(model_cfg.get("channels", [64, 64]))
        self.kernel_sizes = list(model_cfg.get("kernel_sizes", [3, 3]))
        beta = float(model_cfg.get("beta", 0.95))
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Window size for membrane potential init
        self.window_size = config.get("data", {}).get("window_size", 21)

        # Build conv + LIF layer pairs
        self.conv_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        in_channels = N_FEATURES
        for out_ch, ks in zip(self.channels, self.kernel_sizes):
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_ch, kernel_size=ks, padding=ks // 2)
            )
            self.lif_layers.append(
                snntorch.Leaky(beta=beta, spike_grad=spike_grad)
            )
            in_channels = out_ch

        # Readout layer (non-spiking): last channel count -> 39 logits
        self.readout = nn.Linear(in_channels, N_FEATURES)

        logger.info(
            "SpikingCNN1D: channels=%s, kernel_sizes=%s, beta=%.3f, "
            "encoding=%s, window_size=%d",
            self.channels,
            self.kernel_sizes,
            beta,
            self.encoder.encoding,
            self.window_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking 1D-CNN.

        Args:
            x: (batch, W, 39) windowed multi-hot input.

        Returns:
            (batch, 39) logits for each part.
        """
        # Encode input into spike trains: (T, B, W, 39)
        spikes = self.encoder(x)
        T, B, W = spikes.size(0), spikes.size(1), spikes.size(2)

        # Initialize membrane potentials: (B, C_out, W) per conv layer
        mems = [
            torch.zeros(B, ch, W, device=x.device)
            for ch in self.channels
        ]

        # Temporal loop
        spike_rec = []
        for t in range(T):
            # Permute for Conv1d: (B, W, 39) -> (B, 39, W)
            cur = spikes[t].permute(0, 2, 1)

            for i, (conv, lif) in enumerate(
                zip(self.conv_layers, self.lif_layers)
            ):
                cur = conv(cur)  # (B, C_out, W)
                cur, mems[i] = lif(cur, mems[i])

            # Global average pool over W: (B, C_out, W) -> (B, C_out)
            pooled = cur.mean(dim=-1)
            spike_rec.append(pooled)

        # Aggregate over time: mean
        spike_out = torch.stack(spike_rec)  # (T, B, last_channel)
        agg = spike_out.mean(dim=0)  # (B, last_channel)

        # Linear readout
        return self.readout(agg)  # (B, 39) logits


MODEL_REGISTRY["spiking_cnn1d"] = SpikingCNN1D
