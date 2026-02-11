"""Phase C Spiking Neural Network models (STORY-6.1).

SpikingTransformer: Spiking transformer with Spiking Self-Attention (SSA)
following the Spikformer design (Zhou et al., 2023). Uses spike-form Q/K/V
without softmax normalization, learnable positional encoding, and LIF neurons
throughout.
"""

import logging

import snntorch
import torch
from snntorch import surrogate
from torch import nn

from c5_snn.models.base import MODEL_REGISTRY, BaseModel
from c5_snn.models.encoding import SpikeEncoder
from c5_snn.utils.exceptions import ConfigError

logger = logging.getLogger("c5_snn")

N_FEATURES = 39


class SpikingSelfAttention(nn.Module):
    """Spiking Self-Attention (SSA) module.

    Computes attention from spike-form Q, K, V matrices (linear projections
    through LIF neurons) with scaled dot-product but NO softmax. Multi-head
    support via reshape.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        beta: LIF membrane decay factor.
        dropout: Dropout rate for attention.
    """

    def __init__(
        self, d_model: int, n_heads: int, beta: float, dropout: float
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # LIF neurons for spike-form Q, K, V and output
        self.lif_q = snntorch.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_k = snntorch.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_v = snntorch.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_out = snntorch.Leaky(beta=beta, spike_grad=spike_grad)

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None

    def init_mem(
        self, batch: int, seq_len: int, d_model: int, device: torch.device
    ) -> tuple:
        """Initialize membrane potentials for Q, K, V, and output LIF."""
        shape = (batch, seq_len, d_model)
        mem_q = torch.zeros(shape, device=device)
        mem_k = torch.zeros(shape, device=device)
        mem_v = torch.zeros(shape, device=device)
        mem_out = torch.zeros(shape, device=device)
        return mem_q, mem_k, mem_v, mem_out

    def forward(
        self,
        x: torch.Tensor,
        mem_q: torch.Tensor,
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        mem_out: torch.Tensor,
    ) -> tuple:
        """Forward pass for one encoding timestep.

        Args:
            x: (batch, W, d_model) input for one encoding timestep.
            mem_q, mem_k, mem_v, mem_out: membrane potentials.

        Returns:
            output: (batch, W, d_model)
            mem_q, mem_k, mem_v, mem_out: updated membrane potentials.
        """
        B, W, D = x.shape

        # Project and pass through LIF neurons (spike-form Q, K, V)
        q, mem_q = self.lif_q(self.q_proj(x), mem_q)
        k, mem_k = self.lif_k(self.k_proj(x), mem_k)
        v, mem_v = self.lif_v(self.v_proj(x), mem_v)

        # Reshape for multi-head: (B, W, D) -> (B, n_heads, W, d_head)
        q = q.view(B, W, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, W, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, W, self.n_heads, self.d_head).transpose(1, 2)

        # Spike-form attention: NO softmax, just scaled dot-product
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)

        # Apply attention to values
        out = attn @ v

        # Reshape back: (B, n_heads, W, d_head) -> (B, W, D)
        out = out.transpose(1, 2).contiguous().view(B, W, D)

        # Output projection + LIF
        out, mem_out = self.lif_out(self.out_proj(out), mem_out)
        return out, mem_q, mem_k, mem_v, mem_out


class SpikingTransformerBlock(nn.Module):
    """Single spiking transformer block: SSA + spiking FFN with residuals.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ffn: Feed-forward hidden dimension.
        beta: LIF membrane decay factor.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        beta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # SSA sub-layer
        self.ssa = SpikingSelfAttention(d_model, n_heads, beta, dropout)

        # Residual LIF after SSA
        self.lif_res1 = snntorch.Leaky(beta=beta, spike_grad=spike_grad)

        # Spiking FFN: Linear -> LIF -> Linear -> LIF
        self.ffn_fc1 = nn.Linear(d_model, d_ffn)
        self.lif_ffn1 = snntorch.Leaky(beta=beta, spike_grad=spike_grad)
        self.ffn_fc2 = nn.Linear(d_ffn, d_model)
        self.lif_ffn2 = snntorch.Leaky(beta=beta, spike_grad=spike_grad)

        # Residual LIF after FFN
        self.lif_res2 = snntorch.Leaky(beta=beta, spike_grad=spike_grad)

        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.d_ffn = d_ffn

    def init_mem(
        self,
        batch: int,
        seq_len: int,
        d_model: int,
        d_ffn: int,
        device: torch.device,
    ) -> dict:
        """Initialize all membrane potentials for this block."""
        return {
            "ssa": self.ssa.init_mem(batch, seq_len, d_model, device),
            "res1": torch.zeros(batch, seq_len, d_model, device=device),
            "ffn1": torch.zeros(batch, seq_len, d_ffn, device=device),
            "ffn2": torch.zeros(batch, seq_len, d_model, device=device),
            "res2": torch.zeros(batch, seq_len, d_model, device=device),
        }

    def forward(self, x: torch.Tensor, mems: dict) -> tuple:
        """Forward pass for one encoding timestep.

        Args:
            x: (batch, W, d_model)
            mems: dict of membrane potentials from init_mem().

        Returns:
            output: (batch, W, d_model)
            mems: updated membrane potentials.
        """
        # SSA with residual
        mem_q, mem_k, mem_v, mem_out = mems["ssa"]
        ssa_out, mem_q, mem_k, mem_v, mem_out = self.ssa(
            x, mem_q, mem_k, mem_v, mem_out
        )
        mems["ssa"] = (mem_q, mem_k, mem_v, mem_out)

        # Residual connection + LIF
        res1, mems["res1"] = self.lif_res1(x + ssa_out, mems["res1"])

        # Spiking FFN
        ffn = self.ffn_fc1(res1)
        ffn, mems["ffn1"] = self.lif_ffn1(ffn, mems["ffn1"])
        if self.ffn_dropout is not None:
            ffn = self.ffn_dropout(ffn)
        ffn = self.ffn_fc2(ffn)
        ffn, mems["ffn2"] = self.lif_ffn2(ffn, mems["ffn2"])

        # Residual connection + LIF
        out, mems["res2"] = self.lif_res2(res1 + ffn, mems["res2"])

        return out, mems


class SpikingTransformer(BaseModel):
    """Spiking Transformer with SSA for CA5 prediction.

    Architecture:
        SpikeEncoder -> input projection (39 -> d_model) -> positional encoding
        -> N x SpikingTransformerBlock (SSA + spiking FFN)
        -> mean(T) -> mean(W) -> Linear -> (batch, 39)

    Follows the Spikformer design (Zhou et al., 2023) adapted for
    time-series multi-label prediction.

    Args:
        config: Experiment config dict. Reads from config["model"]:
            d_model (int): Model dimension. Default 128.
            n_heads (int): Number of attention heads. Default 4.
            n_layers (int): Number of transformer blocks. Default 2.
            d_ffn (int): FFN hidden dimension. Default 256.
            beta (float): LIF membrane decay factor. Default 0.95.
            dropout (float): Dropout rate. Default 0.1.
            max_window_size (int): Max window size for PE. Default 100.
            encoding (str): Spike encoding mode. Default "direct".
            timesteps (int): Time steps for rate_coded. Default 10.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config.get("model", {})

        # Encoding front-end
        self.encoder = SpikeEncoder(config)

        # Architecture params
        self.d_model = int(model_cfg.get("d_model", 128))
        self.n_heads = int(model_cfg.get("n_heads", 4))
        self.n_layers = int(model_cfg.get("n_layers", 2))
        self.d_ffn = int(model_cfg.get("d_ffn", 256))
        self.beta = float(model_cfg.get("beta", 0.95))
        self.dropout_rate = float(model_cfg.get("dropout", 0.1))
        self.max_window_size = int(model_cfg.get("max_window_size", 100))

        # Validate d_model divisible by n_heads
        if self.d_model % self.n_heads != 0:
            raise ConfigError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )

        # Input projection: 39 -> d_model
        self.input_proj = nn.Linear(N_FEATURES, self.d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.max_window_size, self.d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                SpikingTransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.d_ffn,
                    self.beta,
                    self.dropout_rate,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Readout layer
        self.readout = nn.Linear(self.d_model, N_FEATURES)

        logger.info(
            "SpikingTransformer: d_model=%d, n_heads=%d, n_layers=%d, "
            "d_ffn=%d, beta=%.3f, dropout=%.2f, max_W=%d, encoding=%s",
            self.d_model,
            self.n_heads,
            self.n_layers,
            self.d_ffn,
            self.beta,
            self.dropout_rate,
            self.max_window_size,
            self.encoder.encoding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking transformer.

        Args:
            x: (batch, W, 39) windowed multi-hot input.

        Returns:
            (batch, 39) logits for each part.
        """
        # Encode input into spike trains: (T, B, W, 39)
        spikes = self.encoder(x)
        T, B, W = spikes.size(0), spikes.size(1), spikes.size(2)

        # Initialize membrane potentials for all blocks
        block_mems = [
            block.init_mem(B, W, self.d_model, self.d_ffn, x.device)
            for block in self.blocks
        ]

        # Temporal loop
        t_records = []
        for t in range(T):
            # Project input features: (B, W, 39) -> (B, W, d_model)
            cur = self.input_proj(spikes[t])

            # Add positional encoding (sliced to actual W)
            cur = cur + self.pos_embed[:, :W, :]

            # Pass through transformer blocks
            for i, block in enumerate(self.blocks):
                cur, block_mems[i] = block(cur, block_mems[i])

            t_records.append(cur)  # (B, W, d_model)

        # Aggregate over T
        agg = torch.stack(t_records).mean(dim=0)  # (B, W, d_model)

        # Aggregate over W (mean pool)
        agg = agg.mean(dim=1)  # (B, d_model)

        # Linear readout
        return self.readout(agg)  # (B, 39)


MODEL_REGISTRY["spiking_transformer"] = SpikingTransformer
