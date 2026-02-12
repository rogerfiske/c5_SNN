"""Tests for SNN models (STORY-4.2, STORY-4.3, STORY-5.1, STORY-6.1)."""

import pytest
import torch

from c5_snn.models.base import MODEL_REGISTRY, get_model
from c5_snn.models.snn_models import SpikingCNN1D, SpikingMLP
from c5_snn.models.snn_phase_b import SpikeGRU
from c5_snn.models.snn_phase_c import SpikingTransformer
from c5_snn.utils.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    hidden_sizes=None,
    beta=0.95,
    encoding="direct",
    timesteps=10,
    window_size=21,
) -> dict:
    """Create a minimal config dict for SpikingMLP."""
    cfg = {
        "model": {
            "type": "spiking_mlp",
            "encoding": encoding,
            "timesteps": timesteps,
            "beta": beta,
        },
        "data": {
            "window_size": window_size,
        },
    }
    if hidden_sizes is not None:
        cfg["model"]["hidden_sizes"] = hidden_sizes
    return cfg


def _binary_input(batch=4, window=21, features=39) -> torch.Tensor:
    """Create a random binary tensor simulating windowed CA5 data."""
    return (torch.rand(batch, window, features) > 0.5).float()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSpikingMLPConstruction:
    """Verify SpikingMLP construction and config parsing."""

    def test_default_config(self):
        """Default hidden_sizes is [256, 128]."""
        model = SpikingMLP(_make_config())
        assert model.hidden_sizes == [256, 128]

    def test_custom_hidden_sizes(self):
        """Custom hidden_sizes from config."""
        model = SpikingMLP(_make_config(hidden_sizes=[512, 256, 128]))
        assert model.hidden_sizes == [512, 256, 128]

    def test_single_hidden_layer(self):
        """Single hidden layer configuration."""
        model = SpikingMLP(_make_config(hidden_sizes=[256]))
        assert model.hidden_sizes == [256]
        assert len(model.fc_layers) == 1
        assert len(model.lif_layers) == 1

    def test_is_base_model(self):
        """SpikingMLP subclasses BaseModel."""
        from c5_snn.models.base import BaseModel

        model = SpikingMLP(_make_config())
        assert isinstance(model, BaseModel)

    def test_is_nn_module(self):
        """SpikingMLP is an nn.Module."""
        model = SpikingMLP(_make_config())
        assert isinstance(model, torch.nn.Module)

    def test_input_size_default_window(self):
        """Input size is W * 39 for default window_size=21."""
        model = SpikingMLP(_make_config())
        assert model.input_size == 21 * 39

    def test_input_size_custom_window(self):
        """Input size adapts to window_size config."""
        model = SpikingMLP(_make_config(window_size=7))
        assert model.input_size == 7 * 39

    def test_layer_counts(self):
        """Number of FC and LIF layers matches hidden_sizes length."""
        model = SpikingMLP(_make_config(hidden_sizes=[256, 128]))
        assert len(model.fc_layers) == 2
        assert len(model.lif_layers) == 2

    def test_readout_layer(self):
        """Readout layer projects from last hidden to 39."""
        model = SpikingMLP(_make_config(hidden_sizes=[256, 128]))
        assert model.readout.in_features == 128
        assert model.readout.out_features == 39

    def test_readout_single_layer(self):
        """Readout with single hidden layer."""
        model = SpikingMLP(_make_config(hidden_sizes=[512]))
        assert model.readout.in_features == 512
        assert model.readout.out_features == 39

    def test_encoding_mode_stored(self):
        """Encoder encoding mode matches config."""
        model = SpikingMLP(_make_config(encoding="direct"))
        assert model.encoder.encoding == "direct"

    def test_rate_coded_encoding(self):
        """Rate-coded encoding is properly configured."""
        model = SpikingMLP(_make_config(encoding="rate_coded", timesteps=5))
        assert model.encoder.encoding == "rate_coded"
        assert model.encoder.timesteps == 5


# ---------------------------------------------------------------------------
# Forward pass — shape correctness
# ---------------------------------------------------------------------------


class TestSpikingMLPForward:
    """Verify SpikingMLP forward pass shapes."""

    def test_forward_shape_default(self):
        """Forward: (4, 21, 39) -> (4, 39)."""
        model = SpikingMLP(_make_config())
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_batch_1(self):
        """Forward: (1, 21, 39) -> (1, 39)."""
        model = SpikingMLP(_make_config())
        x = _binary_input(batch=1, window=21)
        out = model(x)
        assert out.shape == (1, 39)

    def test_forward_shape_large_batch(self):
        """Forward: (64, 21, 39) -> (64, 39)."""
        model = SpikingMLP(_make_config())
        x = _binary_input(batch=64, window=21)
        out = model(x)
        assert out.shape == (64, 39)

    def test_forward_shape_window_7(self):
        """Forward with W=7: (4, 7, 39) -> (4, 39)."""
        model = SpikingMLP(_make_config(window_size=7))
        x = _binary_input(batch=4, window=7)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_rate_coded(self):
        """Forward with rate_coded encoding: (4, 21, 39) -> (4, 39)."""
        model = SpikingMLP(_make_config(encoding="rate_coded", timesteps=10))
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_single_hidden(self):
        """Forward with single hidden layer: (4, 21, 39) -> (4, 39)."""
        model = SpikingMLP(_make_config(hidden_sizes=[256]))
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_three_hidden(self):
        """Forward with three hidden layers: (4, 21, 39) -> (4, 39)."""
        model = SpikingMLP(_make_config(hidden_sizes=[256, 128, 64]))
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_output_dtype(self):
        """Output dtype is float32."""
        model = SpikingMLP(_make_config())
        x = _binary_input()
        out = model(x)
        assert out.dtype == torch.float32

    def test_forward_all_zeros(self):
        """All-zeros input produces output (near bias values)."""
        model = SpikingMLP(_make_config())
        x = torch.zeros(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)
        # Output should be near readout bias (LIF neurons never spike)
        assert torch.isfinite(out).all()

    def test_forward_all_ones(self):
        """All-ones input produces valid output."""
        model = SpikingMLP(_make_config())
        x = torch.ones(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Backward pass — surrogate gradients
# ---------------------------------------------------------------------------


class TestSpikingMLPBackward:
    """Verify backward pass through surrogate gradients."""

    def test_backward_completes(self):
        """Backward pass completes without error (surrogate gradients)."""
        model = SpikingMLP(_make_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()
        # If we get here, surrogate gradients worked

    def test_backward_with_bce_loss(self):
        """Backward with BCEWithLogitsLoss (actual training loss)."""
        model = SpikingMLP(_make_config())
        x = _binary_input()
        target = (torch.rand(4, 39) > 0.5).float()  # (batch, 39)
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out, target
        )
        loss.backward()

    def test_gradients_exist(self):
        """FC layers have gradients after backward."""
        model = SpikingMLP(_make_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()
        for fc in model.fc_layers:
            assert fc.weight.grad is not None
            assert fc.weight.grad.abs().sum() > 0
        assert model.readout.weight.grad is not None

    def test_backward_rate_coded(self):
        """Backward completes with rate_coded encoding."""
        model = SpikingMLP(_make_config(encoding="rate_coded", timesteps=5))
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()


# ---------------------------------------------------------------------------
# Learnable parameters
# ---------------------------------------------------------------------------


class TestSpikingMLPParameters:
    """Verify SpikingMLP has learnable parameters."""

    def test_has_parameters(self):
        """Model has learnable parameters."""
        model = SpikingMLP(_make_config())
        params = list(model.parameters())
        assert len(params) > 0

    def test_parameter_count_reasonable(self):
        """Parameter count matches expected architecture."""
        model = SpikingMLP(_make_config(hidden_sizes=[256, 128]))
        total_params = sum(p.numel() for p in model.parameters())
        # FC1: 819*256 + 256 = 209920 + 256
        # FC2: 256*128 + 128 = 32768 + 128
        # Readout: 128*39 + 39 = 4992 + 39
        expected = (819 * 256 + 256) + (256 * 128 + 128) + (128 * 39 + 39)
        assert total_params == expected

    def test_parameters_require_grad(self):
        """All parameters require gradient."""
        model = SpikingMLP(_make_config())
        for param in model.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# Membrane potential reset
# ---------------------------------------------------------------------------


class TestMembraneReset:
    """Verify membrane potentials are reset each forward call."""

    def test_consecutive_forwards_independent(self):
        """Two consecutive forward passes produce the same output for same input."""
        model = SpikingMLP(_make_config())
        model.eval()
        x = _binary_input()

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        torch.testing.assert_close(out1, out2)

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs with rate_coded encoding.

        With rate_coded (T>1), LIF neurons accumulate membrane potential
        across timesteps, so all-ones input (always spiking) produces
        different output than random binary input.
        """
        model = SpikingMLP(
            _make_config(encoding="rate_coded", timesteps=10)
        )
        model.eval()
        x1 = torch.zeros(4, 21, 39)
        x2 = _binary_input(batch=4, window=21)

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        assert not torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


class TestSpikingMLPRegistry:
    """Verify SpikingMLP is registered in MODEL_REGISTRY."""

    def test_registered(self):
        """'spiking_mlp' is in MODEL_REGISTRY."""
        assert "spiking_mlp" in MODEL_REGISTRY

    def test_registry_class(self):
        """Registry entry is SpikingMLP class."""
        assert MODEL_REGISTRY["spiking_mlp"] is SpikingMLP

    def test_get_model_creates_instance(self):
        """get_model with spiking_mlp config creates SpikingMLP."""
        config = _make_config()
        model = get_model(config)
        assert isinstance(model, SpikingMLP)

    def test_get_model_custom_config(self):
        """get_model passes config to SpikingMLP."""
        config = _make_config(hidden_sizes=[512, 256])
        model = get_model(config)
        assert model.hidden_sizes == [512, 256]

    def test_exported_from_models(self):
        """SpikingMLP is accessible from c5_snn.models."""
        from c5_snn.models import SpikingMLP as Mlp

        assert Mlp is SpikingMLP


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------


class TestConfigFile:
    """Verify config file can create SpikingMLP."""

    def test_load_snn_phase_a_config(self):
        """Load configs/snn_phase_a_mlp.yaml and create model."""
        import yaml

        with open("configs/snn_phase_a_mlp.yaml") as f:
            config = yaml.safe_load(f)

        model = get_model(config)
        assert isinstance(model, SpikingMLP)
        assert model.hidden_sizes == [256, 128]

    def test_config_forward_shape(self):
        """Model from config produces correct output shape."""
        import yaml

        with open("configs/snn_phase_a_mlp.yaml") as f:
            config = yaml.safe_load(f)

        model = get_model(config)
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)


# ===========================================================================
# SpikingCNN1D tests (STORY-4.3)
# ===========================================================================


def _make_cnn_config(
    channels=None,
    kernel_sizes=None,
    beta=0.95,
    encoding="direct",
    timesteps=10,
    window_size=21,
) -> dict:
    """Create a minimal config dict for SpikingCNN1D."""
    cfg = {
        "model": {
            "type": "spiking_cnn1d",
            "encoding": encoding,
            "timesteps": timesteps,
            "beta": beta,
        },
        "data": {
            "window_size": window_size,
        },
    }
    if channels is not None:
        cfg["model"]["channels"] = channels
    if kernel_sizes is not None:
        cfg["model"]["kernel_sizes"] = kernel_sizes
    return cfg


# ---------------------------------------------------------------------------
# SpikingCNN1D — Construction
# ---------------------------------------------------------------------------


class TestSpikingCNN1DConstruction:
    """Verify SpikingCNN1D construction and config parsing."""

    def test_default_channels(self):
        """Default channels is [64, 64]."""
        model = SpikingCNN1D(_make_cnn_config())
        assert model.channels == [64, 64]

    def test_default_kernel_sizes(self):
        """Default kernel_sizes is [3, 3]."""
        model = SpikingCNN1D(_make_cnn_config())
        assert model.kernel_sizes == [3, 3]

    def test_custom_channels(self):
        """Custom channels from config."""
        model = SpikingCNN1D(
            _make_cnn_config(channels=[32, 64, 128], kernel_sizes=[3, 3, 3])
        )
        assert model.channels == [32, 64, 128]

    def test_single_conv_layer(self):
        """Single conv layer configuration."""
        model = SpikingCNN1D(
            _make_cnn_config(channels=[64], kernel_sizes=[3])
        )
        assert model.channels == [64]
        assert len(model.conv_layers) == 1
        assert len(model.lif_layers) == 1

    def test_is_base_model(self):
        """SpikingCNN1D subclasses BaseModel."""
        from c5_snn.models.base import BaseModel

        model = SpikingCNN1D(_make_cnn_config())
        assert isinstance(model, BaseModel)

    def test_is_nn_module(self):
        """SpikingCNN1D is an nn.Module."""
        model = SpikingCNN1D(_make_cnn_config())
        assert isinstance(model, torch.nn.Module)

    def test_layer_counts(self):
        """Number of conv and LIF layers matches channels length."""
        model = SpikingCNN1D(_make_cnn_config(channels=[64, 64]))
        assert len(model.conv_layers) == 2
        assert len(model.lif_layers) == 2

    def test_readout_layer(self):
        """Readout layer projects from last channel to 39."""
        model = SpikingCNN1D(_make_cnn_config(channels=[64, 128]))
        assert model.readout.in_features == 128
        assert model.readout.out_features == 39

    def test_readout_single_layer(self):
        """Readout with single conv layer."""
        model = SpikingCNN1D(
            _make_cnn_config(channels=[32], kernel_sizes=[5])
        )
        assert model.readout.in_features == 32
        assert model.readout.out_features == 39

    def test_conv_in_channels(self):
        """First conv layer has 39 input channels."""
        model = SpikingCNN1D(_make_cnn_config())
        assert model.conv_layers[0].in_channels == 39

    def test_conv_padding_preserves_w(self):
        """Conv1d padding is kernel_size // 2 (same padding)."""
        model = SpikingCNN1D(
            _make_cnn_config(channels=[64], kernel_sizes=[5])
        )
        assert model.conv_layers[0].padding == (2,)  # 5 // 2 = 2

    def test_encoding_mode_stored(self):
        """Encoder encoding mode matches config."""
        model = SpikingCNN1D(_make_cnn_config(encoding="rate_coded"))
        assert model.encoder.encoding == "rate_coded"

    def test_window_size_stored(self):
        """Window size is stored for membrane init."""
        model = SpikingCNN1D(_make_cnn_config(window_size=7))
        assert model.window_size == 7


# ---------------------------------------------------------------------------
# SpikingCNN1D — Forward pass
# ---------------------------------------------------------------------------


class TestSpikingCNN1DForward:
    """Verify SpikingCNN1D forward pass shapes."""

    def test_forward_shape_default(self):
        """Forward: (4, 21, 39) -> (4, 39)."""
        model = SpikingCNN1D(_make_cnn_config())
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_batch_1(self):
        """Forward: (1, 21, 39) -> (1, 39)."""
        model = SpikingCNN1D(_make_cnn_config())
        x = _binary_input(batch=1, window=21)
        out = model(x)
        assert out.shape == (1, 39)

    def test_forward_shape_large_batch(self):
        """Forward: (64, 21, 39) -> (64, 39)."""
        model = SpikingCNN1D(_make_cnn_config())
        x = _binary_input(batch=64, window=21)
        out = model(x)
        assert out.shape == (64, 39)

    def test_forward_shape_window_7(self):
        """Forward with W=7: (4, 7, 39) -> (4, 39)."""
        model = SpikingCNN1D(_make_cnn_config(window_size=7))
        x = _binary_input(batch=4, window=7)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_rate_coded(self):
        """Forward with rate_coded encoding: (4, 21, 39) -> (4, 39)."""
        model = SpikingCNN1D(
            _make_cnn_config(encoding="rate_coded", timesteps=10)
        )
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_single_conv(self):
        """Forward with single conv layer: (4, 21, 39) -> (4, 39)."""
        model = SpikingCNN1D(
            _make_cnn_config(channels=[64], kernel_sizes=[3])
        )
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_three_conv(self):
        """Forward with three conv layers: (4, 21, 39) -> (4, 39)."""
        model = SpikingCNN1D(
            _make_cnn_config(
                channels=[32, 64, 128], kernel_sizes=[3, 3, 3]
            )
        )
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_kernel_5(self):
        """Forward with kernel_size=5: (4, 21, 39) -> (4, 39)."""
        model = SpikingCNN1D(
            _make_cnn_config(channels=[64, 64], kernel_sizes=[5, 5])
        )
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_output_dtype(self):
        """Output dtype is float32."""
        model = SpikingCNN1D(_make_cnn_config())
        x = _binary_input()
        out = model(x)
        assert out.dtype == torch.float32

    def test_forward_all_zeros(self):
        """All-zeros input produces valid output."""
        model = SpikingCNN1D(_make_cnn_config())
        x = torch.zeros(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)
        assert torch.isfinite(out).all()

    def test_forward_all_ones(self):
        """All-ones input produces valid output."""
        model = SpikingCNN1D(_make_cnn_config())
        x = torch.ones(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SpikingCNN1D — Backward pass
# ---------------------------------------------------------------------------


class TestSpikingCNN1DBackward:
    """Verify backward pass through surrogate gradients."""

    def test_backward_completes(self):
        """Backward pass completes without error."""
        model = SpikingCNN1D(_make_cnn_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()

    def test_backward_with_bce_loss(self):
        """Backward with BCEWithLogitsLoss."""
        model = SpikingCNN1D(_make_cnn_config())
        x = _binary_input()
        target = (torch.rand(4, 39) > 0.5).float()
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out, target
        )
        loss.backward()

    def test_gradients_exist(self):
        """Conv layers have gradients after backward."""
        model = SpikingCNN1D(_make_cnn_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()
        for conv in model.conv_layers:
            assert conv.weight.grad is not None
            assert conv.weight.grad.abs().sum() > 0
        assert model.readout.weight.grad is not None

    def test_backward_rate_coded(self):
        """Backward completes with rate_coded encoding."""
        model = SpikingCNN1D(
            _make_cnn_config(encoding="rate_coded", timesteps=5)
        )
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()


# ---------------------------------------------------------------------------
# SpikingCNN1D — Learnable parameters
# ---------------------------------------------------------------------------


class TestSpikingCNN1DParameters:
    """Verify SpikingCNN1D has learnable parameters."""

    def test_has_parameters(self):
        """Model has learnable parameters."""
        model = SpikingCNN1D(_make_cnn_config())
        params = list(model.parameters())
        assert len(params) > 0

    def test_parameter_count_reasonable(self):
        """Parameter count matches expected architecture."""
        model = SpikingCNN1D(
            _make_cnn_config(channels=[64, 64], kernel_sizes=[3, 3])
        )
        total_params = sum(p.numel() for p in model.parameters())
        # Conv1: in=39, out=64, kernel=3 -> 39*64*3 + 64 = 7488 + 64
        # Conv2: in=64, out=64, kernel=3 -> 64*64*3 + 64 = 12288 + 64
        # Readout: 64*39 + 39 = 2496 + 39
        expected = (39 * 64 * 3 + 64) + (64 * 64 * 3 + 64) + (64 * 39 + 39)
        assert total_params == expected

    def test_parameters_require_grad(self):
        """All parameters require gradient."""
        model = SpikingCNN1D(_make_cnn_config())
        for param in model.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# SpikingCNN1D — Membrane potential reset
# ---------------------------------------------------------------------------


class TestSpikingCNN1DMembraneReset:
    """Verify membrane potentials are reset each forward call."""

    def test_consecutive_forwards_independent(self):
        """Two consecutive forward passes produce the same output."""
        model = SpikingCNN1D(_make_cnn_config())
        model.eval()
        x = _binary_input()

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        torch.testing.assert_close(out1, out2)

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs with rate_coded."""
        model = SpikingCNN1D(
            _make_cnn_config(encoding="rate_coded", timesteps=10)
        )
        model.eval()
        x1 = torch.zeros(4, 21, 39)
        x2 = _binary_input(batch=4, window=21)

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        assert not torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# SpikingCNN1D — Model registry
# ---------------------------------------------------------------------------


class TestSpikingCNN1DRegistry:
    """Verify SpikingCNN1D is registered in MODEL_REGISTRY."""

    def test_registered(self):
        """'spiking_cnn1d' is in MODEL_REGISTRY."""
        assert "spiking_cnn1d" in MODEL_REGISTRY

    def test_registry_class(self):
        """Registry entry is SpikingCNN1D class."""
        assert MODEL_REGISTRY["spiking_cnn1d"] is SpikingCNN1D

    def test_get_model_creates_instance(self):
        """get_model with spiking_cnn1d config creates SpikingCNN1D."""
        config = _make_cnn_config()
        model = get_model(config)
        assert isinstance(model, SpikingCNN1D)

    def test_get_model_custom_config(self):
        """get_model passes config to SpikingCNN1D."""
        config = _make_cnn_config(channels=[32, 64, 128])
        model = get_model(config)
        assert model.channels == [32, 64, 128]

    def test_exported_from_models(self):
        """SpikingCNN1D is accessible from c5_snn.models."""
        from c5_snn.models import SpikingCNN1D as Cnn

        assert Cnn is SpikingCNN1D


# ---------------------------------------------------------------------------
# SpikingCNN1D — Config file loading
# ---------------------------------------------------------------------------


class TestCNNConfigFile:
    """Verify config file can create SpikingCNN1D."""

    def test_load_snn_phase_a_cnn_config(self):
        """Load configs/snn_phase_a_cnn.yaml and create model."""
        import yaml

        with open("configs/snn_phase_a_cnn.yaml") as f:
            config = yaml.safe_load(f)

        model = get_model(config)
        assert isinstance(model, SpikingCNN1D)
        assert model.channels == [64, 64]
        assert model.kernel_sizes == [3, 3]

    def test_cnn_config_forward_shape(self):
        """Model from config produces correct output shape."""
        import yaml

        with open("configs/snn_phase_a_cnn.yaml") as f:
            config = yaml.safe_load(f)

        model = get_model(config)
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)


# ===========================================================================
# SpikeGRU tests (STORY-5.1)
# ===========================================================================


def _make_gru_config(
    hidden_size=128,
    num_layers=1,
    beta=0.95,
    dropout=0.0,
    encoding="direct",
    timesteps=10,
    window_size=21,
) -> dict:
    """Create a minimal config dict for SpikeGRU."""
    return {
        "model": {
            "type": "spike_gru",
            "encoding": encoding,
            "timesteps": timesteps,
            "beta": beta,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        },
        "data": {
            "window_size": window_size,
        },
    }


# ---------------------------------------------------------------------------
# SpikeGRU — Construction
# ---------------------------------------------------------------------------


class TestSpikeGRUConstruction:
    """Verify SpikeGRU construction and config parsing."""

    def test_default_hidden_size(self):
        """Default hidden_size is 128."""
        model = SpikeGRU(_make_gru_config())
        assert model.hidden_size == 128

    def test_custom_hidden_size(self):
        """Custom hidden_size from config."""
        model = SpikeGRU(_make_gru_config(hidden_size=256))
        assert model.hidden_size == 256

    def test_default_num_layers(self):
        """Default num_layers is 1."""
        model = SpikeGRU(_make_gru_config())
        assert model.num_layers == 1

    def test_custom_num_layers(self):
        """Custom num_layers from config."""
        model = SpikeGRU(_make_gru_config(num_layers=2))
        assert model.num_layers == 2
        assert len(model.rlif_layers) == 2

    def test_default_beta(self):
        """Default beta is 0.95."""
        model = SpikeGRU(_make_gru_config())
        assert model.beta == 0.95

    def test_custom_beta(self):
        """Custom beta from config."""
        model = SpikeGRU(_make_gru_config(beta=0.80))
        assert model.beta == 0.80

    def test_is_base_model(self):
        """SpikeGRU subclasses BaseModel."""
        from c5_snn.models.base import BaseModel

        model = SpikeGRU(_make_gru_config())
        assert isinstance(model, BaseModel)

    def test_is_nn_module(self):
        """SpikeGRU is an nn.Module."""
        model = SpikeGRU(_make_gru_config())
        assert isinstance(model, torch.nn.Module)

    def test_rlif_layer_count(self):
        """Number of RLeaky layers matches num_layers."""
        model = SpikeGRU(_make_gru_config(num_layers=3))
        assert len(model.rlif_layers) == 3

    def test_input_projection(self):
        """Input projection maps 39 features to hidden_size."""
        model = SpikeGRU(_make_gru_config(hidden_size=64))
        assert model.fc_input.in_features == 39
        assert model.fc_input.out_features == 64

    def test_readout_layer(self):
        """Readout layer projects from hidden_size to 39."""
        model = SpikeGRU(_make_gru_config(hidden_size=64))
        assert model.readout.in_features == 64
        assert model.readout.out_features == 39

    def test_encoding_mode_stored(self):
        """Encoder encoding mode matches config."""
        model = SpikeGRU(_make_gru_config(encoding="rate_coded"))
        assert model.encoder.encoding == "rate_coded"

    def test_dropout_none_single_layer(self):
        """Dropout is None for single-layer models."""
        model = SpikeGRU(_make_gru_config(num_layers=1, dropout=0.1))
        assert model.dropout is None

    def test_dropout_active_multi_layer(self):
        """Dropout is active for multi-layer models with dropout > 0."""
        model = SpikeGRU(_make_gru_config(num_layers=2, dropout=0.1))
        assert model.dropout is not None

    def test_no_dropout_when_zero(self):
        """Dropout is None when rate is 0.0 even with multi-layer."""
        model = SpikeGRU(_make_gru_config(num_layers=2, dropout=0.0))
        assert model.dropout is None


# ---------------------------------------------------------------------------
# SpikeGRU — Forward pass
# ---------------------------------------------------------------------------


class TestSpikeGRUForward:
    """Verify SpikeGRU forward pass shapes."""

    def test_forward_shape_default(self):
        """Forward: (4, 21, 39) -> (4, 39)."""
        model = SpikeGRU(_make_gru_config())
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_batch_1(self):
        """Forward: (1, 21, 39) -> (1, 39)."""
        model = SpikeGRU(_make_gru_config())
        x = _binary_input(batch=1, window=21)
        out = model(x)
        assert out.shape == (1, 39)

    def test_forward_shape_large_batch(self):
        """Forward: (64, 21, 39) -> (64, 39)."""
        model = SpikeGRU(_make_gru_config())
        x = _binary_input(batch=64, window=21)
        out = model(x)
        assert out.shape == (64, 39)

    def test_forward_shape_window_7(self):
        """Forward with W=7: (4, 7, 39) -> (4, 39)."""
        model = SpikeGRU(_make_gru_config(window_size=7))
        x = _binary_input(batch=4, window=7)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_rate_coded(self):
        """Forward with rate_coded encoding: (4, 21, 39) -> (4, 39)."""
        model = SpikeGRU(
            _make_gru_config(encoding="rate_coded", timesteps=5)
        )
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_multi_layer(self):
        """Forward with 2 layers: (4, 21, 39) -> (4, 39)."""
        model = SpikeGRU(_make_gru_config(num_layers=2))
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_multi_layer_rate_coded(self):
        """Forward with 2 layers + rate_coded: (4, 21, 39) -> (4, 39)."""
        model = SpikeGRU(
            _make_gru_config(
                num_layers=2, encoding="rate_coded", timesteps=3
            )
        )
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_small_hidden(self):
        """Forward with small hidden_size=32: (4, 21, 39) -> (4, 39)."""
        model = SpikeGRU(_make_gru_config(hidden_size=32))
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_large_hidden(self):
        """Forward with large hidden_size=256: (4, 21, 39) -> (4, 39)."""
        model = SpikeGRU(_make_gru_config(hidden_size=256))
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_output_dtype(self):
        """Output dtype is float32."""
        model = SpikeGRU(_make_gru_config())
        x = _binary_input()
        out = model(x)
        assert out.dtype == torch.float32

    def test_forward_all_zeros(self):
        """All-zeros input produces valid output."""
        model = SpikeGRU(_make_gru_config())
        x = torch.zeros(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)
        assert torch.isfinite(out).all()

    def test_forward_all_ones(self):
        """All-ones input produces valid output."""
        model = SpikeGRU(_make_gru_config())
        x = torch.ones(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SpikeGRU — Backward pass
# ---------------------------------------------------------------------------


class TestSpikeGRUBackward:
    """Verify backward pass through surrogate gradients."""

    def test_backward_completes(self):
        """Backward pass completes without error."""
        model = SpikeGRU(_make_gru_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()

    def test_backward_with_bce_loss(self):
        """Backward with BCEWithLogitsLoss."""
        model = SpikeGRU(_make_gru_config())
        x = _binary_input()
        target = (torch.rand(4, 39) > 0.5).float()
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out, target
        )
        loss.backward()

    def test_gradients_exist(self):
        """Input projection and readout have gradients after backward."""
        model = SpikeGRU(_make_gru_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.fc_input.weight.grad is not None
        assert model.fc_input.weight.grad.abs().sum() > 0
        assert model.readout.weight.grad is not None

    def test_backward_rate_coded(self):
        """Backward completes with rate_coded encoding."""
        model = SpikeGRU(
            _make_gru_config(encoding="rate_coded", timesteps=3)
        )
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()

    def test_backward_multi_layer(self):
        """Backward completes with multi-layer model."""
        model = SpikeGRU(_make_gru_config(num_layers=2, dropout=0.1))
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()


# ---------------------------------------------------------------------------
# SpikeGRU — Learnable parameters
# ---------------------------------------------------------------------------


class TestSpikeGRUParameters:
    """Verify SpikeGRU has learnable parameters."""

    def test_has_parameters(self):
        """Model has learnable parameters."""
        model = SpikeGRU(_make_gru_config())
        params = list(model.parameters())
        assert len(params) > 0

    def test_parameters_require_grad(self):
        """All parameters require gradient."""
        model = SpikeGRU(_make_gru_config())
        for param in model.parameters():
            assert param.requires_grad

    def test_parameter_count_includes_recurrent(self):
        """Parameter count includes recurrent weights from RLeaky."""
        model = SpikeGRU(_make_gru_config(hidden_size=64))
        total_params = sum(p.numel() for p in model.parameters())
        # fc_input: 39*64 + 64 = 2560
        # RLeaky recurrent: 64*64 = 4096 (V matrix, all_to_all)
        # readout: 64*39 + 39 = 2535
        # Total minimum: 2560 + 4096 + 2535 = 9191
        assert total_params >= 9000


# ---------------------------------------------------------------------------
# SpikeGRU — Membrane potential reset
# ---------------------------------------------------------------------------


class TestSpikeGRUMembraneReset:
    """Verify membrane potentials are reset each forward call."""

    def test_consecutive_forwards_independent(self):
        """Two consecutive forward passes produce the same output."""
        model = SpikeGRU(_make_gru_config())
        model.eval()
        x = _binary_input()

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        torch.testing.assert_close(out1, out2)

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs."""
        model = SpikeGRU(_make_gru_config())
        model.eval()
        x1 = torch.zeros(4, 21, 39)
        x2 = _binary_input(batch=4, window=21)

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        assert not torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# SpikeGRU — Model registry
# ---------------------------------------------------------------------------


class TestSpikeGRURegistry:
    """Verify SpikeGRU is registered in MODEL_REGISTRY."""

    def test_registered(self):
        """'spike_gru' is in MODEL_REGISTRY."""
        assert "spike_gru" in MODEL_REGISTRY

    def test_registry_class(self):
        """Registry entry is SpikeGRU class."""
        assert MODEL_REGISTRY["spike_gru"] is SpikeGRU

    def test_get_model_creates_instance(self):
        """get_model with spike_gru config creates SpikeGRU."""
        config = _make_gru_config()
        model = get_model(config)
        assert isinstance(model, SpikeGRU)

    def test_get_model_custom_config(self):
        """get_model passes config to SpikeGRU."""
        config = _make_gru_config(hidden_size=256)
        model = get_model(config)
        assert model.hidden_size == 256

    def test_exported_from_models(self):
        """SpikeGRU is accessible from c5_snn.models."""
        from c5_snn.models import SpikeGRU as Gru

        assert Gru is SpikeGRU


# ---------------------------------------------------------------------------
# SpikeGRU — Config file loading
# ---------------------------------------------------------------------------


class TestSpikeGRUConfigFile:
    """Verify config file can create SpikeGRU."""

    def test_load_snn_phase_b_config(self):
        """Load configs/snn_phase_b.yaml and create model."""
        import yaml

        with open("configs/snn_phase_b.yaml") as f:
            config = yaml.safe_load(f)

        model = get_model(config)
        assert isinstance(model, SpikeGRU)
        assert model.hidden_size == 128
        assert model.num_layers == 1

    def test_config_forward_shape(self):
        """Model from config produces correct output shape."""
        import yaml

        with open("configs/snn_phase_b.yaml") as f:
            config = yaml.safe_load(f)

        model = get_model(config)
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)


# ===========================================================================
# SpikingTransformer tests (STORY-6.1)
# ===========================================================================


def _make_transformer_config(
    d_model=128,
    n_heads=4,
    n_layers=2,
    d_ffn=256,
    beta=0.95,
    dropout=0.1,
    max_window_size=100,
    encoding="direct",
    timesteps=10,
    window_size=21,
) -> dict:
    """Create a minimal config dict for SpikingTransformer."""
    return {
        "model": {
            "type": "spiking_transformer",
            "encoding": encoding,
            "timesteps": timesteps,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "d_ffn": d_ffn,
            "beta": beta,
            "dropout": dropout,
            "max_window_size": max_window_size,
        },
        "data": {
            "window_size": window_size,
        },
    }


# ---------------------------------------------------------------------------
# SpikingTransformer — Construction
# ---------------------------------------------------------------------------


class TestSpikingTransformerConstruction:
    """Verify SpikingTransformer construction and config parsing."""

    def test_default_d_model(self):
        """Default d_model is 128."""
        model = SpikingTransformer(_make_transformer_config())
        assert model.d_model == 128

    def test_custom_d_model(self):
        """Custom d_model from config."""
        model = SpikingTransformer(_make_transformer_config(d_model=64))
        assert model.d_model == 64

    def test_default_n_heads(self):
        """Default n_heads is 4."""
        model = SpikingTransformer(_make_transformer_config())
        assert model.n_heads == 4

    def test_default_n_layers(self):
        """Default n_layers is 2."""
        model = SpikingTransformer(_make_transformer_config())
        assert model.n_layers == 2

    def test_custom_n_layers(self):
        """Custom n_layers from config."""
        model = SpikingTransformer(_make_transformer_config(n_layers=3))
        assert model.n_layers == 3
        assert len(model.blocks) == 3

    def test_default_d_ffn(self):
        """Default d_ffn is 256."""
        model = SpikingTransformer(_make_transformer_config())
        assert model.d_ffn == 256

    def test_default_beta(self):
        """Default beta is 0.95."""
        model = SpikingTransformer(_make_transformer_config())
        assert model.beta == 0.95

    def test_default_max_window_size(self):
        """Default max_window_size is 100."""
        model = SpikingTransformer(_make_transformer_config())
        assert model.max_window_size == 100

    def test_is_base_model(self):
        """SpikingTransformer subclasses BaseModel."""
        from c5_snn.models.base import BaseModel

        model = SpikingTransformer(_make_transformer_config())
        assert isinstance(model, BaseModel)

    def test_is_nn_module(self):
        """SpikingTransformer is an nn.Module."""
        model = SpikingTransformer(_make_transformer_config())
        assert isinstance(model, torch.nn.Module)

    def test_block_count(self):
        """Number of transformer blocks matches n_layers."""
        model = SpikingTransformer(_make_transformer_config(n_layers=3))
        assert len(model.blocks) == 3

    def test_input_projection(self):
        """Input projection maps 39 features to d_model."""
        model = SpikingTransformer(_make_transformer_config(d_model=64))
        assert model.input_proj.in_features == 39
        assert model.input_proj.out_features == 64

    def test_readout_layer(self):
        """Readout layer projects from d_model to 39."""
        model = SpikingTransformer(_make_transformer_config(d_model=64))
        assert model.readout.in_features == 64
        assert model.readout.out_features == 39

    def test_pos_embed_shape(self):
        """Positional embedding has shape (1, max_window_size, d_model)."""
        model = SpikingTransformer(
            _make_transformer_config(d_model=64, max_window_size=100)
        )
        assert model.pos_embed.shape == (1, 100, 64)

    def test_encoding_mode_stored(self):
        """Encoder encoding mode matches config."""
        model = SpikingTransformer(
            _make_transformer_config(encoding="rate_coded")
        )
        assert model.encoder.encoding == "rate_coded"

    def test_d_model_not_divisible_by_n_heads_raises(self):
        """ConfigError raised when d_model not divisible by n_heads."""
        with pytest.raises(ConfigError, match="divisible"):
            SpikingTransformer(
                _make_transformer_config(d_model=100, n_heads=3)
            )


# ---------------------------------------------------------------------------
# SpikingTransformer — Forward pass
# ---------------------------------------------------------------------------


class TestSpikingTransformerForward:
    """Verify SpikingTransformer forward pass shapes."""

    def test_forward_shape_default(self):
        """Forward: (4, 21, 39) -> (4, 39)."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_batch_1(self):
        """Forward: (1, 21, 39) -> (1, 39)."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=1, window=21)
        out = model(x)
        assert out.shape == (1, 39)

    def test_forward_shape_large_batch(self):
        """Forward: (64, 21, 39) -> (64, 39)."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=64, window=21)
        out = model(x)
        assert out.shape == (64, 39)

    def test_forward_shape_window_7(self):
        """Forward with W=7: (4, 7, 39) -> (4, 39)."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=7)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_window_60(self):
        """Forward with W=60: (4, 60, 39) -> (4, 39)."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=60)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_window_90(self):
        """Forward with W=90: (4, 90, 39) -> (4, 39)."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=90)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_rate_coded(self):
        """Forward with rate_coded encoding: (4, 21, 39) -> (4, 39)."""
        model = SpikingTransformer(
            _make_transformer_config(encoding="rate_coded", timesteps=5)
        )
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_single_layer(self):
        """Forward with single transformer layer: (4, 21, 39) -> (4, 39)."""
        model = SpikingTransformer(_make_transformer_config(n_layers=1))
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_shape_small_d_model(self):
        """Forward with d_model=32: (4, 21, 39) -> (4, 39)."""
        model = SpikingTransformer(
            _make_transformer_config(d_model=32, n_heads=4, d_ffn=64)
        )
        x = _binary_input(batch=4, window=21)
        out = model(x)
        assert out.shape == (4, 39)

    def test_forward_output_dtype(self):
        """Output dtype is float32."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input()
        out = model(x)
        assert out.dtype == torch.float32

    def test_forward_all_zeros(self):
        """All-zeros input produces valid output."""
        model = SpikingTransformer(_make_transformer_config())
        x = torch.zeros(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)
        assert torch.isfinite(out).all()

    def test_forward_all_ones(self):
        """All-ones input produces valid output."""
        model = SpikingTransformer(_make_transformer_config())
        x = torch.ones(4, 21, 39)
        out = model(x)
        assert out.shape == (4, 39)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# SpikingTransformer — Backward pass
# ---------------------------------------------------------------------------


class TestSpikingTransformerBackward:
    """Verify backward pass through surrogate gradients."""

    def test_backward_completes(self):
        """Backward pass completes without error (surrogate gradients)."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()

    def test_backward_with_bce_loss(self):
        """Backward with BCEWithLogitsLoss (actual training loss)."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input()
        target = (torch.rand(4, 39) > 0.5).float()
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out, target
        )
        loss.backward()

    def test_gradients_exist(self):
        """Input projection and readout have gradients after backward."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.input_proj.weight.grad is not None
        assert model.input_proj.weight.grad.abs().sum() > 0
        assert model.readout.weight.grad is not None

    def test_pos_embed_has_grad(self):
        """Positional embedding gets gradients."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert model.pos_embed.grad is not None

    def test_backward_rate_coded(self):
        """Backward completes with rate_coded encoding."""
        model = SpikingTransformer(
            _make_transformer_config(encoding="rate_coded", timesteps=3)
        )
        x = _binary_input()
        out = model(x)
        loss = out.sum()
        loss.backward()


# ---------------------------------------------------------------------------
# SpikingTransformer — Learnable parameters
# ---------------------------------------------------------------------------


class TestSpikingTransformerParameters:
    """Verify SpikingTransformer has learnable parameters."""

    def test_has_parameters(self):
        """Model has learnable parameters."""
        model = SpikingTransformer(_make_transformer_config())
        params = list(model.parameters())
        assert len(params) > 0

    def test_parameters_require_grad(self):
        """All parameters require gradient."""
        model = SpikingTransformer(_make_transformer_config())
        for param in model.parameters():
            assert param.requires_grad

    def test_parameter_count_reasonable(self):
        """Parameter count is reasonable for the architecture."""
        model = SpikingTransformer(
            _make_transformer_config(d_model=64, n_heads=4, d_ffn=128)
        )
        total_params = sum(p.numel() for p in model.parameters())
        # input_proj: 39*64 + 64 = 2560
        # pos_embed: 100*64 = 6400
        # 2 blocks each with SSA (Q/K/V/O proj) + FFN
        # readout: 64*39 + 39 = 2535
        assert total_params > 10000


# ---------------------------------------------------------------------------
# SpikingTransformer — Membrane potential reset
# ---------------------------------------------------------------------------


class TestSpikingTransformerMembraneReset:
    """Verify membrane potentials are reset each forward call."""

    def test_consecutive_forwards_independent(self):
        """Two consecutive forward passes produce the same output."""
        model = SpikingTransformer(_make_transformer_config())
        model.eval()
        x = _binary_input()

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        torch.testing.assert_close(out1, out2)

    def test_different_inputs_different_outputs(self):
        """Different inputs produce different outputs with rate_coded."""
        model = SpikingTransformer(
            _make_transformer_config(encoding="rate_coded", timesteps=10)
        )
        model.eval()
        x1 = torch.zeros(4, 21, 39)
        x2 = _binary_input(batch=4, window=21)

        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)

        assert not torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# SpikingTransformer — Model registry
# ---------------------------------------------------------------------------


class TestSpikingTransformerRegistry:
    """Verify SpikingTransformer is registered in MODEL_REGISTRY."""

    def test_registered(self):
        """'spiking_transformer' is in MODEL_REGISTRY."""
        assert "spiking_transformer" in MODEL_REGISTRY

    def test_registry_class(self):
        """Registry entry is SpikingTransformer class."""
        assert MODEL_REGISTRY["spiking_transformer"] is SpikingTransformer

    def test_get_model_creates_instance(self):
        """get_model with spiking_transformer config creates instance."""
        config = _make_transformer_config()
        model = get_model(config)
        assert isinstance(model, SpikingTransformer)

    def test_get_model_custom_config(self):
        """get_model passes config to SpikingTransformer."""
        config = _make_transformer_config(d_model=64, n_heads=4)
        model = get_model(config)
        assert model.d_model == 64

    def test_exported_from_models(self):
        """SpikingTransformer is accessible from c5_snn.models."""
        from c5_snn.models import SpikingTransformer as Txf

        assert Txf is SpikingTransformer


# ---------------------------------------------------------------------------
# SpikingTransformer — Config file loading
# ---------------------------------------------------------------------------


class TestSpikingTransformerConfigFile:
    """Verify config file can create SpikingTransformer."""

    def test_load_snn_phase_c_config(self):
        """Load configs/snn_phase_c.yaml and create model."""
        import yaml

        with open("configs/snn_phase_c.yaml") as f:
            config = yaml.safe_load(f)

        model = get_model(config)
        assert isinstance(model, SpikingTransformer)
        assert model.d_model == 64
        assert model.n_heads == 2
        assert model.n_layers == 6
        assert model.d_ffn == 128

    def test_config_forward_shape(self):
        """Model from config produces correct output shape."""
        import yaml

        with open("configs/snn_phase_c.yaml") as f:
            config = yaml.safe_load(f)

        model = get_model(config)
        x = _binary_input(batch=4, window=90)
        out = model(x)
        assert out.shape == (4, 39)


# ---------------------------------------------------------------------------
# SpikingTransformer — Variable window size
# ---------------------------------------------------------------------------


class TestSpikingTransformerVariableWindow:
    """Verify SpikingTransformer works with variable window sizes W=7-90."""

    def test_window_7(self):
        """W=7 produces correct output shape."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=7)
        assert model(x).shape == (4, 39)

    def test_window_14(self):
        """W=14 produces correct output shape."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=14)
        assert model(x).shape == (4, 39)

    def test_window_30(self):
        """W=30 produces correct output shape."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=30)
        assert model(x).shape == (4, 39)

    def test_window_60(self):
        """W=60 produces correct output shape."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=60)
        assert model(x).shape == (4, 39)

    def test_window_90(self):
        """W=90 produces correct output shape."""
        model = SpikingTransformer(_make_transformer_config())
        x = _binary_input(batch=4, window=90)
        assert model(x).shape == (4, 39)

    def test_same_model_different_windows(self):
        """Same model handles different W in sequence (no reinstantiation)."""
        model = SpikingTransformer(_make_transformer_config())
        model.eval()

        with torch.no_grad():
            out7 = model(_binary_input(batch=2, window=7))
            out21 = model(_binary_input(batch=2, window=21))
            out60 = model(_binary_input(batch=2, window=60))

        assert out7.shape == (2, 39)
        assert out21.shape == (2, 39)
        assert out60.shape == (2, 39)
