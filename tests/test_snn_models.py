"""Tests for SNN models (STORY-4.2)."""

import torch

from c5_snn.models.base import MODEL_REGISTRY, get_model
from c5_snn.models.snn_models import SpikingMLP

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
