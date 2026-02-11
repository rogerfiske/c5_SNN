"""Tests for spike encoding layer (STORY-4.1)."""

import pytest
import torch

from c5_snn.models.encoding import SpikeEncoder
from c5_snn.utils.exceptions import ConfigError
from c5_snn.utils.seed import set_global_seed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(encoding: str = "direct", timesteps: int = 10) -> dict:
    """Create a minimal config dict for SpikeEncoder."""
    return {"model": {"encoding": encoding, "timesteps": timesteps}}


def _binary_input(batch: int = 4, window: int = 21, features: int = 39) -> torch.Tensor:
    """Create a random binary tensor simulating windowed CA5 data."""
    set_global_seed(42)
    return (torch.rand(batch, window, features) > 0.5).float()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSpikeEncoderConstruction:
    """Verify SpikeEncoder construction and config parsing."""

    def test_default_config(self):
        """Default encoding is 'direct'."""
        encoder = SpikeEncoder({"model": {}})
        assert encoder.encoding == "direct"

    def test_direct_mode(self):
        """Explicit direct mode."""
        encoder = SpikeEncoder(_make_config("direct"))
        assert encoder.encoding == "direct"

    def test_rate_coded_mode(self):
        """Rate-coded mode with timesteps."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=5))
        assert encoder.encoding == "rate_coded"
        assert encoder.timesteps == 5

    def test_default_timesteps(self):
        """Default timesteps is 10."""
        encoder = SpikeEncoder(_make_config("rate_coded"))
        assert encoder.timesteps == 10

    def test_invalid_encoding_raises_config_error(self):
        """Invalid encoding mode raises ConfigError."""
        with pytest.raises(ConfigError, match="Unknown encoding mode"):
            SpikeEncoder(_make_config("latency"))

    def test_invalid_encoding_shows_valid_options(self):
        """Error message includes valid options."""
        with pytest.raises(ConfigError, match="direct"):
            SpikeEncoder(_make_config("invalid_mode"))

    def test_empty_config(self):
        """Empty config uses defaults."""
        encoder = SpikeEncoder({})
        assert encoder.encoding == "direct"
        assert encoder.timesteps == 10

    def test_is_nn_module(self):
        """SpikeEncoder is an nn.Module."""
        encoder = SpikeEncoder(_make_config())
        assert isinstance(encoder, torch.nn.Module)

    def test_no_learnable_parameters(self):
        """SpikeEncoder has no learnable parameters."""
        encoder = SpikeEncoder(_make_config("rate_coded"))
        assert len(list(encoder.parameters())) == 0


# ---------------------------------------------------------------------------
# Direct mode
# ---------------------------------------------------------------------------


class TestDirectMode:
    """Verify direct encoding mode."""

    def test_output_shape(self):
        """Direct mode: (batch, W, 39) -> (1, batch, W, 39)."""
        encoder = SpikeEncoder(_make_config("direct"))
        x = _binary_input(batch=4, window=21, features=39)
        out = encoder(x)
        assert out.shape == (1, 4, 21, 39)

    def test_output_shape_different_sizes(self):
        """Direct mode works with different batch/window sizes."""
        encoder = SpikeEncoder(_make_config("direct"))
        x = _binary_input(batch=8, window=7, features=39)
        out = encoder(x)
        assert out.shape == (1, 8, 7, 39)

    def test_values_unchanged(self):
        """Direct mode passes binary values through unchanged."""
        encoder = SpikeEncoder(_make_config("direct"))
        x = _binary_input()
        out = encoder(x)
        torch.testing.assert_close(out[0], x)

    def test_all_zeros(self):
        """All-zeros input -> all-zeros output."""
        encoder = SpikeEncoder(_make_config("direct"))
        x = torch.zeros(2, 5, 39)
        out = encoder(x)
        assert out.sum().item() == 0.0
        assert out.shape == (1, 2, 5, 39)

    def test_all_ones(self):
        """All-ones input -> all-ones output."""
        encoder = SpikeEncoder(_make_config("direct"))
        x = torch.ones(2, 5, 39)
        out = encoder(x)
        assert (out == 1.0).all()

    def test_deterministic_regardless_of_seed(self):
        """Direct mode is deterministic regardless of seed."""
        encoder = SpikeEncoder(_make_config("direct"))
        x = _binary_input()

        set_global_seed(1)
        out1 = encoder(x)

        set_global_seed(999)
        out2 = encoder(x)

        torch.testing.assert_close(out1, out2)

    def test_num_steps_property(self):
        """Direct mode num_steps is 1."""
        encoder = SpikeEncoder(_make_config("direct"))
        assert encoder.num_steps == 1


# ---------------------------------------------------------------------------
# Rate-coded mode
# ---------------------------------------------------------------------------


class TestRateCodedMode:
    """Verify rate-coded encoding mode."""

    def test_output_shape(self):
        """Rate-coded: (batch, W, 39) -> (T, batch, W, 39)."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=10))
        x = _binary_input(batch=4, window=21, features=39)
        out = encoder(x)
        assert out.shape == (10, 4, 21, 39)

    def test_output_shape_timesteps_5(self):
        """Rate-coded with T=5."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=5))
        x = _binary_input(batch=2, window=7, features=39)
        out = encoder(x)
        assert out.shape == (5, 2, 7, 39)

    def test_output_shape_timesteps_1(self):
        """Rate-coded with T=1 (single Bernoulli sample)."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=1))
        x = _binary_input()
        out = encoder(x)
        assert out.shape == (1, 4, 21, 39)

    def test_output_is_binary(self):
        """Rate-coded output values are 0 or 1."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=10))
        x = _binary_input()
        out = encoder(x)
        assert ((out == 0.0) | (out == 1.0)).all()

    def test_all_zeros_input(self):
        """All-zeros input -> all-zeros output (P=0 never fires)."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=10))
        x = torch.zeros(2, 5, 39)
        out = encoder(x)
        assert out.sum().item() == 0.0

    def test_all_ones_input(self):
        """All-ones input -> all-ones output (P=1 always fires)."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=10))
        x = torch.ones(2, 5, 39)
        out = encoder(x)
        assert (out == 1.0).all()

    def test_binary_input_deterministic(self):
        """Binary (0/1) input is deterministic since P=0 never fires, P=1 always fires."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=10))
        x = _binary_input()

        set_global_seed(1)
        out1 = encoder(x)

        set_global_seed(999)
        out2 = encoder(x)

        # For binary inputs, rate encoding is deterministic
        torch.testing.assert_close(out1, out2)

    def test_determinism_with_same_seed(self):
        """Same seed produces identical rate-coded output for fractional values."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=10))
        # Use fractional values to exercise stochastic path
        x = torch.full((2, 5, 39), 0.5)

        set_global_seed(42)
        out1 = encoder(x)

        set_global_seed(42)
        out2 = encoder(x)

        torch.testing.assert_close(out1, out2)

    def test_different_seeds_can_differ_fractional(self):
        """Different seeds can produce different output for fractional values."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=20))
        x = torch.full((4, 10, 39), 0.5)

        set_global_seed(42)
        out1 = encoder(x)

        set_global_seed(999)
        out2 = encoder(x)

        # With P=0.5 and enough timesteps, outputs should differ
        assert not torch.equal(out1, out2)

    def test_num_steps_property(self):
        """Rate-coded num_steps matches timesteps config."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=15))
        assert encoder.num_steps == 15


# ---------------------------------------------------------------------------
# Integration with windowed data format
# ---------------------------------------------------------------------------


class TestEncoderIntegration:
    """Verify encoder works with typical CA5 data shapes."""

    def test_typical_ca5_shape_direct(self):
        """Direct mode with typical CA5 dimensions."""
        encoder = SpikeEncoder(_make_config("direct"))
        x = _binary_input(batch=64, window=21, features=39)
        out = encoder(x)
        assert out.shape == (1, 64, 21, 39)

    def test_typical_ca5_shape_rate_coded(self):
        """Rate-coded with typical CA5 dimensions."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=10))
        x = _binary_input(batch=64, window=21, features=39)
        out = encoder(x)
        assert out.shape == (10, 64, 21, 39)

    def test_window_size_7(self):
        """Works with W=7 (minimum viable window)."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=5))
        x = _binary_input(batch=4, window=7, features=39)
        out = encoder(x)
        assert out.shape == (5, 4, 7, 39)

    def test_single_sample(self):
        """Works with batch=1."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=3))
        x = _binary_input(batch=1, window=21, features=39)
        out = encoder(x)
        assert out.shape == (3, 1, 21, 39)

    def test_output_dtype_float32(self):
        """Output dtype is float32."""
        encoder = SpikeEncoder(_make_config("rate_coded", timesteps=5))
        x = _binary_input()
        out = encoder(x)
        assert out.dtype == torch.float32

    def test_gradient_flows_through(self):
        """Verify that gradients can flow through the encoder (for rate_coded)."""
        encoder = SpikeEncoder(_make_config("direct"))
        x = torch.randn(2, 5, 39, requires_grad=True)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


class TestSnnTorchImport:
    """Verify snnTorch integration with PyTorch."""

    def test_snntorch_imports(self):
        """snntorch can be imported."""
        import snntorch
        assert hasattr(snntorch, "__version__")

    def test_snntorch_version(self):
        """snntorch version is 0.9.1."""
        import snntorch
        assert snntorch.__version__ == "0.9.1"

    def test_spikegen_available(self):
        """snntorch.spikegen is available."""
        from snntorch import spikegen
        assert hasattr(spikegen, "rate")

    def test_leaky_neuron_available(self):
        """snntorch.Leaky neuron is available (needed by STORY-4.2)."""
        import snntorch
        assert hasattr(snntorch, "Leaky")

    def test_spikegen_rate_basic(self):
        """spikegen.rate works with a simple tensor."""
        from snntorch import spikegen
        x = torch.tensor([[1.0, 0.0, 1.0]])
        spikes = spikegen.rate(x, num_steps=3)
        assert spikes.shape == (3, 1, 3)

    def test_encoder_exported_from_models(self):
        """SpikeEncoder is accessible from c5_snn.models."""
        from c5_snn.models import SpikeEncoder as Enc
        assert Enc is SpikeEncoder
