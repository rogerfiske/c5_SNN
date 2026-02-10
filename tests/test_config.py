"""Tests for YAML config loading."""

import pytest

from c5_snn.utils.config import load_config
from c5_snn.utils.exceptions import ConfigError


class TestLoadConfig:
    def test_load_valid_yaml(self, tmp_path):
        """load_config on a valid YAML file returns a dict."""
        config_file = tmp_path / "valid.yaml"
        config_file.write_text("experiment:\n  name: test\n  seed: 42\n")

        result = load_config(str(config_file))

        assert isinstance(result, dict)
        assert result["experiment"]["name"] == "test"
        assert result["experiment"]["seed"] == 42

    def test_load_nonexistent_path_raises_config_error(self):
        """load_config on a non-existent path raises ConfigError."""
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml_raises_config_error(self, tmp_path):
        """load_config on an invalid YAML file raises ConfigError."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text(":\n  invalid: yaml: [\n")

        with pytest.raises(ConfigError, match="Failed to parse YAML"):
            load_config(str(bad_file))

    def test_load_empty_file_raises_config_error(self, tmp_path):
        """load_config on an empty file raises ConfigError (safe_load returns None)."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")

        with pytest.raises(ConfigError, match="must be a YAML mapping"):
            load_config(str(empty_file))

    def test_load_default_config(self):
        """The project's default.yaml loads successfully."""
        result = load_config("configs/default.yaml")

        assert isinstance(result, dict)
        assert result["experiment"]["seed"] == 42
        assert result["data"]["window_size"] == 21
        assert result["model"]["type"] == "frequency_baseline"
