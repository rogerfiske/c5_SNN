"""YAML experiment config loader."""

from pathlib import Path

import yaml

from c5_snn.utils.exceptions import ConfigError


def load_config(path: str) -> dict:
    """Load a YAML config file and return the parsed dict.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config as a dictionary.

    Raises:
        ConfigError: If the file does not exist, YAML parsing fails,
            or the result is not a dict.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config: {path}\n{e}") from e

    if not isinstance(data, dict):
        raise ConfigError(
            f"Config must be a YAML mapping (dict), got {type(data).__name__}: {path}"
        )

    return data
