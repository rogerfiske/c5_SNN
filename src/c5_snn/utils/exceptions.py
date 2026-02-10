"""Custom exception hierarchy for the c5_SNN pipeline."""


class C5SNNError(Exception):
    """Base exception for the pipeline."""


class DataValidationError(C5SNNError):
    """Raised when input data fails integrity checks."""


class ConfigError(C5SNNError):
    """Raised when YAML config is missing required fields or has invalid values."""


class TrainingError(C5SNNError):
    """Raised when training encounters an unrecoverable error (NaN loss, OOM)."""
