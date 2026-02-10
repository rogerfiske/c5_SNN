"""Centralized logging configuration for the c5_SNN pipeline."""

import logging
import sys

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure the c5_snn logger with console and optional file output.

    Args:
        level: Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output. If provided, a FileHandler
            is added alongside the console handler.
    """
    logger = logging.getLogger("c5_snn")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output on repeated calls
    logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler (stderr)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logger.level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Optional file handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
