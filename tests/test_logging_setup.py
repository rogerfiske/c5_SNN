"""Tests for centralized logging configuration."""

import logging

from c5_snn.utils.logging import setup_logging


class TestSetupLogging:
    def test_sets_debug_level(self):
        """setup_logging('DEBUG') sets the c5_snn logger to DEBUG level."""
        setup_logging("DEBUG")
        logger = logging.getLogger("c5_snn")
        assert logger.level == logging.DEBUG

    def test_sets_info_level_by_default(self):
        """setup_logging() defaults to INFO level."""
        setup_logging()
        logger = logging.getLogger("c5_snn")
        assert logger.level == logging.INFO

    def test_console_handler_present(self):
        """setup_logging adds a StreamHandler to the c5_snn logger."""
        setup_logging("INFO")
        logger = logging.getLogger("c5_snn")

        stream_handlers = [
            h for h in logger.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) == 1

    def test_format_matches_expected_pattern(self, capfd):
        """Logger output matches the expected format pattern."""
        setup_logging("INFO")
        logger = logging.getLogger("c5_snn.test_format")
        logger.info("test message")

        captured = capfd.readouterr()
        # Format: "2026-02-10 12:00:00,000 [INFO] c5_snn.test_format: test message"
        assert "[INFO]" in captured.err
        assert "c5_snn.test_format" in captured.err
        assert "test message" in captured.err

    def test_file_handler_created_when_log_file_provided(self, tmp_path):
        """A FileHandler is created when log_file is provided."""
        log_file = tmp_path / "test.log"
        setup_logging("INFO", log_file=str(log_file))

        logger = logging.getLogger("c5_snn")
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

        # Write a message and verify it appears in the file
        logger.info("file handler test")
        file_handlers[0].flush()

        log_content = log_file.read_text()
        assert "file handler test" in log_content

    def test_repeated_calls_do_not_duplicate_handlers(self):
        """Calling setup_logging twice should not create duplicate handlers."""
        setup_logging("INFO")
        setup_logging("DEBUG")

        logger = logging.getLogger("c5_snn")
        # Should have exactly 1 handler (StreamHandler), not 2
        assert len(logger.handlers) == 1
