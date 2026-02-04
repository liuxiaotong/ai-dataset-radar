"""Centralized logging configuration for AI Dataset Radar."""

import logging
import os
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Set up logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.
        format_string: Custom format string for log messages.

    Returns:
        Configured root logger for the application.
    """
    # Get level from environment or use default
    level = os.environ.get("RADAR_LOG_LEVEL", level).upper()

    # Default format with timestamp and level
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Get the root logger for our application
    logger = logging.getLogger("radar")
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "radar") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name, will be prefixed with 'radar.' if not already.

    Returns:
        Logger instance.
    """
    if not name.startswith("radar"):
        name = f"radar.{name}"
    return logging.getLogger(name)


# Initialize default logger on module import
_default_logger = setup_logging()
