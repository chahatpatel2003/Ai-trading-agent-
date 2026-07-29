"""
Structured logging setup using structlog.
All modules import get_logger() from here.
"""
import logging
import sys
from pathlib import Path

import structlog


def setup_logging(log_dir: str = "./logs", level: str = "INFO") -> None:
    """Configure structured logging to both console and file."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # File handler
    file_handler = logging.FileHandler(log_path / "trading_bot.log")
    file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger().addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = None) -> structlog.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name or "gold_bot")
