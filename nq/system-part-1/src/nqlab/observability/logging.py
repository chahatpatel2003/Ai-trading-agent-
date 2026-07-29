"""Logging setup: console or JSON formatting, idempotent configuration.

Kept dependency-light (stdlib ``logging``). The JSON formatter emits one object
per line so logs are machine-parseable in CI / batch runs; the console formatter
is human-friendly for interactive use.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from nqlab.config.models import LoggingConfig

_ROOT_LOGGER_NAME = "nqlab"
_CONFIGURED = False

# Standard LogRecord attributes we never duplicate into JSON "extra".
_RESERVED = set(
    logging.makeLogRecord({}).__dict__.keys()
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _build_formatter(fmt: str, include_timestamp: bool) -> logging.Formatter:
    if fmt == "json":
        return JsonFormatter()
    if include_timestamp:
        pattern = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    else:
        pattern = "%(levelname)-7s | %(name)s | %(message)s"
    return logging.Formatter(pattern, datefmt="%Y-%m-%dT%H:%M:%S%z")


def configure_logging(config: LoggingConfig) -> None:
    """Configure the ``nqlab`` logger tree. Idempotent (safe to call repeatedly)."""
    global _CONFIGURED

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(config.level.upper())
    logger.propagate = False

    formatter = _build_formatter(config.format, config.include_timestamp)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for existing in logger.handlers:
            existing.setFormatter(formatter)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a namespaced child of the ``nqlab`` logger."""
    if name is None or name == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
