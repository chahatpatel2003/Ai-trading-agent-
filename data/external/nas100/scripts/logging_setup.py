"""logging_setup.py - UTC file + console loggers (download, validation, errors)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import config

_FMT = "%(asctime)s UTC | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


class _UTC(logging.Formatter):
    converter = time.gmtime


def _logger(name: str, logfile: Path, *, level=logging.DEBUG, console=True) -> logging.Logger:
    lg = logging.getLogger(name)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    if not any(getattr(h, "_nas_file", None) == str(logfile) for h in lg.handlers):
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(_UTC(_FMT, _DATEFMT))
        setattr(fh, "_nas_file", str(logfile))
        lg.addHandler(fh)
    if console and not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in lg.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(_UTC(_FMT, _DATEFMT))
        lg.addHandler(ch)
    return lg


def setup_logging() -> tuple[logging.Logger, logging.Logger, logging.Logger]:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    dl = _logger("download", config.DOWNLOAD_LOG)
    val = _logger("validation", config.VALIDATION_LOG)
    err = _logger("errors", config.ERROR_LOG, level=logging.WARNING)
    return dl, val, err
