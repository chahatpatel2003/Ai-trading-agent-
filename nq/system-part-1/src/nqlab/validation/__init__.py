"""Validation gauntlet: the machine that judges edge vs. noise."""

from __future__ import annotations

from nqlab.validation.gauntlet import ValidationResult, run_gauntlet
from nqlab.validation.verdict import ACCEPT, NEEDS_MORE_DATA, REJECT

__all__ = ["ACCEPT", "NEEDS_MORE_DATA", "REJECT", "ValidationResult", "run_gauntlet"]
