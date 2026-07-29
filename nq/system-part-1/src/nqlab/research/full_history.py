"""Full-history orchestration: ensure bars/continuous/features, then validate."""

from __future__ import annotations

from datetime import date

from nqlab.config.models import AppConfig
from nqlab.data.bars_full import build_bars_full
from nqlab.data.continuous import build_continuous
from nqlab.features.build_full import build_features_full


def ensure_full_history(
    config: AppConfig,
    *,
    root: str = "NQ",
    timeframe: str = "5m",
    start: date,
    end: date,
    force: bool = False,
    verify: bool = False,
) -> str:
    """Incrementally build per-contract bars, the continuous series, and features.

    Returns the continuous symbol (e.g. ``NQ.c``).
    """
    build_bars_full(
        config, timeframe=timeframe, start=start, end=end, roots=(root,),
        force=force, verify=verify,
    )
    continuous = build_continuous(config, root=root, timeframe=timeframe, start=start, end=end)
    build_features_full(
        config, symbol=continuous.symbol, timeframe=timeframe, start=start, end=end, force=force
    )
    return continuous.symbol
