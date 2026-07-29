"""Phase 28 — mechanical LTF-confirmation rulebook for manual HTF-level trading.

Not a full auto-strategy. The HTF level/bias is given; this module measures, for each MECHANICALLY
defined LTF confirmation, its forward reliability AFTER an HTF-POI touch — so a manual trader knows
which confirmation to wait for. Every confirmation is an exact, PIT-safe rule (no discretionary
words): each is defined in code over the audited structure / daily-bias / htf_poi feature columns.

Direction is set by the POI: a support POI (sell-side liquidity / discount) allows LONG only; a
resistance POI allows SHORT only. Confirmation must occur within the recent-touch window (the
``at_*_poi_recent`` flag is the "after HTF level touch" gate). Win rate is measured at fixed 1:1
(and 1.5R / 2R) with exact costs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.models import AppConfig
from nqlab.domain.strategy.base import StrategyParams, attach_signals

_PARAMS = StrategyParams("p28", "1")


@dataclass(frozen=True)
class Confirmation:
    name: str
    definition: str           # exact mechanical rule (documented)
    long: Callable[[], pl.Expr]
    short: Callable[[], pl.Expr]


def _c(name: str, definition: str, long: pl.Expr, short: pl.Expr) -> Confirmation:
    return Confirmation(name, definition, lambda: long, lambda: short)


# All confirmations are mechanical over audited PIT-safe columns. Long = support-POI reaction,
# short = resistance-POI reaction.
CONFIRMATIONS: dict[str, Confirmation] = {
    "sweep_reclaim": _c(
        "sweep+reclaim",
        "swept a confirmed swing (swept_low/high==1) AND reclaim close back (reclaim_up/dn==1)",
        (pl.col("swept_low") == 1) & (pl.col("reclaim_up") == 1),
        (pl.col("swept_high") == 1) & (pl.col("reclaim_dn") == 1),
    ),
    "mss": _c(
        "MSS",
        "market-structure shift: mss_up/mss_dn==1 (close beyond the opposing confirmed swing)",
        pl.col("mss_up") == 1, pl.col("mss_dn") == 1,
    ),
    "choch": _c(
        "CHoCH",
        "change of character: choch_up/choch_dn==1 (first counter-trend structure break)",
        pl.col("choch_up") == 1, pl.col("choch_dn") == 1,
    ),
    "ifvg": _c(
        "IFVG reclaim",
        "inverted-FVG reclaim: ifvg_bull/ifvg_bear==1 (price reclaims an inverted gap)",
        pl.col("ifvg_bull") == 1, pl.col("ifvg_bear") == 1,
    ),
    "cisd": _c(
        "CISD",
        "change in state of delivery: cisd_up/cisd_dn==1 (displacement past prior bar extreme)",
        pl.col("cisd_up") == 1, pl.col("cisd_dn") == 1,
    ),
    "displacement": _c(
        "displacement",
        "displacement candle (body>=0.6 range, range>=1.5 ATR) closing in trade direction",
        (pl.col("displacement") == 1) & (pl.col("close") > pl.col("open")),
        (pl.col("displacement") == 1) & (pl.col("close") < pl.col("open")),
    ),
    "micro_swing": _c(
        "micro-swing reclaim",
        "reclaim of the most recent confirmed swing (reclaim_up/reclaim_dn==1)",
        pl.col("reclaim_up") == 1, pl.col("reclaim_dn") == 1,
    ),
    "vwap_reclaim": _c(
        "VWAP reclaim",
        "candle low<=session VWAP and close>VWAP (long); high>=VWAP and close<VWAP (short)",
        (pl.col("low") <= pl.col("vwap_session")) & (pl.col("close") > pl.col("vwap_session")),
        (pl.col("high") >= pl.col("vwap_session")) & (pl.col("close") < pl.col("vwap_session")),
    ),
    # --- small explainable combinations ---
    "sweep+mss": _c(
        "sweep + MSS",
        "sweep_reclaim AND mss in the trade direction",
        (pl.col("swept_low") == 1) & (pl.col("reclaim_up") == 1) & (pl.col("mss_up") == 1),
        (pl.col("swept_high") == 1) & (pl.col("reclaim_dn") == 1) & (pl.col("mss_dn") == 1),
    ),
    "sweep+cisd": _c(
        "sweep + CISD",
        "sweep_reclaim AND cisd in the trade direction",
        (pl.col("swept_low") == 1) & (pl.col("reclaim_up") == 1) & (pl.col("cisd_up") == 1),
        (pl.col("swept_high") == 1) & (pl.col("reclaim_dn") == 1) & (pl.col("cisd_dn") == 1),
    ),
    "mss+displacement": _c(
        "MSS + displacement",
        "mss AND a displacement candle in the trade direction",
        (pl.col("mss_up") == 1) & (pl.col("displacement") == 1) & (pl.col("close") > pl.col("open")),  # noqa: E501
        (pl.col("mss_dn") == 1) & (pl.col("displacement") == 1) & (pl.col("close") < pl.col("open")),  # noqa: E501
    ),
    "conservative": _c(
        "conservative (sweep+reclaim + (MSS or CISD))",
        "sweep_reclaim AND (mss OR cisd) in the trade direction",
        (pl.col("swept_low") == 1) & (pl.col("reclaim_up") == 1)
        & ((pl.col("mss_up") == 1) | (pl.col("cisd_up") == 1)),
        (pl.col("swept_high") == 1) & (pl.col("reclaim_dn") == 1)
        & ((pl.col("mss_dn") == 1) | (pl.col("cisd_dn") == 1)),
    ),
}


def confirmation_signal(
    frame: pl.DataFrame,
    conf: Confirmation,
    *,
    stop_pts: float,
    target_pts: float,
    window: tuple[int, int] = (570, 720),
    use_bias: bool = False,
    bias_def: str = "a",
) -> pl.DataFrame:
    """Build a signal: long after a support-POI touch + bullish confirmation; short mirror.

    ``use_bias`` additionally requires daily-bias + HTF-trend alignment in the trade direction.
    """
    in_window = (pl.col("minute_of_day") >= window[0]) & (pl.col("minute_of_day") < window[1])
    long = (pl.col("at_support_poi_recent") == 1) & conf.long() & in_window
    short = (pl.col("at_resistance_poi_recent") == 1) & conf.short() & in_window
    if use_bias:
        long = long & (pl.col(f"aligned_long_{bias_def}") == 1)
        short = short & (pl.col(f"aligned_short_{bias_def}") == 1)
    return attach_signals(
        frame, long=long, short=short, stop_pts=pl.lit(stop_pts), target_pts=pl.lit(target_pts)
    )


def _streaks(nets: list[float]) -> tuple[int, int, int]:
    cur = longest = n3 = n5 = 0
    runs: list[int] = []
    for x in nets:
        if x < 0:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
        longest = max(longest, cur)
    if cur:
        runs.append(cur)
    n3 = sum(1 for r in runs if r >= 3)
    n5 = sum(1 for r in runs if r >= 5)
    return longest, n3, n5


def evaluate_confirmation(
    config: AppConfig,
    conf: Confirmation,
    frame: pl.DataFrame,
    *,
    tf: str,
    rules: Any,
    stop_pts: float = 10.0,
    r_mult: float = 1.0,
    window: tuple[int, int] = (570, 720),
    use_bias: bool = False,
) -> dict[str, Any]:
    """Backtest one confirmation at fixed R after an HTF-POI touch; return the reliability panel."""
    sig = confirmation_signal(
        frame, conf, stop_pts=stop_pts, target_pts=stop_pts * r_mult, window=window,
        use_bias=use_bias,
    ).filter(pl.col("is_rth"))
    bt = run_backtest(
        sig, symbol="NQ.c", timeframe=tf, strategy=_PARAMS, config=config, rules=rules,
        entry_mode="next_open", gate_entries=False, primary_cutoff=False,
    )
    nets = [t.net_pnl for t in bt.trades]
    n = len(nets)
    if n == 0:
        return {"name": conf.name, "tf": tf, "n": 0}
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x < 0]
    longest, n3, n5 = _streaks(nets)
    mfe = sum(t.mfe for t in bt.trades) / n
    mae = sum(t.mae for t in bt.trades) / n
    return {
        "name": conf.name, "tf": tf, "n": n,
        "win_rate": round(len(wins) / n, 4),
        "pf": round(sum(wins) / -sum(losses), 3) if losses else None,
        "expectancy": round(sum(nets) / n, 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "avg_mfe": round(mfe, 2),
        "avg_mae": round(mae, 2),
        "mfe_mae": round(mfe / mae, 3) if mae > 0 else None,
        "max_losing_streak": longest, "n_3loss": n3, "n_5loss": n5,
        "net": round(sum(nets), 2),
    }
