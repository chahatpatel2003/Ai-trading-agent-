"""Phase 33 — micro-scalping families + grid evaluation over conservative-fill outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from nqlab.research.scalp import STOPS, TARGETS, trade_pnl

_C = pl.col
_D = pl.col("d")


@dataclass(frozen=True)
class Fam:
    name: str
    long: pl.Expr
    short: pl.Expr


def families() -> list[Fam]:
    up = _C("close") > _C("close").shift(1).over(_D)
    dn = _C("close") < _C("close").shift(1).over(_D)
    stretched_dn = _C("dist_vwap_atr") < -1.5
    stretched_up = _C("dist_vwap_atr") > 1.5
    return [
        # 1/2. VWAP (≈POC) magnet: stretched from VWAP then micro reclaim toward it
        Fam("vwap_magnet", stretched_dn & up, stretched_up & dn),
        # 3. micro extension-reversion: large stretch then failed continuation
        Fam("ext_reversion", (_C("dist_vwap_atr") < -2.5) & up, (_C("dist_vwap_atr") > 2.5) & dn),
        # 4. micro range rotation: tag rolling-range edge and reject back in
        Fam(
            "range_rotation",
            (_C("low") <= _C("rng_lo")) & (_C("close") > _C("rng_lo")) & up,
            (_C("high") >= _C("rng_hi")) & (_C("close") < _C("rng_hi")) & dn,
        ),
        # 5. failed micro breakout: break range edge then close back inside
        Fam(
            "failed_breakout",
            (_C("low") < _C("rng_lo")) & (_C("close") > _C("rng_lo")),
            (_C("high") > _C("rng_hi")) & (_C("close") < _C("rng_hi")),
        ),
        # 6. micro momentum burst: range expansion breaking the recent micro high/low
        Fam(
            "momentum_burst",
            ((_C("high") - _C("low")) > 2.0 * _C("micro_atr")) & up & (_C("close") > _C("rng_hi5")),
            ((_C("high") - _C("low")) > 2.0 * _C("micro_atr")) & dn & (_C("close") < _C("rng_lo5")),
        ),
        # 7. trend pullback to VWAP: above/below VWAP, tag it, hold
        Fam(
            "trend_pullback_vwap",
            (_C("close") > _C("vwap")) & (_C("low") <= _C("vwap")) & up,
            (_C("close") < _C("vwap")) & (_C("high") >= _C("vwap")) & dn,
        ),
        # 8. liquidity sweep of opening range edge + reclaim
        Fam(
            "or_sweep_reclaim",
            (_C("low") < _C("or_lo")) & (_C("close") > _C("or_lo")) & (_C("mod") > 585),
            (_C("high") > _C("or_hi")) & (_C("close") < _C("or_hi")) & (_C("mod") > 585),
        ),
    ]


def _valid(frame: pl.DataFrame) -> pl.Expr:
    return (
        (_C("scalp_valid") == 1)
        & _C("micro_atr").is_not_null()
        & (_C("micro_atr") > 0)
        & _C("vwap").is_not_null()
        & _C("rng_lo").is_not_null()
    )


def _panel(
    pnl: np.ndarray, target: int, stop: int, ts: np.ndarray, pv: float = 2.0, comm: float = 0.5
) -> dict[str, Any]:
    n = len(pnl)
    if n == 0:
        return {"n": 0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    aw = float(wins.mean()) if wins.size else 0.0
    al = float(-losses.mean()) if losses.size else 0.0
    be = al / (aw + al) if (aw + al) > 0 else None  # breakeven win rate
    eq = np.cumsum(pnl)
    mdd = float((np.maximum.accumulate(eq) - eq).max())
    # year stats
    yrs = np.array([int(str(t)[:4]) for t in ts])
    ypos = sum(1 for y in np.unique(yrs) if pnl[yrs == y].sum() > 0)
    return {
        "n": n,
        "win_rate": round(float((pnl > 0).mean()), 4),
        "breakeven_wr": round(be, 4) if be else None,
        "edge_vs_be": round(float((pnl > 0).mean()) - be, 4) if be else None,
        "pf": round(float(wins.sum() / -losses.sum()), 3) if losses.size else None,
        "expectancy": round(float(pnl.mean()), 3),
        "avg_win": round(aw, 2),
        "avg_loss": round(-al, 2),
        "net": round(float(pnl.sum()), 2),
        "max_dd": round(mdd, 2),
        "monthly": round(float(pnl.sum()) / 66, 2),
        "years_pos": ypos,
    }


def evaluate(
    frame: pl.DataFrame,
    tf_min: float,
    *,
    cost_mult: float = 1.0,
    margin: int = 1,
    stop_slip: float = 1.0,
    instrument: str = "MNQ",
) -> list[dict[str, Any]]:
    """Sweep families x targets x stops at one fill/cost setting (default 1-tick-through, 1x)."""
    base = frame.filter(_valid(frame))
    rows: list[dict[str, Any]] = []
    for fam in families():
        for direction, expr in (("long", fam.long), ("short", fam.short)):
            sub = base.filter(expr.fill_null(value=False))
            if sub.height < 50:
                continue
            ts = sub["d"].to_numpy()
            n_days = sub["d"].n_unique()
            for t in TARGETS:
                for s in STOPS:
                    if t >= s + 4:  # skip nonsensical target>=stop+large
                        continue
                    pnl = trade_pnl(
                        sub,
                        direction=direction,
                        target=t,
                        stop=s,
                        margin=margin,
                        stop_slip_ticks=stop_slip,
                        cost_mult=cost_mult,
                        instrument=instrument,
                    )
                    p = _panel(pnl, t, s, ts)
                    if p.get("n", 0) == 0:
                        continue
                    rows.append(
                        {
                            "family": fam.name,
                            "direction": direction,
                            "target": t,
                            "stop": s,
                            "margin": margin,
                            "cost_mult": cost_mult,
                            "instrument": instrument,
                            "trades_per_day": round(p["n"] / n_days, 2),
                            **p,
                        }
                    )
    return rows
