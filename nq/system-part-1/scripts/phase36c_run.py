"""Phase 36C — 3-5 point momentum-burst scalp test (continuation, not mean-reversion).

Does NQ have a real high-win-rate momentum-CONTINUATION edge after costs and realistic fills? Honesty
metric: edge over the breakeven win rate (>= 5pp required). Reuses Phase-33 sub-minute bars + the
conservative-fill scalp engine. Order-flow trigger (G) is MISSING (sub-minute bars are trades-only
OHLCV; MBP1 ends 2025-05) — reported, not faked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nqlab.research.scalp import TICK, add_scalp_outcomes, trade_pnl

OUT = Path("docs/phase36c")
OUT.mkdir(parents=True, exist_ok=True)
_C = pl.col
_D = pl.col("d")


def add_momentum(b: pl.DataFrame) -> pl.DataFrame:
    rng = _C("high") - _C("low")
    pos_in_range = (_C("close") - _C("low")) / pl.max_horizontal(rng, pl.lit(TICK))
    up = _C("close") > _C("close").shift(1).over(_D)
    dn = _C("close") < _C("close").shift(1).over(_D)
    b = b.with_columns(
        rng.alias("_rng"), pos_in_range.alias("_pir"),
        up.cast(pl.Int8).alias("_up"), dn.cast(pl.Int8).alias("_dn"),
        rng.rolling_median(window_size=50).over(_D).alias("_rngmed"),
        (_C("vwap") - _C("vwap").shift(5).over(_D)).alias("_vwap_slope"),
        (_C("close") - _C("close").shift(3).over(_D)).alias("_mv3"),
    )
    # consecutive up/down counts
    b = b.with_columns(
        (_C("_up") + _C("_up").shift(1).over(_D) + _C("_up").shift(2).over(_D)).alias("_up3"),
        (_C("_dn") + _C("_dn").shift(1).over(_D) + _C("_dn").shift(2).over(_D)).alias("_dn3"),
    )
    return b


def triggers() -> dict[str, tuple[pl.Expr, pl.Expr]]:
    """name -> (long_expr, short_expr). All PIT-safe (current/past bars only)."""
    return {
        # A. consecutive-candle momentum (3 up closes, final strong, >=3pt thrust)
        "consec3": (
            (_C("_up3") == 3) & (_C("_pir") > 0.75) & (_C("_mv3") >= 3),
            (_C("_dn3") == 3) & (_C("_pir") < 0.25) & (_C("_mv3") <= -3),
        ),
        # B. range-expansion burst (wide bar, strong close, breaks recent micro high)
        "range_expansion": (
            (_C("_rng") >= 1.5 * _C("_rngmed")) & (_C("_pir") > 0.75) & (_C("close") > _C("rng_hi5")),
            (_C("_rng") >= 1.5 * _C("_rngmed")) & (_C("_pir") < 0.25) & (_C("close") < _C("rng_lo5")),
        ),
        # C. breakout momentum (close clears rolling 10m high by >=1 tick, strong)
        "breakout_mom": (
            (_C("close") > _C("rng_hi") + TICK) & (_C("close").shift(1).over(_D) <= _C("rng_hi")),
            (_C("close") < _C("rng_lo") - TICK) & (_C("close").shift(1).over(_D) >= _C("rng_lo")),
        ),
        # D. VWAP-slope momentum continuation
        "vwap_mom": (
            (_C("close") > _C("vwap")) & (_C("_vwap_slope") > 0) & (_C("close") > _C("rng_hi5")),
            (_C("close") < _C("vwap")) & (_C("_vwap_slope") < 0) & (_C("close") < _C("rng_lo5")),
        ),
        # E. opening-drive momentum (first 30 min, thrust)
        "opening_drive": (
            (_C("mod") < 600) & (_C("_up3") >= 2) & (_C("_pir") > 0.7) & (_C("close") > _C("rng_hi5")),
            (_C("mod") < 600) & (_C("_dn3") >= 2) & (_C("_pir") < 0.3) & (_C("close") < _C("rng_lo5")),
        ),
    }


def _panel(pnl: np.ndarray, days: np.ndarray) -> dict[str, Any]:
    n = len(pnl)
    if n == 0:
        return {"n": 0}
    w, ls = pnl[pnl > 0], pnl[pnl < 0]
    aw = float(w.mean()) if w.size else 0.0
    al = float(-ls.mean()) if ls.size else 0.0
    be = al / (aw + al) if (aw + al) > 0 else None
    wr = float((pnl > 0).mean())
    return {"n": n, "win_rate": round(wr, 4), "breakeven_wr": round(be, 4) if be else None,
            "edge_vs_be": round(wr - be, 4) if be else None,
            "pf": round(float(w.sum() / -ls.sum()), 3) if ls.size else None,
            "expectancy": round(float(pnl.mean()), 3),
            "trades_per_day": round(n / len(np.unique(days)), 2)}


def main() -> None:
    # Part 2: breakeven math table (1 MNQ, audited cost, 1-tick-through fill, 1-tick stop slip)
    print("=== BREAKEVEN MATH (1 MNQ, $2/pt, RT comm $0.50; entry+stop slip 1tk; target exact) ===")
    print("  target/stop : net_win  net_loss  breakeven_wr  wr_for_PF1.4  wr_for_PF1.7")
    be_rows = []
    for t in (3, 4, 5):
        for s in (3, 4, 5, 6):
            net_win = t * 2 - TICK * 2 - 0.50              # entry slip only on win (target exact)
            net_loss = -(s * 2 + TICK * 2 * 2 + 0.50)      # entry + stop slip
            be = -net_loss / (net_win - net_loss)
            wr14 = 1.4 * -net_loss / (net_win + 1.4 * -net_loss)
            wr17 = 1.7 * -net_loss / (net_win + 1.7 * -net_loss)
            print(f"  {t}/{s} : ${net_win:.2f}  ${net_loss:.2f}  {be:.3f}  {wr14:.3f}  {wr17:.3f}")
            be_rows.append({"target": t, "stop": s, "net_win": round(net_win, 2),
                            "net_loss": round(net_loss, 2), "breakeven_wr": round(be, 3),
                            "wr_pf14": round(wr14, 3), "wr_pf17": round(wr17, 3)})
    pl.DataFrame(be_rows).write_csv(OUT / "phase36c_breakeven.csv")

    rows: list[dict[str, Any]] = []
    for tf, tfmin in (("15s", 0.25), ("1m", 1.0)):
        b = add_momentum(pl.read_parquet(f"docs/phase33/bars_{tf}.parquet"))
        b = add_scalp_outcomes(b, tf_min=tfmin, hold_min=3.0)  # short hold for momentum
        valid = (_C("scalp_valid") == 1) & _C("micro_atr").is_not_null() & (_C("micro_atr") > 0) & _C("rng_hi5").is_not_null()
        base = b.filter(valid)
        for name, (lo, sh) in triggers().items():
            for direction, expr in (("long", lo), ("short", sh)):
                sub = base.filter(expr.fill_null(value=False))
                if sub.height < 100:
                    continue
                days = sub["d"].to_numpy()
                for tgt in (3, 4, 5):
                    for stp in (3, 4, 5, 6):
                        for margin, cost in ((1, 1.0), (0, 1.0), (1, 2.0)):
                            pnl = trade_pnl(sub, direction=direction, target=tgt, stop=stp,
                                            margin=margin, stop_slip_ticks=1.0, cost_mult=cost)
                            p = _panel(pnl, days)
                            if p.get("n", 0) == 0:
                                continue
                            rows.append({"tf": tf, "trigger": name, "direction": direction,
                                         "target": tgt, "stop": stp, "margin": margin,
                                         "cost_mult": cost, **p})
    df = pl.DataFrame(rows, infer_schema_length=None)
    df.sort("edge_vs_be", descending=True).write_csv(OUT / "phase36c_leaderboard.csv")
    (OUT / "_mom.json").write_text(json.dumps(rows, indent=2, default=str))

    cols = ["tf", "trigger", "direction", "target", "stop", "margin", "cost_mult", "n",
            "trades_per_day", "win_rate", "breakeven_wr", "edge_vs_be", "pf"]
    print(f"\ncandidates: {df.height}")
    print("\n=== TOP 15 by EDGE over breakeven (realistic: margin=1, 1x cost), n>=200 ===")
    real = df.filter((_C("margin") == 1) & (_C("cost_mult") == 1.0) & (_C("n") >= 200))
    with pl.Config(tbl_rows=17, fmt_str_lengths=16, tbl_width_chars=200):
        print(real.sort("edge_vs_be", descending=True).select(cols).head(15))
    print("\n=== TOP by PF (realistic), n>=200 ===")
    with pl.Config(tbl_rows=10, fmt_str_lengths=16, tbl_width_chars=200):
        print(real.sort("pf", descending=True).select(cols).head(8))
    full = real.filter((_C("edge_vs_be") >= 0.05) & (_C("pf") >= 1.4))
    print(f"\n>>> momentum scalps meeting bar (edge>=5pp, PF>=1.4, margin1, 1x, n>=200): {full.height}")
    if full.height:
        with pl.Config(tbl_rows=12, fmt_str_lengths=16): print(full.sort("edge_vs_be", descending=True).select(cols))
    print(f"\nmax edge_vs_be (realistic): {real['edge_vs_be'].max()} | max PF: {real['pf'].max()}")
    # zero-cost touch (most optimistic) — is there ANY raw momentum edge?
    print("\n=== zero-cost touch-fill upper bound (margin=0) ... checking raw edge ===")
    opt = df.filter((_C("margin") == 0) & (_C("cost_mult") == 1.0) & (_C("n") >= 200))
    print(f"  best edge_vs_be at touch/1x: {opt['edge_vs_be'].max()} | best PF: {opt['pf'].max()}")


if __name__ == "__main__":
    main()
