"""Phase 39 step 1 — PIT-safe day-level regime table + per-sleeve per-regime PnL.

Regimes are measured at the FIRST RTH bar of each day (known at the open -> PIT-safe): trend
(bull/bear/neutral via htf/4h), volatility tercile (own ATR-at-open terciles, since frame vol_regime
is ~always 'high'), chop (low ADX / compression), and a trend-day proxy (prior-day type). Per-sleeve
per-regime PnL is reported on TRAIN (2021-22) and FULL so regime gates can be decided on train and
validated OOS. Writes docs/phase39/day_regime.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from nqlab.config.loader import load_config
from nqlab.discovery import phase29 as P29

OUT = Path("docs/phase39")
OUT.mkdir(parents=True, exist_ok=True)
C = pl.col
MEMBERS = ["od_cont_struct", "asia_sweep_fade", "disp_pb_fvg", "trend_adx_cont", "dow_long_mon",
           "ir30_break", "ir60_break", "ir90_break", "ir60_fade"]


def build_regime() -> dict[str, dict]:
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    fr = pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet")
    op = (fr.filter(C("is_rth")).sort(["trading_day", "minute_of_day"])
          .group_by("trading_day").first()
          .select("trading_day", "htf_bull", "htf_bear", "trend_4h_bull", "adx", "atr",
                  "compression_setup", "prev_day_type", "gap_dir", "vol_regime_ratio"))
    # ATR terciles (own) -> vol regime; ADX terciles -> trend strength
    a = op["atr"].to_numpy()
    qlo, qhi = np.nanquantile(a, [0.33, 0.66])
    adx = op["adx"].to_numpy()
    adx_lo = np.nanquantile(adx, 0.33)
    out: dict[str, dict] = {}
    for r in op.iter_rows(named=True):
        atr_v, adx_v = r["atr"], r["adx"]
        vol = "vol_hi" if atr_v >= qhi else ("vol_lo" if atr_v < qlo else "vol_mid")
        trend = "bull" if r["htf_bull"] == 1 else ("bear" if r["htf_bear"] == 1 else "neutral")
        chop = bool((adx_v is not None and adx_v < 20) or r["compression_setup"] == 1)
        out[str(r["trading_day"])] = {
            "trend": trend, "vol": vol, "chop": chop,
            "adx": round(float(adx_v), 1) if adx_v is not None else None,
            "trend_str": "adx_hi" if (adx_v or 0) >= np.nanquantile(adx, 0.66) else
                         ("adx_lo" if (adx_v or 99) < adx_lo else "adx_mid"),
            "pdt": r["prev_day_type"], "gap_up": (r["gap_dir"] == 1),
        }
    print(f"ATR terciles: <{qlo:.1f} lo, >={qhi:.1f} hi | ADX lo<{adx_lo:.1f}")
    return out


def main() -> None:
    reg = build_regime()
    (OUT / "day_regime.json").write_text(json.dumps(reg, default=str))
    store = json.loads((Path("docs/phase38") / "_trades.json").read_text())

    def yr(d: str) -> int:
        return int(d[:4])

    # per-sleeve per-regime PnL (train vs full), for trend & vol & chop
    def regime_pnl(nm: str, key: str, train_only: bool) -> dict[str, float]:
        agg: dict[str, float] = {}
        for t in store[nm]:
            if train_only and not (2021 <= yr(t["day"]) <= 2022):
                continue
            rv = reg[t["day"]][key] if t["day"] in reg else None
            k = str(rv)
            agg[k] = agg.get(k, 0.0) + t["net"]
        return {k: round(v, 0) for k, v in sorted(agg.items())}

    print("\n=== per-sleeve PnL by TREND regime (full | train) ===")
    for nm in MEMBERS:
        print(f"{nm:18s} full={regime_pnl(nm,'trend',False)}  train={regime_pnl(nm,'trend',True)}")
    print("\n=== per-sleeve PnL by VOL regime (full | train) ===")
    for nm in MEMBERS:
        print(f"{nm:18s} full={regime_pnl(nm,'vol',False)}  train={regime_pnl(nm,'vol',True)}")
    print("\n=== per-sleeve PnL by CHOP (True=chop/lowADX) (full | train) ===")
    for nm in MEMBERS:
        print(f"{nm:18s} full={regime_pnl(nm,'chop',False)}  train={regime_pnl(nm,'chop',True)}")
    # regime day counts
    from collections import Counter
    print("\nregime day counts:",
          "trend", dict(Counter(v["trend"] for v in reg.values())),
          "| vol", dict(Counter(v["vol"] for v in reg.values())),
          "| chop", dict(Counter(v["chop"] for v in reg.values())))


if __name__ == "__main__":
    main()
