"""Phase 34A — manual chart replay pack: render 100 finalist trades for visual fill/logic audit."""

from __future__ import annotations

import csv
import zoneinfo
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nqlab.config.loader import load_config
from nqlab.discovery import phase29 as P29

OUT = Path("docs/phase34a_trade_replay")
OUT.mkdir(parents=True, exist_ok=True)
PRE, POST = 30, 8  # bars around the trade


def _select(df: pl.DataFrame, strat: str) -> list[tuple[str, dict]]:
    df = df.with_columns(pl.col("net_pnl").cast(pl.Float64))
    wins = df.filter(pl.col("net_pnl") > 0)
    loss = df.filter(pl.col("net_pnl") < 0)
    rng = np.random.default_rng(34)

    def pick(frame: pl.DataFrame, k: int) -> list[dict]:
        if frame.height == 0:
            return []
        idx = rng.choice(frame.height, size=min(k, frame.height), replace=False)
        return [frame.row(int(i), named=True) for i in idx]

    out = []
    out += [(f"{strat}_randwin", r) for r in pick(wins, 13)]
    out += [(f"{strat}_randloss", r) for r in pick(loss, 12)]
    out += [(f"{strat}_bigwin", r) for r in wins.sort("net_pnl", descending=True).head(13).to_dicts()]
    out += [(f"{strat}_bigloss", r) for r in loss.sort("net_pnl").head(12).to_dicts()]
    return out


def _chart(bars: pl.DataFrame, t: dict, tag: str, fname: Path) -> None:
    ets = datetime.fromisoformat(t["entry_ts"])
    xts = datetime.fromisoformat(t["exit_ts"])
    day = bars.filter(pl.col("trading_day").cast(pl.Utf8) == t["trading_day"]).sort("ts")
    if day.height == 0:
        return
    tss = day["ts"].to_list()
    ei = min(range(len(tss)), key=lambda i: abs((tss[i] - ets).total_seconds()))
    xi_full = min(range(len(tss)), key=lambda i: abs((tss[i] - xts).total_seconds()))
    lo, hi = max(0, ei - PRE), min(day.height, xi_full + POST + 1)  # span full entry->exit hold
    w = day[lo:hi]
    o, h, low, c = (w[col].to_numpy() for col in ("open", "high", "low", "close"))
    x = np.arange(len(o))
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(o)):
        col = "#2da44e" if c[i] >= o[i] else "#d1242f"
        ax.plot([i, i], [low[i], h[i]], color=col, lw=0.8)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o[i], c[i])), 0.6, abs(c[i] - o[i]) + 1e-6,
                                   color=col, alpha=0.8))
    if "vwap_session" in w.columns:
        ax.plot(x, w["vwap_session"].to_numpy(), color="#0969da", lw=1.2, label="VWAP")
    if "vp_prior_poc" in w.columns and w["vp_prior_poc"].null_count() < w.height:
        ax.plot(x, w["vp_prior_poc"].to_numpy(), color="#bf8700", lw=1.0, ls="--", label="prior POC")
    eix = ei - lo
    ax.scatter([eix], [t["entry_px"]], marker="^" if t["direction"] == "long" else "v",
               s=130, color="black", zorder=5, label=f"entry {t['direction']}")
    xix = xi_full - lo
    ax.scatter([xix], [t["exit_px"]], marker="x", s=160, color="purple", zorder=6,
               label=f"exit {t['exit_reason']}")
    ax.axhline(t["stop_px"], color="#d1242f", ls=":", lw=1, label="stop")
    if t["target_px"] != "EOD":
        ax.axhline(float(t["target_px"]), color="#2da44e", ls=":", lw=1, label="target")
    et = zoneinfo.ZoneInfo("America/New_York")
    e_et, x_et = ets.astimezone(et), xts.astimezone(et)
    ax.set_title(f"{tag} | {t['trading_day']} {e_et:%H:%M}->{x_et:%H:%M} ET | {t['direction']} "
                 f"{t['exit_reason']} | hold {int((xts-ets).total_seconds()/60)}m | "
                 f"net ${t['net_pnl']:.0f}", fontsize=10)
    ax.set_xlabel("5m bar")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fname, dpi=95)
    plt.close(fig)


def main() -> None:
    cfg = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    bars = pl.read_parquet(cfg.data.artifacts_root / "phase30" / "frame_5m_full_vp.parquet").select(
        ["ts", "trading_day", "open", "high", "low", "close", "vwap_session", "vp_prior_poc"]
    )
    index = []
    for strat in ("idb", "swd"):
        df = pl.read_csv(f"docs/phase34a/phase34a_trades_{strat}.csv")
        for tag, t in _select(df, strat):
            tid = f"{tag}_{t['trade_id']}"
            fn = OUT / f"{tid}.png"
            _chart(bars, t, tag, fn)
            index.append({"trade_id": tid, "chart_file": fn.name, "strategy": strat,
                          "date": t["trading_day"], "direction": t["direction"],
                          "exit_reason": t["exit_reason"], "net_pnl": t["net_pnl"],
                          "pass_fail_review": "manual_review", "suspicious_reason": ""})
    with (OUT / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    print(f"rendered {len(index)} charts -> {OUT}")


if __name__ == "__main__":
    main()
