"""
verify_symbol.py
================

SMALL verification (NOT a bulk download): pull the most recent ~3 trading days of
USATECHIDXUSD from Dukascopy, compute stats, sanity-check the price level against
the Nasdaq-100, draw 3 charts, and write a verification report with a verdict.

Outputs -> D:\\NAS100 Verification Test
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, r"D:\NAS 100 Data\scripts")
import convert
import dukascopy

VERIFY_DIR = Path(r"D:\NAS100 Verification Test")
RAW_DIR = VERIFY_DIR / "raw"
CHARTS_DIR = VERIFY_DIR / "charts"
INSTRUMENT_NAME = "USA 100 Technical Index (Nasdaq-100 CFD)"  # Dukascopy catalog name

# Approximate recent index levels used only to NAME a mismatch as evidence.
INDEX_BANDS = {
    "Nasdaq-100 (USATECHIDXUSD)": (12000, 36000),
    "S&P 500 (USA500IDXUSD)": (3500, 8000),
    "Dow 30 (USA30IDXUSD)": (28000, 50000),
    "Russell 2000 (USSC2000IDXUSD)": (1200, 3500),
}


def find_recent_days(session, n=3, max_back=25):
    days, back = [], 0
    d = datetime.now(timezone.utc).date()
    while len(days) < n and back <= max_back:
        dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        if dt.weekday() < 5:
            has = any(dukascopy.decode_hour(
                dukascopy.fetch_hour(session, dt.replace(hour=h), max_retries=3), dt.replace(hour=h))
                for h in (14, 15, 16, 13, 17))
            if has:
                days.append(d)
        d -= timedelta(days=1)
        back += 1
    return sorted(days)


def candlesticks(df: pl.DataFrame, title: str, out: Path):
    """Draw a simple OHLC candlestick chart using an integer x-axis (no gaps)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    x = range(df.height)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i in x:
        up = c[i] >= o[i]
        color = "#1a9850" if up else "#d73027"
        ax.vlines(i, l[i], h[i], color=color, linewidth=0.6)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o[i], c[i])), 0.6,
                                   max(abs(c[i] - o[i]), 1e-6), color=color))
    # sparse time labels
    ts = df["timestamp"]
    step = max(1, df.height // 12)
    ax.set_xticks(list(x)[::step])
    ax.set_xticklabels([str(ts[i])[5:16] for i in list(x)[::step]], rotation=45, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def ohlc(df_ticks: pl.DataFrame, every: str) -> pl.DataFrame:
    return (df_ticks.group_by_dynamic("timestamp", every=every, closed="left", label="left")
            .agg(pl.col("mid").first().alias("open"), pl.col("mid").max().alias("high"),
                 pl.col("mid").min().alias("low"), pl.col("mid").last().alias("close"),
                 pl.len().alias("ticks"))
            .drop_nulls("open"))


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    session = dukascopy.make_session()

    print(f"Symbol: {dukascopy.config.SYMBOL}")
    print("Locating most recent trading days with data ...")
    days = find_recent_days(session, n=3)
    if not days:
        print("NO DATA returned for USATECHIDXUSD in the last ~25 days.")
        (VERIFY_DIR / "verification_report.json").write_text(json.dumps(
            {"symbol": dukascopy.config.SYMBOL, "conclusion": "INCORRECT INSTRUMENT",
             "evidence": "No tick data returned from Dukascopy for this symbol."}, indent=2))
        return 2
    print("Days:", [str(d) for d in days])

    # detect price factor from the most recent day
    recs_probe = []
    last_day = datetime(days[-1].year, days[-1].month, days[-1].day, tzinfo=timezone.utc)
    for h in (15, 14, 16, 13, 17):
        recs_probe = dukascopy.decode_hour(dukascopy.fetch_hour(session, last_day.replace(hour=h)), last_day.replace(hour=h))
        if recs_probe:
            break
    asks = [r[1] for r in recs_probe[:5000]]
    raw_med = sorted(asks)[len(asks) // 2]
    factor = dukascopy.detect_price_factor(asks)
    disambig = {str(f): round(raw_med / f, 3) for f in dukascopy.config.PRICE_FACTOR_CANDIDATES}
    print(f"raw median ask={raw_med}  detected factor={factor}  candidates->price {disambig}")

    # download all hours of the chosen days
    all_records = []
    for d in days:
        for h in range(24):
            dt = datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc)
            raw = dukascopy.fetch_hour(session, dt, max_retries=5)
            if raw:
                (RAW_DIR / f"{dt:%Y%m%d_%H}h_ticks.bi5").write_bytes(raw)
                all_records.extend(dukascopy.decode_hour(raw, dt))

    df = convert.add_derived(convert.build_frame(all_records, factor))
    df.write_parquet(VERIFY_DIR / "ticks_sample.parquet", compression="snappy")

    bid, ask, mid = df["bid"], df["ask"], df["mid"]
    latest = float(mid[-1])
    stats = {
        "instrument_name": INSTRUMENT_NAME,
        "symbol": dukascopy.config.SYMBOL,
        "price_factor": factor,
        "raw_median_ask": raw_med,
        "factor_disambiguation": disambig,
        "trading_days": [str(d) for d in days],
        "row_count": df.height,
        "first_timestamp": str(df["timestamp"][0]),
        "last_timestamp": str(df["timestamp"][-1]),
        "min_bid": float(bid.min()), "max_bid": float(bid.max()),
        "min_ask": float(ask.min()), "max_ask": float(ask.max()),
        "min_price": float(mid.min()), "max_price": float(mid.max()),
        "latest_price": latest,
        "average_spread": float((ask - bid).mean()),
    }

    # ---- price sanity / conclusion ----
    matches = [name for name, (lo, hi) in INDEX_BANDS.items() if lo <= latest <= hi]
    is_nasdaq = any("Nasdaq-100" in m for m in matches)
    if is_nasdaq:
        conclusion = "VERIFIED NAS100 CFD"
        evidence = (f"Latest price {latest:.2f} sits in the Nasdaq-100 band (12000-36000) and is "
                    f"inconsistent with S&P500 (~3500-8000), Dow (~28000-50000) or Russell (~1200-3500). "
                    f"Dukascopy catalogs USATECHIDXUSD as '{INSTRUMENT_NAME}'. Factor {factor} (3-decimal "
                    f"index CFD) yields {latest:.2f}; other divisors give implausible values {disambig}.")
    else:
        likely = matches[0] if matches else "unknown"
        conclusion = "INCORRECT INSTRUMENT"
        evidence = (f"Latest price {latest:.2f} does NOT match the Nasdaq-100 (~12000-36000). "
                    f"It looks more like: {likely}. Investigate alternative symbol.")

    stats["conclusion"] = conclusion
    stats["evidence"] = evidence
    (VERIFY_DIR / "verification_report.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    # ---- charts ----
    # 1) tick chart (mid)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df["timestamp"].to_list(), mid.to_list(), linewidth=0.4, color="#1f77b4")
    ax.set_title(f"USATECHIDXUSD tick mid-price  ({days[0]} .. {days[-1]})")
    ax.set_ylabel("Mid price"); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(CHARTS_DIR / "tick_chart.png", dpi=120); plt.close(fig)

    # 2) 1-min, 3) 5-min OHLC
    candlesticks(ohlc(df, "1m"), "USATECHIDXUSD 1-minute OHLC (mid)", CHARTS_DIR / "ohlc_1min.png")
    candlesticks(ohlc(df, "5m"), "USATECHIDXUSD 5-minute OHLC (mid)", CHARTS_DIR / "ohlc_5min.png")

    # ---- console summary ----
    print("\n================ VERIFICATION ================")
    for k in ["instrument_name", "symbol", "price_factor", "row_count", "first_timestamp",
              "last_timestamp", "min_bid", "max_bid", "min_ask", "max_ask",
              "min_price", "max_price", "latest_price", "average_spread"]:
        print(f"  {k:18s}: {stats[k]}")
    print(f"  factor_disambig   : {disambig}")
    print(f"\n  CONCLUSION: {conclusion}")
    print(f"  EVIDENCE  : {evidence}")
    print("  charts ->", CHARTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
