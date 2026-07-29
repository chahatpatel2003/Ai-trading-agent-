#!/usr/bin/env python
"""Build a sample of NQ front-month 1-minute bars (READ-ONLY on raw data),
run the leakage build gate, and write:

    outputs/nq_bars_sample.parquet
    outputs/nq_bars_sample_schema.md
    outputs/leakage_audit_nq_pipeline.json

Exit code != 0 if any gate check FAILs. Default window includes MLK 2024
(2024-01-15) to exercise holiday trade-date handling.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.audit.leakage import LeakageAuditor                     # noqa: E402
from tslab.audit.pipeline_gate import (BarIntegrityCheck,          # noqa: E402
                                       PrefixInvarianceCheck,
                                       RollCausalityCheck)
from tslab.config import load_config                               # noqa: E402
from tslab.data.nq_pipeline import build_nq_bars, load_pipeline_params  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-08")
    ap.add_argument("--end", default="2024-01-19")
    ap.add_argument("--no-quotes", action="store_true")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    cfg = load_config()
    params = load_pipeline_params(cfg.project_root / "configs")
    quotes = not args.no_quotes

    print(f"building NQ bars {start}..{end} (quotes={quotes}) ...", flush=True)
    full = build_nq_bars(cfg, start, end, include_quotes=quotes, params=params)
    print(f"  {len(full.bars):,} bars, sessions={len(full.stats['sessions'])}, "
          f"rolls={len(full.roll_log)}, skipped={full.stats['skipped_sessions']}")

    # prefix run: drop the last session, everything shared must be identical
    sessions = sorted(set(full.bars["session"]))
    prefix_end = sessions[-2] if len(sessions) > 1 else sessions[-1]
    print(f"prefix rebuild {start}..{prefix_end} for the truncation gate ...",
          flush=True)
    prefix = build_nq_bars(cfg, start, prefix_end, include_quotes=quotes,
                           params=params)

    auditor = (LeakageAuditor()
               .register(BarIntegrityCheck(full.bars, cfg, params.bar_seconds))
               .register(RollCausalityCheck(full.roll_log))
               .register(PrefixInvarianceCheck(full.bars, prefix.bars)))
    results = auditor.run_all()
    outdir = cfg.paths.outputs_dir
    LeakageAuditor.write_report(results, outdir, "nq_pipeline")
    for r in results:
        print(f"  [{r.status.value}] {r.name}: {r.message}")

    sample_path = outdir / "nq_bars_sample.parquet"
    full.bars.to_parquet(sample_path)
    write_schema_report(full, sample_path, outdir / "nq_bars_sample_schema.md",
                        start, end, quotes)
    print(f"wrote {sample_path}\nwrote {outdir / 'nq_bars_sample_schema.md'}")

    if not LeakageAuditor.passed(results):
        print("LEAKAGE GATE FAILED", file=sys.stderr)
        return 1
    return 0


def write_schema_report(res, parquet_path: Path, md_path: Path,
                        start, end, quotes: bool) -> None:
    b = res.bars
    lines = [
        "# NQ 1-minute bar sample — schema report", "",
        f"- window: {start} .. {end}   (quotes joined: {quotes})",
        f"- rows: {len(b):,}   sessions: {b['session'].nunique()}   "
        f"symbols: {sorted(b['symbol'].unique())}",
        f"- index: `bar_open_utc` ({b.index.dtype}), strictly increasing",
        f"- file: `{parquet_path}`", "",
        "## Columns", "", "| column | dtype | meaning |", "|---|---|---|",
    ]
    meaning = {
        "open": "first trade price in bar", "high": "max trade price",
        "low": "min trade price", "close": "last trade price",
        "volume": "total contracts traded", "n_trades": "trade prints",
        "buy_vol": "aggressor-buy volume (Databento side='B')",
        "sell_vol": "aggressor-sell volume (side='A')",
        "delta": "buy_vol - sell_vol ('N' prints excluded from both legs)",
        "rv": "sum of squared tick-to-tick log returns within bar",
        "vwap": "volume-weighted average trade price",
        "bid_close": "last bid price <= bar end (known at info_time)",
        "ask_close": "last ask price <= bar end",
        "bid_sz_close": "last bid size", "ask_sz_close": "last ask size",
        "spread_close_ticks": "last spread in ticks",
        "spread_mean_ticks": "mean spread over non-wide quote updates",
        "n_quotes": "top-of-book updates in bar",
        "session": "trade date (holiday-aware; MLK Monday books to Tuesday)",
        "symbol": "front-month contract (prior-session-volume roll)",
        "info_time": "bar_open + 60s — EARLIEST usable decision time",
        "secs_to_flatten": "seconds from info_time to the (early-close aware) "
                           "Topstep flatten deadline",
        "entry_ok": "entry window open, evaluated AT info_time",
        "rth": "bar open inside NQ RTH reference window on its session date",
    }
    for col in b.columns:
        lines.append(f"| {col} | {b[col].dtype} | {meaning.get(col, '')} |")
    lines += ["", "## Roll log", ""]
    for e in res.roll_log:
        lines.append(f"- {e['session']}: front = **{e['front']}** "
                     f"(prev {e['previous']}, decided from "
                     f"{e['decided_from_session']} volumes)")
    lines += ["", "## Filter stats", ""]
    for s in res.stats["trade_filter"]:
        lines.append(f"- trades {s}")
    for s in res.stats["quote_filter"][:5]:
        lines.append(f"- quotes {s}")
    lines += ["", "## Notes", "",
              "- Raw data opened read-only via pyarrow dataset scans; "
              "row-group pruning on ts_event.",
              "- Bad prints dropped: nonpositive px/size and >0.5% deviation "
              "from the rolling median of strictly prior trades.",
              "- Crossed quotes dropped; spreads > 25 ticks excluded from "
              "spread_mean_ticks.",
              "- Delta sign assumes Databento aggressor-side convention "
              "(B = buy aggressor).",
              "- MBP-1 coverage ends 2025-06-30; later windows produce "
              "trade-only bars (quote columns absent)."]
    md_path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
