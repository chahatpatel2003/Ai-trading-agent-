"""Phase 30 orchestrator — VP-feature scan + winner enhancement, resumable, honest.

Usage:
    python scripts/phase30_run.py scan | analyze | robust | all

Same protocol as Phase 29 (run each candidate ONCE over 2021-01..2026-06 through the audited engine
at 1 MNQ, EOD-flat; derive year/month/LOYO/walk-forward from stored trades; cost-sensitivity = top-N
re-run). The frame additionally carries the PIT-safe ``vp_*`` volume-profile columns.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29  # noqa: N812
from nqlab.discovery import phase29_stats as S  # noqa: N812
from nqlab.discovery import phase30 as P  # noqa: N812

START = date(2021, 1, 1)
END = date(2026, 6, 30)
TIMEFRAME = "5m"
RUN = "run1"


def _paths(config: Any) -> dict[str, Path]:
    base = config.data.artifacts_root / "phase30" / RUN
    base.mkdir(parents=True, exist_ok=True)
    (base / "trades").mkdir(exist_ok=True)
    return {
        "base": base,
        "trades": base / "trades",
        "frame": config.data.artifacts_root / "phase30" / f"frame_{TIMEFRAME}_full_vp.parquet",
        "summary": base / "scan_summary.jsonl",
        "candidates": base / "candidates.csv",
        "leaderboard": base / "leaderboard.csv",
        "rejected": base / "rejected.csv",
        "robust": base / "robust.json",
        "log": base / "progress.log",
    }


def _log(paths: dict[str, Path], msg: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with paths["log"].open("a") as fh:
        fh.write(line + "\n")


def _safe(name: str) -> str:
    return name.replace("|", "__").replace("/", "_").replace(":", "-")


def _read_parquet(path: Path, tries: int = 4) -> pl.DataFrame:
    last: Exception | None = None
    for _ in range(tries):
        try:
            return pl.read_parquet(path)
        except Exception as exc:
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"failed to read {path}: {last}")


def _ensure_frame(config: Any, paths: dict[str, Path]) -> pl.DataFrame:
    if paths["frame"].exists():
        t = time.time()
        fr = _read_parquet(paths["frame"])
        _log(
            paths,
            f"loaded cached VP frame rows={fr.height} cols={len(fr.columns)} "
            f"in {time.time() - t:.1f}s",
        )
        return fr
    _log(paths, f"building VP frame {TIMEFRAME} [{START}..{END}] …")
    t = time.time()
    fr = P.build_vp_frame(config, timeframe=TIMEFRAME, start=START, end=END)
    fr.write_parquet(paths["frame"])
    _log(
        paths,
        f"built+cached VP frame rows={fr.height} cols={len(fr.columns)} in {time.time() - t:.1f}s",
    )
    return fr


_intraday = P29.intraday_only


def scan(config: Any, rules: Any, paths: dict[str, Path]) -> None:
    fr = _ensure_frame(config, paths)
    cands = P.generate_candidates()
    _log(paths, f"scan: {len(cands)} candidates")
    done = {p.stem for p in paths["trades"].glob("*.parquet") if not p.name.startswith("._")}
    t0 = time.time()
    for i, c in enumerate(cands):
        stem = _safe(c.name)
        if stem in done:
            continue
        res = P.run_candidate(fr, c, config=config, rules=rules)
        tf = P29.trades_frame(res, c)
        tf.write_parquet(paths["trades"] / f"{stem}.parquet")
        with paths["summary"].open("a") as fh:
            fh.write(json.dumps({"candidate": c.name, "n_raw": tf.height}) + "\n")
        if (i + 1) % 50 == 0:
            _log(paths, f"  {i + 1}/{len(cands)} done ({(i + 1) / (time.time() - t0):.1f}/s)")
    _log(paths, f"scan complete in {time.time() - t0:.1f}s")


_MIN_TRADES = 100  # VP setups are rarer than Phase-29 indicators -> slightly lower floor
_MIN_YEARS = 3


def _candidate_metrics(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    files = sorted(p for p in paths["trades"].glob("*.parquet") if not p.name.startswith("._"))
    for f in files:
        raw = _read_parquet(f)
        if raw.height == 0:
            rows.append(
                {"candidate": f.stem.replace("__", "|"), "n": 0, "reject_reason": "no trades"}
            )
            continue
        meta = raw.row(0, named=True)
        kept, dropped = _intraday(raw)
        panel = S.compute_panel(kept)
        yr = S.by_year(kept)
        ms = S.month_stats(kept)
        rows.append(
            {
                "candidate": meta["candidate"],
                "family": meta["family"],
                "kind": meta["kind"],
                "cross_day_dropped": dropped,
                **panel,
                "years_traded": len(yr),
                "years_profitable": sum(1 for v in yr.values() if (v.get("net") or 0) > 0),
                "year_concentration": S.year_concentration(kept),
                "winning_month_pct": ms.get("winning_month_pct"),
                "red_months": ms.get("red_months"),
                "best_month": ms.get("best_month"),
                "worst_month": ms.get("worst_month"),
                "best_month_share": ms.get("best_month_share"),
                "yearly_net": {y: v.get("net") for y, v in yr.items()},
                "yearly_pf": {y: v.get("pf") for y, v in yr.items()},
                "yearly_wr": {y: v.get("win_rate") for y, v in yr.items()},
                "yearly_n": {y: v.get("n") for y, v in yr.items()},
            }
        )
    return rows


def _screen(r: dict[str, Any]) -> str | None:
    if r.get("n", 0) == 0:
        return "no trades"
    if (r.get("n") or 0) < _MIN_TRADES:
        return f"too few trades ({r.get('n')} < {_MIN_TRADES})"
    if (r.get("years_traded") or 0) < _MIN_YEARS:
        return f"too few years traded ({r.get('years_traded')} < {_MIN_YEARS})"
    if (r.get("net") or 0) <= 0:
        return "net negative over full history"
    pf = r.get("pf")
    if pf is None or pf <= 1.0:
        return f"PF <= 1.0 ({pf})"
    return None


def analyze(config: Any, paths: dict[str, Path]) -> None:
    rows = _candidate_metrics(paths)
    _log(paths, f"analyze: {len(rows)} candidate result files")
    for r in rows:
        r["reject_reason"] = _screen(r)
    passers = [r for r in rows if r["reject_reason"] is None]
    rejects = [r for r in rows if r["reject_reason"] is not None]
    passers.sort(
        key=lambda r: (
            -(r.get("pf") or 0),
            -(r.get("ret_over_maxdd") or 0),
            -(r.get("win_rate") or 0),
            (r.get("max_dd") or 1e18),
        )
    )
    cols = [
        "candidate",
        "family",
        "kind",
        "n",
        "net",
        "pf",
        "win_rate",
        "expectancy",
        "avg_win",
        "avg_loss",
        "max_dd",
        "avg_dd",
        "ret_over_maxdd",
        "sharpe",
        "sortino",
        "calmar",
        "max_losing_streak",
        "avg_mfe",
        "avg_mae",
        "mfe_mae",
        "trades_per_day",
        "years_traded",
        "years_profitable",
        "year_concentration",
        "winning_month_pct",
        "red_months",
        "best_month_share",
        "cross_day_dropped",
        "reject_reason",
    ]

    def _flat(rs: list[dict[str, Any]]) -> pl.DataFrame:
        return pl.DataFrame([{c: r.get(c) for c in cols} for r in rs], infer_schema_length=None)

    _flat([*passers, *rejects]).write_csv(paths["candidates"])
    _flat(passers).write_csv(paths["leaderboard"])
    _flat(rejects).sort("candidate").write_csv(paths["rejected"])
    (paths["base"] / "candidates_full.json").write_text(json.dumps(rows, indent=2, default=str))
    _log(paths, f"analyze: {len(passers)} passers, {len(rejects)} rejects")
    _log(paths, "TOP 15 by PF:")
    for r in passers[:15]:
        _log(
            paths,
            f"  PF={r.get('pf'):.2f} wr={(r.get('win_rate') or 0) * 100:.0f}% "
            f"n={r.get('n')} ret/DD={r.get('ret_over_maxdd')} DD=${r.get('max_dd'):.0f} "
            f"yrs+={r.get('years_profitable')}/{r.get('years_traded')} :: {r.get('candidate')}",
        )


def _panel_period(trades: pl.DataFrame, y0: int, y1: int) -> dict[str, Any]:
    t = trades.with_columns(pl.col("entry_ts").dt.year().alias("_y"))
    return S.compute_panel(t.filter(pl.col("_y").is_between(y0, y1)).drop("_y"))


def robust(config: Any, rules: Any, paths: dict[str, Path], top_n: int = 30) -> None:
    lb = pl.read_csv(paths["leaderboard"])
    if lb.height == 0:
        paths["robust"].write_text(json.dumps([], indent=2))
        _log(paths, "robust: empty leaderboard")
        return
    top = lb.head(top_n)["candidate"].to_list()
    cmap = {c.name: c for c in P.generate_candidates()}
    fr = _ensure_frame(config, paths)
    years = list(range(2021, 2027))
    out: list[dict[str, Any]] = []
    for name in top:
        c = cmap[name]
        kept, _ = _intraday(_read_parquet(paths["trades"] / f"{_safe(name)}.parquet"))
        loyo = {y: _panel_period(kept, y, y) for y in years}
        wfo = {f"test_{y}": _panel_period(kept, y, y) for y in range(2022, 2027)}
        cost = {}
        for mult in (0.0, 0.5, 1.0, 2.0):
            res = P.run_candidate(
                fr, c, config=config, rules=rules, slippage_mult=mult, commission_mult=mult
            )
            k2, _ = _intraday(P29.trades_frame(res, c))
            p = S.compute_panel(k2)
            cost[f"{mult}x"] = {k: p.get(k) for k in ("n", "pf", "win_rate", "net", "max_dd")}
        out.append({"candidate": name, "loyo": loyo, "walk_forward": wfo, "cost_sensitivity": cost})
        _log(paths, f"robust done: {name}")
    paths["robust"].write_text(json.dumps(out, indent=2, default=str))
    _log(paths, f"robust complete for {len(out)} finalists")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    config = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    paths = _paths(config)
    _log(paths, f"=== Phase 30 {mode} === frame=[{START}..{END}] tf={TIMEFRAME}")
    if mode in ("scan", "all"):
        scan(config, rules, paths)
    if mode in ("analyze", "all"):
        analyze(config, paths)
    if mode in ("robust", "all"):
        robust(config, rules, paths)


if __name__ == "__main__":
    main()
