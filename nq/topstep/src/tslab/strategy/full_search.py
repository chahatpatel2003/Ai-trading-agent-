"""Phase 3C full-history walk-forward search over the verified fold plan.

Protocol (leakage-safe by construction, no fitting inside folds):
  1. The 17-fold plan is loaded from outputs/fold_plan.json and RE-verified
     (embargo intervals + exact label purge) before any evaluation.
  2. Every candidate carries fixed a-priori parameters. Per fold, candidates
     are ranked on TRAIN sessions only (daily-P&L t-stat over all train
     sessions, zeros included; minimum train trades; positive net).
  3. The headline result is the WALK-FORWARD STREAM: each fold's train-winner
     evaluated on that fold's held-out test quarter, concatenated. Per-
     candidate aggregate OOS is reported only as stability evidence.
  4. Lifecycle Monte Carlo (TopstepSimulator) runs only for the walk-forward
     stream and the top-K global candidates.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..audit.leakage import FoldEmbargoCheck, Status, TruncationInvarianceCheck
from ..config import Config
from ..features.core import FeatureParams, compute_features
from ..features.labels import check_label_purge
from ..sim.costs import CostModel
from .backtest import BacktestResult, Trade, run_backtest
from .evaluate import LifecycleStats, lifecycle_mc, max_drawdown
from .rules import TradeRule

MIN_TRAIN_TRADES = 60


def load_fold_plan(path: Path) -> List[Dict[str, pd.Timestamp]]:
    raw = json.loads(Path(path).read_text())
    return [{k: pd.Timestamp(f[k]) for k in
             ("train_start", "train_end", "test_start", "test_end")}
            for f in raw["folds"]]


def sessions_in(bars: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp
                ) -> List[date]:
    m = (bars.index >= lo) & (bars.index < hi)
    return sorted(set(bars.loc[m, "session"]))


def train_tstat(daily: pd.Series, n_sessions: int) -> float:
    """t-stat of mean daily P&L over ALL train sessions (flat days = 0)."""
    if n_sessions < 2:
        return float("-inf")
    mu = daily.sum() / n_sessions
    var = (daily ** 2).sum() / n_sessions - mu ** 2
    if var <= 0:
        return float("-inf")
    return float(mu / math.sqrt(var / n_sessions))


@dataclass
class FoldCandidate:
    name: str
    train_net: float
    train_trades: int
    train_tstat: float
    test_net: float
    test_trades: int
    test_trade_list: List[Trade] = field(default_factory=list)


def evaluate_fold(grid: Sequence[TradeRule], bars: pd.DataFrame,
                  feats: pd.DataFrame, fold: Dict[str, pd.Timestamp],
                  cfg: Config, cm: CostModel,
                  instrument: str = "NQ") -> Dict[str, FoldCandidate]:
    tr_m = np.asarray((bars.index >= fold["train_start"])
                      & (bars.index < fold["train_end"]))
    te_m = np.asarray((bars.index >= fold["test_start"])
                      & (bars.index < fold["test_end"]))
    btr, ftr = bars[tr_m], feats[tr_m]
    bte, fte = bars[te_m], feats[te_m]
    tr_sessions = sorted(set(btr["session"]))
    assert not set(tr_sessions) & set(bte["session"])   # hard leakage guard
    out: Dict[str, FoldCandidate] = {}
    for rule in grid:
        r_tr = run_backtest(rule, btr, ftr, cfg, cm, instrument=instrument)
        r_te = run_backtest(rule, bte, fte, cfg, cm, instrument=instrument)
        assert r_tr.violations == 0 and r_te.violations == 0
        out[rule.name] = FoldCandidate(
            name=rule.name, train_net=r_tr.total_net,
            train_trades=len(r_tr.trades),
            train_tstat=train_tstat(r_tr.daily_net(), len(tr_sessions)),
            test_net=r_te.total_net, test_trades=len(r_te.trades),
            test_trade_list=r_te.trades)
    return out


def select_winner(fold_result: Dict[str, FoldCandidate],
                  min_trades: int = MIN_TRAIN_TRADES) -> Optional[str]:
    """Train-only selection: highest train t-stat among candidates with
    enough train trades and positive train net. None = stay flat."""
    best, best_key = None, (float("-inf"), float("-inf"))
    for name, fc in fold_result.items():
        if fc.train_trades < min_trades or fc.train_net <= 0:
            continue
        key = (fc.train_tstat, fc.train_net)
        if key > best_key:
            best, best_key = name, key
    return best


@dataclass
class SearchOutput:
    fold_table: List[Dict]
    wf_sessions: List[date]
    wf_trades_by_session: Dict[date, List[Trade]]
    wf_daily: pd.Series
    candidate_summary: pd.DataFrame
    top_global: List[str]
    top_candidate_trades: Dict[str, Dict[date, List[Trade]]]
    violations: int


def run_search(grid: Sequence[TradeRule], bars: pd.DataFrame,
               feats: pd.DataFrame, folds: List[Dict], cfg: Config,
               progress=print, min_trades: int = MIN_TRAIN_TRADES,
               instrument: str = "NQ") -> SearchOutput:
    cm = CostModel(cfg)
    fold_table: List[Dict] = []
    wf_trades: Dict[date, List[Trade]] = {}
    wf_sessions: List[date] = []
    agg: Dict[str, Dict] = {r.name: {"train_tstats": [], "test_nets": [],
                                     "test_trades": 0, "trades": {}}
                            for r in grid}
    for k, fold in enumerate(folds):
        import time as _t
        t0 = _t.time()
        res = evaluate_fold(grid, bars, feats, fold, cfg, cm,
                            instrument=instrument)
        winner = select_winner(res, min_trades=min_trades)
        te_sessions = sessions_in(bars, fold["test_start"], fold["test_end"])
        wf_sessions += te_sessions
        for s in te_sessions:
            wf_trades[s] = []
        if winner is not None:
            for t in res[winner].test_trade_list:
                wf_trades[t.session].append(t)
        for name, fc in res.items():
            a = agg[name]
            a["train_tstats"].append(fc.train_tstat)
            a["test_nets"].append(fc.test_net)
            a["test_trades"] += fc.test_trades
            for t in fc.test_trade_list:
                a["trades"].setdefault(t.session, []).append(t)
        w = res[winner] if winner else None
        fold_table.append({
            "fold": k, "winner": winner,
            "train_tstat": round(w.train_tstat, 2) if w else None,
            "train_net": round(w.train_net) if w else None,
            "train_trades": w.train_trades if w else 0,
            "test_net": round(w.test_net) if w else 0,
            "test_trades": w.test_trades if w else 0,
            "test_start": str(fold["test_start"].date()),
            "test_end": str(fold["test_end"].date()),
        })
        progress(f"[fold {k:02d}] winner={winner}  "
                 f"train_t={fold_table[-1]['train_tstat']}  "
                 f"test_net={fold_table[-1]['test_net']}  "
                 f"({_t.time() - t0:.0f}s)")

    wf_daily = pd.Series(
        {s: sum(t.net_usd for t in wf_trades[s]) for s in wf_sessions},
        dtype=float).sort_index()

    rows = []
    for name, a in agg.items():
        tstats = [t for t in a["train_tstats"] if np.isfinite(t)]
        nets = a["test_nets"]
        all_t = [t for lst in a["trades"].values() for t in lst]
        wins = sum(1 for t in all_t if t.net_usd > 0)
        rows.append({
            "name": name,
            "mean_train_tstat": np.mean(tstats) if tstats else float("-inf"),
            "oos_net": float(np.sum(nets)),
            "oos_trades": a["test_trades"],
            "oos_win_rate": (wins / len(all_t)) if all_t else float("nan"),
            "trades_per_session": (len(all_t) / max(len(wf_sessions), 1)),
            "avg_win": (float(np.mean([t.net_usd for t in all_t
                                       if t.net_usd > 0])) if wins else 0.0),
            "avg_loss": (float(np.mean([t.net_usd for t in all_t
                                        if t.net_usd <= 0]))
                         if len(all_t) > wins else 0.0),
            "pos_fold_share": float(np.mean([n > 0 for n in nets])),
            "n_folds_qualified": len(tstats),
        })
    summary = (pd.DataFrame(rows)
               .sort_values("mean_train_tstat", ascending=False)
               .reset_index(drop=True))
    top_global = list(summary.head(5)["name"])
    top_trades: Dict[str, Dict[date, List[Trade]]] = {}
    for name in top_global:
        per = {s: [] for s in wf_sessions}     # include flat OOS sessions
        per.update(agg[name]["trades"])
        top_trades[name] = per
    return SearchOutput(fold_table=fold_table, wf_sessions=wf_sessions,
                        wf_trades_by_session=wf_trades, wf_daily=wf_daily,
                        candidate_summary=summary, top_global=top_global,
                        top_candidate_trades=top_trades, violations=0)


# --------------------------------------------------------------------------- #
# verification helpers
# --------------------------------------------------------------------------- #

def candidate_gates(row: Dict, min_win: float = 0.55, min_tps: float = 1.0,
                    min_share: float = 0.5) -> bool:
    """Pre-registered candidate-level qualification for tail confirmation:
    aggregate-OOS win rate, frequency, profitability, fold stability."""
    return bool(row["oos_win_rate"] >= min_win
                and row["trades_per_session"] >= min_tps
                and row["oos_net"] > 0
                and row["pos_fold_share"] >= min_share)


def tail_split(all_sessions: Sequence[date], tail_start: date,
               n_train: int = 250, embargo: int = 1
               ) -> Tuple[List[date], List[date]]:
    """Train sessions = the n_train sessions ending `embargo` sessions before
    the first reserved-tail session; tail = all sessions >= tail_start."""
    tail = [s for s in all_sessions if s >= tail_start]
    pre = [s for s in all_sessions if s < tail_start]
    if len(pre) < n_train + embargo:
        raise ValueError("not enough pre-tail sessions")
    train = pre[-(n_train + embargo):-embargo] if embargo else pre[-n_train:]
    return train, tail


def verify_fold_plan(folds: List[Dict], end_times: pd.DataFrame,
                     embargo: pd.Timedelta) -> List:
    results = [FoldEmbargoCheck(folds, embargo).run()]
    purge = check_label_purge(end_times, folds)
    from ..audit.leakage import CheckResult
    results.append(CheckResult(
        "label_purge_per_fold", Status.PASS if not purge else Status.FAIL,
        "no train label ends past train_end" if not purge
        else f"{len(purge)} violations"))
    return results


def audit_signal(rule: TradeRule, bars: pd.DataFrame, cfg: Config,
                 n_probes: int = 4):
    fp = FeatureParams()

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        fe = compute_features(df, cfg, fp)
        return pd.DataFrame({"sig": rule.entry_signal(fe, cfg).astype(float)},
                            index=df.index)

    r = TruncationInvarianceCheck(bars, signal_fn, n_probes=n_probes).run()
    r.name = f"signal_truncation:{rule.name}"
    return r


def mc_for_pool(trades_by_session: Dict[date, List[Trade]], cfg: Config,
                size: str = "50k", plan: str = "standard",
                personal_dll: Optional[float] = 1000.0,
                n_runs: int = 400, horizon: int = 500,
                seed: int = 29) -> LifecycleStats:
    pool = [trades_by_session[s] for s in sorted(trades_by_session)]
    return lifecycle_mc(pool, cfg, size=size, plan=plan,
                        personal_dll=personal_dll, n_runs=n_runs,
                        horizon_sessions=horizon, seed=seed)
