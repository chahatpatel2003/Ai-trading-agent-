"""Walk-forward candidate search.

Protocol (leakage-safe by construction):
  1. folds = make_session_folds(sessions, n_train, n_test, embargo>=1) —
     validated by FoldEmbargoCheck before any evaluation.
  2. Candidates carry FIXED a-priori parameters (nothing is fitted inside a
     fold); "training" is pure candidate SELECTION on train sessions.
  3. Per fold: backtest on train sessions -> rank; backtest on test sessions
     only to report the selected candidates' out-of-sample results.
  4. The winner's OOS trades feed the lifecycle Monte Carlo for the payout-EV
     objective; its entry signal is re-audited with TruncationInvarianceCheck
     as a final gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..audit.leakage import (FoldEmbargoCheck, LeakageAuditor, Status,
                             TruncationInvarianceCheck)
from ..config import Config
from ..features.core import FeatureParams, compute_features
from ..features.labels import make_session_folds, required_embargo
from ..sim.costs import CostModel
from .backtest import BacktestResult, run_backtest
from .evaluate import Score, lifecycle_mc, simulate_account
from .rules import TradeRule


@dataclass
class SearchResult:
    scores: List[Score]
    best: Optional[Score]
    fold_report: List[Dict]
    audit_results: List
    audited_ok: bool


def _sessions_between(bars: pd.DataFrame, lo: pd.Timestamp,
                      hi: pd.Timestamp) -> List[date]:
    m = (bars.index >= lo) & (bars.index < hi)
    return sorted(set(bars.loc[m, "session"]))


def walkforward_search(candidates: Sequence[TradeRule], bars: pd.DataFrame,
                       feats: pd.DataFrame, cfg: Config,
                       n_train: int = 5, n_test: int = 2,
                       embargo_sessions: int = 1,
                       mc_runs: int = 200, mc_horizon: int = 250,
                       personal_dll: Optional[float] = None,
                       size: str = "50k", plan: str = "standard"
                       ) -> SearchResult:
    sessions = sorted(set(bars["session"]))
    folds = make_session_folds(sessions, cfg, n_train, n_test,
                               embargo_sessions)
    if not folds:
        raise ValueError("not enough sessions for a single fold")
    emb = required_embargo([])   # session-bounded labels -> 24h
    fold_check = FoldEmbargoCheck(folds, emb).run()
    if fold_check.status == Status.FAIL:
        raise ValueError(f"fold construction failed embargo check: "
                         f"{fold_check.message}")

    cm = CostModel(cfg)
    per_fold_train: Dict[str, List[float]] = {c.name: [] for c in candidates}
    per_fold_test: Dict[str, List[float]] = {c.name: [] for c in candidates}
    fold_report: List[Dict] = []

    def bt(rule: TradeRule, sess_list: List[date]) -> BacktestResult:
        m = bars["session"].isin(sess_list).to_numpy()
        return run_backtest(rule, bars[m], feats[m], cfg, cm)

    for k, f in enumerate(folds):
        train_sess = _sessions_between(bars, f["train_start"], f["train_end"])
        test_sess = _sessions_between(bars, f["test_start"], f["test_end"])
        assert not set(train_sess) & set(test_sess)
        ranked = []
        for c in candidates:
            tr = bt(c, train_sess)
            te = bt(c, test_sess)
            per_fold_train[c.name].append(tr.total_net)
            per_fold_test[c.name].append(te.total_net)
            ranked.append((c.name, tr.total_net, len(tr.trades),
                           te.total_net, len(te.trades)))
        ranked.sort(key=lambda r: r[1], reverse=True)
        fold_report.append({"fold": k, "train_sessions": train_sess,
                            "test_sessions": test_sess, "ranking": ranked})

    # select by MEAN TRAIN net (candidates with any train trades)
    def train_key(c: TradeRule) -> float:
        v = per_fold_train[c.name]
        return sum(v) / len(v) if v else float("-inf")

    ordered = sorted(candidates, key=train_key, reverse=True)
    scores: List[Score] = []
    best: Optional[Score] = None
    for c in ordered:
        oos_sessions: List[date] = []
        for f in folds:
            oos_sessions += _sessions_between(bars, f["test_start"],
                                              f["test_end"])
        oos = bt(c, sorted(set(oos_sessions)))
        pool = list(oos.by_session().values())
        life = lifecycle_mc(pool, cfg, size=size, plan=plan,
                            personal_dll=personal_dll, n_runs=mc_runs,
                            horizon_sessions=mc_horizon) if pool else None
        sc = Score.evaluate(c.name, per_fold_train[c.name],
                            per_fold_test[c.name], oos, life)
        scores.append(sc)
        if best is None and sc.feasible and life is not None:
            best = sc

    # final leakage gate on the selected candidate's entry signal
    audit_results = []
    audited_ok = True
    if best is not None:
        rule = next(c for c in candidates if c.name == best.rule_name)
        fp = FeatureParams()

        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            fe = compute_features(df, cfg, fp)
            return pd.DataFrame(
                {"sig": rule.entry_signal(fe, cfg).astype(float)},
                index=df.index)

        auditor = (LeakageAuditor()
                   .register(TruncationInvarianceCheck(bars, signal_fn,
                                                       n_probes=4)))
        audit_results = auditor.run_all()
        audit_results.append(fold_check)
        audited_ok = LeakageAuditor.passed(audit_results)
    return SearchResult(scores=scores, best=best, fold_report=fold_report,
                        audit_results=audit_results, audited_ok=audited_ok)
