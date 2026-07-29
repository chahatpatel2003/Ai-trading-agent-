"""Leakage audit framework.

Every strategy/feature pipeline must pass this auditor before its backtest
numbers are believed. The framework is check-based: checks are registered
with the auditor and produce PASS / FAIL / WARN / SKIP results plus details.

Implemented checks (real, testable now):

  TruncationInvarianceCheck
      The workhorse. For probe timestamps t: features computed on data[:t]
      must equal the FULL PREFIX of features computed on the whole history —
      every row <= t, not just the row at t, so leaks confined to the warm-up
      region (e.g. bfill of rolling windows) are caught too. Catches centered
      windows, full-sample normalization/statistics, sorting on future info,
      target leakage through joins. Vacuous comparisons (both NaN) are
      counted; a check with zero effective comparisons WARNs instead of
      passing. Non-numeric columns are compared by equality.

  FoldEmbargoCheck
      Walk-forward folds must be temporally ordered, non-overlapping (train
      vs its own test AND test-vs-test across folds), separated by an
      embargo >= the label horizon. When given the bar index and a horizon
      in ROWS, the required embargo becomes gap-aware: the max wall-clock
      span any `horizon_rows` consecutive bars cover (sessions/weekends make
      row-horizons much longer in time than bar_minutes * rows).

  LabelHorizonCheck
      Fixed-horizon mode (horizon_rows): labels claimed to use only (t, t+h]
      must be reproducible from data truncated at t+h and NaN for the last h
      rows. Variable-horizon mode (label_end_times): each non-NaN label must
      have its end time within the available data and be reproducible from
      data truncated at its end time (supports "return to session close").

Skeleton checks (interface fixed, marked SKIP until wired to a model layer):

  SessionBoundaryCheck   — no same-session feature uses post-cutoff bars
  ShuffledTargetCheck    — model trained on shuffled labels must score ~chance
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class LeakageCheck:
    """Base class. Subclasses implement run() and set a unique name."""
    name: str = "base"

    def run(self) -> CheckResult:  # pragma: no cover - abstract
        raise NotImplementedError


def _compare_series(a: pd.Series, b: pd.Series, rtol: float, atol: float
                    ) -> Tuple[np.ndarray, int]:
    """Elementwise match mask + count of effective (non-both-NaN) comparisons.
    Numeric dtypes use isclose; everything else uses equality."""
    a_na, b_na = pd.isna(a.values), pd.isna(b.values)
    both_nan = a_na & b_na
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        ok = np.isclose(a.values.astype(float), b.values.astype(float),
                        rtol=rtol, atol=atol, equal_nan=True)
    else:
        ok = (a.values == b.values) | both_nan
    effective = int((~both_nan).sum())
    return ok, effective


# --------------------------------------------------------------------------- #
# concrete checks
# --------------------------------------------------------------------------- #

class TruncationInvarianceCheck(LeakageCheck):
    """feature_fn(df) -> DataFrame aligned to df.index. For each probe time t,
    recompute on df[df.index <= t]; EVERY row of the truncated result must
    match the corresponding prefix of the full-sample result. Any mismatch =
    some feature value depends on data after t."""

    name = "truncation_invariance"

    def __init__(self, df: pd.DataFrame, feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
                 n_probes: int = 8, rtol: float = 1e-9, atol: float = 1e-12,
                 seed: int = 7):
        self.df = df
        self.feature_fn = feature_fn
        self.n_probes = n_probes
        self.rtol, self.atol = rtol, atol
        self.seed = seed

    def run(self) -> CheckResult:
        if len(self.df) < 10:
            return CheckResult(self.name, Status.SKIP, "not enough rows to probe")
        full = self.feature_fn(self.df)
        if not full.index.equals(self.df.index):
            return CheckResult(self.name, Status.FAIL,
                               "feature_fn output index does not match input index")
        rng = np.random.default_rng(self.seed)
        lo = max(5, len(self.df) // 10)
        rand = rng.choice(np.arange(lo, len(self.df)),
                          size=min(self.n_probes, len(self.df) - lo),
                          replace=False)
        # fixed EARLY probes inside the warm-up region: leaks like bfill of a
        # rolling window are only visible when the truncation is shorter than
        # the warm-up itself (full-prefix comparison at late probes cannot
        # see them because the backfill source already exists)
        early = [p for p in (5, 10, lo // 2) if 0 < p < len(self.df)]
        probe_pos = sorted({int(p) for p in rand} | set(early))
        bad: List[Dict[str, Any]] = []
        n_effective = 0
        for pos in probe_pos:
            t = self.df.index[pos]
            trunc = self.feature_fn(self.df[self.df.index <= t])
            prefix = full.iloc[: pos + 1]
            if not trunc.index.equals(prefix.index):
                return CheckResult(self.name, Status.FAIL,
                                   "feature_fn output index does not match "
                                   "input index under truncation")
            for col in full.columns:
                ok, eff = _compare_series(prefix[col], trunc[col],
                                          self.rtol, self.atol)
                n_effective += eff
                if not ok.all():
                    for row_pos in np.flatnonzero(~ok)[:3]:
                        bad.append({
                            "probe_time": str(t), "column": str(col),
                            "row_time": str(prefix.index[row_pos]),
                            "full_sample_value": _f(prefix[col].iloc[row_pos]),
                            "truncated_value": _f(trunc[col].iloc[row_pos]),
                        })
        if bad:
            return CheckResult(
                self.name, Status.FAIL,
                f"{len(bad)} feature value(s) change when the future is removed "
                "— features depend on future data",
                {"mismatches": bad[:20], "n_probes": len(probe_pos)})
        if n_effective == 0:
            return CheckResult(
                self.name, Status.WARN,
                "all comparisons were vacuous (both-NaN) — feature too sparse "
                "for this probe set; increase probes or probe non-NaN rows")
        return CheckResult(self.name, Status.PASS,
                           f"features identical under truncation at "
                           f"{len(probe_pos)} probes "
                           f"({n_effective} effective comparisons)")


class FoldEmbargoCheck(LeakageCheck):
    """folds: sequence of dicts with train_start/train_end/test_start/test_end
    (tz-aware timestamps). embargo: minimum gap between train_end and
    test_start (>= label horizon in TIME).

    Optional gap-awareness: pass bar_index (the actual bar DatetimeIndex) and
    horizon_rows (label horizon in rows) and the required embargo becomes
    max(embargo, max wall-clock span of any horizon_rows consecutive bars) —
    a row-based label crossing a weekend spans days, not minutes.
    """

    name = "fold_embargo"

    def __init__(self, folds: Sequence[Dict[str, pd.Timestamp]], embargo: pd.Timedelta,
                 bar_index: Optional[pd.DatetimeIndex] = None,
                 horizon_rows: Optional[int] = None):
        self.folds = list(folds)
        self.embargo = embargo
        self.bar_index = bar_index
        self.horizon_rows = horizon_rows

    def _required_embargo(self) -> pd.Timedelta:
        req = self.embargo
        if self.bar_index is not None and self.horizon_rows:
            idx = pd.DatetimeIndex(self.bar_index)
            h = self.horizon_rows
            if len(idx) > h:
                spans = idx[h:] - idx[:-h]
                req = max(req, spans.max())
        return req

    def run(self) -> CheckResult:
        required = self._required_embargo()
        problems: List[str] = []
        warnings: List[str] = []
        for i, f in enumerate(self.folds):
            tr0, tr1 = f["train_start"], f["train_end"]
            te0, te1 = f["test_start"], f["test_end"]
            if not (tr0 < tr1 and te0 < te1):
                problems.append(f"fold {i}: degenerate interval")
                continue
            if tr1 > te0:
                problems.append(f"fold {i}: train overlaps test ({tr1} > {te0})")
            elif te0 - tr1 < required:
                problems.append(
                    f"fold {i}: embargo violated (gap {te0 - tr1} < required "
                    f"{required})")
            if tr0 >= te0:
                problems.append(f"fold {i}: train starts after test starts")
        # cross-fold: test windows must not overlap each other (double-counted
        # OOS periods inflate aggregate statistics)
        order = sorted(range(len(self.folds)),
                       key=lambda i: self.folds[i]["test_start"])
        if order != list(range(len(self.folds))):
            warnings.append("folds are not listed in temporal test order")
        for a, b in zip(order, order[1:]):
            fa, fb = self.folds[a], self.folds[b]
            if fa["test_end"] > fb["test_start"]:
                problems.append(
                    f"folds {a} and {b}: test windows overlap "
                    f"({fa['test_end']} > {fb['test_start']})")
        if problems:
            return CheckResult(self.name, Status.FAIL,
                               f"{len(problems)} fold problem(s)",
                               {"problems": problems, "warnings": warnings,
                                "required_embargo": str(required)})
        if warnings:
            return CheckResult(self.name, Status.WARN, "; ".join(warnings),
                               {"required_embargo": str(required)})
        return CheckResult(self.name, Status.PASS,
                           f"{len(self.folds)} folds ordered, train/test and "
                           f"test/test non-overlapping, embargo >= {required}")


class LabelHorizonCheck(LeakageCheck):
    """Two modes:

    Fixed horizon (horizon_rows=h): label_fn(df) uses only (t, t+h] info.
    Verifies (a) the last h labels are NaN and (b) labels are unchanged when
    data beyond t+h rows is discarded.

    Variable horizon (label_end_times): a Series aligned to df.index giving
    each label's INFORMATION end time — the closing time of the last bar the
    label uses (bar open + bar duration), not the open stamp. Verifies
    (a) every non-NaN label ends within the frame's information horizon
    (data_end_time; inferred as last open + one index step when omitted) and
    (b) each probed label is reproducible from df truncated to bars that are
    fully closed by its own end time (strict cut on open stamps).
    """

    name = "label_horizon"

    def __init__(self, df: pd.DataFrame, label_fn: Callable[[pd.DataFrame], pd.Series],
                 horizon_rows: Optional[int] = None,
                 label_end_times: Optional[pd.Series] = None,
                 data_end_time: Optional[pd.Timestamp] = None,
                 n_probes: int = 5, seed: int = 11):
        if (horizon_rows is None) == (label_end_times is None):
            raise ValueError("pass exactly one of horizon_rows / label_end_times")
        self.df = df
        self.label_fn = label_fn
        self.h = horizon_rows
        self.end_times = label_end_times
        self.data_end_time = data_end_time
        self.n_probes = n_probes
        self.seed = seed

    def run(self) -> CheckResult:
        labels = self.label_fn(self.df)
        if not labels.index.equals(self.df.index):
            return CheckResult(self.name, Status.FAIL, "label index mismatch")
        if self.end_times is not None:
            return self._run_variable(labels)
        return self._run_fixed(labels)

    def _run_fixed(self, labels: pd.Series) -> CheckResult:
        h = self.h
        tail = labels.iloc[-h:]
        if h > 0 and tail.notna().any():
            return CheckResult(
                self.name, Status.FAIL,
                f"labels within the final horizon window are non-NaN "
                f"({int(tail.notna().sum())} of {h}) — horizon is understated")
        rng = np.random.default_rng(self.seed)
        n = len(self.df)
        lo = max(1, n // 10)
        hi = n - h - 1
        if hi <= lo:
            return CheckResult(self.name, Status.SKIP, "sample too short for probes")
        bad = []
        for pos in sorted(rng.choice(np.arange(lo, hi),
                                     size=min(self.n_probes, hi - lo), replace=False)):
            cut = self.df.iloc[: pos + h + 1]        # keep exactly t..t+h
            lab_cut = self.label_fn(cut)
            t = self.df.index[pos]
            a, b = labels.loc[t], lab_cut.loc[t]
            if _both_nan(a, b):
                continue
            if not np.isclose(a, b, equal_nan=True):
                bad.append({"probe_time": str(t), "full": _f(a), "truncated": _f(b)})
        if bad:
            return CheckResult(self.name, Status.FAIL,
                               "labels depend on data beyond the stated horizon",
                               {"mismatches": bad})
        return CheckResult(self.name, Status.PASS,
                           f"labels respect horizon of {h} rows")

    def _run_variable(self, labels: pd.Series) -> CheckResult:
        ends = self.end_times
        if not ends.index.equals(self.df.index):
            return CheckResult(self.name, Status.FAIL,
                               "label_end_times index mismatch")
        if self.data_end_time is not None:
            horizon = self.data_end_time
        else:
            step = (self.df.index[-1] - self.df.index[-2]
                    if len(self.df) > 1 else pd.Timedelta(0))
            horizon = self.df.index[-1] + step
        overrun = labels.notna() & (ends > horizon)
        if overrun.any():
            return CheckResult(
                self.name, Status.FAIL,
                f"{int(overrun.sum())} non-NaN label(s) claim an end time "
                "beyond the available data")
        candidates = np.flatnonzero(labels.notna().values)
        if len(candidates) == 0:
            return CheckResult(self.name, Status.WARN, "no non-NaN labels to probe")
        rng = np.random.default_rng(self.seed)
        picks = sorted(rng.choice(candidates,
                                  size=min(self.n_probes, len(candidates)),
                                  replace=False))
        bad = []
        for pos in picks:
            t = self.df.index[pos]
            # bars fully closed by the label's end time: open stamp strictly
            # before a close-time end (60s-grid bars => open < end ⇔ close <= end)
            cut = self.df[self.df.index < ends.iloc[pos]]
            lab_cut = self.label_fn(cut)
            if t not in lab_cut.index:
                bad.append({"probe_time": str(t), "error": "row missing after cut"})
                continue
            a, b = labels.loc[t], lab_cut.loc[t]
            if _both_nan(a, b):
                continue
            if not np.isclose(a, b, equal_nan=True):
                bad.append({"probe_time": str(t), "full": _f(a), "truncated": _f(b)})
        if bad:
            return CheckResult(self.name, Status.FAIL,
                               "labels depend on data beyond their declared "
                               "end times", {"mismatches": bad})
        return CheckResult(self.name, Status.PASS,
                           f"labels respect their declared end times "
                           f"({len(picks)} probes)")


class SessionBoundaryCheck(LeakageCheck):
    """SKELETON: verify no feature used for a trade in session S incorporates
    bars stamped after the trade's entry time or after S's flatten cutoff.
    Needs the (future) trade log + feature matrix linkage; interface fixed now
    so pipelines are built against it."""

    name = "session_boundary"

    def __init__(self, trades: Optional[pd.DataFrame] = None,
                 features: Optional[pd.DataFrame] = None):
        self.trades, self.features = trades, features

    def run(self) -> CheckResult:
        if self.trades is None or self.features is None:
            return CheckResult(self.name, Status.SKIP,
                               "no trade log yet (strategy layer not built)")
        raise NotImplementedError("wire up when the strategy layer exists")


class ShuffledTargetCheck(LeakageCheck):
    """SKELETON: train_and_score(X, y) -> float metric. With y shuffled, the
    metric must collapse to ~chance; if it doesn't, features encode the label
    (leakage) or the CV is broken. Interface fixed; wired when a model layer
    exists."""

    name = "shuffled_target"

    def __init__(self, train_and_score: Optional[Callable] = None,
                 X: Optional[pd.DataFrame] = None, y: Optional[pd.Series] = None,
                 chance_level: float = 0.0, tolerance: float = 0.05):
        self.train_and_score = train_and_score
        self.X, self.y = X, y
        self.chance_level, self.tolerance = chance_level, tolerance

    def run(self) -> CheckResult:
        if self.train_and_score is None:
            return CheckResult(self.name, Status.SKIP,
                               "no model layer yet (foundation stage)")
        raise NotImplementedError("wire up when the model layer exists")


# --------------------------------------------------------------------------- #
# auditor
# --------------------------------------------------------------------------- #

class LeakageAuditor:
    def __init__(self) -> None:
        self.checks: List[LeakageCheck] = []

    def register(self, check: LeakageCheck) -> "LeakageAuditor":
        self.checks.append(check)
        return self

    def run_all(self) -> List[CheckResult]:
        results = []
        for c in self.checks:
            try:
                results.append(c.run())
            except Exception as e:
                results.append(CheckResult(c.name, Status.FAIL,
                                           f"check crashed: {type(e).__name__}: {e}"))
        return results

    @staticmethod
    def passed(results: Sequence[CheckResult]) -> bool:
        return not any(r.status == Status.FAIL for r in results)

    @staticmethod
    def write_report(results: Sequence[CheckResult], outputs_dir: Path,
                     stamp: str = "latest") -> Path:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        path = outputs_dir / f"leakage_audit_{stamp}.json"
        with open(path, "w") as f:
            json.dump([{"name": r.name, "status": r.status.value,
                        "message": r.message, "details": r.details}
                       for r in results], f, indent=2, default=str)
        return path


def _both_nan(a: Any, b: Any) -> bool:
    try:
        return bool(np.isnan(a) and np.isnan(b))
    except TypeError:
        return False


def _f(x: Any) -> Any:
    try:
        return float(x)
    except (TypeError, ValueError):
        return str(x)
