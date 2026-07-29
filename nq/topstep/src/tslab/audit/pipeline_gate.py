"""Build gate for the NQ bar pipeline: leakage/integrity checks that must
PASS before a bar file is allowed to feed research. Runs inside the sample
builder and the test suite via the LeakageAuditor framework."""
from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .. import sessions as S
from ..config import Config
from .leakage import CheckResult, LeakageCheck, Status


class BarIntegrityCheck(LeakageCheck):
    """Structural discipline of a bar frame:
      - tz-aware UTC index, strictly increasing, unique
      - info_time == bar_open + bar_seconds for EVERY row (the earliest
        usable decision time)
      - every bar lies inside its session's [open, globex_close) window and
        the session is a valid trade date (weekday, non-holiday)
      - entry_ok is False at/after the (early-close aware) flatten deadline
      - entry_ok matches in_entry_window evaluated AT info_time (sampled)
    """

    name = "bar_integrity"

    def __init__(self, bars: pd.DataFrame, cfg: Config, bar_seconds: int = 60,
                 entry_sample: int = 2000, seed: int = 5):
        self.bars, self.cfg = bars, cfg
        self.bar_seconds = bar_seconds
        self.entry_sample, self.seed = entry_sample, seed

    def run(self) -> CheckResult:
        b, sc = self.bars, self.cfg.sessions
        problems: List[str] = []
        if b.empty:
            return CheckResult(self.name, Status.FAIL, "bar frame is empty")
        idx = b.index
        if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
            return CheckResult(self.name, Status.FAIL, "index not tz-aware")
        if not idx.is_monotonic_increasing:
            problems.append("index not monotonic increasing")
        if idx.duplicated().any():
            problems.append(f"{int(idx.duplicated().sum())} duplicate bar stamps")
        want = idx + pd.Timedelta(seconds=self.bar_seconds)
        if not (pd.DatetimeIndex(b["info_time"]) == want).all():
            problems.append("info_time != bar_open + bar_seconds somewhere")

        bad_sessions = set()
        for sess, g in b.groupby("session", sort=False):
            if sess.weekday() >= 5 or sess in sc.holidays:
                problems.append(f"session {sess} is not a valid trade date")
                bad_sessions.add(sess)
                continue
            lo = S.session_open_utc(sess, sc)
            hi = S.globex_close_utc(sess, sc)
            if g.index.min() < lo or g.index.max() >= hi:
                problems.append(f"session {sess}: bars outside [open, close)")
            if S.session_date(g.index.min().to_pydatetime(), sc) != sess:
                problems.append(f"session {sess}: first bar maps elsewhere")
            flatten = S.eod_flatten_utc(sess, sc)
            late = g[pd.DatetimeIndex(g["info_time"]) >= flatten]
            if late["entry_ok"].any():
                problems.append(f"session {sess}: entry_ok past flatten deadline")

        rng = np.random.default_rng(self.seed)
        picks = rng.choice(len(b), size=min(self.entry_sample, len(b)),
                           replace=False)
        for i in picks:
            row = b.iloc[i]
            if row["session"] in bad_sessions:
                continue
            t = pd.Timestamp(row["info_time"]).to_pydatetime()
            expect = (S.in_entry_window(t, sc)
                      and pd.Timestamp(row["info_time"])
                      < S.eod_flatten_utc(row["session"], sc))
            if bool(row["entry_ok"]) != expect:
                problems.append(f"entry_ok mismatch at {b.index[i]}")
                break

        if problems:
            return CheckResult(self.name, Status.FAIL,
                               f"{len(problems)} integrity problem(s)",
                               {"problems": problems[:20]})
        return CheckResult(self.name, Status.PASS,
                           f"{len(b)} bars structurally sound "
                           f"({len(picks)} entry_ok spot checks)")


class RollCausalityCheck(LeakageCheck):
    """Roll decisions must be made strictly from earlier sessions and the
    front contract must never move backwards."""

    name = "roll_causality"

    def __init__(self, roll_log: List[Dict]):
        self.roll_log = roll_log

    def run(self) -> CheckResult:
        from ..data.nq_pipeline import parse_contract
        problems = []
        prev = None
        for e in self.roll_log:
            if e["decided_from_session"] is None:
                problems.append(f"{e['session']}: roll with no prior session")
            elif e["decided_from_session"] >= e["session"]:
                problems.append(f"{e['session']}: decided from "
                                f"{e['decided_from_session']} (not strictly prior)")
            c = parse_contract(e["front"], e["session"].year)
            if c is None:
                problems.append(f"{e['session']}: unparseable front {e['front']}")
            elif prev is not None and c < prev:
                problems.append(f"{e['session']}: rolled backwards to {e['front']}")
            else:
                prev = c
        if problems:
            return CheckResult(self.name, Status.FAIL,
                               f"{len(problems)} roll problem(s)",
                               {"problems": problems})
        return CheckResult(self.name, Status.PASS,
                           f"{len(self.roll_log)} roll decision(s), all from "
                           "strictly prior sessions, monotone contracts")


class PrefixInvarianceCheck(LeakageCheck):
    """The pipeline run on [start, end-1] must produce bit-identical bars for
    the shared date range as the run on [start, end]: nothing in a bar may
    depend on later sessions (window-level truncation invariance)."""

    name = "pipeline_prefix_invariance"

    def __init__(self, bars_full: pd.DataFrame, bars_prefix: pd.DataFrame):
        self.full, self.prefix = bars_full, bars_prefix

    def run(self) -> CheckResult:
        if self.prefix.empty:
            return CheckResult(self.name, Status.SKIP, "empty prefix run")
        common = self.prefix.index
        missing = common.difference(self.full.index)
        if len(missing):
            return CheckResult(self.name, Status.FAIL,
                               f"{len(missing)} prefix bars absent from full run")
        a = self.full.loc[common, self.prefix.columns]
        if not a.equals(self.prefix):
            neq = [c for c in self.prefix.columns
                   if not a[c].equals(self.prefix[c])]
            return CheckResult(self.name, Status.FAIL,
                               "bars change when later sessions are added — "
                               "pipeline output depends on the future",
                               {"differing_columns": neq})
        return CheckResult(self.name, Status.PASS,
                           f"{len(common)} bars identical under window truncation")
