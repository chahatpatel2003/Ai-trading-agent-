"""Phase 30 — volume/market-profile strategy factory (standalone families + winner enhancement).

Reuses the Phase-29 audited-engine wrapper, EOD-flat [0,960] block, 1-MNQ sizing, ``intraday_only``
EOD guarantee, and ``trades_frame``. Adds: (A-H) standalone volume-profile families over the
PIT-safe ``vp_*`` columns; VP-filtered variants of the two Phase-29 winners (``inside_day_break``,
``sweep_displacement``) plus their un-filtered baselines on the same exit grid (so improvement is
apples-to-apples); and VP-aware exits (target the prior POC / opposite value edge) and value-edge
stops.

Direction-correct dynamic targets: a "to_poc" long targets the POC only when it lies above entry
(else falls back to a 2R target); mirrored for shorts and the opposite edge. Stops are ATR or
"value-edge" (just beyond prior VAL for longs / VAH for shorts), all clamped to [0.5, 8] ATR so no
degenerate fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.backtest.result import BacktestResult
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.phase29 import (
    EOD_BLOCK,
    FAMILIES,
    SESSIONS,
    Family,
    intraday_only,  # noqa: F401  (re-exported for the runner)
    trades_frame,  # noqa: F401  (re-exported for the runner)
)
from nqlab.domain.strategy.base import ExecConfig, StrategyParams, TrailingConfig, attach_signals
from nqlab.features.volume_profile import add_volume_profile

_C = pl.col
_FALSE = pl.lit(value=False)
_BIG = 100000.0
_VP_REQ = ("vp_prior_poc", "vp_prior_vah", "vp_prior_val")


def _vp_families() -> dict[str, Family]:
    """Standalone VP families (A-H) + VP-filtered winners + un-filtered winner baselines."""
    base_idb = FAMILIES["inside_day_break"]
    base_swd = FAMILIES["sweep_displacement"]
    fams: list[Family] = [
        # --- A. Value-area rejection (revert from outside the edge back into value) ---
        Family("vp_va_rejection", "vp", _C("vp_val_reclaim_up"), _C("vp_vah_reject_dn"), _VP_REQ),
        # --- B. Value-area breakout / acceptance ---
        Family("vp_va_breakout", "vp", _C("vp_vah_break_up"), _C("vp_val_break_dn"), _VP_REQ),
        # --- C. POC reclaim ---
        Family("vp_poc_reclaim", "vp", _C("vp_poc_reclaim_up"), _C("vp_poc_reclaim_dn"), _VP_REQ),
        # --- D. LVN rejection ---
        Family(
            "vp_lvn_rejection",
            "vp",
            _C("vp_lvn_reject_up"),
            _C("vp_lvn_reject_dn"),
            (*_VP_REQ, "vp_prior_lvn_dn", "vp_prior_lvn_up"),
        ),
        # --- E. LVN breakout / vacuum ---
        Family(
            "vp_lvn_breakout",
            "vp",
            _C("vp_lvn_break_up"),
            _C("vp_lvn_break_dn"),
            (*_VP_REQ, "vp_prior_lvn_up", "vp_prior_lvn_dn"),
        ),
        # --- F. HVN magnet (drift into a nearby high-volume node with momentum) ---
        Family(
            "vp_hvn_magnet",
            "vp",
            (_C("vp_dist_hvn_up_atr") > 0)
            & (_C("vp_dist_hvn_up_atr") < 1.0)
            & (_C("close") > _C("close").shift(1)),
            (_C("vp_dist_hvn_dn_atr") > 0)
            & (_C("vp_dist_hvn_dn_atr") < 1.0)
            & (_C("close") < _C("close").shift(1)),
            (*_VP_REQ, "vp_prior_hvn_up", "vp_prior_hvn_dn"),
        ),
        # --- G. Outside-value re-entry (open outside value, re-enter toward POC) ---
        Family(
            "vp_outside_reentry",
            "vp",
            (_C("vp_open_loc") == -1) & _C("vp_val_reclaim_up"),
            (_C("vp_open_loc") == 1) & _C("vp_vah_reject_dn"),
            _VP_REQ,
        ),
        # --- H. Inside-value breakout (open inside value, break and hold an edge) ---
        Family(
            "vp_inside_breakout",
            "vp",
            (_C("vp_open_loc") == 0) & _C("vp_vah_break_up"),
            (_C("vp_open_loc") == 0) & _C("vp_val_break_dn"),
            _VP_REQ,
        ),
        # --- Winner baselines (no VP filter) on the same exit grid ---
        Family("idb_base", "combo_base", base_idb.long, base_idb.short, base_idb.requires),
        Family("swd_base", "combo_base", base_swd.long, base_swd.short, base_swd.requires),
        # --- inside_day_break + VP filters ---
        Family(
            "idb_dir_poc",
            "combo",
            base_idb.long & (_C("close") > _C("vp_prior_poc")),
            base_idb.short & (_C("close") < _C("vp_prior_poc")),
            (*base_idb.requires, *_VP_REQ),
        ),
        Family(
            "idb_outside_value",
            "combo",
            base_idb.long & (_C("vp_pos") > 0),
            base_idb.short & (_C("vp_pos") < 0),
            (*base_idb.requires, *_VP_REQ),
        ),
        Family(
            "idb_clean_path",
            "combo",
            base_idb.long
            & (_C("close") > _C("vp_prior_poc"))
            & (_C("vp_dist_hvn_up_atr").is_null() | (_C("vp_dist_hvn_up_atr") > 1.0)),
            base_idb.short
            & (_C("close") < _C("vp_prior_poc"))
            & (_C("vp_dist_hvn_dn_atr").is_null() | (_C("vp_dist_hvn_dn_atr") > 1.0)),
            (*base_idb.requires, *_VP_REQ),
        ),
        # --- sweep_displacement + VP filters ---
        Family(
            "swd_at_edge",
            "combo",
            base_swd.long & (_C("vp_pos") <= 0),
            base_swd.short & (_C("vp_pos") >= 0),
            (*base_swd.requires, *_VP_REQ),
        ),
        Family(
            "swd_not_chop",
            "combo",
            base_swd.long & (_C("vp_d_poc_atr") < -0.5),
            base_swd.short & (_C("vp_d_poc_atr") > 0.5),
            (*base_swd.requires, *_VP_REQ),
        ),
        Family(
            "swd_clean_path",
            "combo",
            base_swd.long
            & (_C("vp_d_poc_atr") < -0.5)
            & (_C("vp_dist_hvn_up_atr").is_null() | (_C("vp_dist_hvn_up_atr") > 1.0)),
            base_swd.short
            & (_C("vp_d_poc_atr") > 0.5)
            & (_C("vp_dist_hvn_dn_atr").is_null() | (_C("vp_dist_hvn_dn_atr") > 1.0)),
            (*base_swd.requires, *_VP_REQ),
        ),
    ]
    return {f.name: f for f in fams}


VP_FAMILIES: dict[str, Family] = _vp_families()

_KIND_SESSIONS: dict[str, tuple[str, ...]] = {
    "vp": ("RTH", "NY_AM", "NY_PM"),
    "combo": ("RTH", "NY_AM"),
    "combo_base": ("RTH", "NY_AM"),
}
# exit: rr1.5/rr2 fixed; to_poc -> prior POC; to_edge -> opposite value edge; eod -> hold to close.
_KIND_EXITS: dict[str, tuple[tuple[str, float], ...]] = {
    "vp": (("rr", 1.0), ("rr", 1.5), ("rr", 2.0), ("to_poc", 0.0), ("to_edge", 0.0)),
    "combo": (("rr", 2.0), ("to_poc", 0.0), ("to_edge", 0.0), ("eod", 0.0)),
    "combo_base": (("rr", 2.0), ("to_poc", 0.0), ("to_edge", 0.0), ("eod", 0.0)),
}
_KIND_STOPS: dict[str, tuple[tuple[str, float], ...]] = {
    "vp": (("atr", 1.0), ("atr", 1.5), ("vedge", 0.0)),
    "combo": (("atr", 1.5), ("atr", 2.5)),
    "combo_base": (("atr", 1.5), ("atr", 2.5)),
}


@dataclass(frozen=True)
class Candidate:
    family: str
    kind: str
    session: str
    stop_model: str  # "atr" | "vedge"
    stop_atr: float
    exit_model: str  # "rr" | "to_poc" | "to_edge" | "eod"
    rr: float

    @property
    def name(self) -> str:
        s = f"a{self.stop_atr:g}" if self.stop_model == "atr" else "vedge"
        e = {"rr": f"rr{self.rr:g}", "to_poc": "toPOC", "to_edge": "toEDGE", "eod": "eod"}[
            self.exit_model
        ]
        return f"{self.family}|{self.session}|{s}|{e}"

    def params(self) -> StrategyParams:
        return StrategyParams(
            "p30",
            "1",
            (
                ("family", self.family),
                ("session", self.session),
                ("stop", self.stop_model),
                ("stop_atr", self.stop_atr),
                ("exit", self.exit_model),
                ("rr", self.rr),
            ),
        )


def generate_candidates() -> list[Candidate]:
    out: list[Candidate] = []
    for fam in VP_FAMILIES.values():
        for session in _KIND_SESSIONS[fam.kind]:
            for stop_model, stop_atr in _KIND_STOPS[fam.kind]:
                for exit_model, rr in _KIND_EXITS[fam.kind]:
                    out.append(
                        Candidate(fam.name, fam.kind, session, stop_model, stop_atr, exit_model, rr)
                    )
    return out


def build_vp_frame(config: AppConfig, *, timeframe: str, start: Any, end: Any) -> pl.DataFrame:
    """Phase-29 feature frame + PIT-safe volume-profile columns."""
    from nqlab.discovery.phase29 import build_phase29_frame

    return add_volume_profile(
        build_phase29_frame(config, timeframe=timeframe, start=start, end=end)
    )


def _clamp(x: pl.Expr, atr: pl.Expr) -> pl.Expr:
    return pl.min_horizontal(pl.max_horizontal(x, 0.5 * atr), 8.0 * atr)


def build_signal_frame(frame: pl.DataFrame, cand: Candidate) -> pl.DataFrame:
    """Attach signal/stop_pts/target_pts for one VP candidate, gated to session + EOD block."""
    fam = VP_FAMILIES[cand.family]
    lo, hi = SESSIONS[cand.session]
    in_session = (_C("minute_of_day") >= lo) & (_C("minute_of_day") < hi)
    if any(req not in frame.columns for req in fam.requires):
        return attach_signals(
            frame, long=_FALSE, short=_FALSE, stop_pts=pl.lit(1.0), target_pts=pl.lit(_BIG)
        ).filter((_C("minute_of_day") >= EOD_BLOCK[0]) & (_C("minute_of_day") <= EOD_BLOCK[1]))
    valid = _C("atr").is_not_null() & (_C("atr") > 0)
    for req in fam.requires:
        valid = valid & _C(req).is_not_null()
    long = fam.long.fill_null(value=False) & in_session & valid
    short = fam.short.fill_null(value=False) & in_session & valid

    atr = _C("atr")
    close, poc, vah, val = _C("close"), _C("vp_prior_poc"), _C("vp_prior_vah"), _C("vp_prior_val")
    if cand.stop_model == "vedge":
        long_stop = _clamp((close - val) + 0.5 * atr, atr)
        short_stop = _clamp((vah - close) + 0.5 * atr, atr)
        stop = (
            pl.when(long)
            .then(long_stop)
            .when(short)
            .then(short_stop)
            .otherwise(cand.stop_atr * atr)
        )
    else:
        stop = cand.stop_atr * atr

    if cand.exit_model in ("eod",):
        target: pl.Expr = pl.lit(_BIG)
    elif cand.exit_model == "rr":
        target = cand.rr * stop
    elif cand.exit_model == "to_poc":
        lt = pl.when(poc > close).then(_clamp(poc - close, atr)).otherwise(2.0 * stop)
        st = pl.when(poc < close).then(_clamp(close - poc, atr)).otherwise(2.0 * stop)
        target = pl.when(long).then(lt).when(short).then(st).otherwise(pl.lit(_BIG))
    else:  # to_edge: long -> VAH, short -> VAL
        lt = pl.when(vah > close).then(_clamp(vah - close, atr)).otherwise(2.0 * stop)
        st = pl.when(val < close).then(_clamp(close - val, atr)).otherwise(2.0 * stop)
        target = pl.when(long).then(lt).when(short).then(st).otherwise(pl.lit(_BIG))

    sig = attach_signals(frame, long=long, short=short, stop_pts=stop, target_pts=target)
    return sig.filter((_C("minute_of_day") >= EOD_BLOCK[0]) & (_C("minute_of_day") <= EOD_BLOCK[1]))


def run_candidate(
    frame: pl.DataFrame,
    cand: Candidate,
    *,
    config: AppConfig,
    rules: TopstepRules,
    slippage_mult: float = 1.0,
    commission_mult: float = 1.0,
) -> BacktestResult:
    """Run one VP candidate through the audited engine (1 MNQ, EOD-flat, full distribution)."""
    sig = build_signal_frame(frame, cand)
    trail = TrailingConfig()
    exec_cfg = ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10_000)
    return run_backtest(
        sig,
        symbol="NQ.c",
        timeframe=config.backtest.timeframe,
        strategy=cand.params(),
        config=config,
        rules=rules,
        primary_cutoff=False,
        entry_mode="next_open",
        gate_entries=False,
        slippage_mult=slippage_mult,
        commission_mult=commission_mult,
        trailing=trail,
        exec_config=exec_cfg,
    )
