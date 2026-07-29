"""Phase 10 — pre-registered lockbox confirmation of frozen Phase-9 survivors.

The scientific contract: **freeze the exact hypotheses first, then test on held-out
data, then recompute significance with the *frozen* trial count** (not the 176
searched in Phase 9). Nothing is tuned after freezing — a frozen hypothesis must
reproduce its Phase-9 strategy fingerprint exactly, or the freeze is rejected.

Honesty note baked into the verdict: a truly clean ACCEPT requires a *pristine*
lockbox — data the Phase-9 selection never observed. If the lockbox window overlapped
Phase-9's out-of-sample window (it did: Phase 9 ran OOS through 2026), the strongest
honest verdict is ``NEEDS_PAPER_LOCKBOX`` (confirm forward), never ``ACCEPT``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from nqlab.config.models import AppConfig, TopstepRules
from nqlab.discovery.deflation import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_variance,
)
from nqlab.discovery.htf_setups import PHASE9_FAMILIES
from nqlab.discovery.phase11_families import PHASE11_FAMILIES
from nqlab.discovery.phase12_families import PHASE12_FAMILIES
from nqlab.discovery.phase13_families import PHASE13_FAMILIES
from nqlab.discovery.phase14_families import PHASE14_FAMILIES
from nqlab.discovery.phase16_families import PHASE16_FAMILIES
from nqlab.discovery.phase17_families import PHASE17_FAMILIES
from nqlab.discovery.phase18_families import PHASE18_FAMILIES
from nqlab.discovery.phase20_families import PHASE20_FAMILIES
from nqlab.discovery.phase21_families import PHASE21_FAMILIES
from nqlab.discovery.phase22_families import PHASE22_FAMILIES
from nqlab.discovery.phase23_families import PHASE23_FAMILIES
from nqlab.discovery.phase24_families import PHASE24_FAMILIES
from nqlab.discovery.phase25_families import PHASE25_FAMILIES
from nqlab.discovery.phase26_families import PHASE26_FAMILIES
from nqlab.discovery.record import build_candidate_row
from nqlab.domain.calendar.sessions import parse_hhmm
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.validation.gauntlet import ValidationResult, run_gauntlet

# A frozen hypothesis can come from any of the live family registries.
_ALL_FAMILIES: dict[str, Any] = {
    **PHASE9_FAMILIES,
    **PHASE11_FAMILIES,
    **PHASE12_FAMILIES,
    **PHASE13_FAMILIES,
    **PHASE14_FAMILIES,
    **PHASE16_FAMILIES,
    **PHASE17_FAMILIES,
    **PHASE18_FAMILIES,
    **PHASE20_FAMILIES,
    **PHASE21_FAMILIES,
    **PHASE22_FAMILIES,
    **PHASE23_FAMILIES,
    **PHASE24_FAMILIES,
    **PHASE25_FAMILIES,
    **PHASE26_FAMILIES,
}

ACCEPT = "ACCEPT"
REJECT = "REJECT"
NEEDS_PAPER_LOCKBOX = "NEEDS_PAPER_LOCKBOX"
ACCEPT_FOR_PAPER = "ACCEPT_FOR_PAPER"
ACCEPT_FOR_FORWARD_PAPER = "ACCEPT_FOR_FORWARD_PAPER"
NEEDS_MORE_DATA = "NEEDS_MORE_DATA"
NEEDS_FORWARD_DATA = "NEEDS_FORWARD_DATA"  # MBP1-required without a clean MBP1 lockbox

_RTH_OPEN_DEFAULT = 570


@dataclass(frozen=True)
class FreezeSpec:
    """A pre-registration pointer: which exact Phase-9 candidate to freeze."""

    name: str
    run_id: str
    fingerprint: str


# The pre-registered Phase 10 freeze set. Selection rule (fixed, applied once):
# the gauntlet-surviving (pre-DSR ACCEPT) variant of each headline Phase-9 setup at
# the smallest risk that still SIZES to a tradeable position in the higher-ATR 2026
# regime (risk=$100 — risk=$25/$50 round to ~0 MNQ contracts in 2026, so they cannot
# be tested on the lockbox at all). All three passed OOS + ruin + cost-stress +
# Topstep + walk-forward in Phase 9 before the n=176 DSR correction.
PHASE10_FREEZE_SPECS: tuple[FreezeSpec, ...] = (
    FreezeSpec("opening_drive_london_5m", "phase9_150k_5m", "ea0e38697b573be9"),
    FreezeSpec("compression_expansion_or_5m", "phase9_50k_5m", "983deaed7389f60d"),
    FreezeSpec("opening_drive_on_5m", "phase9_100k_5m", "20993de318148d3f"),
)


@dataclass(frozen=True)
class FrozenHypothesis:
    """An immutable frozen hypothesis: exact family + params + market + provenance."""

    name: str
    family: str
    timeframe: str
    account: str
    params: tuple[tuple[str, Any], ...]  # sorted (key, value) items
    phase9_run_id: str
    phase9_fingerprint: str
    phase9_oos_net: float
    phase9_dsr: float
    frozen_at: str

    def param_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def config_hash(self) -> str:
        """Content hash over the *frozen, executable* config (tamper detection)."""
        # Reuse the strategy fingerprint: identical params -> identical hash.
        return str(self.to_strategy().params.fingerprint())

    def to_strategy(self, rth_start_min: int = _RTH_OPEN_DEFAULT) -> Any:
        return _ALL_FAMILIES[self.family](self.param_dict(), rth_start_min=rth_start_min)

    def to_json(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "family": self.family,
            "timeframe": self.timeframe,
            "account": self.account,
            "params": self.param_dict(),
            "phase9_run_id": self.phase9_run_id,
            "phase9_fingerprint": self.phase9_fingerprint,
            "phase9_oos_net": self.phase9_oos_net,
            "phase9_dsr": self.phase9_dsr,
            "frozen_at": self.frozen_at,
        }
        d["config_hash"] = self.config_hash()
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> FrozenHypothesis:
        fh = FrozenHypothesis(
            name=d["name"],
            family=d["family"],
            timeframe=d["timeframe"],
            account=d["account"],
            params=tuple(sorted(d["params"].items())),
            phase9_run_id=d["phase9_run_id"],
            phase9_fingerprint=d["phase9_fingerprint"],
            phase9_oos_net=d["phase9_oos_net"],
            phase9_dsr=d["phase9_dsr"],
            frozen_at=d["frozen_at"],
        )
        stored = d.get("config_hash")
        if stored is not None and stored != fh.config_hash():
            msg = f"frozen hypothesis {fh.name!r} has been tampered with (hash mismatch)"
            raise ValueError(msg)
        return fh


def freeze_from_phase9(
    tracker: ExperimentTracker, specs: tuple[FreezeSpec, ...], *, frozen_at: str
) -> list[FrozenHypothesis]:
    """Build immutable frozen hypotheses from stored Phase-9 candidates.

    Verifies each reconstructed strategy reproduces its Phase-9 fingerprint exactly —
    this is the 'no tuning after freeze' guarantee.
    """
    out: list[FrozenHypothesis] = []
    for spec in specs:
        run = tracker.get_run(spec.run_id)
        if run is None:
            msg = f"no Phase-9 run {spec.run_id!r} to freeze {spec.name!r} from"
            raise ValueError(msg)
        rows = tracker.load_candidates(spec.run_id)
        match = next((r for r in rows if r["fingerprint"] == spec.fingerprint), None)
        if match is None:
            msg = f"fingerprint {spec.fingerprint!r} not found in run {spec.run_id!r}"
            raise ValueError(msg)
        fh = FrozenHypothesis(
            name=spec.name,
            family=str(match["family"]),
            timeframe=str(run["timeframe"]),
            account=str(run["account"]).lower(),
            params=tuple(sorted(json.loads(str(match["params"])).items())),
            phase9_run_id=spec.run_id,
            phase9_fingerprint=spec.fingerprint,
            phase9_oos_net=float(match["oos_net"] or 0.0),
            phase9_dsr=float(match["deflated_sharpe"] or 0.0),
            frozen_at=frozen_at,
        )
        if fh.config_hash() != spec.fingerprint:
            msg = (
                f"frozen params for {spec.name!r} do not reproduce the Phase-9 fingerprint "
                f"({fh.config_hash()} != {spec.fingerprint}); refusing to freeze a tuned config"
            )
            raise ValueError(msg)
        out.append(fh)
    return out


def select_survivors(
    tracker: ExperimentTracker,
    run_ids: tuple[str, ...],
    *,
    top_n: int = 5,
    min_oos_trades: int = 20,
    min_risk: float = 100.0,
) -> list[FreezeSpec]:
    """Pre-registration selection: pick the top validation-gauntlet survivors to freeze.

    Selects candidates that passed EVERY economic gate on the validation window
    (``gauntlet_verdict == 'ACCEPT'``, i.e. positive OOS + ruin + cost-stress +
    Topstep + walk-forward before the multiple-testing penalty), traded enough, and
    size to a real position (risk >= ``min_risk``). Ranked by validation DSR then OOS.
    Uses ONLY validation results — never the lockbox.
    """
    cands: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for rid in run_ids:
        run = tracker.get_run(rid)
        if run is None:
            continue
        for r in tracker.load_candidates(rid):
            if r.get("gauntlet_verdict") != "ACCEPT":
                continue
            if int(r.get("oos_trades") or 0) < min_oos_trades:
                continue
            params = json.loads(str(r.get("params") or "{}"))
            if float(params.get("risk_per_trade") or 0.0) < min_risk:
                continue
            cands.append((rid, r, run, params))
    cands.sort(
        key=lambda c: (
            float(c[1].get("deflated_sharpe") or 0.0),
            float(c[1].get("oos_net") or 0.0),
        ),
        reverse=True,
    )
    specs: list[FreezeSpec] = []
    seen: set[tuple[str, str, str, object]] = set()
    for rid, r, run, params in cands:
        fam, tf, acct = str(r["family"]), str(run["timeframe"]), str(run["account"]).lower()
        key = (fam, tf, acct, params.get("risk_per_trade"))
        if key in seen:
            continue
        seen.add(key)
        name = f"{fam}_{tf}_{acct}_r{int(float(params.get('risk_per_trade') or 0))}"
        specs.append(FreezeSpec(name, rid, str(r["fingerprint"])))
        if len(specs) >= top_n:
            break
    return specs


def registry_path(config: AppConfig, name: str = "phase10") -> Path:
    return config.data.curated_root / f"{name}_frozen.json"


def save_registry(path: Path, frozen: list[FrozenHypothesis]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([fh.to_json() for fh in frozen], indent=2, sort_keys=True))


def load_registry(path: Path) -> list[FrozenHypothesis]:
    if not path.exists():
        msg = f"no frozen registry at {path} — run `lockbox freeze` first"
        raise FileNotFoundError(msg)
    return [FrozenHypothesis.from_json(d) for d in json.loads(path.read_text())]


def phase10_verdict(
    result: ValidationResult, dsr: float, threshold: float, *, pristine: bool
) -> tuple[str, str]:
    """Map the lockbox gauntlet result + frozen-count DSR to a Phase-10 verdict."""
    gv = result.verdict  # the economic gauntlet verdict on the lockbox
    if gv == "NEEDS_MORE_DATA":
        return NEEDS_PAPER_LOCKBOX, "too few lockbox trades to judge; needs a forward paper lockbox"
    if gv == "REJECT":
        return REJECT, "; ".join(result.reasons)
    # gv == ACCEPT: cleared every economic gate (OOS, walk-forward, cost stress,
    # Monte-Carlo, Topstep) on the lockbox.
    if dsr < threshold:
        return REJECT, f"fails Deflated-Sharpe at frozen count (DSR={dsr:.2f} < {threshold:.2f})"
    if not pristine:
        return NEEDS_PAPER_LOCKBOX, (
            f"clears the gauntlet AND DSR (={dsr:.2f}) at the frozen count, but the lockbox "
            "window was inside Phase 9's out-of-sample window (not pristine) — confirm on a "
            "forward paper lockbox before ACCEPT"
        )
    return ACCEPT, f"passes the untouched lockbox and DSR (={dsr:.2f}) at the frozen trial count"


def phase11_verdict(
    result: ValidationResult, dsr: float, threshold: float, *, pristine: bool
) -> tuple[str, str]:
    """Phase-11 paper-readiness verdict on the reserved (pristine) lockbox.

    ACCEPT_FOR_PAPER requires clearing every economic gate on the untouched lockbox
    AND the Deflated-Sharpe bar at the frozen trial count.
    """
    gv = result.verdict
    if gv == "NEEDS_MORE_DATA":
        return NEEDS_MORE_DATA, "too few lockbox trades to judge; " + "; ".join(result.reasons)
    if gv == "REJECT":
        return REJECT, "; ".join(result.reasons)
    if dsr < threshold:
        return REJECT, f"fails Deflated-Sharpe at frozen count (DSR={dsr:.2f} < {threshold:.2f})"
    if not pristine:  # safety net — Phase 11 reserves a clean lockbox, so this should not fire
        return NEEDS_MORE_DATA, "lockbox not pristine; needs a forward paper lockbox"
    return ACCEPT_FOR_PAPER, (
        f"passes the untouched 2025-2026 lockbox and DSR (={dsr:.2f}) at the frozen trial count"
    )


def phase12_verdict(
    result: ValidationResult, dsr: float, threshold: float, *, pristine: bool
) -> tuple[str, str]:
    """Phase-12 verdict. The 2025-2026 final test is NOT pristine for the trend_day
    family (Phase 11 already inspected it there), so the ceiling is
    ACCEPT_FOR_FORWARD_PAPER (confirm on a forward feed), never ACCEPT_FOR_PAPER.
    """
    gv = result.verdict
    if gv == "NEEDS_MORE_DATA":
        return NEEDS_MORE_DATA, "too few final-test trades to judge; " + "; ".join(result.reasons)
    if gv == "REJECT":
        return REJECT, "; ".join(result.reasons)
    if dsr < threshold:
        return REJECT, f"fails Deflated-Sharpe at frozen count (DSR={dsr:.2f} < {threshold:.2f})"
    return ACCEPT_FOR_FORWARD_PAPER, (
        f"clears the full gauntlet AND DSR (={dsr:.2f}) at the frozen count on the 2025-2026 "
        "final test; but that window was inspected for the trend_day family in Phase 11, so the "
        "honest ceiling is forward-paper confirmation, not ACCEPT_FOR_PAPER"
    )


def run_lockbox(
    config: AppConfig,
    frozen: list[FrozenHypothesis],
    *,
    tracker: ExperimentTracker,
    run_id: str,
    lockbox_start: date,
    lockbox_end: date,
    frame_for: Callable[[str], pl.DataFrame],
    rules_for: Callable[[str], TopstepRules],
    pristine: bool,
    created_at: str,
    symbol: str = "NQ.c",
    verdict_fn: Callable[..., tuple[str, str]] = phase10_verdict,
) -> list[dict[str, Any]]:
    """Run every frozen hypothesis on the lockbox window and record a Phase-10 verdict.

    Significance is recomputed with ``n_trials = len(frozen)`` (the frozen count), NOT
    the Phase-9 search size. ``frame_for`` / ``rules_for`` are injected so this is
    testable on synthetic data.
    """
    v = config.validation
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    evaluated: list[tuple[FrozenHypothesis, ValidationResult]] = []
    for fh in frozen:
        result = run_gauntlet(
            config,
            fh.to_strategy(rth_start_min),
            symbol=symbol,
            timeframe=fh.timeframe,
            start=lockbox_start,
            end=lockbox_end,
            rules=rules_for(fh.account),
            feature_frame=frame_for(fh.timeframe),
        )
        evaluated.append((fh, result))

    # Deflate with the FROZEN trial count (not 176).
    n_frozen = len(frozen)
    sr0 = expected_max_sharpe(sharpe_variance([r.in_sample_sharpe for _, r in evaluated]), n_frozen)
    run_meta = {
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": "mixed",
        "account": "frozen",
        "start_date": lockbox_start.isoformat(),
        "end_date": lockbox_end.isoformat(),
        "n_trials": n_frozen,
        "pbo": None,
        "dsr_threshold": v.dsr_threshold,
        "status": "complete",
        "created_at": created_at,
    }
    tracker.upsert_run(run_meta)

    rows: list[dict[str, Any]] = []
    for trial, (fh, result) in enumerate(evaluated):
        dsr = deflated_sharpe_ratio(
            result.in_sample_sharpe,
            sr0,
            int(result.in_sample_n or 0),
            result.in_sample_skew,
            result.in_sample_kurt,
        )
        verdict, reason = verdict_fn(result, dsr, v.dsr_threshold, pristine=pristine)
        row = build_candidate_row(
            result,
            run_id=run_id,
            trial=trial,
            family=fh.family,
            fingerprint=fh.phase9_fingerprint,
            params=fh.param_dict(),
            feature_set="9",
            train_start=v.train_start,
            train_end=v.train_end,
            val_start=v.test_start,
            val_end=v.test_end,
            config_hash=fh.config_hash(),
            data_version="lockbox",
            git_hash="lockbox",
            artifacts_path="",
            created_at=created_at,
        )
        row["deflated_sharpe"] = dsr
        row["sr0"] = sr0
        row["gauntlet_verdict"] = result.verdict
        row["final_verdict"] = verdict
        row["rejection_reason"] = reason
        row["strategy_name"] = fh.name  # surface the frozen hypothesis name in reports
        # Key the stored row by the unique frozen name (the real fingerprint lives in
        # config_hash). Two frozen hypotheses that differ only by account share a
        # strategy fingerprint, so without this they would collide on (run_id, fingerprint).
        row["fingerprint"] = fh.name
        tracker.save_candidate(row)
        rows.append(row)
    return rows
