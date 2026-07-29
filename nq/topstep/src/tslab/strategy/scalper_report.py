"""Reporting, viability gating, and aggregation for the tick-scalper
walk-forward framework (scripts/run_scalper_wf.py).

Pure functions with no data dependencies, so every rule is unit-testable
(tests/test_scalper_framework.py).

Hard rules encoded here (post-mortem of the 2026-07-05 NQ rf diagnosis,
71.88% win / -$41,630):
  * A policy with qualifies=False (no positive train-EV grid cell) is
    never traded and never pooled into a headline aggregate.
  * Win rate is only a valid edge signal when RR >= 1 or expectancy > 0;
    otherwise it is a barrier-geometry artifact (a signal-free random
    walk touches +T before -S with probability ~S/(S+T)) and the report
    must carry HIGH_WIN_WARNING.
  * A shuffled-label control can never be marked viable, whatever its
    stats.
"""
from __future__ import annotations

import math

import numpy as np

WIN_GATE = 0.55          # mandate win-rate gate; see HIGH_WIN_WARNING
CONTROL_SUFFIX = "_shuf"  # model-name convention marking control arms
HIGH_WIN_WARNING = (
    "HIGH WIN RATE IS INVALID AS AN EDGE SIGNAL: RR < 1 and expectancy "
    "<= 0. With target T << stop S a signal-free random walk wins "
    "~S/(S+T) of trades; the shuffled-label control reproduces this. "
    "Do not gate on win rate alone.")
CONTROL_WARNING = (
    "SHUFFLED-LABEL CONTROL: diagnostic arm only; can never be marked "
    "viable regardless of its statistics.")
REJECTED_NOTE = ("no qualifying policy in any fold; nothing traded in "
                 "headline")


def effective_rr(target_ticks: float, stop_ticks: float,
                 fee_rt_ticks: float, stop_slip_ticks: float = 1.0
                 ) -> float:
    """Net reward:risk of the policy geometry.

    Win pays target minus round-trip fee; loss pays stop plus stop
    slippage plus round-trip fee.
    """
    win = target_ticks - fee_rt_ticks
    loss = stop_ticks + stop_slip_ticks + fee_rt_ticks
    if loss <= 0:
        return float("inf")
    return win / loss


def is_control_model(name: str) -> bool:
    """Shuffled-label control models are named <base>_shuf by
    convention; this single predicate carries the pipeline-level
    'control can never be viable' wiring."""
    return name.endswith(CONTROL_SUFFIX)


def realized_rr(avg_win: float, avg_loss: float) -> float:
    """|avg_win / avg_loss| from realized trade P&L. avg_win must be
    a genuine winner mean (> 0); a negative value would let abs()
    fake RR >= 1, so it maps to NaN."""
    if avg_win < 0 or avg_loss == 0 or not math.isfinite(avg_loss):
        return float("nan")
    return abs(float(avg_win) / float(avg_loss))


def breakeven_win_rate(avg_win: float, avg_loss: float) -> float:
    """p* with p*avg_win + (1-p)*avg_loss = 0, i.e. |L| / (W + |L|)."""
    w, loss = float(avg_win), abs(float(avg_loss))
    if w + loss <= 0:
        return float("nan")
    return loss / (w + loss)


def expectancy(win_rate: float, avg_win: float, avg_loss: float
               ) -> float:
    """Per-trade expectancy; avg_loss is signed (negative)."""
    return win_rate * avg_win + (1.0 - win_rate) * avg_loss


def gate_fold_policy(pol: dict | None) -> tuple[bool, str]:
    """(traded, reason). A fold policy is traded only if it carries a
    literal qualifies=True from train-time selection. Anything else —
    False, missing key, None, truthy int/str from a lossy round-trip —
    is rejected. This replaces the old fallback that simulated the
    least-negative rejected candidate anyway."""
    if pol is None:
        return False, "no_policy_found"
    q = pol.get("qualifies")
    if q is True:
        return True, "qualified"
    if q is False:
        return False, "rejected_at_selection_train_ev_not_positive"
    return False, "qualifies_flag_missing_or_invalid"


def policy_econ(pol: dict, fee_rt_ticks: float,
                stop_slip_ticks: float = 1.0) -> dict:
    """Geometry economics of a policy, independent of any trades."""
    t, s = float(pol["T"]), float(pol["S"])
    return dict(
        target_ticks=t, stop_ticks=s,
        fee_rt_ticks=round(fee_rt_ticks, 4),
        stop_slip_ticks=stop_slip_ticks,
        effective_rr=round(effective_rr(t, s, fee_rt_ticks,
                                        stop_slip_ticks), 4),
        geom_breakeven_win_rate=round(
            breakeven_win_rate(t - fee_rt_ticks,
                               -(s + stop_slip_ticks + fee_rt_ticks)),
            4))


def assess_viability(*, trades: int, win: float | None,
                     avg_win: float, avg_loss: float,
                     ev_trade: float | None, net: float,
                     qualified: bool, is_control: bool = False,
                     tps: float | None = None,
                     win_gate: float = WIN_GATE) -> dict:
    """Viability verdict + hard warnings for a stats block.

    valid_win_metric: win rate may be read as an edge signal only when
    RR >= 1 or expectancy > 0. viable requires a qualified non-control
    arm passing every mandate gate: win >= win_gate, tps >= 1 (when
    known), expectancy > 0, net > 0 — plus a valid win metric.
    """
    warnings: list[str] = []
    rr = realized_rr(avg_win, avg_loss)
    be = breakeven_win_rate(avg_win, avg_loss)
    ev = ev_trade if ev_trade is not None else float("nan")
    valid_win = bool(trades) and (
        (math.isfinite(rr) and rr >= 1.0)
        or (math.isfinite(ev) and ev > 0))
    if (trades and win is not None and math.isfinite(win)
            and win >= win_gate and not valid_win):
        warnings.append(HIGH_WIN_WARNING)
    if is_control:
        warnings.append(CONTROL_WARNING)
    if not qualified:
        warnings.append("policy rejected at selection; not traded in "
                        "headline")
    mandate = dict(
        win_ge_gate=bool(trades and win is not None
                         and math.isfinite(win) and win >= win_gate),
        tps_ge_1=(None if tps is None else bool(tps >= 1.0)),
        rr_ge_1_or_ev_gt_0=valid_win,
        ev_gt_0=bool(math.isfinite(ev) and ev > 0),
        net_gt_0=bool(net > 0))
    viable = bool(
        trades and qualified and not is_control and valid_win
        and mandate["win_ge_gate"] and mandate["tps_ge_1"] is not False
        and mandate["ev_gt_0"] and mandate["net_gt_0"])
    return dict(
        viable=viable, valid_win_metric=valid_win,
        mandate_gates=mandate,
        realized_rr=round(rr, 4) if math.isfinite(rr) else None,
        breakeven_win_rate=round(be, 4) if math.isfinite(be) else None,
        observed_win_rate=(round(win, 4) if win is not None
                           and math.isfinite(win) else None),
        warnings=warnings)


def aggregate_folds(stats: list[dict], pols: list[dict], *,
                    n_sess_total: int, n_folds_total: int,
                    fee_rt_ticks: float, stop_slip_ticks: float = 1.0,
                    is_control: bool = False) -> dict:
    """Pool per-fold simulate() outputs for QUALIFIED folds only.

    stats/pols are aligned lists covering only the folds that passed
    gate_fold_policy. Passing a rejected policy is a programming error
    and raises rather than silently polluting the headline.
    """
    for p in pols:
        if p.get("qualifies") is not True:
            raise ValueError(
                "rejected policy (qualifies is not True) passed to "
                "headline aggregation — gate_fold_policy must exclude "
                "it")
    if len(stats) != len(pols):
        raise ValueError("stats and pols must be aligned")
    if n_folds_total < len(stats):
        raise ValueError("n_folds_total < number of traded folds")
    n_rejected = n_folds_total - len(stats)
    if not stats:
        out = dict(trades=0, n_folds=0, n_folds_rejected=n_rejected,
                   n_folds_total=n_folds_total, note=REJECTED_NOTE,
                   viable=False, valid_win_metric=False,
                   warnings=["all folds rejected at selection "
                             "(train EV <= 0); headline intentionally "
                             "empty"])
        if is_control:
            out["warnings"].append(CONTROL_WARNING)
            out["role"] = "control"
        return out

    n = sum(r["trades"] for r in stats)
    nw = sum(r["n_wins"] for r in stats)
    nl = n - nw
    at = sum(r["attempts"] for r in stats)
    net = sum(r["net"] for r in stats)
    gross = sum(r["gross_usd"] for r in stats)
    costs = sum(r["costs_usd"] for r in stats)
    avg_win = (sum(r["avg_win"] * r["n_wins"] for r in stats) / nw
               if nw else 0.0)
    avg_loss = (sum(r["avg_loss"] * (r["trades"] - r["n_wins"])
                    for r in stats) / nl if nl else 0.0)
    win = nw / n if n else None
    ev = net / n if n else None
    tps = n / max(n_sess_total, 1)
    verdict = assess_viability(
        trades=n, win=win, avg_win=avg_win, avg_loss=avg_loss,
        ev_trade=ev, net=net, qualified=True, is_control=is_control,
        tps=tps)
    t_counts: dict[str, int] = {}
    s_counts: dict[str, int] = {}
    geom = []
    for p in pols:
        t_counts[str(p["T"])] = t_counts.get(str(p["T"]), 0) + 1
        s_counts[str(p["S"])] = s_counts.get(str(p["S"]), 0) + 1
        geom.append(effective_rr(p["T"], p["S"], fee_rt_ticks,
                                 stop_slip_ticks))
    def _nanmean(vals):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        return round(float(arr.mean()), 3) if arr.size else None

    out = dict(
        trades=n, attempts=at,
        fill_rate=round(n / at, 4) if at else None,
        tps=round(tps, 3),
        target_ticks=t_counts, stop_ticks=s_counts,
        effective_rr_geom_min=round(min(geom), 4),
        effective_rr_geom_max=round(max(geom), 4),
        avg_win=round(avg_win, 4), avg_loss=round(avg_loss, 4),
        gross=round(gross, 2), costs=round(costs, 2),
        net=round(net, 2),
        ev_trade=round(ev, 4) if n else None,
        ev_day=round(net / max(n_sess_total, 1), 3),
        win=round(win, 4) if n else None,
        adv_1s=_nanmean([r["adv_1s"] for r in stats]),
        miss_fmid_5s=_nanmean([r["miss_fmid_5s"] for r in stats]),
        max_dd=round(min(r["max_dd"] for r in stats), 2),
        killed=sum(r["killed"] for r in stats),
        pos_folds=sum(1 for r in stats if r["net"] > 0),
        n_folds=len(stats), n_folds_rejected=n_rejected,
        n_folds_total=n_folds_total)
    out.update(verdict)
    if is_control:
        out["role"] = "control"
    return out


def write_scalper_md(report: dict) -> str:
    """Markdown summary with the mandatory columns per instrument x
    model x arm: target/stop ticks, effective RR, avg win/loss,
    breakeven vs observed win rate, gross and net P&L, viability, plus
    the shuffled-label control section and hard warnings."""
    lines = [f"# {report.get('phase', 'scalper')} report", ""]
    cols = ("| model | arm | folds q/total | trades | target | stop | "
            "RR (realized) | RR geom | avg win | avg loss | "
            "breakeven win | observed win | gross | net | viable |")
    sep = "|" + "---|" * 15
    for inst, rep_i in report.get("instruments", {}).items():
        lines += [f"## {inst}", "", cols, sep]
        warn_block: list[str] = []
        for m, mb in rep_i.items():
            for a, r in (mb.get("arms") or {}).items():
                if r is None:
                    continue
                if r.get("trades"):
                    row = (f"| {m} | {a} "
                           f"| {r['n_folds']}/{r['n_folds_total']} "
                           f"| {r['trades']} | {r['target_ticks']} "
                           f"| {r['stop_ticks']} "
                           f"| {r.get('realized_rr')} "
                           f"| {r['effective_rr_geom_min']}-"
                           f"{r['effective_rr_geom_max']} "
                           f"| {r['avg_win']} | {r['avg_loss']} "
                           f"| {r.get('breakeven_win_rate')} "
                           f"| {r.get('win')} | {r['gross']} "
                           f"| {r['net']} | {r['viable']} |")
                else:
                    row = (f"| {m} | {a} "
                           f"| {r.get('n_folds', 0)}"
                           f"/{r['n_folds_total']} | 0 "
                           f"| - | - | - | - | - | - | - | - | - | - "
                           f"| False ({r.get('note', '')}) |")
                lines.append(row)
                for w in r.get("warnings", []):
                    warn_block.append(f"- **{m}/{a}**: {w}")
        if warn_block:
            lines += ["", "### Warnings", *warn_block]
        lines.append("")
    ctrl = report.get("controls", {})
    if ctrl:
        lines += ["## Shuffled-label control", "",
                  CONTROL_WARNING, ""]
        for inst, c in ctrl.items():
            note = c.get("note", "traded control folds are diagnostic "
                                 "only")
            lines.append(f"- **{inst}**: trades={c.get('trades', 0)}, "
                         f"win={c.get('win')}, net={c.get('net')}, "
                         f"viable={c.get('viable')} ({note})")
        lines.append("")
    return "\n".join(lines)
