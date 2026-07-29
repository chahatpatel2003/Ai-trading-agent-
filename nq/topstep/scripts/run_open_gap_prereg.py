#!/usr/bin/env python
"""Pre-registered cash-open gap continuation kill-test (edge-thesis audit
kill-test #1). NOT a search: exactly 5 registered hypotheses, mechanical
dev-only parameter freezing, one-shot 2025-H1 validation.

Protocol (enforced by this script):
  freeze    Derive + freeze parameters from DEV sessions only
            (2021-01-01..2024-12-31) via the DECLARED_RULES below, write
            outputs/open_gap_prereg.{json,md} with a SHA256 of the frozen
            config. Refuses to run if the prereg already exists.
  validate  One-shot evaluation on 2025-H1 (2025-01-02..2025-06-30).
            Refuses to run without a frozen prereg, refuses to run twice
            (kill-test report existing = consumed), re-verifies the
            prereg hash, emits leakage audit + shuffled control, writes
            outputs/open_gap_killtest_report.{json,md}.

Registered hypotheses (max 5, frozen):
  H1  gap >= +thr AND first post-OR 1m close > OR_high  -> LONG continuation
  H2  gap <= -thr AND first post-OR 1m close < OR_low   -> SHORT continuation
  H3  gap >= +thr AND first post-OR 1m close < OR_low   -> SHORT failed-gap fade
  H4  gap <= -thr AND first post-OR 1m close > OR_high  -> LONG failed-gap fade
  H5  symmetric-side clause: short-side results (H2+H3) are reported and
      gated separately so long-side beta cannot carry the verdict.

DECLARED_RULES (fixed before any dev data was inspected):
  gap threshold  = max(12.16 NQ pts [ml-pass distilled prior, not re-fit],
                       P60 of |gap| over dev sessions), rounded to 0.25
  OR window      = argmax over {15, 30} min of total dev net (H1..H4,
                   R=1.0, MNQ costs); tie -> 30
  R multiples    = {1.0 (PRIMARY, gated), 1.5, 2.0 (secondary, reported)}
  confirmation   = first 1m close beyond OR extreme after OR completes,
                   deadline 10:30 CT; entry at NEXT 1m bar open (market)
  stop           = opposite OR extreme; target = entry +/- R * risk
  intrabar tie   = stop fills first (conservative)
  time-stop      = exit at close of first bar >= 14:30 CT
  max 1 trade/session (first confirmation wins); 1 contract
  instruments    = MNQ primary (gates), NQ control (reported)
  costs          = tslab CostModel (verified fees + spread + slippage);
                   entry market, stop exit 'stop', target exit 'limit',
                   time-stop exit 'market'

Usage: run_open_gap_prereg.py {freeze|validate}
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.config import load_config          # noqa: E402
from tslab.sim.costs import CostModel         # noqa: E402

DEV_START, DEV_END = "2021-01-01", "2024-12-31"
VAL_START, VAL_END = "2025-01-02", "2025-06-30"
GAP_ANCHOR_PTS = 12.16          # ml-pass distilled prior (frozen, not re-fit)
GAP_PCTL = 60                   # declared percentile floor rule
OR_CANDIDATES = (15, 30)        # declared; frozen at freeze-time by rule
R_PRIMARY = 1.0
R_SECONDARY = (1.5, 2.0)
CONFIRM_DEADLINE_CT = "10:30"
TIMESTOP_CT = "14:30"
MIN_RTH_MINUTES = 300           # skip early-close sessions
SHUFFLE_SEED = 20260705
GATES = {
    "net_gt_0": "validation MNQ net (R=1.0, all hypotheses) > 0",
    "ev_trade_gt_0": "validation MNQ EV/trade > 0",
    "ev_month_gt_0": "mean validation monthly MNQ net > 0",
    "rr_ge_1": "geometric RR >= 1 by construction AND realized "
               "|avgWin/avgLoss| >= 0.85 (declared slippage tolerance)",
    "sufficient_trades": ">= 20 validation trades for a binding verdict; "
                         "12-19 = WEAK_SAMPLE; < 12 = INSUFFICIENT_SAMPLE",
    "short_side_alive": ">= 5 short-side trades AND short-side net > "
                        "-(2x total friction on those trades)",
    "regime_spread": "if net > 0: no single validation month > 60% of "
                     "the sum of positive monthly P&L",
    "leakage_clean": "all leakage audit checks pass",
}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
PREREG_JSON = OUT / "open_gap_prereg.json"
PREREG_MD = OUT / "open_gap_prereg.md"
REPORT_JSON = OUT / "open_gap_killtest_report.json"
REPORT_MD = OUT / "open_gap_killtest_report.md"
LEAK_JSON = OUT / "leakage_audit_open_gap.json"


def sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def load_sessions(start: str, end: str) -> tuple[dict, dict]:
    """({session_date: rth-bars DataFrame}, meta). RTH bars only, prior
    RTH close attached. Sessions whose front-month SYMBOL differs from
    the prior session's closing symbol are skipped: an old-contract
    close vs new-contract open embeds quarterly roll basis, not a
    market gap (pre-validation audit finding)."""
    cols = ["open", "high", "low", "close", "session", "rth",
            "bar_open_utc", "entry_ok", "symbol"]
    df = pd.read_parquet(OUT / "nq_bars_full.parquet", columns=cols)
    if "bar_open_utc" not in df.columns:      # stored as the index
        df = df.reset_index()
    df["session"] = pd.to_datetime(df["session"]).dt.date
    lo, hi = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    # need one session before `start` for the first gap's prior close
    df = df[df["rth"]]
    df["ct"] = df["bar_open_utc"].dt.tz_convert("America/Chicago")
    out, prior_close, prior_sess, prior_sym = {}, None, None, None
    skipped = dict(roll_boundary=0, early_close=0, odd_open=0)
    for sess, g in df.groupby("session", sort=True):
        g = g.sort_values("ct")
        sym = str(g["symbol"].iloc[0])
        if prior_close is not None and lo <= sess <= hi:
            if sym != prior_sym:
                skipped["roll_boundary"] += 1
            elif len(g) < MIN_RTH_MINUTES:
                skipped["early_close"] += 1
            elif g["ct"].iloc[0].strftime("%H:%M") != "08:30":
                skipped["odd_open"] += 1
            else:
                gg = g.reset_index(drop=True)
                gg.attrs["prior_close"] = prior_close
                gg.attrs["prior_session"] = str(prior_sess)
                out[sess] = gg
        prior_close = float(g["close"].iloc[-1])
        prior_sess, prior_sym = sess, sym
    return out, dict(sessions=len(out), skipped=skipped)


def day_signal(bars: pd.DataFrame, gap_thr: float, or_min: int,
               deadline_ct: str = CONFIRM_DEADLINE_CT):
    """Return (hyp, side, entry_idx, entry_px, stop_px, risk_pts) or None.
    Uses only information available at each bar close."""
    gap = float(bars["open"].iloc[0]) - bars.attrs["prior_close"]
    if abs(gap) < gap_thr:
        return None
    orb = bars.iloc[:or_min]
    or_hi, or_lo = float(orb["high"].max()), float(orb["low"].min())
    deadline = bars["ct"].iloc[0].normalize() + pd.Timedelta(
        hours=int(deadline_ct[:2]), minutes=int(deadline_ct[3:]))
    deadline = deadline.tz_localize(None)
    for i in range(or_min, len(bars) - 1):
        ct = bars["ct"].iloc[i].tz_localize(None)
        if ct > deadline:
            return None
        c = float(bars["close"].iloc[i])
        up, dn = c > or_hi, c < or_lo
        if not (up or dn):
            continue
        if gap > 0:
            hyp, side = ("H1", 1) if up else ("H3", -1)
        else:
            hyp, side = ("H4", 1) if up else ("H2", -1)
        entry_idx = i + 1
        if not bool(bars["entry_ok"].iloc[entry_idx]):
            return None
        entry_px = float(bars["open"].iloc[entry_idx])
        stop_px = or_lo if side == 1 else or_hi
        risk = (entry_px - stop_px) * side
        if risk <= 0:
            return None
        return dict(hyp=hyp, side=side, entry_idx=entry_idx,
                    entry_px=entry_px, stop_px=stop_px, risk_pts=risk,
                    gap_pts=gap, or_hi=or_hi, or_lo=or_lo)
    return None


def run_bracket(bars, sig, r_mult, timestop_ct: str = TIMESTOP_CT):
    """(exit_px, exit_type, exit_ct). Conservative: stop-first intrabar."""
    side, e = sig["side"], sig["entry_px"]
    stop, tgt = sig["stop_px"], e + side * r_mult * sig["risk_pts"]
    ts_h, ts_m = int(timestop_ct[:2]), int(timestop_ct[3:])
    for i in range(sig["entry_idx"], len(bars)):
        ct = bars["ct"].iloc[i]
        if ct.hour > ts_h or (ct.hour == ts_h and ct.minute >= ts_m):
            return float(bars["close"].iloc[i]), "time", str(ct)
        hi, lo = float(bars["high"].iloc[i]), float(bars["low"].iloc[i])
        hit_stop = lo <= stop if side == 1 else hi >= stop
        hit_tgt = hi >= tgt if side == 1 else lo <= tgt
        if hit_stop:                      # conservative: stop first
            return stop, "stop", str(ct)
        if hit_tgt:
            return tgt, "target", str(ct)
    return float(bars["close"].iloc[-1]), "eod", str(bars["ct"].iloc[-1])


def evaluate(sessions: dict, gap_thr: float, or_min: int, r_mult: float,
             cm: CostModel, cfg, gap_override: dict | None = None) -> list:
    """One trade row per triggering session. gap_override (shuffled
    control) substitutes each session's gap value pre-threshold."""
    trades = []
    for sess, bars in sessions.items():
        if gap_override is not None:
            b = bars.copy()
            b.attrs["prior_close"] = float(bars["open"].iloc[0]) - \
                gap_override[sess]
            bars_used = b
        else:
            bars_used = bars
        sig = day_signal(bars_used, gap_thr, or_min)
        if sig is None:
            continue
        exit_px, exit_type, exit_ct = run_bracket(bars, sig, r_mult)
        pts = (exit_px - sig["entry_px"]) * sig["side"]
        ei = sig["entry_idx"]
        row = dict(session=str(sess), hyp=sig["hyp"], side=sig["side"],
                   gap_pts=round(sig["gap_pts"], 2),
                   risk_pts=round(sig["risk_pts"], 2),
                   entry_px=sig["entry_px"], exit_px=exit_px,
                   exit_type=exit_type, exit_ct=exit_ct,
                   confirm_ct=str(bars["ct"].iloc[ei - 1]),
                   entry_ct=str(bars["ct"].iloc[ei]),
                   or_end_ct=str(bars["ct"].iloc[or_min]),
                   pts=round(pts, 2))
        for inst in ("MNQ", "NQ"):
            pv = cfg.instruments[inst].point_value
            exit_ot = {"stop": "stop", "target": "limit",
                       "time": "market", "eod": "market"}[exit_type]
            cost = (cm.side_cost(inst, 1, "market").total
                    + cm.side_cost(inst, 1, exit_ot).total)
            row[f"net_{inst}"] = round(pts * pv - cost, 2)
            row[f"cost_{inst}"] = round(cost, 2)
        trades.append(row)
    return trades


def stats(trades, inst="MNQ"):
    if not trades:
        return dict(trades=0, net=0.0)
    net = [t[f"net_{inst}"] for t in trades]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]
    aw = float(np.mean(wins)) if wins else 0.0
    al = float(np.mean(losses)) if losses else 0.0
    monthly = {}
    for t in trades:
        m = t["session"][:7]
        monthly[m] = round(monthly.get(m, 0.0) + t[f"net_{inst}"], 2)
    return dict(
        trades=len(net), net=round(float(np.sum(net)), 2),
        ev_trade=round(float(np.mean(net)), 4),
        win=round(len(wins) / len(net), 4),
        avg_win=round(aw, 2), avg_loss=round(al, 2),
        realized_rr=round(abs(aw / al), 4) if al else None,
        gross=round(float(np.sum(net))
                    + sum(t[f"cost_{inst}"] for t in trades), 2),
        costs=round(sum(t[f"cost_{inst}"] for t in trades), 2),
        exit_mix={e: sum(1 for t in trades if t["exit_type"] == e)
                  for e in ("target", "stop", "time", "eod")},
        by_hyp={h: dict(n=sum(1 for t in trades if t["hyp"] == h),
                        net=round(sum(t[f"net_{inst}"] for t in trades
                                      if t["hyp"] == h), 2))
                for h in ("H1", "H2", "H3", "H4")},
        monthly=monthly,
        max_dd=round(float((lambda s: (s - s.cummax()).min())(
            pd.Series(net).cumsum())), 2))


def freeze() -> int:
    if PREREG_JSON.exists():
        print(f"REFUSED: {PREREG_JSON} exists — prereg is immutable.")
        return 1
    cfg = load_config()
    cm = CostModel(cfg)
    dev, dev_meta = load_sessions(DEV_START, DEV_END)
    gaps = {s: float(b["open"].iloc[0]) - b.attrs["prior_close"]
            for s, b in dev.items()}
    p60 = float(np.percentile(np.abs(list(gaps.values())), GAP_PCTL))
    gap_thr = round(max(GAP_ANCHOR_PTS, p60) * 4) / 4
    or_nets = {}
    for om in OR_CANDIDATES:
        tr = evaluate(dev, gap_thr, om, R_PRIMARY, cm, cfg)
        or_nets[om] = stats(tr)["net"]
    or_min = max(OR_CANDIDATES,
                 key=lambda om: (or_nets[om], om))   # tie -> 30
    frozen = dict(
        gap_threshold_pts=gap_thr, or_minutes=int(or_min),
        r_primary=R_PRIMARY, r_secondary=list(R_SECONDARY),
        confirm_deadline_ct=CONFIRM_DEADLINE_CT, timestop_ct=TIMESTOP_CT,
        stop_rule="opposite OR extreme", intrabar_rule="stop-first",
        max_trades_per_session=1, contracts=1,
        instrument_primary="MNQ", instrument_control="NQ",
        dev_window=[DEV_START, DEV_END], val_window=[VAL_START, VAL_END],
        min_rth_minutes=MIN_RTH_MINUTES, shuffle_seed=SHUFFLE_SEED)
    prereg = dict(
        name="open-gap continuation kill-test",
        registered_utc=pd.Timestamp.now('UTC').isoformat(),
        hypotheses={
            "H1": "gap>=+thr & OR-hold up -> long continuation",
            "H2": "gap<=-thr & OR-hold down -> short continuation",
            "H3": "gap>=+thr & OR-fail down -> short failed-gap fade",
            "H4": "gap<=-thr & OR-fail up -> long failed-gap fade",
            "H5": "symmetric clause: short side (H2+H3) gated separately"},
        declared_rules=dict(
            gap_threshold=f"max({GAP_ANCHOR_PTS} anchor, P{GAP_PCTL} of "
                          f"dev |gap|) rounded to 0.25",
            or_window=f"argmax dev net over {OR_CANDIDATES} at R=1 MNQ, "
                      f"tie->30",
            r_multiples="1.0 primary (gated); 1.5, 2.0 secondary"),
        derivation=dict(dev_sessions=len(dev), dev_meta=dev_meta,
                        dev_gap_p60=round(p60, 2),
                        or_selection_dev_net=or_nets),
        interpretation_clauses=[
            "GAP-SPECIFICITY (pre-committed before validation): a PASS "
            "is claimed as a GAP edge only if validation MNQ net at "
            "R=1.0 exceeds the shuffled-gap control net; otherwise the "
            "verdict is labeled *_GENERIC_ORB — the gap condition then "
            "merely selects which days trade a generic opening-range "
            "breakout (dev already shows control parity).",
            "ROLL FIX (pre-validation re-freeze): sessions whose "
            "front-month symbol differs from the prior session's are "
            "skipped — old-contract close vs new-contract open embeds "
            "quarterly roll basis, not a market gap."],
        frozen=frozen, gates=GATES,
        runner_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        config_sha256=sha(frozen))
    PREREG_JSON.write_text(json.dumps(prereg, indent=2))
    lines = [f"# Pre-registration: {prereg['name']}", "",
             f"Registered {prereg['registered_utc']} — config SHA256 "
             f"`{prereg['config_sha256']}`", "",
             "## Hypotheses (frozen, max 5)"]
    lines += [f"- **{k}**: {v}" for k, v in prereg["hypotheses"].items()]
    lines += ["", "## Frozen parameters"]
    lines += [f"- {k}: `{v}`" for k, v in frozen.items()]
    lines += ["", "## Declared derivation rules"]
    lines += [f"- {k}: {v}" for k, v in prereg["declared_rules"].items()]
    lines += ["", f"## Derivation record",
              f"- dev sessions: {len(dev)}; P{GAP_PCTL}|gap| = "
              f"{p60:.2f} pts; OR selection dev nets: {or_nets}", "",
              "## Gates (frozen)"]
    lines += [f"- **{k}**: {v}" for k, v in GATES.items()]
    lines += ["", "## Interpretation clauses (pre-committed)"]
    lines += [f"- {c}" for c in prereg["interpretation_clauses"]]
    lines += ["", f"Runner SHA256 at freeze: "
              f"`{prereg['runner_sha256']}`", "",
              "Validation (2025-H1) is one-shot; this file and the "
              "runner may not change after `validate` runs."]
    PREREG_MD.write_text("\n".join(lines))
    print(f"FROZEN: thr={gap_thr} pts, OR={or_min}m, dev nets={or_nets}, "
          f"sha={prereg['config_sha256'][:12]}")
    return 0


def validate() -> int:
    if not PREREG_JSON.exists():
        print("REFUSED: no prereg — run freeze first.")
        return 1
    if REPORT_JSON.exists():
        print(f"REFUSED: {REPORT_JSON} exists — validation is one-shot.")
        return 1
    prereg = json.loads(PREREG_JSON.read_text())
    fz = prereg["frozen"]
    if sha(fz) != prereg["config_sha256"]:
        print("REFUSED: prereg hash mismatch — config was edited.")
        return 1
    # frozen behavioral params must match the module constants actually
    # used by day_signal/run_bracket/load_sessions (binds hash to code)
    assert fz["confirm_deadline_ct"] == CONFIRM_DEADLINE_CT
    assert fz["timestop_ct"] == TIMESTOP_CT
    assert fz["min_rth_minutes"] == MIN_RTH_MINUTES
    assert fz["r_primary"] == R_PRIMARY
    cfg = load_config()
    cm = CostModel(cfg)
    dev, dev_meta = load_sessions(*fz["dev_window"])
    val, val_meta = load_sessions(*fz["val_window"])
    thr, om = fz["gap_threshold_pts"], fz["or_minutes"]

    res = {"dev": {}, "val": {}}
    trades_val_primary = None
    for label, sess in (("dev", dev), ("val", val)):
        for r in [fz["r_primary"]] + fz["r_secondary"]:
            tr = evaluate(sess, thr, om, r, cm, cfg)
            key = f"R{r}"
            res[label][key] = dict(
                MNQ=stats(tr, "MNQ"), NQ=stats(tr, "NQ"),
                trades=tr if (label == "val" and r == fz["r_primary"])
                else len(tr))
            if label == "val" and r == fz["r_primary"]:
                trades_val_primary = tr
        # shuffled-gap control at primary R (fixed seed, per-window perm)
        rng = np.random.default_rng(fz["shuffle_seed"])
        keys = list(sess.keys())
        gaps = [float(b["open"].iloc[0]) - b.attrs["prior_close"]
                for b in sess.values()]
        gap_over = dict(zip(keys, rng.permutation(gaps)))
        ctr = evaluate(sess, thr, om, fz["r_primary"], cm, cfg,
                       gap_override=gap_over)
        res[label]["shuffled_control_R1"] = dict(MNQ=stats(ctr, "MNQ"))
        # unconditional baseline (no gap condition) at primary R
        base = evaluate(sess, 0.0, om, fz["r_primary"], cm, cfg)
        res[label]["all_days_baseline_R1"] = dict(MNQ=stats(base, "MNQ"))

    # ---- leakage audit (real checks, parsed timestamps) -----------------
    dev_dates = sorted(str(s) for s in dev)
    val_dates = sorted(str(s) for s in val)
    tp = trades_val_primary

    def _causal(t):
        conf = pd.Timestamp(t["confirm_ct"])
        ent = pd.Timestamp(t["entry_ct"])
        orend = pd.Timestamp(t["or_end_ct"])
        exi = pd.Timestamp(t["exit_ct"])
        dl_h, dl_m = (int(fz["confirm_deadline_ct"][:2]),
                      int(fz["confirm_deadline_ct"][3:]))
        return (conf >= orend and ent > conf and exi >= ent
                and (conf.hour, conf.minute) <= (dl_h, dl_m))

    dev_gaps = np.abs([float(b["open"].iloc[0]) - b.attrs["prior_close"]
                       for b in dev.values()])
    thr_rederived = round(
        max(GAP_ANCHOR_PTS, float(np.percentile(dev_gaps, GAP_PCTL)))
        * 4) / 4
    leak = dict(
        dev_window_respected=(dev_dates[0] >= fz["dev_window"][0]
                              and dev_dates[-1] <= fz["dev_window"][1]),
        val_window_respected=(val_dates[0] >= fz["val_window"][0]
                              and val_dates[-1] <= fz["val_window"][1]),
        no_dev_val_overlap=not set(dev_dates) & set(val_dates),
        threshold_rederived_from_dev=thr_rederived == thr,
        confirmation_causality=all(_causal(t) for t in tp) if tp
        else True,
        one_trade_per_session=(len({t["session"] for t in tp})
                               == len(tp)) if tp else True,
        frozen_params_bound_to_code=True,   # asserted at function entry
        session_meta=dict(dev=dev_meta, val=val_meta),
        runner_sha256=hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        prereg_runner_sha256=prereg.get("runner_sha256"),
        runner_unchanged_since_freeze=prereg.get("runner_sha256")
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    leak["all_pass"] = all(v is True for k, v in leak.items()
                           if isinstance(v, bool))
    LEAK_JSON.write_text(json.dumps(leak, indent=2))

    # ---- gates on validation MNQ primary --------------------------------
    vs = res["val"][f"R{fz['r_primary']}"]["MNQ"]
    n = vs.get("trades", 0)
    shorts = [t for t in (tp or []) if t["side"] == -1]
    short_net = round(sum(t["net_MNQ"] for t in shorts), 2)
    short_friction = round(sum(t["cost_MNQ"] for t in shorts), 2)
    monthly = vs.get("monthly", {})
    pos_sum = sum(v for v in monthly.values() if v > 0)
    max_month_share = (max(monthly.values()) / pos_sum
                       if pos_sum > 0 and monthly else None)
    months = sorted({t["session"][:7] for t in (tp or [])} |
                    set(monthly.keys()))
    ev_month = (vs.get("net", 0.0) / 6.0)
    gates = dict(
        net_gt_0=vs.get("net", 0.0) > 0,
        ev_trade_gt_0=vs.get("ev_trade", 0.0) > 0 if n else False,
        ev_month_gt_0=ev_month > 0,
        rr_ge_1=(((vs.get("realized_rr") is None and n > 0
                   and vs.get("win", 0) == 1.0)          # all-winners
                  or (vs.get("realized_rr") or 0) >= 0.85)
                 if n else False),
        sufficient_trades=("SUFFICIENT" if n >= 20 else
                           "WEAK_SAMPLE" if n >= 12 else
                           "INSUFFICIENT_SAMPLE"),
        short_side_alive=(len(shorts) >= 5
                          and short_net > -2 * short_friction),
        regime_spread=(max_month_share is None
                       or max_month_share <= 0.60),
        leakage_clean=leak["all_pass"],
    )
    hard = ["net_gt_0", "ev_trade_gt_0", "ev_month_gt_0", "rr_ge_1",
            "short_side_alive", "regime_spread", "leakage_clean"]
    sc_val_net = res["val"]["shuffled_control_R1"]["MNQ"].get("net", 0.0)
    gap_specific = vs.get("net", 0.0) > sc_val_net
    if gates["sufficient_trades"] == "INSUFFICIENT_SAMPLE":
        verdict = "INSUFFICIENT_SAMPLE"
    elif all(gates[g] for g in hard):
        verdict = ("PASS" if gates["sufficient_trades"] == "SUFFICIENT"
                   else "PASS_WEAK_SAMPLE")
        if not gap_specific:      # pre-committed interpretation clause
            verdict += "_GENERIC_ORB"
    else:
        verdict = "FAIL"
    report = dict(
        name=prereg["name"], prereg_sha256=prereg["config_sha256"],
        run_utc=pd.Timestamp.now('UTC').isoformat(),
        frozen=fz, results=res, gates=gates, verdict=verdict,
        gate_definitions=prereg["gates"],
        short_side=dict(n=len(shorts), net=short_net,
                        friction=short_friction),
        gap_specific=gap_specific,
        shuffled_control_val_net=sc_val_net,
        max_month_share=(round(max_month_share, 4)
                         if max_month_share is not None else None),
        ev_month=round(ev_month, 2), val_months=months,
        leakage_audit=leak)
    REPORT_JSON.write_text(json.dumps(report, indent=2, default=str))

    dv = res["dev"][f"R{fz['r_primary']}"]["MNQ"]
    sc_d = res["dev"]["shuffled_control_R1"]["MNQ"]
    sc_v = res["val"]["shuffled_control_R1"]["MNQ"]
    md = [f"# Open-gap kill-test report — verdict: **{verdict}**", "",
          f"Run {report['run_utc']} | prereg `{prereg['config_sha256'][:12]}` "
          f"| frozen: thr={fz['gap_threshold_pts']} pts, OR={om}m, "
          f"R primary={fz['r_primary']}", "",
          "## Validation 2025-H1 (MNQ primary, R=1.0)", "",
          f"- trades={vs.get('trades', 0)}, net=${vs.get('net', 0)}, "
          f"EV/trade=${vs.get('ev_trade', 0)}, win={vs.get('win')}, "
          f"realized RR={vs.get('realized_rr')}",
          f"- gross=${vs.get('gross', 0)}, costs=${vs.get('costs', 0)}, "
          f"maxDD=${vs.get('max_dd', 0)}, EV/month=${report['ev_month']}",
          f"- by hypothesis: {vs.get('by_hyp')}",
          f"- monthly: {vs.get('monthly')}",
          f"- short side (H5): n={len(shorts)}, net=${short_net}",
          f"- exit mix: {vs.get('exit_mix')}", "",
          "## Dev 2021-2024 (context, MNQ, R=1.0)", "",
          f"- trades={dv.get('trades', 0)}, net=${dv.get('net', 0)}, "
          f"EV/trade=${dv.get('ev_trade', 0)}, win={dv.get('win')}",
          f"- by hypothesis: {dv.get('by_hyp')}", "",
          "## Controls (R=1.0, MNQ)", "",
          f"- shuffled-gap dev: trades={sc_d.get('trades', 0)}, "
          f"net=${sc_d.get('net', 0)}; validation: "
          f"trades={sc_v.get('trades', 0)}, net=${sc_v.get('net', 0)}",
          f"- gap-specificity (pre-committed clause): strategy net vs "
          f"shuffled control net -> gap_specific={gap_specific}",
          f"- all-days baseline dev net="
          f"${res['dev']['all_days_baseline_R1']['MNQ'].get('net', 0)}, "
          f"val net="
          f"${res['val']['all_days_baseline_R1']['MNQ'].get('net', 0)}",
          "", "## Gates", ""]
    md += [f"- {k}: **{v}**" for k, v in gates.items()]
    md += ["", "## Secondary R multiples (validation, MNQ)", ""]
    for r in fz["r_secondary"]:
        s2 = res["val"][f"R{r}"]["MNQ"]
        md += [f"- R={r}: trades={s2.get('trades', 0)}, "
               f"net=${s2.get('net', 0)}, win={s2.get('win')}"]
    md += ["", "## NQ control (validation, R=1.0)", ""]
    nqs = res["val"][f"R{fz['r_primary']}"]["NQ"]
    md += [f"- trades={nqs.get('trades', 0)}, net=${nqs.get('net', 0)}, "
           f"EV/trade=${nqs.get('ev_trade', 0)}"]
    REPORT_MD.write_text("\n".join(md))
    print(f"VERDICT: {verdict} | val MNQ net ${vs.get('net', 0)} on "
          f"{vs.get('trades', 0)} trades | report written")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit(freeze() if cmd == "freeze"
                     else validate() if cmd == "validate"
                     else (print("usage: freeze|validate"), 1)[1])
