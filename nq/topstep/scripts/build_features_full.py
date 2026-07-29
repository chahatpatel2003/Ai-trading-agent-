#!/usr/bin/env python
"""Phase 3B-2: full-history NQ features + labels + walk-forward fold plan.

Reads outputs/nq_bars_full.parquet (1.92M bars), writes:
    outputs/nq_features_full.parquet
    outputs/nq_labels_full.parquet          (labels + session)
    outputs/nq_label_endtimes_full.parquet  (per-label information end times)
    outputs/fold_plan.json
    outputs/feature_schema_full.md / label_schema_full.md
    outputs/leakage_audit_full.json

Audits (exit != 0 on any FAIL):
    - feature truncation invariance (6 prefix recomputes at full scale)
    - no-future-session: perturbing an interior session leaves all earlier
      features bit-identical
    - no same-bar decision leak (latest_usable spot checks)
    - label structure at full scale, vectorized: fixed-horizon NaN tails per
      session; end_times >= decision info_time; end_times <= session close
      (session-boundedness); flatten labels end before the deadline
    - label horizon recompute checks on two subsets (quotes era 2023,
      trade-only era 2025)
    - fold plan: FoldEmbargoCheck (>= 1 session boundary) + exact per-fold
      label PURGE check (no train label ends past train_end)
"""
from __future__ import annotations

import json
import sys
import time as _time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab import sessions as S                                    # noqa: E402
from tslab.audit.leakage import (CheckResult, FoldEmbargoCheck,    # noqa: E402
                                 LabelHorizonCheck, LeakageAuditor,
                                 Status, TruncationInvarianceCheck)
from tslab.config import load_config                               # noqa: E402
from tslab.features.core import (FeatureParams, compute_features,  # noqa: E402
                                 feature_columns, latest_usable)
from tslab.features.labels import (LabelParams, check_label_purge, # noqa: E402
                                   compute_labels, make_session_folds,
                                   required_embargo)

N_TRAIN, N_TEST, EMBARGO = 250, 63, 1        # ~1y train, ~1q test, 1 session


def main() -> int:
    t0 = _time.time()
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    fparams, lparams = FeatureParams(), LabelParams()
    bars = pd.read_parquet(outdir / "nq_bars_full.parquet")
    sessions = sorted(set(bars["session"]))
    print(f"bars={len(bars):,} sessions={len(sessions)}", flush=True)

    t = _time.time()
    feats = compute_features(bars, cfg, fparams)
    print(f"features: {len(feature_columns(feats))} cols in "
          f"{_time.time() - t:.0f}s", flush=True)
    t = _time.time()
    labres = compute_labels(bars, cfg, lparams)
    print(f"labels: {len(labres.labels.columns)} cols in "
          f"{_time.time() - t:.0f}s", flush=True)

    feats.to_parquet(outdir / "nq_features_full.parquet")
    lab_out = labres.labels.copy()
    lab_out["session"] = bars["session"]
    lab_out.to_parquet(outdir / "nq_labels_full.parquet")
    labres.end_times.to_parquet(outdir / "nq_label_endtimes_full.parquet")

    results = []

    # ---- feature truncation invariance (full scale) ----
    t = _time.time()
    r = TruncationInvarianceCheck(
        bars, lambda df: compute_features(df, cfg, fparams), n_probes=6).run()
    r.name = "feature_truncation_invariance_full"
    results.append(r)
    print(f"  [{r.status.value}] {r.name} ({_time.time() - t:.0f}s): "
          f"{r.message}", flush=True)

    # ---- no future session data: perturb an interior session ----
    t = _time.time()
    mid = sessions[len(sessions) // 2]
    tampered = bars.copy()
    m = tampered["session"] == mid
    tampered.loc[m, ["open", "high", "low", "close"]] += 500.0
    feats_t = compute_features(tampered, cfg, fparams)
    early = bars["session"] < mid
    fcols = feature_columns(feats)
    try:
        pd.testing.assert_frame_equal(feats.loc[early, fcols],
                                      feats_t.loc[early, fcols])
        results.append(CheckResult(
            "no_future_session_data", Status.PASS,
            f"perturbing {mid} left {int(early.sum()):,} earlier feature "
            "rows bit-identical"))
    except AssertionError as e:
        results.append(CheckResult("no_future_session_data", Status.FAIL,
                                   f"earlier features changed: {e}"))
    print(f"  [{results[-1].status.value}] no_future_session_data "
          f"({_time.time() - t:.0f}s)", flush=True)
    del feats_t, tampered

    # ---- no same-bar decision leak ----
    rng = np.random.default_rng(23)
    bad = 0
    for i in rng.choice(np.arange(1, len(bars)), size=50, replace=False):
        at = bars.index[i] + pd.Timedelta(seconds=30)   # mid-bar decision
        row = latest_usable(feats, at)
        if row is None or row.name >= bars.index[i]:
            bad += 1
    results.append(CheckResult(
        "no_same_bar_decision_leak", Status.PASS if bad == 0 else Status.FAIL,
        f"{50 - bad}/50 mid-bar decisions resolved to a fully closed bar"))

    # ---- label structure at full scale (vectorized) ----
    info = pd.DatetimeIndex(bars["info_time"])
    sess_col = bars["session"]
    close_map = sess_col.map({s: S.globex_close_utc(s, cfg.sessions)
                              for s in sessions})
    flat_map = sess_col.map({s: S.eod_flatten_utc(s, cfg.sessions)
                             for s in sessions})
    pos_from_end = bars.groupby("session").cumcount(ascending=False)
    problems = []
    for h in lparams.fwd_horizons:
        lab = labres.labels[f"lab_fwd_{h}"]
        n_bad = int(lab[pos_from_end < h].notna().sum())
        if n_bad:
            problems.append(f"lab_fwd_{h}: {n_bad} non-NaN session-tail rows")
    for col in labres.end_times.columns:
        e = pd.DatetimeIndex(labres.end_times[col])
        ok = e.notna()
        if (e[ok] < info[ok]).any():
            problems.append(f"{col}: end before decision time")
        if (e[ok] > pd.DatetimeIndex(close_map)[ok]).any():
            problems.append(f"{col}: end past session close (not bounded)")
    e = pd.DatetimeIndex(labres.end_times["lab_flat_ret"])
    ok = e.notna()
    if (e[ok] >= pd.DatetimeIndex(flat_map)[ok]).any():
        problems.append("lab_flat_ret: end at/after flatten deadline")
    results.append(CheckResult(
        "label_structure_full", Status.PASS if not problems else Status.FAIL,
        ("all fixed tails NaN; all end_times in [decision, session close]; "
         "flat labels end before the deadline" if not problems
         else f"{len(problems)} problem(s)"),
        {"problems": problems}))
    print(f"  [{results[-1].status.value}] label_structure_full", flush=True)

    # ---- label horizon recompute checks on two era subsets ----
    for tag, lo, hi in (("2023-quotes", date(2023, 3, 6), date(2023, 3, 8)),
                        ("2025-tradeonly", date(2025, 9, 8), date(2025, 9, 10))):
        sub = bars[(bars["session"] >= lo) & (bars["session"] <= hi)]
        sub_res = compute_labels(sub, cfg, lparams)
        for spec in sub_res.specs:
            if spec.kind == "fixed":
                r = LabelHorizonCheck(
                    sub, lambda df, n=spec.name:
                        compute_labels(df, cfg, lparams).labels[n],
                    horizon_rows=spec.horizon_rows, n_probes=3).run()
            else:
                r = LabelHorizonCheck(
                    sub, lambda df, n=spec.name:
                        compute_labels(df, cfg, lparams).labels[n],
                    label_end_times=sub_res.end_times[spec.name],
                    data_end_time=pd.Timestamp(sub["info_time"].iloc[-1]),
                    n_probes=3).run()
            r.name = f"label_horizon:{spec.name}@{tag}"
            results.append(r)
        print(f"  label horizon checks @{tag} done", flush=True)

    # ---- fold plan + embargo + purge ----
    folds = make_session_folds(sessions, cfg, N_TRAIN, N_TEST, EMBARGO)
    emb = required_embargo(labres.specs)
    r = FoldEmbargoCheck(folds, emb).run()
    results.append(r)
    purge = check_label_purge(labres.end_times, folds)
    results.append(CheckResult(
        "label_purge_per_fold", Status.PASS if not purge else Status.FAIL,
        (f"{len(folds)} folds x {len(labres.end_times.columns)} labels: no "
         "train label ends past train_end" if not purge
         else f"{len(purge)} violation(s)"), {"violations": purge[:10]}))
    print(f"  [{r.status.value}] fold_embargo | "
          f"[{results[-1].status.value}] label_purge_per_fold", flush=True)

    fold_plan = {
        "n_train_sessions": N_TRAIN, "n_test_sessions": N_TEST,
        "embargo_sessions": EMBARGO, "n_folds": len(folds),
        "embargo_time_check": str(emb),
        "folds": [{
            "fold": k,
            "train_start": str(f["train_start"]), "train_end": str(f["train_end"]),
            "test_start": str(f["test_start"]), "test_end": str(f["test_end"]),
        } for k, f in enumerate(folds)],
    }
    (outdir / "fold_plan.json").write_text(json.dumps(fold_plan, indent=2))

    _write_feature_schema(feats, outdir / "feature_schema_full.md")
    _write_label_schema(labres, outdir / "label_schema_full.md", folds)
    LeakageAuditor.write_report(results, outdir, "full")

    n_fail = sum(1 for r in results if r.status == Status.FAIL)
    print(f"\n{len(results)} checks, {n_fail} FAIL | "
          f"total {(_time.time() - t0) / 60:.1f} min", flush=True)
    for r in results:
        if r.status == Status.FAIL:
            print(f"  FAIL {r.name}: {r.message}", flush=True)
    return 1 if n_fail else 0


def _write_feature_schema(feats, path):
    fcols = feature_columns(feats)
    lines = ["# Feature schema (full NQ history)", "",
             f"- rows: {len(feats):,}   feature columns: {len(fcols)}",
             "- usable only from info_time (bar close); select rows with "
             "`latest_usable`", "",
             "| column | dtype | NaN% |", "|---|---|---:|"]
    for c in fcols:
        lines.append(f"| {c} | {feats[c].dtype} | "
                     f"{feats[c].isna().mean() * 100:.1f} |")
    path.write_text("\n".join(lines))


def _write_label_schema(labres, path, folds):
    lines = ["# Label schema (full NQ history)", "",
             f"- rows: {len(labres.labels):,}   folds: {len(folds)}",
             "- decision at info_time(T), entry open[T+1]; all labels "
             "session-bounded; end_times stored per label", "",
             "| label | kind | max horizon (bars) | NaN% |",
             "|---|---|---:|---:|"]
    for s in labres.specs:
        lines.append(f"| {s.name} | {s.kind} | {s.horizon_rows} | "
                     f"{labres.labels[s.name].isna().mean() * 100:.1f} |")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
