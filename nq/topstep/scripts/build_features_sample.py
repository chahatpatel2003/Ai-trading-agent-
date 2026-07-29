#!/usr/bin/env python
"""Phase 2C gate: compute features + labels on the Phase 2B NQ bar sample,
run the leakage auditor over ALL of them, and write:

    outputs/nq_features_sample.parquet, outputs/nq_labels_sample.parquet
    outputs/feature_schema.md, outputs/label_schema.md
    outputs/leakage_audit_features.json

Exit != 0 if any check FAILs. Label-horizon probes run on a 2-session subset
(recomputing the label scan per probe is the cost driver); truncation runs on
the full sample.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tslab.audit.leakage import (FoldEmbargoCheck, LabelHorizonCheck,   # noqa: E402
                                 LeakageAuditor, TruncationInvarianceCheck)
from tslab.config import load_config                                    # noqa: E402
from tslab.features.core import (FeatureParams, compute_features,       # noqa: E402
                                 feature_columns)
from tslab.features.labels import (LabelParams, compute_labels,         # noqa: E402
                                   make_session_folds, required_embargo)


def main() -> int:
    cfg = load_config()
    outdir = cfg.paths.outputs_dir
    bars_path = outdir / "nq_bars_sample.parquet"
    if not bars_path.exists():
        print("run scripts/build_nq_sample.py first", file=sys.stderr)
        return 2
    bars = pd.read_parquet(bars_path)
    fparams, lparams = FeatureParams(), LabelParams()

    print(f"bars: {len(bars):,}  sessions: {bars['session'].nunique()}")
    feats = compute_features(bars, cfg, fparams)
    labres = compute_labels(bars, cfg, lparams)
    print(f"features: {len(feature_columns(feats))} columns   "
          f"labels: {len(labres.labels.columns)} columns")

    auditor = LeakageAuditor()
    auditor.register(TruncationInvarianceCheck(
        bars, lambda df: compute_features(df, cfg, fparams), n_probes=8))

    sessions = sorted(set(bars["session"]))
    sub = bars[bars["session"].isin(sessions[:2])]
    for spec in labres.specs:
        name = spec.name
        if spec.kind == "fixed":
            auditor.register(LabelHorizonCheck(
                sub, _label_fn(cfg, lparams, name),
                horizon_rows=spec.horizon_rows, n_probes=4))
        else:
            ends = compute_labels(sub, cfg, lparams).end_times[name]
            auditor.register(LabelHorizonCheck(
                sub, _label_fn(cfg, lparams, name),
                label_end_times=ends, n_probes=4))

    folds = make_session_folds(sessions, cfg, n_train=5, n_test=2,
                               embargo_sessions=1)
    auditor.register(FoldEmbargoCheck(folds, required_embargo(labres.specs)))

    results = auditor.run_all()
    LeakageAuditor.write_report(results, outdir, "features")
    ok = True
    for r in results:
        print(f"  [{r.status.value}] {r.name}: {r.message}")
        ok &= r.status.value != "FAIL"

    feats.to_parquet(outdir / "nq_features_sample.parquet")
    lab_out = labres.labels.copy()
    lab_out["session"] = bars["session"]
    lab_out.to_parquet(outdir / "nq_labels_sample.parquet")
    _write_feature_schema(feats, outdir / "feature_schema.md")
    _write_label_schema(labres, outdir / "label_schema.md", folds)
    print(f"wrote {outdir}/nq_features_sample.parquet, nq_labels_sample.parquet")
    print(f"wrote {outdir}/feature_schema.md, label_schema.md")
    if not ok:
        print("LEAKAGE GATE FAILED", file=sys.stderr)
    return 0 if ok else 1


def _label_fn(cfg, lparams, name):
    from tslab.features.labels import compute_labels
    return lambda df: compute_labels(df, cfg, lparams).labels[name]


DESC = {
    "f_ret_": "log return over N closed bars (per contract block)",
    "f_atr_ticks": "ATR(14) in ticks", "f_range_atr": "bar range / ATR",
    "f_retstd_": "std of 1-bar log returns over window",
    "f_rv_": "sum of within-bar realized variance over window",
    "f_sma_dist_": "(close - SMA_w) / ATR", "f_z_": "close z-score vs window",
    "f_svwap_dist_atr": "(close - session-anchored VWAP) / ATR",
    "f_sess_cumvol": "cumulative session volume",
    "f_vol_ratio": "bar volume / rolling median",
    "f_ntrades_z": "trade-count z-score",
    "f_spread_ratio": "bar mean spread / rolling median spread",
    "f_book_imb": "(bid_sz - ask_sz)/(bid_sz + ask_sz) at bar close",
    "f_quote_ratio": "quote updates / rolling median",
    "f_delta_": "aggressor delta / volume over window",
    "f_cvd_sess": "session cumulative delta / cumulative volume",
    "f_buy_share_15": "aggressor buy share over 15 bars",
    "f_min_since_open": "minutes since session open (Sun/holiday-aware)",
    "f_min_to_flatten": "minutes from info_time to flatten deadline",
    "f_ct_hour": "Chicago hour of bar open", "f_dow": "session weekday",
    "f_rth": "bar inside RTH reference window",
    "f_prior_high_dist": "(close - prior session high) / ATR",
    "f_prior_low_dist": "(close - prior session low) / ATR",
    "f_prior_close_gap": "(close - prior session close) / ATR",
    "f_in_prior_range": "close within prior session range",
    "f_or_high_dist": "(close - opening range high)/ATR, NaN until OR complete",
    "f_or_low_dist": "(close - opening range low)/ATR, NaN until OR complete",
    "f_or_break_up": "close above OR high", "f_or_break_dn": "close below OR low",
}


def _desc(col: str) -> str:
    for k, v in DESC.items():
        if col == k or col.startswith(k):
            return v
    return ""


def _write_feature_schema(feats: pd.DataFrame, path: Path) -> None:
    fcols = feature_columns(feats)
    lines = ["# Feature schema (NQ sample)", "",
             f"- rows: {len(feats):,}   feature columns: {len(fcols)}",
             "- row at bar_open T is usable ONLY from info_time = T+60s "
             "(select with `latest_usable`)", "",
             "| column | dtype | NaN% | meaning |", "|---|---|---:|---|"]
    for c in fcols:
        nan = feats[c].isna().mean() * 100
        lines.append(f"| {c} | {feats[c].dtype} | {nan:.1f} | {_desc(c)} |")
    path.write_text("\n".join(lines))


def _write_label_schema(labres, path: Path, folds) -> None:
    lines = ["# Label schema (NQ sample)", "",
             "- decision at info_time(T); entry = open[T+1]; all labels "
             "session-bounded (never cross a session/holiday boundary)",
             "- embargo: 1 full session boundary (>= 24h); folds built by "
             f"make_session_folds — {len(folds)} folds on the sample", "",
             "| label | kind | max horizon (bars) | NaN% | meaning |",
             "|---|---|---:|---:|---|"]
    for s in labres.specs:
        nan = labres.labels[s.name].isna().mean() * 100
        lines.append(f"| {s.name} | {s.kind} | {s.horizon_rows} | "
                     f"{nan:.1f} | {s.description} |")
    lines += ["", "Notes:",
              "- lab_tb ties (target and stop pierced in one bar) resolve "
              "stop-first (conservative).",
              "- deadline bar = last bar whose close time is strictly before "
              "the early-close-aware flatten deadline.",
              "- per-row information end times are stored alongside labels "
              "and drive LabelHorizonCheck (variable mode)."]
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
