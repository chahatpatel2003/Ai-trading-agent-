# topstep-lab

Leakage-proof strategy-discovery system for Topstep futures trading (NQ, GC).
This repo currently contains the **foundation only** — no strategies, no
optimization, by design. Every later layer builds on the guarantees here.

## Objective (fixed, encoded in `configs/topstep.yaml`)

Maximize **expected net payout before account failure**:
`E[ 0.9 × payouts − subscriptions − activation − resets ]` — not lifetime
survival, not combine pass-rate. Hard constraints: no rule violations, no
account-stacking exploit, no martingale, no lookahead bias.

## Guarantees

1. **Raw data is read-only.** Loaders/scanners open parquet files through
   pyarrow readers only; config validation refuses an outputs dir inside a
   data root; the inventory refuses paths outside declared roots.
2. **No lookahead.** Bars are stamped with OPEN time (Databento); loaders add
   `info_time = open + duration` and all feature joins must use it. The
   leakage auditor (`tslab.audit.leakage`) verifies truncation-invariance,
   fold embargo, and label horizons; strategy code must pass it before its
   backtests are believed.
3. **Verified rules.** Topstep parameters were verified against live official
   pages on 2026-07-02 — see `outputs/topstep_rules_verification.md` for
   sources. `verified: true` blocks in configs mark confirmed values.

## Layout

    configs/            instruments, costs, topstep accounts/payouts, sessions, paths
    src/tslab/
      config.py         typed+validated config loading (frozen dataclasses)
      sessions.py       CT session clock: 17:00 CT roll, 15:10 CT flatten,
                        DST-safe, 2016-2026 holiday/early-close calendar
      data/inventory.py read-only parquet scanner (schema, ranges, gaps, overlaps)
      data/loader.py    GC bar loader (implemented)
      data/nq_pipeline.py  NQ trades+MBP-1 -> front-month 1-min bars with
                        microstructure aggregates; prior-session-volume roll
      data/gc_pipeline.py  GC compatibility adapter to the same bar schema
      features/core.py  39 causal features (usable only from info_time;
                        latest_usable() enforces no-same-bar decisions)
      features/labels.py session-bounded labels: fixed-horizon, triple
                        barrier, MFE/MAE, hold-to-flatten; end-time metadata
                        feeds LabelHorizonCheck + session-fold embargo
      audit/leakage.py  leakage checks: truncation-invariance, embargo, label horizon
      audit/pipeline_gate.py  build gate: bar integrity, roll causality,
                        prefix (window-truncation) invariance
      sim/costs.py      fees (verified TopstepX schedule) + spread + slippage model
      sim/topstep.py    account lifecycle: trailing MLL, DLL, consistency target,
                        combine→XFA, scaling plan, payouts (90/10), fee ledger
      strategy/rules.py    declarative candidates: filters + trade rule (stop/
                        target/RR, time stop, max trades, daily loss/profit locks)
      strategy/backtest.py deterministic bar-loop engine; flatten violations
                        structurally impossible; CostModel per exit type
      strategy/evaluate.py trades -> TopstepSimulator (MAE marks, DLL lockouts,
                        greedy payouts) + lifecycle Monte Carlo (payout-EV)
      strategy/search.py   walk-forward search: session folds + embargo check,
                        train-only selection, OOS reporting, signal re-audit
    scripts/scan_data.py        run the inventory
    scripts/build_nq_sample.py  build NQ bar sample + leakage gate
    scripts/build_features_sample.py  features+labels + leakage gate
    scripts/run_smoke_search.py       tiny walk-forward search (machinery check)
    tests/                 170 tests covering all of the above
    outputs/               generated reports (gitignored) + rules verification doc

## Usage

    .venv/bin/python -m pytest                 # run tests
    .venv/bin/python scripts/scan_data.py      # scan all datasets (read-only)
    .venv/bin/python scripts/scan_data.py --dataset gc.ohlcv_1m

## Data (external T9 drive, read-only)

| dataset | kind | span | size |
|---|---|---|---|
| `nq.mbp1` | Databento MBP-1 ticks (top-of-book) | 2021 → 2025H1 | ~200 GB |
| `nq.trades` | Databento trade prints | 2021 → 2026H1 | ~20 GB |
| `gc.ohlcv_1m` | 1-min bars, continuous `GC.v.0` | 2016 → 2025 | ~53 MB |

NQ files contain outrights + calendar spreads + far months → must be filtered
(`outright_mask`) and rolled without lookahead. `gc_2024_jan.parquet`
duplicates part of `gc_2024.parquet` (the scanner flags the overlap; the
loader dedups).

## Key modeling notes

- **MLL** trails end-of-day balance highs, locks at start balance, breach
  checked in real time against equity (incl. unrealized).
- **DLL**: effective = min(Topstep DLL, personal DLL); personal DLL is
  configurable to 300/500/750/1000 (`risk_overrides`). Hitting it = lockout,
  not failure.
- **Combine**: min 2 days; best day ≤ 50% of target or the target grows.
- **GC is currently volatility-capped at 3/6/9 contracts** (MGC 30/60/90) — modeled.
- **Holiday calendar populated 2016–2026** (103 holidays, 95 early closes) in
  sessions.yaml, measured from recorded Globex behavior in the local GC/NQ
  data (2026-H2 entries pattern-derived, flagged VERIFY vs the CME notice).
  Halt holidays book to the next trade date; full closures kill the prior
  evening's session too; `sessions.validate_calendar()` guards any new range.
- Costs use the official TopstepX round-turns (NQ $3.80, MNQ $1.24, GC $4.24,
  MGC $1.74) plus conservative spread/slippage assumptions.

## Next stages (not started)

1. NQ tick → bar/feature pipeline with no-lookahead roll logic + book-aware fills.
2. Feature/label library that passes the leakage auditor by construction.
3. Walk-forward strategy search under the payout-EV objective (combine +
   funded Monte Carlo over the simulator).
