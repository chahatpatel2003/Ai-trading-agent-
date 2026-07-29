# Phase 27 — Backtest Integrity Audit + Clean High-PF Re-run

Before trusting any strategy result, audit the backtester for hindsight/leakage and fix the Phase-26
process. The user's concern was correct: Phase 26's `scan` **ranked train+val by PF**, which leaks
the validation set into selection.

**Result: AUDIT = PASS, process FIXED, clean Phase-26 re-run = REJECT.** No hindsight/leakage was
found; the leaky ranking did **not** manufacture a false positive (best PF unchanged); and under the
strict train-only-selection protocol the high-PF search still fails.

> **No high-PF NQ strategy was found under the clean non-hindsight protocol.**

---

## Part 1–4 — Backtest integrity audit: PASS

**Method.** The proof is *prefix-invariance*: a value at bar *k* computed on the FULL series must
equal the value computed on only the prefix `[0..k]`. If adding future bars ever changes a past
value or signal, that is hindsight. (`tests/unit/test_phase27_integrity.py`, +the pre-existing PIT
suite for structure/htf_context/context/build/mbp1.)

| check | module(s) | result |
|---|---|---|
| prefix==full features | structure, htf_context, context, build, mbp1 (existing) | PASS |
| prefix==full features | **htf_poi, daily_bias, indicators (ADX/Donchian/Bollinger), order-flow** | **PASS** |
| prefix==full **signal** | **Phase-26 strategy signal** (end-to-end) | **PASS** |
| entry after signal | engine `next_open` — signal at *t* fills at *t+1* open (entry_minute 605 ≠ signal 600) | PASS |
| stop before target | same-bar double-touch takes the **STOP** (conservative) | PASS |
| no future bar in exit | stops/targets evaluated only on bars after entry | PASS |
| MFE/MAE not used in decisions | tracked for reporting only | PASS |

**Stop/target + cost (Part 4):** TARGET = exact limit fill; STOP/EOD slipped 1 tick; same-bar
stop+target → STOP (no assuming the favorable fill); commissions/slippage applied per fill; NQ/MNQ
point values correct. **Exact cost model:** NQ $0.85/side (RT $1.70), MNQ $0.25/side; slippage 1
tick (0.25 pt)/fill; NQ win/loss **$6.70 / $11.70**, MNQ **$1.00 / $1.50**.

**Replay / trade-audit (Part 2):** `nqlab phase27 audit` exports a 123-trade CSV
(`phase27_trade_audit.csv`) with `signal_ts, entry_ts, entry_price, stop, target, exit_ts,
exit_price, exit_reason, mfe_after_entry, mae_after_entry, uses_future_data`. Every row shows
`signal_ts` strictly **before** `entry_ts` (e.g. signal 14:30 → entry 14:35) and
`uses_future_data = false`.

```
AUDIT: PASS — no hindsight/leakage detected.
```

## Part 5 — Phase 26 process correction

The leaky `scan` (ranked by validation PF) is superseded by **`phase26 clean`**, which enforces the
correct protocol:

1. **Search/rank on 2021–2023 only** (PF, return/maxDD, Sharpe).
2. **Freeze** the top-N train candidates.
3. **Test once on 2024** (validation is pass/fail **test-only**, never a ranking input).
4. **Lockbox 2025–2026 touched only for validation-passers** (PF ≥ 2.0 on val).

Confirmed: validation is no longer used for selection; the lockbox is untouched until a candidate
passes validation.

## Part 6 — Clean high-PF re-run

Ranked/frozen on **TRAIN 2021–2023 only**, then tested once on 2024:

| strategy (frozen by train) | train PF | train n | val PF | val n | val ret/DD | val Sharpe |
|---|--:|--:|--:|--:|--:|--:|
| trend-follow 15m | 1.23 | 610 | 1.46 | 211 | 3.37 | 2.51 |
| vol-expansion | 1.19 | 476 | 0.99 | 175 | −0.05 | −0.15 |
| HTF-POI swing | 1.19 | 877 | 1.02 | 274 | 0.17 | 0.19 |
| vol-expansion | 1.17 | 421 | 0.95 | 147 | −0.23 | −0.53 |
| HTF-POI swing | 1.15 | 846 | 1.12 | 300 | 1.21 | 0.86 |
| HTF-POI swing | 1.14 | 794 | 1.14 | 284 | 1.09 | 0.88 |
| HTF-POI swing | 1.14 | 304 | 0.96 | 100 | −0.10 | −0.26 |
| breakout | 1.10 | 614 | 1.03 | 196 | 0.14 | 0.20 |

- **Best TRAIN PF = 1.23** (the selection metric) — already far below the 2.0 minimum, so the best
  candidate fails the bar on the training set itself.
- Tested once on 2024: best val PF **1.46** (the trend-follow), most others **degrade** out-of-sample
  (1.19 → 0.99, etc.). **0 validation-passers**; lockbox **untouched**.
- The leaky train+val ranking gave best val PF 1.46; the clean train-only ranking gives the same
  1.46 — **the process flaw did not change the conclusion** (because nothing passed under either
  protocol and train PFs were independently weak ≤ 1.23).

## Verdict

```
Integrity audit: PASS (no hindsight/leakage).
Clean Phase-26 high-PF re-run: REJECT — best train PF 1.23, best validation PF 1.46, 0 passers,
lockbox 2025-2026 untouched.
```

**Final statement:** *No high-PF NQ strategy was found under the clean non-hindsight protocol.* The
Phase-26 conclusion is now confirmed under a correct, leakage-free, train-only-selection process:
NQ bar/price-action data does not contain a PF ≥ 2.0 edge at these horizons, and the earlier
ranking flaw was not what produced the negative result.

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **291 tests**, incl. 7 new integrity/leakage tests).
- ✅ Backtest integrity audit PASS — prefix-invariance proven for every feature layer + the
  Phase-26 signal; engine entry/exit ordering verified; trade-audit CSV exported.
- ✅ Phase-26 selection corrected: validation is test-only (`phase26 clean`); lockbox untouched
  until freeze.
- ✅ Clean re-run produced; no forced acceptance.

## Reproduction

```
nqlab phase27 audit                          # cost model + trade-audit CSV + leakage-test summary
pytest tests/unit/test_phase27_integrity.py  # prefix==full + stop-first + next-open proofs
nqlab phase26 clean --timeframes 5m,15m      # CLEAN: rank on TRAIN only, val test-only, lockbox for passers
```
