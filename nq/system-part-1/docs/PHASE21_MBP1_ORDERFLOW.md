# Phase 21 — MBP1 Order-Flow Microstructure Discovery (final NQ-only NY-AM attempt)

**Goal:** test whether MBP1 order flow contains a real tradable edge that bars and indicators
could not see (Phases 16–20 found none). This was the last untested data class.

**Status: built, run end-to-end, verdict = NO edge — all 24 candidates REJECT at validation.**
Order-flow signals (OFI/delta/CVD/absorption/liquidity) show **no directional edge before costs**,
with MFE ≈ MAE — the same structural result as the bar edge map. The 2025-H1 MBP1 lockbox was
**never touched** (nothing earned it). No forced acceptance.

---

## Data limitation (and verdict ceiling)

MBP1 exists only **through 2025-H1**. So: search/train **2021–2023**, validation **2024**, MBP1
lockbox **2025-01…2025-06 only** (6 months); there is no MBP1 for 2025-H2/2026-H1. Therefore the
best attainable verdict is **`ACCEPT_FOR_FORWARD_PAPER`** (never live) — confirmed by the rig's
forward-paper ceiling. None reached it.

## What was built

1. **Order-flow feature layer** (`features/orderflow.py`, tested): derived dynamics from the
   bar-level MBP1 columns (which already include `ofi`, `delta`, `cvd`, `tob_imbalance`,
   `bid/ask_depletion`, `trade_aggression`, `spread`, `liq_spread_ratio`) — rolling z-scores
   (`ofi_z`, `delta_z`, `cvd_mom_z`), OFI/delta acceleration, CVD slope, and 3-bar imbalance
   persistence. PIT-safe.
2. **Six MBP1 families A–F** (`phase21_families.py`, MBP1-required): absorption reversal,
   imbalance continuation, failed-aggressive-flow trap, liquidity-vacuum breakout, delta-divergence
   reversal, and the adverse-selection-avoidance filter — each combining order flow with key
   levels (swept swings, prior-bar range, VWAP). *(A breakout bug — `close > roll_high` where the
   rolling max included the current bar, giving 0 trades — was found and fixed with a shifted
   prior-bar level, so all six families genuinely fire.)*
3. **Reused the payout rig** (gauntlet + DSR + cost-stress, daily-control sweep, two-stage funded
   scaling, fleet) with the 2025-H1 MBP1 lockbox and `ACCEPT_FOR_FORWARD_PAPER` ceiling. CLI
   `phase21 search/freeze/lockbox/report`.

## Result — 0 of 24 cleared validation

12 candidates × {5m, 1m} = 24. Gauntlet: **REJECT 23, NEEDS_MORE_DATA 1, ACCEPT 0.** The dominant
rejection reason is **"no directional edge before costs."** Order-flow excursion diagnostics
(train 2021–2023):

| family (tf) | trades | gross **before** costs | cost drag | MFE | MAE | dir-edge before costs? |
|---|--:|--:|--:|--:|--:|:--:|
| liq_vacuum (5m, z1.25) | 321 | **+$1,748** | $805 | 29.0 | 26.7 | yes (faint) |
| liq_vacuum (5m, z0.75) | 575 | −$9 | $1,499 | 28.3 | 26.2 | no |
| imbalance_cont (5m) | 131 | −$1,772 | $302 | 30.7 | 31.4 | no |
| imbalance_cont (1m) | 597 | −$14,379 | $2,877 | 14.2 | 13.7 | no |
| absorption_rev (1m) | 176 | −$4,182 | $995 | 13.0 | 12.1 | no |
| delta_div (1m, z0.75) | 843 | −$73,890 (net) | large | 13.7 | 13.7 | no |

- **MFE ≈ MAE for every order-flow signal** (ratios ~0.95–1.10) — order flow predicts NQ NY-AM
  direction no better than price bars. This matches the Phase-19 edge map exactly.
- The **single** signal with a positive *pre-cost* edge — liquidity-vacuum breakout (5m) — makes
  only ≈ **+$942 net over three years** after its $805 cost drag, **fails cost/slippage stress**
  (dies at 2× slippage), and has **53% risk of ruin** with only 0.75 positive years. Not tradable.
- On 1m, order flow loses heavily (gross −$9k to −$74k) — costs + no edge, as in Phase 18.

## Verdict

```
REJECT (all 24). MBP1 order-flow contains no robust directional edge before costs.
2025-H1 MBP1 lockbox left untouched (nothing earned a lockbox test).
```

## FINAL PROGRAM VERDICT

> **"NQ-only NY-AM automated Topstep payout strategy was not found with available data."**

Every data class and strategy family has now been tested and honestly rejected:

| phase | approach | outcome |
|---|---|---|
| 16 | payout-first structure/liquidity | REJECT (collapsed OOS) |
| 17 | A+ large-R asymmetric | REJECT (collapsed OOS) |
| 18 | high-win-rate scalps | REJECT (no edge before costs, 28–37% win) |
| 19 | edge-discovery map | **no directional bar edge** (MFE ≈ MAE everywhere) |
| 20 | classic indicators | REJECT (no edge / blow-up / cost drag) |
| 21 | **MBP1 order-flow** | **REJECT (no edge before costs, MFE ≈ MAE)** |

The converging evidence — a structural edge map (19) plus five independent strategy phases across
price-action, indicators, and order flow — is that **NQ NY-AM intraday direction is, after
realistic costs, indistinguishable from a symmetric random walk in every context, indicator, and
order-flow bucket tested.** There is no payout-grade directional edge to build a bot on with the
available NQ-only data.

**Recommendation: stop the NQ-only NY-AM automated-bot search.** Continuing to generate strategy
families against the same edgeless data will keep producing rejects. If the project is to continue,
it requires a *different* premise — additional instruments / cross-asset signals, a non-intraday
horizon, a non-directional (e.g. volatility/options) structure, or genuinely new data — none of
which are in scope for the NQ-only NY-AM mandate. The only thing that could revisit Phase 21's
result is **post-2025-H1 MBP1 forward data**, and even then the pre-cost MFE≈MAE finding makes a
robust edge unlikely.

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **258 tests**, incl. order-flow + Phase-21 tests).
- ✅ 2025-H1 MBP1 lockbox **never touched** (0 validation survivors).
- ✅ Every candidate has a verdict + reason; diagnostics-first "no edge before costs" active; the
  inclusive-rolling breakout bug was caught and fixed so all six families fire.
- ✅ No forced acceptance.
- ✅ Final report states the program conclusion plainly.

## Reproduction

```
nqlab phase21 search --timeframes 5m,1m --account 150k   # train 2021-23 / val 2024
nqlab phase21 freeze --top-n 8                            # 0 survivors -> nothing frozen
# (no lockbox run: nothing cleared validation; 2025-H1 MBP1 lockbox stays untouched)
```
