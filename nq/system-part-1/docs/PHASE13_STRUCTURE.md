# Phase 13 — NQ-only Market Structure + Liquidity + Displacement Discovery

**Goal:** a genuinely new NQ-only signal source — market structure (swings, BOS, CHoCH,
MSS), liquidity (sweeps, reclaims, equal highs/lows), displacement (impulse candles),
and fair value gaps — searched under the reserved-lockbox protocol. Because these
families have never touched 2025-2026, the final lockbox is **pristine**, so
`ACCEPT_FOR_PAPER` is a reachable verdict.

**Verdict: the strongest validation signal in the project — but not confirmable on
held-out data. 0 ACCEPT_FOR_PAPER, 0 REJECT, 6 NEEDS_MORE_DATA.** The `bos_continuation`
family posted the **highest validation DSR the project has produced (0.757)**, ruin 0%,
100% positive years; `fvg_displacement` was close behind. But every validation survivor
is a **15m** setup, and at the risk levels that validated (≤$150) those setups size to
**~0 MNQ contracts in the higher-ATR 2026 regime** — so they generate too few held-out
trades to confirm. The variants that *would* trade enough (5m, or risk $200) did not
pass validation, so freezing them would be lockbox-peeking. No structure strategy both
validates and confirms.

NQ only; no ES. Retired edges (Phase 6-8 order-flow, Phase 9 HTF, Phase 11/12 trend_day)
were not touched.

---

## Features built (PIT-safe; the swing is the lookahead trap)

`features/structure.py` adds, on the existing frame, a full structure layer:

- **Structure:** confirmation-lagged swing highs/lows (a fractal swing is exposed only
  `swing_k` bars after it prints), HH/HL/LH/LL, BOS, CHoCH, MSS (CHoCH + displacement),
  `struct_dir`.
- **Liquidity:** equal highs/lows, sweep + reclaim of prior-day / swing levels,
  distance to swings.
- **Displacement:** impulse candle (body-to-range + ATR expansion), range expansion.
- **FVG:** bullish/bearish fair value gaps + mitigation/entry zones.
- **Premium/discount:** dealing-range midpoint, in-premium / in-discount.

Every swing-derived column uses a centered fractal → `shift(k)` → forward-fill so its
value at row *t* depends only on rows ≤ *t*; FVGs (3-bar) and displacement (1-bar) are
knowable at the forming bar. **`test_structure_pit.py`** proves it: prefix==full at many
bars, plus a direct test that a swing-high price never appears before its confirmation
bar.

Five families (all asymmetric, NY-AM, trades-only): **A** liquidity-sweep + MSS
reversal, **B** BOS continuation, **C** failed-breakout trap, **D** FVG-displacement
continuation, **E** premium/discount liquidity model.

---

## Protocol & validation (2024)

Search/train 2021-2023, validation 2024 (search loads only through 2024-12-31), final
untouched lockbox 2025-2026. 480 candidates (5m+15m × 50/100/150K × {MNQ-only,
adaptive} × risk {$50-$200}), strict locked-buffer Topstep sim, conservative
costs/slippage, diagnostics-first prefilter, DSR/PBO, Monte-Carlo ruin.

| Validation finding | |
|---|---|
| Gauntlet-survivors (passed 2024 economics) | 24 (16 `bos_continuation`, 8 `fvg_displacement`) |
| Best: `bos_continuation` 15m r$100 | 115 trades, +$2,759 OOS-2024, ruin 0%, WF 67%, **100% pos-yrs, valDSR 0.757** |
| `fvg_displacement` 15m r$100 | 235 trades, +$2,220, WF 83%, 100% pos-yrs, valDSR 0.40 |
| 5m structure families | **failed validation** (none survived the 2024 gauntlet) |
| 15m PBO / 5m PBO | ~10% / ~67% |

`bos_continuation`'s validation DSR (0.757) is the strongest single validation result
across all 13 phases.

## Final lockbox (untouched 2025-2026, pristine — ACCEPT_FOR_PAPER reachable)

| Frozen (validation survivor) | Held-out trades (OOS gate) | Full-window 2025-26 trades | Verdict |
|---|---:|---:|---|
| bos_continuation 15m r$100 (50/100/150K) | 0 (21 full-window) | 21 | NEEDS_MORE_DATA |
| bos_continuation 15m r$150 | 0 | ~70 | NEEDS_MORE_DATA |
| fvg_displacement 15m r$100 (100/150K) | 0 | 36 | NEEDS_MORE_DATA |

Best frozen profile (bos_continuation 15m, full-window 2025-26): win 33%, profit factor
1.07, expectancy $3.9, net +$82, max DD $554, ruin 0%, liquidity-target hit 29% — far
from paper-ready even before the trade-count problem.

## Why it cannot be confirmed (the regime/sizing wall, diagnosed)

A direct check of the 2025-2026 trade counts by risk and timeframe:

| Family / TF | risk $100 | risk $200 |
|---|---:|---:|
| fvg_displacement 15m | 36 trades, +$900 | 155 trades, +$118 |
| bos_continuation 15m | 21 trades, +$82 | 90 trades, +$485 |
| **fvg_displacement 5m** | **563 trades, +$2,899** | **1,017 trades, +$7,489** |

Two facts close the phase:

1. **The validated 15m setups don't size in 2026.** At a 1.5×ATR stop and ≤$150 risk, a
   15m bar's ATR (in points) makes one MNQ contract cost more than the risk budget in the
   higher-ATR 2025-2026 regime, so the setup fires but trades ~0 contracts in the OOS gate.
   risk-$200 sizes enough (90-155 trades) — but risk-$200 15m **was not a validation
   survivor**, so freezing it is not validation-justified.
2. **The high-frequency 5m `fvg_displacement` looks excellent on held-out data**
   (563 trades, +$2,899) — **but it failed the 2024 validation gauntlet**, so freezing it
   on the strength of its 2025-2026 numbers would be exactly the lockbox-peeking the
   protocol forbids.

So no structure strategy clears **both** gates: validate on 2024 *and* trade enough on
the untouched 2025-2026 to be confirmable.

---

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **203 tests**, incl. the no-lookahead structure
  tests and the swing-confirmation test).
- ✅ No-lookahead structure/liquidity tests pass (swings confirmation-lagged).
- ✅ Clean lockbox reserved before search (search ends 2024-12-31; lockbox `pristine=True`
  verified at run time).
- ✅ Discovery completes; every frozen candidate received a verdict.
- ✅ Nothing promoted — no structure setup passes validation AND the final lockbox.
  Nothing forced, no lockbox-peeking.

## Honest answer

> Market structure + liquidity + displacement is the most promising *validation* signal
> the project has found (`bos_continuation`, DSR 0.757, 100% positive years). It is the
> first family class whose **leaderboard is not overfit at 15m** (PBO ~10%). But it
> cannot be **confirmed**: the setups that validate are 15m and do not size on the
> higher-ATR 2025-2026 regime at validated risk, and the variants that trade enough (5m,
> or risk $200) did not validate. The honest verdict is **NEEDS_MORE_DATA** — confirm on
> a **forward feed** (especially a lower-ATR period, or a larger account that affords 15m
> contracts), where the validated 15m structure edge could finally accumulate enough
> trades to clear the bar.

Per the standing rule — *no edge until proven; reject honestly, do not force an accept* —
Phase 13 ends with **zero strategies promoted**, the structure signal flagged as the
project's best validation lead pending forward confirmation.

---

## Reproduction

```
nqlab features structure --no-build                 # PIT coverage report
nqlab discover structure-setups --no-build          # search: train 2021-23 / validation 2024
nqlab phase13 freeze --min-trades 60                # freeze validation survivors
nqlab phase13 lockbox --no-build                    # untouched 2025-2026 final test
nqlab report phase13                                # registry + verdicts + full stats + liquidity-hit
```
