# Phase 24 — Topstep Prop-Style 1:1 Scalping Models

Tests four specific prop-style high-win-rate models as **strict, filtered** prop versions (not the
rough generic forms from earlier phases):

1. **Opening Range Fade** (high win-rate failed-breakout reclaim)
2. **Liquidity Sweep Reversal**
3. **VWAP Reversion with Distance Filter**
4. **Micro Pullback Scalps in Trend**

**Answer: No.** Tested strictly at fixed 1:1 R:R, one trade/day, the best after-cost win rate is
**45%** (VWAP reversion), the others 27–37% — none reach the 60% floor (let alone 70%). 0 of 32
candidates cleared validation; the lockbox was never touched. Not a cost problem (zero-cost win
rates are within 1–2 pts of full-cost). No forced acceptance; the models were tested with their
filters, not watered down.

---

## What was built (strict, filtered)

- **Model 1 ORF** — first failed breakout that sweeps ≥2 pt beyond the OR and closes back inside;
  filtered for OR size (≥5 pt and ≤6×ATR) and non-expansion (no strong-impulse) opens.
- **Model 2 Sweep Reversal** — swing-liquidity sweep + fast reclaim + (displacement or CHoCH)
  confirmation.
- **Model 3 VWAP Reversion** — price stretched ≥1.0 ATR from VWAP, exhaustion/reclaim candle,
  ADX < 35 (skip only violent trends that walk the band).
- **Model 4 Micro Pullback in Trend** — 4h-trend + ADX ≥ 20 (trending, not chop) + a shallow VWAP
  pullback reclaimed in the trend direction.

All NQ NY-AM, **1 trade/day** (first qualifying signal), fixed **1:1** in {8/8, 10/10, 12/12,
15/15} pt, $200 risk via MNQ sizing. ADX/EMA computed PIT-safe (a NaN-poisoning bug in the ADX
EWM was found and fixed so the VWAP/pullback filters actually evaluate). Reused the payout rig with
a **60% win-rate floor** and the `ACCEPT_FOR_FORWARD_PAPER` ceiling.

## Exact cost model (audited Phase 22 — reprinted)

| | NQ | MNQ |
|---|---|---|
| commission / side | $0.85 | $0.25 |
| slippage | 1 tick / fill; entry always; STOP/EOD slipped; **TARGET = exact limit** | same |
| **cost / trade (win / loss)** | **$6.70 / $11.70** | **$1.00 / $1.50** |

## Data protocol

Train 2021–2023, validation 2024, lockbox 2025–2026 **untouched** (no candidate earned a freeze).

## Result — 0 of 32 cleared validation

16 candidates (4 models × 4 sizes) × {5m, 1m} = 32. All REJECT. Per-model 1:1 (10/10) economics,
validation 2024, with cost sensitivity (0× vs 1×):

| model | trades | win % (0×) | win % (1×) | PF (0×) | PF (1×) |
|---|--:|--:|--:|--:|--:|
| 3 VWAP reversion (distance) | 82 | **45%** | 43% | 0.82 | 0.69 |
| 1 Opening Range Fade | 146 | 37% | 36% | 0.59 | 0.53 |
| 2 Liquidity Sweep Reversal | 47 | 34% | 32% | 0.52 | 0.44 |
| 4 Micro Pullback in Trend | 157 | 27% | 27% | 0.38 | 0.34 |

- **No model reaches 60%** (or 50%) win at 1:1. The best — VWAP reversion with a distance filter —
  is 45% (PF 0.69). A 1:1 system needs ~57%+ for PF 1.30.
- **Zero-cost ≈ full-cost** for every model → the shortfall is missing directional edge, not costs.
- **Mean-reversion fades win most, trend-continuation wins least:** VWAP reversion (45%) and
  (Phase 23) reclaim+MSS (45%) are the two strongest 1:1 setups found anywhere, yet both cap at
  ~45%. The with-trend micro-pullback is *worst* (27%) — at 1:1, shallow pullbacks deepen past the
  10-pt stop before reaching the 10-pt target more often than not.

## Verdict

```
REJECT (all 32). None of the four strict prop-style 1:1 models reaches the 60% win-rate /
PF 1.30 bar after costs (best 45%). Lockbox 2025-2026 left untouched.
```

## Direct answer

> **"Can these four exact prop-style 1:1 scalping models produce a Topstep-usable NQ NY-AM
> strategy?"**
>
> **No.** Tested strictly with their filters at fixed 1:1, the four models win 27–45% after costs
> (best: VWAP reversion 45%), short of the 60% minimum and the 70% ideal. It is not a cost problem
> (zero-cost win rates match). The repeated finding holds: at 1:1 on NQ NY-AM, the symmetric target
> is reached < 50% of the time, so a 60–70% win-rate 1:1 prop scalp does not exist in this data —
> consistent with the Phase-19 edge map (~52% directional accuracy, MFE ≈ MAE) and the Phase-22/23
> 1:1 results (best 38–45%).

Across Phases 16–24, every strategy class — structure/liquidity, large-R, high-win-rate scalps,
the edge map, classic indicators, MBP1 order flow, the HTF-POI→reclaim sequence, and now the four
prop-style 1:1 models — has been tested and honestly rejected. The program conclusion stands:
**"NQ-only NY-AM automated Topstep payout strategy was not found with available data."**

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **274 tests**, incl. Phase-24 + NaN-safe-ADX tests).
- ✅ Lockbox 2025-2026 **never touched** (0 validation survivors).
- ✅ Models tested as strict, filtered prop versions (not watered-down generic tests); an ADX bug
  was fixed so the filters genuinely evaluate.
- ✅ Every candidate has a verdict + reason; 60% win floor enforced; cost sensitivity reported;
  no forced acceptance.

## Reproduction

```
nqlab phase22 cost-audit                                  # exact cost model (reused)
nqlab phase24 search --timeframes 5m,1m --account 150k    # four strict prop-style 1:1 models
nqlab phase24 freeze --top-n 8                            # 0 survivors -> nothing frozen
```
