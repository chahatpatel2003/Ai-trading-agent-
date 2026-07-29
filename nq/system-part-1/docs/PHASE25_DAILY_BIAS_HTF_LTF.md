# Phase 25 — Daily Bias + HTF Trend + HTF POI + LTF Reclaim/IFVG/CISD/MSS

The most complete discretionary model tested: **daily bias + HTF-trend alignment as MANDATORY
gates** on top of the HTF-POI → LTF-reclaim sequence.

> Daily bias + HTF trend must agree → price trades into an HTF POI → LTF reclaim/IFVG/CISD/MSS
> confirmation → enter only in the daily/HTF bias direction.

**Answer: No.** With the mandatory alignment gate, the complete model wins **25–45% at 1:1** (best
IFVG 45%, robust-sample families 29–45%) — below the 55% floor, implying gross PF ~0.82 (losing
before costs). Critically, the **mandatory daily-bias + HTF-trend alignment does NOT improve win
rate** (clean test below), tripping the spec's first early-reject. 0 of 32 candidates cleared
validation; the lockbox was never touched. Adversarially verified (5 agents): all claims hold,
both verdict judges REJECT. No forced acceptance; the model was tested as the complete sequence.

---

## What was built (mandatory gate, not watered down)

- **`features/daily_bias.py`** (PIT-safe, tested): two explainable daily-bias definitions tested
  separately — **A** daily structure (prior-day HH/HL vs LH/LL) and **B** gap + prior-day candle —
  the **HTF trend** (1H + 4H agreement), the **mandatory alignment flags** (`aligned_long/short`),
  and **CISD** (displacement closing through a prior extreme) + **IFVG** (inverted-FVG reclaim)
  confirmations.
- **8 families** (`phase25_families.py`), every one gated by `aligned_long/short` (no trade unless
  daily bias AND HTF trend agree) + an HTF-POI touch: **A** IFVG reclaim, **B** CISD, **C**
  MSS/CHoCH, **D** sweep+reclaim, **E** displacement+FVG, **F** conservative combo, plus secondary
  models **G** momentum breakout+pullback and **H** pivot/S-R reclaim. Fixed 1:1 / 1.5R, 1 trade/day.
- Reused the payout rig (gauntlet + DSR + cost-stress, daily controls, Topstep 50/100/150K,
  scaling) with a **55% win-rate floor**, the audited Phase-22 cost model, and the
  `ACCEPT_FOR_FORWARD_PAPER` ceiling.

## Exact cost model (audited Phase 22 — reprinted)

NQ commission $0.85/side ($1.70 RT), MNQ $0.25 ($0.50 RT); slippage 1 tick/fill (entry always,
STOP/EOD slipped, TARGET = exact limit); NQ win/loss $6.70/$11.70, MNQ $1.00/$1.50.

## Data protocol

Train 2021–2023, validation 2024, lockbox 2025–2026 **untouched** (no candidate earned a freeze).
Daily bias, HTF POI, CISD/IFVG all PIT-safe (prior-session structure; shifted rolling levels).

## Result — 0 of 32 cleared validation

Per-family 1:1 (10/10, bias-def A) win rate, validation 2024:

| family | n | win % |
|---|--:|--:|
| **A IFVG reclaim** | 58 | **45%** |
| H pivot / S-R | 107 | 36% |
| B CISD | 52 | 29% |
| C MSS/CHoCH | 65 | 32% |
| E displacement + FVG | 41 | 32% |
| G momentum pullback | 92 | 32% |
| D sweep + reclaim | 4 | 25% (thin) |
| F conservative combo | 1 | — (n=1, noise) |

Every robust-sample family wins **29–45%**, below 55% (and below the 50% breakeven for 1:1).

## The decisive test — does the mandatory alignment improve win rate?

Clean apples-to-apples (same MSS@POI setup, with vs without the daily-bias+HTF-trend gate, val 2024):

| MSS @ POI | trades | win % |
|---|--:|--:|
| **without** bias+trend gate | 175 | **34.3%** |
| **with** bias+trend gate | 65 | **32.3%** |

The mandatory alignment **does not improve win rate** — it slightly lowers it (34.3% → 32.3%) and
just cuts the trade count (175 → 65). (A pivot/S-R pair nominally "improves" 5.3% → 35.5%, but its
"without" arm is a degenerate n=19 baseline with an inverted trade-count direction, so it is not a
clean toggle.) This trips early-reject #1. The best gated win rate (45%, IFVG) equals the **ungated
Phase 23 ceiling (34–45%)**, confirming the daily-bias + HTF-trend filter adds **no edge** — just
selectivity.

## Verdict

```
REJECT (all 32). The complete daily-bias + HTF-trend + HTF-POI + LTF-reclaim model wins <=45%
at 1:1 (best IFVG 45%); the mandatory alignment does not improve win rate; gross PF ~0.82.
Lockbox 2025-2026 untouched.
```

## Direct answer

> **"Does daily bias + HTF trend + HTF POI + LTF reclaim/IFVG/CISD/MSS produce a Topstep-usable NQ
> NY-AM strategy?"**
>
> **No.** Tested as the complete sequence with daily-bias + HTF-trend alignment **mandatory**, the
> model wins 25–45% at 1:1 — below the 55% floor and well below PF 1.30 (45% at 1:1 ⇒ gross PF
> ~0.82, losing before costs). The mandatory alignment provides no win-rate improvement (clean test:
> 34.3% → 32.3%), and the best result matches the ungated Phase-23 ceiling, so daily bias + HTF
> trend add selectivity but no edge. This is exactly what the Phase-19 edge map predicts (~52%
> directional accuracy, MFE ≈ MAE): no asymmetry to harvest at 1:1, in any context or confirmation
> stack tested.

Across Phases 16–25 — structure/liquidity, large-R, scalps, the edge map, classic indicators,
MBP1 order flow, HTF-POI→reclaim, prop-style 1:1 models, and now the full daily-bias+HTF-trend
sequence — every approach has been honestly rejected. The program conclusion stands: **"NQ-only
NY-AM automated Topstep payout strategy was not found with available data."**

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **279 tests**, incl. daily-bias + Phase-25 tests).
- ✅ Lockbox 2025-2026 **never touched** (0 validation survivors).
- ✅ Daily-bias + HTF-trend alignment enforced as a MANDATORY gate in every family (verified by the
  with/without comparison); the complete model tested, not watered down.
- ✅ Every candidate has a verdict + reason; 55% win floor enforced; the decisive
  alignment-improvement test run; adversarially verified (judges: REJECT). No forced acceptance.

## Reproduction

```
nqlab phase22 cost-audit                              # exact cost model (reused)
nqlab phase25 search --timeframes 5m --account 150k   # bias-gated complete model, train/val
nqlab phase25 freeze --top-n 8                        # 0 survivors -> nothing frozen
```
