# Phase 23 — HTF POI Reaction + LTF Reclaim/MSS Model

Tests one specific discretionary sequence as a **complete model** (not generic components):

> HTF POI → price trades into the POI → sweep/reaction → LTF reclaim / MSS(CHoCH) confirmation →
> enter from the POI reaction.

**Answer: No.** Tested as the full sequence across four entry variants, the best after-cost win
rate is **45%** (the most-conservative reclaim+MSS variant, on only 11 trades), most variants win
**~34%**, and none reach the 55% floor (let alone 70%). 0 of 72 candidates cleared validation; the
lockbox was never touched. The model was tested faithfully, not watered down. No forced acceptance.

---

## What was built (the complete sequence)

1. **PIT-safe HTF POI layer** (`features/htf_poi.py`, tested): HTF points of interest from
   confirmed-and-past liquidity only — 1H/4H rolling swing extremes (prior 12/48 5m bars, shifted),
   prior day/week high & low, overnight high & low. Exposes the nearest **support POI** (level
   at/below price) and **resistance POI** (at/above), a "traded into and reclaimed the POI within
   tolerance" touch flag, and a **recent-touch window** (touched within the last K bars) so the LTF
   confirmation can fire a few bars *after* the touch — modelling the real sequence rather than
   demanding everything on one bar.
2. **Four entry variants** (`phase23_families.py`), each the full sequence (POI reaction +
   variant-specific LTF confirmation), 1 trade/day, NY-AM:
   - **A reclaim-close** — enter when the LTF candle closes back through the POI;
   - **B MSS/CHoCH confirm** — POI reaction + a bullish/bearish LTF structure shift after it;
   - **C displacement + FVG retrace** — POI reaction + displacement + LTF FVG mitigation;
   - **D conservative reclaim + MSS** — require both the reclaim and a structure shift.
3. Fixed stop (10 pt) with **1:1 / 1.5R / 2R** targets, POI-tolerance grid {2, 5, 10 pt}; reused
   the payout rig (gauntlet + DSR + cost-stress, daily controls, Topstep 50/100/150K, scaling) with
   a **55% win-rate floor** and the `ACCEPT_FOR_FORWARD_PAPER` ceiling.

*Faithfulness note:* two modelling fixes were required to test the sequence honestly — (i) a
recent-touch window so MSS/CHoCH can confirm *after* the POI touch, and (ii) dropping a spurious
`in_discount` AND-condition at the confirmation bar (a bullish MSS structurally lifts price out of
discount, so AND-ing it made the test vacuously empty). The support/resistance POI itself encodes
the discount/premium location. Both changes make the test *more* faithful, not weaker.

## Exact cost model (audited, Phase 22 — reprinted)

| | NQ | MNQ |
|---|---|---|
| commission / side | $0.85 | $0.25 |
| slippage | 1 tick (0.25 pt) per fill; entry always; STOP/EOD slipped; **TARGET = exact limit** | same |
| **cost / trade (win / loss)** | **$6.70 / $11.70** | **$1.00 / $1.50** |

Cost sensitivity is reported per-variant below (0× ≈ 1×) — costs are not the deciding factor.

## Data protocol

Train/search 2021–2023, validation 2024, lockbox 2025–2026 **untouched** (no candidate earned a
freeze). HTF POIs are PIT-safe (rolling extremes shifted; prior-period levels), no lookahead.

## Result — 0 of 72 cleared validation

36 candidates (4 variants × 3 tolerances × 3 R-multiples) × {5m, 15m} = 72. All REJECT. Per-variant
1:1 (10/10, tol 5) economics on validation 2024:

| variant | trades | win % (0× cost) | win % (1× cost) | PF (1×) |
|---|--:|--:|--:|--:|
| A reclaim-close | 39 | 21% | 18% | 0.20 |
| B MSS/CHoCH confirm | 175 | 34% | 34% | 0.48 |
| C displacement + FVG | 131 | 34% | 33% | 0.45 |
| **D reclaim + MSS (conservative)** | 11 | **45%** | **45%** | 0.77 |

- **No variant reaches 55%** win at 1:1 — the best (conservative D) is 45% on a tiny 11-trade
  sample. A 1:1 system needs ~57%+ to clear PF 1.30; the model produces 34–45%.
- **Zero-cost ≈ 1×-cost** for every variant → the shortfall is missing directional edge, not costs.
- **More confirmation helps marginally but not enough:** stacking reclaim + MSS lifts win rate to
  45% (highest) but collapses frequency to ~11 trades/year — confirming the components add a little
  selectivity but never cross the threshold, and the trade-off destroys payout frequency.

## Verdict

```
REJECT (all 72). The HTF-POI -> LTF-reclaim/MSS sequence does not produce a Topstep-usable
NQ strategy. Best after-cost win rate 45% (n=11); most ~34%; none >= 55%. Lockbox untouched.
```

## Direct answer

> **"Does the specific HTF POI → LTF reclaim/MSS model produce a Topstep-usable NQ strategy?"**
>
> **No.** Tested as the complete sequence (POI touch → reclaim → MSS/CHoCH/displacement/FVG), the
> best win rate is 45% (conservative variant, 11 trades) and the typical variant wins ~34% at 1:1
> — far short of the 55% floor and the 60–70% target. It is not a cost problem (zero-cost win rates
> are identical). Adding more confirmation raises win rate slightly but never to threshold and
> guts the trade count. This is the same conclusion the Phase-19 edge map predicted (~52%
> directional accuracy, MFE ≈ MAE): conditioning on HTF-POI + LTF-reclaim does not lift NQ NY-AM
> directional accuracy into payout-grade territory.

Consistent with Phases 16–22, the overall program conclusion stands: **"NQ-only NY-AM automated
Topstep payout strategy was not found with available data."** Both the generic ICT/SMC components
(earlier phases) and now the complete discretionary HTF-POI→LTF-reclaim sequence fail.

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **269 tests**, incl. HTF-POI + Phase-23 tests).
- ✅ Lockbox 2025-2026 **never touched** (0 validation survivors).
- ✅ HTF POIs PIT-safe (verified by a prefix==full test); the complete sequence modelled, not
  watered down into generic FVG/OB/bias entries.
- ✅ Every candidate has a verdict + reason; 55% win floor enforced; no forced acceptance.
- ✅ The phase question answered directly: the model is **not** Topstep-usable.

## Reproduction

```
nqlab phase22 cost-audit                                  # exact cost model (reused here)
nqlab phase23 search --timeframes 5m,15m --account 150k   # full HTF-POI->LTF-reclaim sequence
nqlab phase23 freeze --top-n 8                            # 0 survivors -> nothing frozen
```
