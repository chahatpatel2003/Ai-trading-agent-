# Phase 19 — NQ NY-AM Edge-Discovery Map (behaviour, not a backtest)

**Goal:** before building any more strategies, find *where NQ actually has tradable directional
edge*. For every NY-AM bar (09:30–12:30 ET) we measured forward outcomes (signed return, MFE, MAE)
over 5/10/15/30/60-min horizons and bucketed by market context, on **train 2021-2023** and
**validation 2024**. The 2025-2026 lockbox was **never touched**. No strategy was backtested, no
parameter optimised.

A context only qualifies as an edge if: **gross edge before costs is positive, MFE clearly > MAE
for one direction, enough sample, after-cost edge big enough for $200 days at safe size, and it
holds in 2024.** Round-trip cost ≈ **0.75 NQ pt** (1-tick slippage/side + $0.85/side commission).

---

## Baseline — is there ANY directional drift? (30-min horizon)

| set | bars | mean fwd | MFE | MAE | **MFE/MAE** | dir acc |
|---|--:|--:|--:|--:|--:|--:|
| train 2021-23 | 27,828 | +0.055 pt | 31.4 | 32.9 | **0.96** | 52.6% |
| val 2024 | 9,324 | +0.239 pt | 32.7 | 35.9 | **0.91** | 52.7% |

**The average NY-AM bar has MFE ≈ MAE (ratio ≤1.0) and ~52.6% directional accuracy.** After the
0.75 pt cost, the expected outcome is negative in *either* direction. This is the structural
explanation for the Phase 16–18 failures: there is no directional drift to capture.

## Table 4 — Time-of-day edge map (30-min forward)

| window | n_tr | dir | aftC_tr | MFE/MAE | n_va | aftC_val | stable? |
|---|--:|--:|--:|--:|--:|--:|:--:|
| 09:30–09:45 | 2,319 | short | +0.14 | 0.97 | 777 | +0.86 | yes (thin) |
| 09:45–10:00 | 2,319 | short | +0.08 | 0.93 | 777 | −0.19 (flips long) | no |
| 10:00–10:30 | 4,638 | short | +0.21 | 0.93 | 1,554 | −0.35 (flips long) | no |
| 10:30–11:00 | 4,638 | long | −0.12 | 0.96 | 1,554 | −0.72 (flips short) | no |
| **11:00–11:30** | 4,638 | long | +0.70 | 1.00 | 1,554 | +0.42 | **yes (thin)** |
| 11:30–12:00 | 4,638 | short | −0.11 | 0.93 | 1,554 | −0.00 | no |
| 12:00–12:30 | 4,638 | long | −0.03 | 0.97 | 1,554 | −0.43 (flips short) | no |

Only **11:00–11:30 (weak long)** and **09:30–09:45 (weak short)** survive into 2024 — both with
MFE/MAE ≈ 0.97–1.0 (no real asymmetry) and after-cost edge < 1 pt. Every other window flips
direction train→val. The open (09:30–09:45) has the largest MFE *and* MAE (~46 pt) — violent
two-way movement, not a directional edge.

## Table 5 — Bias-edge map (is long/short bias predictive?)

| context | n_tr | dir | aftC_tr | MFE/MAE | val | predictive? |
|---|--:|--:|--:|--:|--:|:--:|
| 4h bull / above VWAP | 13,698 | long | −0.59 | 0.94 | flips/thin | **no** |
| 4h bear / below VWAP | 12,532 | long | −0.66 | 0.97 | flips short | **no** |
| premium (short bias) | 14,807 | short | −0.74 | 0.96 | flips long | **no** |
| discount (long bias) | 12,986 | long | −0.60 | 0.96 | flips short | **no** |
| 1h bull + 4h bull | 12,103 | long | −0.49 | 0.96 | long (thin) | **no** |
| gap-up + 4h-bull | 8,229 | long | +0.51 | 0.96 | +1.53 | weak momentum only |

**HTF trend, VWAP location, and premium/discount are NOT directionally predictive** — every
large bias bucket is after-cost negative with MFE/MAE < 1.0. The only mild signal is **gap-up +
4h-bull → a weak long momentum drift** (0.5–1.5 pt, but MFE/MAE 0.96 < 1, so choppy and cost-eaten).

## Table 1 — Top long-edge contexts (ranked by validation after-cost)

| context | n_tr/va | aftC_tr | aftC_val | MFE/MAE_val | honest read |
|---|--:|--:|--:|--:|---|
| prev-day = expansion (next day) | 792/288 | +5.54 | +19.46 | **1.51** | outlier-driven (huge magnitude, moderate n) → NEEDS_MORE_DATA |
| 11:00–11:30 / normal regime | 3,122/1,021 | +0.17 | +2.31 | 1.03 | large sample but MFE/MAE≈1.0 → thin |
| gap-up + 4h-bull | 8,229/3,078 | +0.51 | +1.53 | 0.96 | weak momentum, MFE/MAE<1 → thin |
| prev-day = trend_down (revert up) | 4,068/1,296 | +0.45 | +0.44 | 0.96 | thin, no asymmetry |
| 11:00–11:30 (all) | 4,638/1,554 | +0.70 | +0.42 | 0.96 | thin, no asymmetry |

## Table 2 — Top short-edge contexts

| context | n_tr/va | aftC_val | MFE/MAE_val | honest read |
|---|--:|--:|--:|---|
| gap = flat | 180/72 | +33.0 | 0.41 | **NOISE** — tiny sample, absurd magnitude, MFE≪MAE |
| 09:30–09:45 | 2,319/777 | +0.86 | 0.86 | mild, MFE/MAE<1, sub-1pt |
| 09:45–10:00 / expansion | 2,067/674 | +0.12 | 0.87 | negligible |

No short context has MFE clearly > MAE with adequate sample; the big numbers are small-sample noise.

## Table 3 — No-trade / danger contexts

- **09:30–09:45 open** — MFE≈MAE≈46 pt: violent whipsaw both ways, no directional edge.
- **10:00–10:30** — highest MAE (38.5), direction flips train→val: chop / reversal traps.
- **Expansion regime entries** — MFE/MAE 0.88–0.95, high MAE: adverse-heavy, late-chase risk.
- **Bias-only entries** (4h trend / VWAP side / premium-discount) — after-cost negative everywhere.
- **10:30–11:00, 11:30–12:00, 12:00–12:30** — direction flips or after-cost negative as standalone.

## Table 6 — Strategy candidate extraction (hypotheses only — NOT backtested)

The edge map does **not** support a strategy-grade hypothesis. The least-weak leads, logged only
as *watch / needs-more-evidence* (every one fails the "MFE clearly > MAE" + "$200-day at safe size"
bar):

1. **11:00–11:30 NY long lean.** Context: late NY-morning, normal (non-expansion) regime. Dir:
   long. Why it might exist: post-European-close drift. Sample: 3,122/1,021 (large), stable.
   Failure mode: **MFE/MAE ≈ 1.0** — no asymmetry; after-cost edge ~0.4–0.5 pt, swamped by cost;
   cannot produce $200 days without dangerous size. Estimated payout potential: negligible.
2. **Day-after-expansion long.** Context: prior day classified `expansion`. Dir: long. MFE/MAE
   1.14–1.51 (the only real asymmetry seen). Failure mode: sample 792/288 with implausibly large
   magnitude (5–19 pt after cost) → almost certainly a handful of big trend days, not a stable
   edge. Verdict: **NEEDS_MORE_DATA**, do not build.
3. **Gap-up + 4h-bull momentum long.** Large sample, but MFE/MAE 0.96 (<1) and 0.5–1.5 pt
   after-cost — a weak drift that costs erase at realistic fills. Too thin.

## Final verdict

- **Did Phase 19 find real context edge?** **No robust, tradable directional edge.** Across every
  adequately-sampled context, MFE/MAE ≈ 0.88–1.07 (≤~1.0) and directional accuracy 50–56%. The
  "MFE clearly > MAE" requirement is met nowhere with sufficient sample; the only buckets that
  exceed it are tiny-sample noise (flat gap; 09:45/normal) or implausible outliers
  (day-after-expansion).
- **Which contexts are worth turning into strategies?** **None at strategy grade.** Two weak leads
  (11:00–11:30 long; day-after-expansion long) are logged as *watch / NEEDS_MORE_DATA* only.
- **Which contexts must be avoided?** The 09:30–09:45 open whipsaw, 10:00–10:30 chop, expansion-
  regime chases, and all bias-only (HTF/VWAP/premium-discount) entries — all after-cost negative.
- **Enough evidence to proceed to Phase 20 strategy construction?** **No.**
- **Pause?** **Yes — pause the NQ-only NY-AM *directional price-action* search.** The edge map
  confirms structurally what Phases 16–18 found empirically: NQ NY-AM 5m has no exploitable
  directional edge before costs. A future phase should NOT build another directional price-action
  family on these contexts; the only justifiable next directions are (a) **MBP1 order-flow
  microstructure** on clean MBP1 periods (a different data edge, verdict-capped at
  `NEEDS_FORWARD_DATA` for the 2026-06 gap), or (b) accept that NQ-only NY-AM directional swing/
  scalp trading does not carry a payout-grade edge and stop.

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **243 tests**, incl. 4 edge-map tests).
- ✅ Lockbox 2025-2026 **never touched** (this phase is train+val behaviour analysis only).
- ✅ No strategy built, no parameters optimised, no forced edge.
- ✅ Edge map + 6 tables + honest verdict produced; the answer is **pause the directional search**.

## Reproduction

```
nqlab phase19 map --horizon 30m     # build the edge map on train 2021-23 + val 2024 (no lockbox)
```
