# Phase 22 — Exact Cost Audit + Targeted 60-70% Win-Rate / 1:1 Topstep Search

Two jobs: (1) audit the exact execution cost model used in every prior phase, and (2) test the
specific claim — **"can we find a 70% win-rate, 1:1, $200-risk, Topstep-safe NQ strategy?"**

**Answer up front: NO.** The best selective 1-trade/day 1:1 setup wins **~38%** (not 60-70%), and
the cost audit proves costs are **not** the cause — at *zero cost* the same setups win ~38%, so the
entries have no directional edge. The cost model is reasonable and if anything mildly optimistic.
0 of 48 candidates cleared validation; the lockbox was never touched. No forced acceptance.

---

## Part 1 — Exact cost model (audited from code)

| item | value |
|---|---|
| commission / side (NQ / MNQ) | **$0.85 / $0.25** |
| round-trip commission (NQ / MNQ) | **$1.70 / $0.50** |
| exchange / NFA / platform fees | folded into `commission_per_side` (not modeled separately) |
| spread assumption | folded into the 1-tick slippage |
| slippage | **1.0 tick = 0.25 pt per fill** (`slippage_ticks`) |
| entry fill | always slipped 0.25 pt adverse |
| exit fill | STOP / EOD / time-stop / halt slipped 0.25 pt; **TARGET = exact limit (no slip)** |
| **cost/trade WIN (NQ)** | **$6.70** = 0.25pt slip ($5) + $1.70 comm (~1.34 ticks) |
| **cost/trade LOSS (NQ)** | **$11.70** = 0.50pt slip ($10) + $1.70 comm (~2.34 ticks) |
| **cost/trade WIN / LOSS (MNQ)** | **$1.00 / $1.50** |
| deducted | round trip (entry fees at open + exit fees at close); slippage embedded in fills |
| varies by timeframe? | no per-trade (lower TF just takes more trades → more total cost) |
| varies by NQ vs MNQ? | yes ($ differ; MNQ costs *more in points*, less in dollars) |
| varies by volatility/liquidity? | **NO — flat 1 tick** (a simplification) |
| config location | `config/instruments/nq.yaml` (comm/tick/pv), `config/backtest/default.yaml` (`slippage_ticks=1.0`), `src/nqlab/backtest/engine.py` (application) |
| prior-phase assumption | identical model in **all** phases; gauntlet cost-stress ran 1×/2×/3× slippage × 1×/1.5× commission |

**Cost sensitivity** (representative 1:1 VWAP-reclaim, validation 2024):

| cost × | trades | win % | PF | exp $/trade | net $ |
|---|--:|--:|--:|--:|--:|
| **0×** | 163 | **31.9%** | 0.47 | −$72 | −$11,800 |
| 0.5× | 163 | 31.9% | 0.45 | −$77 | −$12,485 |
| 1× | 163 | 31.3% | 0.42 | −$83 | −$13,575 |
| 2× | 163 | 30.7% | 0.38 | −$94 | −$15,360 |

**Assessment:** Win rate and PF barely move from 0× to 2× (31.9% → 30.7%). **Costs are NOT the
problem — the edge is absent before costs.** A 1:1 system needs ~50% wins just to break even and
>57% to clear PF 1.30; these setups win ~32% even with zero cost (the 10-pt stop is hit far more
than the 10-pt target). The model is **reasonable and mildly optimistic** (exact target fills, a
flat 1-tick slippage that under-models fast-tape slippage), so real-world results would be **no
better** than reported. Prior phases were not rejected because costs were too harsh.

## Part 2 — Targeted 1:1 search (1 trade/day, $200 risk)

Six selective one-trade-per-day families (A first-clean-bias, B continuation, C sweep-reversal,
D OR-failed-breakout, E VWAP-reclaim, F MBP1-confirmed), fixed 1:1 in {10/10, 8/8, 12/12, 15/15}
points, first-of-day signal gate, NY-AM, on 5m + 15m. Train 2021-23 / validation 2024.

**Result: 0 of 48 cleared validation (all REJECT).** Per-family 1:1 win rate (10/10, val 2024):

| family (A–E) | trades | win % (0× cost) | win % (1× cost) | PF (1×) |
|---|--:|--:|--:|--:|
| A first-clean-bias | 242 | 38% | 38% | 0.57 |
| B continuation scalp | 171 | 26% | 26% | 0.32 |
| C sweep reversal | 116 | 34% | 33% | 0.45 |
| D OR failed-breakout | 216 | 32% | 31% | 0.41 |
| E VWAP reclaim | 163 | 32% | 31% | 0.42 |

- **No 1:1 family exceeds 38% win rate** — the 60% floor (let alone 70%) is not remotely reached.
- **Zero-cost win rate = 1×-cost win rate** for every family → the shortfall is a missing
  directional edge, not transaction costs. At 1:1, NQ NY-AM entries are hit on the stop side ~62%
  of the time.
- (F MBP1-confirmed is MBP1-required → its verdict would be capped at `NEEDS_FORWARD_DATA`; it also
  did not clear validation.)

Because nothing cleared validation, the streak/Topstep panel is moot (no finalist), and the
**2025-2026 lockbox was never touched.**

## Verdict

```
REJECT (all 48). No 60%+ win-rate 1:1 NQ NY-AM setup exists; the best is ~38%.
2025-2026 lockbox left untouched.
```

## Direct answer to the phase question

> **"Can we find a 70% win-rate, 1:1, $200-risk, Topstep-safe NQ strategy?"**
>
> **No.** Across six selective one-trade-per-day setups and four 1:1 target sizes, the best
> after-cost win rate is **~38%**, and at **zero cost** it is still ~38% — so this is not a cost
> problem, it is a no-edge problem: NQ NY-AM intraday entries do not reach the symmetric 1:1 target
> more often than the stop (they reach it *less* often, ~32-38%). A 1:1 system needs ~57%+ wins to
> clear PF 1.30; nothing comes close. The cost audit additionally confirms prior-phase rejections
> were **not** caused by overly conservative costs — the model is mildly optimistic.

This is consistent with Phase 19's edge map (MFE ≈ MAE, ~52% directional accuracy) and Phases
16–21: **there is no payout-grade directional edge in NQ-only NY-AM bars or order flow.** The
overall program conclusion stands: *"NQ-only NY-AM automated Topstep payout strategy was not found
with available data."*

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **263 tests**, incl. Phase-22 + streak-stat tests).
- ✅ Exact cost model printed (`nqlab phase22 cost-audit`) with 0×/0.5×/1×/2× sensitivity.
- ✅ Lockbox 2025-2026 **never touched** (0 validation survivors).
- ✅ Every candidate has a verdict + reason; 60% win-rate floor enforced; no forced acceptance.
- ✅ The phase question is answered directly: **no 70% (or 60%) win-rate 1:1 NQ strategy found.**

## Reproduction

```
nqlab phase22 cost-audit                                # exact cost model + 0x/0.5x/1x/2x sensitivity
nqlab phase22 search --timeframes 5m,15m --account 150k # 1-trade/day 1:1 families, train/val
nqlab phase22 freeze --top-n 8                          # 0 survivors -> nothing frozen
```
