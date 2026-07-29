# Phase 12 — trend_day Frequency Expansion

**Goal:** take the one real lead — Phase 11's `trend_day`, which had the cleanest
validation result in the project but **failed only because it traded too rarely to
confirm** — and search for *related* NQ-only strategies that keep the same edge logic
but produce enough trades to judge on held-out data. Not random discovery; a focused
expansion of one lead.

**Verdict: the frequency expansion succeeded — and the confirmation is negative.**
0 ACCEPT_FOR_FORWARD_PAPER, 3 REJECT, 3 NEEDS_MORE_DATA. The expansion did what it was
asked: `trend_open_drive` on 5m now trades **hundreds** of times on the held-out
window (vs the base `trend_day`'s ~0). With enough trades to *judge*, the best variant
**passes every economic gate on the untouched 2025-2026 final test** (positive, ruin
1%, cost-stress survived, walk-forward 67%, 100% positive years) but is **statistically
insignificant** (profit factor 1.09, DSR 0.16) → REJECT. The trend_day edge, even
frequency-expanded, is real-but-marginal, not confirmable.

> **Verdict ceiling (honesty):** the 2025-2026 window was already inspected for the
> trend_day family in Phase 11, so it is **not pristine**. Phase 12's strongest possible
> verdict is `ACCEPT_FOR_FORWARD_PAPER` (confirm on a forward feed), never
> `ACCEPT_FOR_PAPER`. Nothing reached even that bar.

NQ only; no ES; the frozen Phase-11 candidate was **not** touched or optimized.

---

## Protocol (reserved split)

| Window | Dates | Use |
|---|---|---|
| Search / train | 2021-01-01 → 2023-12-31 | fit/score |
| Validation | 2024 | OOS gate for freezing |
| Final test | 2025-01-01 → 2026-06-30 | one shot (not pristine → forward-paper ceiling) |

Search loads only through 2024-12-31. Five related families, all keeping the identical
trend-day gate (HTF-4h bull, expanding vol, right side of VWAP) but changing the entry
trigger to add trades: **trend_pullback** (re-enter each VWAP pullback), **trend_open_drive**
(retest of the opening range that holds), **trend_range_expansion** (compression → NY
expansion), **trend_late_confirm** (second leg after the prior-day extreme breaks),
**trend_no_fade** (continuation, explicitly excluding range/compression days). 480
candidates (5m+15m × 50/100/150K × {MNQ-only, adaptive} × risk {$50-$200}).

---

## The expansion worked: more confirmable survivors

| | Phase 11 (base trend_day) | Phase 12 (expansion) |
|---|---|---|
| Validation gauntlet-survivors | 30 | **70** |
| Best held-out trade count (5m) | ~0 (could not judge) | **311** (judgeable) |

`trend_open_drive` dominated validation (15m, +$2,162 OOS-2024, validation DSR 0.70,
100% positive years) and — crucially — its **5m** form trades 200-300×/year, finally
giving the lead enough held-out trades to confirm or reject. (The high-DSR **15m**
variants, like Phase 11's base trend_day, still collapse on 2025-2026 — at $100-$150
risk a 1.5×ATR stop sizes to ~0 MNQ contracts in the higher-ATR 2026 regime — so they
land at NEEDS_MORE_DATA. The freeze used a **validation-frequency floor (≥100 trades)**
to prioritize confirmable candidates.)

## Final test (untouched 2025-2026) — full practical stats

| Frozen variant | Trades | Win% | PF | Expectancy | Net | Monthly | Max DD | Ruin | Cost | WF+ | Pos-yr | P(pass 50K) | DSR(n=6) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|---|
| **trend_open_drive 5m r$150 (150K)** | 311 | 38% | **1.09** | $6.7 | +$2,083 | $119 | $1,929 | 1% | **pass** | 67% | 100% | 38% | **0.16** | **REJECT** |
| trend_open_drive 5m r$100 | 146 | 38% | 1.03 | $1.6 | +$233 | $13 | $1,303 | 0% | **FAIL** | — | 50% | 0% | 0.09 | REJECT |
| trend_open_drive 15m r$150 | 34 | 26% | 0.75 | −$20 | −$696 | −$40 | $1,084 | 0% | fail | 0% | 0% | 0% | 0.12 | NEEDS_MORE_DATA |
| trend_pullback 15m | 18 | 33% | 1.16 | $9.6 | — | — | — | — | — | — | — | — | — | NEEDS_MORE_DATA |

(Trade counts here are over the full 2025-2026 window; the gauntlet's OOS-gate split
saw 75 / 25 / 0 trades respectively, which is what the verdict uses.)

**Why each failed:**

- **trend_open_drive 5m r$150 (the best, and judgeable):** on the untouched final test
  it is **positive (+$2,083 over 18 months), ruin 1%, not blown, survives cost stress,
  walk-forward 67%, profitable in 100% of years** — it clears every *economic* gate.
  But **profit factor is only 1.09, win rate 38%, and DSR(n=6) = 0.16 ≪ 0.90** — the
  edge is real but far too small/noisy to be statistically significant even at the
  frozen trial count. **REJECT.** (P(pass 50K) ≈ 38%, P(pass 100K) ≈ 0.5%, P(pass 150K)
  ≈ 0 — far short of a reliable Topstep pass.)
- **trend_open_drive 5m r$100:** profit factor 1.03 — the edge is below transaction
  costs; **fails cost/slippage stress** and is profitable in only 1/2 years. **REJECT.**
- **15m variants:** still trade too rarely / turn negative on 2025-2026 (win 26%, PF
  0.75) → **NEEDS_MORE_DATA**.

**Best account / instrument:** among the tradeable variants, 5m `trend_open_drive` at
150K, MNQ-only sizing, gave the best Topstep pass odds (38% at 50K) — but it is still a
REJECT.

---

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **197 tests**, incl. Phase-12 family
  determinism, the `ACCEPT_FOR_FORWARD_PAPER` cap, and the full practical-stats panel).
- ✅ Every candidate received `ACCEPT_FOR_FORWARD_PAPER` / `REJECT` / `NEEDS_MORE_DATA`.
- ✅ No candidate promoted: none clears DSR on the final test, and the rest fail cost
  stress or trade too rarely. The forward-paper ceiling is correctly enforced (final
  test not pristine). Full practical stats reported for every variant. Nothing forced.

## Honest answer

> **Did frequency expansion turn trend_day into a confirmable strategy?** It produced
> variants that trade enough to *be judged* — which Phase 11 could not achieve — and the
> judgment is negative. The best frequency-expanded variant (`trend_open_drive` 5m)
> survives ruin, costs, Topstep, walk-forward, and multi-year consistency on the
> untouched 2025-2026 data, but its edge is a profit factor of ~1.09 that does not clear
> statistical significance (DSR 0.16). That is the most precise finding the project has
> produced: the trend_day edge is **real but sub-significant** — not a confirmation
> failure for lack of data, but a genuine "the edge is too small."

This closes the trend_day lead at the strategy-search level: more entries do not change
that the per-trade edge is ~at the cost line. The only remaining honest avenue is a
**forward paper feed** (out of scope — no live infrastructure), which would either grow
the sample enough to push a true >1.09-PF edge over the DSR bar, or confirm the
rejection. Per the standing rule — *no edge until proven; reject honestly, do not force
an accept* — Phase 12 ends with **zero strategies promoted**.

---

## Reproduction

```
nqlab phase12 search  --no-build                  # train 2021-23 / validation 2024
nqlab phase12 freeze  --min-trades 100            # frequency floor -> confirmable survivors
nqlab phase12 lockbox --no-build                  # untouched 2025-2026 final test
nqlab report phase12                              # registry + verdicts + full practical stats
```
