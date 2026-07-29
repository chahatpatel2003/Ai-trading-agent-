# Phase 18 — High-Win-Rate Frequent NY-AM Scalp Payout Search

**Hypothesis (new direction):** the Phase 16/17 large-R (3-5 ATR) approach failed — maybe Topstep
payouts need a *higher-win-rate, higher-frequency* NY-AM strategy (many controlled green days, lots
of $200+ winning days, smooth equity) instead of rare big winners.

**Status: built, run end-to-end, hypothesis FALSIFIED — all 96 candidates REJECT at validation.**
The high-win-rate premise did not hold: NQ NY-AM scalp entries have **no directional edge before
costs** and realize **28-37% win rates (not 55-70%)**, PF ~0.44. The lockbox was never touched
(correctly preserved — nothing earned the right to be tested on it). No forced acceptance.

---

## What was built

1. **Engine daily controls** (`ExecConfig` + engine): `daily_profit_target_usd`, `max_daily_trades`
   (+ existing `max_daily_losses`) — stop NEW entries after +$X / N trades / N losses today. They
   apply only on the gated real account (the ungated gauntlet still measures the raw edge).
2. **Six high-win-rate scalp families** (`phase18_families.py`), NY-AM 5m & 1m, **fixed-point**
   (or tight ATR) stops/targets, R:R 0.8-1.5: VWAP reclaim, opening-range mean-reversion, pullback
   continuation, liquidity sweep micro-reversal, range rotation, trend-day scalp.
3. **Daily risk-control matrix sweep** at the lockbox — per survivor, sweep profit-target
   {none/200/300/500} × max-loss {none/1/2} × max-trades {none/3/5/10} and keep the best (the
   "best risk setting"). [No survivor reached this stage.]
4. **Phase-18 bar + green-day stats** — win-rate floor ≥ 55%, PF ≥ 1.30, ruin ≤ 10%, green months
   ≥ 75%, payout-eligible ≥ 60%; added green-day rate / red days / worst-day / worst-week. Verdict
   ceiling capped at **`ACCEPT_FOR_FORWARD_PAPER`** (hypothesis test). `NEEDS_FORWARD_DATA` for MBP1
   families without a clean MBP1 lockbox.

## Data protocol (reserved lockbox)

Search/train **2021–2023**, validation **2024** (never loads past 2024-12-31), reserved **pristine**
lockbox **2025-2026**. Because nothing cleared validation, the lockbox was **not touched** — it
stays pristine for a future hypothesis.

## Result — all 96 candidates REJECT at validation

Search: 6 families × 4 fixed-point R:R configs × 2 risk = 48 per timeframe, on **5m and 1m**
(96 total). Every single candidate is net-negative in 2021-2024 with **zero positive years**:

| timeframe | candidates | gauntlet ACCEPT | best full_net (2021-24) | best OOS (2024) |
|---|--:|--:|--:|--:|
| 5m | 48 | **0** | −$71,488 | −$19,078 |
| 1m | 48 | **0** | −$65,569 | −$22,652 |

Representative validation-year (2024) economics — the hypothesis is falsified on its own terms:

| family | config | trades | **win rate** | PF | net | avg win | avg loss |
|---|---|--:|--:|--:|--:|--:|--:|
| sweep micro-reversal | s10/t8 | 358 | **36.6%** | 0.43 | −$19,156 | $108 | −$147 |
| VWAP reclaim | s8/t10 | 1,540 | **28.0%** | 0.45 | −$93,910 | $176 | −$153 |
| OR mean-reversion | s10/t8 | 721 | **37.3%** | 0.44 | −$37,258 | $108 | −$147 |

Why it fails (from per-candidate diagnostics):
- **`direction_edge_before_costs = False` for every candidate** — gross P&L is negative *before*
  commissions/slippage. The entries do not predict direction.
- **Symmetric MAE ≈ MFE** (e.g. 5m: avg_mae 15.98 ≈ avg_mfe 16.59) with **negative MFE-capture**
  (−0.08 to −0.19) — post-entry moves are coin-flips, and the small fixed targets/stops just
  realize that randomness; the stop is hit more often than the target.
- **Win rate is 28-37%, not the hypothesized 55-70%** — at R:R ~1 that gives PF ~0.44 (deeply
  losing). High frequency then *compounds* the cost drag ($5k-$24k over 2021-24).

So the premise — "high win rate may rescue payouts" — is empirically false here: these NQ NY-AM
scalp signals have no edge to begin with, and trading them more often loses faster.

## Verdict

```
REJECT (all 96). The high-win-rate frequent-scalp hypothesis is falsified at validation:
NQ NY-AM scalp entries have no directional edge before costs and realize ~30-37% win rates.
Lockbox 2025-2026 left pristine (nothing earned a lockbox test).
```

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **239 tests**, incl. 7 Phase-18 tests + engine daily-control
  tests).
- ✅ Lockbox respected — never touched (0 validation survivors); stays pristine.
- ✅ Every candidate carries a verdict + reason (all REJECT, with the no-edge-before-costs reason).
- ✅ No forced acceptance, no tiny-edge acceptance.
- ✅ The high-win-rate frequent-trade hypothesis was specifically tested (as required) and
  **falsified** — the report says plainly: not business-worthy.

## Honest framing

> Phase 18 specifically tested the new high-win-rate / high-frequency direction the previous
> large-R approach motivated. It built the engine daily controls, six fixed-point scalp families on
> 5m and 1m, the daily-control matrix sweep, and the green-day payout panel — then ran a genuine
> reserved-lockbox search. The result is unambiguous: the scalp entries have no directional edge
> before costs and realize 28-37% win rates, so no high-win-rate scalp clears PF 1.30. Both the
> large-R (16/17) and high-win-rate (18) hypotheses have now failed on NQ-only NY-AM data. Honest
> verdict: **REJECT all.**

## Reproduction

```
nqlab phase18 search --timeframes 5m,1m --account 150k   # train 2021-23 / val 2024 (reserves 25-26)
nqlab phase18 freeze --top-n 8                            # 0 survivors -> nothing frozen
# (no lockbox run: nothing cleared validation; 2025-2026 stays pristine)
```
