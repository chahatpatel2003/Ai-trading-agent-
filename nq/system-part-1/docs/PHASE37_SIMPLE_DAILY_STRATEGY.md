# Phase 37 — Simplest NQ Intraday Strategy That Trades Most RTH Days (disciplined, adversarial)

Goal: the simplest NQ/MNQ rule that trades **most RTH days** (0–3 trades/day), steady ROI, low
drawdown, Topstep-viable. Anti-bias protocol: disprove first, one trade/day max (first valid signal,
no forcing), **a-priori rules with NO tuning**, realistic costs, MNQ, train/val/test + per-year
walk-forward + Monte-Carlo trade-order reshuffle + Topstep trailing sim + **comparison to a dumb
always-long baseline**. Audited engine.

**Headline (blunt).**

```
FAILED. No simple strategy that trades most RTH days has a real, out-of-sample, Topstep-viable edge.
THE KILLER: a DUMB "buy the open, sell the close, every day" baseline made +$23,742 (1 MNQ, 6 yrs),
  PF 1.16, positive in ALL THREE splits (test 2025-26 PF 1.25) — and EVERY simple intraday strategy
  LOST TO IT. The best (ORB breakout) made +$10,127 — under half the baseline — with edge +1.7pp,
  2025 NET NEGATIVE, and test-split PF 0.965 (degrading out-of-sample).
Every frequent simple rule (ORB break/fade, VWAP revert, trend pullback, PDH/PDL fade) has edge over
  breakeven ~0 or NEGATIVE, loses to buy-and-hold-intraday, and breaches a $3K trailing account with
  ~100% Monte-Carlo probability.
The ONLY simple rule with real edge (inside-day breakout, PF 1.38, +7.5pp) trades just 10.7% of days
  (DISQUALIFIED by "trades most days") AND its test-period PF is 0.74 (degrades OOS) — curve-fit/fragile.
Verdict: every candidate = FAIL.
```

The intraday rules add **negative value**: they churn commissions/slippage around NQ's long drift and
underperform simply being long. This re-confirms ~10 phases: NQ has **no simple, frequent, after-cost
intraday edge.**

---

## Trading-day coverage
1,406 RTH days (2021-258, 2022-258, 2023-257, 2024-259, 2025-257, 2026H1-117). Splits: **train 2021-22,
validation 2023-24, test 2025-26.**

## Per-strategy results (1 MNQ, audited costs, one-trade/day, hold/RR a-priori)

| strategy | exact rule (first per day) | days% | trades/day | win% | BE win% | **edge** | PF | avg W / avg L | max DD | net $ | train/val/**test** PF | MC breach $3K | verdict |
|---|---|--:|--:|--:|--:|--:|--:|--|--:|--:|--|--:|--|
| **BASELINE_long** | long first RTH bar → EOD | 96% | 1.0 | 43.8% | 40.2% | +3.6pp | **1.16** | $287 / −$193 | $4,317 | **+23,742** | 1.10/1.14/**1.25** | 100% | (dumb beta, beats all) |
| orb_break | close breaks OR hi/lo → EOD (2×ATR stop) | 95% | 1.25 | 38.3% | 36.6% | +1.7pp | 1.074 | $229 / −$132 | $5,135 | +10,127 | 1.16/1.10/**0.97** | 100% | **FAIL** |
| pdhl_fade | first tag of PDH/PDL → fade, 2R (1.5×ATR) | 64% | 1.08 | 36.2% | 35.2% | +1.0pp | 1.043 | $161 / −$87 | $2,497 | +2,312 | **0.92**/1.10/1.12 | 87% | **FAIL** |
| trend_pullback_vwap | above VWAP, tag VWAP, hold → EOD | 86% | 1.24 | 28.0% | 27.9% | +0.1pp | 1.006 | $267 / −$104 | $6,476 | +690 | 1.08/0.92/1.01 | 100% | **FAIL** |
| vwap_revert | >1.5 ATR from VWAP → fade, 2R | 99% | 1.36 | 33.5% | 34.6% | **−1.1pp** | 0.954 | $162 / −$86 | $8,594 | −5,015 | 0.98/0.96/0.92 | 100% | **FAIL** |
| orb_fade | poke OR then close back inside → fade, 2R | 88% | 1.21 | 31.7% | 34.4% | **−2.6pp** | 0.887 | $191 / −$100 | $18,246 | −11,425 | 0.98/0.78/0.90 | 100% | **FAIL** |
| inside_day_break | prior inside day + OR break → EOD | **11%** | 1.20 | 43.3% | 35.8% | **+7.5pp** | **1.383** | $246 / −$137 | $2,039 | +5,311 | 2.32/1.66/**0.74** | 7% | **FAIL** (infrequent + OOS-degrades) |
| BASELINE_short | short first RTH bar → EOD | 97% | 1.0 | 38.3% | 38.0% | +0.3pp | 1.013 | $308 / −$189 | $8,398 | +2,131 | 1.14/0.94/0.97 | 100% | (fighting the drift) |

## The 12 required outputs (worst-to-least-bad of the real strategies)

**orb_break** (least-bad frequent strategy): rules = take the first 5m close beyond the 09:30-10:00
opening range, hold to EOD, 2×ATR disaster stop. Frequency **1.25 trades/day, 95% of days**. Win rate
**38%**, avg win $229 / avg loss −$132, **PF 1.074**. Max DD **$5,135** (1 MNQ). Daily loss behavior:
loses ~62% of days, relies on a few big trend-day EOD holds. **Topstep: FAIL** — MC breach $3K = 100%,
$4.5K = 98%. **OOS: FAIL** — train PF 1.16 → test PF **0.965** (2025 net −$2,477). **MC ruin: ~certain**
on a $3K account. **vs dumb baseline: LOSES** ($10,127 vs $23,742). **Verdict: FAIL.**

**inside_day_break** (only real edge): PF 1.383, edge +7.5pp, low DD $2,039, MC breach $3K only 7% —
the only Topstep-survivable candidate — **but trades just 11% of days (DISQUALIFIED: not "most days",
≈ the once-a-week setups excluded) and test-split PF 0.736 (2025 −$956, 2026 −$745).** This is the
Phase-30 `idb` edge, and the audit (Phase 34A) already flagged it as fragile (81% of profit from 5
trades). Out-of-sample it degrades. **Verdict: FAIL** (frequency + OOS).

The other four (orb_fade, vwap_revert, trend_pullback, pdhl_fade) have edge ≈ 0 or negative and/or
lose money; raw-trade samples confirm ordinary entries/exits, no bug — they simply have no edge.

## Raw trades (proof, not just summaries)
`orb_break` first days: 2021-01-04 short → +$337 (EOD), 01-05 long → −$88 (stop), 01-06 long → −$143
(stop), 01-07 long → +$107 (EOD), 01-08 two trades both stopped (−$135, −$86). Typical: small stops,
occasional EOD trend win — net barely positive, then costs. Full lists in `docs/phase37/_simple.json`.

## Final answer (your research target)
- **Is there a simple NQ strategy that trades most days with steady ROI + low DD + Topstep-viable?**
  **No.** Every frequent simple rule fails: edge ≈ 0/negative, loses to buy-and-hold-intraday, ~100%
  $3K breach. The only real-edge rule (inside-day breakout) trades 11% of days and degrades OOS.
- **Best by PnL:** the **dumb always-long baseline** ($23,742) — which is not a strategy (long beta)
  and is itself not $3K-viable. That it wins is the whole point: the intraday rules subtract value.
- **Curve-fit / weak / not viable?** orb_break, trend_pullback, pdhl_fade = **weak/no-edge**;
  inside_day_break = **fragile/curve-fit OOS** (test PF 0.74); orb_fade, vwap_revert = **negative**.
  All = **not Topstep-viable** (≈100% $3K breach) except the disqualified inside_day_break.
- **What actually works in NQ (from the whole program, for honesty):** only **infrequent
  mean-reversion-from-extension** (swd PF 1.53, breakout-acceptance +7.6pp) — ~1 trade/**week**, which
  **violates your "trades most days" requirement**. Best use of those is the **Phase-36 payout-first
  cycle** (~62% payout/3.4% blow on a 150K), not a daily-income strategy.

## Verdicts
| strategy | verdict |
|---|---|
| orb_break | **FAIL** — no edge OOS (test PF 0.97), loses to baseline, ~100% $3K breach |
| orb_fade | **FAIL** — negative (PF 0.89), −$11.4K |
| vwap_revert | **FAIL** — negative edge (−1.1pp), −$5.0K |
| trend_pullback_vwap | **FAIL** — edge ≈ 0 (+$690/6yr), val PF 0.92 |
| pdhl_fade | **FAIL** — marginal (+1pp), train PF 0.92, 87% $3K breach |
| inside_day_break | **FAIL** — real edge but 11% of days (disqualified) + test PF 0.74 (curve-fit/fragile) |
| **Overall research target** | **FAIL** — no simple, most-days, low-DD, Topstep-viable NQ intraday edge exists |

### Bottom line (no spin)
I tried to find the simple daily workhorse and could not — and the data says plainly why: **a brain-dead
buy-the-open/sell-the-close rule beats every clever intraday strategy I built**, because NQ's only
free money intraday is its long drift, and the rule-based systems just pay costs to capture less of
it. The frequency you require and a real edge are mutually exclusive in NQ (re-confirming Phase 31/34B).
If you want "trades most days," accept there is no edge to harvest. If you want an edge, accept ~1
trade/week (the mean-reversion-from-extension setups) and run it through the Phase-36 payout cycle.
**Do not keep searching for a simple high-frequency NQ intraday edge — ~10 phases now say it isn't there.**

```
Reproduction: python scripts/phase37_simple.py   # -> docs/phase37/phase37_simple.csv + _simple.json
```
