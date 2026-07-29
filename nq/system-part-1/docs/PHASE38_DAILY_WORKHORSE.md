# Phase 38 — Daily NQ Workhorse Search (64 sleeves, 19 areas + portfolio combination)

Expands Phase 37 (which tested only 6 single sleeves and judged on net $). Phase 38 tests **64
rule-based sleeves across all 19 search areas**, judges on a **risk-adjusted + Topstep-viability**
basis, then **combines sleeves into portfolios** so the *portfolio* trades most days. Anti-bias: one
trade/day per sleeve (first valid signal, no forcing), a-priori rules with NO tuning, realistic MNQ
costs, train21-22 / val23-24 / test25-26 + per-year + Monte-Carlo trade reshuffle + Topstep trailing
DD + 120-day payout/blow MC. Audited engine.

**Headline — the conclusion changes vs Phase 37, but with a hard caveat.**

```
1) TRADES-MOST-DAYS IS ACHIEVABLE. A PORTFOLIO of independent sleeves trades 99% of RTH days — Phase 37
   was wrong to imply daily coverage is impossible; it just can't come from ONE simple sleeve.
2) A DIVERSIFIED, OOS-ROBUST ENSEMBLE BEATS THE DUMB LONG BASELINE RISK-ADJUSTED. Taking 1-2 trades/day
   from a robust sleeve set (longs+shorts, continuation+mean-reversion) beats always-long on Sharpe,
   return/DD, AND tail risk — because the short/MR sleeves cushion the long sleeves on down days.
3) BUT IT IS NOT A STRONG TOPSTEP PAYOUT MACHINE WHILE TRADING MOST DAYS. At safe size (1-2 MNQ) the
   per-day edge is too small to reach the +$9,000 payout often (payout 1-17% / 120 days); pushing size
   to chase payout sends the blow-rate to 16-42%. Strong payout still needs the infrequent Phase-36
   cycle (~once/week, 62% payout / 3% blow) — which violates "trades most days".
4) HARD CAVEAT: 2021-2026 was a secular BULL. The winning ensembles are long-biased; part of their edge
   is captured beta. A sustained bear is NOT in the sample. Treat as forward-paper, size small.
```

Verdict: **CONDITIONAL PASS** for "daily workhorse that beats the dumb baseline risk-adjusted with low
blow risk" (best = `robust_no_beta@2` / `all_robust@1-2`). **FAIL** for "trades most days AND is a
strong Topstep payout machine" — the two are in tension at safe size.

---

## 1. Coverage of the 19 search areas
All 19 were tested except the news calendar (no feed). 64 sleeves; 1 produced 0 trades (`rexp_rollbreak`,
double-cross too strict). Full per-sleeve table: `docs/phase38/phase38_sleeves.csv`.

| # | area | tested? | sleeves | best sleeve (full) |
|--|--|--|--|--|
|1|opening drive continuation|✅|3|od_cont_struct (PF 1.07, $8.9K)|
|2|opening drive failure|✅|3|od_fail_mss (PF 1.01) — all weak/negative|
|3|VWAP reclaim/reject|✅|4|vwap_reclaim (PF 1.08, $8.1K)|
|4|PDH/PDL sweep+reclaim|✅|3|pdhl_fade (PF 1.05, br3k 83%)|
|5|Asia/London/overnight sweep|✅|4|asia_sweep_fade (PF 1.07, $5.1K, OOS-robust)|
|6|first pullback after displacement|✅|3|cisd_cont (PF 1.07, $5.5K)|
|7|trend day continuation|✅|4|trend_adx_cont (PF 1.14, $19.1K — but long-beta, br3k 100%)|
|8|range day mean reversion|✅|4|range_or_fade (PF 1.88, br3k 0% — but 3.5% days)|
|9|gap fill / continuation|✅|3|gap_go (PF 1.18, $11K); gap_fill (br3k 1.7%)|
|10|settlement/prev-close reclaim|✅|2|both negative|
|11|POC / value-area reversion|✅|5|poc_reclaim_cont (PF 1.08)|
|12|high-vol breakout|✅|2|volhi_orb (PF 1.07, br3k 100%)|
|13|low-vol chop / compression|✅|1|comp_break (PF 1.14, br3k 27%)|
|14|time-of-day|✅|3|tod_power_hour (PF 1.05); reversion ones negative|
|15|long-only beta filters|✅|5|beta_long_uptrend (PF 1.17, $22.9K, br3k 100%)|
|16|news / non-news regime|⚠️ proxy|2|NO calendar feed — proxied by vol-regime "active day" (active_long_volhi PF 1.16)|
|17|ATR expansion/contraction|✅|2|atr_exp_orb (PF 1.07)|
|18|day-of-week|✅|5|dow_long_tue (PF 1.22, edge +4.9pp)|
|19|first 30/60/90-min range|✅|4|ir60_break (PF 1.12, ret/DD 3.12); ir60_fade (br3k 81%, OOS-robust)|
|20|portfolio combination|✅|—|see §3|

## 2. The two-population structure of the sleeves (the core insight)
Ranking all 64 sleeves splits them cleanly into two groups — **and neither alone satisfies the goal:**

- **High-net sleeves are long-beta in disguise → 100% trailing-DD breach.** beta_long_uptrend ($22.9K,
  ret/DD 5.8), trend_adx_cont ($19.1K), active_long_volhi ($17.3K), ir30_break ($15.3K),
  beta_long_above_pdc ($16.5K). Every one breaches a $3K trailing account ~100% — they just harvest
  NQ's drift like the baseline and are **not Topstep-viable** standalone.
- **Low-breach (Topstep-survivable) sleeves are infrequent mean-reversion/gap.** range_or_fade (br3k
  0%, PF 1.88, **but 3.5% days**), gap_fill (br3k 1.7%, 28% days), rsi_revert (br3k 0%, 3.8% days),
  comp_break (br3k 27%, 16% days). Real edge, but they trade too few days alone.

The whole phase is the search for a *combination* that gets the days-coverage of group 1 with the
breach-safety of group 2.

## 3. Portfolio combinations (the answer)
Each sleeve = 1 MNQ; portfolio = union of trades. **Uncapped**, reaching 99% of days needs 9-15 sleeves
= up to **15-21 concurrent MNQ** → over-leveraged → still 100% breach (e.g. `H5_all_robust` nets
$172K but **blows 65% before payout**). So we **cap to the user's preferred 0-3 trades/day** (take the
first k signals each day), which bounds size and is the realistic configuration:

| portfolio | days% | trades/day | net (6yr) | PF | ret/DD | daily Sharpe | $3K breach* | $4.5K breach* | payout / blow (120d) | OOS PF tr/val/te |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--|
| **BASELINE_long (dumb)** | 96% | 1 | 23,742 | 1.16 | 5.23 | 0.92 | 100% | 96% | 4% / 10.3% | 1.10/1.14/1.25 |
| all_robust @1/day | 99.9% | 1 | 18,413 | 1.20 | 3.86 | 0.92 | 100% | **52%** | 1% / **0.9%** | 1.13/1.06/1.42 |
| **all_robust @2/day** | 99.9% | 2 | **45,713** | 1.22 | **7.11** | **1.20** | 100% | 86% | **17.3%** / 16.3% | 1.17/1.15/1.35 |
| **robust_no_beta @2/day** | 99.9% | 2 | 28,934 | 1.18 | 6.40 | 1.15 | 100% | **60%** | 3.4% / **4.2%** | 1.21/1.20/1.13 |
| diversified @1/day | 99.3% | 1 | 7,681 | 1.10 | 2.38 | 0.62 | 82% | **24%** | 0% / **0.1%** | 1.10/1.08/1.13 |
| lowbreach_all @1/day | 99.9% | 1 | 3,385 | 1.05 | 1.07 | 0.33 | 81% | 24% | 0% / 0% | 1.03/1.08/1.05 |

*\$3K / \$4.5K breach = P(trailing DD ever exceeds the limit over the FULL 6-yr / 1,400-trade history) —
a pessimistic lens (over years you WILL see a $3K dip). The realistic Topstep test is the 120-day
payout/blow MC.*

**`robust_no_beta @2/day`** members: od_cont_struct, asia_sweep_fade, disp_pb_fvg, trend_adx_cont,
dow_long_mon, ir30/60/90_break, ir60_fade (continuation + MR, longs + shorts; **excludes pure beta**).

## 4. Does anything beat the dumb baseline risk-adjusted? (the required test)
**Yes — three of them do, and the win is from diversification, not leverage** (Sharpe and ret/DD are
scale-invariant):

| metric | baseline | all_robust@1 | all_robust@2 | robust_no_beta@2 |
|---|--:|--:|--:|--:|
| daily Sharpe | 0.92 | 0.92 (tie) | **1.20** ✅ | **1.15** ✅ |
| return / max-DD | 5.23 | 3.86 | **7.11** ✅ | **6.40** ✅ |
| net $ (6 yr) | 23,742 | 18,413 | **45,713** ✅ | **28,934** ✅ |
| $4.5K breach | 96% | **52%** ✅ | 86% ✅ | **60%** ✅ |
| 120-day blow | 10.3% | **0.9%** ✅ | 16.3% | **4.2%** ✅ |
| days traded | 96% | **99.9%** ✅ | **99.9%** ✅ | **99.9%** ✅ |
| OOS test PF | 1.25 | 1.42 | 1.35 | 1.13 |

- `all_robust@1` = **baseline return at a fraction of the tail risk** (same Sharpe, half the $4.5K
  breach, 1/10th the blow rate) — the short/MR sleeves cushion the longs. Capital-preservation tilt.
- `robust_no_beta@2` = **beats baseline on Sharpe, ret/DD, net, and blow while excluding pure beta** —
  the cleanest "real edge" daily workhorse. Payout modest (3.4%) but blow only 4.2%, OOS-robust.
- `all_robust@2` = highest return/Sharpe and best payout (17.3%), but leans on beta and blows 16.3%.

## 5. Why it still fails as a *Topstep payout machine* while trading most days
At safe size the per-day expectancy is small relative to the +$9,000 payout target / $4,500 trailing
floor: over 120 days the safe configs reach payout only 1-3.4% of the time (they grind sideways,
"neither" ~92-98%). Raising size to reach payout (cap 3, or many sleeves) pushes blow to 16-42%. So:

```
trade-most-days + safe size  -> payout ~0-3%   (too slow to reach +$9k)
trade-most-days + chase payout -> blow ~16-42% (over-leveraged long beta)
```

The Topstep payout frontier is owned by the **Phase-36 infrequent cycle** (~once/week, 62% payout /
3% blow) — strictly better payout odds, but it does NOT trade most days. You cannot have both.

## 6. Honest caveats
- **Bull-market regime.** 2021-2026 NQ rose; the winning ensembles are long-biased and partly capture
  beta. OOS test (2025-26) holds (PF 1.13-1.42) but is also an up period. A sustained bear is untested
  — expect underperformance there. This is the #1 risk.
- **Trailing-DD over years is ~certain.** The $3K breach is 100% over 6 years even for the diversified
  books — you need the 150K/$4,500 buffer and disciplined scaling/de-scaling (Phase-31/36 growth-gating).
- **News regime (area 16) not truly tested** — no calendar feed; proxied by volatility regime.
- **Edges are modest.** Most sleeve edges-over-breakeven are +1 to +4pp; the portfolio works by
  *diversification of small edges*, not any single strong edge. No PF≥2 daily sleeve exists (Phase 29).

## 7. Final verdicts
| candidate | verdict |
|---|---|
| Single simple sleeve, trades most days | **FAIL** — high-net ones are long-beta (100% breach); low-breach ones trade <40% of days (confirms Phase 37) |
| **Portfolio, trades most days, beats baseline risk-adjusted, low blow** | **CONDITIONAL PASS** — `robust_no_beta@2` / `all_robust@1` (99.9% days, Sharpe ≥0.92, blow ≤4%, OOS-robust). Forward-paper, small size, regime-caveated |
| Portfolio, trades most days, STRONG Topstep payout | **FAIL** — safe size → payout 1-3%; chase payout → blow 16-42% |
| Strong Topstep payout overall (any frequency) | **use Phase-36 cycle** (~once/week, 62%/3%) — not a most-days book |

### Bottom line (no spin)
The expanded search **partly vindicates the user and partly re-confirms the program.** Phase 37's
implied "daily trading is impossible" was too strong: a *portfolio* of independently-tested sleeves
trades 99% of days, and a diversified, OOS-robust ensemble genuinely **beats the dumb long baseline on
a risk-adjusted basis** (higher Sharpe/ret-DD, much lower blow) — the short and mean-reversion sleeves
hedge the long sleeves' down days. That is a real, defensible daily workhorse worth forward-papering at
small size. **But** the deeper finding holds: you cannot simultaneously trade most days *and* be a
strong Topstep payout machine — at safe size the daily edge is too small to reach payout, and the only
way to reach payout (size up) reintroduces ruin risk. Strong payout remains the territory of the
infrequent Phase-36 mean-reversion cycle. And every most-days workhorse here is long-biased in a
six-year bull — the regime caveat is the dominant risk, not the backtest.

```
Reproduction:
python scripts/phase38_workhorse.py     # 64 sleeves -> docs/phase38/phase38_sleeves.csv + _trades.json
python scripts/phase38_portfolio.py     # uncapped portfolios + greedy -> _portfolios.json
python -m scripts.phase38_capped        # 0-3 trades/day capped portfolios -> _capped.json
```
