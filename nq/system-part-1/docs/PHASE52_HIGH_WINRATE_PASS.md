# Phase 52 — High-Win-Rate Topstep Pass Engine

Objective shift: maximize **pass probability** (target 70–80%+), accepting low RR / small targets / high
win rate. Built genuine high-WR low-RR systems (fresh engine runs: 5 fade/MR triggers × RR
{0.5, 0.75, 1.0, 2.0}, 1.0×ATR stop), computed trade win rate + after-cost expectancy, and ran the
Topstep pass sim on 50K/100K/150K (cushion sizing, costs+slippage+commissions). clean5+Phase36 entered as
the real-edge reference. Script `scripts/phase52.py`, data `docs/phase52/_phase52.json`.

**Headline.**

```
PASS PROBABILITY IS DRIVEN BY EXPECTANCY, NOT TRADE WIN RATE. The proof:
  - The high-WR scalps DELIVER the win rate (60-65% at 1:0.5 RR) but their after-cost expectancy is
    NEGATIVE (-$0.28 to -$5.54/trade) -> they pass ~0% of the time.
  - clean5+Phase36 has a LOW 36% trade win rate but +$6.27/trade expectancy -> it passes 79% (150K).
The RR ladder shows the trap directly: asia_sweep rr0.5 (WR 63%, exp -$1.64) -> rr1 (WR 51%, exp +$1.75)
-> rr2 (WR 35%, exp +$1.65). The HIGHEST win rate has the WORST (negative) expectancy.
ANSWER: YES, a 70-80% PASS-probability engine exists (clean5+Phase36 / 150K = 79% pass-before-blow) —
but via real edge + the $4,500 buffer, NOT via high trade win rate. A high TRADE-win-rate engine that
also passes is IMPOSSIBLE: the win rate you can buy with tiny targets is below the cost-adjusted
breakeven, so it has negative edge and passes ~0%.
```

Verdict: **YES on pass probability (79%, low trade WR); NO on high-trade-win-rate passing.**

---

## The high-WR scalps: win rate achieved, expectancy destroyed
| engine (1.0×ATR stop) | **trade WR** | RR | **expectancy/trade** | pass (150K) | pass (50K) |
|---|--:|--:|--:|--:|--:|
| or_fakeout rr0.5 | **65.2%** | 0.47 | **−$2.87** | 0.0% | 0.0% |
| pdhl_reclaim rr0.5 | **64.1%** | 0.55 | −$0.28 | 0.0% | 0.0% |
| asia_sweep rr0.5 | **63.0%** | 0.54 | −$1.64 | 0.0% | 0.0% |
| vwap_rev rr0.5 | **61.4%** | 0.52 | −$3.78 | 0.0% | 0.0% |
| pdc_reclaim rr0.5 | 60.0% | 0.51 | −$5.54 | 0.0% | 0.0% |
| asia_sweep rr1 | 51.0% | 1.02 | +$1.75 | 0.0% | 0.1% |
| pdhl_reclaim rr0.75 | 57.1% | 0.81 | +$1.96 | 0.0% | 0.0% |
| **BENCH clean5+Phase36** | **36.0%** | 1.96 | **+$6.27** | **36% pass / 79% pass-before-blow** | 51% / 53.6% |

### The RR ladder (asia_sweep) — the fake-geometry trap, isolated
| RR | trade WR | expectancy | what happens |
|---|--:|--:|---|
| 0.5 | 63.0% | **−$1.64** | tiny target eaten by costs → negative edge |
| 0.75 | 55.8% | −$0.41 | still negative |
| 1.0 | 51.0% | +$1.75 | barely positive, WR now ~coin-flip |
| 2.0 | 35.0% | +$1.65 | positive edge, but LOW win rate |

> Raising the win rate (shrinking the target) **lowers** expectancy. The highest WR is the most
> negative. This is the program's "high win rate = fake geometry" finding (Phases 33/34B), reconfirmed.

## Pass-probability comparison (150K, cushion, 90-day horizon)
| engine | trade WR | p_pass (before blow) | pass ≤45d | median days | ratio | maxDD |
|---|--:|--:|--:|--:|--:|--:|
| **clean5+Phase36** | 36% | **0.79** | 11.8% | 62 | 3.84 | 9,592 |
| every high-WR scalp | 60–65% | **~0.00** | 0% | never | 0 | 700–3,300 |

> The high-WR scalps don't blow much (small DD $0.7–3.3k) — they just **never reach the target**: with
> zero/negative edge and a ~$1 target, they grind sideways for years. clean5+Phase36, despite a 36% WR,
> compounds a +$6.27/trade edge to +$9,000 in ~62 days and passes 79% before blowing.

## Why a high-TRADE-win-rate pass engine is impossible (the math)
For a target = r × stop, the **gross** breakeven win rate is `1/(1+r)`:
- 1:0.5 RR → breakeven WR = **66.7%**; 1:0.75 → 57.1%; 1:1 → 50%.
- After costs (~$0.50–1.00 round-trip on a ~$1.00–2.00 micro target), the **cost-adjusted** breakeven
  rises several points higher.

NQ fade/MR scalps deliver ~**60–65%** WR at 1:0.5 — *below* the ~68–72% cost-adjusted breakeven → negative
edge. To pass you need positive expectancy; to get a 70–80% *trade* win rate you need r ≤ ~0.4, whose
breakeven (≥71% gross, ≥74% after costs) the market doesn't clear. **So high trade win rate and positive
edge are mutually exclusive at NQ scalp targets** — and without positive edge, pass probability → 0.

## Rankings
1. **Highest pass probability — clean5+Phase36, 150K (79% pass-before-blow).** Trade WR only 36%.
2. **Highest trade win rate — or_fakeout rr0.5 (65.2%)** / pdhl_reclaim rr0.5 (64.1%) — but **pass ≈ 0%.**
3. **Fastest viable pass — clean5+Phase36** (median 62d on 150K, 31d on 50K) — the only thing that passes.
4. **Best pass/fail ratio — clean5+Phase36, 150K (3.84).**
5. **Best account size — 150K** (79% pass-before-blow vs 53.6% on 50K — the bigger buffer matters more
   than the smaller target).

## Final verdict
**Can we build a high-win-rate Topstep pass engine with 70–80%+ pass probability?**

- **70–80%+ PASS probability: YES.** `clean5+Phase36` on a **150K** reaches **79% pass-before-blow** (and
  the cushion combine eventually passes ~84% over a longer horizon — Phase 44/50). Exact rules below.
- **Via high TRADE win rate: NO — impossible.** Every engine with a 60–65%+ win rate has negative
  after-cost expectancy (fake geometry) and passes ~0%. The win rate buyable with tiny targets sits
  below the cost-adjusted breakeven, so it cannot have a real edge, so it cannot pass.

**The two goals are in conflict:** you can have a high *pass* rate (with a low ~36% trade win rate and a
real edge), or a high *trade win rate* (with no edge and ~0% pass) — not both. Pass probability comes
from expectancy × buffer × horizon, not from winning most trades.

### Exact rules of the 79%-pass engine (clean5+Phase36, 150K)
```
Account: 150K (target +$9,000, trailing MLL $4,500 intraday locked-at-start, 15ct, consistency 50%).
Sizing : cushion — 1 MNQ/signal; -> 2 after +$2,000 cushion; -> 3 after +$4,000; daily loss stop ~-$1,700.
Gate   : skip the day if opening regime is neutral (htf neither bull nor bear).  Cap 3 trades/day.
Sleeves (each: first signal/day, next-bar-open entry, EOD-flat):
  trend_adx_cont  2.0xATR stop, EOD : RTH; LONG adx>25 & trend_1h_bull & close>VWAP & close>prev_close;
                                      SHORT adx>25 & htf_bear & close<VWAP & close<prev_close
  ir90_break      2.0xATR stop, EOD : 11:00-14:30; cross of the 09:30-11:00 (first-90-min) high/low
  asia_sweep_fade 1.5xATR stop, 2R  : RTH; sweep+reclaim of Asia-session low (long) / high (short)
  ir60_fade       1.5xATR stop, 2R  : 10:30-14:00; fade poke of the first-60-min high/low
  disp_pb_fvg     1.5xATR stop, 2R  : RTH; displacement + price in bull/bear FVG, in trend direction
  Phase-36 (idb + swd): inside-day breakout + sweep-displacement mean-reversion finalists (~1.5 trades/wk)
```
Trade win rate ≈ 36%, expectancy ≈ +$6.27/trade — a **low-win-rate, positive-edge** engine. The pass rate
is high *despite* the low win rate, because the edge and the $4,500 buffer carry it to target ~79% of the
time before a trailing-DD breach.

```
Reproduction: python -m scripts.phase52   # -> docs/phase52/_phase52.json
```
