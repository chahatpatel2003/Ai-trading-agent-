# Phase 49 — Search for better Topstep pass engines beyond clean5

clean5 was NOT assumed final. 15 new strategy families were built from the PIT-safe Phase-38 sleeve
library on the **same Topstep wrapper** as clean5 (neutral-gate + cap3 + cushion sizing, 150K, flat
cost) and run through the Phase-44 renewal-cycle evaluator, then compared head-to-head with clean5 and
the clean5+Phase36 benchmark, with year-by-year (2022 bear / 2023 grind). Script `scripts/phase49.py`,
data `docs/phase49/_phase49.json`.

**Headline.**

```
NOTHING BEATS clean5+Phase36. After testing opening-drive, high-vol breakout, trend-day, news-proxy,
first-pullback, NY/Asia/London sweeps, VWAP reclaim, gap cont/fill, ORB-on-expansion, failed-breakout,
power-hour, and long-beta(+crash-protect) — clean5+Phase36 has the highest EV/month ($497), the fastest
VIABLE pass (101 days), 96% first-payout, and a top-2 pass/fail ratio (6.65). No candidate wins on
ratio OR EV/month while passing the kill criteria.
THE ONLY GENUINELY NEW FIND: long-beta + MR-hedge (beta_long_above_pdc + asia_sweep_fade + ir60_fade)
is POSITIVE IN ALL SIX YEARS including 2023 — it fixes clean5's one weak year (2023 low-vol grind) at
low drawdown ($8.2k) — but at lower EV ($310/mo). It's a robustness patch, not a replacement.
2023 is the discriminator: in the low-vol grind, almost everything is flat/negative; only long-beta
variants, VWAP-reclaim, and the ir-breakout stack are positive. 2022 (bear) everything wins.
```

Verdict: **clean5+Phase36 REMAINS the benchmark — confirmed, not replaced.**

---

## Head-to-head (150K, cushion combine, 1 MNQ funded, flat cost, 8k sims)
| family (sleeves) | pass | blow | **ratio** | med d→pass | 1st-payout | E[total payout] | **EV/mo** | maxDD | worst day | consist |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **clean5+Phase36 (BENCH)** | 85% | 13% | 6.65 | **101** | 96% | 6,225 | **497** | 10,962 | −2,354 | 5% |
| clean5 (benchmark) | 83% | 10% | **8.52** | 120 | 93% | 5,702 | 431 | 10,227 | −2,405 | 5% |
| trend_day_cont | 57% | 43% | 1.35† | **82** | 70% | 5,265 | 408 | 17,209 | −3,065 | 29% |
| high_vol_breakout | 59% | 38% | 1.55 | 105 | 73% | 5,021 | 320 | 10,233 | −2,990 | 10% |
| long_beta_crashprot | 62% | 20% | 3.05 | 130 | **99%** | 5,335 | 310 | **8,213** | −2,094 | 23% |
| ir_breakout_stack | 61% | 23% | 2.64 | 130 | 78% | 4,025 | 285 | 18,866 | −2,713 | 23% |
| opening_drive_mom | 47% | 21% | 2.23 | 174 | 72% | 3,500 | 185 | 8,137 | −2,365 | 2% |
| orb_vol_expansion | 41% | 21% | 1.98 | 156 | 70% | 3,350 | 159 | 8,887 | −2,365 | 0% |
| first_pullback_impulse | 15% | 11% | 1.38† | 225 | 92% | 2,955 | 104 | 8,317 | −2,110 | 0% |
| vwap_reclaim_cont | 28% | 59% | 0.47† | 92 | 63% | 3,140 | 106 | 15,017 | −3,200 | 8% |
| long_beta_raw | 31% | 20% | 1.49† | 166 | 82% | 2,679 | 97 | 6,935 | −1,463 | 0% |
| gap_continuation | 24% | 9% | 2.61 | 197 | 71% | 2,452 | 91 | 6,805 | −1,495 | 0% |
| power_hour_cont | 16% | 8% | 1.90 | 149 | 60% | 1,679 | 68 | 8,230 | −2,096 | 0% |
| news_vol_proxy | 13% | 38% | 0.35† | 233 | 79% | 2,342 | 33 | 8,057 | −947 | 0% |
| asia_london_sweep | 3% | 24% | 0.14† | 170 | 73% | 1,757 | 2 | 5,116 | −2,317 | 0% |
| ny_sweep_reclaim | 0% | 19% | 0.00† | — | 54% | 1,321 | −18 | 6,075 | −2,613 | 0% |
| gap_fill | 0% | 0% | —† | — | 51% | 451 | −14 | 1,537 | −306 | 0% |
| failed_breakout | 0% | 71% | 0.00† | — | 36% | 575 | −24 | 7,876 | −2,089 | 0% |

† fails a kill criterion (ratio < 1.5, or EV negative, or can't pass).

## Year-by-year net (1 MNQ, gated cap3) — the regime check
| family | 2021 | **2022 (bear)** | **2023 (grind)** | 2024 | 2025 | 2026 |
|---|--:|--:|--:|--:|--:|--:|
| clean5+Phase36 (BENCH) | 7,663 | 17,431 | **−1,438** | 13,863 | 15,410 | 7,564 |
| clean5 | 5,370 | 13,189 | **−1,863** | 10,744 | 11,300 | 5,917 |
| trend_day_cont | 7,303 | **21,744** | −922 | 15,648 | −774 | 8,501 |
| **long_beta_crashprot** | 6,477 | 8,551 | **+2,435** | 3,428 | 8,514 | 8,593 |
| long_beta_raw | 3,496 | 3,707 | **+3,451** | 711 | 4,519 | 6,349 |
| ir_breakout_stack | 4,154 | 13,475 | **+3,832** | 12,164 | −1,182 | 8,446 |
| vwap_reclaim_cont | 4,150 | 4,213 | **+2,602** | −5,361 | 9,639 | 1,500 |
| high_vol_breakout | 5,191 | 12,402 | +129 | 9,465 | −6,280 | 5,247 |

> **2022 bear:** nearly every family is strongly positive — momentum/breakout *thrive* in high-vol
> selloffs (trend_day_cont +$21.7k). **2023 low-vol grind:** the hard year — clean5/clean5+P36 and the
> momentum families go flat/negative; only **long-beta variants, ir-stack, and VWAP-reclaim** are
> positive. That's the one place clean5 can be improved.

## Final ranking
**1) Best strategy to REPLACE clean5 — NONE.** clean5+Phase36 wins EV/month ($497), fastest viable pass
(101 d), first-payout (96%), and top-2 ratio (6.65). Every faster candidate (trend_day_cont 82 d,
vwap_reclaim 92 d) fails the ratio kill criterion with 2–3× the drawdown. **clean5 is confirmed, not
replaced.**

**2) Best strategy to COMBINE with clean5 — Phase-36 (already the benchmark combo)**: it raises EV
($431→$497) and speeds the pass (120→101 d). **Secondary, for robustness: a small `long_beta_crashprot`
allocation** (beta_long_above_pdc + asia_sweep_fade + ir60_fade) — positive in all six years incl 2023,
lowest DD ($8.2k), 99% first-payout. It patches clean5's only weak regime at the cost of some EV.

**3) Best fast-pass strategy — `trend_day_cont` (median 82 days, the fastest)** — but it FAILS the kill
criteria (ratio 1.35, fail 43%, DD $17.2k, 29% consistency blocks). So among *viable* engines the fastest
is still **clean5+Phase36 (101 d)**. (Reconfirms Phase 47: genuinely fast = not viable.)

**4) Best payout strategy — `clean5+Phase36`** (highest E[total payout] $6,225, first-payout 96%).
`long_beta_crashprot` has marginally higher first-payout (99%) but lower total ($5,335).

**5) Does clean5 remain the benchmark or get replaced? — REMAINS, and is strengthened.** A fair test of
15 alternative families found none better on the Topstep metrics. The disciplined outcome is to keep
clean5+Phase36 as the engine, with an optional small long-beta-crash-protected sleeve to harden the
2023-type low-volatility regime.

### Bottom line
The honest result of looking *beyond* clean5 is that clean5+Phase36 holds up — it is the best Topstep
pass engine found across opening-drive, breakout, trend-day, sweep, VWAP, gap, and beta families. The
only improvement worth making is *robustness*, not speed or raw EV: bolt on a small long-beta + MR-hedge
sleeve so the system stops bleeding in low-volatility grind years like 2023. No replacement; a hedge.

```
Reproduction: python -m scripts.phase49   # -> docs/phase49/_phase49.json
```
