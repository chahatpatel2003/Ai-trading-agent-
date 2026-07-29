# Phase 57b — Non-Retail / Data-Driven GC Pattern Discovery

**Verdict: real recurring STRUCTURE found, but NO tradeable EDGE.** Every data-surfaced
pattern fails the strict train/val/test + cost gate. The apparent "edge" is the move
already having happened before entry (MFE ≈ MAE) plus long-gold beta in 2023–25. GC's
discovery does **not** beat normal GC strategies and does **not** beat NQ clean5+Phase36.

Method: PIT-clean (08:20–13:30 ET pit), 2572 sessions, **roll days excluded** (51, flagged
from raw `instrument_id`). No look-ahead. Split **train 2016–20 / val 2021–22 / test 2023–25**.
Survivor gate: expectancy **>0 in train AND val AND test**, **≥7/10 positive years**, **n≥150**,
full MGC cost ($3.48 RT). Reuses the Phase-57 cost engine and Topstep sim.

## What the data DID reveal (genuine recurring structure)
1. **Volatility expansion windows** concentrate at **08:30 ET** (macro releases — biggest, vol 3401) and **10:00 ET**, plus the 08:20 pit open. Stable, not assumed.
2. **RTH high/low formation time:** the day's extreme is set **right at the 08:20 COMEX open** far more than any other time (~282/294 sessions; 37% of extremes in the first hour vs ~15% in the last). Opening drive sets the extreme.
3. **Day-type mix is remarkably stable:** ~61% trending / 39% range, and the *trending share is 44–51% every single year* (2016→2025). Robust regime constant.
4. **Continuation rises monotonically with opening displacement:** P(day closes in first-30min direction) = 0.63 all days → **0.77 (>0.2 ATR) → 0.82 (>0.3 ATR)**. Clean, intuitive, non-retail.
5. **Refuted assumptions (data said no):** range *compression → expansion* (tight-open and wide-open both expand 0.60×ATR later — no edge); overnight-gap → RTH fade/continuation (corr +0.07 ≈ none).

## Why the structure is NOT tradeable (the mechanism)
- **MFE/MAE profile, strong-open (>0.3 ATR) entries at 09:00, n=710:** MFE median \$59 vs **MAE median \$59** (symmetric), final P(>0) = **0.50**, median \$0. The 82% "continuation" is the early move *already in the books* — the forward path from entry is a coin flip. No asymmetry to monetize.
- **Strict split kills every conditional trigger.** Eight magnitude-gated continuation variants: all show big **positive test** (+\$19–43) but **negative validation** (−\$2 to −\$13). The positive test sample is the 2023–25 gold melt-up (1810→4584), i.e. long beta — exactly what the rules reject. Survivors: **none**.
- **Beta-exposure split (the airtight check):** strong-open continuation, LONG-only = exp −\$0.6 (train −5.8 / val −4.3 / test +10), SHORT-only = train +6 / **val −18.6** / test +37. Incoherent across periods and direction → not a real edge; if it were, both sides would profit consistently.
- **Other listed structures, strict split:** sweep→stall→continuation exp −\$3.5 (neg all splits); failed-auction revert −\$5.6 (neg val/test). Rejected.

## Final answers
1. **Best non-retail GC pattern:** statistically, *opening-displacement continuation* (strong 30-min open displacement → day continues), alongside *RTH extreme forms at the 08:20 open* and *vol clusters at 08:30/10:00 ET*. These are the strongest recurring behaviors.
2. **Exact rules (of the best candidate):** at 09:00 ET, if |close−open(08:20)| > 0.3×ATR(10), trade in that direction, stop = opening range, hold to pit close (13:30 ET).
3. **Why it works statistically:** it **doesn't work as a trade.** The directional autocorrelation is descriptive (early move + multi-year drift); from the entry point MFE≈MAE and outcome is 50/50. It is path-momentum already realized, not forward-predictive.
4. **How often it appears:** strong-open >0.3 ATR ≈ **71 days/yr**; >0.4 ATR ≈ 41 days/yr (ample sample — not a tiny-sample reject).
5. **EV/month/account:** effectively **≤ \$0 robustly** — positive only in the bull-year out-of-sample slice (beta), negative in validation; fails the consistency gate.
6. **Beats normal GC strategies?** **No.** No validated pattern exceeds the Phase-57 ceiling (best stable ≈ \$20/mo, 50K gap-continuation).
7. **Beats NQ clean5+Phase36?** **No.** NQ has a genuinely positive-expectancy engine (~\$400/mo, ~79% pass); GC discovery yields no validated positive-EV pattern.

## Bottom line
The data-driven search was honest and found real GC microstructure, but the strict
val/test + cost gate (and the MFE≈MAE diagnostic) show none of it is a regime-robust,
cost-surviving edge. Consistent with the whole NQ arc (no PF≥2 intraday edge; high WR =
fake geometry; anti-bias). **Recommendation unchanged: stay on NQ; GC adds nothing.**

Code: `scripts/phase57b_{data,descriptive,discover,final}.py`.
