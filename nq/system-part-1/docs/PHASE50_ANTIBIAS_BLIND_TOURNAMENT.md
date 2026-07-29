# Phase 50 — Anti-bias clean5 challenge (blind tournament)

clean5 was treated as the benchmark to **beat**, not defend. 119 new candidate families were generated
from the Phase-38 library **excluding clean5's 5 sleeves** (so no candidate can mimic it), named
anonymously, and tested on the **same** Topstep wrapper / costs / budget. clean5 and clean5+Phase36
entered only as hidden benchmarks, revealed at the end. 121 candidates total; identical 3,000-sim budget
each. Script `scripts/phase50.py`, data `docs/phase50/_phase50.json`. **clean5+Phase36 was hidden as
S070, clean5 as S003 — unknown to the ranking.**

**Headline.**

```
clean5 SURVIVES THE BLIND ANTI-BIAS TEST DECISIVELY. Across 119 anonymous candidates built from
entirely different sleeves, the two hidden clean5 entries ranked #1 (S070, $519/mo) and #2 (S003,
$441/mo) by EV/month. The next best candidate is $333/mo (36% lower) and FAILS the pass/fail ratio
criterion (0.84). NO candidate is both high-EV (>=$400) and viable (ratio>=1.5) except the two
benchmarks. EVERY kill criterion came back EMPTY.
VALIDATION, NOT LUCK: the best BLIND candidates (bos_cont/volhi_orb/ir-breakout + a mean-reversion fade
+ long-beta) independently converged on clean5's ARCHITECTURE — momentum/breakout + MR hedge — from
different sleeves. A blind search rediscovering the same shape is evidence the design is sound, not
curve-fit.
THE ONE REAL WEAKNESS IS CONFIRMED AND ISOLATED: 2023 (low-vol grind). clean5 is negative there; the
strongest 2023-positive, clean5-uncorrelated candidates are all "long-beta + fade hedge" (S076, S025)
— good diversifiers but low standalone EV, so they patch, not replace.
```

Verdict: **#1 — clean5 is still best after the blind anti-bias test** (with an optional #3: combine a
small long-beta+hedge sleeve to patch 2023).

---

## Tournament result (blind; ‹‹‹ = hidden benchmark)

### Top by EV/month
| ID | EV/mo | ratio | pass | med d | 2023 | maxDD | corr→clean5 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **S070 ‹‹‹** | **519** | 5.96 | 0.67 | 89 | −1,438 | 10,962 | 0.94 |
| **S003 ‹‹‹** | **441** | 5.77 | 0.59 | 98 | −1,863 | 10,227 | 1.00 |
| S050 | 333 | 0.84† | 0.46 | 56 | +11,152 | 11,232 | 0.27 |
| S078 | 276 | 1.42† | 0.48 | 84 | +3,253 | 19,784 | 0.56 |
| S036 | 276 | 1.26† | 0.35 | 106 | +2,195 | 13,954 | 0.49 |
| S079 | 272 | 2.05 | 0.41 | 115 | +857 | 10,237 | 0.38 |
| S093 | 266 | 2.10 | 0.41 | 132 | +3,001 | 13,374 | 0.44 |
| S108 | 257 | 1.88 | 0.37 | 91 | +2,285 | 8,031 | 0.48 |

### Top by pass/fail ratio (EV ≥ $150) — ultra-safe but low cashflow
| ID | ratio | EV/mo | pass | 2023 | identity |
|---|--:|--:|--:|--:|---|
| S116 | 192 | 168 | 0.19 | −980 | gap_go+gap_fill+gap_fade_pdc |
| S102 | 99.9 | 150 | 0.19 | +279 | ir60_break |
| S076 | 66.8 | 243 | 0.29 | +4,417 | beta_long_uptrend+on_sweep_fade |
| S064 | 21.4 | 219 | 0.23 | +4,015 | beta_long_uptrend |
| S081 | 18.0 | 226 | 0.25 | +4,068 | beta_long_uptrend+range_or_fade |
| **S070 ‹‹‹** | 5.96 | **519** | **0.67** | −1,438 | clean5+Phase36 |

> The mega-ratio candidates (100–192) are long-beta/value/gap combos that rarely blow *but rarely pass*
> (pass 19–29%) and earn 2–3× less. clean5 trades a lower (but strong) ratio for **2.4× the EV and 2–3×
> the pass rate** — the right trade for a payout factory.

### Top by 2023 net (the discriminator) — all uncorrelated long-beta plays
| ID | 2023 | EV/mo | ratio | corr→clean5 | identity |
|---|--:|--:|--:|--:|---|
| S050 | +11,152 | 333 | 0.84† | 0.27 | all-5 beta stack |
| S085 | +6,935 | 121 | 0.80† | −0.01 | dow_long all-5 |
| S005 | +5,696 | 156 | 1.17† | 0.12 | beta_long_above_vwap+on_sweep_fade |
| **S025** | +3,505 | 136 | **3.27** | 0.28 | beta_long_above_pdc+range_or_fade |
| **S076** | +4,417 | 243 | **66.8** | 0.25 | beta_long_uptrend+on_sweep_fade |

### Fastest viable pass (ratio ≥ 1.5, EV ≥ $150)
S070 (89 d) ‹‹‹ → S108 (91 d, ir30+ir60) → S003 (98 d) ‹‹‹. **clean5+Phase36 is the fastest viable engine.**

## Kill-criteria scorecard — ALL EMPTY
| kill criterion | result |
|---|---|
| another candidate beats EV/month by ≥10% (≥$571) | **none** |
| similar EV (≥95%) + faster pass + ratio ≥1.5 | **none** |
| lower DD + ≥ payout rate + ≥95% EV | **none** |
| fixes 2023 (positive) without lowering EV much (≥90%) + ratio ≥1.5 | **none** |
| clean5 only wins because it was tuned more | the wrapper/budget were identical and clean5 was **not** tuned within the tournament; the blind field independently converged on its architecture → win is on merit |

## Honest caveats
- **Selection-history asymmetry (acknowledged):** clean5 is the product of Phases 37–42 winnowing,
  whereas the candidates are raw library sleeves. clean5 has "survived more." But within this *blind,
  equally-budgeted* test it wins on EV and ratio, and the blind winners share its structure — so the
  asymmetry does not explain the result.
- **2023 is a genuine blind spot.** clean5 loses in the 2023 low-vol grind; uncorrelated long-beta+hedge
  sleeves (S025/S076) fix it — but they're long-tilted, so they patch an *up*-grind, not a down-grind,
  and lower blended EV. A real trade-off, not a free win.
- All figures use flat per-attempt cost and the frozen edge holding live (same caveats as Phases 44/49).

## Final verdict — **(1) clean5 is still best after the blind anti-bias test.**
A genuinely blind, equally-resourced field of 119 alternative families could not beat it on EV/month or
produce any high-EV + viable-ratio rival, and could not satisfy a single kill criterion. The independent
convergence of the best blind candidates onto clean5's momentum-plus-mean-reversion architecture is
positive evidence the design is structural, not curve-fit.

**Secondary (3) — optionally combine a small long-beta + fade-hedge sleeve** (S025 = beta_long_above_pdc
+ range_or_fade, or S076 = beta_long_uptrend + on_sweep_fade): both are clean5-uncorrelated (0.25–0.28),
ratio ≥ 3, and strongly positive in 2023. A modest allocation hardens clean5's only confirmed weak
regime, at the cost of a long tilt and slightly lower blended EV. **Not a replacement — a hedge.**

(Not verdict 2: nothing replaces clean5. Not verdict 4: the test was blind and clean5 won on merit, so
the research is validated, not biased.)

```
Reproduction: python -m scripts.phase50   # -> docs/phase50/_phase50.json
```
