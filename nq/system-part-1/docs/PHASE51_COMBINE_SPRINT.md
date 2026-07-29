# Phase 51 — Topstep Combine Sprint Engine Search (user-labeled "Phase 50")

Single job: pass the combine **fast** (target 10–20 trading days), fail ≤ pass, pass/fail ratio ≥ 1.25
(ideally 1.5+). Not funded EV, not long-term ROI, not protecting clean5 (it's only the benchmark to
beat). Searched 14 high-expectancy pass-engine families × 50K/100K/150K × sizing (cushion, aggressive
cushion) with a daily loss stop. Frozen signals, flat cost, 6,000 sims/cell. Script `scripts/phase51.py`,
data `docs/phase51/_phase51.json`.

**Headline.**

```
FAIL — fast passing is NOT realistically possible. There is a hard frontier across every engine,
account, and sizing:
  - FAST configs (aggressive sizing -> pass-within-20-days ~15-22%, median 15-18 d): pass/fail ratio
    0.30-0.55 (fail >> pass) -> KILLED. On a 50K their worst day (-$3,153) exceeds the $2,000 MLL, so
    a single bad day blows the account.
  - VIABLE configs (ratio >= 1.25): ALL slow — median 28-60 days, pass-within-20 <= 5%.
No engine/account/sizing is both fast (<=20 d) and viable (ratio >=1.25). The relaxed 1.25 bar (vs the
1.5 used in Phase 47) did NOT change the verdict. This is the THIRD independent confirmation (Phases 41,
47, 51) that the edge cannot sprint.
```

Verdict: **FAIL.** Plan for a ~4–8 week combine (slow cushion), not a 2–4 week sprint.

---

## The speed/viability frontier (best cells per regime)

### Fast but KILLED (ratio < 1.25, fail > pass)
| account | engine / sizing | pass≤10 | pass≤20 | fail | ratio | med d | worst day | consist | why killed |
|---|---|--:|--:|--:|--:|--:|--:|--:|---|
| 50K | momentum_stack / aggr | 10.3% | 21.8% | 67% | 0.46 | 15 | −3,153 | 39% | fail≫pass; 1 day > MLL; consistency |
| 50K | clean5+P36 / aggr | 10.8% | 21.5% | 64% | 0.55 | 16 | −2,541 | 32% | fail>pass; 1 day > MLL |
| 50K | hivol_trend / aggr | 8.7% | 18.9% | 73% | 0.37 | 15 | −3,153 | 28% | fail≫pass; 1 day > MLL |
| 50K | first_pullback / aggr | 8.0% | 17.1% | 63% | 0.52 | 19 | −1,535 | 9% | fail>pass |
| 150K | momentum_stack / aggr | 2.2% | 8.0% | 57% | 0.36 | 27 | −3,153 | 19% | fail≫pass |

### Viable (ratio ≥ 1.25) but SLOW (not a sprint)
| account | engine / sizing | pass≤20 | fail | ratio | med d | note |
|---|---|--:|--:|--:|--:|---|
| 150K | clean5+P36 / cushion | 0.5% | 3.5% | **4.78** | 40 | best ratio, far too slow |
| 100K | clean5+P36 / cushion | 5.3% | 17% | 1.84 | 36 | slow |
| 150K | opening_drive / cushion | 0% | 2.2% | 1.60 | 41 | slow |
| 50K | clean5+P36 / cushion | 13.3% | 42% | 1.01† | 28 | closest to fast, but ratio < 1.25 |
| 50K | ir90_aggr / cushion | 0.5% | 10% | 1.03† | 47 | slow + ratio < 1.25 |

† just under the 1.25 bar. The "never-blow" high-ratio engines (gap_and_go 100K ratio 5.08; ir90/news_vol
150K ratio 99.9) all have **pass≤20 = 0%** — they survive by trading so little they can't pass in 20 days
(median 50–60 d).

## Kill-criteria scorecard (every fast config)
| criterion | fast configs |
|---|---|
| fail probability > pass probability | **YES** (fail 57–73%) |
| pass/fail ratio < 1.25 | **YES** (0.30–0.55) |
| only works by oversized risk | **YES** (aggressive sizing is the only thing that's fast) |
| violates consistency | **YES** (19–39% of attempts blocked by best-day > 50%) |
| one bad day can destroy the account | **YES** on 50K (worst day −$3,153 > $2,000 MLL) |

## Final output
**1) Best fast-pass strategy — NONE qualifies.** The fastest engines are `momentum_stack` (volhi_orb +
od_cont_struct + ir30_break) and `clean5+Phase36` on aggressive sizing — ~15–16 day median, ~22%
pass-within-20 — but both fail every kill criterion (ratio 0.46–0.55, fail 64–67%, single day > MLL).
There is no viable fast-pass strategy.

**2) Best account size for fast pass — 50K** *if* fast passing worked (smallest target $3,000 → the
quickest passes when they happen). But 50K is also the most fragile: at the sizing needed for speed, one
bad day (−$3,153) exceeds the $2,000 MLL. So 50K is "fastest" and "most likely to blow." 100K/150K are
slower (bigger targets) but survive longer.

**3) Best sizing plan — cushion (1 → 2 after cushion → 3), and it is NOT fast.** Cushion is the only
sizing that keeps ratio ≥ 1.25 (best: 150K cushion ratio 4.78), but its median pass is 28–40 days.
Aggressive sizing is the only thing that sprints, and it inverts the ratio. There is no sizing that is
both fast and viable. (Daily loss stop and profit lock help consistency slightly but cannot fix the
core problem.)

**4) Is fast passing realistically possible? — NO.** To reach even the $3,000 (50K) target in 10–20
days you must size up enough that drawdown breaches the MLL more often than you pass. The edge
(~$37/MNQ/day) simply cannot cover the target that fast within the contract/drawdown limits.

**5) FAIL — clearly.** Fast Topstep passing is not achievable for this (or any tested) engine. The
realistic combine is a **slow cushion grind: ~4–8 weeks, ratio ~1.8 (100K) to ~4.8 (150K)**. Build the
plan around a multi-week pass, not a sprint. (Confirms Phases 41 and 47; the relaxed 1.25 bar did not
change the result.)

```
Reproduction: python -m scripts.phase51   # -> docs/phase51/_phase51.json
```
