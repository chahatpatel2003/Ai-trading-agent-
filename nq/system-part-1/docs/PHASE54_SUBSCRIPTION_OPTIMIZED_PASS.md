# Phase 54 — Topstep subscription-time-optimized pass plan (150K, clean5+Phase36)

Problem: the high-pass-probability engine (cushion, ~79% pass-before-blow) has a **median ~62-day** pass,
so under **monthly** billing a campaign pays ~**9.8 months** of subscription per pass — EV-eroding.
Objective here: maximize **net EV per subscription month**, not pass probability. Tested time caps
(20/30/45/60 d), progress gates (must be above $X by day D else reset), and front-loaded sizing, via a
campaign sim (re-attempt until pass, accruing monthly + reset fees). Funded payout held constant from a
1-MNQ funded sim ($4,892, ~3 funded months). Costs: monthly $150, reset $150, activation $150. Script
`scripts/phase54.py`, data `docs/phase54/_phase54.json`.

**Headline.**

```
UNDER MONTHLY BILLING, SPEED BEATS SAFETY — the opposite of the pass-probability optimum (Phase 42).
The subscription-optimal plan is FRONT-LOADED AGGRESSIVE sizing (2->3->4 MNQ) + a 60-day time cap +
NO progress gate + daily loss -$1,000: EV/month $387 vs the cushion baseline's $244 (+59%). It resolves
each attempt in ~32 days median (pass OR blow), so it bleeds far fewer subscription months per pass
(6.2 vs 9.8), even though it blows ~30% of attempts (which we accept).
PROGRESS GATES HURT. Resetting attempts that are "behind schedule" kills viable ones — reset rate jumps
33% -> 78% and EV/month falls (aggr cap60: no-gate $387 > med-gate $302 > aggr-gate $250). The only
useful early exit is the natural fast blow + the 60-day cap.
TOO-TIGHT CAPS BACKFIRE. cap20/cap30 reset 96-99% of attempts -> a reset-fee spiral -> EV negative.
cap60 is the sweet spot.
```

Verdict: **front-loaded 2→3→4 MNQ, cap 60 days, no gate, daily stop −$1,000** — the best plan for
passing without wasting 2–3 months of subscriptions.

---

## The baseline problem
| plan | per-attempt pass | blow | reset | med pass d | **months/pass** | cost/pass | EV/month |
|---|--:|--:|--:|--:|--:|--:|--:|
| cushion, cap120, no gate (the 79%-pass engine) | 48.5% | 9.7% | 41.8% | 70 | **9.77** | $1,778 | **$244** |

> Pass-before-blow is 83%, but campaigns pay ~9.8 months of subscription per pass because the cushion
> grind is slow and the long tail (cap120) is expensive.

## Optimization matrix (daily loss −$1,000; EV/month is the target metric)
| sizing | cap | gate | pass | blow | reset | med pass d | months/pass | cost/pass | **EV/month** |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|
| **aggr 2→3→4** | **60** | **none** | 37% | 30.5% | 32.5% | 32 | **6.19** | $1,330 | **$387** |
| front 2→3 | 60 | none | 33% | 26% | 41% | 34 | 7.32 | $1,551 | $324 |
| aggr 2→3→4 | 60 | med | 17.6% | 4.2% | 78.1% | 21 | 6.91 | $1,899 | $302 |
| aggr 2→3→4 | 45 | none | 28.8% | 23% | 48.2% | 27 | 8.08 | $1,734 | $285 |
| aggr 2→3→4 | 45 | med | 16.6% | 4.0% | 79.4% | 20 | 7.26 | $2,002 | $282 |
| aggr 2→3→4 | 30 | med | 14.9% | 4.1% | 81% | 19 | 7.62 | $2,117 | $261 |
| front 2→3 | 60 | med | 15.8% | 2.6% | 81.5% | 23 | 8.12 | $2,219 | $240 |
| cushion 123 | 60 | none | 16.5% | 4.4% | 79.1% | 40 | 16.81 | $3,395 | $76 |
| aggr 2→3→4 | 20 | none | 8.6% | 8.4% | 83% | 16 | 10.63 | $3,190 | $125 |
| cushion 123 | 30 | any | 3–4% | ~0% | 96–97% | 25 | 18–20 | $5,300 | −20 |
| cushion 123 | 20 | any | 0.3–0.4% | 0% | 99.6% | 19 | ~20 | $6,000 | −45 to −52 |

(Full 36-cell grid in `_phase54.json`.)

## Why front-loaded beats cushion (under monthly billing)
- **Cushion (start 1 MNQ)** is built for *survival* — it passes 79% but slowly (median 62–70 d). Under
  monthly billing the slow grind + long tail costs ~9.8 months/pass.
- **Front-loaded (start 2, scale to 3–4)** reaches the target faster (median 32 d) *and* blows faster
  when it's going to blow. Both outcomes resolve the attempt quickly, so you pay fewer subscription
  months per pass (6.2). The higher blow rate (30%) is acceptable because the objective is EV/month and
  the user accepts post-attempt blow.
- This is an **objective-dependent reversal of Phase 42**: under-sizing maximizes *pass probability*;
  front-loaded sizing maximizes *EV per subscription month* when billing is monthly. Both are right for
  their objective.

## Best cutoff / reset / sizing (the user's asks)
- **Best cutoff day (time cap): 60.** Long enough that aggressive sizing's occasional slow winner still
  passes; short enough to abandon the dead grind. cap20/30 trigger a reset-fee spiral (96–99% reset).
- **Best reset rule: NONE beyond the natural blow + the 60-day cap.** Progress gates (day-10/20/30
  thresholds) *reduce* EV/month by resetting attempts that would have passed. Do **not** cancel on being
  "behind schedule"; only stop on a blow or at day 60.
- **Best sizing plan: aggressive front-loaded** — **2 MNQ until +$1,500 cushion → 3 MNQ until +$4,000 →
  4 MNQ thereafter**, daily loss stop **−$1,000**. (front 2→3 is a slightly safer, slightly lower-EV
  alternative.)
- **1 NQ:** not used — its worst day exceeds the $4,500 MLL → near-certain fast blow → reset churn that
  raises cost/pass (does not improve EV/month).

## Final output — the best plan
```
Account : Topstep 150K (target +$9,000, trailing MLL $4,500, 15ct, consistency 50%), clean5+Phase36 signals.
Sizing  : FRONT-LOADED — 2 MNQ until +$1,500 profit; 3 MNQ until +$4,000; 4 MNQ thereafter.
Daily   : hard daily loss stop -$1,000.  Cap 3 trades/day.  Skip neutral-trend days (gate).
Reset   : abandon/cancel the combine ONLY if (a) it blows the trailing DD, or (b) it has not passed by
          trading day 60. NO interim progress-gate cancellation.
On pass : switch to FUNDED mode = 1 MNQ grinder, extract every payout immediately (Phase 42/44).
```
**Expected economics:** per attempt 37% pass / 30% blow / 33% time-out, median pass ~32 days; campaign
pays ~6.2 subscription months + ~1.7 resets per pass (cost-to-pass ≈ $1,330); **net EV ≈ $387 / month**
(vs $244 for the slow cushion plan) — a 59% improvement, achieved by spending fewer months in the combine.

## Honest caveats
- This **raises per-attempt blow to ~30%** and drops the pass/fail ratio to ~1.2 (vs cushion's 4–8). It
  is the right choice ONLY under the stated objective (EV per subscription month, monthly billing,
  accept-blow). If your real cost is flat per-attempt (Phase 44), or you want maximum pass probability,
  use the cushion plan instead.
- EV magnitudes depend on the assumed fees ($150 monthly/reset/activation) and the frozen edge holding
  live; the *ranking* (front-loaded + cap60 + no-gate) is robust to fee level.
- Funded payout is held constant here; the combine choice doesn't change funded behavior (Phase 44/46).

```
Reproduction: python -m scripts.phase54   # -> docs/phase54/_phase54.json
```
