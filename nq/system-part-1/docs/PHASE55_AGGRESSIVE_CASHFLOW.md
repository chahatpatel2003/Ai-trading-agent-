# Phase 55 — Aggressive cashflow-optimized Topstep cycle (accept high fail)

Mandate: stop rejecting on fail rate; accept 40–60% fail and ratio < 1.25 **if** EV/month improves.
Pass within 20–22 trading days, optimize EV/month after monthly subscription + reset + activation,
repeatable-cycle cashflow, not survival. Larger sizing allowed (fixed 3/4/5 MNQ, 1 NQ where the contract
cap permits); reset after 20–22 days; funded stays 1-MNQ. Simulated 3-month (63-day) cashflow timelines
(combine → pass → 1-MNQ funded → payouts → blow → restart). Per-account fees (monthly = reset =
activation): 50K $49 / 100K $99 / 150K $199. clean5+Phase36 signals. Script `scripts/phase55.py`.

**Headline — answered on cashflow terms, not safety.**

```
GOING AGGRESSIVE DOES NOT IMPROVE CASHFLOW. Optimizing EV/month directly (not rejecting on fail rate),
the BEST cell in the entire 50K/100K/150K x sizing x cap matrix is 50K fixed-5 MNQ cap-20 = +$8/month
(break-even). Every 100K and 150K aggressive plan is NEGATIVE (-$20 to -$443/month). And the best plan
LOSES money over 3 months 82% of the time (profits only 18% of windows).
TWO REASONS (neither is "safety"):
  1. Pass-within-22-days is only ~15-22% per attempt even at max size -> you pay 5-7 capped attempts
     (~months + resets) per pass, which swamps the funded payout — worst on 150K ($199 fee x many
     attempts vs a $9,000 target).
  2. The funded FIRST PAYOUT takes ~48 days, so even when you pass, a 63-day window rarely leaves time
     to collect a payout. A 3-month horizon is too short for this cycle to pay.
The 20-22 day cap is too aggressive on TIME, not size — bigger sizing can't fix it (confirms Phase 54:
cap20/30 = reset spiral -> negative EV).
```

Verdict: **No aggressive fast-cycle plan produces reliable positive cashflow.** Best is 50K break-even.

---

## Aggressive matrix (EV/month is the optimized metric; fail rate NOT a filter)
### 50K (target $3k, MLL $2k, fee $49)
| sizing / cap | pass | fail | ratio | med pass d | **EV/mo** | P(profit 3mo) | P(loss 3mo) | worst 3mo |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **fixed5 / cap20** | 15.3% | 42.9% | 0.36 | 14 | **+8** | 18.4% | 81.6% | −784 |
| fixed5 / cap22 | 18.3% | 45.1% | 0.41 | 14 | −3 | 19.1% | 80.9% | −882 |
| fixed4 / cap20 | 13.8% | 33.6% | 0.41 | 15 | −17 | 14.6% | 85.4% | −784 |
| fixed3 / cap22 | 12.3% | 24.3% | 0.51 | 18 | −62 | 11.2% | 88.8% | −735 |
| front234 / cap22 | 10.4% | 9.7% | 1.07 | 16 | −71 | 9.0% | 91.0% | −637 |

### 100K (target $6k, MLL $3k, fee $99)
| sizing / cap | pass | fail | EV/mo | P(profit 3mo) | worst 3mo |
|---|--:|--:|--:|--:|--:|
| **1NQ / cap22** | 22.2% | 63.3% | **−20** | 30.7% | −2,475 |
| fixed5 / cap22 | 18.6% | 43.8% | −103 | 19.4% | −1,782 |
| fixed3 / cap22 | 11.8% | 24.4% | −184 | 11.0% | −1,485 |

### 150K (target $9k, MLL $4.5k, fee $199)
| sizing / cap | pass | fail | EV/mo | P(profit 3mo) | worst 3mo |
|---|--:|--:|--:|--:|--:|
| **1NQ / cap22** | 22.0% | 63.5% | **−250** | 28.0% | **−5,174** |
| fixed5 / cap20 | 15.7% | 42.5% | −307 | 17.5% | −3,582 |
| fixed3 / cap22 | 12.5% | 24.6% | −443 | 11.2% | −2,786 |

(Full grid in `_phase55.json`.)

## Reading it
- **50K is the only account where aggressive is even break-even** (cheapest fee, smallest $3k target).
  100K/150K aggressive plans are all negative — the fee scales up ($99/$199) and the bigger target needs
  more capped attempts, so the funded payout can't cover the cost-to-pass.
- **Larger size raises pass-within-22 (fixed5 ~18% vs fixed3 ~12%) but also blow (43% vs 24%)** — and the
  EV barely moves, because the binding cost is the *number of paid attempts*, not the blow rate.
- **1 NQ** passes fastest (~22% in ~11 days) and has the highest P(profit-3mo) (~28–31%), but its
  worst-3mo is catastrophic (−$2,475 on 100K, **−$5,174 on 150K** — exceeds the MLL), EV is still
  negative, and on 50K it isn't allowed (10 > 5-contract cap). **1 NQ is drawdown-disallowed** on every
  account (worst day > MLL); it does not improve cashflow.
- **P(make money over 3 months) is 11–31%** across all aggressive plans → you **lose money over a
  3-month window 69–89% of the time.** The rare profitable windows (a fast pass + an early payout)
  carry the near-zero/negative average.

## Why the 3-month window loses even when EV-positive elsewhere
Two structural lags make a 3-month aggressive horizon unprofitable:
1. **Combine cost lag:** at ~15–22% pass-per-attempt you burn several capped attempts (≈ a subscription
   month each) before a pass — most of the 63-day window is spent paying for failed combines.
2. **Payout lag:** the funded first payout needs ~5 winning days + ~48 days on average — so even a pass
   late in the window yields **no collected payout** before the 3 months end.
A 22-day cap maximizes both lags' damage: it resolves attempts before the edge can reach the target
(only the fastest ~18% pass), creating a reset spiral, and it leaves little funded runway.

## Best aggressive plan (if you insist on aggression)
**50K, fixed 5 MNQ, reset at day 20, daily loss −$1,000, funded → 1 MNQ.**
Pass 15%, fail 43%, EV ≈ **+$8/month** (break-even), worst 3-month −$784, P(profit over 3 months) 18%.
It is the *least-bad* aggressive plan — smallest fee, smallest target — but it does **not** generate
meaningful cashflow.

## Final verdict (cashflow, not safety)
Optimizing purely for EV/month and accepting 40–60% fail, **aggressive 20–22 day cycling fails to
produce positive cashflow** — break-even at best (50K) and clearly negative on 100K/150K, losing money
over ~80% of 3-month windows. This is **not** a safety rejection: the math itself is break-even-to-
negative because the fast-pass rate is too low and the funded-payout lag is too long for a tight cap.

If cashflow is the goal, the better plan is the **patient front-loaded one from Phase 54** (front 2→3→4
MNQ, **60-day** cap, no progress gate → ~$387/month over a full cycle): give the edge ~30–40 days to
reach target rather than capping at 22. Going aggressive on **size** is fine; going aggressive on **time
(a 20–22 day cap)** is what destroys the cashflow.

```
Reproduction: python -m scripts.phase55   # -> docs/phase55/_phase55.json
```
