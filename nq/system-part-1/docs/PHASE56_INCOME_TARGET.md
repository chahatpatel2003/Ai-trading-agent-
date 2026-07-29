# Phase 56 — $1,000+/month/account income search (final verdict)

Hard requirement: **≥ $1,000 net EV/month/account** after all Topstep costs (ideally $2,000+), pass in
30–45 d, slower payout OK, blow-after-payout OK. clean5+Phase36 (~$400–500/mo) fails the bar. Searched
the higher-expectancy families with **aggressive funded sizing** (now allowed) and **flat per-attempt
cost** (the most favorable case). Engines × 50/100/150K × funded size {1,2,3,5 MNQ} × payout cap {M, 2M}.
Script `scripts/phase56.py`, data `docs/phase56/_phase56.json`.

**Headline.**

```
NO ROBUST STRATEGY REACHES $1,000/month/account. Four cells nominally clear $1k — ALL at 5 MNQ funded
(the contract cap) with a 2x payout cap, on momentum/breakout engines. Stress-testing the best one
(50K momentum_stack, 5 MNQ, cap 2M, nominal "$1,688/mo") demolishes it:
  - MEDIAN funded payout = $0. 67% of funded accounts collect NOTHING. Only 33% ever clear $1,000.
  - 73% blow, mean ~47 funded days before blow.
  - The "EV" is a right-skewed mean carried by a ~27% survivor tail (top 10% of accounts hold 41% of all
    payout $). It is a LOTTERY, not income.
  - OOS: train E[payout] $15,242 -> test $8,207 (-46%). The edge halves out of sample.
These trip every kill criterion: impossible sizing (5 MNQ on a $2,000 MLL), hidden curve-fit (corner
solution: max contracts + max payout cap + OOS-fragile momentum), fails OOS, and the mean is not
realizable as monthly income (median $0).
THE ROBUST CEILING (survivable, median-positive, OOS-valid) is ~$400/month/account (clean5+Phase36,
1 MNQ funded on a 150K). Every config above ~$600 relies on aggressive funded sizing whose survival is
3-30% and whose median income is far below the mean.
```

Verdict: **Topstep is NOT worth it at the $1,000+/month/account bar with this data.** The realistic,
robust ceiling is **~$400–600/month/account.**

---

## The four nominal "$1k" hits — and why they are lottery artifacts
| cell | nominal EV/mo | funded size | funded survival | the reality |
|---|--:|--:|--:|---|
| 50K momentum_stack cap2M | $1,688 | **5 MNQ** | 28% | **median payout $0**, 73% blow, OOS −46% |
| 50K trend_day_cont cap2M | $1,321 | 5 MNQ | 18% | momentum engine, OOS-fragile, tail-carried |
| 100K momentum_stack cap2M | $1,110 | 5 MNQ | 12% | 88% blow; mean from rare survivors |
| 50K ir_breakout_stack cap2M | $1,102 | 5 MNQ | 20% | corner solution, low median |

### Stress test of the best hit (50K momentum_stack, 5 MNQ, cap 2M) — 3 seeds
| metric | value |
|---|--:|
| mean funded payout | ~$11,200 |
| **median funded payout** | **$0** |
| P(payout = $0) | **67%** |
| P(payout > $1,000) | 33% |
| funded blow rate | 73% |
| mean days before blow | 47 |
| top-10% of accounts' share of all payout $ | **41%** |
| OOS train → test payout | $15,242 → **$8,207 (−46%)** |

> The mean is a mirage: two-thirds of accounts collect nothing, the account blows in ~7 weeks, the income
> is concentrated in a lucky tail, and it halves out of sample. This is the textbook profile of a
> sizing-artifact lottery — exactly what the kill criteria are meant to reject.

## Why $1,000/month/account is structurally out of reach (the math)
- **The funded MLL caps per-account extraction.** On a 150K the trailing buffer is $4,500; once you draw
  it down you blow. The total $ extractable per account-life is bounded by how much "house money" the
  edge can build above that buffer — roughly **$5–7k per account-life regardless of sizing** (bigger size
  blows faster, it doesn't extract more). Sizing up just converts steady extraction into a faster
  lottery.
- **The edge is fixed and modest.** Six years of search (Phases 29–52) found no NQ intraday PF ≥ 2; the
  best engines run ~$37/MNQ/day. A blind 119-candidate tournament (Phase 50) found nothing better than
  clean5+Phase36. There is no "larger edge" hiding in this NQ-only data.
- **Therefore EV/cycle ≈ (≤$7k payout − ~$200 cost) and cycle ≈ 8–12 months ⇒ ~$400–600/month/account**
  for a robust (survivable) plan. To clear $1,000 you'd need the funded buffer to be ~2–3× larger or the
  edge ~2–3× stronger — neither exists here.

## Robust ceiling (survivable, OOS-valid, median-positive)
| account | engine | funded | EV/month/account | funded survival |
|---|---|--:|--:|--:|
| 150K | clean5+Phase36 | 1 MNQ | ~$388 | 34% |
| 150K | clean5+Phase36 | 2 MNQ | ~$514 | 2% (tail-carried) |
| 100K | clean5+Phase36 | 1 MNQ | ~$344 | 20% |
| 50K | clean5+Phase36 | 1 MNQ | ~$409 | 3% |

> The only *survivable* configuration is **1 MNQ funded (~$400/month/account)**. Everything labeled
> $500–870 earlier relies on 2–5 MNQ funded with 2–30% survival — i.e., the same tail-carried lottery,
> just less extreme. So the honest, reliable number is **~$400/month/account**.

## Final verdict
**Either find $1,000+/month/account, or say Topstep is not worth it with this data → it is NOT worth it.**

- **No robust strategy reaches $1,000/month/account.** The only cells that nominally do are 5-MNQ-on-a-
  tiny-MLL lotteries with **median income $0, 73% blow, and OOS that halves** — they fail impossible-
  sizing, curve-fit, and OOS kill criteria.
- **The robust, survivable ceiling is ~$400/month/account** (clean5+Phase36, 1 MNQ funded), maybe
  ~$500–600 if you accept higher variance with 2 MNQ funded. That is **less than half the requirement.**
- **The cap is structural, not a search failure:** the $4,500 MLL bounds per-account extraction and the
  NQ edge (no PF ≥ 2 in 6 years) can't refill it fast enough. More search won't change this.
- **If you need $1,000+/month, the only honest path is more accounts, not more per-account** — e.g. ~3
  accounts × ~$400 ≈ $1,200/month total (the Phase-46 factory), accepting the operational load and
  lumpy, fee-heavy cashflow. But **per-account $1,000+ is not achievable** with this edge.

Honest bottom line: with this data, Topstep is a **~$400/month/account** grind at best — a small
supplementary income, not a $1–2k/month/account business. At the income bar you set, **it is not worth
it.**

```
Reproduction: python -m scripts.phase56   # -> docs/phase56/_phase56.json
```
