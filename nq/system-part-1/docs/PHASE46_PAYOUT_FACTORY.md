# Phase 46 — Multi-account Topstep 150K payout factory

Builds a repeatable multi-account factory (combine → funded → blow → restart) and tests how account
count and strategy allocation interact through **correlation**. Correlation is modeled by construction:
all accounts run over the **same calendar dates**, so same-strategy accounts share that day's PnL
(correlated); strategy-mix and staggered starts decorrelate. **Monthly** combine cost + activation +
reset modeled (the realistic, conservative billing — unlike Phase 44's flat cost). Frozen signals only
(clean5+gate harsh, Phase-36 idb+swd). 150K each (target $9k, MLL $4.5k, 15ct), funded = 1 MNQ, cushion
combine sizing. Fees assumed: $150/mo combine, $150 activation, $150 reset. Script `scripts/phase46.py`.

**Headline.**

```
THE FACTORY'S VALUE IS RELIABILITY, NOT HIGHER RETURNS. Per-account EV is ~constant (~$160-180/mo under
monthly billing); adding accounts scales total EV roughly linearly. The win from more accounts is
DECORRELATION, and it ONLY appears if you STAGGER starts: P(all accounts blow in one month) falls
6% -> 0.5% (1 -> 5 accts) and P(>=1 payout per month) rises 10% -> 19%. SYNCED same-strategy accounts
do NOT diversify — P(all-blow) stays ~6% at every count (they share the same bad day).
DO NOT DEDICATE WHOLE ACCOUNTS TO PHASE-36 OR A-ONLY. Those engines are too slow to cover their own
monthly combine fee -> mixed/diversified portfolios collapse to ~$0 or NEGATIVE EV. The Phase-36 edge
must be COMBINED INSIDE each account (the "combo"), not run as standalone accounts.
BEST FACTORY = N staggered "combo" (clean5+Phase-36) accounts. ~70% of account-time is spent in the
(long) combine paying fees; only ~30% is funded earning -> the factory is fee-heavy and payouts are
lumpy (a payout in only ~1 month in 5, even diversified).
```

Verdict: **3–5 staggered combo accounts** is the sweet spot — modest, real, but fee-heavy and lumpy
income. Worth running only mechanically and only if the monthly combine fee is cheap.

---

## Portfolio matrix (steady-state monthly metrics; `avg funded` corrected to value/N)

### `staggered_combo` — clean5+Phase-36 on every account, staggered starts (THE WINNER)
| accts | monthly cost | monthly payout | **net EV/mo** | EV/acct | payout-prob/mo | worst month | 5th-pct month | **P(all blow same mo)** | avg funded | 
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 115 | 290 | 175 | 175 | 9.8% | −379 | −264 | 6.0% | 0.32 |
| 2 | 230 | 547 | 318 | 159 | 13.7% | −757 | −450 | 2.4% | 0.62 |
| 3 | 358 | 827 | 469 | 156 | 16.4% | −1,136 | −636 | 1.2% | 0.86 |
| **5** | 599 | 1,428 | **830** | 166 | **19.2%** | −1,757 | −1,050 | **0.5%** | 1.43 |
| 10 | 1,213 | 2,695 | 1,482 | 148 | 17.9% | −3,514 | −2,071 | 0.7% | 2.72 |

### `synced_combo` — same strategy, same start (correlated baseline)
| accts | net EV/mo | payout-prob/mo | worst month | **P(all blow same mo)** |
|--:|--:|--:|--:|--:|
| 1 | 183 | 10.0% | −379 | 6.1% |
| 3 | 558 | 10.1% | −1,136 | 5.9% |
| 5 | 832 | 9.4% | −1,893 | 5.7% |
| 10 | 2,018 | 10.6% | −3,786 | 6.3% |

> Synced gives marginally **higher** raw EV (accounts fully utilized, no stagger ramp) but **9× the
> correlated-ruin risk** (P(all-blow) ~6% at every N vs 0.5–1% staggered) and **half the monthly payout
> reliability** (10% vs 19%). For a *cashflow factory*, staggered wins decisively.

### Allocations that dedicate accounts to Phase-36 / A-only (REJECTED)
| scheme | net EV/mo @5 | net EV/mo @10 |
|---|--:|--:|
| mixed_staggered (combo + phase36) | 252 | 48 |
| grinder_vs_infreq (clean5 + phase36) | 132 | **−65** |
| diversified (clean5/phase36/aonly) | **−156** | **−184** |

> Whole accounts on phase36/aonly **can't cover their monthly combine fee** → they drag the portfolio to
> zero or negative. Decorrelation (P(all-blow)=0) is real but worthless if the EV is gone. **Combine the
> edges inside one account, don't split them across accounts.**

## Final ranking — best Topstep payout factory

**1) Best account count — `5` (sweet spot); scale to `10` only with capital.**
At 5 staggered combo accounts the decorrelation is essentially captured (P(all-blow) 0.5%, payout in
~19% of months) at ~$830/mo net EV. Beyond 5, total EV keeps scaling (~$1,482/mo at 10) but per-account
efficiency dips slightly and absolute monthly swings grow (worst month −$3.5k) with **no further
diversification gain**. 1–2 accounts leave reliability on the table (P(all-blow) 2–6%, payout 10–14%).

**2) Best strategy mix — `combo (clean5 + Phase-36) on EVERY account`, never dedicated phase36/aonly accounts.**
The Phase-36 mean-reversion edge belongs *inside* each account (it diversifies intraday and lifts the
pass rate); splitting it into its own slow accounts destroys EV.

**3) Best sizing plan — cushion combine (1 MNQ → 2 after +$2k → 3 after +$4k), 1 MNQ funded, daily stop ≈ −$1,700**,
with **staggered start dates** (~1 month apart) so accounts sit at different equity levels and never share
a drawdown window.

**4) Best overall factory — 5 staggered `combo` 150K accounts:** ~$830/mo net EV, payout in ~19% of
months, P(all blow in one month) 0.5%, ~1.4 of 5 accounts funded at any time, worst month ≈ −$1.8k.

## Honest caveats (do not skip)
- **Monthly billing roughly thirds the EV** vs Phase 44's flat-cost model ($166/acct/mo here vs ~$452
  flat). The factory is positive-EV only while the **combine fee is cheap**; at higher monthly fees it
  goes negative (the rejected mixes already show how thin the margin is).
- **~70% of account-time is unproductive** (in the multi-month combine paying fees); only ~30% is funded
  and earning. This is structural — the combine is slow.
- **Payouts are lumpy:** even 5 diversified accounts pay in only ~1 month in 5. This is irregular
  side-income, not a salary.
- **Funded accounts blow after 1–2 payouts by design** (we optimize repeatable cashflow, not longevity).
- Everything rests on the frozen edge holding live and on the fee assumptions; treat as forward-paper
  first (Phase 40 readiness applies per-account).

### Bottom line
A multi-account Topstep factory is **real but modest**: run the **combo engine on every account, stagger
the starts, hold ~5 accounts**. The point of multiple accounts is *reliability through decorrelation*
(more months with a payout, near-zero chance of all blowing at once), not higher per-account returns.
Do not dilute it with phase36-only accounts, and do not expect a salary — expect lumpy ~$800/mo net at
5 accounts under cheap monthly fees, with most of your account-time spent grinding combines.

```
Reproduction: python -m scripts.phase46   # -> docs/phase46/_phase46.json
```
