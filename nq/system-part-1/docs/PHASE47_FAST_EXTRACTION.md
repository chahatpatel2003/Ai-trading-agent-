# Phase 47 — Fast Topstep payout extraction test

Goal: can we pass in 1–2 weeks (5–15 trading days), grab 1–3 payouts fast, and accept blow-after-payout —
optimizing **cash extracted before blow**, not longevity? Frozen signals only; flat per-attempt cost
(50K $49 / 100K $99 / 150K $199). Tests 7 strategies × 3 accounts × sizing (1–5 MNQ, 1 NQ, cushion,
scale-up). Script `scripts/phase47.py`, data `docs/phase47/_phase47.json`.

**Headline.**

```
FAST EXTRACTION IS NOT REALISTIC. There is a hard frontier:
  - FAST (pass within 15 days >10% of the time) => fail > pass, pass/fail ratio 0.3–0.6 => KILLED.
  - SAFE (ratio >=1.5) => passes within 15 days <3% of the time (avg ~40 days) => NOT fast.
You cannot have both. The edge (~$37/MNQ/day) is too small to reach even the $3,000 (50K) target in
1–2 weeks without sizing that breaches the MLL more often than it passes.
SPEED BUYS ZERO EV. 150K cushion (slow, ratio 3.56) yields EV ~$742/mo; 150K at 1 NQ (fast, ratio 0.34,
74% blow, worst day −$6,366 > the $4,500 MLL) yields the SAME ~$744/mo. Aggressive sizing just swaps a
4% blow rate for a 74% blow rate at identical expected value. It is strictly dominated.
"FIRST PAYOUT FAST" ALSO FAILS: funded first payout takes ~41–53 days (you need 5 winning days +
consistency), never 1–2 weeks — at any size.
```

Verdict: **Abandon the fast premise.** Every fast config fails the kill criteria (fail>pass, ratio<1.5,
oversized risk) and offers no EV edge over the slow cushion plan. The realistic cycle stays slow.

---

## Part A — sizing sweep (combo engine), the speed/survival frontier

### 50K (target $3,000, MLL $2,000, max 5)
| sizing | pass≤5 | pass≤10 | pass≤15 | fail | **ratio** | avg d→pass | P1st | P2nd | P3rd | E[total payout] | EV/mo | worst day | maxDD | consist |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| flat1 | 0% | 2.1% | 4.8% | 32% | 1.10 | 32 | 72% | 43% | 19% | 2,515 | 371 | −1,136 | 2,858 | 8% |
| flat3 | 3.5% | 11.4% | 17.4% | 65% | 0.51† | 17 | 72% | 43% | 19% | 2,472 | 418 | −1,910 | 6,981 | 29% |
| flat5 | 3.5% | 11.7% | 15.9% | 74% | 0.32† | 15 | 72% | 43% | 18% | 2,489 | 423 | −3,183 | 11,135 | 31% |
| cushion | 0.8% | 4.2% | 8.7% | 43% | 0.91† | 29 | 71% | 43% | 19% | 2,473 | 354 | −1,569 | 7,155 | 25% |

### 100K (target $6,000, MLL $3,000, max 10)
| sizing | pass≤5 | pass≤10 | pass≤15 | fail | **ratio** | avg d→pass | EV/mo | worst day | maxDD |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| flat1 | 0% | 0% | 0.5% | 14% | 0.94† | 41 | 411 | −1,718 | 4,280 |
| flat5 | 2.1% | 8.6% | 14.4% | 70% | 0.40† | 19 | 519 | −3,183 | 12,062 |
| **cushion** | 0% | 0.8% | 2.7% | 17% | **1.83** | 36 | 495 | −2,240 | 7,059 |

### 150K (target $9,000, MLL $4,500, max 15)
| sizing | pass≤5 | pass≤10 | pass≤15 | fail | **ratio** | avg d→pass | P1st | P2nd | P3rd | E[total payout] | EV/mo | worst day | maxDD |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| flat1 | 0% | 0% | 0.3% | 2.9% | 1.23 | 37 | 88% | 64% | 34% | 4,478 | 610 | −1,718 | 4,457 |
| flat5 | 0.8% | 5.1% | 11.4% | 60% | 0.61† | 25 | 88% | 65% | 35% | 4,499 | 726 | −3,183 | 13,405 |
| **cushion** | 0% | 0% | 0.1% | 4.2% | **3.56** | 40 | 88% | 64% | 35% | 4,496 | **742** | −2,470 | 8,674 |
| 1 NQ (=10 MNQ) | 3.2% | 10.4% | 15.0% | 74% | 0.34† | 16 | 89% | 65% | 35% | 4,490 | 744 | **−6,366** | 23,525 |

† fails the kill criterion *pass/fail ratio < 1.5*.

## Part B — strategy sweep (150K, flat3 "fast" size)
| strategy | pass≤15 | fail | ratio | avg d→pass | E[total payout] | EV/mo | notes |
|---|--:|--:|--:|--:|--:|--:|---|
| combo (clean5+P36) | 4.4% | 42% | 0.77† | 31 | 4,481 | 670 | best fast-ish |
| A-only | 1.3% | 34% | 0.59† | 36 | 4,047 | 472 | |
| ir90+trend_adx | 0.3% | 27% | 0.50† | 41 | 3,464 | 334 | |
| trend_adx only | 0% | 26% | 0.34† | 41 | 2,612 | 175 | weak |
| hivol trend-day | 3.1% | 36% | 0.58† | 32 | 3,260 | 328 | |
| **Phase-36 only** | 0% | 5% | 0.00 | — | 1,451 | **−98** | can't pass $9k fast at all → EV/cycle −$197k |
| aggr4 + scale-up funded | 7.7% | 53% | 0.70† | 30 | 5,034 | **784** | highest EV but ratio 0.70 (killed), 11% consistency viol |

## The decisive comparison
| | 150K cushion (slow) | 150K 1 NQ (fast) |
|---|--:|--:|
| pass ≤15 days | 0.1% | 15.0% |
| fail before pass | 4.2% | 74.0% |
| pass/fail ratio | **3.56** | 0.34 |
| worst day | −$2,470 | **−$6,366** (>MLL) |
| max drawdown | $8,674 | $23,525 |
| EV / month | $742 | $744 |

> Identical EV. The only thing speed buys is a **74% blow rate instead of 4%** and a worst day that
> breaches the MLL. Fast sizing is pure downside.

## Kill-criteria scorecard (every fast config)
| criterion | fast configs (flat3-5, 1NQ) |
|---|---|
| fail probability > pass probability | **YES — killed** (fail 42–74%) |
| pass/fail ratio < 1.5 | **YES — killed** (0.3–0.8) |
| first-payout probability too low | no (P1st 72–89%) — but takes ~41–53 days |
| EV negative after real costs | no (flat cost keeps them +EV) — but no edge over slow |
| works only by oversized risk | **YES** (1 NQ worst day −$6,366 > $4,500 MLL) |
| best day violates consistency | elevated (15–31% of attempts blocked) |

## Final ranking
**1) Fastest pass — `50K, flat 4–5 MNQ`** (pass≤15 ~16%, pass≤5 ~3.5%). **But ratio 0.32–0.38, 74% blow,
consistency-blocked 31% — KILLED.** It's a losing lottery; "fastest" ≠ viable.

**2) Safest pass — `150K cushion`** (ratio 3.56, blow 4%). **But pass≤15 = 0.1%, avg 40 days — not fast.**
Safe = slow. (Closest "ratio≥1.5 with any speed": 100K cushion, ratio 1.83, pass≤15 2.7%.)

**3) Best first payout — `150K`** (P 1st 88%) — but ~52 days to get it, not 1–2 weeks.

**4) Best 2–3 payout path — `150K cushion / flat1, 1 MNQ funded`** (P 1st/2nd/3rd = 88% / 64% / 35%,
E[total] $4,496). The 150K's larger buffer survives to the most payouts. A-only similar with fewer.

**5) Best multi-account — `5 staggered combo 150K` (Phase 46)** — but multi-account does **not** make fast
extraction work; it only diversifies the *slow* cycle into more reliable cashflow.

**6) Is fast payout extraction realistic? — NO.** Decisively. The edge cannot reach the profit target in
1–2 weeks within contract/drawdown limits; doing it by oversized risk fails more than it passes (ratio
0.3–0.8) and gains zero EV over the slow plan. First payouts also take ~2 months regardless. The honest
play is the **slow cushion combine → 1 MNQ funded → collect 1–3 payouts over ~2–5 months → accept blow →
restart** (Phase 42/44). Speed is a trap: same money, far more ruin.

```
Reproduction: python -m scripts.phase47   # -> docs/phase47/_phase47.json
```
