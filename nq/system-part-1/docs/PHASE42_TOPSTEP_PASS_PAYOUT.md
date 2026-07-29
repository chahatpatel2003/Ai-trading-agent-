# Phase 42 — Topstep 150K Pass + Payout System (frozen clean5+gate, Topstep wrapper only)

Optimizes **PASS probability + FUNDED PAYOUT probability**, not net ROI. Two separate modes
(Combine Pass / Funded Payout) wrapped around the **frozen** Phase-40 clean5+gate signals — the signal
engine is unchanged. Harsh fills. 150K rules: target +$9,000, trailing MLL $4,500 (intraday, locked at
start), max 15 contracts, consistency 50% (best day < ½ of total profit). Monte-Carlo over contiguous
real day-windows. Scripts: `scripts/phase42.py {combine,refine,funded,lifecycle,booster}`.

**Headline.**

```
THE FIX FOR THE COMBINE IS UNDER-SIZING, NOT OVER-SIZING. "cushion123" — start 1 MNQ, scale to 2 only
after +$2,000 cushion, 3 after +$4,000 — turns the combine from a coin-flip into pass:fail ~4:1
(pass 30% / fail 7% in 90 days). Phase 41's 2->3->4 mode failed because it started big; starting at 1
MNQ when you have no buffer (most exposed to the trailing DD) and scaling only with house money is the
whole trick. Every plan that starts >=2 MNQ has fail > pass.
FUNDED MODE MUST BE 1 MNQ. At 1 MNQ: 82% reach a payout within 6 months (avg first payout ~48 days),
~$3,700 withdrawn / 6mo -- BUT only ~62% of funded accounts survive 6 months. At 2 MNQ survival
collapses to 23%. The funded account is a fragile early-extraction machine, not a perpetual one.
BOOSTERS HURT. Adding momentum/beta sleeves (gap_go, volhi_orb, dow_tue) raises drawdown -> combine
ratio 5.3->2.0, funded survival 64%->59%. Drawdown is the binding constraint; clean5 stays frozen.
LIFECYCLE EV ~ +$1,549 per pass+6mo-funded cycle at a $165/mo combine fee; breakeven fee ~$290/mo.
Positive but thin and fee-sensitive.
```

Verdict: **CONDITIONAL YES** — tradeable on Topstep with positive EV, but only with a cheap combine
fee (≤ ~$290/mo), strict under-sizing, and early payout extraction. It is a low-return disciplined
grind with real account-death risk, not a wealth machine.

---

## 1-3. Combine Pass Mode matrix (frozen clean5+gate, harsh, 90-day horizon, 20k sims)
Selected variants (full set in `docs/phase42/_combine.json`):

| variant | pass30 | pass60 | pass90 | fail90 | **ratio** | med days | worst DD | worst day | best day | consist viol | alive@90 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| size flat1 | 1.2% | 3.5% | 6.6% | 5.3% | 1.24 | 58 | 4,619 | −1,136 | 3,321 | 0% | 88% |
| size flat2 | 7.3% | 18% | 28% | 33% | 0.86 | 45 | 8,031 | −2,273 | 6,642 | 1% | 39% |
| size flat3 | 13% | 26% | 37% | 45% | 0.82 | 44 | 8,997 | −3,409 | 9,964 | 4% | 19% |
| size flat5 | 22% | 34% | 35% | 63% | 0.56 | 25 | 12,631 | −5,681 | 16,606 | 12% | 1% |
| size cushion234 (P41) | 12% | 26% | 38% | 35% | 1.09 | 45 | 9,128 | −4,045 | 13,285 | 8% | 28% |
| **size cushion123** | 2.4% | 15% | 28% | 7.4% | **3.8** | 58 | 8,244 | −2,909 | 9,964 | 5% | 64% |
| mixed MNQ/NQ | 16% | 32% | 39% | 51% | 0.77 | 36 | 23,506 | −9,364 | 33,212 | 12% | 10% |
| A-only @flat2 | 1.2% | 9% | 21% | 18% | 1.19 | 64 | 6,793 | −2,273 | 7,972 | 5% | 61% |
| best sleeve (ir90) @flat2 | 0% | 0% | 0% | 0% | — | — | 1,860 | −999 | 3,250 | 0% | 100% |
| trend-day-only @flat2 | 3.4% | 10% | 15% | 32% | 0.47 | 43 | 8,600 | −2,273 | 6,642 | 1% | 53% |
| pstop +1000 @flat2 | 3.6% | 11% | 20% | 37% | 0.53 | 53 | 7,550 | −2,273 | 7,972 | 1% | 44% |
| lstop −750 @flat2 | 8.1% | 19% | 29% | 32% | 0.93 | 43 | 8,133 | −1,493 | 6,642 | 1% | 39% |

**Refinements on the cushion family (the winners):**

| variant | pass30 | pass60 | pass90 | fail90 | **ratio** | med days | worst day | consist viol | alive@90 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cushion123 (base) | 2.4% | 15% | 28% | 7.4% | 3.8 | 58 | −2,909 | 5% | 64% |
| **cushion123 + lstop −750** | 3.1% | 16% | 30% | 7.2% | **4.15** | 56 | −1,910 | 5% | 63% |
| cushion1234 + lstop −1000 | 4.9% | 17% | 31% | 9.0% | 3.44 | 53 | −1,910 | 4% | 60% |
| cushion123 + A-only | 0% | 1% | 8% | 0.9% | 9.07 | 76 | −3,409 | 7% | 91% |
| cushion123 + skip-weak | 0% | 3% | 8% | 0.5% | 17.5 | 66 | −1,823 | 7% | 91% |

> Key reads: **only the cushion-start family clears the kill criteria** (ratio ≥ 1.5, fail < pass).
> Filters like A-only / skip-weak push the ratio sky-high (9–17) but starve the pass rate (8%) — too
> slow. The daily **loss stop −$750** is a free improvement (caps worst day −2,909 → −1,910, lifts
> ratio 3.8 → 4.15). No 30-day pass is realistic (best ~5%); this is a **45–90 day** combine.

## 4. Funded Payout Mode (post-pass, 126-day / 6-month horizon, 20k sims)
Model: winning day = +$150; payout when ≥5 winning days + positive profit + consistency (best day ≤ ½
profit); withdraw profit down to start buffer, $2,500/request cap; trailing MLL locks at start once
+$4,500 reached.

| funded plan | payout ≤3mo | payout ≤6mo | avg 1st payout | payouts/6mo | $ withdrawn/6mo | **survive 6mo** | worst DD |
|---|--:|--:|--:|--:|--:|--:|--:|
| **1 MNQ grinder** | 58% | 82% | 48 d | 1.81 | $3,660 | **63%** | $3,750 |
| 1 MNQ + lstop −750 | 59% | 82% | 48 d | 1.83 | $3,705 | 62% | $3,772 |
| 2 MNQ grinder | 55% | 69% | 40 d | 1.65 | $3,984 | **23%** | $5,339 |
| 2 MNQ + lstop −1000 | 56% | 70% | 40 d | 1.74 | $4,205 | 24% | $5,397 |

> **1 MNQ is mandatory in funded.** At 2 MNQ the median drawdown exceeds the $4,500 MLL → 77% of
> accounts die within 6 months. Even at 1 MNQ, ~38% die within 6 months — payouts arrive early
> (avg 48 days, 58% within 3 months), so the rational play is **extract every eligible payout
> immediately**, banking 1–2 withdrawals before the account's ~38%/6mo death risk catches up.

## 5. Combined lifecycle EV (cushion123 combine → 1 MNQ funded → withdraw early)
| input | value |
|---|--:|
| Combine pass within 126 days | 39.2% |
| Combine fail within 126 days | 12.0% |
| Combine still grinding at 126 days | 48.8% |
| Median days to pass | 77 |
| Avg months paying combine fee | 4.79 |
| Funded E[withdrawn] over 6mo (1 MNQ) | $3,713 |
| Funded survive 6mo | 62% |
| **Assumptions** | combine fee **$165/mo**, activation **$149** |
| Expected combine fees to reach a pass | **$2,016** |
| Funded EV (6mo, after activation) | $3,564 |
| **EV per combine attempt** | **+$607** |
| **EV per full lifecycle (pass + 6mo funded)** | **+$1,549** |
| **Break-even combine fee** | **$292/mo** |

## 6. Boosters beyond clean5 — tested, rejected (apples-to-apples, Phase-38 source)
| sleeve set | combine pass90 | combine fail90 | ratio | funded survive6mo | $ withdrawn/6mo |
|---|--:|--:|--:|--:|--:|
| **clean5 (base)** | 31% | 5.9% | **5.29** | **64%** | $3,826 |
| clean5 + gap_go | 25% | 12.6% | 1.96 | 59% | $3,447 |
| clean5 + gap_go + dow_tue | 26% | 11.7% | 2.23 | 61% | $3,457 |
| clean5 + volhi_orb | 26% | 11.6% | 2.26 | 61% | $3,789 |

> Every booster **raised drawdown and lowered both the combine ratio and funded survival.** Drawdown
> is the binding constraint, so more (even higher-edge) signals hurt. **No booster is added.**

## Kill-criteria scorecard
| criterion | result | verdict |
|---|---|---|
| fail probability > pass probability | combine 7% vs 30% | **PASS** |
| pass/fail ratio < 1.5 | 4.15 (cushion123+lstop750) | **PASS** |
| payout probability too low | 82% reach payout / 6mo | **PASS** |
| works only by oversized risk | works by *under*-sizing (1→2→3) | **PASS** (inverted) |
| best day violates consistency | 5% of combine accounts blocked | ⚠️ low, watch |
| max DD regularly approaches $4,500 | funded median DD $3,750; 38% die/6mo | ⚠️ **funded weak spot** |
| funded cannot survive after payout | survives to 1st payout (82%); 38% die by 6mo | ⚠️ partial |

## Final ranking — the best Topstep 150K plan

**1) Best Combine pass plan — `cushion123 + daily loss stop −$750`**
Start **1 MNQ** per signal; scale to **2 MNQ after +$2,000** cushion, **3 MNQ after +$4,000**; never
more. Daily loss stop **−$750**. The 5 frozen sleeves + neutral-trend gate, ≤3 trades/day.
→ pass 30% / fail 7% in 90 days (**ratio 4.15**), median **56 days**, worst day −$1,910, ~5% consistency
blocks. *(Faster alt: `cushion1234 + lstop −1000` → median 53 d, pass 31%, ratio 3.44.)*

**2) Best Funded payout plan — `1 MNQ flat grinder, extract early`**
Drop to **1 MNQ** on funding; take all gated signals; **withdraw the moment eligible** (5 winning days
≥$150 + consistency). → 82% reach a payout within 6mo, ~1.8 payouts, ~$3,700 withdrawn; ~62% survive
6mo. **Never run funded at 2 MNQ** (23% survival).

**3) Best full-lifecycle plan**
Buy combine → run cushion123+lstop750 → on pass, switch to 1 MNQ funded → withdraw early/often → if the
funded account dies (~38%/6mo), re-buy a combine. EV ≈ **+$1,549 per pass+6mo cycle** (+$607 per combine
attempt) at $165/mo fee; **break-even at ~$292/mo**.

**4) Is this actually worth trading on Topstep? — CONDITIONAL YES (thin).**
- **Positive EV only if the combine fee is ≤ ~$290/mo.** Undiscounted 150K combines run ~$300–360/mo
  → near/below break-even; use discounted resets to stay profitable. The edge is real but small, and
  Topstep's fees eat most of it.
- It passes the "no oversized risk" test by *inverting* it — the whole edge is **under-sizing** (1 MNQ
  until you have house money). Anyone sizing up to pass faster will fail more than they pass (Phase 41).
- The **funded account is fragile** (~38% die within 6 months; DD regularly nears the $4,500 MLL). The
  plan only works because payouts arrive early — you bank 1–2 withdrawals before the death risk hits.
- **Realistic for:** a disciplined trader who values funded income, executes mechanically, and uses
  cheap/discounted combines. **Not** a wealth machine — expect ~$1.5k EV per successful funded cycle,
  with patience (≈2–3 months to pass) and a real chance of repeated combine fees.

### Bottom line
A genuine, rules-based Topstep 150K plan exists: **cushion123 combine (ratio ~4:1) → 1 MNQ funded
grinder, extract payouts early.** It is positive-EV on conservative (harsh-fill) assumptions, but
thin and fee-sensitive, and the funded account must be treated as a fragile early-extraction vehicle,
not a perpetual income stream. The single most important rule is the counter-intuitive one: **size up
only with house money — never start big.**

```
Reproduction:
python -m scripts.phase42 combine    # combine matrix      -> docs/phase42/_combine.json
python -m scripts.phase42 refine     # cushion refinements
python -m scripts.phase42 funded     # funded payout mode
python -m scripts.phase42 lifecycle  # combined EV          -> docs/phase42/_lifecycle.json
python -m scripts.phase42 booster    # booster search (rejected)
```
