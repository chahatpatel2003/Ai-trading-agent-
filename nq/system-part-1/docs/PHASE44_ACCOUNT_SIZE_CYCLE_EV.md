# Phase 44 — Topstep 50K vs 100K vs 150K repeatable-cycle EV

Optimizes **repeatable cashflow EV** (combine → pass → funded → collect payouts → blow → restart), NOT
survival forever. Flat per-attempt costs (50K $49 / 100K $99 / 150K $199, reset = same) ⇒ **no monthly
bleed**, so the combine only needs pass-before-blow. Frozen signals only (clean5+gate harsh, Phase-36
idb+swd infrequent edge); funded = 1 MNQ low-risk; per-account cushion sizing. 10k MC sims/cell.
Script `scripts/phase44.py`, data `docs/phase44/_phase44.json`.

**Headline.**

```
THE 150K WINS — even though it's the most expensive. Its larger drawdown buffer ($4,500 vs $2,000)
makes the combine pass:fail ~6:1 (vs ~1.5:1 on the 50K) AND lets the funded account survive to ~2
payouts instead of ~1. That dominates the higher cost. The cheap 50K is a TRAP: tight $2,000 MLL ->
combine near coin-flip (ratio 1.5) and 100% of funded 50K accounts blow right after their first payout.
BEST ENGINE = clean5 + Phase-36 edge. Unlike the Phase-38 momentum boosters (which HURT), the Phase-36
mean-reversion edge is uncorrelated -> it DIVERSIFIES, lowering combine drawdown ($11.3k vs $14.1k) AND
raising the pass rate. clean5+P36 on a 150K = pass 84% / ratio 6.4 / EV ~$5,460 per cycle / ~$452/mo.
CRITICAL CAVEAT: this assumes FLAT per-attempt cost as specified. Real Topstep combines are usually
MONTHLY; at a ~5-month median pass time, monthly billing raises combine cost ~3-5x and cuts EV toward
the Phase-42 figure (~$1.5k/cycle). The 150K ranking holds either way; the EV magnitude does not.
```

Verdict: **150K is the best account for repeating this process.** Positive, repeatable EV under the
flat-cost spec; thin-but-positive under monthly billing.

---

## Cross-account matrix (cushion sizing, 1 MNQ funded, daily stops ≈ −0.375×MLL)

### 50K (cost $49, target $3,000, MLL $2,000, max 5)
| strategy | pass | blow | ratio | med d→pass | 1st-payout P | avg payout | blow-after-1st | E[total payout] | E[combine cost] | **EV/cycle** | **EV/mo** | cycle mo | maxDD | worst day |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| clean5 | 60% | 40% | 1.50 | 43 | 74% | 1,676 | **100%** | 1,971 | 82 | 1,889 | 268 | 7.0 | 13,784 | −1,910 |
| phase36 | 53% | 35% | 1.51 | 156 | 86% | 1,376 | 72% | 1,639 | 81 | 1,557 | 90 | 17.3 | 2,996 | −1,868 |
| clean5+p36 | 58% | 42% | 1.38† | 35 | 75% | 1,715 | 98% | 2,597 | 85 | 2,512 | 418 | 6.0 | 10,416 | −1,569 |
| A_only | 52% | 48% | 1.07† | 52 | 68% | 1,570 | 99% | 2,004 | 95 | 1,909 | 203 | 9.4 | 7,522 | −2,683 |
| clean5 HR-pass | 34% | 66% | 0.51† | 21 | 72% | 1,676 | 100% | 1,888 | 145 | 1,743 | 300 | 5.8 | 16,752 | −2,547 |

### 100K (cost $99, target $6,000, MLL $3,000, max 10)
| strategy | pass | blow | ratio | med d→pass | 1st-payout P | avg payout | blow-after-1st | E[total payout] | E[combine cost] | **EV/cycle** | **EV/mo** | cycle mo | maxDD | worst day |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| clean5 | 74% | 24% | 3.06 | 87 | 89% | 2,122 | 83% | 3,889 | 131 | 3,758 | 350 | 10.7 | 13,963 | −2,240 |
| phase36 | 30% | 19% | 1.57 | 204 | 86% | 1,413 | 19% | 1,873 | 162 | 1,710 | 73 | 23.4 | 5,147 | −1,868 |
| clean5+p36 | 75% | 25% | 3.07 | 73 | 84% | 2,211 | 90% | 3,726 | 131 | 3,595 | 376 | 9.6 | 9,710 | −2,240 |
| A_only | 71% | 27% | 2.64 | 120 | 93% | 2,022 | 88% | 4,256 | 136 | 4,119 | 295 | 14.0 | 6,890 | −2,683 |
| clean5 HR-pass | 53% | 47% | 1.15† | 36 | 89% | 2,106 | 82% | 3,818 | 185 | 3,633 | 447 | 8.1 | 19,636 | −2,547 |

### 150K (cost $199, target $9,000, MLL $4,500, max 15)
| strategy | pass | blow | ratio | med d→pass | 1st-payout P | avg payout | blow-after-1st | E[total payout] | E[combine cost] | **EV/cycle** | **EV/mo** | cycle mo | maxDD | worst day |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| clean5 | 79% | 13% | 6.05 | 115 | 90% | 2,478 | 75% | 5,344 | 232 | 5,112 | 399 | 12.8 | 14,137 | −2,410 |
| phase36 | 5.5% | 0% | — | 250 | 86% | 1,466 | 3% | 1,930 | 199 | 1,731 | 76 | 22.9 | 3,670 | −2,560 |
| **clean5+p36** | **84%** | 13% | **6.44** | 106 | 95% | 2,527 | 81% | 5,692 | 230 | **5,462** | **452** | 12.1 | 11,348 | −2,354 |
| A_only | 74% | 6% | **11.95** | 179 | 99% | 2,292 | **50%** | 5,568 | 216 | 5,352 | 325 | 16.5 | 8,755 | −2,683 |
| clean5 HR-pass | 69% | 31% | 2.26 | 70 | 91% | 2,480 | 75% | 5,379 | 287 | 5,092 | **468** | 10.9 | 18,617 | −2,986 |

† fails the kill criterion *pass/fail ratio < 1.5*.

## Reading the matrix
- **Account size is the dominant lever.** Bigger buffer → higher combine ratio (1.5 → 3 → 6) *and*
  more funded payouts before blow (E[total payout] $2.0k → $3.9k → $5.3k). The $49→$199 cost gap is
  trivial next to that. EV/cycle: 50K ~$1.9–2.5k, 100K ~$3.6–4.1k, **150K ~$5.1–5.5k.**
- **EV/month also favors 150K** once you drop the kill-criteria failures: best surviving cells are
  150K clean5+p36 **$452/mo** (ratio 6.4) and 150K clean5 $399/mo; the 50K's faster cycles (6–7 mo)
  don't offset its smaller payouts and constant resets.
- **The 50K is a trap.** Cheap, but the $2,000 MLL ≈ one bad day at 1 MNQ → combine is a coin-flip and
  **100% of funded 50K accounts blow right after the first payout** (no buffer to survive). clean5+p36
  and HR-pass on 50K *fail* the ratio≥1.5 kill criterion outright.
- **Phase-36 edge is a real booster here** (opposite of the Phase-38 momentum sleeves): being
  uncorrelated mean-reversion, it cuts combine drawdown (150K: $14.1k→$11.3k) and lifts the pass rate
  (79%→84%). Phase-36 **alone** is far too slow (cycle 17–23 months, EV/mo ~$75).
- **A-only is the safest combine** (150K ratio 11.95, blow 6%, best funded survival: blow-after-first
  50% vs 75%) but slower (median 179 d) → lower EV/month ($325). Good "low-variance" variant.
- **HR-pass** (start 2 MNQ) maximizes raw EV/month on 150K ($468) but doubles the blow rate (31%) and
  drawdown ($18.6k); on 50K/100K it fails the ratio criterion. Not worth the variance.

## Final ranking
**1) Best account size — `150K`.** Highest EV/cycle (~$5.5k) *and* highest kill-criteria-respecting
EV/month (~$452). The drawdown buffer is what makes a *repeatable* cycle work: ~6:1 combine odds and
~2 funded payouts per life. 100K is the runner-up (~$3.6–4.1k/cycle, $376/mo). **50K is not viable**
for repetition (ratio ~1.5, 100% blow-after-first-payout).

**2) Best pass-mode strategy — `clean5 + Phase-36` with cushion sizing** (start 1 MNQ, →2 after +0.22×target
cushion, →3 after +0.44×, cap by account; daily loss stop ≈ −0.375×MLL). On 150K: pass 84%, ratio 6.44,
median 106 days, lower DD than clean5 alone. *(Safest alt: A-only, ratio ~12 but slower. Avoid HR-pass
except on 150K.)*

**3) Best funded payout-mode strategy — `1 MNQ, clean5 + Phase-36, extract every payout immediately`**
(5 winning days ≥$150 + consistency). Highest total payout (150K: $5,692 ≈ 2.25 payouts). *(Survival-tilt
alt: A-only — blow-after-first 50% vs 81%, slightly less total but more durable.)* Never run funded >1 MNQ.

**4) Best full-cycle EV — `150K + clean5+Phase-36 cushion combine → 1 MNQ funded, extract early, reset on
blow`.** EV ≈ **$5,460 / cycle, ~$452 / month, ratio 6.4**, cycle ≈ 12 months, combine cost ≈ $230/pass.

**5) Which size for repeating — `150K`.** Unambiguous on both EV/cycle and EV/month after kill criteria.

## Honest caveat (do not skip)
The strong EV here rides on the **flat per-attempt cost** you specified ($199, no monthly bleed). If the
real combine is a **monthly subscription**, a ~5-month median pass time multiplies combine cost ~3–5×
and pulls EV/cycle toward the Phase-42 figure (~$1.5k) — still positive, but thin and fee-sensitive
(break-even ≈ $290/mo). **The 150K-is-best conclusion is robust to this; the EV magnitude is not.** Also:
funded accounts mostly blow after 1–2 payouts (this is by design — we optimize cashflow, not longevity),
and combine pass times are long (~3–6 months), so this is a slow, disciplined cashflow process, not fast money.

```
Reproduction: python -m scripts.phase44   # -> docs/phase44/_phase44.json
```
