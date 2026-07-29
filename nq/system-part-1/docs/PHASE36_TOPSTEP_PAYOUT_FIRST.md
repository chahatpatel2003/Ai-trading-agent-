# Phase 36 — Topstep Payout-First Account Lifecycle Search

Different objective: not long-term strategy quality, but **can a strategy reach first payout before the
account blows**, and is the rinse-and-repeat cycle positive-EV? Built a daily/account-cycle
first-passage Monte-Carlo simulator over the FROZEN, Phase-34A-verified finalists. Rules are read from
config (printed below) and the winning-day/consistency/fee/cap assumptions are **configurable and MUST
be verified against the live Topstep dashboard** — they change often.

**Headline (honest).**

```
Payout-before-blow IS achievable — and payout-first is structurally MORE favorable than the
long-term-income objective, because the binding constraint is TIMEOUT (grinding to 5 winning days),
NOT blowing up.
  • Best config: PORTFOLIO (idb+swd) at 2 MNQ on a 150K account -> P(first payout)=62%, P(blow)=3.4%,
    median 66 trading days (~3 months), profit@payout ~$3,100.
  • 1-MNQ configs NEVER blow (DD << $4.5K MLL) but time out 40-44% -> ~56-61% payout.
  • 0 of 96 configs reach the >=70% "preferred" bar; max = 62% (barely clears the 60% minimum).
EV is MARGINALLY-TO-MODESTLY POSITIVE for the best config (~+$1,000/cycle after fees + a 90%-split/cap
overlay; ~$2,400/yr net rinse-repeat) BUT it is fragile:
  • a BAD START (first ~20 days from the worst quartile) halves payout to ~38-40%,
  • EV is thin and fee/rule-sensitive; first-payout caps + profit split are NOT precisely modeled,
  • 35-44% of cycles time out (months of fees, no payout).
Verdict: POSITIVE_EV_BUT_HIGH_RISK (portfolio 2 MNQ on 150K). NOT a clear "buy accounts" — forward-
paper and verify the exact funded-account rules first. The safe 1-MNQ version is barely-positive EV.
```

---

## 1. Rule assumptions (from `config/risk/topstep_*.yaml` — VERIFY on dashboard)

| account | size | trailing MLL | profit target | max contracts | trailing | lock |
|---|--:|--:|--:|--:|--|--|
| 50K | $50,000 | **$2,000** | $3,000 | 5 | intraday | at start |
| 100K | $100,000 | **$3,000** | $6,000 | 10 | intraday | at start |
| 150K | $150,000 | **$4,500** | $9,000 | 15 | intraday | at start |

- **Trailing floor** = `min(peak − MLL, start)` (locked at start once the buffer is banked) — verified against `TopstepAccount`.
- **Payout model (configured / tested):** **5 winning days**, winning-day threshold tested at **$150 and $200** (config uses $200), **consistency: largest winning day ≤ 50% of total profit**, account above the starting balance.
- **NOT precisely modeled (caveats → payout-dollar is an UPPER BOUND):** first-payout caps, profit split (overlaid a 90% / per-account cap as a sensitivity), $50-minimum-profit-day rule, combine→funded two-stage transition (this sim is the *funded-account payout* path; passing the combine to reach the profit target is a separate, slower first-passage). **Fees assumed** (must verify): sub ≈ $49/99/149 per month (50/100/150K) + reset on blow.

## 2. Account-lifecycle leaderboard (4,000 iid-bootstrap cycles, $200 winning-day, 150-day cap)

| strategy | sizing | account | **P(payout)** | P(blow) | P(timeout) | median days | profit@payout |
|---|--|--|--:|--:|--:|--:|--:|
| **portfolio** | **2 MNQ** | **150K** | **62.1%** | **3.4%** | 34.6% | 66 | $3,080 |
| portfolio | 3 MNQ | 150K | 59.4% | 15.9% | 24.8% | 63 | $4,659 |
| portfolio | 2 MNQ | 100K | 59.3% | 16.2% | 24.5% | 65 | $3,138 |
| portfolio | 1 MNQ | 150K | 56.2% | **0.0%** | 43.9% | 86 | $1,886 |
| breakout_acceptance | 3 MNQ | 150K | 55.8% | 5.0% | 39.2% | 71 | $2,805 |
| portfolio | 1 MNQ | 50K | 55.5% | 6.9% | 37.7% | 86 | $1,875 |
| portfolio | 3 MNQ | 100K | 52.5% | **38.2%** | 9.4% | 57 | $4,421 |
| portfolio | 5 MNQ | 150K | 49.1% | **43.7%** | 7.2% | 51 | $7,182 |
| swd | 3 MNQ | 150K | 49.9% | 3.1% | 47.0% | 79 | $3,510 |
| idb | 1 MNQ | 150K | 19.8% | 0.0% | 80.2% | — | — |

> **Sizing is the dial between timeout and blow.** Too little (1 MNQ) → 0% blow but ~44% timeout
> (can't bank 5 winning days fast enough at <1 trade/day). Too much (5 MNQ) → fast winning days but
> ~44% blow. **2 MNQ on the large 150K MLL is the sweet spot** (62% payout, 3.4% blow). idb alone is
> the worst (80% timeout) — too few trades to assemble 5 winning days; the **portfolio** (idb+swd,
> 465 trades, more winning days) is the right vehicle. At a $150 winning-day threshold the portfolio
> 1-2 MNQ/150K reaches ~61% with ~0% blow.

## 3. Daily / payout-day behavior
At 1 MNQ the finalists trade <1×/day, but a single swd/idb/breakout_acceptance **win already exceeds
$150-200**, so a winning day ≈ a winning trade. Payout-day rate ≈ trades/day × win rate ≈ **8-12% of
days** at 1 MNQ (→ ~62-86 days to bank 5 winning days). Sizing up raises the per-day $ (faster
winning days) and the drawdown together. There is **no fake high-win-rate daily target** here — the
winning days come from genuine (Phase-34A-verified) wins, not tiny-target geometry.

## 4. Robustness + EV (best configs)

| config | iid P(pay)/blow | block-boot P(pay)/blow | **bad-start P(pay)/blow** | EV/cycle pre-cap | EV/cycle (90%+cap) |
|---|--|--|--|--:|--:|
| **portfolio 2 MNQ / 150K** | 60%/3% | 61%/2% | **40%/6%** | **+$1,205** | **+$1,016** |
| portfolio 2 MNQ / 100K | 59%/18% | 62%/13% | **39%/25%** | +$1,417 | +$1,184 |
| portfolio 1 MNQ / 150K ($150) | 60%/0% | 61%/0% | 39%/0% | +$266 | +$166 |
| portfolio 1 MNQ / 150K ($200) | 55%/0% | 56%/0% | 38%/0% | +$230 | +$125 |
| breakout_acceptance 2 MNQ / 150K | 54%/0% | 56%/0% | 37%/1% | +$259 | +$156 |

- **Block bootstrap ≈ iid** → robust to autocorrelation.
- **Bad-start is the killer**: a worst-quartile opening ~20 days drops payout to **38-40%** (and blow
  to 6%/150K, 25%/100K). Sequence risk dominates the downside.
- **EV is positive but thin** and fee/rule-sensitive: the 2-MNQ/150K cycle averages ~96 days (≈4.5
  months × ~$149 sub ≈ $670 fees) and yields ~$3,100 profit on the 62% of cycles that pay → ~+$1,000
  net per cycle *after* a 90%-split and a per-account cap overlay. **Rinse-repeat (12 mo):** ~1.5
  payouts/yr → ~$4,200 gross − ~$1,800 sub ≈ **~$2,400/yr net** — *if* rules/fees hold and the edge
  persists. Halve the payout (harsher real caps) or double the fees and it approaches break-even.

## 5. Account-size comparison
| account | best config | P(payout) | P(blow) | EV/cycle (cap) | note |
|---|---|--:|--:|--:|---|
| **150K** | portfolio 2 MNQ | **62%** | **3.4%** | **+$1,016** | **best balance** — big MLL allows 2 MNQ safely |
| 100K | portfolio 2 MNQ | 59% | 16% | +$1,184 | higher EV but 16% blow (25% bad-start) — riskier |
| 50K | portfolio 1 MNQ | 56% | 7% | +$125 | MLL too tight to size up; slow, low EV; cheapest sub |

> **150K wins** for the payout-first objective: its $4,500 MLL is the only one that lets the portfolio
> run 2 MNQ (needed to beat timeout) while keeping blow ≈ 3%. 50K can't size up without blowing; 100K
> at 2 MNQ takes meaningful blow risk for marginally more EV.

## 6. Final answers
- **Is payout-before-blow possible?** **Yes** — ~62% on the best config (portfolio 2 MNQ / 150K),
  with only ~3% blow. The usual failure is *timeout* (no payout in the window), not ruin.
- **Best account?** **150K.** **Best strategy?** the **idb+swd portfolio** (enough winning days).
  **Best daily target/stop?** loose caps — these are low-frequency; "stop after the qualifying day"
  matters more than a tight target/stop. **Sizing:** 2 MNQ on 150K.
- **First-payout probability?** **~62%** (iid & block-bootstrap); **~40% on a bad start.**
- **Blow probability?** **~3.4%** (150K, 2 MNQ); ~16% on 100K, ~0% at 1 MNQ.
- **Rinse-repeat positive EV?** **Marginally yes** (~+$1,000/cycle, ~$2,400/yr net) under the stated
  fee/cap/split assumptions — but thin and fragile to bad starts and to the real (unmodeled) payout
  caps/splits.
- **Forward-paper ready?** **Yes, as a forward-paper experiment** — and the only program objective
  that comes out plausibly positive-EV. **Live-ready?** **No** — verify exact funded rules + fees,
  and the edge is modest and bad-start-sensitive.
- **Buy accounts or not?** **Not a clear yes.** It is *plausibly* positive-EV on a 150K with the
  portfolio at 2 MNQ, but the margin is thin, bad-start halves payout, and it hinges on unverified
  payout-cap/split rules. **Forward-paper one 150K-equivalent first; verify the dashboard rules; only
  scale to real account purchases if forward results match.**

## Verdict
| candidate | verdict |
|---|---|
| **portfolio (idb+swd) 2 MNQ / 150K, $200 winning-day** | **POSITIVE_EV_BUT_HIGH_RISK** — 62% payout, 3.4% blow, ~+$1,000/cycle EV; bad-start → 40% payout; fee/cap-sensitive |
| portfolio 1 MNQ / 150K (or $150 threshold) | **RESEARCH_ONLY** — 0% blow but ~56-61% payout, EV ~break-even-to-+$166; safest, slowest |
| portfolio 2-3 MNQ / 100K | **POSITIVE_EV_BUT_HIGH_RISK** (16-38% blow) — higher EV, materially higher ruin risk |
| breakout_acceptance / swd standalone | **RESEARCH_ONLY** — too few winning days alone (timeout 39-47%) |
| 5 MNQ anything / 50K 2+ MNQ | **REJECT_BLOWUP_RISK** (40%+ blow) |

### Bottom line
For the *first time in the program*, an objective comes out plausibly **positive-EV**: the payout-first
cycle on a **150K account with the idb+swd portfolio at 2 MNQ** reaches first payout ~62% of the time
with ~3% blow and ~+$1,000/cycle EV after fees. It is **not** a money machine — bad starts halve it,
the margin is thin, and it rests on unverified Topstep payout caps/splits and the modest (real, audited)
edge persisting forward. **Forward-paper it and verify the live rules before buying accounts.** This is
a genuinely better use of these small edges than trying to run them as a standalone income strategy.

```
Reproduction:
python scripts/phase36_payout.py    # rules + day-series + lifecycle MC -> docs/phase36/phase36_lifecycle.csv
# EV/fees + block-bootstrap + bad-start reproduced inline (see report)
```
