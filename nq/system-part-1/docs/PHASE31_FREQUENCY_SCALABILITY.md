# Phase 31 — Trade-Frequency + Scalable-Risk Requirement (reject low-frequency 1-MNQ edges)

Business correction: a strategy that trades ~1×/week, only works at 1 MNQ, and can't make meaningful
monthly profit within a 100K / **$3,000** account is **not useful**. Phase 31 re-judges everything by
**frequency AND scalability AND edge**, and searches for setups that are all three. No new rules were
optimized — this re-screens the full Phase-29 + Phase-30 universe (1,317 candidates, all FROZEN) and
builds multi-sleeve portfolios.

**Headline result (exact).**

```
NO candidate is SCALABLE_FREQUENT. Across 1,317 tested candidates:
  • 0 have >=2 trades/week AND PF>=1.4 (best PF among >=2/week setups = 1.244).
  • 0 make >=$500/month at $3K-safe sizing (BEST in the whole universe = $235/month).
  • High-frequency families (7-21 trades/wk) have 1-MNQ drawdowns of $4,000-5,700 -> can't even
    trade 1 MNQ on a $3K account.
  • Multi-sleeve portfolios reach the frequency bar (3.8 trades/wk, PF ~1.40) but their Monte-Carlo
    p99 drawdown is $5,400-6,200 (~20-27% breach at 1 MNQ/sleeve) -> fail $3K.
The 100K/$3K + $500/mo + >=2/wk + PF>=1.4 goal is NOT achievable with the NQ 5m intraday edges found.
The $3K drawdown is the binding constraint (it requires return/maxDD >= ~11; best achieved = 8).
```

---

## Part 1 — Reclassification of the Phase 30 finalists (exact)

| finalist | trades | /week | /month | max safe MNQ (MC p99<$3K) | $/month at safe size | $3K breach prob (block-boot) | classification |
|---|--:|--:|--:|--:|--:|--:|---|
| swd_clean_path \| RTH \| toPOC | 244 | **0.86** | 3.7 | 1 | **$117** | ~1% | **RESEARCH_ONLY_1_MNQ_EDGE** |
| idb_base \| RTH \| eod | 221 | **0.85** | 3.3 | 1 | **$123** | ~11% (borderline) | **RESEARCH_ONLY_1_MNQ_EDGE** |
| portfolio (idb + swd, 1 each) | 465 | **1.64** | 7.0 | <1 (MC p99 $4,944) | **$240** | **~21%** | **REJECT_NOT_SCALABLE** for the goal |

All three fail the new bars: < 2 trades/week, < $500/month at $3K-safe size, and (for the portfolio)
> 5% breach probability even at 1 MNQ/sleeve. They remain valid *research* edges, not a business.

---

## Part 2 — Full-universe screen (1,317 candidates)

Scalability model = fixed contracts (N MNQ): drawdown scales linearly, so `max_safe_N =
floor($3,000 / intraday-maxDD_1MNQ)`. Monthly $ uses the full 66-month calendar span (dead months = $0).

| classification | count |
|---|--:|
| REJECT_NO_EDGE (PF < 1.0) | 934 |
| FREQUENT_BUT_NO_EDGE (≥2/wk, 1.0 ≤ PF < 1.4) | 368 |
| REJECT_NOT_SCALABLE (PF ≥ 1.4 but < $500/mo at safe size) | 10 |
| RESEARCH_ONLY_1_MNQ_EDGE (PF ≥ 1.4, max safe = 1 MNQ) | 5 |
| **SCALABLE_FREQUENT_FORWARD_PAPER** | **0** |
| SCALABLE_BUT_TOO_LOW_FREQUENCY | 0 |

**The frequency ↔ edge frontier (why nothing qualifies):**

| regime | example | trades | /wk | PF | 1-MNQ intraday DD | max safe MNQ | $/mo at safe |
|---|---|--:|--:|--:|--:|--:|--:|
| highest edge | swd_clean_path\|RTH\|toPOC | 244 | 0.86 | **1.535** | $1,223 | 2 | $234 |
| best ≥300-trade edge | swd_base\|RTH\|toPOC | 302 | 1.07 | 1.433 | $1,363 | 2 | $231 |
| best ≥2/wk by PF | ema_cross_21_50\|NY_AM\|eod | 992 | 3.5 | **1.244** | $4,126 | **0** | $0 |
| high frequency | macd_hist_expand\|RTH\|eod | 4,138 | 14.6 | 1.132 | $4,140 | **0** | $0 |
| very high frequency | fvg_cont\|RTH\|rr2 | 5,937 | 20.9 | 1.062 | $4,642 | **0** | $0 |

- **Edge requires patience:** every PF ≥ 1.4 setup (15 of them) trades ≤ **1.07/week** and makes
  ≤ **$235/month** at $3K-safe size.
- **Frequency requires accepting noise:** every ≥ 2/week setup has **PF ≤ 1.244**, and the genuinely
  high-frequency ones have 1-MNQ drawdowns **> $3,000** (so `max_safe_N = 0` — not even 1 MNQ fits).
- **Best `$/month at safe size` in the entire universe = $234.75.** Target was ≥ $500.

**The binding math.** $500/mo over 66 months on a $3K-max-DD account ⇒ total $33K ⇒ **return/maxDD ≥ 11**.
The best return/(intraday maxDD) anywhere in the universe is **8.0**, and the high-ratio ones are
either too rare to be frequent or have absolute DD too large to scale. The constraint set is
over-determined for a $3K account.

---

## Part 3 — Multi-sleeve frequent portfolios (the last shot)

Diverse FROZEN sleeves, 1 MNQ each, concurrency-aware intraday drawdown + day-block-bootstrap MC
(1,000 sims). `phase31_portfolios.csv`.

| portfolio | sleeves | trades | /week | PF | net (1×) | $/mo (1×) | intraday DD (1×) | **MC p99 DD (1×)** | **breach prob (1×)** | max safe MNQ (MC) | $/mo @ safe |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| swd + idb (Phase 30) | 2 | 465 | 1.64 | 1.481 | 15,834 | 240 | 2,172 | **4,944** | 21% | 0 | $0 |
| swd-family multi | 3 | 1,030 | 3.63 | 1.374 | 13,187 | 200 | 1,753 | **5,283** | 19% | 0 | $0 |
| **diverse3** (swd+idb+sweepALL) | 3 | 1,066 | **3.76** | **1.396** | 19,071 | 289 | 2,317 | **5,412** | 27% | 0 | $0 |
| diverse4 (+trend) | 4 | 2,070 | 7.30 | 1.247 | 27,552 | 417 | 3,737 | **9,144** | 87% | 0 | $0 |
| freq sweeps | 3 | 1,100 | 3.88 | 1.266 | 8,257 | 125 | 2,117 | **6,204** | 30% | 0 | $0 |

- Portfolios **do** clear the frequency bar (3.6–7.3 trades/week) and the best reaches **PF 1.396** —
  but **none** survive $3K: MC p99 drawdown is **$4,900–9,100**, with **19–87% breach probability at
  just 1 MNQ per sleeve.** Scaling up is impossible; even 1× fails the < 5% breach rule.
- Diversification raised aggregate return/DD to ~8.2 (diverse3) — better than any single sleeve, but
  still far below the ~11 needed, and the **realistic (Monte-Carlo) drawdown is ~2.3× the historical**
  because frequency adds concurrent positions and order-dependent risk.

---

## Part 4 — Final classifications

| candidate / group | trades | /wk | PF | $/mo @ safe | $3K breach | **classification** |
|---|--:|--:|--:|--:|--:|---|
| swd_clean_path\|RTH\|toPOC (1 MNQ) | 244 | 0.86 | 1.535 | $117 | ~1% | RESEARCH_ONLY_1_MNQ_EDGE |
| swd_base / swd_at_edge\|RTH\|toPOC | ~300 | 1.07 | 1.43–1.45 | ~$230 | (2 MNQ unsafe by MC) | RESEARCH_ONLY_1_MNQ_EDGE |
| idb_base\|RTH\|eod (1 MNQ) | 221 | 0.85 | 1.44 | $123 | ~11% | RESEARCH_ONLY_1_MNQ_EDGE |
| portfolio swd+idb / diverse3 | 465–1066 | 1.6–3.8 | 1.40–1.48 | $0 safe | 21–27% | **REJECT_NOT_SCALABLE** |
| ema_cross / supertrend / macd / ema_stack / fvg (≥2/wk) | 600–5,900 | 2–21 | 1.06–1.24 | $0 | n/a (DD>$3K @1MNQ) | **FREQUENT_BUT_NO_EDGE** (368) |
| everything PF < 1.0 | — | — | <1.0 | — | — | **REJECT_NO_EDGE** (934) |
| **anything** | — | — | — | — | — | **SCALABLE_FREQUENT_FORWARD_PAPER = NONE (0)** |

### Required per-candidate metrics
Full tables (total trades, trades/yr/mo/wk, longest no-trade gap, $/mo at safe sizing, max DD,
intraday DD, MC p99 DD, breach prob, PF, win rate, expectancy, max losing streak, 100K/$3K pass/fail,
scalable pass/fail, classification) are in **`docs/phase31/phase31_screen.csv`** (1,317 rows) and
**`docs/phase31/phase31_portfolios.csv`**.

---

## Part 5 — Honest verdict + the path forward

**There is no strategy in the tested NQ 5m universe that is simultaneously frequent (≥2/week),
edge-bearing (PF ≥ 1.4), scalable to ≥2 MNQ within a $3K drawdown, and worth ≥$500/month.** Zero of
1,317 candidates and zero of 5 portfolios qualify. This is not a tuning gap — it is the
**frequency↔edge↔drawdown trilemma**: NQ intraday edge concentrates in *patient* setups (PF 1.4–1.5,
~1/week), while frequency only comes from noisier setups (PF ≤ 1.24) whose drawdowns already exceed
$3K at 1 MNQ.

**To make this a real business, one constraint must move:**
1. **Loosen the drawdown** to ~$5–6K. Then swd/portfolio scale to 2–3 MNQ and ~$350–700/month become
   reachable (return/DD ~8 supports it). This is the single highest-leverage change.
2. **Accept ~$200–300/month** at the strict $3K cap using the RESEARCH_ONLY edges at 1 MNQ.
3. **Find a genuinely higher-frequency, small-DD edge** — which the 5m families don't contain. That
   needs **1m/3m bars** (not yet built for 2025-H2→2026) or **order-flow/MBP1** (data ends 2025-05).
   This is the only route that could deliver frequency *and* a $3K-survivable drawdown, and it is
   unproven.

**Recommendation:** do **not** deploy any of these to a $3K/100K account expecting $500+/month — none
support it. The realistic next experiment is to **build 1m/3m bars for the full history and re-run the
frequency-aware factory there**, and in parallel **re-spec the account to ~$5–6K drawdown** where the
existing edges become a viable ~$400–700/month business at 2–3 MNQ.

```
Reproduction:
python scripts/phase31_screen.py       # 1,317-candidate frequency+scalability screen -> phase31_screen.csv
python scripts/phase31_portfolio.py    # multi-sleeve portfolios + concurrency MC -> phase31_portfolios.csv
```
