# Phase 11 — Confirmed-Enough Strategy Search (reserved-lockbox protocol)

**Goal:** find an NQ-only strategy *historically confirmed enough to move to
paper/shadow trading* — using, for the first time, a **clean final lockbox reserved
BEFORE searching**. This is the protocol the earlier phases lacked: Phase 9 searched
straight through 2026 (no holdout), and Phase 10's lockbox was therefore contaminated.

**Verdict: no paper-ready strategy. 0 ACCEPT_FOR_PAPER, 1 REJECT, 4 NEEDS_MORE_DATA.**
The new trend-day family was the strongest thing any phase has produced on validation
(+$2,350 on 2024, 100% positive years, validation DSR 0.62) — but on the **untouched
2025-2026 lockbox** it either traded too rarely to confirm (15m) or failed cost
stress (5m). The reserved lockbox made `ACCEPT_FOR_PAPER` genuinely reachable, and
nothing earned it. Not forced.

---

## The split (lockbox reserved before any search)

| Window | Dates | Use |
|---|---|---|
| Search / train | 2021-01-01 → 2023-12-31 | fit/score candidates |
| Validation | 2024-01-01 → 2024-12-31 | OOS gate for freezing |
| **Final lockbox (untouched)** | 2025-01-01 → 2026-06-30 | one-shot confirmation |

The search command loads data **only through 2024-12-31** — the 2025-2026 lockbox is
never read until candidates are frozen. The lockbox run verified `pristine=True`
(lockbox start is strictly after every search run's end), so `ACCEPT_FOR_PAPER` is a
reachable verdict, not a foregone `NEEDS_PAPER_LOCKBOX`.

## What was searched (new NQ families, not retired / not Phase-9)

Phase 9 was momentum-only; Phase 11 adds the missing classes:

- **`trend_day`** — trend-day detection + continuation: HTF-4h bull, expanding vol,
  price above session VWAP and breaking the opening range; ride with a wide target.
- **`range_fade`** — *mean reversion*: on a low-vol day after a range/compression
  prior day, fade a poke through the prior-day extreme that reclaims.
- **`vwap_revert`** — *mean reversion*: in a low-vol regime, fade an over-extension
  away from session VWAP back toward it.

All NQ-only, trades-derivable (keeps the lockbox clean), NY-AM, asymmetric (tight
stop / wider target), run on 5m+15m × 50K/100K/150K × {MNQ-only, adaptive} ×
risk {$50,$100,$150,$200}: **480 candidates**, strict locked-buffer Topstep simulator,
diagnostics-first prefilter.

---

## Validation (2024) leaderboard — trend_day wins, mean-reversion fails

- **`trend_day` dominated**: 30 of 480 candidates cleared the full 2024 economic
  gauntlet (positive OOS, ruin ≤ 10%, cost-stress, Topstep, walk-forward). All 30 are
  `trend_day`. Best: **15m, risk $100 — +$2,350 OOS-2024, ruin 0%, cost-pass,
  walk-forward 75%, 100% positive years, validation DSR 0.62** (the strongest single
  validation result in the whole project).
- **`range_fade` and `vwap_revert` did not survive validation** — the mean-reversion
  families were rejected or trade-starved on 2024. Fading NQ in the NY-AM session is
  not an edge here.
- Post-DSR (n=80) accepts: **0** (PBO 66-69% at 5m, 21% at 15m), as expected pre-freeze.

## Frozen hypotheses (top validation survivors, selected before the lockbox)

| Name | Family | TF | Acct | Risk | Val OOS (2024) | Val DSR |
|---|---|---|---|---|---:|---:|
| trend_day_15m_{50k,100k,150k}_r100 | trend_day | 15m | 50/100/150K | $100 | +2,350 | 0.62 |
| trend_day_5m_150k_r100 | trend_day | 5m | 150K | $100 | +1,239 | 0.42 |
| trend_day_15m_150k_r150 | trend_day | 15m | 150K | $150 | +2,057 | 0.27 |

Frozen into a hash-verified registry (`phase11_frozen.json`); each reproduces its
exact validation strategy fingerprint (no tuning after freeze).

---

## Final lockbox (untouched 2025-2026) — the decision

| Frozen hypothesis | Lockbox trades (OOS) | OOS net | Ruin | Cost-stress | WF+ | Pos-yr | DSR(n=5) | Verdict |
|---|---:|---:|---:|:--:|---:|---:|---:|---|
| trend_day 15m (50/100/150K) | **0** | — | 0% | pass | 100% | 100% | 0.49 | NEEDS_MORE_DATA |
| trend_day 5m 150K | 48 | +$11 | 0% | **FAIL** | 53% | 50% | 0.06 | **REJECT** |
| trend_day 15m 150K (r$150) | 0 | — | — | fail | 0% | 0% | 0.13 | NEEDS_MORE_DATA |

**Why each failed:**

- **trend_day 15m** (the validation star) fired only **11 times in the entire
  1.5-year lockbox and 0 times in the 2026-H1 OOS gate.** A 15m trend-day break with
  HTF-bull + vol-expansion confluence is a *rare* event (~7/year); the strong 2024
  validation number rested on relatively few trades, and the held-out window simply
  did not produce enough to judge. Honest verdict: **NEEDS_MORE_DATA** — not refuted,
  but not confirmable on the data we have.
- **trend_day 5m** traded enough (48 OOS) but is **barely positive (+$11) and fails
  cost/slippage stress at baseline 1×/1× commission**, and was profitable in only 1
  of 2 lockbox years. The 5m edge inverts under realistic costs. **REJECT.**
- **range_fade / vwap_revert** never reached the freeze — they failed the 2024
  validation gauntlet outright.

## Paper/shadow readiness

| Strategy | Verdict | Exp. monthly $ | Exp. max DD $ | Account |
|---|---|---:|---:|---|
| trend_day 15m | NEEDS_MORE_DATA | — | ~$422 | — |
| trend_day 5m | REJECT | — | ~$1,593 | — |

**No strategy is paper/shadow ready.** The single most promising candidate is
`trend_day` on 15m (strong, clean validation; survives ruin/cost/Topstep; 100%
positive validation years) — its only failure is **frequency**: it does not generate
enough held-out trades to confirm. That is a "watch and accumulate forward data"
candidate, not a "paper-trade now" one.

---

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **192 tests**, incl. Phase-11 family
  determinism, `ACCEPT_FOR_PAPER`/REJECT/NEEDS_MORE_DATA verdict reachability,
  survivor selection, and frozen-family reconstruction).
- ✅ A clean final lockbox was **reserved before searching** (search loaded only
  through 2024-12-31; lockbox `pristine=True` verified at run time).
- ✅ Frozen hypotheses saved to a hash-verified registry; lockbox run completes;
  every frozen candidate received a verdict.
- ✅ No strategy accepted unless it passed the untouched lockbox — none did. Nothing
  forced.

## Honest answer & the standing candidate

> **Did we find a strategy confirmed enough to paper trade?** No. With a properly
> reserved, genuinely untouched 2025-2026 lockbox — where `ACCEPT_FOR_PAPER` was fully
> reachable — the strongest candidate (trend_day, 15m) is **promising but
> unconfirmable on available data** (too few held-out trades), and the tradeable 5m
> variant **fails cost stress**. The aggressive search did surface a real lead
> (trend-day momentum is the first family to post a clean, high-DSR, 100%-positive-year
> validation), but "confirmed enough" requires held-out evidence the data cannot yet
> supply.

**Recommended next step (not done here):** put `trend_day` 15m on a **forward
paper/shadow** feed and accumulate trades until there are ≥30 forward signals, then
re-run `phase11 lockbox` against that forward window. This is the only honest route to
`ACCEPT_FOR_PAPER` for a low-frequency edge. Building the paper feed was out of scope.

Per the standing rule — *no edge until proven; reject honestly, do not force an
accept* — Phase 11 ends with **zero strategies promoted to paper**, `trend_day` 15m
flagged NEEDS_MORE_DATA as the lead to watch, and everything else rejected.

---

## Reproduction

```
nqlab phase11 search  --no-build      # train 2021-23 / validation 2024 (lockbox reserved)
nqlab phase11 freeze                  # freeze top validation survivors (never sees the lockbox)
nqlab phase11 lockbox --no-build      # one shot on the untouched 2025-2026 lockbox
nqlab report phase11                  # registry + lockbox + paper-readiness verdicts
```
