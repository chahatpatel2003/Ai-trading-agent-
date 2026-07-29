# Phase 7 — Survival Optimization of the Positive-OOS Setup Families

**Question:** Phase 6 produced a handful of contextual setups with *positive
out-of-sample* PnL (`bias_pullback`, `pd_sweep_rev`) and a measurable *directional
edge before costs* (`cvd`-confirmed NY-AM pullbacks). Can better exits, smaller
risk, session filtering, and Topstep-aware sizing turn that signal into a
**Topstep-survivable** strategy?

**Verdict: No. 0 of 165 candidates accepted.** The positive-OOS signal is real
but **not survivable**. It is too small to clear realistic costs at any size that
also survives the Topstep trailing drawdown, and too inconsistent across years and
walk-forward windows to trust. Rejected honestly — full reasoning below.

No acceptance rule was relaxed. No new strategy family was searched. The split,
the gauntlet, and the acceptance gates are exactly the Phase 5/6 ones.

---

## What was run

NQ 1-minute, continuous (`NQ.c`), MBP1-derived setup features.
Train **2021-01-01 → 2023-12-31**, out-of-sample **2024-01-01 → 2025-06-30**
(the MBP1-availability split — MBP1 does not exist past 2025-H1, and we did not
pretend it does). Topstep 50K rules. Validation runs with `gate_entries=False`
so the full trade distribution is scored (the account *records* drawdown/ruin but
does not halt mid-sample), while a separate chronological run checks whether the
account actually blows.

| Run | `run_id` | Candidates | ACCEPT | REJECT | NEEDS_MORE_DATA | PBO |
|---|---|---:|---:|---:|---:|---:|
| Exit grid (stop × target × trail × min-hold, fixed risk) | `phase7_exits_50k` | 48 | **0** | 32 | 16 | 84.3% |
| Risk sweep ($25–$250, fixed trailing exit) | `phase7_risk_50k` | 21 | **0** | 14 | 7 | 71.0% |
| Combined survival (stop × target × trail × risk × session) | `phase7_survival_50k` | 96 | **0** | 64 | 32 | 86.8% |
| **Total** | | **165** | **0** | 110 | 55 | — |

Reproduce any leaderboard with `nqlab report phase7 <run_id>`.

PBO (probability of backtest overfitting) is **71–87%** across all three sweeps:
the leaderboard ordering is mostly overfit noise, exactly what you expect when no
genuine survivable configuration exists in the grid.

The 55 `NEEDS_MORE_DATA` verdicts are all `cvd_pullback`: the NY-AM CVD-pullback
filter is so selective on 2024–2025H1 that it produces too few OOS trades to judge
(≈0 net, near-zero trade count). Its Phase 6 directional edge does not survive
being gated down to a tradeable, survivable setup — there simply isn't enough of it.

---

## The core finding: the survival vs. profit scissor

The risk sweep on `bias_pullback` (identical entries and exit; **only the dollar
risk per trade changes**) is the whole story in one table:

| Risk/trade | OOS net | Full-run net | Risk of ruin | Max DD | Walk-fwd + | Pos. years | Blows account |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| **$25** | **−128** | +253 | **0%** | $310 | 54% | 60% | No |
| $50 | −337 | −2,290 | 100% | $2,463 | 29% | 0% | **Yes** |
| $75 | +88 | −2,623 | 100% | $3,678 | 29% | 40% | **Yes** |
| $100 | +637 | −4,367 | 100% | $5,563 | 29% | 20% | **Yes** |
| $150 | +1,765 | −5,813 | 100% | $8,868 | 36% | 60% | **Yes** |
| $200 | +2,407 | −8,440 | 100% | $12,205 | 36% | 40% | **Yes** |
| $250 | **+2,905** | −8,761 | 100% | $13,627 | 29% | 40% | **Yes** |

Read it top to bottom:

- **Size up enough to make money → guaranteed ruin.** OOS net climbs monotonically
  to +$2,905, but from $50 upward the **risk of ruin is 100%** and the max drawdown
  ($2.5K → $13.6K) blows the Topstep **$2,000 trailing drawdown every single time**.
  The "full-run net" is deeply *negative* (−$8,761 at $250) precisely because the
  account dies early in the chronological run and never compounds. The positive OOS
  number is an artifact of scoring an un-halted account.
- **Size down enough to survive → no edge left.** At **$25** (which forces MNQ
  micro-sizing) ruin is 0% and the account survives — but OOS net is **−$128**. The
  edge that survives the drawdown constraint is worth *less than zero* after costs.

There is **no risk level where OOS > 0 and ruin ≤ 10% simultaneously.** That is the
definition of an unsurvivable edge, and it is not fixable by sizing — sizing is a
linear scalar on a distribution whose *shape* is the problem.

---

## Why the "survivable-looking" candidates still fail

Three candidates in the 96-run cleared the two hardest gates at once — **positive
OOS, ruin ≤ 10%, account not blown.** All three are `pd_sweep_rev` at $50 risk with
a wide stop (2.5 ATR) and wide target (4.0 ATR). All three were still **rejected**:

| OOS net | Ruin | Max DD | Walk-fwd + | Pos. years | Cost-stress | Why rejected |
|---:|---:|---:|---:|---:|:--:|---|
| +284 | 0% | $893 | 50% | 40% | **fails** | Edge gone at 1×/2×/3× slippage and 1.5× commission |
| +472 | 0% | $795 | 36% | 40% | **fails** | Walk-forward < 50% **and** fails cost stress |
| +41 | 0% | $746 | 29% | 20% | **fails** | Profitable in only 1/5 years; WF 29%; fails cost stress |

The edge that *survives the drawdown* does **not survive transaction costs** (it
inverts under realistic slippage/commission) and does **not persist** (profitable
in 1–2 of 5 years, walk-forward positive in well under half of windows). This is the
same signal as the high-risk rows above, just shrunk to a survivable size — and at
that size it is indistinguishable from noise plus a cost drag.

---

## Diagnostics: the edge is real, but the wrong shape

MAE/MFE diagnostics confirm the Phase 6 observation rather than overturning it:

- `direction_edge_before_costs = 1` (TRUE) on the `bias_pullback` candidates — **the
  entry direction is correct more often than not, before costs.** The signal is not
  imaginary.
- **avg MAE ≈ 12–17 pts, avg MFE ≈ 19–23 pts.** Trades go favorable, but the adverse
  excursion is nearly as large — so a stop tight enough to bound the Topstep drawdown
  is hit by normal noise first, and a stop loose enough to stay in is what produces
  the $2.5K–$13.6K drawdowns above.
- **`mfe_capture ≈ 0`** — exits capture almost none of the favorable excursion. We
  swept fixed targets (2.0/3.0/4.0 ATR), ATR trailing, breakeven, min-hold (0/6 bars),
  and time stops; **none** converted MFE into realized, survivable PnL. The wide
  (4.0 ATR) target was consistently the *least bad* exit — i.e. the edge needs room
  the Topstep drawdown won't grant.

In plain terms: there is a faint directional tilt, but the trade's noise band is
wider than the Topstep loss budget. You cannot place a stop that both respects the
$2,000 trailing drawdown and gives the edge enough room to pay — the two constraints
have no overlap.

---

## Best parameter zones (reporting only — none accepted)

From the survival run's mean-OOS-by-parameter analysis, the *least bad* settings were:

- **Session:** NY-AM (09:30–12:00) beats all-day for `bias_pullback`/`cvd` (consistent
  with Phase 6's NY-AM directional edge).
- **Target:** wide — **4.0 ATR** dominated 2.0 ATR (mean OOS +620 vs the rest for
  `bias_pullback`). The edge is a runner, not a scalp.
- **Stop:** 1.5 ATR best by mean OOS, but this is exactly what drives ruin to 100%.
- **Trailing:** ATR trailing helped `bias_pullback` mean OOS but was *catastrophic*
  for `pd_sweep_rev` (−$1,526 mean) — it banks noise and cuts runners.
- **Risk:** lower is always safer; the only ruin-safe tier ($25) is unprofitable.

These zones are documented for the record. **None produces an acceptable candidate** —
the best-OOS zones are precisely the high-ruin zones.

---

## Acceptance gates — all held strict

No gate was loosened to force a pass. Candidates were rejected for, in order of
frequency across the 165 trials:

1. **Not consistent across years** — profitable in only 0–1 of 5 years (38 trials).
2. **Walk-forward < 50%** of windows positive (≈20 trials).
3. **Risk of ruin > 10%** / blows the Topstep account in the chronological run.
4. **Fails cost/slippage stress** — edge inverts at 2–3× slippage or 1.5× commission.
5. **OOS net ≤ 0** outright.
6. **DSR ≈ 0 / PBO 71–87%** — no deflated significance; ordering is overfit.

Most rejected candidates fail **several** of these at once.

---

## Conclusion

**The positive-OOS signal from Phase 6 cannot be converted into a Topstep-survivable
strategy, and we are rejecting it honestly.**

The failure is not "no edge." There is a faint, real directional edge (positive
direction-edge-before-costs, positive OOS at large size). The failure is **structural
and specific:**

> The edge's per-trade noise band (MAE ≈ MFE ≈ 15–20 pts) is **wider than the Topstep
> $2,000 trailing-drawdown budget.** At sizes large enough for the edge to clear
> commissions and slippage, the drawdown blows the account with ~100% probability. At
> sizes small enough to survive the drawdown, costs eat the entire edge and OOS goes
> negative. No exit rule we swept (fixed, ATR, trailing, breakeven, min-hold, time
> stop) closes the gap, because the problem is the *shape* of the trade distribution,
> not the *scale* of the bet. The residual positive-OOS configurations that do survive
> ruin then fail cost-stress and year-consistency — i.e. they are noise.

This is a clean, decisive negative result, and the platform behaved exactly as
designed: it surfaced a plausible signal, stress-tested it against survival rather
than raw PnL, and refused to deploy it.

### What this rules out, and what it doesn't

- **Ruled out:** making `bias_pullback` / `pd_sweep_rev` / NY-AM `cvd_pullback`
  survivable on **NQ 1-minute** via exit/risk/session tuning alone. Closed.
- **Not ruled out (future phases, not this one):** (a) a *higher-timeframe* expression
  where the edge's natural noise band is a smaller fraction of the drawdown budget;
  (b) edges whose MAE is structurally smaller than their MFE (asymmetric setups), so a
  protective stop fits inside the Topstep budget; (c) the larger 100K/150K accounts,
  whose wider trailing drawdown changes the scissor — though that trades survival
  headroom for capital, and must clear the same gauntlet.

Per the standing rule — *no edge until proven; reject honestly, do not force an
accept* — Phase 7 ends with **zero strategies promoted** and the positive-OOS setup
families formally retired on NQ 1m.
