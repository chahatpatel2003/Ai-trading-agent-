# Phase 9 — NQ-only Higher-Timeframe Context & Asymmetric Liquidity Discovery

**Question:** Phases 6–8 retired the intraday order-flow edges (`bias_pullback`,
`pd_sweep_rev`, `cvd_pullback`): real direction tilt, but sub-transaction-cost and
killed by the Topstep ruin/cost scissor. Is there a **new NQ-only signal source** in
higher-timeframe *context* (prior-day/week/month liquidity, overnight range, session
narrative, day-type regime) traded as **asymmetric** setups (intended MFE ≫ MAE)?

**Verdict: No accepted strategy — but this is the first signal source to clear the
structural scissor and the full gauntlet, failing *only* on multiple-testing
deflation.** 0 of 1,056 candidates ACCEPT. Unlike Phases 7–8, a large set of
candidates survive ruin **and** cost stress **and** walk-forward **and** Topstep —
they are then honestly rejected because their in-sample Sharpe does not clear the
Deflated-Sharpe bar over a 176-candidate search. No gate was relaxed; nothing was
force-accepted. ES was not used (no ES/NQ lead-lag). The retired Phase 6–8 edges were
not touched.

---

## What was built (NQ-only, trades-derivable, PIT-safe)

- **HTF context layer** (`features/htf_context.py`, `build_htf_context_features`):
  prior-day open/range, prior-week & prior-month high/low, overnight (ETH) high/low/
  range, NY-open location within the overnight range, gap direction + intraday
  gap-fill flag, inside/outside/NR/WR prior-day flags, **rule-based prior-day-type**
  (trend/range/reversal/compression/expansion/neutral), 1h/4h trend bias, volatility
  regime, and distance to the nearest liquidity level. Every column is shifted/
  trailing/exposed-after-close — proven by a no-lookahead test
  (`tests/unit/test_htf_context_pit.py`): the value at row *t* computed on the full
  series equals the value computed on the prefix `[0..t]`.
- **Five asymmetric setup families** (`discovery/htf_setups.py`), each tight-stop /
  wide-target by construction: **A** HTF-bias + liquidity sweep + reclaim, **B**
  opening drive through the overnight/London range (bias + vol confirm), **C** gap
  go/fade, **D** prior-week/month liquidity sweep-or-breakout, **E** compression →
  NY expansion break.
- **Diagnostics-first prefilter** (`discovery/prefilter.py`): before the full
  gauntlet, one cheap backtest rejects candidates with too few trades
  (NEEDS_MORE_DATA), no directional edge before costs, or **MAE ≥ MFE** (not
  asymmetric). Passing candidates go through the unchanged Phase 5–8 gauntlet.
- **CLI**: `nqlab features htf-context` (build + PIT coverage report),
  `nqlab discover htf-setups` (Phase-9 discovery), `nqlab report phase9` (matrix +
  leaderboard). `make check` is green.

Trades-only families → out-of-sample is the full **2024 → 2026** window (MBP1 not
required). Train **2021–2023**.

---

## The discovery matrix (`nqlab report phase9`)

176 candidates per cell (5 families × grids × risk {25,50,100,200} × sizing
{mnq_only, adaptive}); NY-AM session. "Feasible floor" = OOS > 0 **and** ruin ≤ 10%
**and** not blown **and** survives cost stress.

| Run | Trials | ACCEPT | Feasible floor | PBO | best MFE/MAE |
|---|---:|---:|---:|---:|---:|
| `phase9_50k_5m`   | 176 | 0 | **22** | **13%** | 3.71 |
| `phase9_100k_5m`  | 176 | 0 | **26** | **15%** | 3.71 |
| `phase9_150k_5m`  | 176 | 0 | **39** | **15%** | 3.71 |
| `phase9_50k_15m`  | 176 | 0 | 8  | 47% | 10.63 |
| `phase9_100k_15m` | 176 | 0 | 16 | 48% | 10.63 |
| `phase9_150k_15m` | 176 | 0 | 26 | 48% | 10.63 |
| **Total** | **1,056** | **0** | **137** | — | — |

Every candidate received a verdict (ACCEPT / REJECT / NEEDS_MORE_DATA; no nulls).

---

## Finding 1 — the asymmetric setups clear the scissor that killed Phases 7–8

In Phase 8, essentially **zero** candidates cleared OOS>0 + ruin≤10% **and** cost
stress: sizing up to clear costs blew the trailing drawdown (~80% ruin), sizing down
to survive ruin inverted under costs. Phase 9 is different: **137 candidates clear
the full survival floor** (OOS>0, ruin≤10%, cost-stress-survived, not blown). The
tight-stop/wide-target structure plus HTF-context entries produce a trade
distribution that is genuinely asymmetric — **median realized MFE/MAE = 1.30, 78% of
candidates > 1.2, max 4.22** — so a stop that respects the Topstep budget still gives
the edge room to pay.

## Finding 2 — they pass the whole gauntlet, then die on multiple-testing deflation

Among the survival-floor candidates the rejection reason is overwhelmingly singular:
**129 candidate-instances carry `gauntlet_verdict = ACCEPT`** — "survives OOS,
walk-forward, cost stress, Monte-Carlo, and Topstep" — and are downgraded to REJECT
**only** by the Deflated-Sharpe multiple-testing step (DSR < 0.90 over 176 trials).
This is a fundamentally higher-quality failure than Phases 6–8 (which failed on
ruin / cost / OOS outright). The signal is real enough to clear every economic and
robustness gate; what it cannot yet do is prove statistical significance *after
correcting for how many variants were searched*.

The strongest individual candidates (all REJECT on deflation):

| Run | Family | OOS | Ruin | Cost-stress | WF+ | Pos. yrs | MFE/MAE | DSR |
|---|---|---:|---:|:--:|---:|---:|---:|---:|
| 50k_5m  | compression_expansion (OR break) | +2,899 | 1% | pass | 82% | 83% | 1.38 | 0.17 |
| 150k_5m | opening_drive (London break) | +5,350 | 5% | pass | 50% | 100% | 1.31 | 0.00 |
| 150k_5m | gap_setup (fade) | +3,629 | 0% | pass | 44% | 67% | 1.49 | 0.01 |

## Finding 3 — which contexts work, and which timeframe

- **Best families:** `opening_drive` (NY drive through the **London** range; mean OOS
  +2,625, 48 survival-floor) and `compression_expansion` (NY break after a narrow
  prior-day/overnight range; 55 survival-floor, best DSR 0.17, WF 82%, 83% positive
  years). `wk_month_liq` and `gap_setup` are middling; `htf_sweep_reclaim` is the
  weakest (mean OOS −419) — sweeping-and-reclaiming a level is not, by itself, an
  edge here.
- **Best liquidity location:** the **London-session range** (for the opening drive)
  and the **opening range after compression** dominate the top of the board.
- **Timeframe: 5m is the signal, not 15m.** 5m has more survivors and a **healthy
  PBO of 13–15%** (the leaderboard ordering is *not* overfit — the opposite of Phase
  7–8's 71–87%). 15m's PBO is 47–48%: too few trades, ordering is overfit. **1m was
  not run** (5m/15m first per the plan; 5m already gives the healthiest result and 1m
  was the most overfit-prone timeframe in Phase 8).
- **Account size:** more survivors at larger accounts (22→26→39 at 5m) — the wider
  trailing drawdown admits more candidates, same mechanism as Phase 8, but here they
  also clear cost stress.

---

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **183 tests**, incl. the HTF no-lookahead test
  and the day-type classifier PIT test).
- ✅ HTF context features PIT-safe; day-type classifier has no lookahead (both proven
  by `test_htf_context_pit.py`).
- ✅ Setup discovery completed: 6 cells, 1,056 candidates.
- ✅ Every candidate received ACCEPT / REJECT / NEEDS_MORE_DATA (no nulls).
- ✅ No strategy accepted: nothing clears DSR ≥ 0.90 after multiple-testing deflation;
    no gate was relaxed and nothing was force-accepted.

---

## Honest answer & what comes next (not done here)

> **Is there a tradeable new NQ-only signal in higher-timeframe context + asymmetric
> liquidity?** The evidence is the most encouraging of any phase. The
> `opening_drive` (London-range break) and `compression_expansion` (OR break after a
> quiet prior day) families produce genuinely asymmetric trade distributions that
> survive realistic costs, the Topstep trailing drawdown, walk-forward stability, and
> multi-year consistency — the exact wall the retired edges could not pass. They are
> rejected **only** because their in-sample Sharpe is not significant after deflating
> for a 176-candidate search.

That specific failure has a specific, honest remedy that is **not** "relax the gate":
a **pre-registered narrow confirmation** of the 2–3 best families (a handful of
hypotheses, not 176) re-run on the held-out window, where the multiple-testing
penalty is small enough that a real edge can clear DSR. That is the natural Phase 10,
and it is deliberately *not* done here — running it now, after seeing the leaderboard,
would itself be the overfitting the DSR is protecting against.

**Retired:** nothing new. The Phase 6–8 edges remain retired and were not re-optimised.
Per the standing rule — *no edge until proven; reject honestly, do not force an
accept* — Phase 9 ends with **zero strategies promoted** and a clearly-identified,
honestly-bounded candidate signal carried forward for narrow confirmation.

---

## Reproduction

```
nqlab features htf-context --timeframe 5m --no-build      # PIT coverage report
nqlab discover htf-setups --timeframe 5m  --account 150k --no-build
nqlab discover htf-setups --timeframe 15m --account 150k --no-build
nqlab report phase9                                        # 6-cell matrix
nqlab report phase9 phase9_50k_5m                          # one cell's leaderboard
```

Raw market data was never modified; no ES data was used. HTF context features are
trades-derivable and computed on the fly (PIT-safe); the full sweep (1,056 candidates,
diagnostics-first prefilter + gauntlet) completed in ~12 minutes on the dev profile.
