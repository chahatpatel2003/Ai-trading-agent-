# Phase 26 — Non-Topstep High-Profit-Factor NQ Discovery

The Topstep payout framework is **paused**. Goal: the highest-quality NQ edge possible — **PF ≥ 2.0
after costs**, controlled drawdown, return/maxDD ≥ 3, Sharpe ≥ 1 — ranked by PF and
return-to-drawdown, with **no** NY-AM restriction, no trailing drawdown, no $200-day rule, no
consistency rule. All sessions, timeframes, and wider asymmetric targets allowed.

**Answer: No.** Across 5 high-PF families × 2 sessions × 2 R-targets × 2 timeframes, the best
validation profit factor is **1.46** (a single-year trend-follow; train PF only 1.24), and **0 of
20 candidates** reach the 2.0 bar. Removing every Topstep constraint did **not** unlock a high-PF
edge — confirming the constraints were never the binding problem. The lockbox was never touched. No
forced acceptance.

---

## What was built

- **`discovery/phase26.py`** — the quality-stats panel (PF, expectancy, max/avg drawdown, net,
  **return/maxDD, Sharpe, Sortino, Calmar**, winning months, max losing streak, MFE/MAE) and the
  high-PF verdict (PF ≥ 2.0 + return/DD ≥ 3 + Sharpe ≥ 1 + winning months ≥ 70%, on validation AND
  lockbox; ACCEPT_FOR_PAPER ceiling since it is non-Topstep).
- **5 high-PF families** (`phase26_families.py`), full RTH (no NY-AM cap) + an NY-PM variant, wide
  asymmetric targets (2R/3R), longer holds, 1 MNQ: trend-following, breakout-continuation,
  mean-reversion, volatility-regime expansion, HTF-POI swing.
- **CLI** `phase26 scan` — backtests train+val ranked by PF, and touches the reserved 2025-2026
  lockbox **only** for validation-passers (none qualified). **5 tests**; **`make check` green
  (284 tests)**.

## Exact costs + sensitivity

NQ $0.85/side, MNQ $0.25, slippage 1 tick/fill (TARGET exact, STOP/EOD slipped). Cost sensitivity on
the families (validation 2024) shows PF is essentially **cost-insensitive** — e.g. trend-follow PF
0.93 (0×) → 0.91 (1×) → 0.88 (2×); NY-PM trend 0.99 (0×) → 0.96 (1×) → 0.93 (2×). The edge is absent
*before* costs, so costs are not what blocks PF 2.0.

## Data protocol

Train 2021–2023, validation 2024, lockbox 2025–2026 **untouched** (0 validation-passers, so the
lockbox was correctly never touched).

## Result — high-PF leaderboard (validation 2024, 1 MNQ, full RTH/PM)

| family | trades | **val PF** | ret/maxDD | Sharpe | win% | val net | train PF |
|---|--:|--:|--:|--:|--:|--:|--:|
| trend-follow | 211 | **1.46** | 3.37 | 2.51 | 38% | +$7,515 | **1.24** |
| trend-follow | 217 | 1.28 | 1.71 | 1.77 | 41% | +$4,512 | 1.05 |
| trend-follow | 353 | 1.25 | 1.86 | 1.81 | 33% | +$6,509 | 1.06 |
| HTF-POI swing | 284–637 | 1.12–1.14 | 1.1–2.2 | 0.9–1.1 | 32–38% | +$1.4–3.7k | 1.06–1.15 |
| breakout-continuation | 391–1010 | 1.01–1.14 | 0.1–1.9 | 0.1–1.2 | 31–40% | +$0.4–4.4k | 0.97–1.10 |
| mean-reversion | 396 | 1.04 | 0.26 | 0.38 | 30% | +$1,264 | 1.01 |
| vol-expansion | 31 | 1.04 | 0.12 | 0.30 | 39% | +$83 | 0.74 |

- **Best PF = 1.46** (trend-follow) — far below the 2.0 minimum, and its **train PF is only 1.24**
  with 38% win rate: 2024 was a strong directional year for NQ, so trend-following looked better
  out-of-sample than in-sample — a single-regime artifact, not a stable edge.
- Every other family sits at **PF 1.0–1.3**. Win rates 30–41% (the trend families rely on a few big
  winners, the mean-reversion on a slight win-rate tilt — neither produces PF 2.0).
- **0 of 20 candidates clear PF 2.0** (or even a *stable* 1.30 on both train and val).

## Verdict

```
REJECT. No high-PF (>=2.0) low-drawdown NQ strategy found, even outside Topstep constraints.
Best validation PF = 1.46 (single-year trend; train PF 1.24). Lockbox 2025-2026 untouched.
```

## Direct answer

> **"Can NQ produce a very high-PF, low-drawdown strategy outside Topstep constraints?"**
>
> **No.** With every Topstep constraint removed — no NY-AM window, no locked-buffer trailing
> drawdown, no $200-day or consistency rule, full RTH + NY-PM sessions, 5m/15m, wide 2R/3R targets,
> trend / breakout / mean-reversion / volatility / HTF-POI families — the best after-cost profit
> factor is **1.46**, and that one is a single-year (2024) trend artifact with a 1.24 train PF.
> Nothing reaches 2.0; nothing is stable across train and validation. **This is the decisive
> result of the whole program:** the failures in Phases 16–25 were *not* caused by Topstep's
> constraints — removing all of them and optimizing purely for profit factor still tops out at
> ~1.46. NQ bar/price-action data simply does not contain a high-PF directional edge at these
> horizons, consistent with the Phase-19 edge map (~52% directional accuracy, MFE ≈ MAE).

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **284 tests**, incl. quality-stats + Phase-26 tests).
- ✅ Lockbox 2025-2026 **never touched** (0 validation-passers).
- ✅ Ranked by PF + return/drawdown (not raw PnL); cost sensitivity reported; no forced acceptance,
  no weak-PF acceptance.
- ✅ The phase question answered directly: **no high-PF low-DD NQ edge exists outside Topstep
  either** — best PF 1.46.

## Reproduction

```
nqlab phase26 scan --timeframes 5m,15m   # high-PF leaderboard (train+val); lockbox only for passers
```
