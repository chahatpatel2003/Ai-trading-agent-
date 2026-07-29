# Phase 29 — Unrestricted Full-History NQ Strategy Factory (EOD-flat only)

Search reset: **all Topstep / NY-AM / fixed-RR / payout constraints dropped.** The only hard rule is
**flat by end of day** (no overnight holds). Goal: find the highest-quality NQ *intraday* strategy by
profit factor, drawdown, win rate, return/DD, risk ratios and **year-by-year stability** — searched
across **every available year (2021 → 2026-H1)** with realistic costs, no hindsight, PIT-safe only.

**Headline result.**

```
A real but MODEST intraday edge exists once the old constraints are removed.
Best full-history after-cost PF ≈ 1.44 (very low drawdown). Best out-of-sample (walk-forward) PF ≈ 1.25-1.30.
NO PF>=2.0 edge was found at adequate sample — consistent with all 28 prior phases.
2 families carry the edge: INSIDE-DAY BREAKOUT (price action) and SWEEP+DISPLACEMENT (ICT), plus
trend-follow held-to-EOD. Mean-reversion / fades / raw MSS-CHoCH have NO edge after costs.
```

This phase **discovers** the best candidates on full history and then **stress-tests** them; per the
honesty rule it does **not** call them "live-ready." Best candidates are marked
`ACCEPT_FOR_FORWARD_PAPER`; nothing is forced.

---

## 0. Integrity (Phase-27 rules re-confirmed before trusting any number)

The audited engine `run_backtest` is reused unchanged (next-open entry; same-bar **stop-first**;
target = exact limit, stop/EOD/forced = 1-tick slipped; MFE/MAE tracked, never used in entry
decisions). New Phase-29 code was proven PIT-safe by **prefix-invariance** (`tests/unit/test_phase29_integrity.py`, all green):

| check | result |
|---|---|
| prefix==full for every new indicator + **Supertrend** (`ema/macd/stoch/bollinger/keltner/donchian/adx/st_dir`) | PASS |
| prefix==full for a family **signal** end-to-end | PASS |
| all 40 families resolve only known (non-future) columns | PASS |
| **EOD-flat guarantee**: `intraday_only` drops any trade whose entry/exit span two CME trading days | PASS |
| Phase-27 suite (engine + existing feature layers) | PASS (unchanged) |
| `pytest` Phase-29 + Phase-27 integrity | **12 tests pass** |

**EOD-flat handling.** Each candidate runs on the contiguous **midnight→16:00 ET block** so the
engine's 15:55 force-flat fires once per trading day and nothing crosses days. On early-close
holidays (RTH ends 13:00, no 15:55 bar) the engine cannot flatten; those few carried trades are
**dropped and counted** (`cross_day_dropped`, typically 0–36 per candidate) so every reported trade
is a true intraday EOD-flat trade.

**Cost model (audited, printed).** NQ comm $0.85/side, MNQ $0.25/side; slippage 1 tick/fill; target
exact, stop/EOD slipped → **NQ win/loss $6.70 / $11.70, MNQ $1.00 / $1.50.** Sensitivity run at
**0× / 0.5× / 1× / 2×** for finalists (below). Sizing fixed at **1 MNQ** (`mnq_fixed1`) so PF /
win-rate / drawdown are directly comparable across all candidates; dollar figures scale linearly with
contracts.

---

## 1. Strategy universe summary

| | |
|---|---|
| Family signals | **40** (classic indicators, price-action, ICT/SMC, volume/volatility, order-flow) |
| Candidates run | **942** (family × session × stop × exit grid) |
| Timeframe | **5m** — the only frame with full coverage **2021-01 → 2026-06**; 1m/15m/MBP1 stop at 2025-05 |
| Years | **2021, 2022, 2023, 2024, 2025, 2026-H1** (all of them) |
| Sessions tested | NY_AM, NY_PM, RTH, LONDON, LONDON→NY, overnight-early (00:00–09:30), ALLDAY |
| Stops | ATR-scaled (1.0–2.5× ATR) | 
| Exits | fixed RR (1, 1.5, 2, 3), ATR-trailing, **hold-to-EOD** |
| Each candidate | run **once** over full history; year/month/LOYO/walk-forward derived from stored trades; cost-sensitivity = re-run |
| Passed broad screen | **351 / 942** (≥150 trades, ≥3 yrs traded, net>0, PF>1); **591 rejected** (saved with reasons) |

**Strongest families** (median PF over their config grid): `inside_day_break` **1.22**,
`sweep_displacement` **1.17**, `gap_fill` 1.09, `ema_cross_21_50` 1.08, `orb_break` 1.04,
`supertrend` 1.03, `ema_stack` 1.03, `premium_discount` 1.02, `macd_hist_expand` 1.02.

**Weakest families** (no edge, 0 passers): `orb_fade` 0.88, `choch` 0.89, `mss` 0.90, `pdhl_break`
0.91, `vwap_revert` 0.91, `bollinger_revert` 0.91, `cvd_divergence` 0.90.

**By kind** (median PF): trend 1.014 · breakout 0.986 · vol 0.980 · ict 0.973 · **mr 0.925** · orderflow 0.920.
**By session** (median PF): **NY_AM 1.002** · LONDON→NY 0.986 · RTH 0.985 · ALLDAY 0.971 · NY_PM 0.938.
**By exit** (median PF / median win-rate): **eod 1.062 / 27%** · rr3 1.015 / 31% · rr2 0.985 / 35% ·
trail 0.973 / 36% · rr1.5 0.973 / 40% · **rr1 0.945 / 49%**.

> The exit table *is* the high-PF-vs-high-win-rate tradeoff, quantified: letting winners run (EOD /
> high RR) maximizes PF at low win rate; tight 1:1 targets maximize win rate at lower PF. You cannot
> have both maxed at once in NQ.

**Coverage gaps (honest).** True **volume/market profile** (VAH/VAL/POC/HVN/LVN) was **not** built —
those features don't exist in the store yet; approximated only via VWAP bands + prior-day levels.
Some ICT (order-block / breaker / mitigation as standalone families) and order-flow (OFI / absorption
/ imbalance) families, and explicit cross-timeframe (4h-trend→15m-entry) entries, were not wired.
Order-flow (`cvd`) is limited to **≤2025-05** (MBP1 availability). These are the clearest "deserves
deeper research" items.

---

## 2. Top-25 leaderboard (full history, after costs, 1 MNQ)

Ranked by PF → return/maxDD → win-rate → maxDD. `inside_day_break|RTH` and `|LONDON_NY` are
**identical** (the setup only fires after the opening range forms, so the two windows capture the same
trades) — shown once, marked ‡.

| # | strategy | n | PF | win% | net $ | maxDD $ | ret/DD | Sharpe | Sortino | yrs+ | yrConc | win-mo% |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | inside_day_break · RTH ‡ · s2.5 · eod | 221 | **1.44** | 45% | 8,120 | 1,880 | 4.32 | 2.55 | 5.19 | 5/6 | 0.36 | 60% |
| 2 | inside_day_break · RTH ‡ · s2.5 · rr2 | 282 | 1.37 | 46% | 8,410 | 1,923 | 4.37 | 2.51 | 5.41 | 5/6 | 0.35 | 59% |
| 3 | sweep_displacement · NY_AM · s1.5 · rr1 | 185 | 1.41 | **60%** | 2,736 | 859 | 3.18 | 2.51 | 3.68 | 4/6 | 0.40 | 60% |
| 4 | sweep_displacement · NY_AM · s1 · rr1.5 | 185 | 1.40 | 49% | 2,235 | 608 | 3.68 | 2.44 | 4.12 | 5/6 | 0.38 | 59% |
| 5 | sweep_displacement · NY_AM · s1 · rr2 | 185 | 1.38 | 41% | 2,448 | 732 | 3.35 | 2.27 | 4.22 | 5/6 | 0.47 | 59% |
| 6 | inside_day_break · RTH ‡ · s1.5 · eod | 293 | 1.34 | 34% | 6,617 | 2,941 | 2.25 | 2.42 | 4.69 | 5/6 | 0.44 | 57% |
| 7 | sweep_displacement · NY_AM · s1.5 · rr1.5 | 184 | 1.33 | 47% | 2,809 | 970 | 2.90 | 2.05 | 3.38 | 5/6 | 0.56 | 57% |
| 8 | inside_day_break · RTH ‡ · s2.5 · rr1.5 | 335 | 1.31 | 49% | 8,041 | 2,693 | 2.99 | 2.56 | 5.26 | **6/6** | 0.41 | 59% |
| 9 | sweep_displacement · NY_AM · s1.5 · rr2 | 184 | 1.30 | 39% | 2,894 | 1,032 | 2.80 | 1.83 | 3.33 | 5/6 | 0.33 | 51% |
| 10 | inside_day_break · NY_AM · s1.5 · rr2 | 247 | 1.28 | 40% | 4,390 | 1,651 | 2.66 | 1.71 | 3.45 | 5/6 | 0.50 | 53% |
| 11 | sweep_displacement · RTH · s1 · rr2 | 314 | 1.27 | 40% | 2,784 | 697 | 4.00 | 1.72 | 3.02 | 5/6 | 0.48 | 57% |
| 12 | inside_day_break · NY_AM · s2.5 · rr1.5 | 194 | 1.27 | 48% | 4,431 | 2,170 | 2.04 | 1.40 | 2.80 | 5/6 | 0.56 | 55% |
| 13 | sweep_displacement · RTH · s1 · rr1.5 | 314 | 1.27 | 47% | 2,436 | 752 | 3.24 | 1.74 | 2.82 | 5/6 | 0.36 | 54% |
| 14 | sweep_displacement · RTH · s1.5 · rr1.5 | 312 | 1.25 | 45% | 3,309 | 1,312 | 2.52 | 1.64 | 2.69 | 5/6 | 0.63 | 55% |
| 15 | sweep_displacement · RTH · s1.5 · rr1 | 313 | 1.25 | 56% | 2,741 | 1,254 | 2.19 | 1.67 | 2.45 | 4/6 | 0.45 | 58% |
| 16 | **ema_cross_21_50 · NY_AM · s1.5 · eod** | 992 | 1.24 | 30% | **15,770** | 3,995 | 3.95 | 1.18 | 2.74 | 4/6 | 0.42 | 52% |
| 17 | inside_day_break · NY_AM · s2.5 · trail1.5 | 373 | 1.24 | 40% | 3,104 | 2,341 | 1.33 | 1.26 | 3.08 | 5/6 | 0.49 | 50% |
| 18 | inside_day_break · NY_AM · s1.5 · trail1.5 | 374 | 1.22 | 40% | 2,882 | 2,555 | 1.13 | 1.17 | 2.83 | 5/6 | 0.53 | 50% |
| 19 | **sweep_displacement · ALLDAY · s1 · rr2** | 601 | 1.21 | 37% | 3,237 | 752 | 4.30 | 1.40 | 2.51 | 5/6 | 0.38 | 58% |
| 20 | inside_day_break · NY_AM · s1.5 · eod | 196 | 1.21 | 33% | 2,961 | 3,594 | 0.82 | 1.28 | 2.30 | 4/6 | **0.99** | 48% |
| 21 | inside_day_break · LONDON→NY · s1.5 · rr2 | 566 | 1.21 | 39% | 6,747 | 2,262 | 2.98 | 2.32 | 4.62 | 5/6 | 0.43 | 62% |
| 22-25 | (more `inside_day_break` / `sweep_displacement` / `gap_fill` / `ema_cross` variants, PF 1.18-1.21) | | | | | | | | | | | |

Full machine-readable tables: `…/curated/artifacts/phase29/run1/{leaderboard.csv, candidates.csv,
rejected.csv, candidates_full.json, robust.json}`.

> **Robustness signal:** the top of the board is **whole families repeating across many
> session/stop/exit settings**, not one lucky config — the opposite of an overfit point. (Row 20 is
> the exception: `yrConc 0.99` = ~all profit from one year → flagged, downgraded.)

---

## 3. Finalist detail

### F1 — `inside_day_break | RTH | stop 2.5×ATR | exit rr2` *(best all-round)*

Inside day yesterday (`prev_inside_day==1`) → today, after the opening range forms (≥10:00 ET), enter
on a 5m **close beyond the OR high/low**; stop 2.5×ATR; target 2×stop; flat by 15:55.

| metric | value | metric | value |
|---|---|---|---|
| family / session / TF | price-action breakout / RTH / 5m | trades | 282 (~47/yr) |
| PF after cost | **1.374** | PF before cost (0×) | 1.391 |
| win rate | 45.7% | expectancy | $29.8 / trade (1 MNQ) |
| avg win / avg loss | $239 / −$148 | MFE/MAE | 1.47 |
| net (1 MNQ) | $8,410 | max drawdown | **$1,923** |
| return / maxDD | **4.37** | avg drawdown | low |
| Sharpe / Sortino / Calmar | 2.51 / 5.41 / 7.35 | max losing streak | 9 |
| winning months | 58.7% | year concentration | 0.35 (healthy) |
| **year-by-year PF** | 2021 **1.38** · 2022 **1.80** · 2023 **1.35** · 2024 **1.61** · 2025 **1.23** · 2026-H1 0.99 | | |
| **cost 0/0.5/1/2×** | PF 1.39 / 1.38 / **1.37** / 1.35 | (edge survives 2× cost) | |
| **walk-forward** (train 2021-23 → pick this family-config → test 2024-26) | **train PF 1.55 → test PF 1.26, +$3,584** | survives | |

Profitable every full year (only partial 2026-H1 is flat); cost-robust; very low DD; deep risk
ratios. The single best risk-adjusted intraday strategy found. **The `eod` sibling (F-variant) hits
PF 1.44 / ret-DD 4.32** with fewer trades (221).

### F2 — `sweep_displacement | NY_AM | stop 1.5×ATR | exit rr1` *(highest win rate)*

NY-AM only: a candle that **sweeps a prior swing low/high AND is a displacement candle** in the
reclaim direction; 1:1 target. **60.0% win rate**, PF 1.41, maxDD just **$859**.

| year | PF | win% | n | | year | PF | win% | n |
|---|--:|--:|--:|---|---|--:|--:|--:|
| 2021 | 0.75 | 37% | 30 | | 2024 | 1.18 | 59% | 39 |
| 2022 | 1.80 | 66% | 38 | | 2025 | 1.61 | 62% | 40 |
| 2023 | 3.60 | 81% | 27 | | 2026-H1 | 0.74 | 45% | 11 |

Cost 0/0.5/1/2× → PF 1.43 / 1.42 / **1.41** / 1.30 (the 1:1 target is the most cost-sensitive style).
Walk-forward: train PF 1.62 → **test PF 1.25, 59% OOS win rate, DD $581.** The genuine "high win
rate" answer — but note **two negative years (2021, partial-2026)**, so less monotone than F1. The
`s1|rr1.5` sibling is steadier (5/6 yrs, PF 1.40, ret/DD 3.68).

### F3 — `ema_cross_21_50 | NY_AM | hold-to-EOD` *(capacity / trend)*

Classic 21/50 EMA cross, ride to EOD. **992 trades**, biggest dollar figure (**+$15,770 / 1 MNQ**),
ret/DD 3.95 — but **win rate 30%**, Sharpe only 1.18, max losing streak 19, and **two losing years
(2023 −$239, 2026-H1 −$1,118)**. Cost-robust (PF 1.24 at 2× vs 1.27 at 0×). High-capacity trend
vehicle, lower quality than F1/F2.

### Also-rans worth noting
- `sweep_displacement | ALLDAY | s1 | rr2`: ret/DD **4.30**, tiny DD $752, **improving** over time
  (2021 −, 2026 +$1,004) — the cleanest "getting better" profile.
- `gap_fill` family: **all 24 configs PF>1** (median 1.09) — robust but marginal; fade an unfilled
  opening gap toward prior close.
- `inside_day_break | LONDON→NY | rr2`: 62% winning months, Sharpe 2.32, 566 trades.

---

## 4. Manual insights (direct answers)

- **Which setups actually show edge?** Two clearly: **inside-day breakout** (volatility-contraction →
  expansion) and **sweep+displacement** (ICT liquidity grab + momentum confirmation). Secondary,
  weaker: trend continuation (EMA-cross / Supertrend / EMA-stack) held to EOD, ORB breakout, gap-fill.
- **Which sessions work best?** **NY-AM** (median PF 1.00, best win rates), then the full RTH /
  London→NY. **NY-PM is the worst** (median PF 0.94) — afternoon chop. Pure overnight/Asia adds
  nothing tradeable in the costed, liquid window.
- **Which timeframes?** Only 5m was searched at full history (1m/15m/MBP1 data stop at 2025-05).
  Extending the two winning families to 1m/3m/15m is the top follow-up.
- **Which exits work best?** **Hold-to-EOD and high RR (rr2/rr3)** maximize PF; **rr1 (1:1)** maximizes
  win rate but lowest PF and most cost-sensitive. ATR-trailing was *mediocre* (median PF 0.97) — it
  gave back too much. Best all-round = **rr2 with a wide-ish ATR stop (2.5×)**.
- **Which confirmations are useless?** **Raw MSS and CHoCH alone (PF ~0.90, 0 passers), raw
  Bollinger/VWAP mean-reversion, ORB-fade, prior-day-high/low breakout, CVD-divergence.** Displacement
  only becomes useful *combined with a sweep*; structure breaks only when *combined* with a setup.
- **Which filters reduce drawdown / improve win rate?** **Requiring an inside-day** before a breakout
  (cuts false breaks → DD collapses to <$2k). **Requiring displacement on the sweep** lifts win rate
  to ~60%. **NY-AM-only** filtering improves quality vs ALLDAY/PM.
- **Which families to abandon?** Mean-reversion (vwap/bollinger/keltner revert), all **fades**, raw
  MSS/CHoCH, PDH/PDL breakout, stochastic/RSI reversal — no after-cost edge in any session/exit.
- **Which deserve deeper research?** (1) Build **true volume/market profile** (VAH/VAL/POC, HVN/LVN)
  — the biggest untested family. (2) The two winners on **1m/3m/15m**. (3) Standalone **order-block /
  breaker** and **order-flow OFI/absorption** families. (4) Explicit **cross-timeframe** (HTF
  trend → LTF entry) stacking on top of inside-day / sweep-displacement.

---

## 5. Verdicts (no forced acceptance)

| candidate | verdict | reason |
|---|---|---|
| `inside_day_break\|RTH\|s2.5\|rr2` (+ `eod`, `rr1.5`) | **ACCEPT_FOR_FORWARD_PAPER** | best risk-adjusted: full-hist PF 1.37-1.44, DD<$2k, 5-6/6 yrs profitable, cost-robust to 2×, **survives walk-forward (train 1.55→test 1.26)**, no single-year dominance. *Not* live-ready: OOS PF only ~1.26. |
| `sweep_displacement\|NY_AM\|s1\|rr1.5` (+ `s1.5\|rr1`) | **ACCEPT_FOR_FORWARD_PAPER** | the high-win-rate option (49-60%), tiny DD, survives WFO (test PF 1.25-1.31). Caveat: 2 negative years → trade smaller / confirm forward. |
| `sweep_displacement\|ALLDAY\|s1\|rr2` | **ACCEPT_FOR_FORWARD_PAPER** | ret/DD 4.3, improving over time; lower PF (1.21) → secondary. |
| `ema_cross_21_50\|NY_AM\|eod`, `supertrend\|ALLDAY\|eod` | **NEEDS_MORE_DATA** | positive + high-capacity but low quality (Sharpe ~1, WR ~21-30%, 2 losing years, streak ≤19, deeper DD). Forward-paper only with caution. |
| `gap_fill` family | **NEEDS_MORE_DATA** | robust-but-marginal (PF ~1.09); not worth capital alone yet. |
| All `inside_day_break\|NY_AM\|eod`-type high-train-PF configs | **REJECT** | overfit the grid: train PF 1.77-1.90 → **test PF 0.83 (negative OOS)**; row-20 also yrConc 0.99. |
| order-flow (`cvd_trend`, `cvd_divergence`) | **REJECT** (edge) / **NEEDS_FORWARD_DATA** (data) | PF<1 over full history *and* MBP1 ends 2025-05 → cannot be validated on recent data regardless. |
| mean-reversion / fades / raw MSS-CHoCH / PDHL-breakout / RSI-Stoch reversal (591 screened out) | **REJECT** | no after-cost edge in any session/exit; reasons saved in `rejected.csv`. |
| **Any "confirmed live-ready" claim** | **withheld** | full-history is discovery only; 2026-H1 is weak/partial for every candidate → genuine forward data still required. |

### The honest bottom line
Removing the constraints **did** uncover a real intraday edge that the NY-AM / 1:1 / Topstep framing
had hidden — but it is **modest (after-cost PF ~1.3-1.44 in-sample, ~1.25 out-of-sample), not the
PF≥2 the brief hoped for.** That ceiling is now confirmed from *both* directions (constrained: phases
16-27; unrestricted: phase 29). The best path forward is to **forward-paper `inside_day_break|RTH|rr2`
and `sweep_displacement|NY_AM`**, and to build the one genuinely untested family — **volume/market
profile** — before concluding the search.

---

## Reproduction

```bash
python scripts/phase29_run.py all          # build frame once, run 942 candidates, analyze, robust
pytest tests/unit/test_phase29_integrity.py tests/unit/test_phase27_integrity.py   # 12 integrity tests
# artifacts: /Volumes/T9/nq data/curated/artifacts/phase29/run1/
#   leaderboard.csv  candidates.csv  rejected.csv  candidates_full.json  robust.json  progress.log
```
