# Phase 39 — Turning robust_no_beta@2 into a Topstep payout-capable daily workhorse

Focused engineering on the Phase-38 lead candidate (NOT a new search). Removed beta, killed redundancy,
added a regime gate + adaptive sizing, and ran Topstep payout/blow Monte-Carlo across 50K/100K/150K.
**Objective = payout ÷ blow** (not net profit). All gates/sizing decided on TRAIN (2021-22), validated
on val+test (2023-26) and under a streak-preserving block bootstrap. Audited per-trade data from Phase 38.

**Headline.**

```
THE CLEANING WORKED. Payout/blow ratio improved 0.89 -> 2.1-2.4 and payout probability ~5x, validated
OOS and under block bootstrap. Drivers: (1) removed dow_long_mon = pure long beta; (2) collapsed 5
correlated breakout/momentum sleeves to one; (3) GATE: skip "neutral-trend" days (every sleeve loses
them) -> halves blow; (4) diversified breadth at 1 MNQ raised daily Sharpe 1.15 -> 1.61.
BUT A HARD CEILING REMAINS: you get EITHER blow<8% (150K, payout ~5-17%) OR high payout (50K/100K,
payout 40-55% but blow 23-44%) -- NOT both. Topstep's tight trailing geometry caps payout/blow at this
Sharpe. The best "blow<8%" config pays out ~16-17% per 120-day window on a 150K (ratio ~2.1).
SURPRISE (good): the de-beta'd book had its BEST year in the 2022 bear -- it is two-sided, NOT
bull-dependent. The Phase-38 "long-biased bull" caveat is largely RETRACTED for this cleaned book.
```

Verdict: **CONDITIONAL PASS / deployable to forward-paper.** The BALANCED version on a **150K** is
genuinely Topstep payout-positive (payout ≈ 2× blow, blow ~7-10%, trades 85% of days). It is **not** a
fast money machine, and the 50K/100K aggressive variants are **not** viable (blow too high). For pure
payout odds the infrequent **Phase-36 cycle still wins** (62%/3.4%, ~once/week) — Phase 39's win is
making a *trades-most-days* book payout-positive.

---

## 1-3. The cleaned book — exact rules, stats, correlations, and what was removed
**robust_no_beta@2** held 9 sleeves. Diagnostics (daily-PnL corr to always-long baseline; long share;
PnL on market up-days vs down-days):

| sleeve | trades | days% | WR | PF | avgW/avgL | maxDD | long% | corr→beta | up-day $ | down-day $ | keep? |
|---|--:|--:|--:|--:|--|--:|--:|--:|--:|--:|--|
| trend_adx_cont | 1793 | 92% | 32 | 1.14 | 269/−108 | 4544 | 50% | 0.25 | 16,520 | 2,606 | **keep** (trend, +bear) |
| ir90_break | 1175 | 79% | 40 | 1.13 | 192/−112 | 3202 | 54% | 0.15 | 12,662 | −2,155 | **keep** (breakout) |
| asia_sweep_fade | 1384 | 74% | 36 | 1.07 | 162/−86 | 2903 | 49% | −0.07 | −1,939 | **7,015** | **keep** (hedge) |
| ir60_fade | 1269 | 81% | 39 | 1.13 | 175/−101 | 3277 | 52% | 0.08 | 8,081 | 2,011 | **keep** (MR, robust) |
| disp_pb_fvg | 1803 | 88% | 35 | 1.02 | 146/−78 | 3263 | 47% | 0.01 | −1,751 | 3,514 | **keep** (uncorrelated) |
| od_cont_struct | 1457 | 91% | 38 | 1.07 | 236/−137 | 5117 | 53% | 0.14 | 13,970 | −5,073 | ✂ redundant (corr 0.86 w/ir30) |
| ir30_break | 1496 | 88% | 40 | 1.13 | 229/−134 | 5922 | 52% | 0.14 | 17,440 | −2,166 | ✂ redundant (corr 0.82 w/ir60) |
| ir60_break | 1357 | 88% | 41 | 1.12 | 203/−128 | 3894 | 54% | 0.15 | 14,071 | −1,921 | ✂ redundant (corr 0.85 w/ir90) |
| **dow_long_mon** | 254 | 18% | 56 | 1.12 | 195/−218 | 2379 | **100%** | 0.25 | **20,240** | **−17,382** | ✂ **PURE LONG BETA** |

- **`dow_long_mon` is secretly just long beta** (100% long, +$20.2K on up-days / −$17.4K on down-days,
  lost money in the 2022 bear). **Removed.**
- **The 5 breakout/momentum sleeves are near-duplicates** (od_cont/ir30/ir60/ir90 pairwise corr
  0.72–0.86). Kept **ir90_break** (best high-vol behaviour, balanced OOS); dropped the other three.
- **Genuine diversifiers kept** — `asia_sweep_fade` (corr −0.07, makes +$7,015 on *down*-days),
  `ir60_fade`, `disp_pb_fvg` profit when the market falls; they hedge `trend_adx_cont`/`ir90_break`.

**Cleaned book = 5 sleeves** (`trend_adx_cont, ir90_break, asia_sweep_fade, ir60_fade, disp_pb_fvg`).
Exact rules (each: first signal/day, EOD-flat, audited next-bar-open fills + costs):

```
trend_adx_cont  stop 2.0×ATR, exit EOD : RTH 09:30-15:00; LONG if ADX>25 & trend_1h_bull & close>VWAP
                                          & close>prev_close ; SHORT if ADX>25 & htf_bear & close<VWAP
                                          & close<prev_close.
ir90_break      stop 2.0×ATR, exit EOD : 11:00-14:30; LONG when close crosses ABOVE the 09:30-11:00
                                          (first-90-min) high ; SHORT crosses BELOW first-90-min low.
asia_sweep_fade stop 1.5×ATR, tgt 2R   : RTH; LONG if low<Asia-session-low & close>Asia-low (sweep+
                                          reclaim) ; SHORT if high>Asia-high & close<Asia-high.
ir60_fade       stop 1.5×ATR, tgt 2R   : 10:30-14:00; SHORT if high>first-60-min high & close back
                                          below it ; LONG if low<first-60-min low & close back above.
disp_pb_fvg     stop 1.5×ATR, tgt 2R   : RTH; LONG if displacement & price in bull FVG & up-bar ;
                                          SHORT if displacement & price in bear FVG & down-bar.
GATE  : skip the whole day if opening regime is "neutral" (htf neither bull nor bear).  ~15% of days.
SIZE  : take up to 3 of these signals per day (chronological), 1 MNQ each.
```

## 4. 2022 bear / regime stress (the good news)
Per-sleeve PnL by year — **2022 (NQ −33%) was the cleaned book's BEST year**, the dumb baseline's worst:

| sleeve | 2021 | **2022 (bear)** | 2023 | 2024 | 2025 | 2026 |
|---|--:|--:|--:|--:|--:|--:|
| trend_adx_cont | 3,513 | **4,259** | 2,532 | 903 | 8,275 | −356 |
| ir90_break | 369 | **5,417** | 645 | 1,733 | −52 | 2,395 |
| ir60_fade | 1,525 | **3,543** | 271 | 1,743 | 2,864 | 145 |
| asia_sweep_fade | 2,713 | 509 | 42 | 1,189 | 993 | −371 |
| dow_long_mon (removed) | 1,184 | **−551** | 2,316 | −362 | −1,255 | 1,527 |
| BASELINE_long (dumb) | 3,549 | **1,960** | 6,179 | 290 | 4,768 | 6,995 |

The breakout + trend + MR sleeves *thrive* in a high-vol bear (two-sided); only the beta sleeve lost.
**This retracts the Phase-38 "long-biased, untested in a bear" caveat** for the cleaned book — it
handled the one bear in-sample well. (Residual caveat: a *low-volatility grind-down* bear is still
untested; 2022 was a high-vol bear, which breakouts like.)

## 5. Regime filters — only one survives
PnL by PIT-safe opening regime (measured at the first RTH bar): **every sleeve loses on neutral-trend
days** (train AND full) → the **neutral-trend gate is the one robust filter**. Chop/low-ADX days are
*not* a clean gate (sleeves stay positive in chop on train), and the frame's `vol_regime` is useless
(1343/1406 days flagged "high"); a self-built ATR-tercile vol split helps sizing, not gating.

| regime (train PnL, all sleeves) | bull | bear | **neutral** |
|---|--:|--:|--:|
| trend_adx_cont | +8,691 | +463 | **−1,382** |
| ir90_break | +4,110 | +1,836 | **−160** |
| ir60_fade | +5,386 | +130 | **−448** |
| ir30/60_break, od_cont, disp_pb | all + | all + | **all negative** |

## 6. Adaptive sizing — breadth beats a size multiplier
Tested 1/2/3 MNQ flat, and adaptive (2–3 MNQ on high-confidence = clear-trend + high-ADX days). Finding:
**raising trades/day (cap) at 1 MNQ improves payout/blow more than raising contract size**, because the
extra trades are diversified (mean rises faster than vol; Sharpe 1.46→1.61), whereas a flat size
multiplier scales blow 1:1. Adaptive 1→2 helped modestly but did not beat `cap3 @ 1 MNQ`. So the
recommended sizing is **flat 1 MNQ, cap 2 (safe) or 3 (balanced)**, scale via *account*, not contracts.

## 7. Payout/blow across accounts (the frontier) — iid / OOS(23-26) / block-bootstrap
Cleaned+gated book; payout = P(reach profit target before trailing-DD blow in a 120-day window, with
the 50% consistency rule + 2-day min):

| config | account | payout | blow | ratio | OOS pay/blow | block pay/blow |
|---|---|--:|--:|--:|--|--|
| RNB2 (start) | 150K | 3.6% | 4.0% | 0.89 | 3.8 / 6.4 | 2.8 / 4.0 |
| **SAFE** clean5+gate ≤2 MNQ | **150K** | 5.4% | **2.2%** | **2.44** | 5.3 / 4.1 | 4.9 / 2.6 |
| **BALANCED** clean5+gate ≤3 MNQ | **150K** | **16.0%** | 7.5% | **2.13** | 17.1 / 10.2 | 16.3 / 7.0 |
| AGGRESSIVE clean5+gate ≤3 MNQ | 100K | 41.1% | 23.4% | 1.75 | — | — |
| (max-payout) clean5+gate ≤3 MNQ | 50K | 54.2% | 41.3% | 1.31 | 50.5 / 44.4 | 55.8 / 39.6 |

Daily-loss-limit: Topstep base accounts have **no separate daily limit** (config `daily_loss_limit=None`);
the **trailing $2,000/$3,000/$4,500 MLL is the binding constraint** and is what the MC enforces.

## 8-10. Final deliverables

**Best SAFE version** — clean5 + neutral-gate, **1 MNQ, ≤2 trades/day, 150K account.**
Days 85% · payout ~5% · **blow 2–4% (all methods)** · ratio ~2.4 · Sharpe 1.46. Capital-preservation:
almost never blows, but slow to reach the $9,000 payout. *Deployable to paper: yes.*

**Best BALANCED version (recommended)** — clean5 + neutral-gate, **1 MNQ, ≤3 trades/day, 150K account.**
Days 85% · **payout 16–17%** (iid/OOS/block agree) · blow **7% (iid/block), ~10% (OOS)** · ratio ~2.1 ·
Sharpe 1.61. Payout ~5× the starting candidate at ~2:1 odds over blow. *Meets the targets* (70–95% days
✓; payout materially up ✓; blow ~7–10%, i.e. at the edge of the 5–8% band). **Topstep payout-positive.**

**Best AGGRESSIVE version** — same book, **100K account** (or 50K for max payout).
Days 85% · payout 41% (100K) / 54% (50K) · blow 23% (100K) / 41% (50K) · ratio 1.75 / 1.31. Fast payout
but **fails the blow<8% target** — only sensible if you treat blown evals as cheap, repeatable resets
(EV is still positive when payout > blow and a payout pays more than the eval fee). Not recommended as a
single-account plan.

### Is it deployable to paper / Topstep-viable?
- **Deployable to forward-paper: YES** — all three. Start with SAFE or BALANCED on a 150K.
- **Topstep-viable: CONDITIONALLY YES (BALANCED on 150K).** It is genuinely payout-positive (≈2:1 over
  blow), trades 85% of days, blow ~7–10%/120-day window, and survived OOS + block-bootstrap + the 2022
  bear. It is **not** strong enough to be a high-frequency cash machine, and the 50K/100K aggressive
  variants are **not** viable (blow 23–41%).
- **Still weaker than the Phase-36 infrequent cycle** (62% payout / 3.4% blow, ~once/week). Phase 39's
  contribution is narrower but real: it makes a *trades-most-days* book payout-positive for the first
  time. If "trade most days" is a hard requirement → use BALANCED/150K; if not → Phase-36 wins on odds.

### Bottom line (no spin)
Cleaning the candidate **worked and is robust**: removing one beta sleeve, deduping the breakouts, and
gating out trendless days lifted payout/blow from 0.89 to ~2.1–2.4 and payout ~5×, holding up OOS,
under streak-preserving bootstrap, and through the 2022 bear (which it *profited* from — retracting the
prior bull-regime worry). But Topstep's trailing geometry imposes a hard ceiling at this Sharpe: you
can have low blow *or* high payout, not both. The honest best is the **BALANCED 150K** book — ~16–17%
payout / ~7–10% blow / 85% of days — **conditionally Topstep-viable and ready for forward-paper**, while
the infrequent Phase-36 cycle remains the higher-odds payout vehicle for anyone not bound to "most days".

```
Reproduction:
python scripts/phase39_regime.py   # PIT-safe day regimes + per-sleeve per-regime PnL -> day_regime.json
python scripts/phase39_sim.py      # cleaned book + gate + sizing + payout MC (50K/100K/150K) -> _sim.json
```
