# Phase 20 — Classic Indicator Strategy Discovery (last non-MBP1, bar-based phase)

**Purpose:** before pivoting to MBP1 order-flow, run one clean standalone indicator phase so we
don't skip a major class of systematic strategies. Find whether simple classic indicators can make
a payout-grade NQ NY-AM Topstep strategy.

**Status: built, run end-to-end, verdict = NO business-worthy strategy — all 105 candidates REJECT
at validation.** Classic indicators fail exactly as the Phase-19 edge map predicted (NQ NY-AM has
no directional edge before costs). Lockbox 2025-2026 **never touched** (nothing earned it). No
forced acceptance.

---

## What was built

1. **Indicator feature layer** (`features/indicators.py`, tested): vectorized PIT-safe EMA/SMA,
   MACD, Stochastic, Bollinger, Keltner, Donchian, Wilder ADX (+DI/−DI) — reusing base `atr`/`rsi`/
   `vwap`.
2. **15 indicator families** (`phase20_families.py`) across all 10 requested classes + combos:
   MA trend (EMA cross / pullback / VWAP-trend / ATR-trail), Bollinger (mean-rev + breakout), RSI
   (reversal + trend), MACD cross, Stochastic reversal, ADX trend, Keltner mean-rev, Donchian
   breakout, and combos (EMA+RSI pullback, Donchian+ADX). NY-AM only, ATR-scaled stops/targets,
   deliberately tiny grids (no overfitting).
3. **Reused the Phase-16/18 payout rig wholesale** — gauntlet + DSR + cost-stress, daily-control
   sweep (profit-target/max-loss/max-trades), two-stage funded scaling, fleet projection, and the
   `ACCEPT_FOR_FORWARD_PAPER`-ceiling verdict. CLI `phase20 search/freeze/lockbox` + `report`.

## Data protocol

Search/train **2021–2023**, validation **2024** (never past 2024-12-31), reserved **pristine**
lockbox **2025–2026**. Searched **1m, 5m, 15m**. Because nothing cleared validation, the lockbox
was **not touched**.

## Result — 0 of 105 cleared validation

35 candidates × 3 timeframes = 105. Gauntlet verdicts: **REJECT 99, NEEDS_MORE_DATA 6, ACCEPT 0.**

Three distinct failure modes (rejection reasons, verbatim):

| failure mode | example | reason |
|---|---|---|
| **No edge before costs** | donchian 5m, ema_cross 1m | "diagnostics-first: no directional edge before costs (gross ≤ 0)" |
| **Blows the account** | donchian / bb_breakout 15m | "blow-up probability 52.8–60.2% > 10%" (positive net, but ruinous DD) |
| **Cost-stress / unstable** | atr_trend, ema_pullback | "fails cost/slippage stress …; walk-forward positive in 25–42% (<50%)" |

Concrete validation-year (2024) economics — none approach PF 1.30:

| family (5m) | trades | win rate | **PF** | net |
|---|--:|--:|--:|--:|
| Donchian breakout (best trend) | 704 | 41.5% | **1.02** | +$1,535 |
| EMA cross | 1,666 | 39.9% | **0.95** | −$10,840 |
| Bollinger mean-reversion | 921 | 36.6% | **0.84** | −$21,403 |
| RSI reversal | 608 | 34.9% | **0.77** | −$20,307 |

- **Trend/breakout** (Donchian, Bollinger breakout, ATR-trail) occasionally print positive net by
  catching a few 2024 trend moves, but at **52–60% risk of ruin** — they blow the locked-buffer
  account, so they are untradeable; PF still only ~1.0.
- **Mean-reversion / oscillators** (Bollinger MR, RSI, Stochastic, Keltner) are **net-losing**
  (PF 0.77–0.84, win rate 35–37%) — costs erase any reversion tendency, consistent with the
  Phase-19 finding that MFE ≈ MAE.
- No family reaches **PF ≥ 1.30, ruin ≤ 10%, and cost-stress survival simultaneously** on any
  timeframe.

## Verdict

```
REJECT (all 105). Classic indicators do not produce a payout-grade NQ NY-AM strategy.
Lockbox 2025-2026 left pristine (nothing earned a lockbox test).
```

This was the **last non-MBP1, price/bar-based strategy phase.** Across Phases 16–20, every
bar/price-action approach has failed on NQ-only NY-AM data:

| phase | approach | outcome |
|---|---|---|
| 16 | payout-first structure/liquidity | REJECT (collapsed OOS) |
| 17 | A+ large-R asymmetric | REJECT (collapsed OOS) |
| 18 | high-win-rate scalps | REJECT (no edge before costs) |
| 19 | edge-discovery map | **no directional edge** (MFE ≈ MAE everywhere) |
| 20 | classic indicators | REJECT (no edge / blow-up / cost drag) |

## Recommendation

**Pivot to MBP1 order-flow microstructure, or pause the NQ-only NY-AM bot search.** The structural
finding (Phase 19) and five independent strategy phases agree: **NQ NY-AM bar/price-action carries
no payout-grade directional edge after costs.** The only materially different, still-untested data
edge is **MBP1 order-flow** (CVD, imbalance, absorption, liquidity) on clean MBP1 periods — and
even that is verdict-capped at `NEEDS_FORWARD_DATA` (the 2026-06 lockbox month has no MBP1). If
MBP1 also shows no edge, the honest conclusion is that an NQ-only NY-AM Topstep payout bot is not
achievable with the available data, and the search should stop.

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **250 tests**, incl. indicator + Phase-20 tests).
- ✅ Lockbox 2025-2026 **never touched** (0 validation survivors).
- ✅ Every candidate has a verdict + reason; payout-first bar enforced; diagnostics-first early
  rejection active ("no edge before costs", "blows account", "cost-stress fails").
- ✅ No forced acceptance, no tiny-edge acceptance.
- ✅ Final report states plainly: classic indicators fail → pivot to MBP1 or pause.

## Reproduction

```
nqlab phase20 search --timeframes 5m,15m,1m --account 150k   # train 2021-23 / val 2024
nqlab phase20 freeze --top-n 8                                # 0 survivors -> nothing frozen
# (no lockbox run: nothing cleared validation; 2025-2026 stays pristine)
```
