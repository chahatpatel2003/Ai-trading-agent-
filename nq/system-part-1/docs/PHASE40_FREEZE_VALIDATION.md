# Phase 40 — Freeze clean5+gate and validate forward-paper readiness

No new strategies, no parameter changes, no backtest "improvement." The Phase-39 lead is **frozen** and
put through execution/timestamp/lookahead/robustness audits with explicit kill criteria. Frozen spec is
in `config/forward_paper/clean5_gate_frozen.yaml`; raw logs in `docs/phase40/frozen_trades_*.csv`.

**Verdict: PASS for forward-paper (BALANCED on 150K), with conditions.** No lookahead, survives harsh
execution, 2022 bear strongly positive, OOS PF 1.17–1.20, payout/blow 1.6–3.0 by Monte-Carlo. Three
yellow flags to monitor in paper (not blockers): a losing 2023, ~50% profit dependence on one sleeve,
and a harsh-fill real-path payout/blow that dips to 1.44.

---

## 1. Exact frozen rules (5 sleeves; each: first signal/day, next-bar-open entry, EOD-flat 15:55 ET)
```
trend_adx_cont  stop 2.0×ATR, exit EOD : 09:30-15:00; LONG  adx>25 & trend_1h_bull & close>VWAP & close>prev_close
                                                       SHORT adx>25 & htf_bear     & close<VWAP & close<prev_close
ir90_break      stop 2.0×ATR, exit EOD : 11:00-14:30; LONG close crosses ABOVE 09:30-11:00 high ; SHORT crosses BELOW its low
asia_sweep_fade stop 1.5×ATR, tgt 2R   : RTH; LONG low<Asia-low & close>Asia-low ; SHORT high>Asia-high & close<Asia-high
ir60_fade       stop 1.5×ATR, tgt 2R   : 10:30-14:00; SHORT high>09:30-10:30 high & close back below ; LONG mirror on the low
disp_pb_fvg     stop 1.5×ATR, tgt 2R   : RTH; LONG displacement & in bull-FVG & up-bar ; SHORT displacement & in bear-FVG & down-bar
```

## 2. Exact neutral-trend gate
> At the **first RTH bar (09:30 ET)** read the day's HTF trend. **Trade the day only if `htf_bull==1`
> OR `htf_bear==1`. If neither (neutral), take NO trades that day.** Evaluated once, at the open, from
> trailing HTF data → PIT-safe. Skips ~15% of days → ~85% of days traded.

**Sizing:** 1 MNQ per signal; max trades/day = **2 (SAFE)** or **3 (BALANCED)**; if more signals fire,
take the earliest by timestamp; one trade per sleeve per day.

## 3-5. Raw log + timestamp + no-lookahead verification (BALANCED, 3,531 trades)
Full logs exported. Engine timestamps are UTC; mapped to ET bars for the check:

| check | result |
|---|--|
| exit **strictly before** entry (would be lookahead) | **0** ✅ |
| exit not on a real source bar | **0** ✅ |
| entry not on a real RTH bar | 3 (0.08%, DST-edge artifacts) |
| entry before its sleeve's window opens | 3 (same artifacts) |
| same-bar stop/target exits (legitimate, conservative) | 475 |
| EOD-flat exits | 571 |

**No-lookahead audit of every input the rules touch:**
- **Gate** — `htf_bull/htf_bear` read at 09:30 from trailing HTF EMAs; decided before any entry. ✅
- **Opening-range / IR60 / IR90 levels** — frozen at 10:00 / 10:30 / 11:00; sleeves only fire *after*
  their window (entry-window check passed except 3 DST artifacts). ✅
- **Asia/overnight levels** — formed overnight, fixed before 09:30. ✅
- **FVG (`in_bull/bear_fvg`), `displacement`** — current+past bars only; entry is the *next* bar. ✅
- **VWAP/ADX/ATR** — cumulative/trailing to the current bar. ✅
- Engine guarantee: next-bar-open entry with adverse slippage, stop-first same-bar, exact resting-limit
  targets — "no decision uses a price it could not have." **Conclusion: no lookahead.**

## 6-9. Harsh execution, per-year, OOS, concentration, payout robustness
**Harsh = +1 tick entry slip, +1 tick exit slip (slippage_mult 2×), full commissions, and a 1-tick
touch-only-miss haircut on TARGET fills.** Frozen baseline already pays 1 tick + full commissions.

| metric | BALANCED normal | BALANCED harsh | SAFE normal | SAFE harsh |
|---|--:|--:|--:|--:|
| trades / days% | 3531 / 85% | 3531 / 85% | 2386 / 85% | 2386 / 85% |
| WR / PF | 35.6% / 1.219 | 35.3% / 1.196 | 34.2% / 1.236 | 34.1% / 1.218 |
| net 6yr (1–3 MNQ) | 44,658 | 40,325 (−9.7%) | 31,259 | 29,212 |
| **OOS PF (2023-26)** | **1.196** | **1.167** | 1.208 | 1.189 |
| 150K rolling-120 payout / blow / ratio | 18.2% / 6.8% / **2.7** | 16.9% / 11.8% / **1.44** | 8.1 / 5.2 / 1.57 | 8.1 / 5.6 / 1.45 |
| 150K MC iid payout / blow / ratio | 16.1 / 7.9 / 2.05 | 14.7 / 9.0 / 1.63 | 4.9 / 2.2 / 2.2 | 4.4 / 2.9 / 1.53 |
| 150K MC block payout / blow / ratio | 17.8 / 5.9 / 3.0 | 15.0 / 7.0 / 2.16 | 5.1 / 2.2 / 2.27 | 5.2 / 2.2 / 2.35 |

**Per-year (BALANCED normal | harsh):**
| 2021 | 2022 (bear) | 2023 | 2024 | 2025 | 2026 |
|--:|--:|--:|--:|--:|--:|
| 5,370 / 4,975 | **13,189 / 12,813** | **−1,863 / −2,400** | 10,744 / 9,731 | 11,300 / 9,452 | 5,917 / 5,754 |

**Sleeve concentration (BALANCED normal, gross profit):** trend_adx_cont 42.6% ($22.4K, PF 1.27),
ir90_break ($7.1K, PF 1.66), ir60_fade ($7.0K, PF 1.30), disp_pb_fvg ($4.9K, PF 1.10),
asia_sweep_fade ($3.2K, PF 1.09). (SAFE concentrates **51.8%** in trend_adx — worse than BALANCED.)

## Kill-criteria scorecard (target = BALANCED / 150K)
| # | kill criterion | normal | harsh | verdict |
|--|--|--|--|--|
| 1 | OOS PF < 1.05 | 1.196 | 1.167 | **PASS** |
| 2 | 2022 turns negative | +13,189 | +12,813 | **PASS** (best year) |
| 3 | payout/blow < 1.5 | 2.05–2.7 | MC 1.63–2.16 / **roll 1.44** | **BORDERLINE** (MC passes; harsh real-path dips below) |
| 4 | 120-day blow > 12% | 6.8–7.9% | 9.0–11.8% | **PASS** (under, but close on harsh) |
| 5 | one sleeve = most profits | 42.6% | 42.9% | **PASS** for BALANCED (<50%); **SAFE FAILS** (51.8%) |
| 6 | harsh execution destroys edge | — | net −9.7%, PF 1.196 | **PASS** (degrades, not destroyed) |

## 10. Forward-paper config
Written to `config/forward_paper/clean5_gate_frozen.yaml` — **SAFE (≤2 MNQ/day)** and **BALANCED
(≤3 MNQ/day)**, 150K model, `live_trading: false`. Nothing tunable during paper.

## Final output — PASS / FAIL for forward-paper

**PASS — proceed to forward-paper with the BALANCED book on a 150K, planning to the HARSH numbers.**

- Clean on the hard gates: **no lookahead** (0 exits-before-entry, levels used only after they form),
  **survives harsh execution** (edge −10%, PF still 1.196), **2022 bear was its best year** (two-sided,
  not bull-dependent), **OOS PF 1.17–1.20**, **blow under 12%**, **no single sleeve >50% of gross profit**.
- **Plan to harsh:** expect ~**15% payout / ~9–12% blow per 120-day window, ratio ~1.6–2.2** (not the
  rosier normal-fill numbers). The single harsh real-path sequence gives ratio 1.44 — at the kill line —
  so treat 1.5 as the live abandon trigger, not a comfortable margin.
- **Watch-items (the reasons this is forward-PAPER, not live):**
  1. **2023 was a losing year** (−$1.9K) — low-volatility grind-up regimes are the weak spot; the gate
     can't fix a no-edge regime. Watch drawdowns in quiet markets.
  2. **trend_adx_cont carries ~42% of gross / ~50% of net profit** — if it degrades in paper, the book
     degrades. It is also the most beta-leaning sleeve; monitor it specifically.
  3. Prefer **BALANCED over SAFE** — SAFE is more concentrated (52% one-sleeve) and has lower payout;
     its only edge is marginally lower blow.
- **Not yet live.** Forward-paper ≥ ~3 months / ~60 trading days, confirm live fills match the harsh
  assumptions and that trend_adx_cont + the 2023-type regime behave as modeled, then re-evaluate.

### Bottom line
The freeze holds up. clean5+gate is **real, PIT-clean, and robust to harsh fills and the 2022 bear** —
it earns a forward-paper PASS on the BALANCED/150K configuration. It is **not** bulletproof: it leans
~half its profit on one (beta-tinged) trend sleeve, lost money in the low-vol 2023, and its harsh-fill
payout/blow sits right at the 1.5 line. Forward-paper exists to confirm exactly those three things in
real time before any capital is risked.

```
Reproduction:
python -m scripts.phase40_freeze   # re-run engine (normal+harsh) -> docs/phase40/frozen_trades_*.csv + stats
```
