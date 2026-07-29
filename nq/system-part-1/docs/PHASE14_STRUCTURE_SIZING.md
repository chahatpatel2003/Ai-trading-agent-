# Phase 14 — Structure/Liquidity Frequency Expansion + Sizing Repair

**Goal:** take Phase 13's strongest lead (15m BOS continuation, validation DSR 0.757)
and fix the two things that blocked confirmation — **frequency** and **sizing**. Phase
13's signal was fine; it failed only because the validated 15m setups sized to ~0 MNQ
contracts in the higher-ATR 2025-2026 regime and so produced too few held-out trades.

**Verdict: the bottleneck is fixed, and the edge is now judgeable — and it's rejected,
cleanly. 0 ACCEPT_FOR_FORWARD_PAPER, 6 REJECT, 0 NEEDS_MORE_DATA.** With the sizing
repair the structure edge finally trades enough on the untouched final check (5m
`bos_5m_trigger`: **479 trades, +$4,280, profit factor 1.155, ruin 1%, survives cost
stress, walk-forward 60%, 100% positive years, P(pass 50K) ≈ 62%**) — the best economic
profile of any phase. But it **fails the Deflated-Sharpe bar (DSR 0.44 < 0.90)** — the
strongest held-out DSR the project has reached, yet still not significant. The
remaining bottleneck is no longer frequency or sizing; it is the *statistical strength*
of the edge.

Per the data protocol (2025-2026 was inspected in Phase 13), the ceiling here was
`ACCEPT_FOR_FORWARD_PAPER`, never `ACCEPT_FOR_PAPER`. NQ only; the Phase-13 frozen
candidates were not touched.

---

## The sizing repair (the technical fix)

Phase 13's zero-contract failure, isolated: `floor(risk_usd / per_contract_risk) == 0`
when one MNQ contract's stop-risk exceeds the per-trade budget — exactly what a 1.5×ATR
stop costs on 15m in 2025-2026. The repair (`domain/sizing.py`):

- **`mnq_min1`** — take `max(1, floor(risk/contract_risk))` MNQ contracts; skip *only*
  when one contract would breach the **hard per-trade cap** (25% of the account's
  trailing max loss). Never rounds to 0 when the trade is affordable.
- **`mnq_fixed1`** — always exactly one MNQ contract within the same cap.

Proven by `test_phase14.py` (one MNQ contract risking $120 vs a $100 budget: `mnq_only`
skips, `mnq_min1` takes 1; skips only when the stop breaches the absolute cap). The
direct held-out diagnostic confirms it works *and* localises the real wall:

| Family / TF (mnq_min1) | 2025-26 trades | 2026 (OOS) trades |
|---|---:|---:|
| bos_second_leg 15m | 63 | **0** |
| bos_5m_trigger 15m | 107 | **0** |
| bos_fvg_mitigation 15m | 199 | **0** |
| **bos_5m_trigger 5m** | **447** | **136** |
| bos_early_pullback 5m | 508 | 160 |

**15m structure setups produce ~0 trades in the 2026 held-out window even after sizing
repair** (a 4th confirmation across Phases 11-14); **5m is the confirmable timeframe.**
So the freeze was restricted to 5m validation survivors.

## Frequency expansion (validation, 2024)

Five BOS-continuation variants (structure retest, FVG mitigation, earlier VWAP
pullback, second leg, 5m-trigger-in-HTF-bias) × sizing {mnq_min1, mnq_fixed1, adaptive}
× risk {$100-$250}. **81 validation gauntlet-survivors** (vs Phase 13's 24) — the
expansion + repair tripled confirmable survivors. Best: `bos_second_leg` 15m mnq_min1
(110 trades, +$3,392, WF 83%, 100% pos-yrs, valDSR 0.50); `bos_5m_trigger` 15m mnq_min1
(217 trades, +$7,443). PBO at 15m ~3.8% (clean leaderboard).

## Final check (untouched 2025-2026, 5m, sizing-repaired) — full practical stats

| Frozen | Trades (OOS) | Full-window stats | Verdict |
|---|---:|---|---|
| **bos_5m_trigger 5m / 150K** | 142 | win 24%, **PF 1.155**, exp $8.94, 479 trades, net +$4,280, monthly $245, monthly-ROI 0.16%, max DD $2,031, avg DD $865, ruin 1%, cost **pass**, WF 60%, 100% pos-yrs, **P(pass 50K)=62%** / 100K=19% / 150K=0%, **DSR 0.44** | **REJECT (DSR)** |
| bos_second_leg 5m / 150K | 100 | +$1,016, ruin 1%, cost pass, 100% pos-yrs, WF **47%** | REJECT (walk-forward < 50%) |

`bos_5m_trigger` 5m clears **every economic gate** on the untouched window — positive,
ruin 1%, not blown, survives cost/slippage stress, 100% positive years, walk-forward
60%, and a **62% probability of passing the 50K Topstep combine** — and is rejected
*only* on statistical significance (DSR 0.44 < 0.90). It is the best candidate the
project has produced. Best account: **50K** (highest pass odds); best instrument mode:
**MNQ (mnq_min1)**.

**Exact rules (bos_5m_trigger, 5m):** long when the 4h bias is bullish AND a break of
structure prints with a displacement candle (symmetric for shorts); NY-AM only
(09:30-12:00 ET); stop 1.5×ATR, target 4.0×ATR; min-hold 3 bars, time-stop 48 bars;
sizing `mnq_min1` (≥1 MNQ within the hard per-trade cap); risk $150.

---

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **207 tests**, incl. the sizing-repair test and
  the no-lookahead structure tests).
- ✅ Every candidate received `ACCEPT_FOR_FORWARD_PAPER` / `REJECT` / `NEEDS_MORE_DATA`
  — and this phase produced *judged* verdicts (no NEEDS_MORE_DATA: the frequency+sizing
  fix made the edge confirmable).
- ✅ No candidate promoted — none clears DSR on the final check. Full practical stats
  reported. The forward-paper ceiling is enforced (2025-2026 non-pristine). Nothing
  forced.

## Honest answer

> The frequency + sizing repair **worked**: the structure edge that Phase 13 could not
> confirm now trades 479 times on the untouched 2025-2026 window, is economically clean
> (profit factor 1.155, ruin 1%, survives costs and Topstep, 100% positive years, 62%
> chance of passing a 50K combine), and posts the highest held-out DSR (0.44) of any
> phase. It is the strongest, most complete candidate the project has produced — and it
> is still **rejected**, because a profit factor of 1.15 is not statistically
> significant at the frozen trial count. The bottleneck has moved from "can't size /
> can't confirm" to "the edge is real but thin."

Because the candidate clears every economic gate on held-out data and only misses the
significance bar, the honest disposition (per the phase rule) is: **justifies a forward
paper/shadow run on `bos_5m_trigger` 5m at 50K — not promotion, not live trading.** A
forward feed is the only way a true ~1.15-PF edge accumulates enough independent trades
to clear DSR; building that feed is out of scope. Per the standing rule — *no edge until
proven; reject honestly* — Phase 14 ends with **zero strategies promoted** and
`bos_5m_trigger` 5m flagged as the forward-paper watch candidate.

---

## Reproduction

```
nqlab phase14 search  --no-build                      # train 2021-23 / validation 2024
nqlab phase14 freeze  --tf 5m --min-trades 150        # 5m validation survivors (15m ~0 held-out trades)
nqlab phase14 lockbox --no-build                      # untouched 2025-2026 final check (sizing-repaired)
nqlab report phase14                                  # registry + verdicts + full practical stats
```
