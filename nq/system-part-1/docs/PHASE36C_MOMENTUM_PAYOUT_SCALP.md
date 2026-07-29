# Phase 36C — Momentum Burst Payout Scalp (3-5 point NQ quick scalps)

One specific hypothesis: short-lived **momentum bursts continue 3-5 points** with a high enough win
rate (after costs/realistic fills) to build Topstep payout days. Tested as pure continuation (not
mean-reversion, not slow idb/swd) on Phase-33 sub-minute bars with the conservative-fill scalp engine.
Honesty metric: **edge over the breakeven win rate** (≥5pp required).

**Headline (the hypothesis is refuted).**

```
3-5pt momentum scalping does NOT work — it has NEGATIVE edge, even before costs.
  • Realistic (1-tick-through fill, 1x cost): best edge-over-breakeven = -8.1pp, best PF = 0.72.
  • Zero-cost touch-fill (most optimistic possible): best edge = -7.0pp, best PF = 0.76.
  • 0 of 720 momentum candidates clear edge>=5pp & PF>=1.4 (or even reach PF 1.0).
Momentum-burst win rates (47-51%) are BELOW the geometric breakeven (56-59%): after a sub-minute
thrust, the 3-5pt continuation is hit LESS often than the target/stop geometry predicts — the burst is
already spent and slightly reverts by the time you can enter on the next bar (entry-after-signal).
This is WORSE than Phase 33 generic scalping (edge ~0) and strictly worse than the Phase 36 payout
baseline (62% payout). Daily-payout / lifecycle sims are moot: a negative-PF strategy fails every
account by construction.
```

Verdict: **REJECT_NO_EDGE** (negative edge) + **REJECT_WORSE_THAN_BASELINE**. Abandon momentum payout scalping.

---

## 1. Universe
- **Triggers (A-F):** consecutive-candle momentum (3 up/down closes, strong close, ≥3pt thrust),
  range-expansion burst (≥1.5× rolling-median range, strong close, breaks micro high), breakout
  momentum (clears rolling 10-min high by ≥1 tick), VWAP-slope continuation, opening-drive (first
  30 min thrust). **G (order-flow OFI/delta): MISSING** — the sub-minute bars are trades-only OHLCV
  (MBP1 ends 2025-05); reported, not faked.
- **Bars:** 15s + 1m (Phase 33, RTH front-month, all years). **Targets** 3/4/5 pt × **stops** 3/4/5/6 pt
  × **fill** {touch, 1-tick-through} × **cost** {1×, 2×}, hold ≤3 min, entry next-bar open, stop-first
  same-bar. **720 candidates.**

## 2. Breakeven math (1 MNQ, $2/pt, RT comm $0.50; entry+stop slip 1 tick; target = exact limit)
| target/stop | net win | net loss | breakeven WR | WR for PF 1.4 | WR for PF 1.7 |
|---|--:|--:|--:|--:|--:|
| 5 / 3 | $9.00 | −$7.50 | 0.455 | 0.538 | 0.586 |
| 5 / 4 | $9.00 | −$9.50 | 0.514 | 0.596 | 0.642 |
| 5 / 5 | $9.00 | −$11.50 | 0.561 | 0.641 | 0.685 |
| 4 / 4 | $7.00 | −$9.50 | 0.576 | 0.657 | 0.701 |
| 4 / 6 | $7.00 | −$13.50 | 0.659 | 0.730 | 0.766 |
| 3 / 3 | $5.00 | −$7.50 | 0.600 | 0.677 | 0.718 |

> Even the most favorable geometry (5pt target / 3pt stop) needs **WR ≥ 53.8% for PF 1.4.** The
> momentum triggers deliver ~47-51% — below breakeven, nowhere near the PF-1.4 requirement.

## 3. Momentum scalp leaderboard (realistic: 1-tick-through fill, 1× cost, n≥200)
| tf | trigger | dir | target | stop | trades/day | win% | breakeven% | **edge** | PF |
|---|---|--|--:|--:|--:|--:|--:|--:|--:|
| 15s | consec3 | long | 5 | 3 | high | 51.2% | 59.3% | **−8.1pp** | 0.72 |
| 15s | breakout_mom | long | 5 | 3 | high | 50.8% | 59.1% | −8.2pp | 0.72 |
| 15s | vwap_mom | long | 5 | 3 | high | 50.4% | 58.9% | −8.5pp | 0.72 |
| 15s | range_expansion | long | 5 | 3 | high | 50.0% | 59.0% | −9.0pp | 0.70 |
| 15s | consec3 | short | 5 | 3 | high | 50.3% | 59.3% | −9.0pp | 0.70 |

> **Every** momentum trigger, direction, target, stop, timeframe is **negative-edge**. The "best" is
> still −8.1pp / PF 0.72. opening-drive and 1m versions are the same or worse. **0 clear the bar.**

## 4–5. Daily payout / lifecycle / fill-cost sensitivity
**Moot.** With max PF 0.72 (negative edge over breakeven, negative expectancy per trade), the daily
P&L is net-losing, so every daily-target / account / sizing configuration produces losing days and
fails every account — there is nothing to make payout days from. Running the lifecycle MC on a
guaranteed loser is pointless: it trivially **loses to the Phase 36 baseline** (62% payout, +EV).
Fill/cost stress only makes it worse (1-tick-through already negative; 2× cost → PF ~0.6).

## 6. Final answers
- **Does 3-5pt momentum scalping work?** **No — it has negative edge** (worse than a coin flip).
- **Win rate achieved?** 47-51%. **Breakeven required?** 54-66% (depending on stop). **Real edge over
  breakeven?** **No — −7 to −9pp** (negative even at zero cost with touch fills).
- **Does 1 / 2 / 3 MNQ work?** No — sizing scales a negative edge into bigger losses; none work.
- **Can it hit +$150 days with ≤10 trades?** Only by luck on a given day; expectation is negative, so
  systematically **no**.
- **Can it beat the Phase 36 baseline?** **No** — it is negative-EV vs the baseline's +EV.
- **Forward-paper ready?** **No.** **Keep or abandon momentum payout scalping?** **Abandon.**

## Verdict
| candidate | verdict |
|---|---|
| all 3-5pt momentum scalps (consec / range-expansion / breakout / VWAP / opening-drive) | **REJECT_NO_EDGE** (negative edge) + **REJECT_WORSE_THAN_BASELINE** |
| order-flow momentum (G) | **NOT TESTED** — sub-minute bars lack OFI/delta (MBP1 ends 2025-05); would need an MBP1-joined sub-minute build, and given A-F's negative edge, low priority |

### Bottom line
The momentum hypothesis is **false for NQ at the 3-5pt scalp horizon**: bursts do **not** continue —
they slightly revert by the next bar, so continuation entries win *below* the breakeven rate (−7 to
−9pp) even before costs. This is consistent with Phase 32 (continuation conditions showed no
asymmetry) and Phase 33 (scalp edge ≈ 0) — and momentum is actually the *anti*-edge. The only edge in
NQ remains **mean-reversion-from-extension** (swd / breakout-acceptance, Phases 30/34B), best deployed
via the **Phase 36 payout-first cycle** — not momentum scalping. Abandon this branch.

```
Reproduction:
python scripts/phase36c_run.py   # momentum triggers + breakeven math + edge grid -> docs/phase36c/
```
