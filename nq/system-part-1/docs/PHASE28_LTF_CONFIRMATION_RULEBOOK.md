# Phase 28 — Mechanical LTF Confirmation Rulebook for Manual HTF-Level Trading

Purpose change: not a full auto-strategy. The HTF level/bias is **given**; this phase measures which
**mechanical** LTF confirmation has the best 1:1 reliability *after* price touches the HTF level, so
a manual trader knows exactly what to wait for. Every confirmation is an exact PIT-safe code rule
(no discretionary words); each is gated by an HTF-POI touch (`at_*_poi_recent`) and direction
(support POI → long only, resistance POI → short only).

**Headline:** the best mechanical confirmation is **IFVG reclaim on 1m — 49% win at 1:1**
(validation 2024, n=2,943, PF 0.93). **No confirmation reaches the 55% bar**; even the best is a
near-coin-flip that loses to costs at strict 1:1. The ranking is still useful as guidance, and two
findings are robust: **1m beats 5m for every confirmation**, and **daily-bias/HTF-trend alignment
does not reliably help**.

---

## Ranking — validation 2024, 1:1 win rate after HTF-POI touch

| rank | confirmation | TF | n | **win@1:1** | PF | MFE/MAE | train win |
|---|---|--|--:|--:|--:|--:|--:|
| 1 | **IFVG reclaim** | 1m | 2,943 | **49%** | 0.93 | 1.02 | 48% |
| 2 | CISD | 1m | 1,377 | 48% | 0.90 | 1.04 | 49% |
| 3 | displacement | 1m | 1,485 | 48% | 0.89 | 1.03 | 49% |
| 4 | CHoCH | 1m | 1,192 | 48% | 0.89 | 1.01 | 46% |
| 5 | MSS | 1m | 330 | 47% | 0.86 | 1.03 | 47% |
| 6 | VWAP reclaim | 1m | 1,602 | 47% | 0.84 | 0.96 | 46% |
| 7 | sweep+reclaim / micro-swing | 1m | 1,059 | 44% | 0.74 | 0.92 | 48% |
| 8 | IFVG reclaim | 5m | 758 | 38% | 0.59 | 1.01 | 38% |
| 9 | CHoCH / sweep+reclaim | 5m | 283–342 | 36% | 0.53 | 1.0–1.1 | 37% |
| 10 | VWAP reclaim | 5m | 1,010 | 34% | 0.50 | 0.91 | 36% |
| 11 | displacement / CISD | 5m | 404–465 | 30% | 0.41 | ~0.96 | 35% |
| 12 | **MSS / MSS+displacement** | 5m | 109 | **28%** | 0.36 | 0.98 | 38% |

(Each confirmation's exact mechanical definition is in `src/nqlab/discovery/phase28.py`. All
columns referenced — sweep/reclaim, MSS/CHoCH, IFVG, CISD, displacement, VWAP — are audited
PIT-safe in Phase 27.)

## The manual trading rulebook (the 10-point output)

1. **Best LTF confirmation to wait for:** **1m IFVG reclaim** (49% at 1:1), or equivalently 1m
   **CISD / CHoCH / displacement** (48%). They are statistically interchangeable (~48–49%).
2. **Worst LTF confirmations to avoid:** anything on **5m**, especially **5m MSS / MSS+displacement
   (28%)** and **5m CISD/displacement (30%)** — materially worse than a coin flip.
3. **Best timeframe:** **1m, decisively.** Every confirmation wins ~10 percentage points more on 1m
   than 5m (e.g. IFVG 49% vs 38%; CHoCH 48% vs 36%; CISD 48% vs 30%). The 1m confirmation is earlier
   and tighter, so the fixed stop is closer to invalidation and the 1:1 target is reached more often.
4. **Best time window after HTF touch:** **no meaningful difference** — 1m IFVG wins 49% in both
   09:30–10:30 (n=1,343) and 10:30–12:00 (n=1,610). Trade either; do not over-filter by sub-window.
5. **Best stop placement:** a tight fixed stop (8–10 pt) sized so 1:1 is reachable. "Beyond the
   confirmation candle" performed similarly; wider HTF-extreme stops lower the win rate per R.
6. **Best target placement:** **1:1 is the most reliable.** Win rate falls at 1.5R/2R (the move
   rarely extends), and MFE/MAE ≈ 1.0 means there is no asymmetry to harvest with wider targets.
7. **Does daily bias help?** **No, not reliably.** Requiring daily-bias + HTF-trend alignment left
   the best confirmation's win rate essentially unchanged (IFVG 1m 49% → 49%) and only nudged a few
   smaller-sample ones (CHoCH 48% → 52% on fewer trades). It mostly cuts trade count without lifting
   reliability — consistent with Phase 25.
8. **Does HTF trend help?** Same answer — the 1H/4H alignment filter (part of the bias gate) did not
   improve win rate; it removed trades. Use it only to reduce frequency, not to raise the win rate.
9. **Mechanical checklist to TAKE the trade** (long; mirror for short):
   - [ ] Price has traded into a confirmed HTF **support** level within the last ~6 bars
     (`at_support_poi_recent`).
   - [ ] On the **1m** chart, a bullish confirmation prints: an **IFVG reclaim** (a prior bear FVG
     that a candle has closed back above), **or** CISD/CHoCH/displacement equivalent.
   - [ ] Time is **09:30–12:00 ET**.
   - [ ] **Enter at the next 1m candle open** after the confirmation close.
   - [ ] **Stop** 8–10 pt below entry (or just beyond the confirmation candle); **target = 1:1**.
   - [ ] Force flat before the cutoff; no overnight hold.
10. **Mechanical checklist to SKIP the trade:**
    - [ ] No HTF level touched in the last ~6 bars → **skip**.
    - [ ] Only a **5m** confirmation (no 1m confirmation) → **skip** (5m confirmations win ≤38%).
    - [ ] Confirmation is **MSS/displacement on 5m** → **skip** (worst, ~28–30%).
    - [ ] Wanting >1:1 (1.5R/2R) → **skip the wider target**; win rate drops, MFE≈MAE.
    - [ ] Outside 09:30–12:00 ET → **skip**.

## Honest verdict

No mechanical LTF confirmation clears the 55% / PF 1.30 manual-edge bar at 1:1 in 2024 validation —
the best (1m IFVG reclaim) is **49%, PF 0.93**, a near-coin-flip that loses to costs on its own. The
rulebook is therefore: **if you trade these reactions mechanically, use 1m IFVG-reclaim / CISD /
CHoCH (≈48–49%), never the 5m versions (≤38%) or 5m MSS (28%), target 1:1, and don't expect the
confirmation alone to give an edge — it removes ~half the bad reactions but not enough to be
profitable mechanically.** Any real edge must come from the *quality of the HTF level selection*
(the discretionary input this phase does not measure), not from the LTF confirmation rule. This is
consistent with Phases 19–27: NQ NY-AM 1:1 directional accuracy is ~50%, MFE ≈ MAE, with no
mechanical confirmation lifting it to a tradable level.

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **294 tests**, incl. Phase-28 confirmation tests).
- ✅ Every confirmation is mechanically defined (no discretionary words) over audited PIT-safe
  columns; conditioned strictly on an HTF-level touch occurring first.
- ✅ Lockbox 2025-2026 **untouched** (no confirmation earned a freeze — best 49% < 55% bar).
- ✅ Output is a ranked rulebook + take/skip checklists, not a single accept/reject.

## Reproduction

```
nqlab phase28 rulebook --timeframes 5m,1m   # ranked confirmation table + best/worst + windows + bias
```
