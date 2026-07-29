# Phase 30 — Volume / Market Profile Expansion + Winner Enhancement

Build the biggest missing family from Phase 29 — **true PIT-safe volume / market profile** — and
re-run the unrestricted EOD-flat factory both *standalone* and as *filters/exits on the two Phase-29
winners* (`inside_day_break`, `sweep_displacement`). Same hard rules: NQ only, EOD-flat, realistic
costs, PIT-safe, no hindsight, all years 2021→2026-H1, full-history + year-by-year + leave-one-year-
out + walk-forward, no forced acceptance.

**Headline result.**

```
Volume profile HELPS — but as an EXIT and a FILTER, not as a standalone edge.
- sweep_displacement improved materially: targeting the prior POC + a "clean-path" (no opposing HVN)
  filter lifts it from PF ~1.27-1.41 (Phase 29) to PF 1.53, ret/DD 7.9, max DD <$1k (1 MNQ), 5/6 yrs.
- inside_day_break was NOT improved: every VP filter REMOVED good trades; the base remains best (1.44).
- Standalone VP families are weak (best PF ~1.25) — VP is a context/exit layer, not a signal by itself.
- Still NO PF>=2.0. Best single strategy ~1.5-1.6.
- Best deliverable is a 2-strategy PORTFOLIO (idb + swd): positive EVERY year, PF 1.48, ret/DD 7.7,
  because idb is early-weighted (2021-24) and swd is recent-weighted (2025-26) — they diversify.
Honesty caveat: a strict grid-selection walk-forward (pick on 2021-23, test 2024-26) is WEAK
(mean OOS PF 0.92) -> forward-paper, not "confirmed".
```

---

## A. Build summary

- **Feature module** `src/nqlab/features/volume_profile.py` — 36 PIT-safe `vp_*` columns from BARS
  (no MBP1 → full history): prior-RTH profile (POC/VAH/VAL/value-area-width/HVN/LVN), developing
  intraday profile (cumulative, reset at RTH open), 5-day composite, ATR-normalized distances, and
  value-behavior event flags (POC reclaim, VA breakout/acceptance, VA rejection, LVN rejection/
  breakout, open-vs-value). Profiles spread each bar's volume across the bins its [low,high] spans;
  70% value area grown from POC by absorbing the larger adjacent bin.
- **Factory** `src/nqlab/discovery/phase30.py` — 8 standalone VP families (A–H), the two winners +
  their VP filters, winner baselines on the same exit grid, and VP-aware exits (target prior POC /
  opposite value edge) + value-edge stops. Reuses the **audited** Phase-29 engine wrapper.
- **Tests** `tests/unit/test_phase30_integrity.py` (5) — prefix-invariance / no-future-leakage for
  every level (POC/VAH/VAL/HVN/LVN, prior+developing+composite); prior == standalone completed-prev-
  session; developing only in RTH and cumulative; VAL ≤ POC ≤ VAH always; all families emit valid
  `{-1,0,1}` signals with positive stops/targets.
- **`make check`: GREEN** — ruff (src+tests) clean, `mypy` 111 files clean, **full pytest suite
  passes** (incl. the 5 new Phase-30 + 5 Phase-29 + 7 Phase-27 integrity tests). EOD-flat still
  enforced via `intraday_only` (cross-day trades on early-close days dropped + counted).

### Integrity (the part that matters most)
Prefix-invariance proves no look-ahead: a VP value at bar *k* on the full series equals the value on
the prefix `[0..k]`. The prior profile uses only the **completed previous** RTH session; the
developing profile uses only bars **up to t** (the final-day POC/VAH/VAL is never used intraday);
composites use only completed prior sessions. **VP integrity = PASS.**

---

## B. Volume-profile universe

| | |
|---|---|
| Candidates | **488** — 360 standalone VP, 96 winner+VP-filter, 32 winner-baseline |
| VP families | 16 (8 standalone A–H, 3 `idb`+VP, 3 `swd`+VP, 2 baselines) |
| Sessions | RTH, NY-AM, NY-PM |
| Timeframe | **5m** (full 2021→2026-H1 coverage; 1m/3m/15m/MBP1 still stop at 2025-05 — limitation unchanged from Phase 29) |
| Years | 2021, 2022, 2023, 2024, 2025, 2026-H1 |
| Exits | rr1 / rr1.5 / rr2, **to-POC**, **to-opposite-edge**, hold-to-EOD |
| Stops | ATR (1.0/1.5/2.5×) and **value-edge** (beyond prior VAL/VAH) |
| Passed broad screen | **207 / 488** (≥100 trades, ≥3 yrs, net>0, PF>1); 281 rejected (saved with reasons) |

Artifacts: `…/curated/artifacts/phase30/run1/{leaderboard,candidates,rejected}.csv`,
`candidates_full.json`, `robust.json`; cached VP frame `…/phase30/frame_5m_full_vp.parquet`.

---

## C. Top-25 leaderboard (full history, after costs, 1 MNQ)

`swd_*` = sweep_displacement + VP; `idb_*` = inside_day_break + VP; `*_base` = winner with VP exits
but **no** VP filter (the apples-to-apples control). `toPOC` = target prior POC, `toEDGE` = opposite
value edge.

| # | strategy | kind | n | PF | win% | net $ | DD $ | ret/DD | Sharpe | Calmar | yrs+ | yrConc |
|--:|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | swd_clean_path · NY_AM · a2.5 · toPOC | combo | 137 | **1.59** | 45% | 5,588 | 1,100 | 5.08 | 2.90 | 9.6 | 5/6 | 0.61 |
| 2 | **swd_clean_path · RTH · a2.5 · toPOC** | combo | 244 | **1.53** | 46% | 7,714 | **977** | **7.89** | 2.67 | 8.8 | 5/6 | 0.41 |
| 3 | swd_at_edge · RTH · a2.5 · toPOC | combo | 299 | 1.45 | 49% | 7,747 | 1,112 | 6.96 | 2.29 | 6.4 | 5/6 | 0.39 |
| 4 | idb_base · RTH · a2.5 · eod | base | 221 | 1.44 | 45% | 8,120 | 1,880 | 4.32 | 2.55 | 7.3 | 5/6 | 0.36 |
| 5 | swd_not_chop · RTH · a2.5 · toPOC | combo | 283 | 1.44 | 48% | 7,385 | 1,204 | 6.13 | 2.27 | 5.9 | **6/6** | 0.41 |
| 6 | swd_clean_path · NY_AM · a1.5 · rr2 | combo | 139 | 1.44 | 40% | — | 762 | 3.94 | — | 7.4 | 5/6 | 0.34 |
| 7 | swd_base · RTH · a2.5 · toPOC | base | 302 | 1.43 | 49% | — | 1,112 | 6.85 | — | 6.2 | 5/6 | 0.40 |
| 8 | swd_clean_path · RTH · a2.5 · toEDGE | combo | 243 | 1.43 | 40% | — | 2,153 | 3.21 | — | 3.6 | **6/6** | 0.39 |
| 9 | swd_at_edge · NY_AM · a2.5 · toPOC | combo | 180 | 1.43 | 49% | — | 1,321 | 3.93 | — | 5.6 | **6/6** | 0.63 |
| 10 | swd_base · NY_AM · a2.5 · toPOC | base | 182 | 1.41 | 48% | — | 1,321 | 3.80 | — | 5.4 | **6/6** | 0.66 |
| 11–15 | more `swd_*·toPOC` + `idb_base·rr2` (PF 1.37–1.41) | | | | | | | | | | | |
| 16 | idb_base · RTH · a2.5 · rr2 | base | 282 | 1.37 | 46% | 8,410 | 1,923 | 4.37 | 2.51 | 7.3 | 5/6 | 0.35 |
| 18 | idb_clean_path · RTH · a2.5 · eod | combo | 168 | 1.35 | — | — | — | 5.1 | — | — | 5/6 | 0.49 |
| 22 | idb_base · RTH · a2.5 · toPOC | base | 343 | 1.32 | 54% | — | 1,745 | 4.22 | — | 7.1 | 5/6 | 0.39 |

Standalone VP families (best of each, all below the combos): `vp_inside_breakout` PF 1.25,
`vp_lvn_rejection` 1.25, `vp_va_breakout` 1.18, `vp_outside_reentry` 1.10, `vp_hvn_magnet` 1.08,
`vp_poc_reclaim` 1.04, `vp_va_rejection` 1.00.

---

## D. Finalist reports

### F1 — `swd_clean_path | RTH | stop 2.5×ATR | target prior POC` *(new best risk-adjusted)*

**Entry:** liquidity sweep of a prior swing + a displacement candle in the reclaim direction (the
Phase-29 `sweep_displacement` trigger), **filtered** to setups whose entry is on the far side of the
prior POC (`|close−POC| > 0.5 ATR`) **and** with **no opposing prior HVN within 1 ATR of the path**
to target. **Stop** 2.5×ATR. **Target** the prior-session POC (clamped to [0.5, 8] ATR). Flat 15:55.

| metric | value | metric | value |
|---|---|---|---|
| trades | 244 (~41/yr) | PF after cost | **1.535** |
| win rate | 45.9% | net (1 MNQ) | $7,714 |
| max drawdown | **$977** | return/maxDD | **7.89** |
| Sharpe / Sortino / Calmar | 2.67 / 5.60 / 8.84 | max losing streak | 8 |
| MFE/MAE | 1.54 | winning months | 57% |
| **year-by-year PF** | 2021 1.50 · 2022 1.39 · 2023 **0.86** · 2024 1.20 · 2025 **2.03** · 2026-H1 **3.58** | year concentration | 0.41 |
| **cost 0/0.5/1/2×** | PF 1.56 / 1.55 / 1.53 / 1.51 (cost-robust) | | |

Positive in 5/6 years (only 2023 mildly negative, −$248). **Caveat:** the edge is *recent-weighted*
— 2025+2026 contribute ~63% of net. Improving, but not yet proven regime-stable forward.

### F2 — `swd_not_chop | RTH | a2.5 | toPOC` *(most year-stable swd)*
PF 1.436, **profitable all 6 years** (2023 PF 1.05), ret/DD 6.13, DD $1,204, 283 trades. Filter:
sweep entries away from the POC (not in value chop). The steadiest sweep variant.

### F3 — `idb_base | RTH | stop 2.5×ATR | hold-to-EOD` *(unchanged Phase-29 winner; VP did not help)*
PF 1.439, DD $1,880, ret/DD 4.32, 5/6 yrs. **Year profile is the mirror of swd**: strong early
(2021 PF 1.67, 2022 1.98, 2024 1.72), fading recently (2025 1.18, 2026 0.98). Every VP filter on it
*reduced* PF (RTH eod: base 1.44 vs dir_poc 1.29 vs clean_path 1.35; NY-AM filters fell below 1.0).

### F4 — **PORTFOLIO: `idb_base|RTH|eod` + `swd_clean_path|RTH|toPOC`** *(best overall)*
1 MNQ each, trades merged:

| metric | value | year | PF |
|---|---|---|---|
| trades | 465 | 2021 | 1.58 |
| PF | **1.48** | 2022 | 1.67 |
| net (1 MNQ each) | **$15,834** | 2023 | 1.12 |
| max drawdown | $2,059 | 2024 | 1.46 |
| return/maxDD | **7.69** | 2025 | 1.49 |
| Sharpe / Sortino / Calmar | 2.63 / 5.66 / 5.68 | 2026-H1 | 1.44 |
| winning months | 62% | year concentration | **0.27** |

**Positive in all 6 years, every year PF 1.12–1.67, no year >27% of profit.** The diversification
fixes the single-strategy regime concentration: idb carries 2021-24, swd carries 2025-26.

---

## E. Manual insights (direct answers)

- **Does volume profile improve NQ edge?** **Yes, conditionally** — as an *exit* and a *filter*, not
  as a standalone signal. Targeting the prior **POC** roughly doubled `sweep_displacement`'s edge vs
  a fixed 2R (swd RTH: rr2 PF 1.20 → toPOC 1.43), and the **clean-path** filter added another step
  (→1.53). Standalone VP families top out at PF ~1.25.
- **Which levels matter most?** **POC** (as a magnet/target) is by far the most useful — the to-POC
  target drives the gains. **HVN** matters as an *obstacle* (the clean-path filter = "don't trade
  toward a nearby opposing HVN"). VAH/VAL/LVN as standalone triggers were weak.
- **Value rejection vs value breakout?** Both weak standalone (`vp_va_rejection` PF 1.00,
  `vp_va_breakout` 1.18). Neither is a tradeable edge on its own; breakout edged out rejection.
- **Does VP improve `inside_day_break`?** **No.** Every VP filter removed good trades and lowered PF;
  base (1.44) beats all filtered variants. A clean breakout already prices in the relevant context.
- **Does VP improve `sweep_displacement`?** **Yes** — the single clearest win of the phase
  (PF ~1.3 → 1.53, ret/DD → 7.9, DD < $1k). VP gives the sweep a *natural target* (POC) and a
  *bad-trade filter* (opposing HVN).
- **Best sessions with VP?** **RTH** and **NY-AM**. RTH gives more trades and lower year-
  concentration (more robust); NY-AM gives the single highest PF (1.59) but more concentrated.
  NY-PM remained weak.
- **Which filters reduce drawdown / improve win rate?** **clean-path** (no opposing HVN) reduces DD
  and lifts PF the most; targeting **POC** raises win rate (more trades reach a nearby realistic
  target) vs distant fixed RR. `at_edge`/`not_chop` were marginal-to-neutral.
- **Which families to abandon?** Standalone `vp_va_rejection`, `vp_poc_reclaim`, `vp_hvn_magnet`,
  `vp_outside_reentry` (PF ≤1.10); all `idb`+VP filters; value-edge stops (generally worse than ATR
  except where they incidentally widened the stop).
- **Which deserve deeper research?** (1) The **idb+swd portfolio** and adding a 3rd uncorrelated
  sleeve. (2) `swd_clean_path` / `swd_not_chop` on **1m/3m** (needs building 2025-H2→2026 bars).
  (3) **POC-as-target** applied to *other* mean-reverting setups (vwap reclaim, value rejection).
  (4) Composite / developing-POC targets (only prior-POC was used as the target here).

---

## Verdicts (no forced acceptance)

| candidate | verdict | reason |
|---|---|---|
| **`idb_base\|RTH\|eod` + `swd_clean_path\|RTH\|toPOC` portfolio** | **ACCEPT_FOR_FORWARD_PAPER** | best overall: PF 1.48, ret/DD 7.7, DD $2.1k, **profitable all 6 years**, yrConc 0.27, cost-robust. The diversification is the real win. Not live-ready until forward-confirmed. |
| `swd_clean_path\|RTH\|a2.5\|toPOC` (+ `swd_not_chop` for 6/6-yr stability) | **ACCEPT_FOR_FORWARD_PAPER** | VP materially improved swd (PF 1.53, ret/DD 7.9, DD <$1k, cost-robust). Caveat: recent-weighted (2025-26 = ~63% of net). |
| `idb_base\|RTH\|eod/rr2` | **ACCEPT_FOR_FORWARD_PAPER** | unchanged from Phase 29 (PF 1.44); VP did not help it. Early-weighted — pairs well with swd. |
| Standalone VP families (A–H) | **REJECT** (as standalone) / **NEEDS_MORE_DATA** | best PF ~1.25, weak ret/DD; VP's value is as exit+filter, not a signal. |
| All `idb`+VP-filter variants | **REJECT** | strictly worse than `idb_base` on every matched config. |
| `swd_clean_path\|NY_AM\|toPOC` (PF 1.59) | **NEEDS_MORE_DATA** | highest PF but yrConc 0.61–0.83 (heavily 2025-26); prefer the RTH version. |
| Any "confirmed / live-ready" claim | **withheld** | grid-selection walk-forward is weak — ranking on 2021-23 then testing 2024-26 gives **mean OOS PF 0.92** (top-train picks overfit). Full-history PF 1.5 is discovery, not confirmation; 2026-H1 is partial. **Forward paper data required.** |

### Honest bottom line
Volume profile delivered the phase's goal — it **improved a Phase-29 winner** (`sweep_displacement`
→ PF 1.53, ret/DD 7.9, sub-$1k DD) via a POC target and an HVN-avoidance filter, and the
`idb`+`swd` **portfolio is positive in every one of the six years** (PF 1.48). But VP is a *context
layer*, not a standalone edge; it did **not** help the breakout family; the **PF ≥ 2.0 ceiling still
holds** (now confirmed across phases 16–30); and the strict walk-forward warns that selecting on the
past does not reliably transfer. The right next step is **forward-paper the portfolio**, not declare
victory.

---

## Reproduction
```bash
python scripts/phase30_run.py all      # build VP frame once, 488 candidates, analyze, robust
pytest tests/unit/test_phase30_integrity.py   # 5 VP integrity tests (prefix-invariance, EOD-flat)
make check                              # ruff + mypy(111) + full suite green
# artifacts: /Volumes/T9/nq data/curated/artifacts/phase30/run1/
```
