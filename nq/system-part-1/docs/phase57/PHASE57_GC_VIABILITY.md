# Phase 57 — GC/MGC Topstep Viability vs NQ

**Verdict: FAIL.** GC/MGC does **not** beat NQ for the Topstep plan, and cannot
reach $1,000+/month/account. Best stable GC outcome ≈ **$20/month/account** (50K,
gap-continuation) — ~20× below NQ's ~$400/mo ceiling and ~50× below the $1k target.
Larger accounts (100K/150K) are **worse** (98–100% of combine attempts never even
reach the profit target within 300 trading days).

---

## 1. Data inventory (printed before any testing)

- **Folder:** `~/Desktop/gc`, 11 parquet files (~58 MB).
- **Symbol:** one only — `GC.v.0` (Databento continuous front-month, **volume-rolled**, NOT back-adjusted). Consistent across all years.
- **MGC:** ❌ **does not exist** — only full-size GC bars are present. MGC modelled by economics ($10/pt, tick 0.1 = $1).
- **Data type:** OHLCV **1-minute bars** (`rtype=33`). **No trades, no MBP1** → no true order flow; VWAP/POC approximated from 1-min volume.
- **Columns:** `open, high, low, close, volume`, `ts_event` (UTC) + `rtype/publisher_id/instrument_id`.
- **Coverage:** **2016-01 → 2025-12, continuous**, ~258 sessions/yr, ~1,374 1-min bars/session (≈23 h Globex). Only missing periods = daily 17:00–18:00 ET maintenance break + weekends.
- **Excluded:** `gc_2024_jan.parquet` (duplicate Jan subset of 2024).
- **Price scale:** ~1060 (2016) → 4584 (2025) — strong multi-year uptrend (regime caveat for any long-biased result).

## 2. Method (same Topstep economics as the NQ benchmark)

- 1-min → **5-min bars** on a gold-session clock (18:00 ET → 17:00 ET, `smin` 0..1380). Sessions: Asia `smin 60–540`, London `540–860`, COMEX open `860` (08:20 ET), no-new-entry `1080` (12:00 ET), pit close `1170`, force-flat `1320` (16:00 ET).
- **Entries:** signal on completed bar → **next-bar-open fill** (no look-ahead) + adverse slippage.
- **MGC costs:** commission $1.48 RT + slippage 1 tick/side ($2) ≈ **$3.48/contract RT** (vs MNQ ~$1.50 — gold costs more per micro).
- **Topstep sim:** identical engine to NQ phases (`ACCOUNTS` cost $49/$99/$199; trailing MLL via per-trade MAE; consistency 50%; min 2 days; cushion sizing 1→2→3; funded sizing swept 1–max, best EV kept).
- Sizing convention matches the NQ benchmark: max **5/10/15 MGC** contracts (conservative micro reading), for apples-to-apples comparison.

## 3. Trade-level results (15 families, full 2016–2025, after costs)

| family | trades | /yr | WR | RR | exp$ | PF | IS exp | OOS exp | OOS PF |
|---|---|---|---|---|---|---|---|---|---|
| 1 open-drive cont | 2548 | 255 | .47 | 0.96 | −4.3 | 0.84 | −4.4 | −4.2 | 0.86 |
| 2 London→NY break | 2276 | 228 | .42 | 1.21 | −3.9 | 0.86 | −5.3 | −3.0 | 0.91 |
| 3 Asia/London sweep | 1774 | 177 | .41 | 1.25 | −3.2 | 0.88 | −1.6 | −4.5 | 0.85 |
| 4 PDH/PDL sweep+reclaim | 1345 | 135 | .43 | 1.07 | −5.7 | 0.79 | −2.7 | −8.8 | 0.72 |
| 5 VWAP reclaim/reject | 2491 | 249 | .32 | 1.82 | −3.2 | 0.84 | −3.1 | −3.3 | 0.84 |
| 6 POC/value-area revert | 1849 | 185 | .44 | 0.99 | −4.7 | 0.79 | −3.2 | −6.5 | 0.79 |
| 7 gap continuation | 1519 | 152 | .41 | 1.36 | −2.0 | 0.94 | −4.7 | **+0.5** | 1.01 |
| 8 gap fill | 1355 | 136 | .52 | 0.85 | −2.6 | 0.91 | −1.1 | −4.1 | 0.84 |
| 9 high-vol breakout | 525 | 53 | .47 | 1.07 | −2.2 | 0.93 | −2.6 | −1.6 | 0.94 |
| 10 failed-breakout rev | 2258 | 226 | .41 | 1.13 | −3.8 | 0.77 | −3.7 | −4.0 | 0.77 |
| **11 first pullback** | **193** | **19** | .45 | 1.47 | **+5.2** | **1.18** | +3.3 | **+7.1** | 1.20 |
| 12 trend-day cont | 714 | 71 | .45 | 1.10 | −2.1 | 0.92 | −6.2 | **+1.3** | 1.04 |
| 13 range-day MR | 816 | 82 | .29 | 1.87 | −2.7 | 0.77 | −2.5 | −4.1 | 0.72 |
| 14 news-vol proxy | 155 | 16 | .45 | 0.94 | −10.8 | 0.75 | −7.5 | −13.5 | 0.75 |
| 15 dollar/rates proxy* | 1596 | 160 | .45 | 1.04 | −3.7 | 0.86 | −4.7 | −2.7 | 0.92 |

\*No DXY/ZB data exists (GC-only) → substituted a 5-day gold-momentum-aligned open breakout.

**Key fact:** even **commission-free** (slippage only), 14 of 15 families are negative/break-even.
Only **first-pullback** has genuine raw edge (+$6.7 zero-comm) — but at **~19 trades/yr** it is
far too infrequent to pass a combine. This is the NQ frequency↔edge trilemma, and GC is **worse**
(higher MGC per-trade cost; no frequent family clears it).

## 4. Topstep EV (best funded sizing; multi-seed stable values)

| candidate | acct | combine resolves? | pass% (of resolved) | EV/month | worst month (3c) |
|---|---|---|---|---|---|
| gap continuation | 50K | ~18% (82% **timeout**) | ~0.45 | **~$20** | −$2,228 |
| P_posOOS (fpb+gap+trend) | 50K | ~26% | ~0.10 | ~$1 | −$2,421 |
| P_posOOS | 100K | ~2% (98% timeout) | noise | ~$30 (unreliable) | −$2,421 |
| any family/portfolio | 150K | ~0% (100% timeout) | — | negative | — |
| first-pullback (the only +edge) | any | ~0% timeout | — | negative | −$842 |

- **100K/150K**: 98–100% of combine attempts never reach the $6k/$9k target within 300 trading days → the strategies are too weak/infrequent. The occasional non-zero EV is statistical noise on a ~2% resolved fraction.
- **50K** is the only size that resolves at all, and only gap-continuation gives a *stable* positive number (~$20/mo) — but 82% of attempts still never finish, and full-sample expectancy is negative (it passes only on lucky windows).
- **First payout probability** for every candidate is low (~0.0–0.27); funded payout EV is small ($0–$250/cycle).

## 5. Final ranking & answers

1. **Best GC/MGC strategy:** none viable. Highest raw edge = **first-pullback-after-impulse** (+$5.2/trade, PF 1.18, OOS +$7.1) but ~19 trades/yr → unusable for a combine. Best *combine-completing* candidate = **gap-continuation on 50K (~$20/mo)**.
2. **Best account size:** **50K** (smallest target is the only one reachable; 100K/150K time out).
3. **Best pass-mode:** cushion 1→2→3 — but edge is absent, so sizing doesn't rescue it.
4. **Best funded payout-mode:** small/low size; payouts are rare and tiny regardless.
5. **Does GC/MGC beat NQ?** **No.** NQ clean5+Phase36 ≈ $400/mo robust ceiling with a genuinely positive-expectancy engine (+$6.27/trade, ~79% combine pass). GC's best is ~$20/mo, unstable, on a negative-expectancy portfolio. **GC is worse than NQ.**
6. **Can GC/MGC reach $1,000+/mo/account?** **No — FAIL by ~20–50×.**

## 6. Why GC fails (mechanism)

- **No frequent positive-edge family.** Gold intraday is ~coin-flip after costs on 1-min/5-min bars (mirrors NQ Phase 29/33: no PF≥2 intraday edge exists).
- **Higher cost hurdle.** MGC RT ≈ $3.48 vs MNQ ≈ $1.50; thin edges that survive on NQ die on gold.
- **Target-in-dollars too big for the edge.** Profit targets ($3k/$6k/$9k) are unreachable within a combine window for weak, infrequent gold edges → mass timeouts at 100K/150K.
- **Data limits.** Only 1-min OHLCV (no order flow / DXY / rates) caps the sweep/VWAP/POC and macro-proxy families.

## 7. Caveats / honest flags

- EV figures for near-zero-expectancy candidates are noisy (high timeout fraction); treated as "≈ $0–$50, unreliable," not precise.
- Continuous `GC.v.0` is not back-adjusted → roll-day gap artifacts capped/filtered, but overnight reference levels on ~4 roll days/yr are imperfect.
- Family parameters were kept deliberately simple (no per-family curve-fitting), consistent with the project's anti-bias stance (Phase 50). Aggressive optimization could nudge a single number but cannot manufacture the missing frequent edge.
- NQ benchmark not re-run live (T9 drive unmounted); used the established prior-phase figures.
