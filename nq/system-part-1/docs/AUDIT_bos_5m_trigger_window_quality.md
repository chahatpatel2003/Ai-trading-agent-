# Audit — bos_5m_trigger trading window + strategy quality

Frozen `bos_5m_trigger` (5m, 50K, `mnq_fixed1` $150, stop 1.0×ATR, target 4.0×ATR, NY-AM
hour-mask 9–12 ∩ `is_rth`). **Frozen SIGNAL unchanged.** Filters are derived on **train+val
(2021-2024)** and applied **out-of-sample** to the **lockbox (2025-2026)**; an in-sample
(lockbox-fit) version is computed only to expose the curve-fit gap. Conclusions below were
adversarially verified (10 refutation skeptics + methodology auditor + 3-lens verdict panel);
two draft overstatements were corrected (noted inline).

## Task 1 — trading window (lockbox, 478 trades)

| ET hour of entry | trades | net | exp/trade |
|---|---:|---:|---:|
| 09:30–09:59 | 315 | +$1,693 | $5.37 |
| 10:00–10:59 | 103 | +$265 | $2.58 |
| 11:00–11:59 | 55 | +$1,235 | $22.45 |
| 12:00 (next-open spillover) | 5 | +$1,602 | $320 (5 trades — noise) |

- **By session:** NY_AM 473, NY_PM 5, **Asia 0, London 0, overnight 0.**
- **First entry 09:35, last entry 12:00 ET.** The 5 "NY_PM" entries are 12:00 *next-open fills
  of 11:55 NY_AM signals* (true NY_AM entries end 11:55). Force-flat 15:55; **no overnight holds.**
- **NY-only: YES** in substance — every trade originates from a 09:30–12:00 NY-morning signal.

## Task 2 — strict NY 09:30–12:30 filter

**No-op.** 478 trades identical with/without it. The frozen signal mask already caps entries at
12:00, the engine force-flattens at 15:55, and there are no overnight holds. The strategy is
*already* a strict NY-AM strategy, tighter than the requested 12:30 cutoff.

## Task 3 — profit-factor filters (PF before/after)

**Out-of-sample** (filter fit on 2021-2024, applied to lockbox 2025-2026):

| filter | dropped | trades | PF | exp | win | net | green/red | maxDD | ruin | P(pass 50K) | payout mo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (NY-AM) | — | 478 | 1.177 | $10.03 | 24.5% | +$4,795 | 12/6 | $1,807 | 32% | 68% | 12 |
| best_hours_only | keep h11 | 61 | **1.451** | $27.49 | 26% | +$1,677 | 9/7 | $1,536 | 0.4% | **2.1%** | 1 |
| remove_worst_hours | h12 | 478 | 1.177 | — | — | +$4,795 | — | — | 32% | 68% | 12 |
| remove_worst_weekdays | Mon/Wed/Fri | 205 | 1.256 | $14.79 | 25% | +$3,031 | 11/7 | $2,186 | 12% | 88% | 7 |
| remove_worst_vol | — (no-op) | 478 | 1.177 | — | — | +$4,795 | — | — | 32% | 68% | 12 |
| **remove_worst_liquidity** | high-spread (lhi) | 366 | **1.270** | $15.11 | 26% | **+$5,532** | 11/6 | $1,692 | **18%** | **82%** | 10 |

**In-sample** (fit on the lockbox itself = curve-fit illusion):

| filter | dropped | PF | net | tell |
|---|---|---:|---:|---|
| weekday | Tue, Fri | 1.332 | +$4,886 | Tue is fit-**best** dow (+11.17) yet dropped → contradicts fit |
| vol | vlo | 1.195 | +$4,485 | vlo is fit-**best** (+4.06) yet dropped |
| liquidity | lmid | 0.998 | −$24 (blown) | lmid is fit-**best** (+6.81) yet dropped → blows up |

## The decisive diagnostic — regime instability

| period | mode | trades | net | exp/trade | outcome |
|---|---|---:|---:|---:|---|
| 2021-2024 | **gated real 50K** | 96 | −$1,104 | — | **BLEW UP by 2021-04-15** (account halted) |
| 2021-2024 | ungated full signal | 1,323 | +$2,579 | +$1.95 | near-flat |
| 2025-2026 | gated lockbox | 478 | +$4,795 | +$10.03 | PF 1.177 |

The only earlier **real** account blew up in early 2021; the raw 4-year edge is ~+$1.95/trade
(near-flat). PF 1.177 / +$4,795 exists **only** in the 18-month lockbox. The fit-best buckets
(Tue, low-vol, mid-liquidity) are the lockbox-worst — the per-bucket signal the filters rely on
is **non-stationary**, so OOS "gains" are within noise. (Methodology severity: **MAJOR**.)

## Final answer

1. **Is it actually NY-session tradable? — YES.** Genuinely and exclusively a NY-morning
   strategy (09:35–11:55 entries + a 12:00 next-open spillover; zero Asia/London/overnight;
   force-flat 15:55). The strict 09:30–12:30 filter is already satisfied.
2. **Can it reach stronger PF without curve-fitting? — NO, not deployably.** The honest,
   economically-motivated, OOS-confirmed, *viable* ceiling is **PF ~1.27** (drop high-spread
   bars — which also cuts ruin 32%→18% and lifts P(pass) 68%→82%). It does **not** reach 1.3.
   - PF ≥1.3 is reachable OOS **only** via `keep hour 11` (PF 1.451) — but that collapses to
     61 trades, **2.1% pass probability, 1 payout month** → economically dead.
   - The only *viable* ≥1.3 (weekday drop, PF 1.332) is **in-sample curve-fitting** whose bucket
     choices contradict the fit set (Tuesday is fit-best yet dropped).
   - And the entire filter exercise rests on a non-stationary bucket signal + a single favorable
     18-month regime; the same strategy blew the 50K account in 2021.

## Verdict: `NOT_ENOUGH_EDGE`

The edge is real-but-thin and **regime-dependent, not proven**. PF cannot robustly clear 1.3
without curve-fitting; the deployable ceiling is ~1.27. Disposition is unchanged from Phase 15:
**forward-paper only, never live** — and if forward-papering, the high-spread liquidity filter is
the one principled tweak worth carrying (it strictly improves ruin/pass without changing the
signal). 2 of 3 verdict judges said `NOT_ENOUGH_EDGE`; the third said `SAFE_FOR_FORWARD_PAPER`
(paper-it-forward) — consistent, since forward-paper is the handling decision and "not enough
proven edge" is the quality verdict.
