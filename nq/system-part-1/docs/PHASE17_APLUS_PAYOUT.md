# Phase 17 — Business-Worthy A+ Topstep Payout Strategy Search

**Goal:** find an NQ-only NY-session strategy actually worth trading for Topstep payouts and
scalable funded accounts — high-conviction, asymmetric, larger winners. Reject anything that
cannot realistically make payouts. **No small-edge acceptance.**

**Status: built, run end-to-end, verdict = NO business-worthy strategy found (all REJECT).**
The A+ rig + funded-withdrawal model work; the search produced 7 validation survivors; all that
were frozen collapsed out-of-sample on the pristine 2025-2026 lockbox. No forced acceptance.

---

## Bar (after commissions + slippage)

PF **≥ 1.30** (prefer ≥ 1.50), ruin ≤ 10%, green months **≥ 75%**, payout-eligible months **≥ 60%**,
≥ 5 $200-days and ≥ 10% $200-day rate, Topstep pass ≥ 70% preferred, survives 50K/100K/150K + cost
stress, scales past 1 MNQ without PF collapse, DSR ≥ 0.90. Accepted only if **every** gate clears.

## New in Phase 17 (vs Phase 16)

1. **Six high-conviction A+ families** (`discovery/phase17_families.py`), NY-AM 5m, asymmetric R:
   - **A** sweep reversal (sweep in discount/premium + displacement),
   - **B** expansion-day continuation (range-expansion structure break + bias),
   - **C** volatility expansion (compression → confirmed expansion + displacement),
   - **D** opening-drive A+ (OR break **and** prior-day liquidity break + bias),
   - **E** failed-breakout trap (sweep → reclaim + displacement),
   - **F** premium/discount model (location + MSS + displacement).
2. **Funded withdrawal model** (`domain/risk/payout.py::simulate_funded_withdrawals`) — Stage B
   now models payout withdrawals → **reduced buffer** → post-payout drawdown / blow-after-payout.
3. **Diagnostics-first rejection** — prefilter requires trade frequency + MFE/MAE asymmetry (≥1.3)
   before the full gauntlet.
4. **`NEEDS_FORWARD_DATA` verdict** — an MBP1-required strategy without a clean MBP1 lockbox
   (2026-06 has no MBP1) cannot exceed it. (Phase-17 families are all trades-derivable, so the
   trades-only lockbox is clean for them and they can reach ACCEPT — none did.)

## Data protocol (reserved lockbox)

Search/train **2021–2023**, validation **2024** (never loads past 2024-12-31), **pristine** lockbox
**2025-01 … 2026-06**. These families were never run on 2025-26 before → genuinely pristine, so
`ACCEPT_FOR_PAPER` was on the table.

## Search (train 2021-23 / val 2024, 150K, 5m)

144 candidates → **7 cleared the validation gauntlet** (113 REJECT, 24 NEEDS_MORE_DATA). The
high-OOS variants ($500 risk) rejected on ruin (0.6–0.86) or thin positive-years; the survivors
were lower-risk `p17_vol_expansion` and `p17_opening_drive_aplus`. Top 3 distinct (family × sizing)
were frozen. Validation DSR was already weak (0.34–0.58 < 0.90).

## Lockbox (pristine 2025-2026) — all REJECT

| strategy | trades | /wk | win | **PF** | exp | green | $200-days (rate) | payout-mo | $/mo | P(pass 50/100/150K) | scale PF@$600 | funded payouts | DSR | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--|--:|--:|--:|--|
| vol_expansion (adaptive) | 134 | 1.9 | 20% | **0.74** | −$25 | 7/16 | 16 (4.3%) | 5 | $52 | 0/0/0% | 0.00 (blown) | 0 | 0.02 | REJECT |
| vol_expansion (mnq_min1) | 136 | 2.0 | 20% | **0.72** | −$27 | 6/16 | 15 (4.0%) | 4 | $41 | 0/0/0% | 0.00 (blown) | 0 | 0.02 | REJECT |
| opening_drive_aplus (adaptive) | 138 | 1.8 | 36% | **0.96** | −$3 | 8/18 | 22 (5.9%) | 7 | $162 | 0.6/0/0% | 0.12 (blown) | 0 | 0.24 | REJECT |

Why every one fails the business bar:
- **PF collapsed out-of-sample** to 0.72–0.96 (all net-losing on the lockbox). The +$1.7–3.2k
  validation OOS edge did not survive into 2025-2026.
- **Green months 38–44%** (bar ≥75%); **$200-day rate 4–6%** (bar ≥10%); **0 funded payouts** —
  net-losing accounts never reach payout eligibility, so withdrawals/fleet payout = $0.
- **Topstep pass probability 0–0.6%**; ruin near 100% at 50K/100K.
- **Funded scaling is fatal:** at $600/trade (4–7 MNQ) PF collapses to 0.0–0.12 and every account
  blows → 5- and 10-account fleet payout $0.
- **DSR 0.02–0.24** ≪ 0.90 — noise at the frozen count.

## Verdict

```
NONE business-worthy. All 3 frozen A+ candidates REJECT on the pristine 2025-2026 lockbox.
No NQ-only NY-session strategy in Phase 17 can realistically make Topstep payouts.
```

The cross-phase pattern (6–17) is now very firmly established: NQ-only intraday setups that look
acceptable in-sample are thin and **do not survive out-of-sample**, and none scale past ~1 MNQ
without the locked trailing drawdown blowing the account. Higher-conviction A+ structure did not
change this — the 2025-2026 regime simply does not reward these setups, and the validation edges
were noise (DSR ≤ 0.24 on the lockbox).

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **232 tests**, incl. 7 Phase-17 tests).
- ✅ Clean lockbox respected — search ended 2024-12-31; lockbox 2025-26 touched once, pristine.
- ✅ Every candidate carries a verdict + reason; payout-first bar enforced; `NEEDS_FORWARD_DATA`
  path implemented for MBP1 families.
- ✅ No forced acceptance, no tiny-edge acceptance — the rig rejected everything honestly.
- ✅ Final report states plainly: **no strategy is business-worthy for Topstep payouts.**

## Honest framing

> Phase 17 did **not** find a scalable Topstep payout strategy. It built six high-conviction A+
> families, a funded payout-withdrawal/post-payout-drawdown model, diagnostics-first rejection, and
> the NEEDS_FORWARD_DATA ceiling — then ran a genuine reserved-lockbox search. The best validation
> survivors were net-losing out-of-sample, made zero funded payouts, and blow the account when
> scaled. The correct, non-negotiable verdict is **REJECT all**.

## Reproduction

```
nqlab phase17 search --timeframes 5m --account 150k   # train 2021-23 / val 2024 (reserves 2025-26)
nqlab phase17 freeze --top-n 8                          # freeze validation survivors (val only)
nqlab phase17 lockbox                                   # ONCE on pristine 2025-2026 lockbox
nqlab report phase17                                    # A+ leaderboard + payout panel + verdict
```
