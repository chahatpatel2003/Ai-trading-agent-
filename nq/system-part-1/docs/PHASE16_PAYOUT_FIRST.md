# Phase 16 — Payout-First Scalable Strategy Discovery

**Goal:** stop accepting "barely profitable" strategies. Find an NQ-only strategy that can
actually support Topstep payouts ($200+ winning days, consistent payout-eligible months) and
scale across multiple funded 150K accounts — or reject honestly.

**Status: built, run end-to-end, verdict = NO business-worthy strategy found (all REJECT).**
The payout-first rig works; the search produced 13 validation survivors; all that were frozen
collapsed out-of-sample on the pristine 2025-2026 lockbox. No forced acceptance.

---

## Reset acceptance bar (after costs)

| gate | threshold |
|---|---|
| profit factor | **≥ 1.30** |
| risk of ruin | ≤ 10% |
| green months | ≥ 85% |
| $200+ winning days | ≥ 5, and ≥ 10% of trading days |
| payout-eligible months | ≥ 50% of months |
| scalability | PF holds (≥1.20) at funded sizing **and** uses > 1 MNQ |
| Deflated Sharpe (frozen count) | ≥ 0.90 |
| Topstep pass probability | ≥ 70% preferred |

A candidate is accepted only if it clears **every** gate. `NONE` did.

## What was built

1. **Payout mechanics** (`domain/risk/payout.py`) — calculated from the simulated daily equity,
   never assumed: $200/$500 winning-day stats, month-by-month payout eligibility (≥5 winning days,
   consistency rule, buffer cleared, not blown), and funded payout projection.
2. **Multi-account 150K fleet scaler** — aggregates the same strategy across 5–10 funded accounts;
   payouts scale ×N but risk is **perfectly correlated** (no diversification — one regime blows the
   whole fleet).
3. **Payout-first stat panel + thresholds + strict verdict** (`discovery/phase16.py`) — win/PF/
   expectancy, trades-per-week/month, green months, $200-days, payout months, per-account pass +
   ruin (50K/100K/150K), best account, two-stage (eval + funded-scale) simulation.
4. **Six payout-oriented families** (`discovery/phase16_families.py`) — opening-drive continuation,
   sweep+displacement reversal, compression→NY expansion, prior-level reaction, MSS+FVG
   continuation, trend-expansion day. All NY-AM (09:30–12:00 ET), 5m, asymmetric R (tight stop /
   3–5 ATR target), reusing the PIT-safe feature layer.
5. **CLI** — `phase16 search` / `freeze` / `lockbox`, `report phase16`.

## Data protocol (reserved lockbox)

Search/train **2021–2023**, validation **2024** (the search never loads past 2024-12-31), final
**pristine** lockbox **2025-01 … 2026-06**. These families were never run on 2025-26 before, so the
lockbox is genuinely pristine → an ACCEPT_FOR_PAPER ceiling was on the table.

## Search (train 2021-23 / val 2024, 150K, 5m)

144 candidates. **13 cleared the validation gauntlet** (positive every year, ruin ≤ 10%,
walk-forward, cost-stress) — all at risk **$150**, families `p16_opening_drive` and
`p16_prior_level`. The higher-risk ($250/$400) variants all rejected on ruin in validation. Top 4
distinct (family × sizing) were frozen. Validation DSR was already weak (0.15–0.58 < 0.90).

## Lockbox (pristine 2025-2026) — all REJECT

| strategy | trades | /mo | win | **PF** | exp | green | $200-days (rate) | payout-mo | $/mo | P(pass 50/100/150K) | DSR | scale PF@$600 | verdict |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--|--:|--:|--|
| opening_drive adaptive | 136 | 7.6 | 26% | **0.93** | −$6 | 10/18 | 26 (7.0%) | 10 | $220 | 0.6/0/0% | 0.28 | 0.34 (blown) | REJECT |
| opening_drive mnq_min1 | 155 | 8.6 | 26% | **0.87** | −$12 | 8/18 | 27 (7.2%) | 8 | $189 | 1.7/0/0% | 0.25 | 0.33 (blown) | REJECT |
| prior_level mnq_min1 | 212 | 11.8 | 29% | **0.95** | −$5 | 9/18 | 38 (10.2%) | 9 | $222 | 8.6/0/0% | 0.19 | 0.28 (blown) | REJECT |
| prior_level adaptive | 167 | 9.3 | 29% | **1.04** | +$3 | 10/18 | 30 (8.0%) | 10 | $203 | 8.0/0/0% | 0.16 | 0.29 (blown) | REJECT |

Why every one fails the business bar:
- **PF collapsed out-of-sample** to 0.87–1.04 (three of four are net-losing on the lockbox). The
  +$2–4.3k validation OOS edge did not survive into 2025-2026.
- **Green months 44–56%** (bar: ≥85%); **$200-day rate 7–10%** (bar: ≥10%).
- **Topstep pass probability ≈ 0% at 150K**, ≤ 8.6% at 50K — these cannot reliably pass.
- **Funded scaling is fatal:** at $600/trade (5–7 MNQ) PF collapses to 0.28–0.34 and **every
  account blows** → fleet payout $0. They only "survive" by trading tiny relative to the buffer,
  exactly the tiny-edge/tiny-size failure the bar is designed to reject.
- **DSR 0.16–0.28** ≪ 0.90 — statistically indistinguishable from noise at the frozen count.

## Verdict

```
NONE business-worthy. All 4 frozen candidates REJECT on the pristine 2025-2026 lockbox.
```

No NQ-only strategy found in Phase 16 meets the payout-first business requirements. The recurring
pattern across Phases 6–16 holds: NQ-only intraday edges that look acceptable in-sample are thin
(PF ≈ 0.9–1.2) and **do not survive out-of-sample**, and none can be sized up without the locked
trailing drawdown blowing the account. `bos_5m_trigger` remains a **paper-watch candidate only**
(Phase 15) and is explicitly **not** the strategy we are looking for.

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **225 tests**, incl. 10 Phase-16 tests).
- ✅ Clean lockbox respected — search ended 2024-12-31; lockbox 2025-26 touched once, pristine.
- ✅ Every candidate carries a verdict + reason; payout-first thresholds enforced.
- ✅ No forced acceptance — the rig rejected everything honestly.
- ✅ Final report states plainly: **no strategy is business-worthy for Topstep payouts.**

## Honest framing

> Phase 16 did **not** find a scalable Topstep payout strategy. It built the payout-first rig
> ($200-day stats, calculated payout projection, multi-account 150K fleet scaler, two-stage
> eval→funded simulation, PF≥1.30 / ruin≤10% / green≥85% / DSR≥0.90 bar) and ran a genuine search.
> The best validation survivors were net-losing out-of-sample and blow the account when scaled.
> The correct, non-negotiable verdict is **REJECT all** — the goal is a scalable payout strategy,
> and we do not have one.

## Reproduction

```
nqlab phase16 search --timeframes 5m --account 150k   # train 2021-23 / val 2024 (reserves 2025-26)
nqlab phase16 freeze --top-n 8                          # freeze validation survivors (val only)
nqlab phase16 lockbox                                   # ONCE on pristine 2025-2026 lockbox
nqlab report phase16                                    # leaderboard + payout panel + verdict
```
