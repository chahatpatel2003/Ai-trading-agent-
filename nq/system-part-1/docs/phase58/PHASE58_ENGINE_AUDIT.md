# Phase 58 — Strategy Engine Bug Audit

## Headline answers
- **Is the engine broken?** **NO.** PnL verified 20/20 manually; economics correct; fills realistic.
- **Is the cost model too harsh?** **NO.** Costs match real specs; zero-cost barely changes results.
- **Is the Topstep wrapper killing otherwise-profitable strategies?** **Partly, and faithfully** — it *caps* the profitable NQ engine (~$744→~$400/mo) and blocks *infrequent* positive GC edges from passing, but it does so by *correct* Topstep rules, not a bug. It does **not** turn a real edge into a loss.
- **Are the strategies actually unprofitable?** **GC: YES** (genuine near-zero/negative raw edge). **NQ: NO** (the best NQ strategy is genuinely profitable, PF 1.22).
- **What explains the failure?** GC has **no frequent positive-edge RTH strategy** — and structurally, **gold's trend is overnight while Topstep forces RTH-flat**, so the tradeable session is a coin flip. NQ "fails" only the $1k/mo *income* bar; its strategy works, the Topstep wrapper just caps extraction.

---

## 1. Contract economics — VERIFIED CORRECT
| instrument | point value | tick | tick value | source | status |
|---|---|---|---|---|---|
| NQ | $20/pt | 0.25 | **$5.00** | `config/instruments/nq.yaml` | ✓ correct |
| MNQ | $2/pt | 0.25 | **$0.50** | config | ✓ correct (confirmed by trade recon) |
| GC | $100/pt | 0.1 | **$10.00** | spec (not in config; GC unused) | ✓ correct |
| MGC | $10/pt | 0.1 | **$1.00** | `phase57_engine.PV=10` | ✓ correct |

- **Commission:** NQ $0.85/side ($1.70 RT), MNQ $0.25/side ($0.50 RT), MGC $0.74/side ($1.48 RT). All realistic/Topstep-plausible.
- **Slippage:** 1 tick/side (`backtest.slippage_ticks=1.0`). MGC RT total ≈ $3.48 = **1.9% of the $179 mean daily RTH range** — not harsh.
- **Spread:** modelled implicitly via 1-tick slippage (gold spread ≈ 1 tick). Adequate.

## 2. Fill logic — REALISTIC, mildly conservative (not buggy)
Real engine (`src/nqlab/backtest/engine.py`):
- **Entry:** `next_open` — signal at bar *t* fills at *t+1* open + adverse slippage (`entry_fill = fill + side*slippage`). No look-ahead. ✓
- **PnL:** `gross = side*(exit−entry)*point_value*contracts`. `side` carries direction → **no sign-flip possible**. ✓
- **Stops/targets:** intrabar vs high/low, **stop assumed first if both touch** (conservative). Forced exits (STOP/EOD/TIME/BLOWN) take adverse slippage; **TARGET fills exact** (resting limit — *not* penalised). ✓
- **Touch misses:** target fills whenever `high≥target` (long) at exact target — no artificial misses. ✓
- Conservatism is realistic; the only stricter-than-NQ spot is the GC engine charging adverse slippage on target exits too (immaterial — see §6).

## 3. Baseline sanity (GC, RTH pit, 1 MGC, normal cost)
| baseline | expectancy | total | note |
|---|---|---|---|
| always-flat | $0 | $0 | control |
| buy-open / sell-close (long every day) | **−$4.63** | −$11,664 | WR 50.4% — RTH is a coin flip |
| sell-open / buy-close (short every day) | −$2.33 | −$5,882 | symmetric |
| random long/short | −$3.65 | −$9,194 | ≈ pure cost bleed |
| **buy & hold 2016→2025 (no flat)** | — | **+$32,522** | **all of gold's trend is OVERNIGHT** |

**Critical structural finding:** gold's bull move lives *overnight/multi-day*; the Topstep-mandated **RTH-flat** session (open→pit close) is a near-coin-flip (WR 50.4%). A futures prop trader who must be flat by close cannot hold the part of gold that actually trends.

## 4. Strategy sanity + manual verification
- **Best NQ** (`frozen clean5+gate BALANCED`, 3,531 trades 2021–2025): exp **+$12.65/trade, WR 35.6%, PF 1.22, +$44,658/MNQ**; profitable **5 of 6 years** (only 2023 −$1,862). **20/20 random trades recomputed to the stored net within $0.02.** No sign flips (shorts profit on down moves). Same-bar fills 13.5% (realistic). → **This strategy genuinely works (~$744/mo/MNQ raw).**
- **Best GC** (`first_pb`): exp +$5.20, PF 1.18 but only **193 trades / 10 yrs**. (`gap_cont`: PF 0.94 — negative.)

## 5. Topstep wrapper — FAITHFUL, no bug
`src/nqlab/domain/risk/topstep.py` audited line-by-line:
- **Trailing MLL:** `floor = peak − MLL`, **locked at starting balance** once reached (`lock_trailing_at_start`) — exact Topstep behaviour. ✓
- **Breach:** `equity ≤ trailing_floor → blown`; intraday basis marks the adverse extreme each bar. ✓
- **Daily loss:** honored if set (combine = null). ✓
- **Consistency:** `best_day ≤ 50% × Σ(positive days)` — correct rule. ✓
- **Pass:** `equity−start ≥ target AND trading_days ≥ min`. ✓
- **Reset/fee:** per-attempt flat cost ($49/$99/$199) at the EV layer — the *most favorable* (no monthly bleed). Not harsh.

## 6. Sensitivity (raw trade edge, NO wrapper)
| family | normal | half cost | zero comm | zero slip | **ZERO ALL** |
|---|---|---|---|---|---|
| gap_cont (frequent) PF | 0.94 | 0.99 | 0.98 | 1.00 | **1.05** |
| first_pb (rare) PF | 1.18 | 1.25 | 1.24 | 1.26 | **1.32** |
| open_drive PF | 0.84 | 0.90 | 0.89 | 0.91 | **0.97** |
| **NQ best PF** | **1.22** | — | **1.23** | — | ~1.23 |

- **Even at literally zero cost**, the frequent GC families are PF ≤ 1.05 (untradeable); only the rare `first_pb` is real (PF 1.32) → **failure is the strategy, not the cost.**
- NQ best is PF 1.22 at normal cost and **1.23 at zero cost** → cost is negligible to its edge.

### No-wrapper vs wrapper-only (50K)
| family | raw exp$/tr | raw PF | wrap pass% | wrap EV/mo |
|---|---|---|---|---|
| gap_cont | −2.00 | 0.94 | 0.50 | **+$34** |
| first_pb | +5.20 | 1.18 | **0.00** | −$3 |
| open_drive | −4.29 | 0.84 | 0.00 | −$5 |

- The wrapper does **not** kill a profitable strategy here — gap_cont's wrapper EV (+$34) is *higher* than its negative raw edge (variance + cushion + early-stop extract a little). 
- For the one genuinely +edge GC strategy (`first_pb`), the wrapper yields pass 0.00 — **because 19 trades/yr can't reach the profit target in the combine window (timeout)**, i.e. a *frequency* failure, not a wrapper defect.

## Conclusion
The platform is **sound**: economics correct, fills realistic, PnL exact, Topstep wrapper faithful. Nothing is artificially manufacturing the failures.
- **GC genuinely lacks a frequent positive-edge RTH strategy** (zero-cost PF ≤ 1.05 for frequent families); and gold's trend is overnight, unreachable under RTH-flat rules → GC is genuinely unsuitable for this Topstep plan.
- **NQ's best strategy genuinely works** (PF 1.22, +$744/mo/MNQ); it misses only the *$1,000/mo income* bar, and the binding constraint there is the real Topstep wrapper economics (MLL caps sizing, blow-ups, combine cost, payout timing → ~$400/mo robust) — not an engine bug. Path to more income = more accounts, not a code fix.

Code: `scripts/phase58_audit.py`; NQ trades `docs/phase40/frozen_trades_BALANCED.csv`.
