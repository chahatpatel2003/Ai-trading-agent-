# Phase 8 — Account-Size & Timeframe Feasibility Test

**Question:** Phase 7 proved the direction edge exists before costs but is not
Topstep-survivable on **50K / NQ / 1m** — the per-trade noise band is wider than the
$2,000 trailing drawdown. Does changing the *battlefield* fix that? Specifically:
does the edge become survivable under a **bigger account** (100K/150K), **different
instrument sizing** (MNQ/NQ/adaptive/max-1-NQ/MNQ-scaling), or a **higher timeframe**
(5m/15m, which might shrink the noise band relative to the bar's ATR)?

**Verdict: No. 0 of 1,161 candidates feasible across every account × timeframe ×
sizing combination.** No new strategies were searched; acceptance rules were not
relaxed. The edge is **formally retired** on NQ/MNQ for 50K/100K/150K and 1m/5m/15m.

This run was executed fresh against the current code (post the Topstep audit), so all
account cells now sweep all three focus families (`bias_pullback`, `pd_sweep_rev`,
`cvd_pullback`) and the matrix is extended to a full 3×3 account × timeframe grid plus
the 150K timeframe sweep. Every candidate received an ACCEPT / REJECT /
NEEDS_MORE_DATA verdict (no nulls). `make check` is green.

---

## The feasibility matrix

NQ/MNQ, train **2021-01-01 → 2023-12-31**, OOS **2024-01-01 → 2025-06-30**, focus
families `bias_pullback`, `pd_sweep_rev`, `cvd_pullback`. Exits seeded by Phase 7
MAE/MFE diagnostics per timeframe (stop beyond MAE, target near MFE, min-hold past the
losing early-bar window, time stop at the holding-zone end), NY-AM session. "Feasible"
= at least one candidate with **OOS > 0 AND ruin ≤ 10% AND not blown AND survives cost
stress**.

| Run (`report phase8 <id>`) | Trials | Accept | Best OOS | Min ruin @ OOS>0 | Feasible |
|---|---:|---:|---:|---:|:--:|
| `phase8_acct_50k_1m`   | 105 | 0 | +3,383  | 0% | **no** |
| `phase8_acct_100k_1m`  | 120 | 0 | +5,017  | 0% | **no** |
| `phase8_acct_150k_1m`  | 135 | 0 | +10,472 | 0% | **no** |
| `phase8_acct_50k_5m`   | 105 | 0 | 0       | — | **no** |
| `phase8_acct_100k_5m`  | 120 | 0 | 0       | — | **no** |
| `phase8_acct_150k_5m`  | 135 | 0 | 0       | — | **no** |
| `phase8_acct_50k_15m`  | 105 | 0 | +184    | 0% | **no** |
| `phase8_acct_100k_15m` | 120 | 0 | +184    | 0% | **no** |
| `phase8_acct_150k_15m` | 135 | 0 | +1,849  | 0% | **no** |
| `phase8_tf_1m_150k`    | 27  | 0 | +10,311 | 0% | **no** |
| `phase8_tf_5m_150k`    | 27  | 0 | 0       | — | **no** |
| `phase8_tf_15m_150k`   | 27  | 0 | +184    | 0% | **no** |
| **Total** | **1,161** | **0** | | | **none** |

Reproduce with `nqlab report phase8` (matrix) or `nqlab report phase8 <run_id>`
(full leaderboard for one combination). PBO is **61–86%** across the cells — the
leaderboard orderings are mostly overfit noise, exactly what you expect when no
genuine survivable configuration exists in the grid.

The NEEDS_MORE_DATA verdicts (the bulk of the 5m/15m cells) are trade-starvation:
the NY-AM context setups fire too rarely at higher timeframes to clear the
`min_trades`/`min_oos_trades` floor. Their Phase 6 directional edge does not survive
being gated down to a tradeable, survivable setup at those timeframes.

---

## Finding 1 — A bigger account widens the drawdown-survivable set, then exposes the *real* problem

On 1m, raising the account does exactly what theory predicts — it grows the set of
candidates that clear the trailing drawdown — and then a *different* gate binds:

| Account | Best OOS (1m) | …its ruin | …its max DD | OOS>0 **and** ruin≤10% **and** not blown | …of those, survive cost stress |
|---|---:|---:|---:|---:|---:|
| 50K  | +3,383  | 76% | $6,196  | **13** | **0** |
| 100K | +5,017  | 81% | $9,931  | **13** | **0** |
| 150K | +10,472 | 79% | $15,109 | **25** | **0** |

Read it carefully:

- **The drawdown problem really does shrink with size.** The "drawdown-survivable"
  set (OOS > 0, ruin ≤ 10%, account not blown) grows from 13 → 13 → **25** as the
  trailing drawdown widens ($2,000 → $3,000 → $4,500). At 150K the bigger budget even
  lets the highest-OOS candidate (`pd_sweep_rev`, +$10,472) *pass* cost stress and
  *not* blow on the single historical path.
- **But it is then killed by Monte-Carlo ruin or by cost stress — never zero of both.**
  The +$10,472 ceiling candidate is rejected purely on a **79% blow-up probability**:
  the one historical ordering survived, but 79% of reshuffled trade orderings breach
  the trailing drawdown. Every one of the 25 *drawdown*-survivable candidates (ruin ≤
  10%) is rejected on **cost/slippage stress** — **0 of 25** survive realistic
  commissions and slippage (and 0 clear walk-forward ≥ 50%).

So the bigger account did not make the edge feasible. It moved the binding constraint
from **"blows the account"** to either **"79% of orderings blow it"** (at sizes that
clear costs) or **"there is no edge after costs"** (at sizes that survive ruin).

The risk scissor on 150K/1m (`pd_sweep_rev`, averaged over the four micro-capable
sizing modes; **only the dollar risk changes**) is the whole story in one table:

| Risk/trade | OOS net | Risk of ruin | Max DD |
|---:|---:|---:|---:|
| $25  | +18    | 0%  | $106    |
| $50  | +286   | 0%  | $681    |
| $75  | +260   | 0%  | $1,911  |
| $100 | −112   | 5%  | $2,479  |
| $150 | +1,818 | 21% | $3,974  |
| $200 | +2,606 | 34% | $5,343  |
| $300 | +3,774 | 58% | $8,821  |
| $500 | +9,059 | 82% | $13,620 |

Size up to make money → ruin explodes (to 82% at $500). Size down to survive ruin →
the edge is ≈ 0 and inverts under costs. There is **no risk level where OOS > 0,
ruin ≤ 10%, and the edge survives transaction costs simultaneously.**

> A note on the ruin numbers (from the Topstep audit): the Monte-Carlo risk-of-ruin
> is evaluated at trade-close cumulative PnL, so it does **not** see within-trade
> intrabar excursions and therefore **understates** ruin — it is *optimistic*, not
> conservative. The authoritative blow-up check is the live engine's intraday `blown`
> flag (an independent hard disqualifier). The 76–82% ruin figures are thus a *lower
> bound*; the true blow rate is at least that high, which only strengthens the
> rejection.

---

## Finding 2 — Instrument sizing is a non-lever at survivable risk

At every risk level low enough to survive the drawdown ($10–$150), the four
micro-capable modes — `adaptive`, `mnq_only`, `max_1_nq`, `mnq_scaling` — produce
**identical** OOS / ruin / drawdown (e.g. 150K/1m/`pd_sweep_rev`/$150: all four give
OOS +2,311, ruin 26%, DD $4,926 on 987 trades). The reason: at $50–$150 of risk one
full NQ contract is unaffordable, so every micro-capable mode collapses to the *same*
small MNQ position.

`nq_only` is **degenerate** at these levels — it can only hold whole NQ contracts, so
at $150 risk it takes ~2 trades all run and posts OOS −153: it simply cannot express
the strategy at survivable size.

The modes only diverge at **$300/$500** (where NQ becomes affordable and
`mnq_scaling` stacks more micros) — and that divergence just walks back up the ruin
side of the scissor: more size, more ruin. Instrument granularity therefore matters
only when the binding constraint is position *quantization*; here the binding
constraint is the **post-cost per-trade edge**, which sizing cannot touch.

---

## Finding 3 — Higher timeframes *reduce* the edge, not the noise band

The hypothesis was that 5m/15m bars would shrink MAE/MFE relative to the drawdown
budget. The data says the opposite — higher timeframes mostly **destroy the signal**:

- **5m:** across all accounts and all sizing modes the **best OOS net is 0** — every
  configuration fails the `min_trades` floor (the NY-AM context setups fire too
  rarely). All 5m cells are NEEDS_MORE_DATA / 0-trade.
- **15m:** a faint signal returns but on **2–9 OOS trades** — best OOS +184 (50K/100K)
  to +1,849 (150K), all either NEEDS_MORE_DATA (too few OOS trades) or rejected. The
  150K/15m ceiling (+1,849) carries 38% ruin on just 2 OOS trades. The multiple-testing
  bar also explodes at higher TF (SR0 rises sharply) because there are too few
  near-independent trades to deflate to significance.

Higher timeframe trades fewer, noisier-per-sample setups — it raises the bar it needs
to clear while lowering the edge it has to clear it with. It is strictly worse here.

---

## Acceptance gates — all held strict

No gate was loosened. The verdict logic (`validation/verdict.py`) is unchanged from
Phase 5/6/7. Candidates were rejected for, across the 1,161 trials:

1. **Trade starvation** (NEEDS_MORE_DATA): < 30 total or < 10 OOS trades — the bulk of
   the 5m/15m cells.
2. **Risk of ruin > 10%** / Monte-Carlo blow-up at sizes large enough to clear costs.
3. **Fails cost/slippage stress** — edge inverts at 1×/2×/3× slippage or 1.5×
   commission (every drawdown-survivable 1m candidate).
4. **Walk-forward < 50%** of windows positive.
5. **Not consistent across years** — profitable in too few of the OOS years.
6. **DSR ≈ 0 / PBO 61–86%** — no deflated significance; ordering is overfit.

Most rejected candidates fail several of these at once.

---

## Decision

Per the Phase 8 rule — *retire if no account size / timeframe / instrument
combination can produce OOS > 0 with ruin ≤ 10% that also survives the full
gauntlet* — the edge is **retired**.

Precise statement of what was found (not over-claimed):

- **OOS > 0 with ruin ≤ 10% IS achievable** — most readily at 150K/1m (25 such
  candidates), some at 50K/100K/1m and a handful at 15m. **Every one fails
  cost/slippage stress and/or walk-forward and year-consistency.** None passes
  acceptance.
- **No combination passes the full acceptance criteria** (OOS > 0, ruin ≤ 10%,
  cost-stress survival, DSR, acceptable PBO, walk-forward stability, multi-year
  profitability, Topstep survival). **0 of 1,161.**
- The bigger account **widens the drawdown-survivable set and reveals that the
  underlying edge does not survive realistic transaction costs.** That is the true
  ceiling.
- **Instrument sizing is irrelevant** at survivable risk; **higher timeframes are
  strictly worse.**

> The Phase 6/7 "direction edge before costs" is real but **sub-transaction-cost**.
> Changing the battlefield — account size, instrument, timeframe — cannot manufacture
> an edge that does not exist after costs. The 50K result was a drawdown problem
> hiding a cost problem; the 100K/150K result is the cost-and-sequence-risk problem,
> exposed.

### What this closes, and what remains (future phases — not this one)

- **Closed:** the `bias_pullback` / `pd_sweep_rev` / NY-AM `cvd_pullback` families on
  NQ/MNQ, 50K–150K, 1m–15m. No battlefield change makes them survivable.
- **Not addressed here:** edges whose raw per-trade gross exceeds costs by a real
  margin (this one's gross edge is ≈ its cost). Finding such an edge means a new
  signal family — explicitly out of Phase 8's scope, and not justified by anything
  seen so far.

Per the standing rule — *no edge until proven; reject honestly, do not force an
accept* — Phase 8 ends with **zero strategies promoted** and the positive-direction
edge formally retired across the tested account sizes, instruments, and timeframes.

---

## Reproduction

```
# data (bars + MBP1 features for 1m/5m/15m) already built; --no-build replays it
nqlab feasibility accounts   --timeframe 1m  --accounts 50k,100k,150k --no-build
nqlab feasibility accounts   --timeframe 5m  --accounts 50k,100k,150k --no-build
nqlab feasibility accounts   --timeframe 15m --accounts 50k,100k,150k --no-build
nqlab feasibility timeframes --timeframes 1m,5m,15m --account 150k     --no-build
nqlab report phase8                                  # 12-cell feasibility matrix
nqlab report phase8 phase8_acct_150k_1m              # one combination's leaderboard
```

Raw market data was never modified; MBP1 missing periods past 2025-H1 were not
fabricated. The full sweep (1,161 candidates through the gauntlet) completed in ~42
minutes on the dev profile.
