"""Monte-Carlo trade reshuffling -> blow-up probability / risk-of-ruin.

The realized trade PnLs are reshuffled many times (seeded -> deterministic).
Each ordering is replayed as an equity path under the funded-account trailing
drawdown; the fraction of paths that breach is the blow-up probability, which we
report as the risk-of-ruin estimate. Reshuffling order (not resampling) keeps the
trade multiset fixed and isolates *path/sequence* risk.

The trailing floor here is identical to :attr:`TopstepAccount.trailing_floor`
(``min(peak - trailing, start)`` under lock). One thing differs by design: breach
is tested only at **trade-close cumulative PnL**, so within-trade intrabar adverse
excursions are not seen (this estimator has no intrabar path to sample). A trade
that dips through the floor intraday but closes back above it is *missed* here, so
this **understates** risk of ruin — the estimate is **optimistic, not
conservative**. The live engine (``trailing_basis="intraday"``) marks the bar's
adverse extreme and therefore applies the **stricter** check; treat its
chronological ``blown`` flag as the authoritative blow-up verdict. ``risk_of_ruin``
adds the orthogonal sequence-risk dimension (across reshufflings) on top of that.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from nqlab.config.models import TopstepRules


@dataclass
class MonteCarloResult:
    runs: int
    blow_up_prob: float
    risk_of_ruin: float
    median_final_pnl: float
    p05_final_pnl: float
    median_max_drawdown: float
    p95_max_drawdown: float


def _breaches(pnls: list[float], start: float, trailing: float, lock: bool) -> bool:
    equity = peak = start
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        floor = min(peak - trailing, start) if lock else peak - trailing
        if equity <= floor:
            return True
    return False


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]


def monte_carlo(
    trade_pnls: list[float], rules: TopstepRules, *, runs: int, seed: int
) -> MonteCarloResult:
    """Reshuffle trade order ``runs`` times and measure path risk."""
    if not trade_pnls:
        return MonteCarloResult(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rng = random.Random(seed)
    base = list(trade_pnls)
    blow_ups = 0
    finals: list[float] = []
    max_dds: list[float] = []

    for _ in range(runs):
        perm = base[:]
        rng.shuffle(perm)
        if _breaches(
            perm, rules.account_size, rules.trailing_max_loss, rules.lock_trailing_at_start
        ):
            blow_ups += 1
        equity = peak = mdd = 0.0
        for p in perm:
            equity += p
            peak = max(peak, equity)
            mdd = max(mdd, peak - equity)
        finals.append(equity)
        max_dds.append(mdd)

    finals.sort()
    max_dds.sort()
    prob = blow_ups / runs
    return MonteCarloResult(
        runs=runs,
        blow_up_prob=prob,
        risk_of_ruin=prob,
        median_final_pnl=round(_percentile(finals, 0.5), 2),
        p05_final_pnl=round(_percentile(finals, 0.05), 2),
        median_max_drawdown=round(_percentile(max_dds, 0.5), 2),
        p95_max_drawdown=round(_percentile(max_dds, 0.95), 2),
    )
