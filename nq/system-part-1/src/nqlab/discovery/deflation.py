"""Multiple-testing correction: Deflated Sharpe Ratio and PBO.

Searching many strategies inflates the best in-sample Sharpe even with no edge.
The Deflated Sharpe Ratio (Bailey & Lopez de Prado) discounts a candidate's
Sharpe by the *expected maximum* Sharpe under the null given the number of trials
and the dispersion of trial Sharpes; it returns the probability the true Sharpe
is positive. PBO (Probability of Backtest Overfitting, CSCV) is a run-level
diagnostic: how often the in-sample-best candidate underperforms out-of-sample.

Uses ``statistics.NormalDist`` (stdlib) — no SciPy dependency.
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist, pvariance

_N = NormalDist()
_EULER = 0.5772156649015329


def expected_max_sharpe(sharpe_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe under the null across ``n_trials`` independent trials."""
    if n_trials <= 1 or sharpe_variance <= 0:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - _EULER) * a + _EULER * b)


def deflated_sharpe_ratio(
    sharpe: float, sr0: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """Probability that the true Sharpe exceeds ``sr0`` (i.e. the deflated benchmark)."""
    if n_obs <= 1:
        return 0.0
    denom = math.sqrt(max(1e-12, 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2))
    z = (sharpe - sr0) * math.sqrt(n_obs - 1) / denom
    return _N.cdf(z)


def sharpe_variance(sharpes: list[float]) -> float:
    """Population variance of the trial Sharpe ratios."""
    return pvariance(sharpes) if len(sharpes) >= 2 else 0.0


def probability_of_backtest_overfitting(matrix: list[list[float]]) -> float | None:
    """CSCV PBO from a candidate x window performance matrix (None if infeasible).

    Windows are split combinatorially into in-sample / out-of-sample halves; for
    each split the in-sample-best candidate's out-of-sample rank gives a logit, and
    PBO is the fraction of splits where that logit is <= 0 (below the OOS median).
    """
    n_candidates = len(matrix)
    if n_candidates < 2:
        return None
    n_windows = len(matrix[0])
    if n_windows < 4:
        return None
    half = n_windows // 2
    window_idx = list(range(n_windows))
    splits = list(combinations(window_idx, half))
    if len(splits) > 2000:  # cap combinatorial blow-up
        splits = splits[:2000]

    below = 0
    counted = 0
    for is_idx in splits:
        oos_idx = [w for w in window_idx if w not in is_idx]
        is_perf = [sum(matrix[c][w] for w in is_idx) for c in range(n_candidates)]
        oos_perf = [sum(matrix[c][w] for w in oos_idx) for c in range(n_candidates)]
        best = max(range(n_candidates), key=lambda c: is_perf[c])
        # Relative OOS rank of the IS-best candidate.
        rank = sum(1 for c in range(n_candidates) if oos_perf[c] <= oos_perf[best])
        w = rank / (n_candidates + 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logit = math.log(w / (1 - w))
        below += 1 if logit <= 0 else 0
        counted += 1
    return below / counted if counted else None
