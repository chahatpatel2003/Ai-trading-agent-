from __future__ import annotations

from nqlab.discovery.deflation import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    sharpe_variance,
)


def test_expected_max_sharpe_increases_with_trials() -> None:
    var = 0.04
    assert expected_max_sharpe(var, 2) < expected_max_sharpe(var, 100)
    assert expected_max_sharpe(var, 1) == 0.0  # one trial -> no inflation


def test_more_trials_lower_deflated_sharpe() -> None:
    # Same observed Sharpe; deflate against a higher SR0 (more trials) -> lower DSR.
    sr0_few = expected_max_sharpe(0.04, 5)
    sr0_many = expected_max_sharpe(0.04, 500)
    dsr_few = deflated_sharpe_ratio(0.30, sr0_few, n_obs=200, skew=0.0, kurtosis=3.0)
    dsr_many = deflated_sharpe_ratio(0.30, sr0_many, n_obs=200, skew=0.0, kurtosis=3.0)
    assert 0.0 <= dsr_many <= dsr_few <= 1.0


def test_strong_sharpe_passes_deflation() -> None:
    dsr = deflated_sharpe_ratio(0.5, 0.05, n_obs=500, skew=0.0, kurtosis=3.0)
    assert dsr > 0.9


def test_sharpe_variance() -> None:
    assert sharpe_variance([0.1]) == 0.0
    assert sharpe_variance([0.0, 0.2, 0.4]) > 0.0


def test_pbo_bounds_and_small_matrix() -> None:
    assert probability_of_backtest_overfitting([[1.0, 2.0]]) is None  # too few candidates
    matrix = [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0], [0.5, 0.5, 0.5, 0.5]]
    pbo = probability_of_backtest_overfitting(matrix)
    assert pbo is not None
    assert 0.0 <= pbo <= 1.0
