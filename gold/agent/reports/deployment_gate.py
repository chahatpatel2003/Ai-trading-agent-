"""
Deployment Gate Checker.
ALL conditions must pass before the bot can be marked 'ready for cautious live testing'.
This is conservative by design.
"""
from dataclasses import dataclass

from backtest.engine import BacktestResult
from backtest.monte_carlo import MonteCarloResult
from backtest.walk_forward import WalkForwardResult
from logs.logger import get_logger

log = get_logger("reports.deployment_gate")


@dataclass
class GateCheckResult:
    gate_name: str
    passed: bool
    actual_value: float
    threshold: float
    message: str


def check_deployment_gates(
    backtest_result: BacktestResult,
    wf_result: WalkForwardResult = None,
    mc_result: MonteCarloResult = None,
    paper_result: dict = None,
    cfg: dict = None,
) -> tuple[bool, list[GateCheckResult]]:
    """
    Run ALL deployment gate checks. ALL must pass.
    Returns (all_passed, list_of_check_results).
    """
    gate_cfg = cfg.get("deployment_gate", {}) if cfg else {}
    min_sharpe = gate_cfg.get("min_oos_sharpe", 0.5)
    max_dd = gate_cfg.get("max_oos_drawdown_pct", 0.15)
    min_pf = gate_cfg.get("min_profit_factor", 1.2)
    max_deviation = gate_cfg.get("max_backtest_paper_deviation_pct", 0.30)
    min_regimes = gate_cfg.get("min_regime_count", 3)

    checks = []

    # Gate 1: Out-of-sample Sharpe ratio
    if wf_result:
        oos_sharpe = wf_result.combined_metrics.sharpe_ratio
    else:
        oos_sharpe = backtest_result.metrics.sharpe_ratio
    checks.append(GateCheckResult(
        gate_name="Out-of-Sample Sharpe Ratio",
        passed=oos_sharpe >= min_sharpe,
        actual_value=oos_sharpe,
        threshold=min_sharpe,
        message=f"OOS Sharpe {oos_sharpe:.3f} {'>='}  threshold {min_sharpe}",
    ))

    # Gate 2: Maximum drawdown below cap
    if wf_result:
        dd = wf_result.combined_metrics.max_drawdown_pct / 100
    else:
        dd = backtest_result.metrics.max_drawdown_pct / 100
    checks.append(GateCheckResult(
        gate_name="Maximum Drawdown",
        passed=dd <= max_dd,
        actual_value=round(dd * 100, 2),
        threshold=max_dd * 100,
        message=f"Max DD {dd*100:.2f}% {'<='} cap {max_dd*100:.1f}%",
    ))

    # Gate 3: Profit factor
    pf = backtest_result.metrics.profit_factor
    checks.append(GateCheckResult(
        gate_name="Profit Factor",
        passed=pf >= min_pf,
        actual_value=pf,
        threshold=min_pf,
        message=f"Profit Factor {pf:.3f} {'>='} threshold {min_pf}",
    ))

    # Gate 4: Paper trading matches backtest
    if paper_result:
        bt_return = backtest_result.metrics.total_return_pct
        paper_return = paper_result.get("total_return_pct", 0)
        if bt_return != 0:
            deviation = abs(paper_return - bt_return) / abs(bt_return)
        else:
            deviation = abs(paper_return) if paper_return != 0 else 0
        checks.append(GateCheckResult(
            gate_name="Paper vs Backtest Deviation",
            passed=deviation <= max_deviation,
            actual_value=round(deviation * 100, 2),
            threshold=max_deviation * 100,
            message=f"Deviation {deviation*100:.1f}% {'<='} tolerance {max_deviation*100:.0f}%",
        ))
    else:
        checks.append(GateCheckResult(
            gate_name="Paper vs Backtest Deviation",
            passed=False,
            actual_value=0,
            threshold=max_deviation * 100,
            message="Paper trading not yet completed",
        ))

    # Gate 5: No single regime dominance (walk-forward window consistency)
    if wf_result and len(wf_result.window_details) >= min_regimes:
        # Check that not all positive returns come from one window
        returns = [w["oos_return"] for w in wf_result.window_details]
        positive_windows = sum(1 for r in returns if r > 0)
        checks.append(GateCheckResult(
            gate_name="Multi-Regime Validation",
            passed=positive_windows >= min_regimes,
            actual_value=positive_windows,
            threshold=min_regimes,
            message=f"{positive_windows} profitable windows out of {len(returns)} (need >= {min_regimes})",
        ))
    else:
        checks.append(GateCheckResult(
            gate_name="Multi-Regime Validation",
            passed=False,
            actual_value=0,
            threshold=min_regimes,
            message="Insufficient walk-forward windows for regime validation",
        ))

    # Gate 6: Monte Carlo robustness
    if mc_result:
        checks.append(GateCheckResult(
            gate_name="Monte Carlo Ruin Probability",
            passed=mc_result.prob_ruin_20pct < 0.10,
            actual_value=mc_result.prob_ruin_20pct,
            threshold=0.10,
            message=f"P(20% ruin) = {mc_result.prob_ruin_20pct:.4f} (need < 0.10)",
        ))
    else:
        checks.append(GateCheckResult(
            gate_name="Monte Carlo Ruin Probability",
            passed=False,
            actual_value=0,
            threshold=0.10,
            message="Monte Carlo analysis not yet run",
        ))

    # Gate 7: Minimum trade count for statistical significance
    min_trades = 30
    trades = backtest_result.metrics.num_trades
    checks.append(GateCheckResult(
        gate_name="Minimum Trade Count",
        passed=trades >= min_trades,
        actual_value=trades,
        threshold=min_trades,
        message=f"{trades} trades (need >= {min_trades} for significance)",
    ))

    all_passed = all(c.passed for c in checks)

    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        log.info(f"gate_check_{status.lower()}", gate=c.gate_name, message=c.message)

    if all_passed:
        log.info("ALL_GATES_PASSED", message="Bot may be considered for CAUTIOUS live testing")
    else:
        failed = [c.gate_name for c in checks if not c.passed]
        log.warning("GATES_FAILED", failed_gates=failed)

    return all_passed, checks


def print_gate_report(checks: list[GateCheckResult]) -> str:
    """Format gate results for display."""
    lines = [
        "=" * 60,
        "DEPLOYMENT GATE CHECK RESULTS",
        "=" * 60,
        "",
    ]

    all_passed = all(c.passed for c in checks)

    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        icon = "[+]" if c.passed else "[X]"
        lines.append(f"  {icon} {c.gate_name}")
        lines.append(f"      {c.message}")
        lines.append("")

    lines.append("-" * 60)
    if all_passed:
        lines.append("RESULT: ALL GATES PASSED")
        lines.append("The bot MAY be considered for cautious live testing.")
        lines.append("This does NOT guarantee profits. Start with reduced size.")
    else:
        lines.append("RESULT: ONE OR MORE GATES FAILED")
        lines.append("Live trading should remain DISABLED.")
        lines.append("Continue paper trading and investigation.")
    lines.append("-" * 60)

    return "\n".join(lines)
