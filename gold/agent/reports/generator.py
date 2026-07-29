"""
Report generator — produces strategy comparison reports.
Rankings are by ROBUSTNESS (Sharpe, stability, drawdown), not raw return.
"""
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult
from backtest.monte_carlo import MonteCarloResult
from backtest.walk_forward import WalkForwardResult
from logs.logger import get_logger

log = get_logger("reports.generator")


def generate_comparison_report(
    backtest_results: list[BacktestResult],
    wf_results: list[WalkForwardResult] = None,
    mc_results: list[MonteCarloResult] = None,
    paper_results: dict = None,
    output_dir: str = "./reports",
) -> str:
    """
    Generate a comprehensive strategy comparison report.
    Strategies are ranked by ROBUSTNESS SCORE, not raw return.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "generated_at": datetime.now().isoformat(),
        "disclaimer": (
            "DISCLAIMER: Past performance does NOT guarantee future results. "
            "Backtest results are hypothetical and subject to overfitting. "
            "Paper trading results may not survive live trading conditions. "
            "See 'Why Paper Profits May Not Survive Live Trading' section."
        ),
        "strategies": {},
        "ranking": [],
    }

    # Collect all strategy metrics
    for bt in backtest_results:
        entry = {
            "backtest_metrics": _metrics_to_dict(bt.metrics),
            "signals_generated": bt.signals_generated,
            "signals_rejected": bt.signals_rejected,
            "rejection_reasons": bt.rejection_reasons,
        }

        # Add walk-forward results if available
        if wf_results:
            wf = next((w for w in wf_results if w.strategy_name == bt.strategy_name), None)
            if wf:
                entry["walk_forward"] = {
                    "oos_metrics": _metrics_to_dict(wf.combined_metrics),
                    "window_details": wf.window_details,
                    "n_windows": len(wf.window_details),
                }

        # Add Monte Carlo results if available
        if mc_results:
            mc = next((m for m in mc_results if m.strategy_name == bt.strategy_name), None)
            if mc:
                entry["monte_carlo"] = {
                    "sharpe_ci": mc.sharpe_ci,
                    "max_dd_ci": mc.max_dd_ci,
                    "return_ci": mc.return_ci,
                    "worst_sharpe": mc.worst_sharpe,
                    "worst_max_dd": mc.worst_max_dd,
                    "prob_ruin_10pct": mc.prob_ruin_10pct,
                    "prob_ruin_20pct": mc.prob_ruin_20pct,
                }

        # Add paper trading results if available
        if paper_results and bt.strategy_name in paper_results:
            entry["paper_trading"] = paper_results[bt.strategy_name]

        report["strategies"][bt.strategy_name] = entry

    # Compute robustness score and rank
    rankings = []
    for name, data in report["strategies"].items():
        score = _compute_robustness_score(data)
        rankings.append({
            "strategy": name,
            "robustness_score": round(score, 4),
            "sharpe": data["backtest_metrics"]["sharpe_ratio"],
            "max_drawdown": data["backtest_metrics"]["max_drawdown_pct"],
            "profit_factor": data["backtest_metrics"]["profit_factor"],
            "stability": data["backtest_metrics"]["stability"],
            "num_trades": data["backtest_metrics"]["num_trades"],
        })

    rankings.sort(key=lambda x: x["robustness_score"], reverse=True)
    report["ranking"] = rankings

    # Add failure analysis
    report["why_strategies_may_fail"] = _failure_analysis()
    report["why_paper_profits_may_not_survive"] = _paper_to_live_warning()

    # Save report
    report_path = out / f"strategy_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Also generate a human-readable summary
    summary_path = out / f"strategy_summary_{timestamp}.txt"
    _write_text_summary(report, summary_path)

    log.info("report_generated", path=str(report_path))
    return str(report_path)


def _compute_robustness_score(data: dict) -> float:
    """
    Robustness score weights stability and risk-adjusted returns
    MORE than raw return. Penalizes high drawdown and low trade count.
    """
    m = data["backtest_metrics"]
    score = 0.0

    # Sharpe (30% weight)
    sharpe = max(m["sharpe_ratio"], -2.0)
    score += 0.30 * min(sharpe / 2.0, 1.0)

    # Stability (20% weight)
    score += 0.20 * m["stability"]

    # Max drawdown penalty (20% weight — lower is better)
    dd = m["max_drawdown_pct"]
    dd_score = max(0, 1.0 - dd / 30.0)  # 0% DD = 1.0, 30% DD = 0.0
    score += 0.20 * dd_score

    # Profit factor (15% weight)
    pf = min(m["profit_factor"], 3.0)
    score += 0.15 * (pf / 3.0)

    # Trade count (15% weight — more trades = more statistical significance)
    trades = m["num_trades"]
    trade_score = min(trades / 100, 1.0)
    score += 0.15 * trade_score

    # Bonus/penalty for walk-forward consistency
    if "walk_forward" in data:
        wf = data["walk_forward"]["oos_metrics"]
        oos_sharpe = wf["sharpe_ratio"]
        # Penalize if OOS Sharpe is much worse than in-sample
        if oos_sharpe < sharpe * 0.5:
            score *= 0.7  # 30% penalty for likely overfitting

    return score


def _metrics_to_dict(m) -> dict:
    return {
        "total_return_pct": m.total_return_pct,
        "cagr_pct": m.cagr_pct,
        "max_drawdown_pct": m.max_drawdown_pct,
        "sharpe_ratio": m.sharpe_ratio,
        "sortino_ratio": m.sortino_ratio,
        "profit_factor": m.profit_factor,
        "win_rate_pct": m.win_rate_pct,
        "expectancy_per_trade": m.expectancy_per_trade,
        "exposure_time_pct": m.exposure_time_pct,
        "num_trades": m.num_trades,
        "avg_win": m.avg_win,
        "avg_loss": m.avg_loss,
        "max_consecutive_wins": m.max_consecutive_wins,
        "max_consecutive_losses": m.max_consecutive_losses,
        "stability": m.stability,
    }


def _failure_analysis() -> dict:
    return {
        "trend_following": (
            "Whipsaws in ranging markets. Moving averages lag inherently. "
            "Gold can range for months, generating many small losses. "
            "Regime filter helps but cannot predict transitions."
        ),
        "mean_reversion": (
            "Gets crushed in strong trends. 'The market can stay irrational "
            "longer than you can stay solvent.' Volatility filter limits trade count. "
            "The concept of 'mean' shifts during regime changes."
        ),
        "breakout": (
            "False breakouts are extremely common. Low trade frequency limits "
            "statistical power. News-driven spikes look like breakouts but reverse. "
            "Asian range may be too tight or too wide."
        ),
        "momentum": (
            "All indicators lag — by the time they confirm, the move may be over. "
            "RSI stays overbought/oversold for extended periods in strong trends. "
            "MACD generates constant whipsaws in choppy markets."
        ),
        "ensemble": (
            "Only as good as its components. Weighting by recent Sharpe is itself "
            "a form of curve-fitting. If all strategies fail in the same regime, "
            "the ensemble fails too. Added complexity may not help."
        ),
    }


def _paper_to_live_warning() -> str:
    return """
WHY PAPER PROFITS MAY NOT SURVIVE LIVE TRADING
================================================

1. SLIPPAGE IS WORSE IN REALITY
   Paper trading assumes constant slippage. Real slippage varies wildly
   during news events, low liquidity, and market gaps. Gold can gap
   significantly on geopolitical events.

2. SPREAD WIDENS UNPREDICTABLY
   Retail spreads on XAUUSD can widen to 5-10x normal during news releases,
   market opens, and low-liquidity periods (e.g., around Christmas/New Year).

3. EXECUTION LATENCY
   Paper trading executes instantly. Real orders take time to fill, and
   prices can move against you during that time. This is especially true
   for strategies that depend on precise entry levels.

4. FILL QUALITY VARIES
   Market orders may not fill at the expected price. Limit orders may not
   fill at all. Partial fills create sizing problems.

5. DATA FEED ISSUES
   Missing bars, delayed data, and price feed disconnections are common.
   Paper trading uses clean historical data; live data is messy.

6. PSYCHOLOGICAL FACTORS
   Fear and greed cause traders to override their system. You may cut
   winners too early, hold losers too long, or skip signals after a
   losing streak.

7. REGIME CHANGES
   The market regime during your backtest/paper period may not persist.
   A strategy optimized for trending markets will fail in range-bound markets.

8. OVERFITTING
   Despite walk-forward testing and Monte Carlo analysis, some degree of
   overfitting to historical patterns is inevitable. The future WILL differ
   from the past.

9. BLACK SWAN EVENTS
   Gold responds violently to geopolitical crises, central bank announcements,
   and systemic financial events that are inherently unpredictable.

10. BROKER-SPECIFIC RISKS
    Real brokers may reject orders, change margin requirements, have outages,
    or modify trading conditions without warning.

RECOMMENDATION: Start live trading with 25-50% of intended position size.
Monitor for at least 2-4 weeks at reduced size. Only scale up if live results
match paper trading within the configured tolerance.
"""


def _write_text_summary(report: dict, path: Path) -> None:
    lines = [
        "=" * 70,
        "GOLD TRADING BOT — STRATEGY COMPARISON REPORT",
        f"Generated: {report['generated_at']}",
        "=" * 70,
        "",
        report["disclaimer"],
        "",
        "-" * 70,
        "STRATEGY RANKING (by robustness score, NOT raw return)",
        "-" * 70,
        "",
    ]

    for i, r in enumerate(report["ranking"], 1):
        lines.append(f"  #{i}  {r['strategy']}")
        lines.append(f"      Robustness Score : {r['robustness_score']}")
        lines.append(f"      Sharpe Ratio     : {r['sharpe']}")
        lines.append(f"      Max Drawdown     : {r['max_drawdown']}%")
        lines.append(f"      Profit Factor    : {r['profit_factor']}")
        lines.append(f"      Stability (R²)   : {r['stability']}")
        lines.append(f"      Total Trades     : {r['num_trades']}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("WHY EACH STRATEGY MAY FAIL")
    lines.append("-" * 70)
    for name, reason in report["why_strategies_may_fail"].items():
        lines.append(f"\n  {name}:")
        lines.append(f"    {reason}")

    lines.append("")
    lines.append(report["why_paper_profits_may_not_survive"])

    with open(path, "w") as f:
        f.write("\n".join(lines))
