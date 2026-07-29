#!/usr/bin/env python3
"""
Strategy Tournament — Backtests all strategies, ranks them, and picks the best.
Outputs a detailed comparison table with all metrics.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backtest.engine import BacktestEngine
from config.loader import load_config
from data.provider import get_data
from data.splitter import chronological_split
from logs.logger import setup_logging
from paper_trade.runner import build_strategies


def run_tournament(config_path=None):
    cfg = load_config(config_path)
    setup_logging(
        log_dir=cfg.get("logging", {}).get("dir", "./logs"),
        level=cfg.get("logging", {}).get("level", "INFO"),
    )

    print("=" * 80)
    print("GOLD TRADING BOT — STRATEGY TOURNAMENT")
    print("=" * 80)

    # Fetch data
    print("\n[1/4] Fetching data from TradingView...")
    df = get_data(cfg)
    print(f"      Loaded {len(df)} bars: {df.index[0]} → {df.index[-1]}")

    # Split
    print("\n[2/4] Splitting data...")
    split = chronological_split(
        df, cfg["data"]["train_pct"], cfg["data"]["val_pct"], cfg["data"]["test_pct"]
    )
    test_data = split.test
    print(f"      Test set: {len(test_data)} bars ({split.split_info['test_range'][0]} → {split.split_info['test_range'][1]})")

    # Build all strategies
    strategies = build_strategies(cfg)
    print(f"\n[3/4] Running {len(strategies)} strategies...\n")

    engine = BacktestEngine(cfg)
    results = []

    for strat in strategies:
        try:
            result = engine.run(strat, test_data)
            results.append(result)
            m = result.metrics
            status = "OK" if m.num_trades > 0 else "NO TRADES"
            print(f"  {strat.name:25s}  trades={m.num_trades:4d}  return={m.total_return_pct:8.2f}%  "
                  f"sharpe={m.sharpe_ratio:7.3f}  maxDD={m.max_drawdown_pct:6.2f}%  "
                  f"winrate={m.win_rate_pct:5.1f}%  PF={m.profit_factor:6.3f}  [{status}]")
        except Exception as e:
            print(f"  {strat.name:25s}  ERROR: {e}")

    # Rank strategies
    print("\n" + "=" * 80)
    print("[4/4] FINAL RANKING")
    print("=" * 80)

    # Filter out strategies with no trades
    valid = [r for r in results if r.metrics.num_trades >= 3]

    if not valid:
        print("\nNo strategies produced enough trades. Try a longer date range.")
        return results

    # Composite score: Sharpe (40%) + Profit Factor (20%) + Win Rate (15%) +
    # Stability (15%) + inverse MaxDD (10%)
    for r in valid:
        m = r.metrics
        sharpe_score = max(m.sharpe_ratio, -2) / 3  # normalize -2 to 3 → -0.67 to 1
        pf_score = min(m.profit_factor, 3) / 3  # cap at 3
        wr_score = m.win_rate_pct / 100
        stab_score = m.stability
        dd_score = max(0, 1 - m.max_drawdown_pct / 30)  # 30% DD = 0 score
        r._composite = (
            sharpe_score * 0.40
            + pf_score * 0.20
            + wr_score * 0.15
            + stab_score * 0.15
            + dd_score * 0.10
        )

    ranked = sorted(valid, key=lambda r: r._composite, reverse=True)

    print(f"\n{'Rank':<5} {'Strategy':<25} {'Composite':>9} {'Sharpe':>8} {'Return%':>9} "
          f"{'MaxDD%':>7} {'WinRate':>8} {'PF':>7} {'Trades':>7} {'Stability':>10}")
    print("-" * 105)

    for i, r in enumerate(ranked, 1):
        m = r.metrics
        marker = " ★ BEST" if i == 1 else ""
        print(f"  #{i:<3} {r.strategy_name:<25} {r._composite:>8.4f} {m.sharpe_ratio:>8.3f} "
              f"{m.total_return_pct:>8.2f}% {m.max_drawdown_pct:>6.2f}% {m.win_rate_pct:>7.1f}% "
              f"{m.profit_factor:>6.3f} {m.num_trades:>6d} {m.stability:>9.4f}{marker}")

    best = ranked[0]
    print(f"\n{'=' * 80}")
    print(f"WINNER: {best.strategy_name}")
    print(f"{'=' * 80}")
    print(f"  Sharpe Ratio:      {best.metrics.sharpe_ratio}")
    print(f"  Total Return:      {best.metrics.total_return_pct}%")
    print(f"  Max Drawdown:      {best.metrics.max_drawdown_pct}%")
    print(f"  Win Rate:          {best.metrics.win_rate_pct}%")
    print(f"  Profit Factor:     {best.metrics.profit_factor}")
    print(f"  Total Trades:      {best.metrics.num_trades}")
    print(f"  Avg Win:           ${best.metrics.avg_win}")
    print(f"  Avg Loss:          ${best.metrics.avg_loss}")
    print(f"  Expectancy/Trade:  ${best.metrics.expectancy_per_trade}")
    print(f"  Stability:         {best.metrics.stability}")
    print(f"  Sortino Ratio:     {best.metrics.sortino_ratio}")
    print(f"  Composite Score:   {best._composite:.4f}")

    return results


if __name__ == "__main__":
    run_tournament()
