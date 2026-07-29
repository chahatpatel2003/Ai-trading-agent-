# Gold Trading Bot (XAUUSD)

A production-ready, risk-first paper-trading bot for gold (XAUUSD) with a strict path to live deployment.

**Live trading is DISABLED by default and requires passing ALL deployment gates.**

---

## DISCLAIMER

This software is for educational and paper-trading purposes only. Past backtest and paper trading performance does NOT guarantee future results. Trading gold involves significant financial risk. You may lose money. The authors make no claims about profitability.

---

## Quick Start

```bash
# 1. Clone and set up environment
cd gold_trading_bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Copy environment config
cp .env.example .env

# 3. Run backtest (with walk-forward + Monte Carlo)
python main.py backtest

# 4. Run paper trading simulation
python main.py paper-trade

# 5. Check deployment gates
python main.py gate-check

# 6. Generate full comparison report
python main.py report

# 7. Run tests
pytest tests/ -v
```

## Project Structure

```
gold_trading_bot/
├── main.py                    # CLI entry point
├── config/
│   ├── default.yaml           # Master configuration
│   └── loader.py              # Config + env loader
├── data/
│   ├── provider.py            # Data fetching (yfinance, CSV)
│   ├── splitter.py            # Train/val/test split, regime labeling
│   └── indicators.py          # Technical indicators (SMA, EMA, ATR, RSI, MACD, etc.)
├── strategies/
│   ├── base.py                # Base strategy interface
│   ├── trend_following.py     # MA crossover + ATR stop + trailing stop
│   ├── mean_reversion.py      # Bollinger Bands + z-score + vol filter
│   ├── breakout.py            # Asian session range breakout
│   ├── momentum.py            # RSI + MACD + ADX filter
│   └── ensemble.py            # Meta-strategy (voting + weighted consensus)
├── risk/
│   └── manager.py             # Position sizing, drawdown limits, kill switches
├── backtest/
│   ├── engine.py              # Bar-by-bar backtest with realistic costs
│   ├── metrics.py             # Sharpe, Sortino, drawdown, profit factor, etc.
│   ├── walk_forward.py        # Walk-forward out-of-sample testing
│   └── monte_carlo.py         # Monte Carlo trade resampling
├── broker/
│   ├── base.py                # Broker abstraction interface
│   └── paper.py               # Paper broker (simulated fills)
├── paper_trade/
│   └── runner.py              # Paper trading simulation loop
├── live_trade_disabled/
│   └── gate.py                # Live trading gate (DISABLED by default)
├── reports/
│   ├── generator.py           # Strategy comparison reports
│   └── deployment_gate.py     # Deployment readiness checker
├── logs/
│   └── logger.py              # Structured logging (structlog)
├── tests/
│   ├── test_risk_manager.py
│   ├── test_indicators.py
│   ├── test_strategies.py
│   ├── test_backtest.py
│   └── test_live_gate.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Strategies

| Strategy | Logic | Best Regime | Worst Regime |
|----------|-------|-------------|--------------|
| Trend Following | MA crossover + regime filter + ATR trailing stop | Strong trends | Ranging/choppy |
| Mean Reversion | Bollinger Band / z-score entry + vol filter | Range-bound | Strong trends |
| Breakout | Asian session range breakout + confirmation bars | Volatile opens | Quiet/low-vol |
| Momentum | RSI + MACD + ADX confirmation | Established trends | Choppy/reversals |
| Ensemble | Voting consensus of above strategies | Mixed | Correlated failures |

## Risk Management

- **Per-trade risk**: 0.25% of equity (configurable, max 0.5%)
- **Daily loss limit**: 2% — stops all new trades for the day
- **Total drawdown limit**: 10% — full kill switch
- **Position sizing**: Based on ATR/volatility and stop distance
- **No martingale**: Position size DECREASES after losses (equity-based)
- **Kill switches**: Abnormal spread, data gaps, consecutive losses
- **News blackout**: Configurable flat exposure during major releases

## CLI Commands

```bash
# Full backtest with all strategies
python main.py backtest

# Backtest a specific strategy
python main.py backtest --strategy trend_following

# Skip Monte Carlo (faster)
python main.py backtest --no-monte-carlo

# Paper trade all strategies
python main.py paper-trade

# Check if ready for live deployment
python main.py gate-check --strategy trend_following

# Attempt live trading (WILL BE BLOCKED by default)
python main.py live
```

## Deployment Gate Checklist

The bot may only be marked "ready for cautious live testing" if ALL are true:

- [ ] Out-of-sample Sharpe ratio >= 0.5
- [ ] Maximum drawdown <= 15%
- [ ] Profit factor >= 1.2
- [ ] Paper trading results match backtest within 30% tolerance
- [ ] Profitable across 3+ market regime windows
- [ ] Monte Carlo P(20% ruin) < 10%
- [ ] Minimum 30 trades for statistical significance
- [ ] ENABLE_LIVE_TRADING=true in .env
- [ ] live_trading_enabled: true in config YAML
- [ ] Risks explicitly acknowledged in code

## Why Paper Profits May Not Survive Live Trading

1. **Slippage is worse in reality** — varies wildly during news/low liquidity
2. **Spreads widen unpredictably** — can reach 5-10x normal during events
3. **Execution latency** — real orders take time; prices move against you
4. **Fill quality varies** — partial fills, requotes, rejections
5. **Data feed issues** — missing bars, delays, disconnections
6. **Psychology** — fear/greed cause system overrides
7. **Regime changes** — the future will differ from the backtest period
8. **Overfitting** — some degree is inevitable despite walk-forward testing
9. **Black swans** — gold reacts violently to unpredictable geopolitical events
10. **Broker risks** — outages, margin changes, policy changes

## Configuration

All parameters are in `config/default.yaml`. Key sections:

- `general` — symbol, timeframe, live trading flag
- `data` — provider, date range, train/val/test split
- `costs` — spread, slippage, commission
- `risk` — per-trade risk, drawdown limits, kill switches
- `strategies` — per-strategy parameters
- `backtest` — walk-forward and Monte Carlo settings
- `paper_trade` — paper trading configuration
- `deployment_gate` — thresholds for go-live decision

## License

For educational and personal use. No warranty. Trade at your own risk.
