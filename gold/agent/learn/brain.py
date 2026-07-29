"""
Learning Brain v2 — Elite Quantitative Gold Trading Intelligence.

Orchestrates ALL systems:
1. Market Observer (multi-TF gold analysis)
2. Market Universe (cross-market: DXY, oil, silver, VIX, bonds, BTC, copper)
3. Prediction Engine (directional predictions with tracking)
4. Correlation Finder (indicator/session/timeframe correlations)
5. Strategy Lab (invent & mentally test strategies)
6. Cross-Market Strategies (DXY divergence, silver lead, VIX spike, etc.)
7. Trade Journal (SQLite full-context logging)
8. Strategy Lifecycle (hypothesis→backtest→paper→live→retire)
9. OANDA Tick Streamer (real-time ticks in background)

Discipline: max 3-4 trades/day, min 71% confidence, min 1.8 R:R.
"""
import json
import time
import threading
from datetime import datetime
from pathlib import Path

import click

from learn.market_observer import MarketObserver
from learn.prediction_engine import PredictionEngine
from learn.correlation_finder import CorrelationFinder
from learn.strategy_lab import StrategyLab
from learn.journal import TradeJournal
from learn.lifecycle import StrategyLifecycle
from learn.risk_manager import RiskManager, RiskConfig
from learn.mental_tracker import MentalTradeTracker, MentalTrade
from learn.session_journal import SessionJournal
from learn.market_thinker import MarketThinker
from learn.self_optimizer import SelfOptimizer
from learn.strategy_evolver import StrategyEvolver
from learn.paper_trader import PaperTrader
from learn.daily_report import DailyReportGenerator
from logs.logger import get_logger

log = get_logger("learn.brain")


class LearningBrain:
    """The central intelligence — thinks like a macro hedge fund manager."""

    def __init__(self, cfg: dict, data_dir: str = "./learn/data"):
        self.cfg = cfg
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.observer = MarketObserver(data_dir)
        self.predictor = PredictionEngine(data_dir)
        self.correlator = CorrelationFinder(data_dir)
        self.lab = StrategyLab(data_dir)

        # New v2 components
        self.journal = TradeJournal(data_dir)
        self.lifecycle = StrategyLifecycle(self.journal, data_dir)
        self.market_universe = None  # lazy init (heavy import)
        self.cross_market_strategies = []

        # Risk manager (YVPH discipline)
        risk_cfg = RiskConfig(
            initial_capital=cfg.get("capital", {}).get("initial", 1000.0),
            max_risk_pct=cfg.get("capital", {}).get("risk_pct", 0.02),
            min_rr_ratio=cfg.get("discipline", {}).get("min_rr_ratio", 2.5),
            max_trades_per_day=cfg.get("discipline", {}).get("max_trades_per_day", 4),
            max_loss_streak=cfg.get("discipline", {}).get("max_loss_streak", 3),
            max_daily_loss=cfg.get("capital", {}).get("max_daily_loss", 60.0),
            min_confidence=cfg.get("discipline", {}).get("min_confidence_to_trade", 0.71),
        )
        self.risk_mgr = RiskManager(risk_cfg, data_dir)
        self.mental_tracker = MentalTradeTracker(data_dir)
        self.session_journal = SessionJournal(data_dir)
        self.thinker = MarketThinker(data_dir)
        self.optimizer = SelfOptimizer(data_dir)
        self.evolver = StrategyEvolver(data_dir)

        # ICT concepts
        from intelligence.ict_concepts import ICTAnalyzer
        self.ict = ICTAnalyzer()

        # Unified brain — ONE decision maker
        from learn.unified_brain import UnifiedBrain
        self.unified = UnifiedBrain()

        # Paper trader (real execution simulation)
        self.paper_trader = PaperTrader(
            data_dir=data_dir,
            initial_equity=risk_cfg.initial_capital,
            spread=cfg.get("costs", {}).get("spread_points", 0.30),
            slippage=cfg.get("costs", {}).get("slippage_points", 0.10),
        )

        # State
        self.mental_entry_prices = {}
        self.mental_bars_since = {}
        self.winning_signals = []
        self.losing_signals = []
        self.cycle_count = 0
        self.trades_today = 0
        self.today_date = None

        # New intelligence sources
        self.cot_parser = None
        self.economic_calendar = None
        self.seasonal_analyzer = None
        self.order_flow = None

        # Daily report
        self.daily_reporter = DailyReportGenerator(self)
        self.last_daily_report_date = None

        # Tick streamer
        self.tick_streamer = None
        self.tick_thread = None

    def _init_market_universe(self):
        """Lazy-init the market universe (avoids slow import at startup)."""
        if self.market_universe is None:
            try:
                from intelligence.market_universe import MarketUniverse
                self.market_universe = MarketUniverse(str(self.data_dir))
                log.info("market_universe_initialized")
            except Exception as e:
                log.warning("market_universe_init_failed", error=str(e))

    def _init_cross_market_strategies(self):
        """Initialize cross-market strategies."""
        if self.cross_market_strategies:
            return
        try:
            from strategies.cross_market import (
                DXYDivergenceStrategy, SilverLeadStrategy, VIXSpikeStrategy,
                CopperGoldRatioStrategy, OilGoldInflationStrategy,
                BTCGoldFlowStrategy, RealYieldStrategy,
                AsianSessionSetupStrategy, MondayGapStrategy,
                RoundNumberStrategy, IndiaSeasonalStrategy,
                MultiMarketMomentumStrategy,
            )
            strategy_classes = [
                DXYDivergenceStrategy, SilverLeadStrategy, VIXSpikeStrategy,
                CopperGoldRatioStrategy, OilGoldInflationStrategy,
                BTCGoldFlowStrategy, RealYieldStrategy,
                AsianSessionSetupStrategy, MondayGapStrategy,
                RoundNumberStrategy, IndiaSeasonalStrategy,
                MultiMarketMomentumStrategy,
            ]
            for cls in strategy_classes:
                try:
                    strat = cls(self.cfg)
                    self.cross_market_strategies.append(strat)
                except Exception as e:
                    log.warning("cross_market_strategy_init_failed",
                                strategy=cls.__name__, error=str(e))
            log.info("cross_market_strategies_loaded",
                     count=len(self.cross_market_strategies))
        except ImportError as e:
            log.warning("cross_market_strategies_import_failed", error=str(e))

    def start(self, interval: int = 300, continuous: bool = True):
        """Start the full learning loop."""
        symbol = self.cfg["data"].get("tradingview_symbol", "XAUUSD")
        exchange = self.cfg["data"].get("tradingview_exchange", "OANDA")

        disc = self.cfg.get("discipline", {})
        max_trades = disc.get("max_trades_per_day", 4)
        min_conf = disc.get("min_confidence_to_trade", 0.71)

        click.echo("=" * 75)
        click.echo("GOLD TRADING BOT — ELITE DEEP LEARNING MODE v2")
        click.echo("=" * 75)
        click.echo("")
        click.echo("  Systems Active:")
        click.echo("    [1] Multi-TF Gold Observer    (5M, 15M, 1H, 4H, 1D)")
        click.echo("    [2] Market Universe           (DXY, oil, silver, VIX, bonds, BTC, copper)")
        click.echo("    [3] Prediction Engine          (predict, track, learn)")
        click.echo("    [4] Cross-Market Correlations  (lead-lag, regime, causality)")
        click.echo("    [5] Strategy Lab               (invent & mentally test)")
        click.echo("    [6] Cross-Market Strategies    (12 macro strategies)")
        click.echo("    [7] Trade Journal              (SQLite full-context logging)")
        click.echo("    [8] Strategy Lifecycle          (hypothesis→backtest→paper→live)")
        click.echo("    [9] OANDA Tick Stream           (real-time background)")
        click.echo("   [10] YVPH Risk Manager           (position sizing + discipline)")
        click.echo("")
        rm = self.risk_mgr
        click.echo(f"  Capital: ${rm.balance:.2f}  |  Risk/trade: "
                   f"${rm.balance * rm.cfg.max_risk_pct:.2f} ({rm.cfg.max_risk_pct:.0%})  |  "
                   f"R:R min: 1:{rm.cfg.min_rr_ratio:.1f}")
        click.echo(f"  Max trades: {rm.cfg.max_trades_per_day}/day  |  "
                   f"Max loss streak: {rm.cfg.max_loss_streak}  |  "
                   f"Daily loss cap: ${rm.cfg.max_daily_loss:.0f}")
        click.echo(f"  Discipline: max {max_trades} trades/day, "
                   f"min {min_conf:.0%} confidence, min 1.8 R:R")
        click.echo("  NO MONEY AT RISK. Bot is THINKING, not trading.")
        click.echo("=" * 75)

        # Initialize heavy components
        click.echo("\n  Initializing market universe...")
        self._init_market_universe()
        click.echo("  Initializing cross-market strategies...")
        self._init_cross_market_strategies()
        click.echo("  Initializing new intelligence sources...")
        self._init_intelligence_sources()

        # Start OANDA tick streaming
        self._start_tick_streamer()

        while True:
            try:
                self.cycle_count += 1
                self._check_daily_reset()
                cycle_start = time.time()

                click.echo(f"\n{'━' * 75}")
                click.echo(f"  LEARNING CYCLE #{self.cycle_count}  |  "
                           f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
                           f"Trades today: {self.trades_today}/{max_trades}")
                click.echo(f"{'━' * 75}")

                # ── Step 1: Fetch gold data (all TFs) ──
                click.echo("\n  [1/8] Fetching gold — all timeframes...")
                all_data = self.observer.fetch_all_timeframes(
                    symbol=symbol, exchange=exchange)
                fetched = {tf: len(df) for tf, df in all_data.items()
                           if df is not None}
                click.echo(f"         Gold TFs: {fetched}")

                # ── Step 2: Fetch cross-market data ──
                market_snapshot = {}
                market_history = {}
                if self.market_universe:
                    click.echo("\n  [2/8] Fetching cross-market universe...")
                    try:
                        market_history = self.market_universe.fetch_all(
                            period="6mo", interval="1d")
                        market_snapshot = self.market_universe.get_snapshot()
                        click.echo(f"         Markets: {list(market_snapshot.keys())[:10]}...")

                        # Composite scores
                        gold_pressure_result = self.market_universe.compute_gold_pressure_score()
                        risk_sentiment_result = self.market_universe.compute_risk_sentiment()
                        regime = self.market_universe.get_regime()

                        # Extract numeric scores from dicts
                        gold_pressure = gold_pressure_result.get("score", 0) if isinstance(gold_pressure_result, dict) else gold_pressure_result
                        risk_sentiment = risk_sentiment_result.get("score", 0) if isinstance(risk_sentiment_result, dict) else risk_sentiment_result

                        self._last_regime = regime
                        click.echo(f"         Gold Pressure: {gold_pressure:+.1f}/100  |  "
                                   f"Risk Sentiment: {risk_sentiment:+.2f}  |  "
                                   f"Regime: {regime}")

                        # Market narrative — explain WHY gold is moving
                        try:
                            market_changes = {}
                            for name, df in market_history.items():
                                if df is not None and len(df) >= 2:
                                    last = float(df["Close"].iloc[-1])
                                    prev = float(df["Close"].iloc[-2])
                                    if prev > 0:
                                        market_changes[name] = round((last - prev) / prev * 100, 2)
                            self._last_market_changes = market_changes
                            gold_chg = market_changes.get("gold", 0)
                            narrative = self.thinker.explain_price_move(
                                gold_chg, market_changes, regime, gold_pressure)
                            click.echo(f"         NARRATIVE: {narrative}")
                        except Exception as e:
                            click.echo(f"         Narrative error: {e}")

                    except Exception as e:
                        click.echo(f"         Cross-market fetch error: {e}")
                        gold_pressure, risk_sentiment, regime = 0, 0, "unknown"
                else:
                    click.echo("\n  [2/8] Cross-market: SKIPPED (not initialized)")
                    gold_pressure, risk_sentiment, regime = 0, 0, "unknown"

                # ── Step 3: Multi-TF observation ──
                click.echo("\n  [3/8] Multi-TF gold analysis...")
                report = self.observer.generate_multi_tf_report(all_data)
                click.echo(report)

                obs_result = (self.observer.observations[-1]
                              if self.observer.observations else {})
                cross_tf = obs_result.get("cross_tf", {})

                # ── Step 4: Cross-market correlations ──
                if self.market_universe and market_history:
                    click.echo("\n  [4/8] Cross-market correlation analysis...")
                    try:
                        corr_matrix = self.market_universe.compute_correlation_matrix()
                        lead_lag = self.market_universe.find_lead_lag()
                        breakdowns = self.market_universe.detect_correlation_breakdown()

                        # Show RMT diagnostics
                        rmt = getattr(self.market_universe, '_rmt_diagnostics', None)
                        if rmt:
                            click.echo(f"         RMT Cleaning: q={rmt['q']:.3f} "
                                       f"MP band=[{rmt['mp_lower']:.2f}, {rmt['mp_upper']:.2f}]  "
                                       f"Signal: {rmt['n_signal']}/{rmt['N']} eigenvalues  "
                                       f"Noise: {rmt['n_noise']}")

                        if corr_matrix is not None and len(corr_matrix) > 0:
                            click.echo("         Key correlations with gold (RMT-cleaned):")
                            for col in corr_matrix.columns[:8]:
                                if col != "gold" and col in corr_matrix.index:
                                    val = corr_matrix.loc["gold", col] if "gold" in corr_matrix.index else 0
                                    click.echo(f"           {col:<15s}: {val:+.3f}")

                        if lead_lag:
                            click.echo("         Lead-lag relationships:")
                            for asset, info in list(lead_lag.items())[:5]:
                                if isinstance(info, dict):
                                    click.echo(f"           {asset}: lag={info.get('best_lag', '?')} "
                                               f"corr={info.get('correlation', 0):.3f}")

                        if breakdowns:
                            click.echo(f"         CORRELATION BREAKDOWNS: {len(breakdowns)}")
                            for b in breakdowns[:3]:
                                click.echo(f"           {b}")

                        self.market_universe.save_state()
                    except Exception as e:
                        click.echo(f"         Correlation analysis error: {e}")
                else:
                    click.echo("\n  [4/8] Cross-market correlations: SKIPPED")

                # ── Step 4b: New Intelligence Sources ──
                click.echo(f"\n  [4b/10] Intelligence sources...")
                self._run_intelligence_sources(market_snapshot)

                # ── Step 4c: ICT Analysis ──
                try:
                    ict_result = self.ict.analyze(
                        df_5m=all_data.get("5m"),
                        df_15m=all_data.get("15m"),
                        df_1h=all_data.get("1h"),
                        df_4h=all_data.get("4h"),
                        df_1d=all_data.get("1d"),
                    )
                    kz = ict_result.get("kill_zone")
                    conf = ict_result.get("confluence", {})
                    nwog = ict_result.get("nwog")

                    click.echo(f"\n  [4c/10] ICT Smart Money Concepts...")
                    click.echo(f"         Kill Zone: {kz['name'] if kz else 'OUTSIDE kill zone'}")
                    if nwog:
                        click.echo(f"         NWOG: ${nwog['low']:.2f}-${nwog['high']:.2f} "
                                   f"(gap ${nwog['gap_size']:.2f} {nwog['direction']})")
                    if conf.get("score", 0) > 0:
                        click.echo(f"         ICT Confluence: {conf['direction']} "
                                   f"(score {conf['score']}, {conf['concepts_aligned']} concepts)")
                        for r in conf.get("reasons", [])[:3]:
                            click.echo(f"           -> {r}")

                    sweeps = ict_result.get("sweeps", [])
                    if sweeps:
                        for s in sweeps:
                            click.echo(f"         LIQUIDITY SWEEP: {s['description']}")

                    mss = ict_result.get("mss")
                    if mss:
                        click.echo(f"         MSS: {mss['description']}")

                    ote = ict_result.get("ote_1h")
                    if ote and ote.get("in_zone"):
                        click.echo(f"         OTE: IN ZONE — {ote['description']}")

                    self._last_ict = ict_result
                except Exception as e:
                    click.echo(f"         ICT analysis error: {e}")
                    self._last_ict = {}

                # ── Step 5: Resolve open trades (mental + paper) ──
                current_price = float(all_data.get("1h", all_data.get("5m", next(iter(all_data.values()))))["close"].iloc[-1])
                click.echo(f"\n  [5/12] Trade Resolution (price=${current_price:.2f})...")
                self._resolve_mental_trades(current_price, market_snapshot,
                                            gold_pressure, risk_sentiment, regime)
                # Paper trade resolution
                paper_resolved = self.paper_trader.check_outcomes(current_price)
                for pt in paper_resolved:
                    symbol = {True: "+", False: "x"}.get(pt.outcome == "WON", "~")
                    click.echo(f"         [{symbol}] PAPER {pt.strategy} {pt.direction}: "
                               f"${pt.fill_price:.2f} -> ${pt.exit_price:.2f}  "
                               f"P&L: ${pt.pnl_dollars:+.2f} [{pt.outcome}]")

                # ── Step 6: UNIFIED BRAIN DECISION ──
                click.echo(f"\n  [6/12] Unified Brain — analyzing all data...")
                self._run_unified_decision(
                    all_data, market_snapshot, market_history,
                    gold_pressure, risk_sentiment, regime,
                    current_price)

                # ── Step 7: Prediction Engine (track existing) ──
                click.echo(f"\n  [7/12] Prediction tracking...")
                self._run_predictions(all_data, cross_tf, market_snapshot,
                                      gold_pressure, regime)

                # ── Step 8: Strategy Lab ──
                click.echo(f"\n  [8/12] Strategy Lab...")
                self._run_strategy_lab(all_data)

                # ── Step 9: Missed Opportunities (with ANSWERS) ──
                click.echo(f"\n  [9/12] Missed Opportunity Check...")
                missed = self.paper_trader.detect_missed_opportunities(
                    all_data, [],
                    current_price,
                    open_mental_trades=self.mental_tracker.get_open_trades(),
                    regime=regime,
                    market_changes=getattr(self, '_last_market_changes', None))
                for m in missed:
                    click.echo(f"         MISSED: {m['question']}")
                    for r in m.get("reasons_missed", [])[:3]:
                        click.echo(f"           WHY: {r}")
                    for a in m.get("action_items", [])[:2]:
                        click.echo(f"           FIX: {a}")

                # ── Step 10: Strategy Invention ──
                if self.cycle_count % 12 == 0 and self.cycle_count > 0:
                    click.echo(f"\n  [10/12] Strategy Invention...")
                    try:
                        all_resolved = (self.mental_tracker.get_resolved_trades() +
                                        self.paper_trader.closed_trades)
                        winners = [t for t in all_resolved
                                   if t.get("outcome") in ("WON", "CORRECT")]
                        losers = [t for t in all_resolved
                                  if t.get("outcome") in ("LOST", "WRONG")]
                        invented = self.paper_trader.invent_strategy(
                            winners, losers, market_snapshot)
                        if invented:
                            click.echo(f"         NEW HYPOTHESIS: {invented['name']}")
                            click.echo(f"           {invented['reasoning']}")
                    except Exception as e:
                        click.echo(f"         Invention error: {e}")

                # ── Step 11: Session Journal ──
                self._check_session_journal()

                # ── Step 12: Progress + Save ──
                progress = self.paper_trader.get_progress()
                click.echo(f"\n  [12/12] Paper Trading Progress: "
                           f"{progress['total_trades']}/{progress['target_trades']} trades  "
                           f"WR={progress['win_rate']:.0%} (target {progress['target_win_rate']:.0%})  "
                           f"Equity=${progress['equity']:.2f}")
                if progress["ready_for_real"]:
                    click.echo(f"         *** MILESTONE: READY FOR REAL MONEY ***")
                self._save_all()

                # Tick streamer status
                if self.tick_streamer:
                    s = self.tick_streamer.analyzer.get_summary()
                    click.echo(f"\n  OANDA Ticks: {s.get('total_ticks', 0)} received, "
                               f"{s.get('spikes_detected', 0)} spikes")

                # Daily report at 10 PM EST (22:00 EST = 02:00 UTC next day, or ~local 10PM)
                self._check_daily_report()

                elapsed = time.time() - cycle_start
                click.echo(f"\n  Cycle #{self.cycle_count} done in {elapsed:.1f}s  |  "
                           f"Trades today: {self.trades_today}")

                # Invent new strategies periodically
                if self.cycle_count % 5 == 0 and len(self.winning_signals) >= 5:
                    self.lab.invent_from_observations(
                        self.winning_signals, self.losing_signals)

                # Weekly report on Sundays
                if datetime.now().weekday() == 6 and self.cycle_count % 20 == 0:
                    click.echo(self.journal.generate_weekly_report())

            except Exception as e:
                click.echo(f"\n  ERROR in learning cycle: {e}")
                log.error("cycle_error", error=str(e), cycle=self.cycle_count)
                import traceback
                traceback.print_exc()

            if not continuous:
                break

            click.echo(f"\n  Next cycle in {interval}s... (Ctrl+C to stop)")
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                click.echo("\n\n  Stopping...")
                self._save_all()
                self._stop_tick_streamer()
                self._print_session_summary()
                break

    def _init_intelligence_sources(self):
        """Initialize COT, economic calendar, seasonal analyzer, order flow."""
        import os

        try:
            from intelligence.cot_report import COTParser
            self.cot_parser = COTParser(str(self.data_dir))
            log.info("cot_parser_initialized")
        except Exception as e:
            log.warning("cot_init_failed", error=str(e))

        try:
            from intelligence.economic_calendar import EconomicCalendar
            self.economic_calendar = EconomicCalendar(str(self.data_dir))
            log.info("economic_calendar_initialized")
        except Exception as e:
            log.warning("econ_cal_init_failed", error=str(e))

        try:
            from intelligence.seasonal import SeasonalAnalyzer
            self.seasonal_analyzer = SeasonalAnalyzer(str(self.data_dir))
            log.info("seasonal_analyzer_initialized")
        except Exception as e:
            log.warning("seasonal_init_failed", error=str(e))

        try:
            token = self.cfg["data"].get("oanda_token") or os.getenv("OANDA_TOKEN")
            account_id = self.cfg["data"].get("oanda_account_id") or os.getenv("OANDA_ACCOUNT_ID")
            if token and account_id:
                from intelligence.order_flow import OrderFlowAnalyzer
                self.order_flow = OrderFlowAnalyzer(
                    token=token, account_id=account_id,
                    practice=True, data_dir=str(self.data_dir))
                log.info("order_flow_initialized")
            else:
                log.warning("order_flow_skipped", reason="no_oanda_credentials")
        except Exception as e:
            log.warning("order_flow_init_failed", error=str(e))

        sources = sum(1 for x in [self.cot_parser, self.economic_calendar,
                                   self.seasonal_analyzer, self.order_flow] if x)
        click.echo(f"    {sources}/4 intelligence sources ready: "
                   f"COT={'OK' if self.cot_parser else 'SKIP'} | "
                   f"Calendar={'OK' if self.economic_calendar else 'SKIP'} | "
                   f"Seasonal={'OK' if self.seasonal_analyzer else 'SKIP'} | "
                   f"OrderFlow={'OK' if self.order_flow else 'SKIP'}")

    def _check_daily_reset(self):
        """Reset daily trade count at midnight."""
        today = datetime.now().date()
        if self.today_date != today:
            if self.today_date is not None:
                # Generate daily summary
                self.journal.generate_daily_summary(str(self.today_date), {
                    "trades_taken": self.trades_today,
                    "predictions_made": len(self.predictor.active_predictions),
                    "predictions_resolved": len(self.predictor.resolved_predictions),
                })
                self.lifecycle.daily_reset()
            self.today_date = today
            self.trades_today = 0

    def _run_intelligence_sources(self, market_snapshot):
        """Run COT, economic calendar, seasonal, and order flow analysis."""

        # COT Report (weekly — only fetch if stale)
        if self.cot_parser:
            try:
                positioning = self.cot_parser.get_gold_positioning()
                if positioning:
                    spec_net = positioning.get("non_commercial_net", 0)
                    comm_net = positioning.get("commercial_net", 0)
                    click.echo(f"         COT: Spec net={spec_net:+,d}  "
                               f"Comm net={comm_net:+,d}  "
                               f"Date={positioning.get('report_date', '?')}")
                    extremes = self.cot_parser.detect_extremes()
                    if extremes:
                        for ex in extremes:
                            click.echo(f"         *** COT SIGNAL: {ex} ***")
                else:
                    click.echo(f"         COT: No data yet (fetches weekly from CFTC)")
            except Exception as e:
                click.echo(f"         COT: Error — {e}")

        # Economic Calendar
        if self.economic_calendar:
            try:
                risk_score = self.economic_calendar.get_event_risk_score()
                events = self.economic_calendar.get_upcoming_events(days_ahead=7)
                reduce = self.economic_calendar.should_reduce_risk()
                click.echo(f"         Calendar: Risk={risk_score}/100  "
                           f"Events next 7d={len(events)}  "
                           f"Reduce risk={'YES' if reduce else 'no'}")
                for ev in events[:3]:
                    click.echo(f"           {ev['date']} {ev['event']} "
                               f"({ev['impact']}) — {ev.get('days_away', '?')}d away")
            except Exception as e:
                click.echo(f"         Calendar: Error — {e}")

        # Seasonal Analysis
        if self.seasonal_analyzer:
            try:
                seasonal_score = self.seasonal_analyzer.get_current_seasonal_score()
                combined = self.seasonal_analyzer.get_combined_seasonal_signal()
                click.echo(f"         Seasonal: score={seasonal_score:+.2f}  "
                           f"combined={combined.get('combined_score', 0):+.2f}  "
                           f"bias={combined.get('bias', 'neutral')}")
            except Exception as e:
                click.echo(f"         Seasonal: Error — {e}")

        # Order Flow (OANDA)
        if self.order_flow:
            try:
                flow = self.order_flow.get_flow_summary()
                if flow:
                    click.echo(f"         OrderFlow: pressure={flow.get('order_pressure_score', 0):+.2f}  "
                               f"positioning={flow.get('positioning_score', 0):+.2f}  "
                               f"extreme={'YES' if flow.get('extreme_alert') else 'no'}  "
                               f"signal={flow.get('recommendation', 'NEUTRAL')}")
                    if flow.get("extreme_alert"):
                        click.echo(f"         *** EXTREME POSITIONING: "
                                   f"{flow.get('recommendation', '')} ***")
                    self.order_flow.save_snapshot()
                else:
                    click.echo(f"         OrderFlow: No data")
            except Exception as e:
                click.echo(f"         OrderFlow: Error — {e}")

    def _resolve_mental_trades(self, current_price, market_snapshot,
                               gold_pressure, risk_sentiment, regime):
        """Check all open mental trades against current price."""
        open_count = len(self.mental_tracker.open_trades)
        if open_count == 0:
            click.echo(f"         No open mental trades")
            return

        click.echo(f"         Checking {open_count} open mental trades...")
        resolved = self.mental_tracker.check_outcomes(current_price)

        for trade in resolved:
            outcome_symbol = {"WON": "+", "LOST": "x", "EXPIRED": "~"}.get(trade.outcome, "?")
            click.echo(f"         [{outcome_symbol}] {trade.strategy} {trade.direction}: "
                       f"${trade.entry_price:.2f} -> ${trade.exit_price:.2f}  "
                       f"P&L: ${trade.pnl_dollars:+.2f} ({trade.pnl_points:+.1f} pts)  "
                       f"[{trade.outcome} after {trade.bars_held} bars]")
            click.echo(f"             MFE: +{trade.max_favorable:.1f} / "
                       f"MAE: {trade.max_adverse:.1f}")

            # Feed result to risk manager
            self.risk_mgr.record_outcome(
                pnl_dollars=trade.pnl_dollars,
                won=(trade.outcome == "WON"),
                strategy=trade.strategy,
                direction=trade.direction,
                entry=trade.entry_price,
                exit_price=trade.exit_price,
            )

            # Feed to thinker — learn strategy-regime performance
            trade_regime = trade.market_context.get("regime", regime) if trade.market_context else regime
            self.thinker.record_strategy_outcome(
                strategy=trade.strategy,
                regime=trade_regime,
                outcome=trade.outcome,
                pnl=trade.pnl_dollars,
                market_context=trade.market_context,
            )

            # Self-optimizer — deep analysis + parameter adjustment
            from dataclasses import asdict
            trade_dict = asdict(trade) if hasattr(trade, '__dataclass_fields__') else trade.__dict__
            calendar_events = []
            if self.economic_calendar:
                try:
                    calendar_events = self.economic_calendar.get_upcoming_events(days_ahead=2)
                except Exception:
                    pass
            diagnosis = self.optimizer.learn_from_trade(trade_dict, calendar_events)
            if diagnosis.get("why"):
                click.echo(f"             DIAGNOSIS: {' | '.join(diagnosis['why'][:2])}")
            if diagnosis.get("adjustments"):
                click.echo(f"             ADJUST: {' | '.join(diagnosis['adjustments'][:2])}")
            if diagnosis.get("event_impact"):
                click.echo(f"             EVENT: {diagnosis['event_impact']}")
            if diagnosis.get("lesson"):
                click.echo(f"             LESSON: {diagnosis['lesson']}")

            # Log to trade journal with full context
            self.journal.log_prediction({
                "trade_id": trade.trade_id + "_resolved",
                "strategy": trade.strategy,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "pnl": trade.pnl_dollars,
                "outcome": trade.outcome,
                "confidence": trade.confidence,
                "reasons": trade.reasons,
                "timeframe": trade.timeframe,
                "bars_held": trade.bars_held,
                "max_favorable": trade.max_favorable,
                "max_adverse": trade.max_adverse,
                "stop_price": trade.stop_price,
                "target_price": trade.target_price,
                "rr_ratio": trade.rr_ratio,
                "risk_dollars": trade.risk_dollars,
            }, {
                "gold_pressure": gold_pressure,
                "risk_sentiment": risk_sentiment,
                "regime": regime,
                "market_snapshot": market_snapshot,
                "balance_after": self.risk_mgr.balance,
            })

        # Show remaining open trades
        remaining = self.mental_tracker.get_open_trades()
        if remaining:
            click.echo(f"         Still open: {len(remaining)} trades")
            for t in remaining:
                click.echo(f"           {t['strategy']} {t['direction']} "
                           f"@ ${t['entry_price']:.2f} "
                           f"(bar {t['bars_elapsed']}/{t['horizon_bars']})")

        # Show running stats
        stats = self.mental_tracker.get_stats()
        if stats.get("total", 0) > 0:
            click.echo(f"         Mental record: {stats['wins']}W-{stats['losses']}L-"
                       f"{stats['expired']}E  "
                       f"WR={stats['win_rate']:.0%}  "
                       f"P&L=${stats['total_pnl']:+.2f}")

    def _check_session_journal(self):
        """Write session journal if a session just ended."""
        for session in ["asian", "london", "new_york"]:
            if self.session_journal.should_write(session):
                try:
                    click.echo(f"\n  [9/10] Writing {session.upper()} session journal...")
                    knowledge = {}
                    try:
                        import json
                        kb_path = self.data_dir / "knowledge_base.json"
                        if kb_path.exists():
                            with open(kb_path) as f:
                                knowledge = json.load(f)
                    except Exception:
                        pass

                    self.session_journal.write_session_summary(
                        session=session,
                        trades=self.mental_tracker.get_resolved_trades(),
                        open_trades=self.mental_tracker.get_open_trades(),
                        risk_status=self.risk_mgr.get_status(),
                        tracker_stats=self.mental_tracker.get_stats(),
                        knowledge=knowledge,
                    )

                    # Write thinker's analytical analysis
                    thinker_analysis = self.thinker.generate_session_analysis(
                        session=session,
                        resolved_trades=self.mental_tracker.get_resolved_trades(),
                        market_changes={},  # filled from cache if available
                        regime=getattr(self, '_last_regime', 'unknown'),
                        gold_change_pct=0,
                        knowledge=knowledge,
                    )
                    journal_path = self.session_journal.journal_dir / f"{str(datetime.now().date())}.md"
                    with open(journal_path, "a") as f:
                        f.write(f"\n{thinker_analysis}\n")

                    # Write daily summary at end of NY session
                    if session == "new_york":
                        # Generate and write daily lessons
                        lessons = self.thinker.generate_daily_lessons(
                            resolved_trades=self.mental_tracker.get_resolved_trades(),
                            regime=getattr(self, '_last_regime', 'unknown'),
                            balance=self.risk_mgr.balance,
                        )
                        self.session_journal.write_daily_summary(
                            risk_status=self.risk_mgr.get_status(),
                            tracker_stats=self.mental_tracker.get_stats(),
                            knowledge=knowledge,
                        )
                        # Append thinker lessons
                        if lessons:
                            with open(journal_path, "a") as f:
                                f.write("\n### Thinker Lessons\n\n")
                                for l in lessons:
                                    f.write(f"- **{l['type']}**: {l['lesson']}\n")
                                    f.write(f"  Evidence: {l['evidence']}\n")
                                f.write("\n")

                        # Strategy Evolution — analyze and rewrite losing strategies
                        try:
                            import json as _json
                            opt_state = _json.load(open(self.data_dir / "optimizer_state.json"))
                            evo_records = self.evolver.daily_review(
                                opt_state, dict(self.predictor.pattern_accuracy))
                            if evo_records:
                                click.echo(f"         EVOLVED {len(evo_records)} strategies")
                                with open(journal_path, "a") as f:
                                    f.write(self.evolver.get_evolution_report())
                                for rec in evo_records:
                                    click.echo(f"           {rec['strategy']}: {rec['action']}")
                                    for d in rec.get('diagnosis', [])[:2]:
                                        click.echo(f"             {d}")
                                    for h in rec.get('hypothesis', [])[:2]:
                                        click.echo(f"             FIX: {h}")
                        except Exception as e:
                            click.echo(f"         Evolution error: {e}")
                except Exception as e:
                    click.echo(f"         Journal error: {e}")

    def _run_unified_decision(self, all_data, market_snapshot, market_history,
                              gold_pressure, risk_sentiment, regime, current_price):
        """ONE brain looks at EVERYTHING, makes ONE call."""
        try:
            # ─── PATIENCE: check existing positions first ───
            # Max 2 positions total (mental + paper combined)
            # If we have open positions, MANAGE them — don't open more
            MAX_OPEN_POSITIONS = 2
            open_mental = len(self.mental_tracker.open_trades)
            open_paper = len(self.paper_trader.open_trades)
            total_open = open_mental + open_paper

            # Also check OANDA directly
            oanda_open = 0
            if self.paper_trader.use_oanda:
                try:
                    oanda_trades = self.paper_trader.broker.get_open_trades()
                    oanda_open = len(oanda_trades)
                except Exception:
                    pass

            if total_open >= MAX_OPEN_POSITIONS or oanda_open >= MAX_OPEN_POSITIONS:
                click.echo(f"         PATIENCE: {total_open} mental + {oanda_open} OANDA positions open — "
                           f"managing existing trades, not opening new ones")
                # Show current positions
                for name, t in self.mental_tracker.open_trades.items():
                    unrealized = current_price - t.entry_price if t.direction == "LONG" else t.entry_price - current_price
                    click.echo(f"           {t.strategy} {t.direction} @ ${t.entry_price:.2f} "
                               f"P&L: {unrealized:+.1f} pts (bar {t.bars_elapsed}/{t.horizon_bars})")
                return

            # Session trade limit: max 5 trades per session
            _session = self.session_journal.get_current_session() if self.session_journal else "unknown"
            session_key = f"trades_{_session}_{datetime.now().strftime('%Y%m%d')}"
            if not hasattr(self, '_session_trade_counts'):
                self._session_trade_counts = {}
            session_trades = self._session_trade_counts.get(session_key, 0)
            if session_trades >= 5:
                click.echo(f"         SESSION LIMIT: {session_trades}/5 trades in {_session} session — waiting")
                return

            # Gather all intelligence
            _cot = None
            if self.cot_parser:
                try: _cot = self.cot_parser.get_gold_positioning()
                except: pass
            _flow = None
            if self.order_flow:
                try: _flow = self.order_flow.get_flow_summary()
                except: pass
            _cal = []
            if self.economic_calendar:
                try: _cal = self.economic_calendar.get_upcoming_events(days_ahead=2)
                except: pass
            _seasonal = None
            if self.seasonal_analyzer:
                try: _seasonal = self.seasonal_analyzer.get_combined_seasonal_signal()
                except: pass
            _patterns = {}
            try:
                import json
                kb = json.load(open(self.data_dir / "knowledge_base.json"))
                _patterns = kb.get("pattern_counts", {})
            except: pass

            ict_result = getattr(self, '_last_ict', {})

            # Get RMT correlations and lead-lag
            _correlations = {}
            _lead_lag = {}
            if self.market_universe:
                try:
                    import json as _j
                    mu_state = _j.load(open(self.data_dir / "market_universe_state.json"))
                    _correlations = mu_state.get("correlation_matrix", {}).get("gold", {})
                    _lead_lag = mu_state.get("lead_lag", {})
                except Exception:
                    pass

            # Get live news sentiment
            _news = None
            if not hasattr(self, '_news_fetcher'):
                try:
                    from intelligence.news_fetcher import NewsFetcher
                    self._news_fetcher = NewsFetcher(cache_seconds=300)
                except Exception:
                    self._news_fetcher = None
            if self._news_fetcher:
                try:
                    _news = self._news_fetcher.get_sentiment()
                    if _news and _news.get("headlines"):
                        click.echo(f"         NEWS: {_news['headlines'][:2]}")
                except Exception:
                    pass

            # THE DECISION — all data, one brain
            decision = self.unified.decide(
                all_data=all_data,
                market_changes=getattr(self, '_last_market_changes', None),
                regime=regime,
                gold_pressure=gold_pressure,
                ict_result=ict_result,
                cot_data=_cot,
                order_flow=_flow,
                calendar_events=_cal,
                seasonal=_seasonal,
                pattern_counts=_patterns,
                correlations=_correlations,
                lead_lag=_lead_lag,
                news_sentiment=_news,
            )

            # Display the decision
            if decision.direction == "FLAT":
                click.echo(f"         DECISION: FLAT — {decision.reasons[0] if decision.reasons else 'no clear signal'}")
                click.echo(f"         Votes: {decision.votes}")
                return

            dir_color = "LONG" if decision.direction == "LONG" else "SHORT"
            click.echo(f"         ╔══════════════════════════════════════════╗")
            click.echo(f"         ║  DECISION: {dir_color} ({decision.strength})  "
                       f"Score: {decision.score:+.1f}  Conf: {decision.confidence:.0%}  ║")
            click.echo(f"         ╚══════════════════════════════════════════╝")
            click.echo(f"         Entry: ${decision.entry_price:.2f}  "
                       f"Stop: ${decision.stop_price:.2f}  "
                       f"Target: ${decision.target_price:.2f}")
            click.echo(f"         Votes: {decision.votes}")
            for r in decision.reasons[:4]:
                click.echo(f"           FOR:     {r}")
            for a in decision.against[:2]:
                click.echo(f"           AGAINST: {a}")

            # Risk manager gate
            rr_ratio = abs(decision.target_price - decision.entry_price) / abs(decision.stop_price - decision.entry_price) if abs(decision.stop_price - decision.entry_price) > 0 else 0
            can_trade, reject = self.risk_mgr.can_trade(
                confidence=decision.confidence, rr_ratio=rr_ratio)
            if not can_trade:
                click.echo(f"         RISK BLOCKED: {reject}")
                return

            # Position sizing
            pos = self.risk_mgr.size_position(
                decision.entry_price, decision.stop_price, decision.direction)
            if not pos:
                click.echo(f"         Cannot size position")
                return

            click.echo(f"         Size: {pos.lot_size:.2f} lots  "
                       f"Risk: ${pos.risk_dollars:.2f}  R:R 1:{rr_ratio:.1f}")

            # Open mental trade
            if self.mental_tracker.can_open("unified"):
                from learn.mental_tracker import MentalTrade
                mt = MentalTrade(
                    trade_id=f"unified_{self.cycle_count}",
                    strategy="unified_brain",
                    direction=decision.direction,
                    entry_price=decision.entry_price,
                    stop_price=decision.stop_price,
                    target_price=decision.target_price,
                    timeframe="1h",
                    confidence=decision.confidence,
                    reasons=decision.reasons[:5],
                    rr_ratio=rr_ratio,
                    risk_dollars=pos.risk_dollars,
                    lot_size=pos.lot_size,
                    entry_time=datetime.now().isoformat(),
                    horizon_bars=20,
                    market_context={
                        "regime": regime,
                        "gold_pressure": gold_pressure,
                        "score": decision.score,
                        "votes": decision.votes,
                    },
                )
                self.mental_tracker.open_trade(mt)
                click.echo(f"         MENTAL TRADE OPENED: {decision.direction} @ ${decision.entry_price:.2f}")
                # Track session trade count
                self._session_trade_counts[session_key] = session_trades + 1

            # Paper trade on OANDA
            if self.paper_trader.can_open("unified"):
                _session = self.session_journal.get_current_session() if self.session_journal else "unknown"
                pt = self.paper_trader.execute_trade(
                    strategy="unified_brain",
                    direction=decision.direction,
                    current_price=decision.entry_price,
                    stop_price=decision.stop_price,
                    target_price=decision.target_price,
                    lots=pos.lot_size,
                    risk_dollars=pos.risk_dollars,
                    confidence=decision.confidence,
                    regime=regime,
                    session=_session,
                    reasons=decision.reasons[:5],
                )
                if pt:
                    click.echo(f"         PAPER TRADE ON OANDA: filled @ ${pt.fill_price:.2f}")
                else:
                    click.echo(f"         PAPER TRADE: could not execute")

        except Exception as e:
            click.echo(f"         Unified brain error: {e}")
            import traceback
            traceback.print_exc()

    def _run_predictions(self, all_data, cross_tf, market_snapshot,
                         gold_pressure, regime):
        """Run prediction engine — check old predictions, make new ones."""
        max_trades = self.cfg.get("discipline", {}).get("max_trades_per_day", 4)

        # Check old predictions
        for tf, df in all_data.items():
            if df is not None and len(df) > 2:
                resolved = self.predictor.check_outcomes(df, tf)
                if resolved:
                    for r in resolved:
                        click.echo(f"         RESOLVED [{r.id}]: {r.outcome} "
                                   f"({r.direction} ${r.entry_price:.2f}→"
                                   f"${r.exit_price:.2f} PnL=${r.pnl_points:+.2f})")

                        # Log to journal with full context
                        self.journal.log_prediction({
                            "trade_id": r.id,
                            "strategy": r.strategy_hypothesis or "prediction_engine",
                            "direction": r.direction,
                            "entry_price": r.entry_price,
                            "exit_price": r.exit_price,
                            "pnl": r.pnl_points,
                            "outcome": r.outcome,
                            "confidence": r.confidence,
                            "reasons": r.reasons,
                            "timeframe": r.timeframe,
                            "bars_held": r.bars_held,
                            "max_favorable": r.max_favorable,
                            "max_adverse": r.max_adverse,
                        }, {
                            "gold_pressure": gold_pressure,
                            "regime": regime,
                            "market_snapshot": market_snapshot,
                        })

                        # Self-optimizer: learn from prediction outcome
                        strat_name = r.strategy_hypothesis or "prediction_engine"
                        pred_trade = {
                            "strategy": strat_name,
                            "direction": r.direction,
                            "outcome": r.outcome,
                            "entry_price": r.entry_price,
                            "exit_price": r.exit_price,
                            "pnl_points": r.pnl_points,
                            "pnl_dollars": r.pnl_points,  # 1:1 for predictions
                            "max_favorable": r.max_favorable,
                            "max_adverse": r.max_adverse,
                            "bars_held": r.bars_held,
                            "target_price": r.target_price,
                            "stop_price": r.stop_price,
                            "reasons": r.reasons,
                            "market_context": {"regime": regime, "gold_pressure": gold_pressure},
                        }
                        calendar_events = []
                        if self.economic_calendar:
                            try:
                                calendar_events = self.economic_calendar.get_upcoming_events(days_ahead=2)
                            except Exception:
                                pass
                        _cot_pred = None
                        if self.cot_parser:
                            try: _cot_pred = self.cot_parser.get_gold_positioning()
                            except: pass
                        diagnosis = self.optimizer.learn_from_trade(
                            pred_trade, calendar_events,
                            market_changes=getattr(self, '_last_market_changes', None),
                            correlations=None,
                            cot_data=_cot_pred,
                            session=self.session_journal.get_current_session() if self.session_journal else None)
                        if diagnosis.get("why"):
                            click.echo(f"             WHY: {' | '.join(diagnosis['why'][:2])}")
                        if diagnosis.get("adjustments"):
                            click.echo(f"             FIX: {' | '.join(diagnosis['adjustments'][:2])}")
                        if diagnosis.get("lesson"):
                            click.echo(f"             LESSON: {diagnosis['lesson']}")

                        # Thinker: learn strategy-regime performance
                        self.thinker.record_strategy_outcome(
                            strategy=strat_name, regime=regime,
                            outcome=r.outcome, pnl=r.pnl_points)

                        # Feed signals for strategy invention
                        if df is not None and len(df) > 50:
                            signals = self.predictor._compute_signals(df)
                            if r.outcome == "CORRECT":
                                self.winning_signals.append(signals)
                            else:
                                self.losing_signals.append(signals)

        # Save after resolving so resolved data persists across restarts
        self.predictor.save_state()

        # Make new predictions (respect trade limit)
        if self.trades_today < max_trades:
            for tf, df in all_data.items():
                if df is not None and len(df) > 50:
                    preds = self.predictor.analyze_and_predict(
                        df, tf, cross_tf)
                    for p in preds:
                        # Evolver filter check — evolved rules from past failures
                        strat_hyp = p.strategy_hypothesis or "unknown"
                        allowed, conf_adj, evo_reasons = self.evolver.check_filters(
                            strat_hyp, p.direction, p.confidence,
                            tf, getattr(self, '_last_regime', 'unknown'))
                        if not allowed:
                            click.echo(f"         EVOLVED-BLOCK [{p.id}] {p.direction} "
                                       f"${p.entry_price:.2f} ({tf}) — "
                                       f"{evo_reasons[0] if evo_reasons else 'evolved filter'}")
                            # Remove from active since it was already added
                            self.predictor.active_predictions = [
                                ap for ap in self.predictor.active_predictions if ap.id != p.id]
                            continue
                        if conf_adj != 0:
                            p.confidence = max(0.0, min(1.0, p.confidence + conf_adj))

                        self.trades_today += 1
                        click.echo(f"         NEW [{p.id}] {p.direction} "
                                   f"${p.entry_price:.2f} ({tf}) "
                                   f"conf={p.confidence:.0%}")
                        for reason in p.reasons[:2]:
                            click.echo(f"             → {reason}")
                        for er in evo_reasons:
                            click.echo(f"             [{er}]")
                        if self.trades_today >= max_trades:
                            click.echo(f"         DAILY LIMIT REACHED "
                                       f"({max_trades} trades)")
                            break
                if self.trades_today >= max_trades:
                    break

        click.echo(f"\n{self.predictor.get_scoreboard()}")

    def _run_cross_market_strategies(self, all_data, market_snapshot,
                                      market_history, gold_pressure,
                                      risk_sentiment, regime):
        """Run cross-market strategies as mental tests."""
        if not self.cross_market_strategies:
            click.echo("         No cross-market strategies loaded")
            return

        # Get the 1H gold data for strategy evaluation
        gold_df = all_data.get("1h")
        if gold_df is None or len(gold_df) < 60:
            gold_df = all_data.get("4h")
            if gold_df is None or (hasattr(gold_df, 'empty') and gold_df.empty):
                gold_df = all_data.get("1d")
        if gold_df is None:
            return

        triggered_count = 0
        for strat in self.cross_market_strategies:
            try:
                # Set market data on strategy if it has the method
                if hasattr(strat, "set_market_data"):
                    strat.set_market_data(market_snapshot)
                if hasattr(strat, "set_market_history"):
                    strat.set_market_history(market_history)

                # Generate signals
                result_df = strat.generate_signals(gold_df)

                # Check if last bar has a signal
                if result_df is not None and len(result_df) > 0:
                    last_signal = int(result_df["signal"].iloc[-1])
                    if last_signal != 0:
                        triggered_count += 1
                        direction = "LONG" if last_signal == 1 else "SHORT"
                        confidence = float(result_df["confidence"].iloc[-1]) if "confidence" in result_df.columns else 0.5
                        price = float(gold_df["close"].iloc[-1])

                        reasons = []
                        if hasattr(strat, "reasons"):
                            reasons = strat.reasons
                        elif hasattr(strat, "last_reasons"):
                            reasons = strat.last_reasons

                        # Compute stop from ATR if available
                        stop_price = None
                        if "stop_price" in result_df.columns:
                            sp = result_df["stop_price"].iloc[-1]
                            if sp == sp:  # not NaN
                                stop_price = float(sp)
                        if stop_price is None:
                            # Fallback: use ATR-based stop
                            import numpy as np
                            atr_col = result_df.get("_atr")
                            if atr_col is not None and len(atr_col) > 0:
                                atr_val = float(atr_col.iloc[-1])
                            else:
                                atr_val = abs(float(gold_df["high"].iloc[-1]) - float(gold_df["low"].iloc[-1]))
                            if direction == "LONG":
                                stop_price = price - atr_val * 1.5
                            else:
                                stop_price = price + atr_val * 1.5

                        # Compute R:R
                        stop_dist = abs(price - stop_price)
                        target_dist = stop_dist * self.risk_mgr.cfg.min_rr_ratio
                        if direction == "LONG":
                            target_price = price + target_dist
                        else:
                            target_price = price - target_dist
                        rr_ratio = target_dist / stop_dist if stop_dist > 0 else 0

                        # Deduplication: skip if strategy already has an open trade
                        if not self.mental_tracker.can_open(strat.name):
                            continue

                        # TREND ALIGNMENT — don't fight the higher timeframe
                        # If 4H/1D says DOWN, don't go LONG (and vice versa)
                        try:
                            htf = all_data.get("4h") or all_data.get("1d")
                            if htf is not None and len(htf) > 20:
                                htf_close = float(htf["close"].iloc[-1])
                                htf_sma20 = float(htf["close"].iloc[-20:].mean())
                                htf_sma50 = float(htf["close"].iloc[-50:].mean()) if len(htf) > 50 else htf_sma20
                                htf_trend = "UP" if htf_close > htf_sma20 > htf_sma50 else \
                                            "DOWN" if htf_close < htf_sma20 < htf_sma50 else "SIDEWAYS"

                                if direction == "LONG" and htf_trend == "DOWN":
                                    click.echo(f"         {strat.name}: {direction} "
                                               f"@ ${price:.2f} — TREND BLOCK: "
                                               f"4H trend is DOWN (price < SMA20 < SMA50)")
                                    continue
                                elif direction == "SHORT" and htf_trend == "UP":
                                    click.echo(f"         {strat.name}: {direction} "
                                               f"@ ${price:.2f} — TREND BLOCK: "
                                               f"4H trend is UP (price > SMA20 > SMA50)")
                                    continue
                        except Exception:
                            pass

                        # ICT kill zone filter — prefer kill zones, don't require them
                        ict = getattr(self, '_last_ict', {})
                        ict_conf = ict.get("confluence", {})
                        if ict.get("in_kill_zone", False):
                            # In kill zone — boost if ICT agrees
                            if ict_conf.get("direction") == direction and ict_conf.get("score", 0) >= 3:
                                confidence = min(1.0, confidence + 0.15)
                            elif ict_conf.get("direction") and ict_conf["direction"] != direction and ict_conf["direction"] != "NEUTRAL":
                                confidence *= 0.8  # ICT disagrees — mild penalty
                        else:
                            # Outside kill zone — small penalty, not a blocker
                            confidence *= 0.9

                        # Thinker: adjust confidence based on regime performance
                        raw_confidence = confidence
                        confidence = self.thinker.get_adjusted_confidence(
                            strat.name, regime, confidence)

                        # Self-optimizer: pre-trade advice using ALL intelligence
                        _cot = None
                        if self.cot_parser:
                            try: _cot = self.cot_parser.get_gold_positioning()
                            except: pass
                        _cal = []
                        if self.economic_calendar:
                            try: _cal = self.economic_calendar.get_upcoming_events(days_ahead=2)
                            except: pass
                        _session = self.session_journal.get_current_session() if self.session_journal else None
                        _mkt_chg = getattr(self, '_last_market_changes', None)

                        advice = self.optimizer.get_pre_trade_advice(
                            strat.name, direction, regime,
                            market_changes=_mkt_chg,
                            cot_data=_cot,
                            session=_session,
                            calendar_events=_cal)
                        if not advice["take_trade"]:
                            click.echo(f"         {strat.name}: {direction} "
                                       f"@ ${price:.2f} — OPTIMIZER BLOCKED: "
                                       f"{advice['warnings']}")
                            continue
                        confidence += advice.get("confidence_modifier", 0)
                        confidence = max(0.0, min(1.0, confidence))

                        # Apply optimized stop/target (FROZEN until 50+ trades)
                        opt = self.optimizer.get_optimal_params(strat.name, direction)
                        if opt.get("sample_size", 0) >= 5:  # Unfrozen: 5 trade minimum for paper trading mandate
                            opt_target = opt["optimal_target"]
                            opt_stop = opt["optimal_stop"]
                            if direction == "LONG":
                                stop_price = price - opt_stop
                                target_price = price + opt_target
                            else:
                                stop_price = price + opt_stop
                                target_price = price - opt_target
                            stop_dist = opt_stop
                            rr_ratio = opt_target / opt_stop if opt_stop > 0 else 2.5
                            click.echo(f"           [OPTIMIZED from {opt['sample_size']} trades: "
                                       f"target={opt_target:.1f} stop={opt_stop:.1f}]")
                        if advice.get("warnings"):
                            for w in advice["warnings"]:
                                click.echo(f"           [{w}]")

                        # Risk manager gate check
                        can_trade, reject_reason = self.risk_mgr.can_trade(
                            confidence=confidence, rr_ratio=rr_ratio)

                        if not can_trade:
                            click.echo(f"         {strat.name}: {direction} "
                                       f"@ ${price:.2f} — BLOCKED: {reject_reason}")
                            continue

                        # Position sizing
                        pos = self.risk_mgr.size_position(price, stop_price, direction)

                        click.echo(f"         {strat.name}: {direction} "
                                   f"@ ${price:.2f} conf={confidence:.0%}")
                        click.echo(f"           Stop ${stop_price:.2f} -> "
                                   f"Target ${target_price:.2f}  "
                                   f"R:R 1:{rr_ratio:.1f}  "
                                   f"Risk ${pos.risk_dollars:.2f} "
                                   f"({pos.lot_size:.2f} lots)")
                        for r in (reasons or [])[:2]:
                            click.echo(f"           -> {r}")

                        # Register with mental tracker for outcome tracking
                        mental_trade = MentalTrade(
                            trade_id=f"xm_{strat.name}_{self.cycle_count}",
                            strategy=strat.name,
                            direction=direction,
                            entry_price=price,
                            stop_price=stop_price,
                            target_price=target_price,
                            timeframe="1h",
                            confidence=confidence,
                            reasons=reasons or [strat.name],
                            rr_ratio=rr_ratio,
                            risk_dollars=pos.risk_dollars if pos else 0,
                            lot_size=pos.lot_size if pos else 0,
                            entry_time=datetime.now().isoformat(),
                            horizon_bars=20,
                            market_context={
                                "gold_pressure": gold_pressure,
                                "risk_sentiment": risk_sentiment,
                                "regime": regime,
                                "balance": self.risk_mgr.balance,
                            },
                        )
                        self.mental_tracker.open_trade(mental_trade)

                        # PAPER TRADE: execute through PaperBroker with real spread/slippage
                        if self.paper_trader.can_open(strat.name):
                            _session = self.session_journal.get_current_session() if self.session_journal else "unknown"
                            pt = self.paper_trader.execute_trade(
                                strategy=strat.name,
                                direction=direction,
                                current_price=price,
                                stop_price=stop_price,
                                target_price=target_price,
                                lots=pos.lot_size if pos else 0,
                                risk_dollars=pos.risk_dollars if pos else 0,
                                confidence=confidence,
                                regime=regime,
                                session=_session,
                                reasons=reasons or [strat.name],
                            )
                            if pt:
                                click.echo(f"           PAPER TRADE: filled @ ${pt.fill_price:.2f} "
                                           f"(slippage ${pt.slippage_cost:.2f}, spread ${pt.spread_cost:.2f})")

                        # Update lifecycle
                        self.lifecycle.record_trade(strat.name, None)

            except Exception as e:
                log.warning("cross_market_strategy_error",
                            strategy=strat.name, error=str(e))

        click.echo(f"         {triggered_count}/{len(self.cross_market_strategies)} "
                   f"strategies triggered")

    def _run_strategy_lab(self, all_data):
        """Run strategy lab — mentally test hypotheses."""
        for tf, df in all_data.items():
            if df is not None and len(df) > 60:
                signals = self.lab.compute_signals(df)
                if not signals:
                    continue

                # Resolve existing mental trades
                resolved = self.lab.resolve_mental_trades(
                    self.lab.hypotheses, df,
                    self.mental_entry_prices, self.mental_bars_since)
                for r in resolved:
                    click.echo(f"         LAB: {r['hypothesis']} {r['outcome']} "
                               f"PnL=${r['pnl']:+.2f}")

                # Check for new triggers
                triggered = self.lab.evaluate_hypotheses(df, signals)
                for hyp in triggered:
                    if hyp.id not in self.mental_entry_prices:
                        self.mental_entry_prices[hyp.id] = signals["price"]
                        self.mental_bars_since[hyp.id] = 0
                        click.echo(f"         LAB ENTRY: {hyp.name} "
                                   f"{hyp.direction} @ ${signals['price']:.2f}")
                break

        click.echo(f"\n{self.lab.get_lab_report()}")

    def _check_daily_report(self):
        """Generate daily report at ~10 PM local time."""
        now = datetime.now()
        today = now.date()
        if now.hour >= 22 and self.last_daily_report_date != today:
            click.echo(f"\n{'=' * 80}")
            click.echo(f"  GENERATING DAILY REPORT...")
            click.echo(f"{'=' * 80}")
            report = self.daily_reporter.generate()
            click.echo(report)
            self.last_daily_report_date = today

    def _start_tick_streamer(self):
        """Start OANDA tick streaming in background."""
        import os
        token = self.cfg["data"].get("oanda_token") or os.getenv("OANDA_TOKEN")
        account_id = self.cfg["data"].get("oanda_account_id") or os.getenv("OANDA_ACCOUNT_ID")

        if not token or not account_id:
            click.echo("\n  OANDA tick streaming: SKIPPED (no credentials)")
            return

        from learn.tick_streamer import TickStreamer
        self.tick_streamer = TickStreamer(
            token=token, account_id=account_id,
            practice=True, data_dir=str(self.data_dir))

        def run():
            try:
                self.tick_streamer.start()
            except Exception as e:
                log.error("tick_streamer_error", error=str(e))

        self.tick_thread = threading.Thread(target=run, daemon=True)
        self.tick_thread.start()
        click.echo("\n  OANDA tick streaming: STARTED (background)")

    def _stop_tick_streamer(self):
        if self.tick_streamer:
            self.tick_streamer.stop()

    def _save_all(self):
        """Save all state."""
        self.observer.save_knowledge()
        self.predictor.save_state()
        self.correlator.save_state()
        self.lab.save_state()
        if self.market_universe:
            try:
                self.market_universe.save_state()
            except Exception:
                pass

    def _print_session_summary(self):
        """Print what was learned this session."""
        click.echo(f"\n{'=' * 75}")
        click.echo(f"  SESSION SUMMARY — {self.cycle_count} cycles completed")
        click.echo(f"{'=' * 75}")

        total = len(self.predictor.resolved_predictions)
        if total > 0:
            correct = sum(1 for p in self.predictor.resolved_predictions
                          if p.outcome == "CORRECT")
            click.echo(f"\n  Predictions: {correct}/{total} correct "
                       f"({correct / total * 100:.1f}%)")

        if self.predictor.confirmed_patterns:
            click.echo(f"\n  Confirmed Patterns:")
            for name, data in sorted(
                    self.predictor.confirmed_patterns.items(),
                    key=lambda x: x[1].get("win_rate", 0), reverse=True):
                click.echo(f"    {name}: {data['status']} "
                           f"(WR={data['win_rate']:.1%})")

        confirmed = [h for h in self.lab.hypotheses
                     if h.status == "CONFIRMED"]
        if confirmed:
            click.echo(f"\n  Confirmed Strategies:")
            for h in confirmed:
                total_h = h.correct + h.wrong
                wr = h.correct / total_h if total_h > 0 else 0
                click.echo(f"    {h.name}: WR={wr:.1%}, PnL=${h.total_pnl:+.1f}")

        click.echo(f"\n  Knowledge saved to {self.data_dir}/")
        click.echo(f"{'=' * 75}")
