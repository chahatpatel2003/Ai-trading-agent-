"""
Daily Report Generator — Comprehensive end-of-day learning report.

Scheduled to run at 10 PM EST (02:00 UTC next day).
Reports on everything the bot learned, tried, discovered, and recommends.

Sections:
1. Executive Summary — what happened today in gold
2. Predictions Made & Outcomes — what the bot predicted, was it right
3. Cross-Market Correlations — what moved with/against gold today
4. Strategy Performance — what strategies worked, which didn't
5. New Discoveries — patterns, correlations, hypotheses invented today
6. Inter-Market Analysis — DXY, oil, VIX, bonds, silver, BTC context
7. Strategy Lab — new strategies invented, mental test results
8. Recommendations — what to add/change to improve accuracy
9. OANDA Tick Analysis — real-time tick insights
10. Tomorrow's Outlook — what the bot expects next
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from logs.logger import get_logger

log = get_logger("learn.daily_report")


class DailyReportGenerator:
    """Generates the comprehensive 10 PM EST daily learning report."""

    def __init__(self, brain):
        """Takes a reference to the LearningBrain instance."""
        self.brain = brain
        self.report_dir = Path(brain.data_dir) / "reports"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> str:
        """Generate the full daily report."""
        now = datetime.now()
        lines = []

        lines.append("=" * 80)
        lines.append(f"  GOLD TRADING BOT — DAILY LEARNING REPORT")
        lines.append(f"  Date: {now.strftime('%A, %B %d, %Y')}")
        lines.append(f"  Generated: {now.strftime('%I:%M %p')} (local time)")
        lines.append(f"  Learning Cycles Completed: {self.brain.cycle_count}")
        lines.append("=" * 80)

        lines.append(self._section_executive_summary())
        lines.append(self._section_predictions())
        lines.append(self._section_cross_market())
        lines.append(self._section_strategies())
        lines.append(self._section_discoveries())
        lines.append(self._section_inter_market())
        lines.append(self._section_strategy_lab())
        lines.append(self._section_tick_analysis())
        lines.append(self._section_recommendations())
        lines.append(self._section_outlook())

        lines.append("\n" + "=" * 80)
        lines.append("  END OF DAILY REPORT")
        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save to file
        path = self.report_dir / f"daily_{now.strftime('%Y%m%d')}.txt"
        with open(path, "w") as f:
            f.write(report)
        log.info("daily_report_saved", path=str(path))

        return report

    def _section_executive_summary(self) -> str:
        """What happened today in gold — the big picture."""
        lines = [f"\n{'─' * 80}", "  1. EXECUTIVE SUMMARY", f"{'─' * 80}"]

        # Gold price from observer
        obs = self.brain.observer.observations
        if obs and len(obs) >= 2:
            latest = obs[-1]
            first_today = obs[-min(len(obs), self.brain.cycle_count)]
            price_now = latest.get("latest_price", 0)
            price_start = first_today.get("latest_price", price_now)
            change = price_now - price_start if price_now and price_start else 0
            change_pct = (change / price_start * 100) if price_start else 0

            lines.append(f"\n  Gold Price: ${price_now:.2f}")
            lines.append(f"  Session Change: ${change:+.2f} ({change_pct:+.2f}%)")

            # Cross-TF consensus
            cross_tf = latest.get("cross_tf", {})
            lines.append(f"  Multi-TF Consensus: {cross_tf.get('confluence', 'N/A')}")
            lines.append(f"  Alignment Score: {cross_tf.get('alignment_score', 0)}")
        else:
            lines.append("\n  Insufficient data for executive summary")

        # Regime
        if self.brain.market_universe:
            try:
                regime = self.brain.market_universe.get_regime()
                lines.append(f"  Macro Regime: {regime.upper()}")
            except Exception:
                pass

        # Trades today
        lines.append(f"\n  Trades Today: {self.brain.trades_today}")
        lines.append(f"  Learning Cycles: {self.brain.cycle_count}")

        return "\n".join(lines)

    def _section_predictions(self) -> str:
        """What the bot predicted and outcomes."""
        lines = [f"\n{'─' * 80}", "  2. PREDICTIONS — What Bot Tried Today", f"{'─' * 80}"]

        pred = self.brain.predictor
        active = pred.active_predictions
        resolved = pred.resolved_predictions

        lines.append(f"\n  Active Predictions: {len(active)}")
        lines.append(f"  Total Resolved: {len(resolved)}")

        if resolved:
            correct = sum(1 for p in resolved if p.outcome == "CORRECT")
            wrong = sum(1 for p in resolved if p.outcome == "WRONG")
            total = len(resolved)
            wr = correct / total * 100 if total > 0 else 0
            pnls = [p.pnl_points for p in resolved if p.pnl_points]

            lines.append(f"  Win Rate: {wr:.1f}% ({correct}W / {wrong}L)")
            if pnls:
                lines.append(f"  Total PnL: ${sum(pnls):+.2f}")
                lines.append(f"  Avg PnL per trade: ${np.mean(pnls):+.2f}")

            # Per-strategy accuracy
            lines.append(f"\n  Per-Strategy Accuracy:")
            lines.append(f"  {'Strategy':<40s} {'W':>3s} {'L':>3s} {'WR%':>6s}")
            lines.append("  " + "-" * 55)
            for key, stats in sorted(pred.pattern_accuracy.items(),
                                     key=lambda x: x[1]["correct"] / max(1, sum(x[1].values())),
                                     reverse=True):
                t = stats["correct"] + stats["wrong"] + stats.get("partial", 0)
                wr_s = stats["correct"] / t * 100 if t > 0 else 0
                status = pred.confirmed_patterns.get(key, {}).get("status", "learning")
                lines.append(f"  {key:<40s} {stats['correct']:>3d} {stats['wrong']:>3d} {wr_s:>5.1f}% [{status}]")

        # Show what's still being watched
        if active:
            lines.append(f"\n  Currently Watching ({len(active)} predictions):")
            for p in active[-5:]:
                lines.append(f"    [{p.id}] {p.direction} ${p.entry_price:.2f} ({p.timeframe}) "
                             f"conf={p.confidence:.0%} bars={p.bars_held}/{p.horizon_bars}")
                lines.append(f"      Reason: {p.reasons[0][:70]}")

        return "\n".join(lines)

    def _section_cross_market(self) -> str:
        """Cross-market correlation findings."""
        lines = [f"\n{'─' * 80}", "  3. CROSS-MARKET CORRELATIONS FOUND TODAY", f"{'─' * 80}"]

        if not self.brain.market_universe:
            lines.append("\n  Cross-market analysis not active")
            return "\n".join(lines)

        mu = self.brain.market_universe

        # Correlation matrix
        try:
            corr = mu.compute_correlation_matrix()
            if corr is not None and "gold" in corr.index:
                lines.append(f"\n  Gold Correlation Matrix (60-day rolling):")
                lines.append(f"  {'Market':<15s} {'Corr':>8s} {'Interpretation'}")
                lines.append("  " + "-" * 60)
                for col in corr.columns:
                    if col == "gold":
                        continue
                    val = corr.loc["gold", col]
                    if abs(val) > 0.01:
                        interp = ""
                        if val > 0.7:
                            interp = "STRONG POSITIVE — moves with gold"
                        elif val > 0.3:
                            interp = "moderate positive"
                        elif val < -0.7:
                            interp = "STRONG INVERSE — moves against gold"
                        elif val < -0.3:
                            interp = "moderate inverse"
                        else:
                            interp = "weak relationship"
                        lines.append(f"  {col:<15s} {val:>+7.3f}  {interp}")
        except Exception as e:
            lines.append(f"\n  Correlation matrix error: {e}")

        # Lead-lag
        try:
            ll = mu.find_lead_lag()
            if ll:
                lines.append(f"\n  Lead-Lag Relationships (who leads gold?):")
                for asset, info in list(ll.items())[:5]:
                    if isinstance(info, dict):
                        lag = info.get("best_lag", 0)
                        c = info.get("correlation", 0)
                        if lag > 0:
                            lines.append(f"    {asset} LEADS gold by {lag} bars (corr={c:.3f})")
                        elif lag < 0:
                            lines.append(f"    {asset} LAGS gold by {abs(lag)} bars (corr={c:.3f})")
                        else:
                            lines.append(f"    {asset} moves simultaneously with gold (corr={c:.3f})")
        except Exception:
            pass

        # Breakdowns
        try:
            breakdowns = mu.detect_correlation_breakdown()
            if breakdowns:
                lines.append(f"\n  *** CORRELATION BREAKDOWNS (unusual behavior) ***")
                for b in breakdowns:
                    if isinstance(b, dict):
                        lines.append(f"    {b.get('market', '?')}: was {b.get('historical_mean', 0):.3f}, "
                                     f"now {b.get('current_corr', 0):.3f} "
                                     f"(z-score: {b.get('z_score', 0):.1f})")
        except Exception:
            pass

        return "\n".join(lines)

    def _section_strategies(self) -> str:
        """Strategy performance today."""
        lines = [f"\n{'─' * 80}", "  4. STRATEGY PERFORMANCE", f"{'─' * 80}"]

        # Cross-market strategy results
        if self.brain.cross_market_strategies:
            lines.append(f"\n  Cross-Market Strategies ({len(self.brain.cross_market_strategies)} active):")
            for strat in self.brain.cross_market_strategies:
                status = "active"
                lines.append(f"    {strat.name:<35s} [{status}]")

        # Strategy lab results
        lab = self.brain.lab
        if lab.hypotheses:
            lines.append(f"\n  Strategy Lab Hypotheses:")
            lines.append(f"  {'Name':<30s} {'Dir':>5s} {'W':>3s} {'L':>3s} {'WR%':>6s} {'PnL':>8s} {'Status'}")
            lines.append("  " + "-" * 70)
            sorted_h = sorted(lab.hypotheses,
                              key=lambda h: h.confidence_score, reverse=True)
            for h in sorted_h:
                total = h.correct + h.wrong
                wr = h.correct / total * 100 if total > 0 else 0
                lines.append(f"  {h.name:<30s} {h.direction:>5s} {h.correct:>3d} "
                             f"{h.wrong:>3d} {wr:>5.1f}% ${h.total_pnl:>+7.1f} {h.status}")

        return "\n".join(lines)

    def _section_discoveries(self) -> str:
        """New patterns and correlations discovered today."""
        lines = [f"\n{'─' * 80}", "  5. NEW DISCOVERIES TODAY", f"{'─' * 80}"]

        # Confirmed patterns from prediction engine
        confirmed = self.brain.predictor.confirmed_patterns
        if confirmed:
            lines.append(f"\n  Confirmed Patterns (passed {self.brain.predictor.min_confirmations}+ tests):")
            for name, data in confirmed.items():
                lines.append(f"    {name}: {data['status']} "
                             f"(WR={data['win_rate']:.1%}, n={data['total_predictions']})")
        else:
            lines.append(f"\n  No patterns confirmed yet (need {self.brain.predictor.min_confirmations}+ tests)")

        # Correlator findings
        corr = self.brain.correlator
        if corr.indicator_lead_lag:
            best_5 = corr.indicator_lead_lag.get("strongest_predictor_5bar")
            best_10 = corr.indicator_lead_lag.get("strongest_predictor_10bar")
            if best_5:
                lines.append(f"\n  Best 5-bar predictor: {best_5}")
            if best_10:
                lines.append(f"  Best 10-bar predictor: {best_10}")

        # Pre-big-move patterns
        if corr.pre_move_patterns:
            squeeze_pct = sum(1 for p in corr.pre_move_patterns
                              if p.get("was_squeeze")) / max(1, len(corr.pre_move_patterns))
            lines.append(f"\n  Pre-Big-Move Analysis ({len(corr.pre_move_patterns)} events):")
            lines.append(f"    {squeeze_pct*100:.0f}% preceded by BB squeeze")
            avg_vol = np.mean([p.get("volume_spike", 1)
                               for p in corr.pre_move_patterns])
            lines.append(f"    Average volume spike on big move: {avg_vol:.1f}x")

        return "\n".join(lines)

    def _section_inter_market(self) -> str:
        """Inter-market performance — DXY, oil, VIX, bonds, silver, BTC."""
        lines = [f"\n{'─' * 80}", "  6. INTER-MARKET PERFORMANCE", f"{'─' * 80}"]

        if not self.brain.market_universe:
            lines.append("\n  Inter-market analysis not active")
            return "\n".join(lines)

        mu = self.brain.market_universe
        snapshot = mu.get_snapshot()

        if snapshot:
            lines.append(f"\n  {'Market':<15s} {'Price':>10s}")
            lines.append("  " + "-" * 30)

            key_markets = ["gold", "silver", "dxy", "wti_oil", "vix",
                           "us10y", "btc", "copper", "sp500", "gdx"]
            for m in key_markets:
                if m in snapshot:
                    val = snapshot[m]
                    if val is not None:
                        lines.append(f"  {m:<15s} {val:>10.2f}")

        # Gold pressure score
        try:
            pressure = mu.compute_gold_pressure_score()
            lines.append(f"\n  Gold Pressure Score: {pressure:+.1f}/100")
            if pressure > 30:
                lines.append(f"    → BULLISH pressure on gold")
            elif pressure < -30:
                lines.append(f"    → BEARISH pressure on gold")
            else:
                lines.append(f"    → NEUTRAL pressure")
        except Exception:
            pass

        # Risk sentiment
        try:
            sentiment = mu.compute_risk_sentiment()
            lines.append(f"  Risk Sentiment: {sentiment:+.2f} (-1=fear, +1=greed)")
        except Exception:
            pass

        return "\n".join(lines)

    def _section_strategy_lab(self) -> str:
        """Strategy lab — what was invented and tested."""
        lines = [f"\n{'─' * 80}", "  7. STRATEGY LAB — Invention & Testing", f"{'─' * 80}"]

        lab = self.brain.lab
        total_hyp = len(lab.hypotheses)
        confirmed = sum(1 for h in lab.hypotheses if h.status == "CONFIRMED")
        promising = sum(1 for h in lab.hypotheses if h.status == "PROMISING")
        testing = sum(1 for h in lab.hypotheses if h.status == "TESTING")
        rejected = sum(1 for h in lab.hypotheses if h.status == "REJECTED")

        lines.append(f"\n  Total Hypotheses: {total_hyp}")
        lines.append(f"  Confirmed: {confirmed}  |  Promising: {promising}  |  "
                     f"Testing: {testing}  |  Rejected: {rejected}")

        # Show confirmed/promising strategies with rules
        for h in lab.hypotheses:
            if h.status in ("CONFIRMED", "PROMISING"):
                total = h.correct + h.wrong
                wr = h.correct / total * 100 if total > 0 else 0
                lines.append(f"\n  *** {h.status}: {h.name} ***")
                lines.append(f"    Direction: {h.direction}  |  WR: {wr:.1f}%  |  PnL: ${h.total_pnl:+.1f}")
                lines.append(f"    Rules: {' AND '.join(h.rules)}")
                lines.append(f"    Origin: {h.origin}")

        # Mental trades open
        open_trades = len(self.brain.mental_entry_prices)
        lines.append(f"\n  Mental Trades Open: {open_trades}")

        return "\n".join(lines)

    def _section_tick_analysis(self) -> str:
        """OANDA real-time tick insights."""
        lines = [f"\n{'─' * 80}", "  8. OANDA REAL-TIME TICK ANALYSIS", f"{'─' * 80}"]

        if not self.brain.tick_streamer:
            lines.append("\n  OANDA tick streaming not active")
            return "\n".join(lines)

        s = self.brain.tick_streamer.analyzer.get_summary()
        lines.append(f"\n  Total Ticks Today: {s.get('total_ticks', 0)}")
        lines.append(f"  Spikes Detected: {s.get('spikes_detected', 0)}")
        lines.append(f"  Spread Widenings: {s.get('spread_widenings', 0)}")
        lines.append(f"  Max Spread Seen: ${s.get('max_spread_seen', 0):.4f}")
        lines.append(f"  Avg Spread: ${s.get('avg_spread', 0):.4f}")
        lines.append(f"  Max Tick Velocity: {s.get('tick_velocity_max', 0):.1f} ticks/sec")

        # Recent spikes
        spikes = s.get("recent_spikes", [])
        if spikes:
            lines.append(f"\n  Recent Spikes:")
            for spike in spikes[-5:]:
                lines.append(f"    {spike.get('direction', '?')} ${spike.get('magnitude', 0):.2f} "
                             f"at ${spike.get('price', 0):.2f} ({spike.get('time', '?')})")

        return "\n".join(lines)

    def _section_recommendations(self) -> str:
        """What to add/change to improve accuracy."""
        lines = [f"\n{'─' * 80}", "  9. RECOMMENDATIONS — How to Improve", f"{'─' * 80}"]

        recs = []

        # Check prediction accuracy
        pred = self.brain.predictor
        if pred.resolved_predictions:
            total = len(pred.resolved_predictions)
            correct = sum(1 for p in pred.resolved_predictions if p.outcome == "CORRECT")
            wr = correct / total if total > 0 else 0

            if wr < 0.5:
                recs.append("PREDICTION ACCURACY LOW (<50%) — Consider:"
                            "\n      - Tightening entry conditions (higher RSI thresholds)"
                            "\n      - Adding volume confirmation filters"
                            "\n      - Using multi-timeframe alignment as a pre-filter")
            if wr > 0.6:
                recs.append(f"GOOD ACCURACY ({wr:.0%}) — Consider increasing confidence thresholds "
                            f"for higher-quality, fewer trades")

        # Check which patterns work
        for key, stats in pred.pattern_accuracy.items():
            total = stats["correct"] + stats["wrong"]
            if total >= 5:
                wr = stats["correct"] / total
                if wr > 0.65:
                    recs.append(f"STRONG PATTERN: '{key}' has {wr:.0%} win rate — "
                                f"consider promoting to primary strategy")
                if wr < 0.35:
                    recs.append(f"WEAK PATTERN: '{key}' only {wr:.0%} win rate — "
                                f"consider disabling or inverting conditions")

        # Cross-market suggestions
        corr = self.brain.correlator
        if corr.day_of_week_stats:
            for day, stats in corr.day_of_week_stats.items():
                if stats["returns"]:
                    avg = np.mean(stats["returns"])
                    if avg < -0.001:
                        recs.append(f"DAY-OF-WEEK: {day} has negative avg return — "
                                    f"consider reducing exposure on {day}s")

        # Strategy lab suggestions
        for h in self.brain.lab.hypotheses:
            if h.status == "CONFIRMED":
                recs.append(f"PROMOTE: '{h.name}' is CONFIRMED — ready for paper trading")

        if not recs:
            recs.append("No specific recommendations yet — bot needs more data cycles to identify improvements")

        for i, rec in enumerate(recs, 1):
            lines.append(f"\n  {i}. {rec}")

        return "\n".join(lines)

    def _section_outlook(self) -> str:
        """What the bot expects tomorrow."""
        lines = [f"\n{'─' * 80}", "  10. TOMORROW'S OUTLOOK", f"{'─' * 80}"]

        # Active predictions direction
        active = self.brain.predictor.active_predictions
        long_count = sum(1 for p in active if p.direction == "LONG")
        short_count = sum(1 for p in active if p.direction == "SHORT")

        if long_count > short_count:
            lines.append(f"\n  Bias: BULLISH ({long_count} LONG vs {short_count} SHORT active predictions)")
        elif short_count > long_count:
            lines.append(f"\n  Bias: BEARISH ({short_count} SHORT vs {long_count} LONG active predictions)")
        else:
            lines.append(f"\n  Bias: NEUTRAL (balanced predictions)")

        # Key levels
        obs = self.brain.observer
        if obs.support_resistance_levels:
            lines.append(f"\n  Key Levels to Watch:")
            for level in obs.support_resistance_levels:
                lines.append(f"    {level.get('type', '?')}: ${level.get('level', 0):.2f}")

        # Squeeze alert
        if obs.observations:
            latest = obs.observations[-1]
            tf_summaries = latest.get("tf_summaries", {})
            squeeze_tfs = [tf for tf, s in tf_summaries.items()
                           if s and s.get("bb_squeeze")]
            if squeeze_tfs:
                lines.append(f"\n  *** BB SQUEEZE ACTIVE on: {', '.join(squeeze_tfs)} ***")
                lines.append(f"      → Big volatility expansion expected — be ready for breakout")

        lines.append(f"\n  The bot will continue learning overnight.")
        lines.append(f"  Next daily report: tomorrow at 10 PM EST.")

        return "\n".join(lines)
