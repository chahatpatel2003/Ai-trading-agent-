"""
Real-Time Tick Streamer for XAUUSD via OANDA Streaming API.

Connects to OANDA's price stream, receives every tick, and:
- Builds live candles (10s, 1m, 5m) from raw ticks
- Detects instant spikes, spread widening, momentum bursts
- Feeds completed candles into the MarketObserver
- Logs every tick for post-session analysis

Requires a free OANDA practice account:
  1. Sign up at https://www.oanda.com (Practice/Demo account — free)
  2. Go to "Manage API Access" in account settings
  3. Generate a personal access token
  4. Set OANDA_TOKEN and OANDA_ACCOUNT_ID in your .env file
"""
import json
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from logs.logger import get_logger

log = get_logger("learn.tick_streamer")


class LiveCandle:
    """Accumulates ticks into a candle of a given duration."""
    def __init__(self, duration_seconds: int):
        self.duration = duration_seconds
        self.reset()

    def reset(self):
        self.open = None
        self.high = None
        self.low = None
        self.close = None
        self.volume = 0
        self.start_time = None
        self.ticks = 0

    def update(self, price: float, timestamp: datetime):
        if self.open is None:
            self.open = price
            self.high = price
            self.low = price
            self.start_time = timestamp
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += 1
        self.ticks += 1

    def is_complete(self, current_time: datetime) -> bool:
        if self.start_time is None:
            return False
        return (current_time - self.start_time).total_seconds() >= self.duration

    def to_dict(self) -> dict:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "ticks": self.ticks,
            "start_time": str(self.start_time),
            "duration_sec": self.duration,
        }


class TickAnalyzer:
    """Real-time analysis on raw tick data."""
    def __init__(self):
        self.tick_buffer = deque(maxlen=1000)  # last 1000 ticks
        self.price_history = deque(maxlen=5000)
        self.spread_history = deque(maxlen=500)
        self.spike_events = []
        self.stats = {
            "total_ticks": 0,
            "spikes_detected": 0,
            "spread_widenings": 0,
            "max_spread_seen": 0,
            "avg_spread": 0,
            "tick_velocity_max": 0,
        }

    def on_tick(self, bid: float, ask: float, timestamp: datetime):
        mid = (bid + ask) / 2
        spread = ask - bid

        self.tick_buffer.append({
            "time": timestamp,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
        })
        self.price_history.append(mid)
        self.spread_history.append(spread)
        self.stats["total_ticks"] += 1

        # Track spread
        if spread > self.stats["max_spread_seen"]:
            self.stats["max_spread_seen"] = round(spread, 4)
        if len(self.spread_history) > 10:
            self.stats["avg_spread"] = round(np.mean(list(self.spread_history)), 4)

        # Spike detection: price moved > $2 in last 5 ticks
        if len(self.price_history) >= 5:
            recent_prices = list(self.price_history)[-5:]
            move = abs(recent_prices[-1] - recent_prices[0])
            if move > 2.0:
                direction = "UP" if recent_prices[-1] > recent_prices[0] else "DOWN"
                self.stats["spikes_detected"] += 1
                event = {
                    "time": str(timestamp),
                    "direction": direction,
                    "magnitude": round(move, 2),
                    "price": round(mid, 2),
                    "spread_at_spike": round(spread, 4),
                }
                self.spike_events.append(event)
                if len(self.spike_events) > 100:
                    self.spike_events = self.spike_events[-100:]
                log.info("spike_detected", **event)

        # Spread widening: spread > 2x average
        if len(self.spread_history) > 50:
            avg = np.mean(list(self.spread_history)[-50:])
            if spread > avg * 2 and avg > 0:
                self.stats["spread_widenings"] += 1

        # Tick velocity: ticks per second over last 10 seconds
        if len(self.tick_buffer) >= 10:
            recent = list(self.tick_buffer)[-10:]
            time_span = (recent[-1]["time"] - recent[0]["time"]).total_seconds()
            if time_span > 0:
                velocity = len(recent) / time_span
                self.stats["tick_velocity_max"] = max(
                    self.stats["tick_velocity_max"], round(velocity, 1)
                )

    def get_summary(self) -> dict:
        summary = dict(self.stats)
        if self.tick_buffer:
            last = self.tick_buffer[-1]
            summary["current_bid"] = last["bid"]
            summary["current_ask"] = last["ask"]
            summary["current_mid"] = last["mid"]
            summary["current_spread"] = last["spread"]
        summary["recent_spikes"] = self.spike_events[-5:]
        return summary


class TickStreamer:
    """
    Connects to OANDA streaming API and processes live ticks.
    Builds candles at multiple granularities and runs real-time analysis.
    """
    def __init__(self, token: str, account_id: str, practice: bool = True,
                 data_dir: str = "./learn/data"):
        self.token = token
        self.account_id = account_id
        self.practice = practice
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Candle builders at different granularities
        self.candles = {
            "10s": LiveCandle(10),
            "1m": LiveCandle(60),
            "5m": LiveCandle(300),
        }
        self.completed_candles = {k: [] for k in self.candles}
        self.analyzer = TickAnalyzer()

        self.running = False
        self.tick_count = 0
        self.start_time = None
        self.last_report_time = None
        self.report_interval = 60  # print summary every 60 seconds

        # Heartbeat watchdog
        self.last_tick_time = None
        self.stale_timeout = 120  # seconds with no tick before force-reconnect
        self._stream_client = None  # ref so watchdog can kill it
        self._watchdog_thread = None

        # Tick log (rotating file)
        self.tick_log_path = self.data_dir / "live_ticks.jsonl"

    def start(self, callback=None):
        """Start streaming ticks with auto-reconnect. Blocks until stopped."""
        self.running = True
        self.start_time = datetime.now()
        self.last_report_time = datetime.now()
        self.last_tick_time = datetime.now()

        print(f"\n  LIVE TICK STREAM STARTED — {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Instrument: XAU_USD (XAUUSD Gold)")
        print(f"  Building candles: 10s, 1m, 5m")
        print(f"  Auto-reconnect: ON")
        print(f"  Heartbeat watchdog: {self.stale_timeout}s timeout")
        print(f"  Press Ctrl+C to stop\n")

        # Start watchdog thread
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="tick-watchdog"
        )
        self._watchdog_thread.start()

        while self.running:
            try:
                self._connect_and_stream(callback)
            except KeyboardInterrupt:
                print("\n  Stream stopped by user.")
                self.running = False
            except Exception as e:
                log.warning("stream_disconnected", error=str(e))
                print(f"\n  [{datetime.now().strftime('%H:%M:%S')}] Stream disconnected: {e}")
                if self.running:
                    wait = 10
                    print(f"  Reconnecting in {wait}s...")
                    time.sleep(wait)

        self._save_session()

    def _watchdog_loop(self):
        """Background thread that detects stale streams and force-reconnects."""
        while self.running:
            time.sleep(30)  # check every 30s
            if not self.running or self.last_tick_time is None:
                continue

            stale_seconds = (datetime.now() - self.last_tick_time).total_seconds()

            # During market hours (Sun 5pm - Fri 5pm ET), ticks should arrive
            # frequently. Outside market hours, be more lenient.
            now = datetime.now()
            weekday = now.weekday()  # 0=Mon ... 6=Sun
            hour = now.hour

            # Market closed: Fri 5pm ET (Sat ~midnight UTC) to Sun 5pm ET
            # Rough check: if Saturday, or Sunday before ~10pm UTC, skip
            is_weekend = (weekday == 5) or (weekday == 6 and hour < 22)
            timeout = self.stale_timeout * 10 if is_weekend else self.stale_timeout

            if stale_seconds > timeout:
                log.warning(
                    "watchdog_stale_stream",
                    stale_seconds=round(stale_seconds),
                    timeout=timeout,
                    tick_count=self.tick_count,
                )
                print(f"\n  [{now.strftime('%H:%M:%S')}] WATCHDOG: No tick for "
                      f"{int(stale_seconds)}s — force-reconnecting...")

                # Kill the current stream connection
                self._kill_stream()
                self.last_tick_time = datetime.now()  # reset so we don't re-trigger

    def _kill_stream(self):
        """Force-close the current OANDA stream so the reconnect loop takes over."""
        try:
            if self._stream_client is not None:
                self._stream_client.close()
                self._stream_client = None
        except Exception:
            pass
        # Also try to close the underlying requests session
        try:
            if hasattr(self, '_stream_response') and self._stream_response:
                self._stream_response.close()
                self._stream_response = None
        except Exception:
            pass

    def _connect_and_stream(self, callback=None):
        """Single connection attempt — streams until disconnected."""
        import oandapyV20
        from oandapyV20.endpoints.pricing import PricingStream

        env = "practice" if self.practice else "live"
        client = oandapyV20.API(access_token=self.token, environment=env)
        self._stream_client = client
        params = {"instruments": "XAU_USD"}
        stream = PricingStream(accountID=self.account_id, params=params)

        log.info("stream_connected", account=self.account_id, env=env,
                 reconnect_count=self.tick_count > 0)
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Connected to OANDA {env} stream "
              f"(total ticks so far: {self.tick_count:,})")

        try:
            for response in client.request(stream):
                if not self.running:
                    break

                # Heartbeat: OANDA sends HEARTBEAT type when no price change
                if response.get("type") == "HEARTBEAT":
                    self.last_tick_time = datetime.now()
                    continue

                if response.get("type") == "PRICE":
                    self.last_tick_time = datetime.now()
                    bid = float(response["bids"][0]["price"])
                    ask = float(response["asks"][0]["price"])
                    ts = datetime.fromisoformat(
                        response["time"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    self._process_tick(bid, ask, ts, callback)
        finally:
            self._stream_client = None

    def _process_tick(self, bid: float, ask: float, ts: datetime, callback=None):
        mid = (bid + ask) / 2
        self.tick_count += 1

        # Feed to analyzer
        self.analyzer.on_tick(bid, ask, ts)

        # Update all candle builders
        for label, candle in self.candles.items():
            candle.update(mid, ts)
            if candle.is_complete(ts):
                completed = candle.to_dict()
                completed["timeframe"] = label
                self.completed_candles[label].append(completed)
                candle.reset()
                candle.update(mid, ts)

                if callback:
                    callback(label, completed)

        # Log tick (every tick to file, summarized to console)
        self._log_tick(bid, ask, mid, ts)

        # Periodic console report
        now = datetime.now()
        if (now - self.last_report_time).total_seconds() >= self.report_interval:
            self._print_live_report(mid, bid, ask)
            self.last_report_time = now

    def _log_tick(self, bid, ask, mid, ts):
        """Append tick to rotating JSONL file."""
        entry = {"t": str(ts), "b": round(bid, 3), "a": round(ask, 3), "m": round(mid, 3)}
        with open(self.tick_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Rotate file if > 50MB
        if self.tick_log_path.stat().st_size > 50_000_000:
            archive = self.data_dir / f"ticks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            self.tick_log_path.rename(archive)

    def _print_live_report(self, price, bid, ask):
        """Print live status to console."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        tps = self.tick_count / elapsed if elapsed > 0 else 0
        summary = self.analyzer.get_summary()

        candle_counts = {k: len(v) for k, v in self.completed_candles.items()}

        print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
              f"${price:.2f}  spread=${ask-bid:.3f}  "
              f"ticks={self.tick_count}  tps={tps:.1f}/s  "
              f"spikes={summary['spikes_detected']}  "
              f"candles: {candle_counts}")

        # Show recent spikes if any
        recent = summary.get("recent_spikes", [])
        if recent:
            last_spike = recent[-1]
            print(f"           Last spike: {last_spike['direction']} ${last_spike['magnitude']} "
                  f"at {last_spike['time']}")

    def _save_session(self):
        """Save all accumulated data from this streaming session."""
        session = {
            "start_time": str(self.start_time),
            "end_time": str(datetime.now()),
            "total_ticks": self.tick_count,
            "tick_stats": self.analyzer.get_summary(),
            "completed_candles": {
                k: len(v) for k, v in self.completed_candles.items()
            },
            "spike_events": self.analyzer.spike_events,
        }

        path = self.data_dir / f"stream_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(path, "w") as f:
            json.dump(session, f, indent=2, default=str)

        # Save completed candles as CSV for each timeframe
        for label, candles in self.completed_candles.items():
            if candles:
                df = pd.DataFrame(candles)
                csv_path = self.data_dir / f"live_candles_{label}.csv"
                # Append if exists
                if csv_path.exists():
                    df.to_csv(csv_path, mode="a", header=False, index=False)
                else:
                    df.to_csv(csv_path, index=False)

        print(f"\n  Session saved: {path}")
        print(f"  Tick log: {self.tick_log_path}")
        log.info("session_saved", ticks=self.tick_count, path=str(path))

    def stop(self):
        self.running = False
