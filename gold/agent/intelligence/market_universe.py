"""
Market Universe — cross-market intelligence for gold trading.

Fetches correlated markets via yfinance, computes correlations, lead-lag
relationships, regime detection, and composite pressure scores.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from logs.logger import get_logger

logger = get_logger("market_universe")

# ---------------------------------------------------------------------------
# Symbol registry
# ---------------------------------------------------------------------------

GOLD = {"gold": "GC=F"}

GOLD_PROXIES = {
    "silver": "SI=F",
    "platinum": "PL=F",
    "gdx": "GDX",
    "gld": "GLD",
}

CURRENCIES = {
    "dxy": "DX-Y.NYB",
    "eurusd": "EURUSD=X",
    "usdjpy": "USDJPY=X",
    "usdchf": "USDCHF=X",
    "audusd": "AUDUSD=X",
    "usdcnh": "CNY=X",
}

COMMODITIES = {
    "wti_oil": "CL=F",
    "brent_oil": "BZ=F",
    "copper": "HG=F",
    "natgas": "NG=F",
}

BONDS = {
    "us10y": "^TNX",
    "us2y": "^IRX",
    "tips": "TIP",
}

EQUITIES = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "vix": "^VIX",
}

CRYPTO = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
}

ALL_MARKETS = {**GOLD, **GOLD_PROXIES, **CURRENCIES, **COMMODITIES, **BONDS, **EQUITIES, **CRYPTO}

# Subset used for intraday fetches (to avoid rate limits)
INTRADAY_KEYS = ["gold", "dxy", "silver", "wti_oil", "vix", "btc"]


class MarketUniverse:
    """Fetches and analyses the full market universe surrounding gold."""

    def __init__(self, data_dir: str = "./learn/data"):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "market_cache"
        self.state_path = self.data_dir / "market_universe_state.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory stores populated by fetch methods
        self._daily: Dict[str, pd.DataFrame] = {}
        self._intraday: Dict[str, pd.DataFrame] = {}
        self._state: dict = self._load_state()

    # ------------------------------------------------------------------
    # 1. fetch_all
    # ------------------------------------------------------------------

    def fetch_all(self, period: str = "6mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """Fetch daily data for every market. Uses CSV cache (stale > 1 day)."""
        result: Dict[str, pd.DataFrame] = {}
        for name, symbol in ALL_MARKETS.items():
            cache_file = self.cache_dir / f"{name}_{interval}.csv"
            df = self._read_cache(cache_file, max_age_seconds=86400)
            if df is not None:
                result[name] = df
                continue
            df = self._download(symbol, period=period, interval=interval, label=name)
            if df is not None and not df.empty:
                df.to_csv(cache_file)
                result[name] = df
        self._daily = result
        logger.info("fetch_all_complete", markets=len(result))
        return result

    # ------------------------------------------------------------------
    # 2. fetch_intraday
    # ------------------------------------------------------------------

    def fetch_intraday(self, interval: str = "1h") -> Dict[str, pd.DataFrame]:
        """Fetch recent intraday data for key markets (max 60 days)."""
        result: Dict[str, pd.DataFrame] = {}
        for name in INTRADAY_KEYS:
            symbol = ALL_MARKETS.get(name)
            if symbol is None:
                continue
            cache_file = self.cache_dir / f"{name}_{interval}.csv"
            df = self._read_cache(cache_file, max_age_seconds=3600)
            if df is not None:
                result[name] = df
                continue
            df = self._download(symbol, period="60d", interval=interval, label=name)
            if df is not None and not df.empty:
                df.to_csv(cache_file)
                result[name] = df
        self._intraday = result
        logger.info("fetch_intraday_complete", markets=len(result))
        return result

    # ------------------------------------------------------------------
    # 3. get_snapshot
    # ------------------------------------------------------------------

    def get_snapshot(self) -> Dict[str, float]:
        """Return latest close price for every market in daily data."""
        self._ensure_daily()
        snapshot: Dict[str, float] = {}
        for name, df in self._daily.items():
            try:
                snapshot[name] = float(df["Close"].dropna().iloc[-1])
            except (IndexError, KeyError):
                pass
        return snapshot

    # ------------------------------------------------------------------
    # 4. compute_correlation_matrix
    # ------------------------------------------------------------------

    def compute_correlation_matrix(self, window: int = 60, clean_rmt: bool = True) -> pd.DataFrame:
        """
        Correlation of daily returns between gold and all markets.

        When clean_rmt=True, applies Marchenko-Pastur / Ledoit-Péché
        cleaning to remove noise eigenvalues from the sample matrix.
        """
        self._ensure_daily()
        returns = self._aligned_returns()
        if "gold" not in returns.columns or returns.shape[1] < 2:
            logger.warning("correlation_matrix_insufficient_data")
            return pd.DataFrame()

        tail = returns.tail(window).dropna()
        corr = tail.corr()

        if clean_rmt and len(tail) > corr.shape[0]:
            try:
                from intelligence.rmt_cleaner import clean_correlation_dataframe
                corr, diag = clean_correlation_dataframe(corr, n_observations=len(tail))
                self._rmt_diagnostics = diag
            except Exception as e:
                logger.warning("rmt_cleaning_failed", error=str(e))

        return corr

    # ------------------------------------------------------------------
    # 5. find_lead_lag
    # ------------------------------------------------------------------

    def find_lead_lag(self, target: str = "gold", max_lag: int = 10) -> Dict[str, dict]:
        """
        Cross-correlation at lags -max_lag..+max_lag for each market vs target.
        Positive best_lag means the market LEADS gold by that many bars.
        Returns dict: {market: {"best_lag": int, "correlation": float}}.
        """
        self._ensure_daily()
        returns = self._aligned_returns().dropna()
        if target not in returns.columns:
            logger.warning("lead_lag_no_target", target=target)
            return {}

        target_series = returns[target].values
        results: Dict[str, dict] = {}

        for col in returns.columns:
            if col == target:
                continue
            other = returns[col].values
            best_corr = 0.0
            best_lag = 0
            for lag in range(-max_lag, max_lag + 1):
                if lag > 0:
                    a, b = target_series[lag:], other[:-lag]
                elif lag < 0:
                    a, b = target_series[:lag], other[-lag:]
                else:
                    a, b = target_series, other
                n = min(len(a), len(b))
                if n < 20:
                    continue
                r, _ = stats.pearsonr(a[:n], b[:n])
                if abs(r) > abs(best_corr):
                    best_corr = r
                    best_lag = lag
            results[col] = {"best_lag": best_lag, "correlation": round(best_corr, 4)}

        # Sort by absolute correlation descending
        results = dict(sorted(results.items(), key=lambda x: abs(x[1]["correlation"]), reverse=True))
        return results

    # ------------------------------------------------------------------
    # 6. detect_correlation_breakdown
    # ------------------------------------------------------------------

    def detect_correlation_breakdown(self, lookback: int = 120, recent: int = 20) -> List[dict]:
        """
        Compare recent vs historical gold-correlation for every market.
        Flag when the shift exceeds 2 standard deviations.
        """
        self._ensure_daily()
        returns = self._aligned_returns().dropna()
        if "gold" not in returns.columns or len(returns) < lookback:
            return []

        gold_ret = returns["gold"]
        alerts: List[dict] = []

        for col in returns.columns:
            if col == "gold":
                continue
            other_ret = returns[col]
            # Compute rolling correlation
            rolling_corr = gold_ret.rolling(recent).corr(other_ret)
            historical = rolling_corr.iloc[-lookback:-recent]
            current_corr = rolling_corr.iloc[-1]

            if historical.std() == 0 or historical.isna().all():
                continue

            z_score = (current_corr - historical.mean()) / historical.std()
            if abs(z_score) > 2.0:
                alerts.append({
                    "market": col,
                    "current_corr": round(float(current_corr), 4),
                    "historical_mean": round(float(historical.mean()), 4),
                    "z_score": round(float(z_score), 2),
                    "direction": "strengthened" if z_score > 0 else "weakened",
                })

        if alerts:
            logger.info("correlation_breakdowns_detected", count=len(alerts))
        return alerts

    # ------------------------------------------------------------------
    # 7. compute_gold_pressure_score
    # ------------------------------------------------------------------

    def compute_gold_pressure_score(self) -> Dict[str, object]:
        """
        Composite score -100 to +100 indicating bullish/bearish gold pressure.
        Each factor is scored via rolling z-score, then weighted.
        """
        self._ensure_daily()
        returns = self._aligned_returns(min_bars=60).dropna()

        factors: Dict[str, Tuple[float, float]] = {}  # name -> (z_score, weight)

        # Helper: z-score of recent N-bar return for a given market
        def _zscore(market: str, window: int = 20, full: int = 120) -> Optional[float]:
            if market not in returns.columns or len(returns) < full:
                return None
            series = returns[market].rolling(window).sum()
            recent = series.iloc[-1]
            hist = series.iloc[-full:]
            if hist.std() == 0:
                return None
            return float((recent - hist.mean()) / hist.std())

        # DXY falling -> bullish gold
        z = _zscore("dxy")
        if z is not None:
            factors["dxy_weakness"] = (-z, 20)  # negative z = falling DXY = bullish

        # VIX rising -> bullish gold
        z = _zscore("vix")
        if z is not None:
            factors["vix_rising"] = (z, 15)

        # Real yields falling (10Y - TIPS) -> bullish gold
        z_10y = _zscore("us10y")
        z_tips = _zscore("tips")
        if z_10y is not None and z_tips is not None:
            factors["real_yields_falling"] = (-(z_10y - z_tips), 25)
        elif z_10y is not None:
            factors["real_yields_falling"] = (-z_10y, 25)

        # Oil rising -> inflation -> bullish gold
        z = _zscore("wti_oil")
        if z is not None:
            factors["oil_inflation"] = (z, 10)

        # Silver outperforming gold -> bullish gold
        z_silver = _zscore("silver")
        z_gold = _zscore("gold")
        if z_silver is not None and z_gold is not None:
            factors["silver_outperform"] = (z_silver - z_gold, 10)

        # BTC falling -> safe haven rotation -> bullish gold
        z = _zscore("btc")
        if z is not None:
            factors["btc_weakness"] = (-z, 10)

        # Copper falling -> growth fears -> bullish gold
        z = _zscore("copper")
        if z is not None:
            factors["copper_weakness"] = (-z, 10)

        # Compute weighted composite
        total_weight = sum(w for _, w in factors.values())
        if total_weight == 0:
            return {"score": 0.0, "factors": {}, "label": "neutral"}

        raw = sum(np.clip(z, -2, 2) / 2 * w for z, w in factors.values())
        score = round(float(np.clip(raw, -100, 100)), 1)

        factor_detail = {
            k: {"z_score": round(z, 3), "weight": w, "contribution": round(np.clip(z, -2, 2) / 2 * w, 1)}
            for k, (z, w) in factors.items()
        }

        label = "strongly_bullish" if score > 50 else "bullish" if score > 15 else \
                "strongly_bearish" if score < -50 else "bearish" if score < -15 else "neutral"

        return {"score": score, "factors": factor_detail, "label": label}

    # ------------------------------------------------------------------
    # 8. compute_risk_sentiment
    # ------------------------------------------------------------------

    def compute_risk_sentiment(self) -> Dict[str, object]:
        """
        Risk sentiment score from -1 (extreme fear) to +1 (extreme greed).
        """
        self._ensure_daily()
        returns = self._aligned_returns(min_bars=60).dropna()
        components: Dict[str, float] = {}

        def _zscore(market: str, window: int = 20, full: int = 120) -> Optional[float]:
            if market not in returns.columns or len(returns) < full:
                return None
            series = returns[market].rolling(window).sum()
            recent = series.iloc[-1]
            hist = series.iloc[-full:]
            if hist.std() == 0:
                return None
            return float((recent - hist.mean()) / hist.std())

        # VIX: high = fear (invert)
        z = _zscore("vix")
        if z is not None:
            components["vix"] = -np.clip(z, -3, 3) / 3

        # S&P 500: rising = greed
        z = _zscore("sp500")
        if z is not None:
            components["sp500"] = np.clip(z, -3, 3) / 3

        # BTC: rising = greed
        z = _zscore("btc")
        if z is not None:
            components["btc"] = np.clip(z, -3, 3) / 3

        # DXY: rising USD = risk-off = fear (invert)
        z = _zscore("dxy")
        if z is not None:
            components["dxy"] = -np.clip(z, -3, 3) / 3

        if not components:
            return {"score": 0.0, "components": {}, "label": "neutral"}

        score = float(np.clip(np.mean(list(components.values())), -1, 1))
        components = {k: round(float(v), 3) for k, v in components.items()}

        label = "extreme_fear" if score < -0.6 else "fear" if score < -0.2 else \
                "extreme_greed" if score > 0.6 else "greed" if score > 0.2 else "neutral"

        return {"score": round(score, 3), "components": components, "label": label}

    # ------------------------------------------------------------------
    # 9. get_regime
    # ------------------------------------------------------------------

    def get_regime(self) -> str:
        """
        Classify current macro regime:
        risk_on, risk_off, inflation_fear, deflation_fear, uncertainty.
        """
        self._ensure_daily()
        returns = self._aligned_returns(min_bars=60).dropna()

        def _trend(market: str, window: int = 20) -> Optional[float]:
            """Cumulative return over window (positive = up)."""
            if market not in returns.columns or len(returns) < window:
                return None
            return float(returns[market].iloc[-window:].sum())

        vix_trend = _trend("vix")
        sp500_trend = _trend("sp500")
        gold_trend = _trend("gold")
        oil_trend = _trend("wti_oil")
        copper_trend = _trend("copper")
        tips_trend = _trend("tips")
        dxy_trend = _trend("dxy")

        # Scoring each regime
        scores = {
            "risk_on": 0.0,
            "risk_off": 0.0,
            "inflation_fear": 0.0,
            "deflation_fear": 0.0,
            "uncertainty": 0.0,
        }

        if sp500_trend is not None and sp500_trend > 0.02:
            scores["risk_on"] += 2
        if sp500_trend is not None and sp500_trend < -0.02:
            scores["risk_off"] += 2

        if vix_trend is not None and vix_trend > 0.05:
            scores["risk_off"] += 2
            scores["uncertainty"] += 1
        if vix_trend is not None and vix_trend < -0.05:
            scores["risk_on"] += 1

        if gold_trend is not None and gold_trend > 0.02:
            scores["risk_off"] += 1
            scores["inflation_fear"] += 1

        if oil_trend is not None and oil_trend > 0.05:
            scores["inflation_fear"] += 2
        if oil_trend is not None and oil_trend < -0.05:
            scores["deflation_fear"] += 2

        if copper_trend is not None and copper_trend < -0.03:
            scores["deflation_fear"] += 1
            scores["risk_off"] += 1
        if copper_trend is not None and copper_trend > 0.03:
            scores["risk_on"] += 1
            scores["inflation_fear"] += 1

        if tips_trend is not None and tips_trend < -0.01:
            scores["inflation_fear"] += 1  # real yields rising or TIPS selling
        if tips_trend is not None and tips_trend > 0.01:
            scores["deflation_fear"] += 1  # flight to inflation protection

        if dxy_trend is not None and dxy_trend > 0.02:
            scores["risk_off"] += 1
        if dxy_trend is not None and dxy_trend < -0.02:
            scores["risk_on"] += 1

        # If top scores are close, it's uncertainty
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] <= 1 and sorted_scores[0] > 0:
            scores["uncertainty"] += 1.5

        regime = max(scores, key=scores.get)

        # Persist to state
        self._state.setdefault("regime_history", [])
        self._state["regime_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "regime": regime,
            "scores": {k: round(v, 2) for k, v in scores.items()},
        })
        # Keep last 100 entries
        self._state["regime_history"] = self._state["regime_history"][-100:]

        logger.info("regime_detected", regime=regime, scores=scores)
        return regime

    # ------------------------------------------------------------------
    # 10. save_state / _load_state
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist correlation matrix, lead-lag findings, and regime history to JSON."""
        try:
            # Correlation matrix (RMT-cleaned)
            corr = self.compute_correlation_matrix()
            if not corr.empty:
                self._state["correlation_matrix"] = corr.round(4).to_dict()

            # RMT diagnostics
            if hasattr(self, '_rmt_diagnostics') and self._rmt_diagnostics:
                self._state["rmt_diagnostics"] = self._rmt_diagnostics

            # Lead-lag
            lead_lag = self.find_lead_lag()
            if lead_lag:
                self._state["lead_lag"] = lead_lag

            self._state["last_updated"] = datetime.utcnow().isoformat()

            with open(self.state_path, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
            logger.info("state_saved", path=str(self.state_path))
        except Exception as e:
            logger.error("state_save_failed", error=str(e))

    def _load_state(self) -> dict:
        """Load persisted state from JSON."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("state_load_failed", error=str(e))
        return {}

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _download(self, symbol: str, period: str, interval: str, label: str) -> Optional[pd.DataFrame]:
        """Download data from yfinance, returning None on failure."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df is None or df.empty:
                logger.warning("download_empty", symbol=symbol, label=label)
                return None
            logger.info("downloaded", label=label, symbol=symbol, rows=len(df))
            return df
        except Exception as e:
            logger.warning("download_failed", symbol=symbol, label=label, error=str(e))
            return None

    def _read_cache(self, path: Path, max_age_seconds: int) -> Optional[pd.DataFrame]:
        """Read CSV cache if it exists and is fresh enough."""
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > max_age_seconds:
            return None
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if df.empty:
                return None
            return df
        except Exception:
            return None

    def _ensure_daily(self) -> None:
        """Lazy-load daily data if not already fetched."""
        if not self._daily:
            self.fetch_all()

    def _aligned_returns(self, min_bars: int = 30) -> pd.DataFrame:
        """Build an aligned DataFrame of daily log returns across all markets."""
        close_dict: Dict[str, pd.Series] = {}
        for name, df in self._daily.items():
            if "Close" in df.columns and len(df) >= min_bars:
                series = df["Close"].copy()
                # Normalize index to date-only to fix timezone misalignment
                idx = pd.to_datetime(series.index, utc=True)
                series.index = idx.normalize().tz_localize(None)
                series = series[~series.index.duplicated(keep='last')]
                close_dict[name] = series

        if not close_dict:
            return pd.DataFrame()

        closes = pd.DataFrame(close_dict)
        # Forward fill small gaps, then compute log returns
        closes = closes.ffill(limit=3)
        returns = np.log(closes / closes.shift(1)).iloc[1:]
        return returns
