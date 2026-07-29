"""
Ensemble / Meta-Strategy
- Collects signals from all sub-strategies
- Requires minimum agreement before taking a trade
- Optionally weights strategies by recent rolling Sharpe ratio
- Only used if out-of-sample validation supports it

WHY THIS MAY FAIL:
- Ensemble is only as good as its components; garbage-in, garbage-out
- Weighting by recent Sharpe is itself a form of curve-fitting to recent data
- If all strategies correlate in their failures, the ensemble fails too
- Added complexity may not improve risk-adjusted returns vs the best single strategy
"""
import numpy as np
import pandas as pd

from strategies.base import BaseStrategy


class EnsembleStrategy(BaseStrategy):
    def __init__(self, cfg: dict, sub_strategies: list[BaseStrategy]):
        super().__init__("ensemble", cfg)
        self.sub_strategies = sub_strategies
        ecfg = cfg["strategies"]["ensemble"]
        self.min_agreement = ecfg["min_agreement"]
        self.weight_by_sharpe = ecfg["weight_by_recent_sharpe"]
        self.sharpe_lookback = ecfg["lookback_for_weights"]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all sub-strategies, then combine via voting / weighted consensus.
        """
        df = df.copy()

        # Collect signals from each sub-strategy
        signal_columns = []
        stop_columns = []
        conf_columns = []
        for strat in self.sub_strategies:
            result = strat.generate_signals(df[["open", "high", "low", "close", "volume"]].copy())
            col_name = f"sig_{strat.name}"
            stop_col = f"stop_{strat.name}"
            conf_col = f"conf_{strat.name}"
            df[col_name] = result["signal"]
            df[stop_col] = result["stop_price"]
            df[conf_col] = result.get("confidence", 0.5)
            signal_columns.append(col_name)
            stop_columns.append(stop_col)
            conf_columns.append(conf_col)

        # Count votes
        signals_df = df[signal_columns]
        long_votes = (signals_df == 1).sum(axis=1)
        short_votes = (signals_df == -1).sum(axis=1)

        # Ensemble signal: require min_agreement strategies to agree
        df["signal"] = 0
        df.loc[long_votes >= self.min_agreement, "signal"] = 1
        df.loc[short_votes >= self.min_agreement, "signal"] = -1

        # If both long and short meet threshold, go flat (conflicting signals)
        conflict = (long_votes >= self.min_agreement) & (short_votes >= self.min_agreement)
        df.loc[conflict, "signal"] = 0

        # Stop price: use the tightest (most conservative) stop among agreeing strategies
        df["stop_price"] = np.nan
        for i, row in df.iterrows():
            if row["signal"] == 1:
                stops = [row[sc] for sc, sig in zip(stop_columns, signal_columns)
                         if row[sig] == 1 and pd.notna(row[sc])]
                if stops:
                    df.at[i, "stop_price"] = max(stops)  # highest stop = tightest for longs
            elif row["signal"] == -1:
                stops = [row[sc] for sc, sig in zip(stop_columns, signal_columns)
                         if row[sig] == -1 and pd.notna(row[sc])]
                if stops:
                    df.at[i, "stop_price"] = min(stops)  # lowest stop = tightest for shorts

        # Confidence: average of agreeing strategies' confidence
        df["confidence"] = 0.0
        for i, row in df.iterrows():
            if row["signal"] != 0:
                confs = [row[cc] for cc, sig in zip(conf_columns, signal_columns)
                         if row[sig] == row["signal"] and pd.notna(row[cc])]
                if confs:
                    df.at[i, "confidence"] = np.mean(confs)

        # Clean up sub-strategy columns
        df.drop(columns=signal_columns + stop_columns + conf_columns, inplace=True)

        return df
