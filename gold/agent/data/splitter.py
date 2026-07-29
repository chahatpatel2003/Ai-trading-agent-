"""
Train / Validation / Test data splitting.
Strictly chronological to prevent lookahead bias.
"""
from dataclasses import dataclass

import pandas as pd

from logs.logger import get_logger

log = get_logger("data.splitter")


@dataclass
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    split_info: dict


def chronological_split(
    df: pd.DataFrame,
    train_pct: float = 0.60,
    val_pct: float = 0.20,
    test_pct: float = 0.20,
) -> DataSplit:
    """
    Split data chronologically — NO shuffling, NO random sampling.
    This prevents lookahead bias entirely.
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-9, "Splits must sum to 1.0"

    n = len(df)
    train_end = int(n * train_pct)
    val_end = int(n * (train_pct + val_pct))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    split_info = {
        "total_bars": n,
        "train_bars": len(train),
        "val_bars": len(val),
        "test_bars": len(test),
        "train_range": (str(train.index[0]), str(train.index[-1])),
        "val_range": (str(val.index[0]), str(val.index[-1])),
        "test_range": (str(test.index[0]), str(test.index[-1])),
    }

    log.info("data_split", **split_info)
    return DataSplit(train=train, validation=val, test=test, split_info=split_info)


def label_market_regime(df: pd.DataFrame, ma_period: int = 200) -> pd.DataFrame:
    """
    Label market regimes based on price relative to a long MA and volatility.
    Regimes: uptrend, downtrend, ranging, high_volatility
    """
    df = df.copy()
    df["ma_regime"] = df["close"].rolling(ma_period, min_periods=ma_period).mean()
    df["returns"] = df["close"].pct_change()
    df["volatility_20"] = df["returns"].rolling(20).std()
    vol_median = df["volatility_20"].median()

    conditions = []
    labels = []

    # High volatility regime (takes priority)
    high_vol = df["volatility_20"] > vol_median * 2
    conditions.append(high_vol)
    labels.append("high_volatility")

    # Uptrend
    uptrend = (~high_vol) & (df["close"] > df["ma_regime"])
    conditions.append(uptrend)
    labels.append("uptrend")

    # Downtrend
    downtrend = (~high_vol) & (df["close"] < df["ma_regime"])
    conditions.append(downtrend)
    labels.append("downtrend")

    import numpy as np
    df["regime"] = np.select(
        conditions, labels, default="ranging"
    )

    # Clean up helper columns
    df.drop(columns=["ma_regime", "returns", "volatility_20"], inplace=True)
    return df


def label_trading_session(df: pd.DataFrame) -> pd.DataFrame:
    """Label each bar with its trading session: asian, london, new_york, overlap."""
    df = df.copy()
    hour = df.index.hour

    sessions = []
    for h in hour:
        if 0 <= h < 8:
            sessions.append("asian")
        elif 8 <= h < 13:
            sessions.append("london")
        elif 13 <= h < 17:
            sessions.append("new_york")
        elif 17 <= h < 22:
            sessions.append("us_afternoon")
        else:
            sessions.append("asian")  # late night wraps into asian
    df["session"] = sessions
    return df
