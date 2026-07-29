"""Classic technical indicators (Phase 20) — vectorized, PIT-safe Polars helpers.

Each ``add_*`` stages the columns an indicator family needs onto a bar frame using only
current-and-past data (rolling / Wilder-EWM / shifted channels), so a signal at bar *t* never
reads bar *t+1*. Reuses the base feature columns where they already exist (``atr``, ``rsi``,
``vwap_session``). Intermediate columns are prefixed ``_`` so they never collide with signals.
"""

from __future__ import annotations

import polars as pl


def ema(span: int, col: str = "close") -> pl.Expr:
    return pl.col(col).ewm_mean(span=span, adjust=False)


def sma(window: int, col: str = "close") -> pl.Expr:
    return pl.col(col).rolling_mean(window_size=window)


def add_macd(
    frame: pl.DataFrame, *, fast: int = 12, slow: int = 26, signal: int = 9
) -> pl.DataFrame:
    """MACD line / signal / histogram."""
    f = frame.with_columns((ema(fast) - ema(slow)).alias("_macd"))
    return f.with_columns(
        pl.col("_macd").ewm_mean(span=signal, adjust=False).alias("_macd_sig"),
    ).with_columns((pl.col("_macd") - pl.col("_macd_sig")).alias("_macd_hist"))


def add_stoch(frame: pl.DataFrame, *, k: int = 14, d: int = 3) -> pl.DataFrame:
    """Stochastic %K / %D."""
    lo = pl.col("low").rolling_min(window_size=k)
    hi = pl.col("high").rolling_max(window_size=k)
    f = frame.with_columns(
        (100.0 * (pl.col("close") - lo) / (hi - lo)).alias("_stoch_k"),
    )
    return f.with_columns(pl.col("_stoch_k").rolling_mean(window_size=d).alias("_stoch_d"))


def add_bollinger(frame: pl.DataFrame, *, n: int = 20, k: float = 2.0) -> pl.DataFrame:
    """Bollinger mid / upper / lower / bandwidth."""
    mid = sma(n)
    sd = pl.col("close").rolling_std(window_size=n)
    return frame.with_columns(
        mid.alias("_bb_mid"),
        (mid + k * sd).alias("_bb_up"),
        (mid - k * sd).alias("_bb_dn"),
        (2.0 * k * sd / mid).alias("_bb_bw"),
    )


def add_keltner(frame: pl.DataFrame, *, n: int = 20, k: float = 1.5) -> pl.DataFrame:
    """Keltner mid (EMA) / upper / lower using ATR."""
    mid = ema(n)
    return frame.with_columns(
        mid.alias("_kc_mid"),
        (mid + k * pl.col("atr")).alias("_kc_up"),
        (mid - k * pl.col("atr")).alias("_kc_dn"),
    )


def add_donchian(frame: pl.DataFrame, *, n: int = 20) -> pl.DataFrame:
    """Donchian channel from the PRIOR n bars (shifted -> no lookahead)."""
    return frame.with_columns(
        pl.col("high").rolling_max(window_size=n).shift(1).alias("_dc_up"),
        pl.col("low").rolling_min(window_size=n).shift(1).alias("_dc_dn"),
    )


def add_adx(frame: pl.DataFrame, *, n: int = 14) -> pl.DataFrame:
    """Wilder ADX with +DI / -DI (EWM alpha = 1/n)."""
    pc = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pc).abs(),
        (pl.col("low") - pc).abs(),
    )
    up = pl.col("high") - pl.col("high").shift(1)
    dn = pl.col("low").shift(1) - pl.col("low")
    pdm = pl.when((up > dn) & (up > 0)).then(up).otherwise(0.0)
    mdm = pl.when((dn > up) & (dn > 0)).then(dn).otherwise(0.0)
    a = 1.0 / n
    f = frame.with_columns(tr.alias("_tr"), pdm.alias("_pdm"), mdm.alias("_mdm")).with_columns(
        pl.col("_tr").ewm_mean(alpha=a, adjust=False).alias("_atrw"),
        pl.col("_pdm").ewm_mean(alpha=a, adjust=False).alias("_pdmw"),
        pl.col("_mdm").ewm_mean(alpha=a, adjust=False).alias("_mdmw"),
    ).with_columns(
        # Guard the divisions: a 0/0 would become NaN and poison the whole EWM chain downstream.
        pl.when(pl.col("_atrw") > 0).then(100.0 * pl.col("_pdmw") / pl.col("_atrw"))
        .otherwise(0.0).alias("plus_di"),
        pl.when(pl.col("_atrw") > 0).then(100.0 * pl.col("_mdmw") / pl.col("_atrw"))
        .otherwise(0.0).alias("minus_di"),
    )
    di_sum = pl.col("plus_di") + pl.col("minus_di")
    dx = pl.when(di_sum > 0).then(
        100.0 * (pl.col("plus_di") - pl.col("minus_di")).abs() / di_sum
    ).otherwise(0.0)
    return f.with_columns(dx.alias("_dx")).with_columns(
        pl.col("_dx").ewm_mean(alpha=a, adjust=False).alias("adx")
    )
