"""Phase 25 — daily bias + HTF-trend alignment + CISD/IFVG confirmations (PIT-safe).

Daily bias and HTF trend are MANDATORY filters for the Phase-25 model: a trade is only allowed when
the daily bias and the higher-timeframe trend agree. We expose two explainable, PIT-safe daily-bias
definitions (tested separately), the HTF trend (1H + 4H agreement), the alignment flags, and the
LTF "change in state of delivery" (CISD) / inverted-FVG (IFVG) reclaim confirmations.

Everything uses confirmed-and-past data only:
- daily bias from PRIOR sessions' structure (the current session's globex open is known pre-NY-AM),
- HTF trend from the existing PIT-safe ``trend_1h_bull`` / ``trend_4h_bull``,
- CISD/IFVG from current-and-past bars (displacement closing through a prior bar; reclaim of a gap).
"""

from __future__ import annotations

import polars as pl


def add_daily_bias(frame: pl.DataFrame) -> pl.DataFrame:
    """Add daily bias (A: structure; B: gap+prior-day) + HTF trend + alignment + CISD/IFVG."""
    # Daily OHLC per trading day, then prior-day structure (shifted -> PIT-safe).
    daily = (
        frame.group_by("trading_day")
        .agg(
            pl.col("open").first().alias("d_open"),
            pl.col("high").max().alias("d_high"),
            pl.col("low").min().alias("d_low"),
            pl.col("close").last().alias("d_close"),
        )
        .sort("trading_day")
    )
    ph1, ph2 = pl.col("d_high").shift(1), pl.col("d_high").shift(2)
    pl1, pl2 = pl.col("d_low").shift(1), pl.col("d_low").shift(2)
    prev_close = pl.col("d_close").shift(1)
    prev_dir = (pl.col("d_close").shift(1) - pl.col("d_open").shift(1)).sign()
    daily = daily.with_columns(
        # A — daily structure: prior day HH+HL (bull) vs LH+LL (bear), else unclear.
        pl.when((ph1 > ph2) & (pl1 > pl2)).then(1)
        .when((ph1 < ph2) & (pl1 < pl2)).then(-1)
        .otherwise(0).alias("bias_a"),
        # B — gap + prior-day: open vs prior close, confirmed by prior-day candle direction.
        pl.when((pl.col("d_open") > prev_close) & (prev_dir > 0)).then(1)
        .when((pl.col("d_open") < prev_close) & (prev_dir < 0)).then(-1)
        .otherwise(0).alias("bias_b"),
    )
    f = frame.join(daily.select("trading_day", "bias_a", "bias_b"), on="trading_day", how="left")

    # HTF trend = 1H and 4H agreement; alignment per bias definition.
    htf_bull = (pl.col("trend_4h_bull") == 1) & (pl.col("trend_1h_bull") == 1)
    htf_bear = (pl.col("trend_4h_bull") == 0) & (pl.col("trend_1h_bull") == 0)
    # CISD: a displacement candle that closes through the prior bar's extreme (state shift).
    cisd_up = (pl.col("displacement") == 1) & (pl.col("close") > pl.col("high").shift(1))
    cisd_dn = (pl.col("displacement") == 1) & (pl.col("close") < pl.col("low").shift(1))
    # IFVG reclaim (approx): a recent bear/bull FVG that price has now inverted by closing back
    # through it (reclaim into the opposite-side gap).
    ifvg_bull = (pl.col("in_bull_fvg") == 1) & (
        pl.col("fvg_bear").rolling_max(window_size=6).over("trading_day") == 1
    )
    ifvg_bear = (pl.col("in_bear_fvg") == 1) & (
        pl.col("fvg_bull").rolling_max(window_size=6).over("trading_day") == 1
    )
    return f.with_columns(
        htf_bull.cast(pl.Int8).alias("htf_bull"),
        htf_bear.cast(pl.Int8).alias("htf_bear"),
        ((pl.col("bias_a") == 1) & htf_bull).cast(pl.Int8).alias("aligned_long_a"),
        ((pl.col("bias_a") == -1) & htf_bear).cast(pl.Int8).alias("aligned_short_a"),
        ((pl.col("bias_b") == 1) & htf_bull).cast(pl.Int8).alias("aligned_long_b"),
        ((pl.col("bias_b") == -1) & htf_bear).cast(pl.Int8).alias("aligned_short_b"),
        cisd_up.cast(pl.Int8).alias("cisd_up"),
        cisd_dn.cast(pl.Int8).alias("cisd_dn"),
        ifvg_bull.cast(pl.Int8).alias("ifvg_bull"),
        ifvg_bear.cast(pl.Int8).alias("ifvg_bear"),
    )
