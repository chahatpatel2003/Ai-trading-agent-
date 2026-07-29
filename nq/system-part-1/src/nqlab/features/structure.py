"""Phase 13 market-structure / liquidity / displacement / FVG features (PIT-safe).

A new NQ-only signal layer: swing structure (BOS / CHoCH / MSS, HH-HL-LH-LL), resting
liquidity (equal highs/lows, sweeps + reclaims), displacement (impulse candles), fair
value gaps, and a premium/discount dealing-range model.

Point-in-time contract — the swing is the trap. A swing high at bar *i* is a fractal
(local max over +/-``swing_k`` bars); it is only *knowable* at bar ``i + swing_k`` once
the right side has printed. Every swing-derived column here is therefore exposed with a
``swing_k``-bar confirmation lag (centered fractal -> ``shift(swing_k)`` -> forward
fill), so a value at row *t* depends only on rows <= *t*. FVGs (3-bar) and displacement
(single bar) are knowable at the forming bar. Verified by ``test_structure_pit.py``: the
value at row *t* on the full series equals the value computed on the prefix ``[0..t]``.
"""

from __future__ import annotations

import polars as pl

STRUCTURE_FEATURE_NAMES: tuple[str, ...] = (
    # swing structure (confirmation-lagged)
    "last_swing_high",
    "last_swing_low",
    "prev_swing_high",
    "prev_swing_low",
    "higher_high",
    "higher_low",
    "lower_high",
    "lower_low",
    # break of structure / change of character / market-structure shift
    "struct_dir",
    "bos_up",
    "bos_dn",
    "choch_up",
    "choch_dn",
    "mss_up",
    "mss_dn",
    # liquidity
    "equal_highs",
    "equal_lows",
    "swept_high",
    "swept_low",
    "reclaim_up",
    "reclaim_dn",
    "dist_swing_high",
    "dist_swing_low",
    # displacement
    "displacement",
    "body_to_range",
    "range_expansion",
    # fair value gaps
    "fvg_bull",
    "fvg_bear",
    "in_bull_fvg",
    "in_bear_fvg",
    # premium / discount dealing range
    "range_mid",
    "in_discount",
    "in_premium",
)


def _swings(df: pl.DataFrame, k: int) -> pl.DataFrame:
    """Confirmation-lagged last/previous swing high & low (PIT-safe)."""
    hi, lo = pl.col("high"), pl.col("low")
    # Centered fractal is future-aware; expose only after k bars have confirmed it.
    frac_hi = hi == hi.rolling_max(window_size=2 * k + 1, center=True)
    frac_lo = lo == lo.rolling_min(window_size=2 * k + 1, center=True)
    df = df.with_columns(
        pl.when(frac_hi.shift(k)).then(hi.shift(k)).alias("_sh_conf"),
        pl.when(frac_lo.shift(k)).then(lo.shift(k)).alias("_sl_conf"),
    )
    df = df.with_columns(
        pl.col("_sh_conf").forward_fill().alias("last_swing_high"),
        pl.col("_sl_conf").forward_fill().alias("last_swing_low"),
    )
    sh, sl = pl.col("last_swing_high"), pl.col("last_swing_low")
    df = df.with_columns(
        pl.when(sh != sh.shift(1))
        .then(sh.shift(1))
        .otherwise(None)
        .forward_fill()
        .alias("prev_swing_high"),
        pl.when(sl != sl.shift(1))
        .then(sl.shift(1))
        .otherwise(None)
        .forward_fill()
        .alias("prev_swing_low"),
    )
    return df


def build_structure_features(
    frame: pl.DataFrame,
    *,
    swing_k: int = 2,
    ext_k: int = 5,
    eq_tol_atr: float = 0.15,
    disp_body: float = 0.6,
    disp_atr: float = 1.5,
    fvg_window: int = 30,
) -> pl.DataFrame:
    """Add Phase-13 structure/liquidity/displacement/FVG columns. Requires OHLC + ``atr``."""
    df = frame.sort("ts")
    df = _swings(df, swing_k)

    sh, sl = pl.col("last_swing_high"), pl.col("last_swing_low")
    psh, psl = pl.col("prev_swing_high"), pl.col("prev_swing_low")
    atr = pl.col("atr")
    close = pl.col("close")

    # Higher-high / lower-low classification of the latest confirmed swings.
    df = df.with_columns(
        (sh > psh).cast(pl.Int8).alias("higher_high"),
        (sl > psl).cast(pl.Int8).alias("higher_low"),
        (sh < psh).cast(pl.Int8).alias("lower_high"),
        (sl < psl).cast(pl.Int8).alias("lower_low"),
        ((sh - psh).abs() < eq_tol_atr * atr).cast(pl.Int8).alias("equal_highs"),
        ((sl - psl).abs() < eq_tol_atr * atr).cast(pl.Int8).alias("equal_lows"),
        (close - sh).alias("dist_swing_high"),
        (close - sl).alias("dist_swing_low"),
    )

    # Break of structure: close beyond the last confirmed swing. struct_dir carries the
    # most recent BOS direction; CHoCH/MSS = a BOS against the prior direction.
    bos_up = close > sh
    bos_dn = close < sl
    df = df.with_columns(
        bos_up.cast(pl.Int8).alias("bos_up"),
        bos_dn.cast(pl.Int8).alias("bos_dn"),
        pl.when(bos_up)
        .then(1)
        .when(bos_dn)
        .then(-1)
        .otherwise(None)
        .forward_fill()
        .fill_null(0)
        .cast(pl.Int8)
        .alias("struct_dir"),
    )
    prior_dir = pl.col("struct_dir").shift(1)
    df = df.with_columns(
        ((pl.col("bos_up") == 1) & (prior_dir == -1)).cast(pl.Int8).alias("choch_up"),
        ((pl.col("bos_dn") == 1) & (prior_dir == 1)).cast(pl.Int8).alias("choch_dn"),
    )

    # Displacement / impulse candle.
    rng = (pl.col("high") - pl.col("low")).clip(lower_bound=1e-9)
    body = (close - pl.col("open")).abs()
    avg_rng = rng.rolling_mean(window_size=20)
    df = df.with_columns(
        (body / rng).alias("body_to_range"),
        (rng > avg_rng).cast(pl.Int8).alias("range_expansion"),
        (((body / rng) >= disp_body) & (rng >= disp_atr * atr)).cast(pl.Int8).alias("displacement"),
    )
    # MSS = a CHoCH delivered with displacement (a structural shift with intent).
    df = df.with_columns(
        ((pl.col("choch_up") == 1) & (pl.col("displacement") == 1)).cast(pl.Int8).alias("mss_up"),
        ((pl.col("choch_dn") == 1) & (pl.col("displacement") == 1)).cast(pl.Int8).alias("mss_dn"),
    )

    # Liquidity sweep + reclaim of the prior-day extreme or last swing (wick beyond,
    # close back inside). Levels are PIT-safe (prior-day / confirmed swing).
    hi_level = (
        pl.max_horizontal(pl.col("prev_day_high"), sh) if "prev_day_high" in df.columns else sh
    )
    lo_level = pl.min_horizontal(pl.col("prev_day_low"), sl) if "prev_day_low" in df.columns else sl
    df = df.with_columns(
        ((pl.col("high") > hi_level) & (close < hi_level)).cast(pl.Int8).alias("swept_high"),
        ((pl.col("low") < lo_level) & (close > lo_level)).cast(pl.Int8).alias("swept_low"),
    )
    df = df.with_columns(
        pl.col("swept_low").alias("reclaim_up"),  # swept the lows then reclaimed -> bullish
        pl.col("swept_high").alias("reclaim_dn"),
    )

    # Fair value gaps (3-bar imbalance) at the forming bar; a forward-filled recent gap
    # gives the mitigation/entry zone.
    df = df.with_columns(
        (pl.col("low") > pl.col("high").shift(2)).cast(pl.Int8).alias("fvg_bull"),
        (pl.col("high") < pl.col("low").shift(2)).cast(pl.Int8).alias("fvg_bear"),
    )
    bull_top = pl.when(pl.col("fvg_bull") == 1).then(pl.col("low")).otherwise(None)
    bull_bot = pl.when(pl.col("fvg_bull") == 1).then(pl.col("high").shift(2)).otherwise(None)
    bear_bot = pl.when(pl.col("fvg_bear") == 1).then(pl.col("high")).otherwise(None)
    bear_top = pl.when(pl.col("fvg_bear") == 1).then(pl.col("low").shift(2)).otherwise(None)
    df = df.with_columns(
        bull_top.forward_fill(limit=fvg_window).alias("_bft"),
        bull_bot.forward_fill(limit=fvg_window).alias("_bfb"),
        bear_bot.forward_fill(limit=fvg_window).alias("_rbb"),
        bear_top.forward_fill(limit=fvg_window).alias("_rbt"),
    )
    df = df.with_columns(
        ((pl.col("low") <= pl.col("_bft")) & (close >= pl.col("_bfb")))
        .cast(pl.Int8)
        .alias("in_bull_fvg"),
        ((pl.col("high") >= pl.col("_rbb")) & (close <= pl.col("_rbt")))
        .cast(pl.Int8)
        .alias("in_bear_fvg"),
    )

    # Premium / discount within the confirmed dealing range.
    df = df.with_columns(((sh + sl) / 2.0).alias("range_mid"))
    df = df.with_columns(
        (close < pl.col("range_mid")).cast(pl.Int8).alias("in_discount"),
        (close > pl.col("range_mid")).cast(pl.Int8).alias("in_premium"),
    )

    drop = [c for c in ("_sh_conf", "_sl_conf", "_bft", "_bfb", "_rbb", "_rbt") if c in df.columns]
    return df.drop(drop).sort("ts")
