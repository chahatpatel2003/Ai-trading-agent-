"""PIT-safe volume / market-profile features (Phase 30).

Builds prior-session, developing-session and multi-day-composite volume profiles from BARS (no
MBP1 needed → full history), and derives value-area behavior labels + normalized distances.

**Look-ahead safety is the whole point.** Three rules, enforced by construction and tested by
prefix-invariance (`tests/unit/test_phase30_integrity.py`):

1. *Prior session* profile (POC/VAH/VAL/HVN/LVN) for trading-day D uses ONLY the fully-completed RTH
   session of the previous trading day — known before D opens. Mapped via the previous day's full
   profile, referenced only on D's bars (all after D-1 closed).
2. *Developing session* profile at bar t uses ONLY the current RTH session's bars from open to t
   (a forward cumulative histogram reset at each RTH open). The value at t never depends on t+1.
3. *Composite* profiles use ONLY the last N completed prior RTH sessions.

A bar's final day POC/VAH/VAL is therefore NEVER used during that day; only the developing (so-far)
profile and completed prior/composite profiles are.

Profile construction from bars: each bar's volume is spread uniformly across the price bins its
[low, high] spans (a standard bar-based approximation when tick volume-at-price is unavailable). The
70% value area is grown from the POC bin by repeatedly absorbing the larger adjacent bin.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

import polars as pl

DEFAULT_BIN = 2.0  # price bin size in points (8 NQ ticks)
DEFAULT_VA_PCT = 0.70  # value-area volume fraction
DEFAULT_COMPOSITE = 5  # composite look-back in completed RTH sessions


@dataclass(frozen=True)
class Profile:
    poc: float
    vah: float
    val: float
    width: float
    hvn: tuple[float, ...]  # ascending HVN prices
    lvn: tuple[float, ...]  # ascending LVN prices


def _add_bar(hist: dict[int, float], lo: float, hi: float, vol: float, bin_size: float) -> None:
    """Spread one bar's volume uniformly across the price bins it spans."""
    if vol <= 0 or hi < lo:
        return
    b0 = math.floor(lo / bin_size)
    b1 = math.floor(hi / bin_size)
    n = b1 - b0 + 1
    share = vol / n
    for b in range(b0, b1 + 1):
        hist[b] = hist.get(b, 0.0) + share


def _profile_from_hist(hist: dict[int, float], bin_size: float, va_pct: float) -> Profile | None:
    """POC / 70%-value-area / HVN / LVN from a {bin_index: volume} histogram."""
    if not hist:
        return None
    total = 0.0
    poc_idx = None
    poc_vol = -1.0
    for b, v in hist.items():
        total += v
        if v > poc_vol:
            poc_vol = v
            poc_idx = b
    if poc_idx is None or total <= 0:
        return None
    # Grow the value area from the POC bin by absorbing the larger adjacent bin until >= va_pct.
    target = va_pct * total
    acc = hist[poc_idx]
    up, dn = poc_idx + 1, poc_idx - 1
    lo_idx = hi_idx = poc_idx
    while acc < target and (up in hist or dn in hist):
        vu = hist.get(up, -1.0)
        vd = hist.get(dn, -1.0)
        if vu < 0 and vd < 0:
            break
        if vu >= vd:
            acc += max(vu, 0.0)
            hi_idx = up
            up += 1
        else:
            acc += max(vd, 0.0)
            lo_idx = dn
            dn -= 1
    poc = (poc_idx + 0.5) * bin_size
    vah = (hi_idx + 1.0) * bin_size
    val = lo_idx * bin_size
    # HVN / LVN: local peaks / volume gaps over the occupied range.
    idxs = sorted(hist)
    mean_v = total / len(idxs)
    hvn: list[float] = []
    lvn: list[float] = []
    for i in range(1, len(idxs) - 1):
        b = idxs[i]
        if idxs[i - 1] != b - 1 or idxs[i + 1] != b + 1:
            continue  # need both immediate neighbors present to call a peak/valley
        v, vl, vr = hist[b], hist[b - 1], hist[b + 1]
        price = (b + 0.5) * bin_size
        if v > vl and v > vr and v > mean_v:
            hvn.append(price)
        elif v < vl and v < vr and v < 0.5 * mean_v:
            lvn.append(price)
    return Profile(poc, vah, val, vah - val, tuple(hvn), tuple(lvn))


def _profile_from_bars(
    los: list[float], his: list[float], vols: list[float], bin_size: float, va_pct: float
) -> Profile | None:
    hist: dict[int, float] = {}
    for lo, hi, v in zip(los, his, vols, strict=True):
        _add_bar(hist, lo, hi, v, bin_size)
    return _profile_from_hist(hist, bin_size, va_pct)


def _nearest(levels: tuple[float, ...], price: float) -> tuple[float | None, float | None]:
    """Nearest level strictly below and strictly above ``price`` from an ascending tuple."""
    if not levels:
        return None, None
    i = bisect.bisect_left(levels, price)
    below = levels[i - 1] if i > 0 else None
    above = levels[i] if i < len(levels) else None
    if above is not None and above == price:  # exact hit -> treat as the 'above' boundary
        above = levels[i + 1] if i + 1 < len(levels) else None
    return below, above


def _day_profiles(frame: pl.DataFrame, bin_size: float, va_pct: float) -> dict[Any, Profile]:
    """Completed RTH profile for every trading day (uses only that day's is_rth bars)."""
    rth = frame.filter(pl.col("is_rth"))
    out: dict[Any, Profile] = {}
    for day, sub in rth.group_by("trading_day", maintain_order=True):
        key = day[0] if isinstance(day, tuple) else day
        prof = _profile_from_bars(
            sub["low"].to_list(), sub["high"].to_list(), sub["volume"].to_list(), bin_size, va_pct
        )
        if prof is not None:
            out[key] = prof
    return out


def add_volume_profile(
    frame: pl.DataFrame,
    *,
    bin_size: float = DEFAULT_BIN,
    va_pct: float = DEFAULT_VA_PCT,
    composite_days: int = DEFAULT_COMPOSITE,
) -> pl.DataFrame:
    """Attach PIT-safe volume-profile columns. ``frame`` must be sorted by ``ts`` and have the
    calendar columns (``trading_day``, ``is_rth``, ``minute_of_day``) + OHLCV + ``atr``."""
    frame = frame.sort("ts")
    day_prof = _day_profiles(frame, bin_size, va_pct)
    days_sorted = sorted(day_prof)
    prev_day = {d: days_sorted[i - 1] for i, d in enumerate(days_sorted) if i > 0}
    # Composite over the last ``composite_days`` COMPLETED prior sessions (per trading day present).
    all_days = list(frame["trading_day"].unique(maintain_order=False))
    all_days_sorted = sorted(all_days)
    comp: dict[Any, Profile | None] = {}
    profiled_before: list[Any] = []  # days (in order) that have a completed profile
    for d in all_days_sorted:
        prior_profiled = [x for x in profiled_before if x < d][-composite_days:]
        if prior_profiled:
            hist: dict[int, float] = {}
            # Rebuild a merged histogram from those sessions' bars.
            sub = frame.filter(pl.col("is_rth") & pl.col("trading_day").is_in(prior_profiled))
            for lo, hi, v in zip(
                sub["low"].to_list(), sub["high"].to_list(), sub["volume"].to_list(), strict=True
            ):
                _add_bar(hist, lo, hi, v, bin_size)
            comp[d] = _profile_from_hist(hist, bin_size, va_pct)
        else:
            comp[d] = None
        if d in day_prof:
            profiled_before.append(d)

    n = frame.height
    days = frame["trading_day"].to_list()
    rthf = frame["is_rth"].to_list()
    los = frame["low"].to_list()
    his = frame["high"].to_list()
    cls = frame["close"].to_list()
    ops = frame["open"].to_list()
    vols = frame["volume"].to_list()

    cols: dict[str, list[float | None]] = {
        c: [None] * n
        for c in (
            "vp_prior_poc",
            "vp_prior_vah",
            "vp_prior_val",
            "vp_prior_vaw",
            "vp_prior_hvn_up",
            "vp_prior_hvn_dn",
            "vp_prior_lvn_up",
            "vp_prior_lvn_dn",
            "vp_dev_poc",
            "vp_dev_vah",
            "vp_dev_val",
            "vp_dev_vaw",
            "vp_comp_poc",
            "vp_comp_vah",
            "vp_comp_val",
        )
    }
    open_loc: list[float | None] = [None] * n  # -1 below prior VA / 0 inside / +1 above (per day)

    cur_day = None
    dev: dict[int, float] = {}
    prior: Profile | None = None
    compp: Profile | None = None
    day_open_loc: float | None = None
    for i in range(n):
        d = days[i]
        if d != cur_day:
            cur_day = d
            dev = {}
            prior = day_prof.get(prev_day.get(d))
            compp = comp.get(d)
            day_open_loc = None
        c = cls[i]
        if prior is not None:
            cols["vp_prior_poc"][i] = prior.poc
            cols["vp_prior_vah"][i] = prior.vah
            cols["vp_prior_val"][i] = prior.val
            cols["vp_prior_vaw"][i] = prior.width
            hdn, hup = _nearest(prior.hvn, c)
            ldn, lup = _nearest(prior.lvn, c)
            cols["vp_prior_hvn_up"][i] = hup
            cols["vp_prior_hvn_dn"][i] = hdn
            cols["vp_prior_lvn_up"][i] = lup
            cols["vp_prior_lvn_dn"][i] = ldn
            if day_open_loc is None and rthf[i]:
                o = ops[i]
                day_open_loc = 1.0 if o > prior.vah else (-1.0 if o < prior.val else 0.0)
            open_loc[i] = day_open_loc
        if compp is not None:
            cols["vp_comp_poc"][i] = compp.poc
            cols["vp_comp_vah"][i] = compp.vah
            cols["vp_comp_val"][i] = compp.val
        if rthf[i]:
            _add_bar(dev, los[i], his[i], vols[i], bin_size)
            dp = _profile_from_hist(dev, bin_size, va_pct)
            if dp is not None:
                cols["vp_dev_poc"][i] = dp.poc
                cols["vp_dev_vah"][i] = dp.vah
                cols["vp_dev_val"][i] = dp.val
                cols["vp_dev_vaw"][i] = dp.width

    frame = frame.with_columns(
        [pl.Series(name, vals, dtype=pl.Float64) for name, vals in cols.items()]
        + [pl.Series("vp_open_loc", open_loc, dtype=pl.Float64)]
    )
    return _derive_vp(frame)


def _crossed_up(level: str) -> pl.Expr:
    return (pl.col("close") > pl.col(level)) & (pl.col("close").shift(1) <= pl.col(level).shift(1))


def _crossed_dn(level: str) -> pl.Expr:
    return (pl.col("close") < pl.col(level)) & (pl.col("close").shift(1) >= pl.col(level).shift(1))


def _derive_vp(frame: pl.DataFrame) -> pl.DataFrame:
    """Vectorized distances, position-vs-value, value-migration and behavior event flags."""
    atr = pl.col("atr")
    close = pl.col("close")
    poc, vah, val = pl.col("vp_prior_poc"), pl.col("vp_prior_vah"), pl.col("vp_prior_val")
    frame = frame.with_columns(
        (close - poc).alias("vp_d_poc"),
        (close - vah).alias("vp_d_vah"),
        (close - val).alias("vp_d_val"),
        ((close - poc) / atr).alias("vp_d_poc_atr"),
        ((pl.col("vp_prior_hvn_up") - close) / atr).alias("vp_dist_hvn_up_atr"),
        ((close - pl.col("vp_prior_hvn_dn")) / atr).alias("vp_dist_hvn_dn_atr"),
        ((pl.col("vp_prior_lvn_up") - close) / atr).alias("vp_dist_lvn_up_atr"),
        ((close - pl.col("vp_prior_lvn_dn")) / atr).alias("vp_dist_lvn_dn_atr"),
        pl.when(close > vah).then(1).when(close < val).then(-1).otherwise(0).alias("vp_pos"),
        (pl.col("vp_dev_poc") - pl.col("vp_dev_poc").shift(6)).alias("vp_dev_poc_shift"),
    )
    return frame.with_columns(
        # POC reclaim / reject
        _crossed_up("vp_prior_poc").alias("vp_poc_reclaim_up"),
        _crossed_dn("vp_prior_poc").alias("vp_poc_reclaim_dn"),
        # Value-area breakout (acceptance) — close crosses the edge outward
        _crossed_up("vp_prior_vah").alias("vp_vah_break_up"),
        _crossed_dn("vp_prior_val").alias("vp_val_break_dn"),
        # Value-area rejection — wick beyond the edge, close back across (reversion)
        ((pl.col("low") <= val) & (close > val)).alias("vp_val_reclaim_up"),
        ((pl.col("high") >= vah) & (close < vah)).alias("vp_vah_reject_dn"),
        # LVN rejection — traded into nearest LVN and closed back away from it
        ((pl.col("low") <= pl.col("vp_prior_lvn_dn")) & (close > pl.col("vp_prior_lvn_dn"))).alias(
            "vp_lvn_reject_up"
        ),
        ((pl.col("high") >= pl.col("vp_prior_lvn_up")) & (close < pl.col("vp_prior_lvn_up"))).alias(
            "vp_lvn_reject_dn"
        ),
        # LVN breakout — close pushes through the nearest LVN (vacuum)
        (
            (close > pl.col("vp_prior_lvn_up"))
            & (close.shift(1) <= pl.col("vp_prior_lvn_up").shift(1))
        ).alias("vp_lvn_break_up"),
        (
            (close < pl.col("vp_prior_lvn_dn"))
            & (close.shift(1) >= pl.col("vp_prior_lvn_dn").shift(1))
        ).alias("vp_lvn_break_dn"),
    )


VP_COLUMNS: tuple[str, ...] = (
    "vp_prior_poc",
    "vp_prior_vah",
    "vp_prior_val",
    "vp_prior_vaw",
    "vp_prior_hvn_up",
    "vp_prior_hvn_dn",
    "vp_prior_lvn_up",
    "vp_prior_lvn_dn",
    "vp_dev_poc",
    "vp_dev_vah",
    "vp_dev_val",
    "vp_dev_vaw",
    "vp_comp_poc",
    "vp_comp_vah",
    "vp_comp_val",
    "vp_open_loc",
    "vp_d_poc",
    "vp_d_vah",
    "vp_d_val",
    "vp_d_poc_atr",
    "vp_dist_hvn_up_atr",
    "vp_dist_hvn_dn_atr",
    "vp_dist_lvn_up_atr",
    "vp_dist_lvn_dn_atr",
    "vp_pos",
    "vp_dev_poc_shift",
    "vp_poc_reclaim_up",
    "vp_poc_reclaim_dn",
    "vp_vah_break_up",
    "vp_val_break_dn",
    "vp_val_reclaim_up",
    "vp_vah_reject_dn",
    "vp_lvn_reject_up",
    "vp_lvn_reject_dn",
    "vp_lvn_break_up",
    "vp_lvn_break_dn",
)
