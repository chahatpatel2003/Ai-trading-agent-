"""Strict mechanical implementation of the user's ICT-style displacement
strategy (manual-discretionary hypothesis, formalized verbatim).

RULES AS WRITTEN
  - Bias: locked at 08:44:59 CT (= 9:44 ET) from the FIRST 15m RTH candle
    (08:30-08:45 CT): close > open -> LONG day, close < open -> SHORT day,
    doji -> no trades. No counter-bias trades ever.
  - Entry TF: 3-minute bars (resampled causally from the gated 1m bars).
  - Displacement candle (a-priori mechanical definition, DISCLOSED — the
    term is discretionary in the source): body >= 70% of range AND range >=
    1.5 x ATR20(3m), colored in the bias direction. Two adjacent
    displacement candles merge into ONE leg.
  - Marking: bullish leg = BODY LOW -> WICK HIGH; bearish = BODY HIGH ->
    WICK LOW (never the full candle).
  - Levels (R = marked range): bullish entry = low + 0.5R (limit), stop =
    low - 0.5R (fib 1.5), target = high + 0.5R (fib -0.5). Bearish mirror.
    Risk = 1.0R, reward = 1.0R.
  - Order life: placed at leg completion (first non-continuation 3m close);
    cancelled if target OR stop level trades before the entry fills; expires
    at the 14:30 CT entry cutoff. One position at a time. Forced flat at the
    last 3m close before the 15:10 CT deadline. Both-in-bar -> stop first.
  - Risk: DLL $1,000/day; max 3 trades/day => $333 cap/trade; day stop at
    -$667 realized (cap + realized can never exceed DLL). NQ = 1 contract
    (trade SKIPPED and counted if 1-contract risk > cap); MNQ qty =
    floor(cap / (R_pts x $2)), capped at 30 micros (= 3 mini-equivalents,
    inside all account limits).

POI VARIANT (separately reported filter, same trades otherwise): the leg's
ORIGIN price must sit inside an ACTIVE bias-aligned 15m or 1h POI — FVG
(3-candle gap, active until filled), OB (last opposite-body candle before an
impulse bar >= 1.5 x ATR14 of that TF), or confirmed swing high/low (2-2
fractal, usable only 2 completed candles later). All POI structures are
built strictly from completed HTF candles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import sessions as S
from ..config import Config
from ..sim.costs import CostModel
from .backtest import Trade

BODY_MIN = 0.70          # displacement: body share of range
RANGE_ATR = 1.5          # displacement: range vs ATR20(3m)
MAX_TRADES = 3
DLL = 1000.0
CAP = DLL / MAX_TRADES   # per-trade risk cap
DAY_STOP = DLL - CAP     # realized day stop
BIAS_LOCK_CT = time(8, 45)      # first 15m candle 08:30-08:45 CT (9:44 ET)
ENTRY_CUTOFF_CT = time(14, 30)
IMPULSE_ATR = 1.5        # OB impulse definition on HTF
MAX_MICROS = 30


def resample(bars1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Causal OHLC resample within each session (bar stamped at open; the
    bar is complete only at open + `minutes`)."""
    g = bars1m.groupby([bars1m["session"],
                        bars1m.index.floor(f"{minutes}min")])
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
    })
    out.index.names = ["session", "bar_open_utc"]
    return out.reset_index(level=0)


@dataclass
class Failures:
    doji_days: int = 0
    no_bias_bar_days: int = 0
    cancelled_stop_first: int = 0
    cancelled_target_first: int = 0
    expired_unfilled: int = 0
    unsizeable_nq: int = 0
    day_stop_hits: int = 0
    max_trades_hits: int = 0
    poi_rejected_legs: int = 0

    def as_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


def _ct(idx, cfg) -> pd.Series:
    tz = cfg.sessions.timezone
    return pd.Series([t.time() for t in idx.tz_convert(tz)], index=idx)


# --------------------------------------------------------------------------- #
# POI structures (HTF, causal)
# --------------------------------------------------------------------------- #

def build_pois(htf: pd.DataFrame, tfmin: int) -> List[Dict]:
    """POI list from completed HTF candles: each with kind, side (+1 zone
    supports longs / -1 shorts), lo, hi, active_from (UTC instant the POI is
    USABLE — always a completed-candle boundary), and for FVG a fill rule."""
    pois: List[Dict] = []
    idx = htf.index
    o = htf["open"].to_numpy(); h = htf["high"].to_numpy()
    l = htf["low"].to_numpy(); c = htf["close"].to_numpy()
    step = pd.Timedelta(minutes=tfmin)
    tr = np.maximum(h - l, np.abs(h - np.r_[c[0], c[:-1]]))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    for i in range(2, len(htf)):
        done = idx[i] + step                       # candle i completed
        # FVG
        if l[i] > h[i - 2]:
            pois.append({"kind": "fvg", "side": +1, "lo": h[i - 2],
                         "hi": l[i], "from": done})
        if h[i] < l[i - 2]:
            pois.append({"kind": "fvg", "side": -1, "lo": h[i],
                         "hi": l[i - 2], "from": done})
        # OB: last opposite body before an impulse candle
        if np.isfinite(atr[i]) and (h[i] - l[i]) >= IMPULSE_ATR * atr[i]:
            if c[i] > o[i] and c[i - 1] < o[i - 1]:
                pois.append({"kind": "ob", "side": +1,
                             "lo": min(o[i - 1], c[i - 1]),
                             "hi": max(o[i - 1], c[i - 1]), "from": done})
            if c[i] < o[i] and c[i - 1] > o[i - 1]:
                pois.append({"kind": "ob", "side": -1,
                             "lo": min(o[i - 1], c[i - 1]),
                             "hi": max(o[i - 1], c[i - 1]), "from": done})
        # swings: STRICT 2-2 fractal at i-2 (ties are not swings), confirmed
        # only once candle i completes
        j = i - 2
        if j >= 2:
            if h[j] > max(h[j - 2:j]) and h[j] > max(h[j + 1:i + 1]):
                pois.append({"kind": "swing", "side": -1, "lo": h[j] - 1e-9,
                             "hi": h[j] + 1e-9, "from": done})
            if l[j] < min(l[j - 2:j]) and l[j] < min(l[j + 1:i + 1]):
                pois.append({"kind": "swing", "side": +1, "lo": l[j] - 1e-9,
                             "hi": l[j] + 1e-9, "from": done})
    return pois


def poi_hit(pois: List[Dict], side: int, price: float,
            at: pd.Timestamp, tol: float) -> bool:
    for p in pois:
        if p["side"] == side and p["from"] <= at \
                and p["lo"] - tol <= price <= p["hi"] + tol:
            return True
    return False


# --------------------------------------------------------------------------- #
# strict backtest
# --------------------------------------------------------------------------- #

TAP_WINDOW_MIN = 30       # displacement must start within this of a POI tap
POI_SESS_15M = 6          # POI freshness: 15m zones from last N sessions
POI_SESS_1H = 15


@dataclass
class PoiFailures(Failures):
    taps_total: int = 0
    legs_no_recent_tap: int = 0


def run_ict_poi_backtest(bars1m: pd.DataFrame, cfg: Config, instrument: str,
                         ltf_minutes: int, exit_model: str = "fixed",
                         target_r: float = 1.0
                         ) -> Tuple[List[Trade], PoiFailures, Dict]:
    """HTF-POI-tap -> LTF-displacement hypothesis (pre-registered v2).

    Sequence per session: bias locked from the first 15m candle; a
    bias-aligned 15m/1h POI (FVG/OB/swing, causal `from` times, freshness-
    limited) must be TAPPED (LTF bar range intersects the zone; knowledge at
    bar close); the next displacement leg on the entry LTF must START within
    TAP_WINDOW_MIN of that tap. Marking/levels/order-life/risk identical to
    the v1 registration. POI type of the triggering tap is recorded."""
    cm = CostModel(cfg)
    pv = cfg.instruments[instrument].point_value
    fail = PoiFailures()
    trades: List[Trade] = []
    bl = resample(bars1m, ltf_minutes)
    b15 = resample(bars1m, 15)
    b60 = resample(bars1m, 60)
    ctl = _ct(bl.index, cfg)
    poi_counts: Dict[str, int] = {}
    sess_list = sorted(set(bl["session"]))
    sess_pos = {s: k for k, s in enumerate(sess_list)}

    for sess, dayl in bl.groupby("session", sort=True):
        tt = ctl.loc[dayl.index]
        d15 = b15[b15["session"] == sess]
        t15 = _ct(d15.index, cfg)
        mask = t15 == time(8, 30)
        if not mask.any():
            fail.no_bias_bar_days += 1
            continue
        row = d15[mask.values].iloc[0]
        if row["close"] > row["open"]:
            bias = +1
        elif row["close"] < row["open"]:
            bias = -1
        else:
            fail.doji_days += 1
            continue

        k0 = sess_pos[sess]
        lo15 = sess_list[max(0, k0 - POI_SESS_15M)]
        lo60 = sess_list[max(0, k0 - POI_SESS_1H)]
        pois = [dict(p, tf=15) for p in build_pois(
            b15[(b15["session"] >= lo15) & (b15["session"] <= sess)], 15)]
        pois += [dict(p, tf=60) for p in build_pois(
            b60[(b60["session"] >= lo60) & (b60["session"] <= sess)], 60)]
        pois = [p for p in pois if p["side"] == bias]
        tick = cfg.instruments["NQ"].tick_size
        tol = 4 * tick

        day = dayl[(tt >= BIAS_LOCK_CT)]
        dtt = ctl.loc[day.index]
        o = day["open"].to_numpy(); h = day["high"].to_numpy()
        l = day["low"].to_numpy(); c = day["close"].to_numpy()
        n = len(day)
        tr = np.maximum(h - l, np.abs(h - np.r_[c[0], c[:-1]]))
        atr = pd.Series(tr).rolling(20).mean().to_numpy()
        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-12)
        disp = ((body / rng >= BODY_MIN) & (rng >= RANGE_ATR * atr)
                & ((c > o) if bias > 0 else (c < o)))
        flat_deadline = S.eod_flatten_utc(sess, cfg.sessions)
        step = pd.Timedelta(minutes=ltf_minutes)

        # precompute per-bar tap (any active bias-aligned POI intersected)
        tap_kind = [None] * n
        for k in range(n):
            bar_close = day.index[k] + step
            for p in pois:
                if p["from"] <= day.index[k] \
                        and l[k] <= p["hi"] + tol and h[k] >= p["lo"] - tol:
                    tap_kind[k] = f"{p['kind']}{p['tf']}"
                    break
        fail.taps_total += sum(1 for x in tap_kind if x)

        day_net = 0.0
        n_trades = 0
        last_tap_close: Optional[pd.Timestamp] = None
        last_tap_k: Optional[str] = None
        i = 20
        while i < n - 1:
            if tap_kind[i]:
                last_tap_close = day.index[i] + step
                last_tap_k = tap_kind[i]
            if not disp[i]:
                i += 1
                continue
            j = i
            if j + 1 < n and disp[j + 1]:
                j = j + 1
            leg_close = day.index[j] + step
            # tap gate: leg START within window of the latest tap
            tap_ok = (last_tap_close is not None
                      and day.index[i] + step - last_tap_close
                      <= pd.Timedelta(minutes=TAP_WINDOW_MIN))
            if not tap_ok:
                fail.legs_no_recent_tap += 1
                i = j + 1
                continue
            if bias > 0:
                lo = min(np.minimum(o, c)[i:j + 1])
                hi = max(h[i:j + 1])
            else:
                hi = max(np.maximum(o, c)[i:j + 1])
                lo = min(l[i:j + 1])
            R = hi - lo
            if R <= 0:
                i = j + 1
                continue
            if bias > 0:
                entry_px, stop_px = lo + .5 * R, lo - .5 * R
            else:
                entry_px, stop_px = hi - .5 * R, hi + .5 * R
            # exit geometry (v3): riskd = |entry - stop| = R; fixed target at
            # entry + k*riskd; p1r2 = half at +1R + runner to +2R (stop
            # unchanged); be1 = full to +2R, stop -> entry the bar AFTER +1R
            riskd = R
            model = exit_model
            if model == "fixed":
                k_tgt = target_r
            elif model in ("p1r2", "be1"):
                k_tgt = 2.0
            else:
                raise ValueError(f"unknown exit_model {exit_model!r}")
            tgt_px = entry_px + bias * k_tgt * riskd
            one_r_px = entry_px + bias * riskd
            risk_per_ct = R * pv
            if instrument == "NQ":
                qty = 1
                if risk_per_ct > CAP:
                    fail.unsizeable_nq += 1
                    i = j + 1
                    continue
            else:
                qty = int(min(MAX_MICROS, CAP // risk_per_ct))
                if qty < 1:
                    i = j + 1
                    continue
            if model == "p1r2" and qty < 2:
                model = "be1"                     # disclosed degeneration
            if n_trades >= MAX_TRADES:
                fail.max_trades_hits += 1
                break
            if day_net <= -DAY_STOP:
                fail.day_stop_hits += 1
                break
            filled = None
            be_armed = False
            partial_done = False
            q1 = qty // 2                          # p1r2 partial leg
            q2 = qty - q1
            parts: List[Tuple[int, float, str]] = []   # (qty, px, order type)
            exit_k = None
            reason = None
            k = j + 1
            while k < n:
                bar_close_time = day.index[k] + step
                if tap_kind[k]:                    # taps keep refreshing
                    last_tap_close = day.index[k] + step
                    last_tap_k = tap_kind[k]
                if dtt.iloc[k] > ENTRY_CUTOFF_CT and filled is None:
                    fail.expired_unfilled += 1
                    break
                if filled is None:
                    hit_entry = l[k] <= entry_px if bias > 0 else h[k] >= entry_px
                    hit_stop = l[k] <= stop_px if bias > 0 else h[k] >= stop_px
                    hit_tgt = h[k] >= tgt_px if bias > 0 else l[k] <= tgt_px
                    if hit_tgt and not hit_entry:
                        fail.cancelled_target_first += 1
                        break
                    if hit_stop and not hit_entry:
                        fail.cancelled_stop_first += 1
                        break
                    if hit_entry:
                        filled = k
                        if hit_stop:               # same-bar: stop first
                            parts.append((qty, stop_px, "stop"))
                            reason, exit_k = "stop", k
                            break
                    k += 1
                    continue
                eff_stop = entry_px if (model == "be1" and be_armed) else stop_px
                hit_stop = l[k] <= eff_stop if bias > 0 else h[k] >= eff_stop
                hit_1r = h[k] >= one_r_px if bias > 0 else l[k] <= one_r_px
                hit_tgt = h[k] >= tgt_px if bias > 0 else l[k] <= tgt_px
                if hit_stop:
                    rem = qty if not partial_done else q2
                    px = eff_stop
                    parts.append((rem, px, "stop"))
                    reason = ("be" if (model == "be1" and be_armed
                                       and px == entry_px)
                              else ("partial+stop" if partial_done else "stop"))
                    exit_k = k
                    break
                if model == "p1r2" and not partial_done and hit_1r:
                    parts.append((q1, one_r_px, "limit"))
                    partial_done = True
                if hit_tgt:
                    rem = qty if (model != "p1r2" or not partial_done) else q2
                    if model == "p1r2" and not partial_done:
                        rem = qty                   # degenerate safety
                    parts.append((rem, tgt_px, "limit"))
                    reason = "partial+target" if partial_done and model == \
                        "p1r2" else "target"
                    exit_k = k
                    break
                if bar_close_time >= flat_deadline - step:
                    rem = qty if not partial_done else q2
                    parts.append((rem, c[k], "market"))
                    reason = "partial+flatten" if partial_done else "flatten"
                    exit_k = k
                    break
                if hit_1r:
                    be_armed = True                # active from NEXT bar
                k += 1
            else:
                if filled is not None:
                    rem = qty if not partial_done else q2
                    parts.append((rem, c[n - 1], "market"))
                    reason = "partial+flatten" if partial_done else "flatten"
                    exit_k = n - 1
                else:
                    fail.expired_unfilled += 1
                    i = j + 1
                    continue
            if filled is None:
                i = j + 1
                continue
            gross = sum(bias * (px - entry_px) * pv * q for q, px, _ in parts)
            costs = cm.side_cost(instrument, qty, "limit").total \
                + sum(cm.side_cost(instrument, q, ot).total
                      for q, _, ot in parts)
            stopped_full = reason == "stop"
            trades.append(Trade(
                session=sess, entry_time=day.index[filled],
                exit_time=day.index[exit_k], direction=bias, qty=qty,
                entry_px=float(entry_px), exit_px=float(parts[-1][1]),
                exit_reason=reason, gross_usd=float(gross),
                costs_usd=float(costs), net_usd=float(gross - costs),
                mae_usd=float(R * pv * qty if stopped_full
                              else 0.75 * R * pv * qty),
                risk_usd=float(R * pv * qty)))
            poi_counts[last_tap_k] = poi_counts.get(last_tap_k, 0) + 1
            day_net += gross - costs
            n_trades += 1
            i = exit_k + 1
    return trades, fail, {"poi_counts": poi_counts}


def run_ict_backtest(bars1m: pd.DataFrame, cfg: Config, instrument: str,
                     poi_filter: bool = False
                     ) -> Tuple[List[Trade], Failures, Dict]:
    cm = CostModel(cfg)
    pv = cfg.instruments[instrument].point_value
    fail = Failures()
    trades: List[Trade] = []
    b3 = resample(bars1m, 3)
    b15 = resample(bars1m, 15)
    b60 = resample(bars1m, 60)
    ct3 = _ct(b3.index, cfg)
    examples: List[Dict] = []

    for sess, day3 in b3.groupby("session", sort=True):
        tt = ct3.loc[day3.index]
        # ---- bias lock from the first 15m RTH candle ----
        d15 = b15[b15["session"] == sess]
        t15 = _ct(d15.index, cfg)
        mask = t15 == time(8, 30)
        if not mask.any():
            fail.no_bias_bar_days += 1
            continue
        row = d15[mask.values].iloc[0]
        if row["close"] > row["open"]:
            bias = +1
        elif row["close"] < row["open"]:
            bias = -1
        else:
            fail.doji_days += 1
            continue

        pois = None
        if poi_filter:
            hist15 = b15[(b15["session"] <= sess)]
            hist60 = b60[(b60["session"] <= sess)]
            pois = (build_pois(hist15.tail(400), 15)
                    + build_pois(hist60.tail(400), 60))
            tick = cfg.instruments["NQ" if instrument in ("NQ", "MNQ")
                                   else instrument].tick_size
            tol = 4 * tick

        day = day3[(tt >= BIAS_LOCK_CT)]
        dtt = ct3.loc[day.index]
        o = day["open"].to_numpy(); h = day["high"].to_numpy()
        l = day["low"].to_numpy(); c = day["close"].to_numpy()
        n = len(day)
        tr = np.maximum(h - l, np.abs(h - np.r_[c[0], c[:-1]]))
        atr = pd.Series(tr).rolling(20).mean().to_numpy()
        body = np.abs(c - o)
        rng = np.maximum(h - l, 1e-12)
        disp = ((body / rng >= BODY_MIN) & (rng >= RANGE_ATR * atr)
                & ((c > o) if bias > 0 else (c < o)))
        flat_deadline = S.eod_flatten_utc(sess, cfg.sessions)

        day_net = 0.0
        n_trades = 0
        i = 20                                     # ATR warm-up
        while i < n - 1:
            if not disp[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and disp[j + 1]:       # merge ADJACENT candles
                j = j + 1
                break                              # spec: two candles max
            # ---- mark the leg ----
            if bias > 0:
                lo = min(np.minimum(o, c)[i:j + 1])     # body low
                hi = max(h[i:j + 1])                    # wick high
            else:
                hi = max(np.maximum(o, c)[i:j + 1])     # body high
                lo = min(l[i:j + 1])                    # wick low
            R = hi - lo
            if R <= 0:
                i = j + 1
                continue
            if bias > 0:
                entry_px, stop_px, tgt_px = lo + 0.5 * R, lo - 0.5 * R, hi + 0.5 * R
            else:
                entry_px, stop_px, tgt_px = hi - 0.5 * R, hi + 0.5 * R, lo - 0.5 * R

            if poi_filter:
                origin = lo if bias > 0 else hi
                if not poi_hit(pois, bias, origin, day.index[i], tol):
                    fail.poi_rejected_legs += 1
                    i = j + 1
                    continue

            # ---- sizing under the DLL arithmetic ----
            risk_per_ct = R * pv
            if instrument == "NQ":
                qty = 1
                if risk_per_ct > CAP:
                    fail.unsizeable_nq += 1
                    i = j + 1
                    continue
            else:
                qty = int(min(MAX_MICROS, CAP // risk_per_ct))
                if qty < 1:
                    i = j + 1
                    continue
            if n_trades >= MAX_TRADES:
                fail.max_trades_hits += 1
                break
            if day_net <= -DAY_STOP:
                fail.day_stop_hits += 1
                break

            # ---- order lifecycle over subsequent 3m bars ----
            filled = None
            k = j + 1
            while k < n:
                bar_close_time = day.index[k] + pd.Timedelta(minutes=3)
                if dtt.iloc[k] > ENTRY_CUTOFF_CT and filled is None:
                    fail.expired_unfilled += 1
                    break
                if filled is None:
                    hit_entry = l[k] <= entry_px if bias > 0 else h[k] >= entry_px
                    hit_stop = l[k] <= stop_px if bias > 0 else h[k] >= stop_px
                    hit_tgt = h[k] >= tgt_px if bias > 0 else l[k] <= tgt_px
                    if hit_tgt and not hit_entry:
                        fail.cancelled_target_first += 1
                        break
                    if hit_stop and not hit_entry:
                        fail.cancelled_stop_first += 1
                        break
                    if hit_entry:
                        filled = k
                        if hit_stop:               # same-bar: stop first
                            exit_px, reason, exit_k = stop_px, "stop", k
                            break
                    k += 1
                    continue
                hit_stop = l[k] <= stop_px if bias > 0 else h[k] >= stop_px
                hit_tgt = h[k] >= tgt_px if bias > 0 else l[k] <= tgt_px
                if hit_stop:
                    exit_px, reason, exit_k = stop_px, "stop", k
                    break
                if hit_tgt:
                    exit_px, reason, exit_k = tgt_px, "target", k
                    break
                if bar_close_time >= flat_deadline - pd.Timedelta(minutes=3):
                    exit_px, reason, exit_k = c[k], "flatten", k
                    break
                k += 1
            else:
                if filled is not None:
                    exit_px, reason, exit_k = c[n - 1], "flatten", n - 1
                else:
                    fail.expired_unfilled += 1
                    i = j + 1
                    continue
            if filled is None:
                i = j + 1
                continue

            gross = bias * (exit_px - entry_px) * pv * qty
            costs = (cm.side_cost(instrument, qty, "limit").total
                     + cm.side_cost(instrument, qty,
                                    "stop" if reason == "stop"
                                    else "limit" if reason == "target"
                                    else "market").total)
            net = gross - costs
            trades.append(Trade(
                session=sess, entry_time=day.index[filled],
                exit_time=day.index[exit_k], direction=bias, qty=qty,
                entry_px=float(entry_px), exit_px=float(exit_px),
                exit_reason=reason, gross_usd=float(gross),
                costs_usd=float(costs), net_usd=float(net),
                # MAE approximation (disclosed): full 1R when stopped, 0.75R
                # otherwise (true path unknown at 3m granularity; chosen
                # conservative for the MLL marks in the account simulator)
                mae_usd=float(R * pv * qty if reason == "stop"
                              else 0.75 * R * pv * qty),
                risk_usd=float(R * pv * qty)))
            if len(examples) < 5 and reason in ("target", "stop"):
                examples.append({
                    "session": str(sess), "bias": "LONG" if bias > 0 else "SHORT",
                    "leg_bars": [str(day.index[i]), str(day.index[j])],
                    "marked_lo": round(float(lo), 2),
                    "marked_hi": round(float(hi), 2),
                    "entry_0.5": round(float(entry_px), 2),
                    "stop_1.5": round(float(stop_px), 2),
                    "target_-0.5": round(float(tgt_px), 2),
                    "qty": qty, "instrument": instrument,
                    "filled": str(day.index[filled]),
                    "exit": f"{reason}@{day.index[exit_k]}",
                    "net_usd": round(float(net), 1)})
            day_net += net
            n_trades += 1
            i = exit_k + 1                          # one position at a time
    return trades, fail, {"examples": examples}
