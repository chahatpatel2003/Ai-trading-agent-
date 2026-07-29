"""Phase 57 — GC/MGC backtest engine (self-contained; mirrors NQ Topstep sim rules).

Per-session numpy arrays on 5m bars (gold session clock). Shared next-bar-open
trade simulator with realistic MGC costs. 15 strategy families. Ported Topstep
combine/funded simulators (identical economics to the NQ benchmark: ACCOUNTS
costs 49/99/199, trailing-MLL via per-trade MAE, consistency 50%, min 2 days).

MGC economics: point_value $10, tick 0.1 ($1). Cost per round turn per contract:
commission $1.48 (0.74/side) + slippage 1 tick entry + 1 tick exit ($2) ~ $3.48.
"""

from __future__ import annotations

from collections import defaultdict

import os
import tempfile

import numpy as np
import polars as pl

CACHE = os.path.join(os.environ.get("NQLAB_SCRATCH", os.path.join(tempfile.gettempdir(), "nqlab-phase57")), "gc_5m.parquet")

# ---- economics ----
PV = 10.0           # MGC $ per 1.0 point
TICK = 0.1
SLIP = 0.1          # 1 tick adverse per side
COMM_RT = 1.48      # round-turn commission per contract

# ---- session windows (smin) ----
ASIA = (60, 540)     # 19:00-03:00 ET
LONDON = (540, 860)  # 03:00-08:20 ET
NY_OPEN = 860        # 08:20 ET (COMEX)
NY_ENTRY_END = 1080  # 12:00 ET (no new entries after)
PIT_CLOSE = 1170     # 13:30 ET
FLAT = 1320          # 16:00 ET force flat
TIME_STOP_BARS = 24  # 2 hours

# Topstep accounts: (cost, target, mll, max_contracts)
ACCOUNTS = {
    "50K": (49.0, 3000.0, 2000.0, 5),
    "100K": (99.0, 6000.0, 3000.0, 10),
    "150K": (199.0, 9000.0, 4500.0, 15),
}
WIN_THR, MIN_WIN = 150.0, 5


# ======================================================================
# Session data
# ======================================================================
def load_sessions():
    g = pl.read_parquet(CACHE).sort("b5")
    sess = []
    for (sd,), d in g.group_by(["session_date"], maintain_order=True):
        d = d.sort("smin")
        sess.append({
            "date": sd,
            "smin": d["smin"].to_numpy(),
            "o": d["open"].to_numpy(),
            "h": d["high"].to_numpy(),
            "l": d["low"].to_numpy(),
            "c": d["close"].to_numpy(),
            "v": d["volume"].to_numpy().astype(float),
        })
    sess.sort(key=lambda s: s["date"])
    # attach prior-session RTH refs + ATR scale
    prev_rth = None
    rngs = []
    for s in sess:
        sm = s["smin"]
        ny = (sm >= NY_OPEN) & (sm <= PIT_CLOSE)
        if ny.any():
            rhi, rlo, rcl = float(s["h"][ny].max()), float(s["l"][ny].min()), float(s["c"][ny][-1])
        else:
            rhi = rlo = rcl = float(s["c"][-1])
        s["rth_hi"], s["rth_lo"], s["rth_close"] = rhi, rlo, rcl
        s["prev"] = prev_rth
        rngs.append(rhi - rlo)
        prev_rth = (rhi, rlo, rcl)
    # rolling ATR(10) of RTH range, shifted (no lookahead)
    r = np.array(rngs)
    atr = np.full(len(r), np.nan)
    for i in range(len(r)):
        lo = max(0, i - 10)
        if i > 0:
            atr[i] = np.nanmean(r[lo:i]) if i > lo else r[max(0, i - 1)]
    med = np.nanmedian(r)
    atr = np.where(np.isnan(atr), med, atr)
    for i, s in enumerate(sess):
        s["atr"] = float(max(atr[i], 1.0))   # $/point range scale, floor 1.0 pt
    return sess


# ======================================================================
# Trade simulator (next-bar-open entry, no look-ahead)
# ======================================================================
def walk(s, sig_idx, direction, stop_px, target_px, flat_smin=FLAT, time_stop=TIME_STOP_BARS):
    """Signal confirmed at bar sig_idx close -> enter at next bar open. Manage on
    completed bars. Returns (net_per_contract, mae_dollars, reason, hold_bars) or None."""
    n = len(s["o"])
    ei = sig_idx + 1
    if ei >= n:
        return None
    entry_raw = s["o"][ei]
    entry = entry_raw + direction * SLIP            # entry slippage adverse
    mae_px = 0.0
    for j in range(ei, n):
        hi, lo, cl, sm = s["h"][j], s["l"][j], s["c"][j], s["smin"][j]
        # adverse excursion vs entry fill
        adv = (entry - lo) if direction > 0 else (hi - entry)
        if adv > mae_px:
            mae_px = adv
        # stop check first (conservative)
        if direction > 0:
            if lo <= stop_px:
                exit_px = min(stop_px, s["o"][j] if s["o"][j] <= stop_px else stop_px)
                return _close(entry, exit_px, direction, mae_px, "stop", j - ei)
            if hi >= target_px:
                return _close(entry, target_px, direction, mae_px, "target", j - ei)
        else:
            if hi >= stop_px:
                exit_px = max(stop_px, s["o"][j] if s["o"][j] >= stop_px else stop_px)
                return _close(entry, exit_px, direction, mae_px, "stop", j - ei)
            if lo <= target_px:
                return _close(entry, target_px, direction, mae_px, "target", j - ei)
        if sm >= flat_smin or (j - ei) >= time_stop:
            return _close(entry, cl, direction, mae_px, "flat", j - ei)
    return _close(entry, s["c"][-1], direction, mae_px, "eod", n - 1 - ei)


def _close(entry, exit_px, direction, mae_px, reason, hold):
    exit_eff = exit_px - direction * SLIP           # exit slippage adverse
    net = direction * (exit_eff - entry) * PV - COMM_RT
    mae_d = max(mae_px * PV + SLIP * PV, 0.0)        # $ adverse incl entry slip
    return (net, mae_d, reason, hold)


def idx_at(s, smin_val):
    """first bar index at/after smin_val, or None."""
    w = np.where(s["smin"] >= smin_val)[0]
    return int(w[0]) if len(w) else None


def window_mask(s, a, b):
    return (s["smin"] >= a) & (s["smin"] < b)


# ======================================================================
# Strategy families. Each returns list of (session_date, entry_smin, net, mae).
# ======================================================================
def _emit(trades, s, sig_idx, direction, stop_px, target_px, **kw):
    r = walk(s, sig_idx, direction, stop_px, target_px, **kw)
    if r is None:
        return
    net, mae, reason, hold = r
    trades.append((s["date"], int(s["smin"][min(sig_idx + 1, len(s["smin"]) - 1)]), net, mae))


def fam_open_drive(sess):
    """1. COMEX open drive continuation: first 15m range, break = continuation."""
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN)
        if oi is None or oi + 3 >= len(s["o"]):
            continue
        orh = float(s["h"][oi:oi + 3].max())
        orl = float(s["l"][oi:oi + 3].min())
        rng = max(orh - orl, 0.3)
        # scan next bars for breakout
        for j in range(oi + 3, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if s["c"][j] > orh:
                _emit(out, s, j, +1, orl, s["o"][j + 1] + 1.5 * rng if j + 1 < len(s["o"]) else orh + 1.5 * rng)
                break
            if s["c"][j] < orl:
                _emit(out, s, j, -1, orh, orl - 1.5 * rng)
                break
    return out


def fam_lon_ny_break(sess):
    """2. London-to-NY breakout: break London range in first NY hours."""
    out = []
    for s in sess:
        lm = window_mask(s, *LONDON)
        if lm.sum() < 5:
            continue
        lhi, llo = float(s["h"][lm].max()), float(s["l"][lm].min())
        rng = max(lhi - llo, 0.4)
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        for j in range(oi, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if s["c"][j] > lhi:
                _emit(out, s, j, +1, lhi - 0.5 * rng, lhi + rng)
                break
            if s["c"][j] < llo:
                _emit(out, s, j, -1, llo + 0.5 * rng, llo - rng)
                break
    return out


def fam_asia_sweep(sess):
    """3. Asia/London sweep into NY: sweep Asia extreme then reclaim -> fade."""
    out = []
    for s in sess:
        am = window_mask(s, *ASIA)
        if am.sum() < 5:
            continue
        ahi, alo = float(s["h"][am].max()), float(s["l"][am].min())
        rng = max(ahi - alo, 0.4)
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        for j in range(oi, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            # swept high then closed back below -> short
            if s["h"][j] > ahi and s["c"][j] < ahi:
                _emit(out, s, j, -1, s["h"][j] + 0.2 * rng, s["c"][j] - rng)
                break
            if s["l"][j] < alo and s["c"][j] > alo:
                _emit(out, s, j, +1, s["l"][j] - 0.2 * rng, s["c"][j] + rng)
                break
    return out


def fam_pdhl_sweep(sess):
    """4. Prior-day high/low sweep & reclaim -> reversal."""
    out = []
    for s in sess:
        if s["prev"] is None:
            continue
        phi, plo, _ = s["prev"]
        rng = max(phi - plo, 0.5)
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        for j in range(oi, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if s["h"][j] > phi and s["c"][j] < phi:
                _emit(out, s, j, -1, s["h"][j] + 0.15 * rng, s["c"][j] - 0.6 * rng)
                break
            if s["l"][j] < plo and s["c"][j] > plo:
                _emit(out, s, j, +1, s["l"][j] - 0.15 * rng, s["c"][j] + 0.6 * rng)
                break
    return out


def fam_vwap(sess):
    """5. VWAP reclaim/rejection from NY open session VWAP."""
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        tp = (s["h"] + s["l"] + s["c"]) / 3.0
        cv = np.cumsum(tp[oi:] * s["v"][oi:])
        cvol = np.cumsum(s["v"][oi:]) + 1e-9
        vwap = cv / cvol
        scale = max(s["atr"] * 0.25, 0.4)
        for k in range(2, len(vwap)):
            j = oi + k
            if s["smin"][j] > NY_ENTRY_END:
                break
            # reclaim: prev close below vwap, now closes above
            if s["c"][j - 1] < vwap[k - 1] and s["c"][j] > vwap[k]:
                _emit(out, s, j, +1, s["l"][j] - 0.1 * scale, s["c"][j] + 1.5 * scale)
                break
            if s["c"][j - 1] > vwap[k - 1] and s["c"][j] < vwap[k]:
                _emit(out, s, j, -1, s["h"][j] + 0.1 * scale, s["c"][j] - 1.5 * scale)
                break
    return out


def fam_poc_va(sess):
    """6. POC/value-area reversion: fade prior-session volume extremes to POC."""
    out = []
    for s in sess:
        if s["prev"] is None:
            continue
        # build prior-session profile from THIS engine's prev arrays is unavailable;
        # approximate value area with prior RTH hi/lo + midpoint POC.
        phi, plo, _ = s["prev"]
        poc = (phi + plo) / 2.0
        va_hi = poc + 0.35 * (phi - poc)
        va_lo = poc - 0.35 * (poc - plo)
        rng = max(phi - plo, 0.5)
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        for j in range(oi, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if s["c"][j] > va_hi and s["c"][j] < phi:
                _emit(out, s, j, -1, phi + 0.1 * rng, poc)
                break
            if s["c"][j] < va_lo and s["c"][j] > plo:
                _emit(out, s, j, +1, plo - 0.1 * rng, poc)
                break
    return out


def fam_gap_cont(sess):
    """7. Gap continuation: NY open gap vs prior pit close -> trade in gap dir."""
    out = []
    for s in sess:
        if s["prev"] is None:
            continue
        _, _, pcl = s["prev"]
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        op = float(s["o"][oi])
        gap = op - pcl
        thr = 0.25 * s["atr"]
        if abs(gap) < thr or abs(gap) > 3 * s["atr"]:  # ignore tiny + roll-artifact gaps
            continue
        d = 1 if gap > 0 else -1
        # enter on first 5m close beyond the open in gap direction
        for j in range(oi, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if d > 0 and s["c"][j] > op:
                _emit(out, s, j, +1, op - 0.5 * abs(gap), op + 1.5 * abs(gap))
                break
            if d < 0 and s["c"][j] < op:
                _emit(out, s, j, -1, op + 0.5 * abs(gap), op - 1.5 * abs(gap))
                break
    return out


def fam_gap_fill(sess):
    """8. Gap fill: fade moderate gap back toward prior close."""
    out = []
    for s in sess:
        if s["prev"] is None:
            continue
        _, _, pcl = s["prev"]
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        op = float(s["o"][oi])
        gap = op - pcl
        thr = 0.25 * s["atr"]
        if abs(gap) < thr or abs(gap) > 2 * s["atr"]:
            continue
        d = -1 if gap > 0 else 1   # fade toward pcl
        # enter when price starts reverting (first close back toward pcl)
        for j in range(oi, min(oi + 12, len(s["o"]))):
            if d > 0 and s["c"][j] > op:        # gap down, reverting up
                _emit(out, s, j, +1, op - 0.8 * abs(gap), pcl)
                break
            if d < 0 and s["c"][j] < op:        # gap up, reverting down
                _emit(out, s, j, -1, op + 0.8 * abs(gap), pcl)
                break
    return out


def fam_vol_break(sess):
    """9. High-volatility breakout: elevated-ATR days, opening-range break, wide target."""
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN)
        if oi is None or oi + 3 >= len(s["o"]):
            continue
        if s["atr"] < np.median([x["atr"] for x in sess]):  # only high-vol regime
            pass  # placeholder; real filter below
        orh = float(s["h"][oi:oi + 3].max())
        orl = float(s["l"][oi:oi + 3].min())
        rng = orh - orl
        if rng < 0.4 * s["atr"]:   # need an energetic open
            continue
        for j in range(oi + 3, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if s["c"][j] > orh:
                _emit(out, s, j, +1, orl, orh + 2.0 * rng)
                break
            if s["c"][j] < orl:
                _emit(out, s, j, -1, orh, orl - 2.0 * rng)
                break
    return out


def fam_fail_break(sess):
    """10. Failed breakout reversal: OR break that closes back in -> reverse."""
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN)
        if oi is None or oi + 3 >= len(s["o"]):
            continue
        orh = float(s["h"][oi:oi + 3].max())
        orl = float(s["l"][oi:oi + 3].min())
        rng = max(orh - orl, 0.3)
        broke = 0
        for j in range(oi + 3, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if broke == 0:
                if s["h"][j] > orh:
                    broke = 1
                elif s["l"][j] < orl:
                    broke = -1
            elif broke == 1 and s["c"][j] < orh:      # failed upside break
                _emit(out, s, j, -1, s["h"][j] + 0.15 * rng, orl)
                break
            elif broke == -1 and s["c"][j] > orl:     # failed downside break
                _emit(out, s, j, +1, s["l"][j] - 0.15 * rng, orh)
                break
    return out


def fam_first_pb(sess):
    """11. First pullback after impulse: strong 30m open move, enter first pullback."""
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN)
        if oi is None or oi + 6 >= len(s["o"]):
            continue
        imp = s["c"][oi + 5] - s["o"][oi]
        if abs(imp) < 0.5 * s["atr"]:
            continue
        d = 1 if imp > 0 else -1
        # wait for a pullback (a bar that closes against d) then re-entry on resumption
        pulled = False
        for j in range(oi + 6, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if not pulled:
                if (d > 0 and s["c"][j] < s["c"][j - 1]) or (d < 0 and s["c"][j] > s["c"][j - 1]):
                    pulled = True
            else:
                if (d > 0 and s["c"][j] > s["h"][j - 1]) or (d < 0 and s["c"][j] < s["l"][j - 1]):
                    stop = s["l"][j] - 0.2 * s["atr"] if d > 0 else s["h"][j] + 0.2 * s["atr"]
                    tgt = s["c"][j] + d * 1.5 * abs(imp)
                    _emit(out, s, j, d, stop, tgt)
                    break
    return out


def fam_trend_day(sess):
    """12. Trend-day continuation: open in lower/upper third + mid-morning extension."""
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN)
        mi = idx_at(s, 960)   # ~09:00 ET, after first 40m
        if oi is None or mi is None or mi + 1 >= len(s["o"]):
            continue
        hi = float(s["h"][oi:mi + 1].max())
        lo = float(s["l"][oi:mi + 1].min())
        rng = max(hi - lo, 0.4)
        cur = s["c"][mi]
        pos = (cur - lo) / rng
        if pos > 0.8:        # trending up, near highs
            for j in range(mi, len(s["o"])):
                if s["smin"][j] > NY_ENTRY_END:
                    break
                if s["c"][j] > hi:
                    _emit(out, s, j, +1, lo, hi + 1.5 * rng)
                    break
        elif pos < 0.2:      # trending down
            for j in range(mi, len(s["o"])):
                if s["smin"][j] > NY_ENTRY_END:
                    break
                if s["c"][j] < lo:
                    _emit(out, s, j, -1, hi, lo - 1.5 * rng)
                    break
    return out


def fam_range_mr(sess):
    """13. Range-day mean reversion: low-vol day, fade overnight extremes to mid."""
    out = []
    med_atr = float(np.median([x["atr"] for x in sess]))
    for s in sess:
        if s["atr"] > 0.9 * med_atr:   # only quieter days
            continue
        on = window_mask(s, ASIA[0], NY_OPEN)   # overnight Asia+London
        if on.sum() < 10:
            continue
        ohi, olo = float(s["h"][on].max()), float(s["l"][on].min())
        mid = (ohi + olo) / 2.0
        rng = max(ohi - olo, 0.4)
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        for j in range(oi, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if s["c"][j] >= ohi:
                _emit(out, s, j, -1, ohi + 0.25 * rng, mid)
                break
            if s["c"][j] <= olo:
                _emit(out, s, j, +1, olo - 0.25 * rng, mid)
                break
    return out


def fam_news_vol(sess):
    """14. News-vol proxy days: 08:30 ET release bar abnormally large -> trade thrust."""
    out = []
    for s in sess:
        bi = idx_at(s, 870)   # 08:30 ET bar (et_mod 510 -> smin 870)
        if bi is None or bi + 1 >= len(s["o"]):
            continue
        bar_rng = s["h"][bi] - s["l"][bi]
        if bar_rng < 0.6 * s["atr"]:   # not a news-impulse day
            continue
        d = 1 if s["c"][bi] > s["o"][bi] else -1
        stop = s["l"][bi] - 0.1 * bar_rng if d > 0 else s["h"][bi] + 0.1 * bar_rng
        tgt = s["c"][bi] + d * 1.5 * bar_rng
        _emit(out, s, bi, d, stop, tgt)
    return out


def fam_rates_proxy(sess):
    """15. Dollar/rates proxy UNAVAILABLE (only GC data). Substitute: multi-day gold
    momentum-aligned open-drive (trade open breakout only WITH the 5-day trend)."""
    out = []
    closes = [s["rth_close"] for s in sess]
    for i, s in enumerate(sess):
        if i < 6:
            continue
        trend = closes[i - 1] - closes[i - 6]   # 5-session momentum (shifted, no lookahead)
        if abs(trend) < 0.3 * s["atr"]:
            continue
        d = 1 if trend > 0 else -1
        oi = idx_at(s, NY_OPEN)
        if oi is None or oi + 3 >= len(s["o"]):
            continue
        orh = float(s["h"][oi:oi + 3].max())
        orl = float(s["l"][oi:oi + 3].min())
        rng = max(orh - orl, 0.3)
        for j in range(oi + 3, len(s["o"])):
            if s["smin"][j] > NY_ENTRY_END:
                break
            if d > 0 and s["c"][j] > orh:
                _emit(out, s, j, +1, orl, orh + 1.8 * rng)
                break
            if d < 0 and s["c"][j] < orl:
                _emit(out, s, j, -1, orh, orl - 1.8 * rng)
                break
    return out


FAMILIES = {
    "1_open_drive": fam_open_drive,
    "2_lon_ny_break": fam_lon_ny_break,
    "3_asia_sweep": fam_asia_sweep,
    "4_pdhl_sweep": fam_pdhl_sweep,
    "5_vwap": fam_vwap,
    "6_poc_va": fam_poc_va,
    "7_gap_cont": fam_gap_cont,
    "8_gap_fill": fam_gap_fill,
    "9_vol_break": fam_vol_break,
    "10_fail_break": fam_fail_break,
    "11_first_pb": fam_first_pb,
    "12_trend_day": fam_trend_day,
    "13_range_mr": fam_range_mr,
    "14_news_vol": fam_news_vol,
    "15_rates_proxy": fam_rates_proxy,
}


# ======================================================================
# Trade-level metrics
# ======================================================================
def trade_metrics(trades):
    if not trades:
        return None
    nets = np.array([t[2] for t in trades])
    yrs = defaultdict(list)
    for t in trades:
        yrs[t[0].year].append(t[2])
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    gp, gl = wins.sum(), -losses.sum()
    n = len(nets)
    wr = len(wins) / n
    aw = wins.mean() if len(wins) else 0.0
    al = losses.mean() if len(losses) else 0.0
    # OOS split: train 2016-2020, test 2021-2025
    tr = np.array([t[2] for t in trades if t[0].year <= 2020])
    te = np.array([t[2] for t in trades if t[0].year >= 2021])
    return {
        "trades": n,
        "per_yr": round(n / 10, 1),
        "wr": round(wr, 3),
        "avg_win": round(float(aw), 1),
        "avg_loss": round(float(al), 1),
        "rr": round(float(aw / -al), 2) if al < 0 else None,
        "expectancy": round(float(nets.mean()), 2),
        "pf": round(float(gp / gl), 2) if gl > 0 else None,
        "total_pnl": round(float(nets.sum()), 0),
        "yearly": {y: round(float(np.sum(v)), 0) for y, v in sorted(yrs.items())},
        "is_exp": round(float(tr.mean()), 2) if len(tr) else None,
        "oos_exp": round(float(te.mean()), 2) if len(te) else None,
        "oos_pf": round(float(te[te > 0].sum() / -te[te < 0].sum()), 2)
        if len(te) and te[te < 0].sum() < 0 else None,
    }


# ======================================================================
# Topstep simulators (ported; sized in MGC contracts)
# ======================================================================
def build_days(trades, sess_dates):
    """trades -> ordered per-session list of (smin, net1, mae1)."""
    bd = defaultdict(list)
    for (d, sm, net, mae) in trades:
        bd[d].append((sm, net, mae))
    return [sorted(bd.get(d, [])) for d in sess_dates]


def cushion(T, mc):
    s, th = (1, 2, 3), (0.22 * T, 0.44 * T)

    def f(bal):
        c = s[0] if bal < th[0] else (s[1] if bal < th[1] else s[2])
        return min(c, mc)
    return f


def sim_combine(days, start, T, M, sized, lstop, horizon=300):
    equity = peak = 0.0
    active = 0
    for di in range(horizon):
        if start + di >= len(days):
            break
        day_pnl = 0.0
        traded = False
        for (_sm, net1, mae1) in days[start + di]:
            if lstop is not None and day_pnl <= lstop:
                break
            bal = equity + day_pnl
            s = sized(bal)
            pk = max(peak, bal)
            if bal - mae1 * s <= min(pk - M, 0.0):
                return ("blow", di + 1)
            day_pnl += net1 * s
            traded = True
            bal = equity + day_pnl
            peak = max(peak, bal)
            if bal <= min(peak - M, 0.0):
                return ("blow", di + 1)
        equity += day_pnl
        if traded:
            active += 1
        if equity >= T and active >= 2 and day_pnl <= 0.5 * equity:
            # consistency: best single day <= 50% of total (approx via running check)
            return ("pass", di + 1)
    return ("timeout", horizon)


def sim_funded(days, start, M, mc, lstop, cap, fsize, horizon=180):
    eq = peak = best = 0.0
    win = pots = 0
    wd = 0.0
    first = None
    locked = False
    for di in range(horizon):
        if start + di >= len(days):
            break
        d = 0.0
        for (_sm, net1, mae1) in days[start + di]:
            if lstop is not None and d <= lstop:
                break
            s = min(fsize, mc)
            bal = eq + d
            floor = 0.0 if locked else min(peak - M, 0.0)
            if bal - mae1 * s <= floor:
                return {"wd": wd, "pots": pots, "first": first, "blew": True, "day": di + 1}
            d += net1 * s
            bal = eq + d
            peak = max(peak, bal)
            if peak >= M:
                locked = True
            if bal <= (0.0 if locked else min(peak - M, 0.0)):
                return {"wd": wd, "pots": pots, "first": first, "blew": True, "day": di + 1}
        eq += d
        best = max(best, d)
        if d >= WIN_THR:
            win += 1
        if win >= MIN_WIN and eq > 0 and best <= 0.5 * eq:
            amt = min(eq, cap)
            wd += amt
            eq -= amt
            pots += 1
            first = first or (di + 1)
            win = 0
            best = 0.0
    return {"wd": wd, "pots": pots, "first": first, "blew": False, "day": horizon}


def topstep_eval(days, acct, fsize, rng, sims=4000):
    cost, T, M, mc = ACCOUNTS[acct]
    n = len(days)
    if n < 320:
        return None
    cs = rng.integers(0, n - 300, sims)
    cres = [sim_combine(days, int(s), T, M, cushion(T, mc), -0.375 * M, 300) for s in cs]
    npass = sum(1 for r in cres if r[0] == "pass")
    nblow = sum(1 for r in cres if r[0] == "blow")
    p_pass = npass / (npass + nblow) if (npass + nblow) else 0.0
    pdays = np.array([r[1] for r in cres if r[0] == "pass"])
    resd = np.array([r[1] for r in cres if r[0] in ("pass", "blow")])
    e_att = 1.0 / p_pass if p_pass > 0 else 999.0
    combine_cost = cost * e_att
    combine_days = e_att * (float(resd.mean()) if resd.size else 300)

    fs = rng.integers(0, n - 180, sims)
    fres = [sim_funded(days, int(s), M, mc, -0.375 * M, M, fsize) for s in fs]
    e_payout = float(np.mean([r["wd"] for r in fres]))
    surv = float(np.mean([0 if r["blew"] else 1 for r in fres]))
    fdur = float(np.mean([r["day"] for r in fres]))
    first_prob = float(np.mean([1 if r["first"] else 0 for r in fres]))
    ev_cycle = e_payout - combine_cost
    cyc_days = combine_days + fdur
    ev_month = ev_cycle / (cyc_days / 21.0) if cyc_days > 0 else None
    return {
        "p_pass": round(p_pass, 3),
        "p_fail": round(nblow / (npass + nblow), 3) if (npass + nblow) else None,
        "med_days_pass": round(float(np.median(pdays)), 0) if pdays.size else None,
        "first_payout_prob": round(first_prob, 3),
        "E_payout": round(e_payout, 0),
        "funded_surv": round(surv, 3),
        "combine_cost": round(combine_cost, 0),
        "EV_cycle": round(ev_cycle, 0),
        "cycle_mo": round(cyc_days / 21.0, 1),
        "EV_month": round(ev_month, 0) if ev_month else None,
    }


if __name__ == "__main__":
    sess = load_sessions()
    print("sessions:", len(sess), sess[0]["date"], "->", sess[-1]["date"])
    print(f"{'family':16s} {'trades':>7s} {'wr':>6s} {'rr':>5s} {'exp':>7s} {'pf':>5s} {'oos_exp':>8s} {'total$':>9s}")
    for name, fn in FAMILIES.items():
        tr = fn(sess)
        m = trade_metrics(tr)
        if m is None:
            print(f"{name:16s}  (no trades)")
            continue
        print(f"{name:16s} {m['trades']:>7d} {m['wr']:>6.3f} {str(m['rr']):>5s} "
              f"{m['expectancy']:>7.2f} {str(m['pf']):>5s} {str(m['oos_exp']):>8s} {m['total_pnl']:>9.0f}")
