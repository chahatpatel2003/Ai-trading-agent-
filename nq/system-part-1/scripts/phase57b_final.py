"""Phase 57b — final decisive diagnostics: beta-exposure split + remaining structures."""
from __future__ import annotations
import numpy as np
from scripts.phase57_engine import load_sessions, walk
from scripts.phase57b_discover import idx_at, validate, NY_OPEN, FLAT_PIT

sess = load_sessions()


def strong_open_cont(direction_filter=None, min_disp=0.3, enter=900):
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN); ei = idx_at(s, enter)
        if oi is None or ei is None or ei + 1 >= len(s["o"]):
            continue
        ropen, cur = s["o"][oi], s["c"][ei]
        if abs(cur - ropen) / s["atr"] < min_disp:
            continue
        d = 1 if cur > ropen else -1
        if direction_filter is not None and d != direction_filter:
            continue
        m = (s["smin"] >= NY_OPEN) & (s["smin"] < enter)
        orh, orl = float(s["h"][m].max()), float(s["l"][m].min())
        rng = max(orh - orl, 0.3)
        stop = cur - d * rng
        r = walk(s, ei, d, stop, 1e9 * d, flat_smin=FLAT_PIT, time_stop=200)
        if r:
            out.append((s["date"], enter, r[0], r[1]))
    return out


print("=== BETA-EXPOSURE SPLIT: strong-open continuation (disp>0.3) ===")
for lab, df in [("LONG-only (up opens)", 1), ("SHORT-only (dn opens)", -1)]:
    tr = strong_open_cont(direction_filter=df)
    nets = np.array([t[2] for t in tr]); yrs = np.array([t[0].year for t in tr])
    tr_e = nets[yrs <= 2020].mean(); va = nets[(yrs >= 2021) & (yrs <= 2022)].mean(); te = nets[yrs >= 2023].mean()
    print(f"  {lab:22s} n={len(tr):4d} exp=${nets.mean():6.1f}  train=${tr_e:6.1f} val=${va:6.1f} test=${te:6.1f}")
print("  -> if edge were real (not beta), BOTH long and short continuation would profit.")


def sweep_stall_cont():
    """Overnight high swept in RTH, price STALLS above it (holds 2 bars), then continue up."""
    out = []
    for s in sess:
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        on = (s["smin"] >= 60) & (s["smin"] < NY_OPEN)
        if on.sum() < 20:
            continue
        onh, onl = float(s["h"][on].max()), float(s["l"][on].min())
        rng = max(onh - onl, 0.4)
        for j in range(oi, len(s["o"])):
            if s["smin"][j] > 1080:
                break
            # swept overnight high, then NEXT bar holds above (stall=acceptance) -> continue long
            if s["h"][j] > onh and j + 1 < len(s["o"]) and s["c"][j] > onh and s["c"][j + 1] > onh:
                r = walk(s, j + 1, 1, onh - 0.3 * rng, s["c"][j + 1] + 1.2 * rng, flat_smin=FLAT_PIT)
                if r:
                    out.append((s["date"], int(s["smin"][j + 1]), r[0], r[1]))
                break
            if s["l"][j] < onl and j + 1 < len(s["o"]) and s["c"][j] < onl and s["c"][j + 1] < onl:
                r = walk(s, j + 1, -1, onl + 0.3 * rng, s["c"][j + 1] - 1.2 * rng, flat_smin=FLAT_PIT)
                if r:
                    out.append((s["date"], int(s["smin"][j + 1]), r[0], r[1]))
                break
    return out


def failed_auction():
    """Open beyond prior-day range (failed acceptance) -> revert into prior range."""
    out = []
    for s in sess:
        if s["prev"] is None:
            continue
        phi, plo, _ = s["prev"]
        oi = idx_at(s, NY_OPEN)
        if oi is None:
            continue
        op = s["o"][oi]
        rng = max(phi - plo, 0.5)
        if op > phi + 0.1 * rng:        # opened above prior range
            for j in range(oi, len(s["o"])):
                if s["smin"][j] > 1020:
                    break
                if s["c"][j] < phi:      # acceptance failed, back inside
                    r = walk(s, j, -1, s["h"][:j + 1].max() if j > oi else op + 0.3 * rng,
                             (phi + plo) / 2, flat_smin=FLAT_PIT)
                    if r:
                        out.append((s["date"], int(s["smin"][min(j + 1, len(s["smin"]) - 1)]), r[0], r[1]))
                    break
        elif op < plo - 0.1 * rng:      # opened below prior range
            for j in range(oi, len(s["o"])):
                if s["smin"][j] > 1020:
                    break
                if s["c"][j] > plo:
                    r = walk(s, j, 1, op - 0.3 * rng, (phi + plo) / 2, flat_smin=FLAT_PIT)
                    if r:
                        out.append((s["date"], int(s["smin"][min(j + 1, len(s["smin"]) - 1)]), r[0], r[1]))
                    break
    return out


print("\n=== REMAINING STRUCTURES (strict val/test split) ===")
print(f"{'structure':26s} {'n':>5s} {'exp$':>6s} {'train':>6s} {'val':>6s} {'test':>6s} {'+yrs':>4s} {'OK':>3s}")
for lab, fn in [("sweep->stall->cont", sweep_stall_cont), ("failed-auction revert", failed_auction)]:
    tr = fn()
    v = validate(tr, lab)
    if v is None:
        print(f"{lab:26s}  (too few: {len(tr)})"); continue
    print(f"{lab:26s} {v['n']:>5d} {v['exp']:>6.1f} {str(v['train']):>6s} {str(v['val']):>6s} "
          f"{str(v['test']):>6s} {v['pos_years']:>4d} {'YES' if v['survive'] else '':>3s}")
