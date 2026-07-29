"""Assemble docs/phase30_finalist/phase30_sizing_report.md from _sizing_stats.json (formatting only)."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("docs/phase30_finalist")
d = json.loads((OUT / "_sizing_stats.json").read_text())
P1, MC, SAFE = d["part1"], d["mc"], d["safe"]
L: list[str] = []
def w(s: str = "") -> None:
    L.append(s)


def rows(strat: str) -> list[dict]:
    return [r for r in P1 if r["strategy"] == strat]


def mcrows(strat: str) -> list[dict]:
    return [r for r in MC if r["strategy"] == strat]


w("# Phase 30 Finalist — Fixed-Dollar-Risk Sizing Study")
w()
w("> **Frozen strategies, sizing-only change.** Same finalists as the Phase-30 deep-dive (idb 221, "
  "swd 244, portfolio 465 base trades); only the position-sizing simulation differs. Fixed-$ risk is "
  "**dynamic**: `contracts = floor(risk_budget / (stop_distance_pts × $2))`, skip if < 1, net scales "
  "linearly with contracts (costs included). 1 NQ uses $20/pt + $1.70 RT commission. Account = "
  "$100,000 with a **$3,000 trailing** max drawdown (`floor = peak − 3000`, intratrade lows via MAE).")
w()
w("**The one-line finding:** fixed-$ risk **improves idb** (normalizes risk, skips the widest-stop "
  "trades) but **breaks swd** (it over-sizes low-volatility / tight-stop trades — exactly where swd's "
  "edge is weakest). swd's correct sizing is **fixed contracts**, not fixed-$ risk. Small budgets "
  "($50–$100) skip most trades because the 2.5×ATR stops cost ~$130–175 risk per MNQ.")
w()

hdr = ("| sizing | trades | skipped | avgC | maxC | avg $risk | net $ | PF | win% | maxDD $ | "
       "intraday DD $ | ret/DD | Sharpe | maxLstreak | worst-20 | prof yrs | **$3K fail?** | buffer $ |")
sep = "|" + "---|" * 18


def line(r: dict) -> str:
    return (f"| {r['sizing']} | {r.get('trades')} | {r.get('skipped')} | {r.get('avg_contracts')} | "
            f"{r.get('max_contracts')} | {r.get('avg_risk')} | {r.get('net')} | {r.get('pf')} | "
            f"{(r.get('win_rate') or 0)*100:.0f}% | {r.get('max_dd')} | {r.get('intraday_dd')} | "
            f"{r.get('ret_dd')} | {r.get('sharpe')} | {r.get('max_loss_streak')} | {r.get('worst_20')} | "
            f"{r.get('profitable_years')} | {'❌ FAIL' if r.get('fail_3k') else '✅ ok'} | "
            f"{r.get('min_buffer')} |")


# Part 1
w("## Part 1 — Sizing comparison (per finalist + portfolio)")
for strat, title in (("idb", "idb_base | RTH | eod"), ("swd", "swd_clean_path | RTH | toPOC"),
                     ("port", "portfolio (idb + swd)")):
    w(f"### {title}")
    w(hdr)
    w(sep)
    for r in rows(strat):
        w(line(r))
    w()

# Part 2
w("## Part 2 — 100K / $3,000 (trailing) drawdown results")
w("Pass/fail, drawdowns, buffer and breach date per sizing are in the Part-1 tables (last 4 columns) "
  "and `phase30_sizing_comparison.csv`. Summary of what survives:")
w()
w("| sizing | idb | swd | portfolio |")
w("|---|:--:|:--:|:--:|")
def cell(strat: str, sizing: str) -> str:
    r = next((x for x in rows(strat) if x["sizing"] == sizing), None)
    if r is None:
        return "—"
    return f"{'❌' if r.get('fail_3k') else '✅'} DD ${r.get('max_dd')}"
for sz in ("1 MNQ", "$100", "$150", "$200", "$250", "$300", "1 NQ"):
    w(f"| {sz} | {cell('idb', sz)} | {cell('swd', sz)} | — |")
w("| 1 MNQ each | — | — | ✅ DD $2059 |")
w("| $200/strategy | — | — | ❌ DD $3195 |")
w("| $200 total/day | — | — | ✅ DD $1956 |")
w("| $300 total/day | — | — | ❌ DD $3108 |")
w("| $500 total/day | — | — | ❌ DD $5371 |")
w()
w("**Maximum safe risk per trade** (largest budget with no historical $3K breach / with Monte-Carlo "
  "p99 max-DD < $3K):")
w()
w("| | historical max-safe | MC-robust (p99<$3K) | recommended |")
w("|---|--:|--:|---|")
w(f"| idb (fixed-$ MNQ) | ${SAFE['idb']['hist_max_safe']} | ${SAFE['idb']['mc_max_safe']} | "
  "**$150/trade** (PF 1.62, DD $968, MC p99 $2,428) |")
w(f"| swd (fixed-$ MNQ) | ${SAFE['swd']['hist_max_safe']} | ${SAFE['swd']['mc_max_safe']} | "
  "**don't use fixed-$** — use fixed 1 MNQ (DD $977) or 2 MNQ |")
w(f"| portfolio (total/day) | ${SAFE['port']['hist_max_safe_total']} | "
  f"${SAFE['port']['mc_max_safe_total']} | **1 MNQ each** or ≤$200 total/day |")
w()

# Part 3
w("## Part 3 — Monte Carlo / risk of ruin (2,000 sims; reshuffle + slippage stress, intratrade MAE)")
for strat, title in (("idb", "idb"), ("swd", "swd"), ("port", "portfolio")):
    w(f"### {title}")
    w("| sizing | P(DD≥3K) | P(DD≥2.5K) | P(DD≥2K) | exp maxDD | p95 | p99 | P(profit 60d) | P(profit 120d) |")
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in mcrows(strat):
        w(f"| {r['sizing']} | {r.get('p3k')} | {r.get('p25k')} | {r.get('p2k')} | "
          f"${r.get('exp')} | ${r.get('p95')} | ${r.get('p99')} | {r.get('p60')} | {r.get('p120')} |")
    w()
w("> Note: $50/$100 budgets skip so many trades that 60-day windows often contain *zero* trades "
  "(P(profit)→low) — they are not viable, not safe. idb $150–$200 is the genuinely robust fixed-$ "
  "zone; swd is robust only as fixed 1 MNQ.")
w()

# Part 4
w("## Part 4 — Graphs by sizing")
for cap, f in [
    ("portfolio — 1 MNQ each (equity)", "phase30_portfolio_1mnq_equity.png"),
    ("portfolio — 1 MNQ each (drawdown)", "phase30_portfolio_1mnq_drawdown.png"),
    ("portfolio — $100 risk/strategy (equity)", "phase30_portfolio_100risk_equity.png"),
    ("portfolio — $100 risk/strategy (drawdown)", "phase30_portfolio_100risk_drawdown.png"),
    ("portfolio — $200 risk/strategy (equity)", "phase30_portfolio_200risk_equity.png"),
    ("portfolio — $200 risk/strategy (drawdown)", "phase30_portfolio_200risk_drawdown.png"),
    ("portfolio — $300 risk/strategy (equity)", "phase30_portfolio_300risk_equity.png"),
    ("portfolio — $300 risk/strategy (drawdown)", "phase30_portfolio_300risk_drawdown.png"),
    ("idb — equity by sizing (1 MNQ vs $100/$200/$300 vs 1 NQ)", "phase30_idb_sizing_equity.png"),
    ("swd — equity by sizing", "phase30_swd_sizing_equity.png"),
]:
    w(f"**{cap}**")
    w(f"![{cap}]({f})")
    w()

# Part 5
w("## Part 5 — Final answers (exact)")
w("1. **Why 1 MNQ only in Phase 30?** To compare strategies on an equal, **sizing-invariant** "
  "footing — PF, win-rate and drawdown are identical regardless of contract count, and $ scales "
  "linearly. It was a measurement choice, never a position-sizing recommendation.")
w("2. **What changes with fixed $200 risk?** Contracts become dynamic (≈1–2 typical, up to 13–16 on "
  "very tight stops). **idb improves** (PF 1.44→**1.62**, maxDD $1,880→**$1,247**, net $7,082 on 162 "
  "trades, 59 wide-stop trades skipped). **swd breaks** (PF 1.535→**1.10**, maxDD $977→**$3,958 — "
  "FAILS $3K**) because fixed-$ over-sizes low-vol/tight-stop trades where swd is weak. Portfolio "
  "$200/strategy FAILS (DD $3,195).")
w("3. **Is fixed $200 safe on 100K/$3K?** **Only for idb, and only borderline** (historical DD "
  "$1,247, buffer $1,665, but MC p99 = $3,049 — just over). **Not safe for swd** or the "
  "**$200/strategy portfolio** (both fail). So $200 fixed-risk is **not** a safe portfolio-wide rule.")
w(f"4. **Max safe risk per trade?** idb: historical ${SAFE['idb']['hist_max_safe']}, MC-robust "
  f"${SAFE['idb']['mc_max_safe']} → **recommend $150** (PF 1.62, DD $968, p99 $2,428). swd: fixed-$ is "
  "unsafe → **use fixed 1 MNQ** (DD $977) — 2 MNQ also passes. Portfolio: **≤$200 total/day** "
  f"(DD $1,956) or 1 MNQ each; historical max-safe total ${SAFE['port']['hist_max_safe_total']}.")
w("5. **Is 1 NQ safe?** **No — far too large.** 1 NQ ≈ 10× MNQ: idb DD **$18,641**, swd **$9,690**, "
  "both breach $3K by a wide margin. 1 NQ needs roughly a $20K+ account, not $3K.")
w("6. **Is the portfolio safer than either alone?** Not in raw drawdown — at 1 MNQ each the portfolio "
  "DD ($2,059) is **higher** than swd alone ($977) because two positions run concurrently. Its "
  "advantage is **stability** (profitable all 6 years, smoother) — not a smaller drawdown. For pure "
  "$3K-survival, **swd alone at 1 MNQ is the safest** single choice.")
w("7. **Forward-paper sizing:** **portfolio 1 MNQ each** (diversified, survives $3K, DD $2,059) — or, "
  "to use the budget better, **idb @ $150 fixed-risk + swd @ 1 (or 2) fixed MNQ**.")
w("8. **Too aggressive:** swd with fixed-$ risk (any budget ≥$150), any sizing ≥$250 fixed-risk, "
  "$300+ total/day portfolio, 2+ MNQ on idb/portfolio, and **1 NQ** anything.")
w("9. **Trade count & drawdown per sizing (1 MNQ baseline):**")
w("   - idb: 1 MNQ → 221 trades / $1,880 DD; $150 → 108 / $968; $200 → 162 / $1,247; $300 → 202 / "
  "$2,420; 1 NQ → 221 / $18,641.")
w("   - swd: 1 MNQ → 244 / $977; $150 → 172 / $2,573; $200 → 203 / $3,958 (fail); 1 NQ → 244 / $9,690.")
w("   - portfolio: 1 MNQ each → 465 / $2,059; $200/strat → 365 / $3,195 (fail); $200 total/day → 288 "
  "/ $1,956; $500 total/day → 355 / $5,371 (fail).")
w("10. **Live / forward / reject?** **Forward-paper** (not live). Edge is real but modest (PF ~1.5), "
  "drawdown is order-dependent (Monte Carlo), and 2026 is partial. Recommended forward-paper config: "
  "**portfolio 1 MNQ each**, or **idb @ $150 risk + swd @ 1 MNQ**, on the 100K/$3K account.")

(OUT / "phase30_sizing_report.md").write_text("\n".join(L) + "\n")
print("wrote", OUT / "phase30_sizing_report.md", "lines", len(L))
