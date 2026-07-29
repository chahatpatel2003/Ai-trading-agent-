"""Assemble docs/phase30_finalist/phase30_finalist_report.md from the computed _stats.json.

No computation of strategy results here — pure formatting of the frozen Phase-30 finalist stats,
feasibility, Monte Carlo and graphs produced by scripts/phase30_finalist_report.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

OUT = Path("docs/phase30_finalist")
d = json.loads((OUT / "_stats.json").read_text())
P, YT, MT, SIMS, MC, RULES = (d["panels"], d["year_tables"], d["month_tables"], d["sims"], d["mc"],
                              d["rules"])
NAMES = {"idb": "idb_base | RTH | eod", "swd": "swd_clean_path | RTH | toPOC",
         "port": "portfolio (idb + swd, 1 MNQ each)"}
L: list[str] = []
def w(s: str = "") -> None:
    L.append(s)


# monthly detail CSV (finalist, month, pnl, trades, win_rate, pf, max_dd)
mrows = [{"finalist": k, **m} for k in ("idb", "swd", "port") for m in MT[k]]
pl.DataFrame(mrows, infer_schema_length=None).write_csv(OUT / "phase30_monthly_detail.csv")

w("# Phase 30 Finalist Deep-Dive — Exact Frozen Stats, 100K/$3K Feasibility, Monte Carlo, Graphs")
w()
w("> **Frozen, not re-optimized.** These are the exact Phase-30 finalists re-run unchanged; the "
  "trade counts reproduce the frozen results (idb 221, swd 244, portfolio 465). All $ figures are "
  "**net of the audited cost model** (MNQ $1.00/$1.50 win/loss, slippage in fills). Sizing baseline "
  "= **1 MNQ**; other sizings scale per-contract (NQ derived exactly from the point move).")
w()

# ---- Part 1
w("## Part 1 — Exact mechanical rules")
for k in ("idb", "swd", "port"):
    r = RULES[k]
    p = P[k]
    w(f"### {r['name']}")
    w(f"- **family:** {r['family']} | **timeframe:** {r['timeframe']} | **session:** {r['session']} "
      f"| **direction:** {r['direction']}")
    w(f"- **entry rule:** {r['entry']}")
    w(f"- **signal rule:** {r['signal']}")
    w(f"- **confirmation:** {r['confirmation']}")
    w(f"- **stop rule:** {r['stop']}  → **stop type:** {r['stop_type']}")
    w(f"- **target rule:** {r['target']}")
    w(f"- **exit rule:** {r['exit']}")
    w(f"- **RR:** {r['rr_type']}")
    w(f"- **realized R:** avg **{p['avg_R']}** | median **{p['median_R']}** | avg win "
      f"**{p['avg_win_R']}R** | avg loss **{p['avg_loss_R']}R** | best **{p['best_R']}R** | worst "
      f"**{p['worst_R']}R**")
    w()

# ---- Part 2
w("## Part 2 — Full stats (1 MNQ)")
keys = [
    ("trades", "total trades"), ("win_rate", "win rate"), ("loss_rate", "loss rate"),
    ("breakeven", "breakeven trades"), ("pf_before", "PF before costs"), ("pf_after", "PF after costs"),
    ("net", "net profit $"), ("gross_profit", "gross profit $"), ("gross_loss", "gross loss $"),
    ("expectancy", "expectancy $/trade"), ("expectancy_month", "expectancy $/month"),
    ("avg_win", "avg win $"), ("avg_loss", "avg loss $"), ("win_loss_ratio", "avg win/avg loss"),
    ("max_win", "max win $"), ("max_loss", "max loss $"),
    ("max_consec_wins", "max consec wins"), ("max_consec_losses", "max consec losses"),
    ("n_3loss", "# 3-loss streaks"), ("n_5loss", "# 5-loss streaks"),
    ("avg_mfe", "avg MFE (pts)"), ("avg_mae", "avg MAE (pts)"), ("mfe_mae", "MFE/MAE"),
    ("max_closed_dd", "max closed-trade DD $"), ("avg_dd", "avg DD $"),
    ("max_dd_range", "max DD date range"), ("recovery_trades", "recovery (trades) after worst DD"),
    ("ret_over_maxdd", "return / maxDD"), ("sharpe", "Sharpe"), ("sortino", "Sortino"),
    ("calmar", "Calmar"), ("trades_per_week", "trades/week"), ("trades_per_month", "trades/month"),
    ("max_trades_day", "max trades in a day"), ("avg_hold_min", "avg hold (min)"),
    ("median_hold_min", "median hold (min)"), ("max_hold_min", "max hold (min)"),
    ("min_hold_min", "min hold (min)"), ("winning_months", "winning months"),
    ("losing_months", "losing months"), ("winning_month_pct", "winning month %"),
    ("profitable_years", "profitable years"), ("losing_years", "losing years"),
    ("best_month", "best month"), ("worst_month", "worst month"),
    ("best_year", "best year"), ("worst_year", "worst year"),
]
w("| metric | idb | swd | portfolio |")
w("|---|--:|--:|--:|")
for kk, label in keys:
    def fmt(v: object) -> str:
        if isinstance(v, list):
            return f"{v[0]}:{v[1]:.0f}"
        return str(v)
    w(f"| {label} | {fmt(P['idb'].get(kk))} | {fmt(P['swd'].get(kk))} | {fmt(P['port'].get(kk))} |")
w()
w("*Intraday (MFE/MAE-aware) max DD: idb $1,933 · swd $1,223 · portfolio $2,145 (see Part 5).*")
w()

# ---- Part 3
w("## Part 3 — Year-by-year")
for k in ("idb", "swd", "port"):
    w(f"### {NAMES[k]}")
    w("| year | trades | net $ | PF | win% | expectancy | max DD $ | ret/DD | maxLstreak | best mo | worst mo |")
    w("|--:|--:|--:|--:|--:|--:|--:|--:|--:|--|--|")
    for r in YT[k]:
        w(f"| {r['year']} | {r['trades']} | {r['net']:.0f} | {r['pf']} | {r['win_rate']*100:.0f}% | "
          f"{r['expectancy']:.1f} | {r['max_dd']:.0f} | {r['ret_dd']} | {r['max_loss_streak']} | "
          f"{r['best_month']} | {r['worst_month']} |")
    w()

# ---- Part 4
w("## Part 4 — Month-by-month net $ (1 MNQ)")
w("Full per-finalist monthly detail (pnl / trades / win% / PF / maxDD) is in "
  "`phase30_monthly_detail.csv`. Combined monthly net below.")
months = sorted({m["month"] for m in MT["port"]})
idb_m = {m["month"]: m["pnl"] for m in MT["idb"]}
swd_m = {m["month"]: m["pnl"] for m in MT["swd"]}
port_m = {m["month"]: m["pnl"] for m in MT["port"]}
w("| month | idb $ | swd $ | port $ | | month | idb $ | swd $ | port $ |")
w("|--|--:|--:|--:|--|--|--:|--:|--:|")
half = (len(months) + 1) // 2
for i in range(half):
    a = months[i]
    rowa = f"| {a} | {idb_m.get(a,0):.0f} | {swd_m.get(a,0):.0f} | {port_m.get(a,0):.0f} |"
    if i + half < len(months):
        b = months[i + half]
        rowb = f" | {b} | {idb_m.get(b,0):.0f} | {swd_m.get(b,0):.0f} | {port_m.get(b,0):.0f} |"
    else:
        rowb = " | | | | |"
    w(rowa + rowb)
w()

# ---- Part 5
w("## Part 5 — 100K account / $3,000 drawdown feasibility")
w("**Drawdown formulas (printed exactly):**")
w("- **Static:** `floor = 100000 − 3000 = 97000` (fixed). Fail if intraday equity < 97000.")
w("- **Trailing:** `floor_t = max(equity τ≤t) − 3000` (trails the peak up, never down).")
w("- **Locked:** `floor_t = min( max(equity τ≤t) − 3000 , 100000 )` (trails until it reaches the "
  "starting balance, then locks at 100000).")
w("- Equity path includes the **intratrade low** (entry − MAE) so an open-position dip can breach. "
  "Peak uses post-trade (EOD-style) equity; intraday-peak trailing would be marginally stricter.")
w()
for model in ("static", "trailing", "locked"):
    w(f"### {model.capitalize()} model")
    w("| strategy | sizing | result | end equity $ | net $ | closed DD $ | intraday DD $ | min buffer $ | breach |")
    w("|---|---|:--:|--:|--:|--:|--:|--:|--|")
    for s in SIMS:
        if s["dd_model"] == model:
            res = "✅ PASS" if s["pass"] else "❌ FAIL"
            w(f"| {s['strategy']} | {s['sizing']} | {res} | {s['ending_equity']:.0f} | "
              f"{s['net']:.0f} | {s['closed_max_dd']:.0f} | {s['intraday_max_dd']:.0f} | "
              f"{s['min_buffer']:.0f} | {s['breach_date'] or '—'} |")
    w()
w("Worst rolling sequences (1 MNQ): idb worst-20 −$1,406 / worst-50 −$1,723; swd −$706 / −$637; "
  "portfolio −$1,199 / −$1,200. (Per-sizing values in `phase30_100k_account_sim.csv`.)")
w()

# ---- Part 6
w("## Part 6 — Monte Carlo / risk of ruin (2,000 sims)")
w("Two views: **reshuffle** destroys all sequencing (worst case, intratrade-MAE-aware); "
  "**month block-bootstrap** preserves intra-month structure (more realistic). Big gaps between "
  "them mean the historical low drawdown is partly *order-luck*.")
w()
w("| strategy | sizing | P(DD≥3K) reshuffle | P(DD≥3K) month-boot | exp maxDD | p95 | p99 | "
  "P(profit 60d) | P(profit 120d) | P(loss 6mo) |")
w("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|")
for m in MC:
    w(f"| {m['strategy']} | {m['sizing']} | {m['p_breach_3000']} | {m['p_breach_3000_boot']} | "
      f"${m['exp_max_dd']:.0f} | ${m['p95_max_dd']:.0f} | ${m['p99_max_dd']:.0f} | "
      f"{m['p_profit_60d']} | {m['p_profit_120d']} | {m['p_loss_6mo']} |")
w()
w("> **Read this carefully.** swd 1 MNQ is genuinely $3K-safe (reshuffle p99 maxDD **$2,969 < 3000**, "
  "month-boot P(breach)=0.01). idb 1 MNQ and the portfolio 1 MNQ are **marginal**: their historical "
  "DD passes ($1.9K), but reshuffle expected maxDD is **~$3,165** (p95 ~$3,900) and month-bootstrap "
  "breach prob ~11% — the low historical DD is partly favorable ordering. swd 2 MNQ is borderline "
  "(month-boot breach 36%). All are ~90% likely profitable after 60 trading days; ≤3% chance of a "
  "losing 6 months.")
w()

# ---- Part 7
w("## Part 7 — 5-year performance graphs")
for cap, f in [
    ("idb equity curve", "phase30_idb_equity_curve.png"),
    ("idb drawdown", "phase30_idb_drawdown_curve.png"),
    ("swd equity curve", "phase30_swd_equity_curve.png"),
    ("swd drawdown", "phase30_swd_drawdown_curve.png"),
    ("portfolio equity curve", "phase30_portfolio_equity_curve.png"),
    ("portfolio drawdown", "phase30_portfolio_drawdown_curve.png"),
    ("monthly PnL (all)", "phase30_monthly_pnl.png"),
    ("yearly PnL", "phase30_yearly_pnl.png"),
    ("rolling 20-trade expectancy / 50-trade PF / drawdown", "phase30_rolling_stats.png"),
    ("trade R distribution", "phase30_trade_distribution.png"),
    ("portfolio win/loss streaks", "phase30_streaks.png"),
]:
    w(f"**{cap}**")
    w(f"![{cap}]({f})")
    w()

# ---- Part 8
w("## Part 8 — Exports")
w("- `phase30_finalists_exact_stats.csv` — one row per finalist, all summary stats")
w("- `phase30_finalist_trades.csv` — every trade (prices, R, MFE/MAE, costs, equity/DD after trade)")
w("- `phase30_100k_account_sim.csv` — all 100K feasibility sims (3 models × sizings)")
w("- `phase30_monte_carlo.csv` — Monte Carlo results")
w("- `phase30_monthly_detail.csv` — per-finalist monthly pnl/trades/win%/PF/maxDD")
w("- this report: `phase30_finalist_report.md` + 11 PNGs")
w()

# ---- Part 9
def f2(v: object) -> str:
    return f"{v[0]}:{v[1]:.0f}" if isinstance(v, list) else str(v)

w("## Part 9 — Final answers (exact)")
w(f"1. **Trade count:** idb **221**, swd **244**, portfolio **465** (same-day EOD-flat trades, "
  "2021→2026-H1).")
w(f"2. **RR:** all **dynamic** (idb = hold-to-EOD, swd = target prior POC). Realized avg R (net): "
  f"idb **{P['idb']['avg_R']}**, swd **{P['swd']['avg_R']}**, port **{P['port']['avg_R']}**; "
  f"avg win {P['swd']['avg_win_R']}R / avg loss {P['swd']['avg_loss_R']}R (swd).")
w(f"3. **Max drawdown (1 MNQ):** closed-trade — idb **${P['idb']['max_closed_dd']:.0f}**, swd "
  f"**${P['swd']['max_closed_dd']:.0f}**, portfolio **${P['port']['max_closed_dd']:.0f}**.")
w("4. **Closed-trade or intraday?** Reported both. Closed-trade above; intraday (MFE/MAE-aware) is "
  "slightly higher: idb $1,933 · swd $1,223 · portfolio $2,145.")
w("5. **Survive 100K / $3K?** At **1 MNQ, yes** on closed+intraday history (all pass trailing). But "
  "Monte Carlo says only **swd is comfortably safe** (p99 maxDD $2,969); idb & portfolio are "
  "**marginal** (expected reshuffle maxDD ~$3,165). At **2 MNQ**: idb & portfolio FAIL trailing; swd "
  "passes history but is risky (month-boot breach 36%). 5 MNQ / 1 NQ: fail (except under the lenient "
  "locked model).")
w("6. **Safest sizing:** **swd_clean_path at 1 MNQ** (lowest DD, p99 < $3K). For idb/portfolio, "
  "1 MNQ is the max and is still marginal for a $3K *trailing* limit.")
w("7. **Best strategy:** **swd** for a tight $3K account (highest PF 1.535, ret/DD 7.9, lowest DD, "
  "safest MC). **Portfolio** for robustness (profitable all 6 years, 2× the $) but needs a larger DD "
  "buffer (~$4–5K) or the locked model. **idb** is the weakest standalone for a $3K limit (deep, "
  "order-dependent DD).")
w("8. **Forward paper ready?** **Yes** — swd 1 MNQ and the portfolio are worth forward-paper.")
w("9. **Live ready?** **No** — PF ~1.5 (not ≥2), DD risk is order-dependent (MC), 2026 is partial, "
  "and the strict grid walk-forward was weak. Forward-paper first.")
w("10. **Next best step:** forward-paper **swd 1 MNQ** (primary) and the **portfolio on a ≥$5K-DD or "
  "locked account**; then build 1m/3m bars for 2025-H2→2026 to test the finalists on faster "
  "timeframes, and add a third uncorrelated sleeve to cut the order-dependent DD.")

(OUT / "phase30_finalist_report.md").write_text("\n".join(L) + "\n")
print("wrote", OUT / "phase30_finalist_report.md", "lines", len(L))
