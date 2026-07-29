"""Phase 32 orchestrator — build the conditional edge map, combos, bad map, prototypes, graphs, report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nqlab.backtest.engine import run_backtest
from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.discovery import phase29 as P29
from nqlab.discovery import phase32 as P32
from nqlab.domain.strategy.base import ExecConfig, StrategyParams, attach_signals

OUT = Path("docs/phase32")
OUT.mkdir(parents=True, exist_ok=True)
FWD = "/Volumes/T9/nq data/curated/artifacts/phase32/forward_5m.parquet"
_C = pl.col

LB_COLS = ["condition", "direction", "bucket", "n", "sample_flag", "mfe60", "mae60", "mfe_mae",
           "p10", "p20", "p30", "p50", "ret60_mean", "ret60_med", "clean", "chop", "exp20_cost",
           "dir_acc", "years_pos", "years_n", "months_pos_pct"]


def _combos(fr: pl.DataFrame) -> list[dict[str, Any]]:
    rth = _C("is_rth")
    combos = [
        ("gapup_unfilled+above_value", "short",
         rth & (_C("gap_dir") == 1) & (_C("gap_filled") == 0) & (_C("vp_pos") == 1)),
        ("sweep_pdh+vah_reject", "short",
         rth & (_C("high") > _C("prev_day_high")) & (_C("vp_vah_reject_dn") == 1)),
        ("sweep_low_reclaim+below_poc", "long",
         rth & (_C("swept_low") == 1) & (_C("reclaim_up") == 1) & (_C("vp_d_poc") < 0)),
        ("gapdn_unfilled+val_reclaim", "long",
         rth & (_C("gap_dir") == -1) & (_C("gap_filled") == 0) & (_C("vp_val_reclaim_up") == 1)),
        ("morning+above_value", "short",
         P32._window(570, 720) & (_C("vp_pos") == 1)),
        ("morning+below_value", "long",
         P32._window(570, 720) & (_C("vp_pos") == -1)),
        ("compression+or_break_up", "long",
         (_C("prev_day_type") == "compression") & (_C("close") > _C("or_high"))
         & (_C("close").shift(1) <= _C("or_high")) & P32._window(600, 900)),
        ("highvol+sweep_high_reclaim", "short",
         rth & (_C("atr_bucket") >= 3) & (_C("swept_high") == 1) & (_C("reclaim_dn") == 1)),
        ("midvol+open_above_value", "short",
         rth & (_C("atr_bucket").is_between(2, 3)) & (_C("vp_open_loc") == 1) & P32._window(570, 720)),
        ("displacement_up+below_poc", "long",
         rth & (_C("displacement") == 1) & (_C("close") > _C("open")) & (_C("vp_d_poc") < 0)),
        ("vah_reject+highvol", "short",
         rth & (_C("vp_vah_reject_dn") == 1) & (_C("atr_bucket") >= 3)),
        ("sweep_onh+above_value", "short",
         rth & (_C("high") > _C("overnight_high")) & (_C("close") < _C("overnight_high")) & (_C("vp_pos") == 1)),
    ]
    valid = fr.filter(_C("fwd_valid") == 1)
    rows = []
    for name, d, expr in combos:
        sub = valid.filter(expr.fill_null(value=False))
        p = P32._panel(sub, d)
        if p.get("n", 0) > 0:
            rows.append({"condition": name, "direction": d, "bucket": "combo", **p})
    return rows


def _prototype(fr: pl.DataFrame, name: str, expr: pl.Expr, direction: str, config: Any,
               rules: Any) -> dict[str, Any]:
    """Backtest a condition as: enter next open on condition, 2.5xATR stop, target prior POC, EOD."""
    atr = _C("atr")
    long = expr.fill_null(value=False) if direction == "long" else pl.lit(value=False)  # noqa: FBT003
    short = expr.fill_null(value=False) if direction == "short" else pl.lit(value=False)  # noqa: FBT003
    valid = atr.is_not_null() & (atr > 0) & _C("vp_prior_poc").is_not_null()
    long = long & valid & _C("is_rth")
    short = short & valid & _C("is_rth")
    stop = 2.5 * atr
    poc, close = _C("vp_prior_poc"), _C("close")
    clamp = lambda x: pl.min_horizontal(pl.max_horizontal(x, 0.5 * atr), 8.0 * atr)  # noqa: E731
    lt = pl.when(poc < close).then(clamp(close - poc)).otherwise(2.0 * stop)  # short targets POC below
    ltl = pl.when(poc > close).then(clamp(poc - close)).otherwise(2.0 * stop)  # long targets POC above
    target = pl.when(long).then(ltl).when(short).then(lt).otherwise(pl.lit(100000.0))
    sig = attach_signals(fr, long=long, short=short, stop_pts=stop, target_pts=target).filter(
        (_C("minute_of_day") >= 0) & (_C("minute_of_day") <= 960)
    )
    ex = ExecConfig(sizing_mode="mnq_fixed1", time_stop_bars=10_000)
    res = run_backtest(sig, symbol="NQ.c", timeframe="5m", strategy=StrategyParams("p32", "1"),
                       config=config, rules=rules, primary_cutoff=False, entry_mode="next_open",
                       gate_entries=False, exec_config=ex)
    tf = P29.trades_frame(res, type("C", (), {"name": name, "family": name, "kind": "p32"})())
    kept, _ = P29.intraday_only(tf)
    if kept.height == 0:
        return {"prototype": name, "trades": 0}
    nets = kept.sort("entry_ts")["net"].to_numpy()
    eq = np.cumsum(nets)
    mdd = float((np.maximum.accumulate(eq) - eq).max())
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    # 100K/$3K trailing pass with intratrade MAE
    realized = peak = 100000.0
    breach = False
    maes = kept.sort("entry_ts")["mae"].to_numpy() * 2.0
    for net, mae in zip(nets, maes, strict=True):
        if realized - mae < peak - 3000:
            breach = True
            break
        realized += net
        peak = max(peak, realized)
    return {
        "prototype": name, "direction": direction, "trades": len(nets),
        "pf": round(float(wins.sum() / -losses.sum()), 3) if losses.size else None,
        "win_rate": round(float((nets > 0).mean()), 4), "net": round(float(nets.sum()), 2),
        "expectancy": round(float(nets.mean()), 2), "max_dd": round(mdd, 2),
        "monthly_avg": round(float(nets.sum()) / 66, 2),
        "pass_100k_3k_1mnq": not breach,
    }


def _plots(df: pl.DataFrame) -> None:
    # 1. MFE/MAE by time of day
    tw = ["time_open_0930_0945", "time_0945_1000", "time_1000_1030", "time_1030_1100",
          "time_1100_1130", "time_1130_1200", "time_lunch_1200_1330", "time_nypm_1330_1500",
          "time_lasthour_1500_1600"]
    sub = df.filter((_C("condition").is_in(tw)) & (_C("direction") == "long"))
    order = {c: i for i, c in enumerate(tw)}
    sub = sub.sort(_C("condition").replace_strict(order, default=99))
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(sub.height)
    ax.bar(x - 0.2, sub["mfe60"].to_list(), 0.4, label="MFE 60m (up)", color="#2da44e")
    ax.bar(x + 0.2, sub["mae60"].to_list(), 0.4, label="MAE 60m (down)", color="#d1242f")
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("time_", "") for c in sub["condition"].to_list()], rotation=40, fontsize=8)
    ax.set_title("MFE vs MAE (60-min, points) by time of day — long perspective")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "phase32_mfe_mae_by_time.png", dpi=110)
    plt.close(fig)
    # 2. exp20 by vol regime
    vol = df.filter(_C("condition").str.starts_with("vol_atr_")).sort("condition")
    fig, ax = plt.subplots(figsize=(10, 5))
    for d, col in (("long", "#1f6feb"), ("short", "#d1242f")):
        s = vol.filter(_C("direction") == d)
        ax.plot(s["condition"].to_list(), s["exp20_cost"].to_list(), "o-", label=d, color=col)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Standardized 20pt 1:1 after-cost expectancy ($) by ATR volatility quintile")
    ax.set_xticklabels([c.replace("vol_atr_", "") for c in vol.filter(_C('direction')=='long')["condition"].to_list()], rotation=30, fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase32_vol_regime_edge.png", dpi=110)
    plt.close(fig)
    # 3. MFE/MAE leaderboard (top 20)
    top = df.filter(_C("n") >= 200).sort("mfe_mae", descending=True).head(20)
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = [f"{r['condition']}|{r['direction'][0]}" for r in top.to_dicts()]
    ax.barh(range(top.height), top["mfe_mae"].to_list(), color="#1f6feb")
    ax.axvline(0.94, ls="--", color="black", lw=1, label="baseline 0.94")
    ax.set_yticks(range(top.height))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Top conditions by MFE/MAE ratio (60-min, n>=200)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "phase32_mfe_mae_leaderboard.png", dpi=110)
    plt.close(fig)
    # 4. clean path top 15
    cp = df.filter(_C("n") >= 200).sort("clean", descending=True).head(15)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(range(cp.height), cp["clean"].to_list(), color="#8250df")
    ax.set_yticks(range(cp.height))
    ax.set_yticklabels([f"{r['condition']}|{r['direction'][0]}" for r in cp.to_dicts()], fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Top conditions by clean-path score (n>=200)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "phase32_clean_path.png", dpi=110)
    plt.close(fig)


def main() -> None:
    config = load_config(profile="dev", overrides=P29.PHASE29_OVERRIDES)
    rules = load_topstep_rules("topstep_150k")
    fr = P32.add_regime_buckets(pl.read_parquet(FWD))

    cond_rows = P32.evaluate(fr, P32.conditions())
    cdf = pl.DataFrame(cond_rows, infer_schema_length=None)
    cdf.sort("mfe_mae", descending=True).select(LB_COLS).write_csv(OUT / "phase32_condition_leaderboard.csv")

    combo_rows = _combos(fr)
    bdf = pl.DataFrame(combo_rows, infer_schema_length=None)
    bdf.sort("mfe_mae", descending=True).select([c for c in LB_COLS if c in bdf.columns]).write_csv(
        OUT / "phase32_combo_leaderboard.csv")

    # bad map: robust, low asymmetry / high chop / negative expectancy
    bad = cdf.filter((_C("n") >= 200) & ((_C("exp20_cost") < -5) | (_C("chop") > 0.7) | (_C("mfe_mae") < 0.9)))
    bad.sort("exp20_cost").select(LB_COLS).write_csv(OUT / "phase32_bad_conditions.csv")

    # slim forward outcomes export
    keep = ["ts", "trading_day", "minute_of_day", "is_rth", "atr", "fwd_valid", "fwd_up_60",
            "fwd_dn_60", "fwd_ret_60", "fwd_up_eod", "fwd_dn_eod", "fwd_L20", "fwd_S20"]
    fr.select([c for c in keep if c in fr.columns]).write_parquet(OUT / "phase32_forward_outcomes.parquet")

    # prototypes from top extension-reversion conditions
    rth = _C("is_rth")
    protos = [
        ("gap_up_unfilled_short_toPOC", rth & (_C("gap_dir") == 1) & (_C("gap_filled") == 0)
         & (_C("vp_pos") == 1) & P32._window(570, 720), "short"),
        ("sweep_pdh_short_toPOC", rth & (_C("high") > _C("prev_day_high"))
         & (_C("close") < _C("prev_day_high")), "short"),
        ("vah_reject_short_toPOC", rth & (_C("vp_vah_reject_dn") == 1), "short"),
        ("val_reclaim_long_toPOC", rth & (_C("vp_val_reclaim_up") == 1), "long"),
        ("sweep_low_reclaim_long_toPOC", rth & (_C("swept_low") == 1) & (_C("reclaim_up") == 1), "long"),
    ]
    proto_rows = [_prototype(fr, n, e, d, config, rules) for n, e, d in protos]
    pl.DataFrame(proto_rows, infer_schema_length=None).write_csv(OUT / "phase32_strategy_prototypes.csv")

    _plots(cdf)
    (OUT / "_edge.json").write_text(json.dumps(
        {"conditions": cond_rows, "combos": combo_rows, "prototypes": proto_rows}, indent=2, default=str))

    print("conditions:", cdf.height, "| combos:", bdf.height, "| bad:", bad.height)
    print("\nPROTOTYPES:")
    for r in proto_rows:
        if r.get("trades", 0) > 0:
            print(f"  {r['prototype']:32s} n={r['trades']} pf={r.get('pf')} wr={r.get('win_rate')} "
                  f"net=${r.get('net')} $/mo=${r.get('monthly_avg')} DD=${r.get('max_dd')} "
                  f"pass3k={r.get('pass_100k_3k_1mnq')}")


if __name__ == "__main__":
    main()
