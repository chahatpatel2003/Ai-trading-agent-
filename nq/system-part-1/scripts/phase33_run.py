"""Phase 33 orchestrator — scalp leaderboard, fill/cost sensitivity, zero-cost edge test, graphs, report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from nqlab.discovery import phase33 as P33
from nqlab.research.scalp import add_scalp_outcomes

OUT = Path("docs/phase33")
TFS = [("15s", 0.25), ("30s", 0.5), ("1m", 1.0)]


def _load(tf: str, tfmin: float) -> pl.DataFrame:
    return add_scalp_outcomes(pl.read_parquet(OUT / f"bars_{tf}.parquet"), tf_min=tfmin, hold_min=15)


def main() -> None:
    frames = {tf: _load(tf, m) for tf, m in TFS}

    # 1. Leaderboard at the REALISTIC setting: 1-tick-through fill, 1x cost, 1 MNQ
    lb_rows: list[dict[str, Any]] = []
    for tf, m in TFS:
        for r in P33.evaluate(frames[tf], m, cost_mult=1.0, margin=1, stop_slip=1.0, instrument="MNQ"):
            lb_rows.append({"timeframe": tf, **r})
    lb = pl.DataFrame(lb_rows, infer_schema_length=None)
    lb.filter(pl.col("n") >= 200).sort("pf", descending=True).write_csv(OUT / "phase33_scalp_leaderboard.csv")

    # 2. Fill / cost / instrument sensitivity (best PF among n>=300, per setting) on 15s
    sens: list[dict[str, Any]] = []
    f15 = frames["15s"]
    settings = [
        ("touch", 0, 1.0, 1.0, "MNQ"), ("1tk_through", 1, 1.0, 1.0, "MNQ"),
        ("2tk_through", 2, 1.0, 1.0, "MNQ"), ("touch_2x", 0, 2.0, 1.0, "MNQ"),
        ("touch_3x", 0, 3.0, 1.0, "MNQ"), ("touch_4x", 0, 4.0, 1.0, "MNQ"),
        ("1tk_2x", 1, 2.0, 2.0, "MNQ"), ("touch_NQ", 0, 1.0, 1.0, "NQ"),
        ("touch_zerocost", 0, 0.0001, 0.0, "MNQ"),
    ]
    for name, mg, cm, ss, instr in settings:
        df = pl.DataFrame(
            P33.evaluate(f15, 0.25, cost_mult=cm, margin=mg, stop_slip=ss, instrument=instr),
            infer_schema_length=None,
        ).filter(pl.col("n") >= 300)
        b = df.sort("pf", descending=True).head(1).to_dicts()[0]
        sens.append({
            "setting": name, "best_pf": b["pf"], "family": b["family"], "dir": b["direction"],
            "target": b["target"], "stop": b["stop"], "win_rate": b["win_rate"],
            "breakeven_wr": b["breakeven_wr"], "edge_vs_be": b["edge_vs_be"],
            "n_pf_ge_1": int((df["pf"] >= 1.0).sum()), "n_pf_ge_14": int((df["pf"] >= 1.4).sum()),
        })
    pl.DataFrame(sens, infer_schema_length=None).write_csv(OUT / "phase33_fill_cost_sensitivity.csv")

    # 3. Best config economics on 1 MNQ (to confirm all account models fail by construction)
    best = lb.filter(pl.col("n") >= 200).sort("pf", descending=True).head(1).to_dicts()[0]

    # ---- graphs ----
    # A. best PF by fill/cost setting
    fig, ax = plt.subplots(figsize=(11, 5))
    s = pl.DataFrame(sens)
    ax.bar(s["setting"].to_list(), s["best_pf"].to_list(),
           color=["#2da44e" if v >= 1 else "#d1242f" for v in s["best_pf"].to_list()])
    ax.axhline(1.0, ls="--", color="black", lw=1, label="breakeven PF")
    ax.axhline(1.4, ls=":", color="grey", lw=1, label="target PF 1.4")
    ax.set_title("Best scalp PF by fill/cost setting (15s, all families, n≥300)")
    ax.set_ylabel("best PF")
    ax.set_xticklabels(s["setting"].to_list(), rotation=35, fontsize=8)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "phase33_fill_cost_pf.png", dpi=110)
    plt.close(fig)
    # B. win rate vs breakeven (zero-cost touch) — points on the diagonal = no edge
    zc = pl.DataFrame(
        P33.evaluate(f15, 0.25, cost_mult=0.0001, margin=0, stop_slip=0.0), infer_schema_length=None
    ).filter(pl.col("n") >= 300)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(zc["breakeven_wr"].to_list(), zc["win_rate"].to_list(), alpha=0.6, color="#1f6feb")
    lims = [0.3, 0.95]
    ax.plot(lims, lims, "--", color="black", label="no-edge line (wr = breakeven)")
    ax.set_xlabel("breakeven win rate (target/stop geometry)")
    ax.set_ylabel("actual win rate")
    ax.set_title("Zero-cost touch-fill: actual vs breakeven win rate\n(on the line = NO edge)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "phase33_winrate_vs_breakeven.png", dpi=110)
    plt.close(fig)

    (OUT / "_scalp.json").write_text(json.dumps(
        {"sensitivity": sens, "best_realistic": best}, indent=2, default=str))
    print("leaderboard rows:", lb.height)
    print("best realistic (1tk,1x,MNQ): PF", best["pf"], best["family"], best["direction"],
          f"t{best['target']}/s{best['stop']}", "net$", best["net"], "monthly$", best["monthly"])
    print("\nFILL/COST SENSITIVITY (best PF / setting):")
    for r in sens:
        print(f"  {r['setting']:16s} bestPF={r['best_pf']} wr={r['win_rate']} be={r['breakeven_wr']} "
              f"edge={r['edge_vs_be']} #PF>=1={r['n_pf_ge_1']} #PF>=1.4={r['n_pf_ge_14']}")


if __name__ == "__main__":
    main()
