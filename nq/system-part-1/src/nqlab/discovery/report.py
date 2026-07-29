"""Discovery reporting: leaderboard, rejection reasons, and parameter zones."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from nqlab.discovery.harness import DiscoveryResult

_VERDICT_STYLE = {"ACCEPT": "green", "REJECT": "red", "NEEDS_MORE_DATA": "yellow"}


def _f(row: dict[str, Any], key: str) -> float:
    return float(row.get(key) or 0.0)


def run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compact summary of a run's rows (for cross-run comparison)."""
    counts = Counter(str(r.get("final_verdict")) for r in rows)
    best_score = max((_f(r, "objective_score") for r in rows), default=0.0)
    best_oos = max((_f(r, "oos_net") for r in rows), default=0.0)
    return {
        "trials": len(rows), "accept": counts["ACCEPT"], "reject": counts["REJECT"],
        "needs_more_data": counts["NEEDS_MORE_DATA"], "best_oos": round(best_oos, 2),
        "best_score": round(best_score, 2),
    }


def leaderboard_table(rows: list[dict[str, Any]], top: int = 20) -> Table:
    table = Table(title=f"discovery leaderboard (top {top})", title_justify="left")
    for col in ("#", "family", "verdict", "oos_net", "ruin%", "wf+", "DSR", "score"):
        table.add_column(col, justify="left" if col in ("family", "verdict") else "right")
    for i, r in enumerate(rows[:top], 1):
        verdict = str(r.get("final_verdict", "?"))
        style = _VERDICT_STYLE.get(verdict, "white")
        table.add_row(
            str(i), str(r.get("family")), f"[{style}]{verdict[:6]}[/]",
            f"{_f(r, 'oos_net'):.0f}", f"{_f(r, 'risk_of_ruin') * 100:.0f}",
            f"{_f(r, 'walk_forward_frac') * 100:.0f}", f"{_f(r, 'deflated_sharpe'):.2f}",
            f"{_f(r, 'objective_score'):.2f}",
        )
    return table


def parameter_zones(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Per family/param: the value with the best mean OOS, and the most unstable value."""
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_family[str(r.get("family"))].append(r)

    zones: dict[str, dict[str, dict[str, Any]]] = {}
    for family, family_rows in by_family.items():
        by_param: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in family_rows:
            for key, val in json.loads(r.get("params") or "{}").items():
                by_param[key][str(val)].append(_f(r, "oos_net"))
        family_zones: dict[str, dict[str, Any]] = {}
        for key, values in by_param.items():
            means = {val: sum(xs) / len(xs) for val, xs in values.items()}
            disp = {
                val: (statistics.pstdev(xs) if len(xs) > 1 else 0.0)
                for val, xs in values.items()
            }
            best = max(means, key=lambda v: means[v])
            unstable = max(disp, key=lambda v: disp[v])
            family_zones[key] = {
                "best_value": best,
                "best_mean_oos": round(means[best], 1),
                "unstable_value": unstable,
            }
        zones[family] = family_zones
    return zones


def _rejection_and_zones(console: Console, rows: list[dict[str, Any]]) -> None:
    reasons: Counter[str] = Counter()
    for r in rows:
        if r.get("final_verdict") == "REJECT":
            first = str(r.get("rejection_reason") or "").split(";")[0].strip()
            if first:
                reasons[first] += 1
    if reasons:
        rt = Table(title="top rejection reasons", title_justify="left")
        rt.add_column("reason")
        rt.add_column("count", justify="right")
        for reason, count in reasons.most_common(8):
            rt.add_row(reason[:64], str(count))
        console.print(rt)

    zt = Table(title="best / unstable parameter zones (by mean OOS net)", title_justify="left")
    for col in ("family", "param", "best value", "mean OOS", "unstable value"):
        zt.add_column(col)
    for family, family_zones in parameter_zones(rows).items():
        for key, zone in family_zones.items():
            zt.add_row(
                family, key, str(zone["best_value"]),
                str(zone["best_mean_oos"]), str(zone["unstable_value"]),
            )
    console.print(zt)


def print_run_rows(console: Console, rows: list[dict[str, Any]], run: dict[str, Any]) -> None:
    """Render a leaderboard report from stored rows + run metadata."""
    console.print(leaderboard_table(rows, 20))
    pbo = "n/a" if run.get("pbo") is None else f"{float(run['pbo']):.1%}"
    counts = Counter(str(r["final_verdict"]) for r in rows)
    console.print(
        f"trials={run.get('n_trials')} PBO={pbo} | "
        f"[green]ACCEPT={counts['ACCEPT']}[/green] [red]REJECT={counts['REJECT']}[/red] "
        f"[yellow]NEEDS_MORE_DATA={counts['NEEDS_MORE_DATA']}[/yellow]"
    )
    _rejection_and_zones(console, rows)


def print_discovery(console: Console, result: DiscoveryResult, top: int = 20) -> None:
    console.print(leaderboard_table(result.leaderboard, top))
    pbo = "n/a" if result.pbo is None else f"{result.pbo:.1%}"
    console.print(
        f"trials={result.n_trials} | SR0 (expected max under null)={result.sr0:.3f} | PBO={pbo}"
    )
    console.print(
        f"[green]accepted:[/green] {len(result.accepted)} | "
        f"[red]rejected:[/red] {len(result.rejected)}"
    )

    reasons: Counter[str] = Counter()
    for r in result.rejected:
        first = str(r.get("rejection_reason") or "").split(";")[0].strip()
        if first:
            reasons[first] += 1
    if reasons:
        rt = Table(title="top rejection reasons", title_justify="left")
        rt.add_column("reason")
        rt.add_column("count", justify="right")
        for reason, count in reasons.most_common(8):
            rt.add_row(reason[:64], str(count))
        console.print(rt)

    zones = parameter_zones(result.leaderboard)
    zt = Table(title="best / unstable parameter zones (by mean OOS net)", title_justify="left")
    for col in ("family", "param", "best value", "mean OOS", "unstable value"):
        zt.add_column(col)
    for family, family_zones in zones.items():
        for key, zone in family_zones.items():
            zt.add_row(
                family, key, str(zone["best_value"]),
                str(zone["best_mean_oos"]), str(zone["unstable_value"]),
            )
    console.print(zt)
