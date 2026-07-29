"""Validation report: serialize and pretty-print a ValidationResult."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.table import Table

from nqlab.validation.gauntlet import ValidationResult

_Metrics = dict[str, object]

_VERDICT_STYLE = {"ACCEPT": "green", "REJECT": "red", "NEEDS_MORE_DATA": "yellow"}


def to_dict(result: ValidationResult) -> dict[str, object]:
    return asdict(result)


def write_validation_report(result: ValidationResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "validation.json"
    path.write_text(json.dumps(asdict(result), indent=2, default=str))
    return path


def print_validation_dict(console: Console, d: dict[str, object]) -> None:
    """Render a validation report (from a ValidationResult dict)."""
    verdict = str(d.get("verdict", "?"))
    style = _VERDICT_STYLE.get(verdict, "white")
    full = cast(_Metrics, d["full"])
    oos = cast(_Metrics, d["out_sample"])
    ins = cast(_Metrics, d["in_sample"])
    mc = cast(_Metrics, d["monte_carlo"])

    console.print(
        f"\n[bold]{d['strategy']} v{d['version']}[/bold] on {d['symbol']} "
        f"({d['timeframe']}, Topstep {d['account']})  →  [{style}]{verdict}[/{style}]"
    )

    summary = Table(show_header=True, title="key metrics", title_justify="left")
    summary.add_column("window")
    for col in ("net_pnl", "trades", "win_rate", "profit_factor", "max_dd"):
        summary.add_column(col, justify="right")

    def _row(label: str, m: _Metrics) -> None:
        summary.add_row(
            label, str(m.get("net_pnl")), str(m.get("trades")),
            str(m.get("win_rate")), str(m.get("profit_factor")),
            str(m.get("max_drawdown")),
        )

    _row("full", full)
    _row("in-sample", ins)
    _row("out-sample", oos)
    console.print(summary)

    yearly = d.get("yearly", {})
    if isinstance(yearly, dict) and yearly:
        yt = Table(title="yearly", title_justify="left")
        yt.add_column("year")
        yt.add_column("net_pnl", justify="right")
        yt.add_column("trades", justify="right")
        for year, m in yearly.items():
            yt.add_row(year, str(m.get("net_pnl")), str(m.get("trades")))
        console.print(yt)

    stress = d.get("cost_stress", [])
    if isinstance(stress, list) and stress:
        st = Table(title="cost / slippage stress", title_justify="left")
        st.add_column("scenario")
        st.add_column("net_pnl", justify="right")
        st.add_column("blown", justify="center")
        for s in stress:
            st.add_row(str(s["label"]), str(s["net_pnl"]), "Y" if s["blown"] else "·")
        console.print(st)

    wf = cast(float, d.get("walk_forward_positive_frac", 0.0))
    ruin = cast(float, mc["blow_up_prob"])
    console.print(
        f"walk-forward positive: {wf:.0%} | blow-up prob (risk-of-ruin): {ruin:.1%} | "
        f"unrealistic-fills dependent: {d.get('unrealistic_fills')}"
    )
    reasons = d.get("reasons", [])
    if isinstance(reasons, list):
        joined = "; ".join(str(r) for r in reasons)
        console.print(f"[{style}]verdict: {verdict}[/{style}] — {joined}")


def print_validation(console: Console, result: ValidationResult) -> None:
    print_validation_dict(console, asdict(result))
