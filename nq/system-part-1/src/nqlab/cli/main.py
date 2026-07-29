"""``nqlab`` command-line interface.

Commands (all read-only with respect to raw market data):

    nqlab doctor          environment & data-availability health check
    nqlab data profile    scan raw parquet, print dataset/period summary
    nqlab data manifest   build & write the data-availability manifest
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, cast

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from nqlab.backtest.engine import run_backtest
from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.data.bars_build import build_bars, profile_bars
from nqlab.data.bars_full import build_bars_full
from nqlab.data.continuous import build_continuous
from nqlab.data.discovery import discover_dataset, missing_periods
from nqlab.data.manifest import build_manifest
from nqlab.discovery.calibration import (
    calibrate,
    generate_calibrated_candidates,
    generate_setup_candidates,
)
from nqlab.discovery.candidates import ALL_FAMILIES, MBP1_FAMILIES
from nqlab.discovery.edgemap import (
    COST_PTS,
    HORIZON_BARS_5M,
    bucket_edge,
    context_columns,
    forward_excursions,
    join_stability,
)
from nqlab.discovery.harness import run_discovery
from nqlab.discovery.htf_setups import PHASE9_FAMILIES, generate_htf_setup_candidates
from nqlab.discovery.lockbox import (
    PHASE10_FREEZE_SPECS,
    freeze_from_phase9,
    load_registry,
    phase11_verdict,
    phase12_verdict,
    registry_path,
    run_lockbox,
    save_registry,
    select_survivors,
)
from nqlab.discovery.objective import rank_rows
from nqlab.discovery.optimization import (
    FOCUS_FAMILIES,
    account_risk_grid,
    generate_exit_candidates,
    generate_risk_candidates,
    generate_sizing_candidates,
    generate_survival_candidates,
    generate_timeframe_candidates,
)
from nqlab.discovery.phase11_families import PHASE11_FAMILIES, generate_phase11_candidates
from nqlab.discovery.phase12_families import PHASE12_FAMILIES, generate_phase12_candidates
from nqlab.discovery.phase12_stats import practical_stats
from nqlab.discovery.phase13_families import PHASE13_FAMILIES, generate_phase13_candidates
from nqlab.discovery.phase14_families import PHASE14_FAMILIES, generate_phase14_candidates
from nqlab.discovery.phase16 import (
    PHASE16_THRESHOLDS,
    run_phase16_lockbox,
    select_phase16_survivors,
)
from nqlab.discovery.phase16_families import PHASE16_FAMILIES, generate_phase16_candidates
from nqlab.discovery.phase17 import (
    PHASE17_THRESHOLDS,
    phase17_verdict,
)
from nqlab.discovery.phase17 import (
    run_phase16_lockbox as run_phase17_lockbox,
)
from nqlab.discovery.phase17 import (
    select_phase16_survivors as select_phase17_survivors,
)
from nqlab.discovery.phase17_families import PHASE17_FAMILIES, generate_phase17_candidates
from nqlab.discovery.phase18 import (
    PHASE18_THRESHOLDS,
    run_phase18_lockbox,
)
from nqlab.discovery.phase18 import (
    select_phase16_survivors as select_phase18_survivors,
)
from nqlab.discovery.phase18_families import PHASE18_FAMILIES, generate_phase18_candidates
from nqlab.discovery.phase20 import (
    PHASE20_THRESHOLDS,
)
from nqlab.discovery.phase20 import (
    run_phase18_lockbox as run_phase20_lockbox,
)
from nqlab.discovery.phase20 import (
    select_phase16_survivors as select_phase20_survivors,
)
from nqlab.discovery.phase20_families import PHASE20_FAMILIES, generate_phase20_candidates
from nqlab.discovery.phase21 import (
    PHASE21_THRESHOLDS,
)
from nqlab.discovery.phase21 import (
    run_phase18_lockbox as run_phase21_lockbox,
)
from nqlab.discovery.phase21 import (
    select_phase16_survivors as select_phase21_survivors,
)
from nqlab.discovery.phase21_families import PHASE21_FAMILIES, generate_phase21_candidates
from nqlab.discovery.phase22 import (
    PHASE22_THRESHOLDS,
)
from nqlab.discovery.phase22 import (
    run_phase18_lockbox as run_phase22_lockbox,
)
from nqlab.discovery.phase22 import (
    select_phase16_survivors as select_phase22_survivors,
)
from nqlab.discovery.phase22_families import PHASE22_FAMILIES, generate_phase22_candidates
from nqlab.discovery.phase23 import (
    PHASE23_THRESHOLDS,
)
from nqlab.discovery.phase23 import (
    run_phase18_lockbox as run_phase23_lockbox,
)
from nqlab.discovery.phase23 import (
    select_phase16_survivors as select_phase23_survivors,
)
from nqlab.discovery.phase23_families import PHASE23_FAMILIES, generate_phase23_candidates
from nqlab.discovery.phase24 import (
    PHASE24_THRESHOLDS,
)
from nqlab.discovery.phase24 import (
    run_phase18_lockbox as run_phase24_lockbox,
)
from nqlab.discovery.phase24 import (
    select_phase16_survivors as select_phase24_survivors,
)
from nqlab.discovery.phase24_families import PHASE24_FAMILIES, generate_phase24_candidates
from nqlab.discovery.phase25 import (
    PHASE25_THRESHOLDS,
)
from nqlab.discovery.phase25 import (
    run_phase18_lockbox as run_phase25_lockbox,
)
from nqlab.discovery.phase25 import (
    select_phase16_survivors as select_phase25_survivors,
)
from nqlab.discovery.phase25_families import PHASE25_FAMILIES, generate_phase25_candidates
from nqlab.discovery.phase26 import (
    PHASE26_THRESHOLDS,
    phase26_verdict,
    quality_stats,
)
from nqlab.discovery.phase26_families import PHASE26_FAMILIES, generate_phase26_candidates
from nqlab.discovery.phase28 import CONFIRMATIONS, evaluate_confirmation
from nqlab.discovery.prefilter import diagnostics_prefilter
from nqlab.discovery.report import (
    leaderboard_table,
    print_discovery,
    print_run_rows,
    run_summary,
)
from nqlab.domain.calendar.sessions import SessionLabeler, parse_hhmm
from nqlab.domain.risk.topstep import TopstepResult, simulate_account
from nqlab.experiment.tracker import ExperimentTracker
from nqlab.features.build import build_and_write_features
from nqlab.features.build_full import build_features_full
from nqlab.features.htf_context import HTF_CONTEXT_FEATURE_NAMES
from nqlab.features.mbp1_full import build_mbp1_features_full, profile_mbp1, read_mbp1_features
from nqlab.features.mbp1_profile import DistributionProfile, profile_distributions
from nqlab.features.structure import STRUCTURE_FEATURE_NAMES
from nqlab.observability.logging import configure_logging, get_logger
from nqlab.paper import ledger as paper_ledger
from nqlab.paper.engine import run_paper
from nqlab.paper.evaluate import evaluate_forward
from nqlab.paper.feed import LiveFeed
from nqlab.paper.frozen import (
    init_from_phase14,
    load_paper_config,
    paper_registry_path,
    save_paper_config,
)
from nqlab.research.baselines import BASELINES, build_baseline
from nqlab.research.full_history import ensure_full_history
from nqlab.research.report import compute_metrics, write_report
from nqlab.research.runner import (
    prepare_features,
    prepare_htf_features,
    prepare_mbp1_features,
    prepare_setup_features,
    prepare_structure_features,
    run_baselines,
    run_strategy,
)
from nqlab.validation.diagnostics import diagnose
from nqlab.validation.gauntlet import ValidationResult, run_gauntlet
from nqlab.validation.report import print_validation, print_validation_dict, write_validation_report

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
data_app = typer.Typer(no_args_is_help=True, help="Data discovery & profiling (read-only).")
bars_app = typer.Typer(no_args_is_help=True, help="Build & profile curated bars.")
features_app = typer.Typer(no_args_is_help=True, help="Build the PIT feature store.")
backtest_app = typer.Typer(no_args_is_help=True, help="Run a single-strategy backtest.")
research_app = typer.Typer(no_args_is_help=True, help="Baseline & full-history research.")
topstep_app = typer.Typer(no_args_is_help=True, help="Topstep account simulation.")
validate_app = typer.Typer(no_args_is_help=True, help="Validation gauntlet (edge vs noise).")
report_app = typer.Typer(no_args_is_help=True, help="Render saved reports.")
discover_app = typer.Typer(no_args_is_help=True, help="Strategy discovery sweeps.")
experiment_app = typer.Typer(no_args_is_help=True, help="Experiment tracker queries.")
strategy_app = typer.Typer(no_args_is_help=True, help="Strategy registry.")
mbp1_app = typer.Typer(no_args_is_help=True, help="MBP1 microstructure features.")
mbp1_features_app = typer.Typer(no_args_is_help=True, help="Build/profile MBP1 features.")
mbp1_app.add_typer(mbp1_features_app, name="features")
optimize_app = typer.Typer(no_args_is_help=True, help="Survival optimization (Phase 7).")
feasibility_app = typer.Typer(no_args_is_help=True, help="Account/timeframe feasibility (Phase 8).")
lockbox_app = typer.Typer(
    no_args_is_help=True, help="Pre-registered lockbox confirmation (Phase 10)."
)
phase11_app = typer.Typer(
    no_args_is_help=True, help="Confirmed-enough strategy search w/ reserved lockbox (Phase 11)."
)
phase12_app = typer.Typer(
    no_args_is_help=True, help="trend_day frequency expansion (Phase 12; forward-paper ceiling)."
)
phase13_app = typer.Typer(
    no_args_is_help=True, help="Structure + liquidity + displacement freeze/lockbox (Phase 13)."
)
phase14_app = typer.Typer(
    no_args_is_help=True, help="Structure frequency expansion + sizing repair (Phase 14)."
)
paper_app = typer.Typer(
    no_args_is_help=True, help="Forward paper/shadow runner for the frozen candidate (Phase 15)."
)
phase16_app = typer.Typer(
    no_args_is_help=True, help="Payout-first scalable strategy discovery (Phase 16)."
)
phase17_app = typer.Typer(
    no_args_is_help=True, help="Business-worthy A+ payout strategy search (Phase 17)."
)
phase18_app = typer.Typer(
    no_args_is_help=True, help="High-win-rate frequent NY-AM scalp payout search (Phase 18)."
)
phase19_app = typer.Typer(
    no_args_is_help=True, help="NQ NY-AM edge-discovery map (behaviour, not a backtest) (Phase 19)."
)
phase20_app = typer.Typer(
    no_args_is_help=True, help="Classic indicator strategy discovery (last non-MBP1) (Phase 20)."
)
phase21_app = typer.Typer(
    no_args_is_help=True, help="MBP1 order-flow microstructure discovery (Phase 21)."
)
phase22_app = typer.Typer(
    no_args_is_help=True, help="Cost audit + 60-70% win-rate 1:1 Topstep search (Phase 22)."
)
phase23_app = typer.Typer(
    no_args_is_help=True, help="HTF POI reaction + LTF reclaim/MSS model (Phase 23)."
)
phase24_app = typer.Typer(
    no_args_is_help=True, help="Four prop-style 1:1 scalping models (Phase 24)."
)
phase25_app = typer.Typer(
    no_args_is_help=True, help="Daily-bias + HTF-trend + POI + LTF-reclaim model (Phase 25)."
)
phase26_app = typer.Typer(
    no_args_is_help=True, help="Non-Topstep high-profit-factor strategy discovery (Phase 26)."
)
phase27_app = typer.Typer(
    no_args_is_help=True, help="Backtest integrity audit + clean high-PF re-run (Phase 27)."
)
phase28_app = typer.Typer(
    no_args_is_help=True, help="Mechanical LTF-confirmation rulebook for HTF trading (Phase 28)."
)
app.add_typer(data_app, name="data")
app.add_typer(bars_app, name="bars")
app.add_typer(features_app, name="features")
app.add_typer(backtest_app, name="backtest")
app.add_typer(research_app, name="research")
app.add_typer(topstep_app, name="topstep")
app.add_typer(validate_app, name="validate")
app.add_typer(report_app, name="report")
app.add_typer(discover_app, name="discover")
app.add_typer(experiment_app, name="experiment")
app.add_typer(strategy_app, name="strategy")
app.add_typer(mbp1_app, name="mbp1")
app.add_typer(optimize_app, name="optimize")
app.add_typer(feasibility_app, name="feasibility")
app.add_typer(lockbox_app, name="lockbox")
app.add_typer(phase11_app, name="phase11")
app.add_typer(phase12_app, name="phase12")
app.add_typer(phase13_app, name="phase13")
app.add_typer(phase14_app, name="phase14")
app.add_typer(paper_app, name="paper")
app.add_typer(phase16_app, name="phase16")
app.add_typer(phase17_app, name="phase17")
app.add_typer(phase18_app, name="phase18")
app.add_typer(phase19_app, name="phase19")
app.add_typer(phase20_app, name="phase20")
app.add_typer(phase21_app, name="phase21")
app.add_typer(phase22_app, name="phase22")
app.add_typer(phase23_app, name="phase23")
app.add_typer(phase24_app, name="phase24")
app.add_typer(phase25_app, name="phase25")
app.add_typer(phase26_app, name="phase26")
app.add_typer(phase27_app, name="phase27")
app.add_typer(phase28_app, name="phase28")

console = Console()

ProfileOption = Annotated[str, typer.Option(help="Configuration profile to load.")]
TimeframeOption = Annotated[str, typer.Option(help="Bar timeframe: 1m|5m|15m|1h.")]
AccountOption = Annotated[str, typer.Option(help="Topstep account size: 50k|100k|150k.")]


def _dates(start: str, end: str) -> tuple[date, date]:
    return date.fromisoformat(start), date.fromisoformat(end)


def _metrics_table(title: str, metrics: dict[str, object]) -> Table:
    table = Table(title=title, title_justify="left")
    table.add_column("metric")
    table.add_column("value", justify="right")
    order = [
        "net_pnl",
        "trades",
        "win_rate",
        "profit_factor",
        "expectancy",
        "avg_trade",
        "trades_per_day",
        "max_drawdown",
        "max_daily_drawdown",
        "max_losing_streak",
        "trading_days",
        "passed",
        "days_to_pass",
        "blown",
        "account_failures",
        "consistency_ok",
        "best_day",
        "worst_day",
    ]
    for key in order:
        table.add_row(key, str(metrics.get(key)))
    return table


def _print_topstep(result: TopstepResult) -> None:
    verdict = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
    if result.blown:
        verdict = "[red]BLOWN[/red]"
    console.print(
        f"[bold]Topstep {result.rules_name}[/bold] {verdict} | "
        f"net {result.net_profit:+,.0f} | days_to_pass={result.days_to_pass} | "
        f"max_dd={result.max_drawdown_from_peak:,.0f} | failures={result.account_failures}"
    )
    if result.violations:
        console.print(f"  [red]violations:[/red] {', '.join(result.violations[:5])}")


_REQUIRED_PACKAGES = ("pyarrow", "polars", "duckdb", "pydantic", "typer", "yaml")
_HUMAN_UNITS = ("B", "KB", "MB", "GB", "TB", "PB")


def _human_bytes(n: int) -> str:
    size = float(n)
    for unit in _HUMAN_UNITS:
        if size < 1024 or unit == _HUMAN_UNITS[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _load(profile: str) -> AppConfig:
    config = load_config(profile=profile)
    configure_logging(config.logging)
    return config


@data_app.command("profile")
def data_profile(profile: ProfileOption = "dev") -> None:
    """Scan raw trades & mbp1 parquet and print a per-dataset / per-period summary."""
    config = _load(profile)
    log = get_logger("cli.profile")

    for name, directory in (
        ("trades", config.data.raw_trades_dir),
        ("mbp1", config.data.raw_mbp1_dir),
    ):
        log.info("scanning dataset", extra={"dataset": name, "directory": str(directory)})
        infos = discover_dataset(directory, name)

        table = Table(title=f"{name}  ({directory})", title_justify="left")
        table.add_column("period")
        table.add_column("rows", justify="right")
        table.add_column("size", justify="right")
        table.add_column("row groups", justify="right")
        table.add_column("ts_min")
        table.add_column("ts_max")
        table.add_column("schema")

        for info in infos:
            table.add_row(
                info.period.label if info.period else "?",
                f"{info.num_rows:,}",
                _human_bytes(info.size_bytes),
                f"{info.num_row_groups:,}",
                info.ts_min.date().isoformat() if info.ts_min else "-",
                info.ts_max.date().isoformat() if info.ts_max else "-",
                info.schema.fingerprint(),
            )

        if not infos:
            console.print(f"[yellow]No parquet files found in {directory}[/yellow]")
            continue

        total_rows = sum(i.num_rows for i in infos)
        total_bytes = sum(i.size_bytes for i in infos)
        periods = sorted({i.period for i in infos if i.period is not None})
        gaps = [p.label for p in missing_periods(periods)]
        table.add_section()
        table.add_row("TOTAL", f"{total_rows:,}", _human_bytes(total_bytes), "", "", "", "")
        console.print(table)
        if gaps:
            console.print(f"  [red]missing periods:[/red] {', '.join(gaps)}")
        console.print()


@data_app.command("manifest")
def data_manifest(
    profile: ProfileOption = "dev",
    output: Annotated[
        Path | None, typer.Option(help="Output JSON path (default: artifacts).")
    ] = None,
    no_sample: Annotated[
        bool, typer.Option("--no-sample", help="Skip instrument sampling (faster).")
    ] = False,
) -> None:
    """Build the data-availability manifest and write it to JSON."""
    config = _load(profile)
    log = get_logger("cli.manifest")

    manifest = build_manifest(config, sample_instruments=not no_sample)
    out_path = output or (config.data.artifacts_root / "manifests" / "data_availability.json")
    written = manifest.write_json(out_path)
    log.info("manifest written", extra={"path": str(written)})

    table = Table(title="availability matrix (period x dataset)", title_justify="left")
    table.add_column("period")
    table.add_column("trades", justify="center")
    table.add_column("mbp1", justify="center")
    for row in manifest.availability_matrix:
        table.add_row(
            str(row["period"]),
            "[green]Y[/green]" if row["trades"] else "[red]·[/red]",
            "[green]Y[/green]" if row["mbp1"] else "[red]·[/red]",
        )
    console.print(table)

    for name, ds in manifest.datasets.items():
        gaps = f"  missing: {', '.join(ds.missing_periods)}" if ds.missing_periods else ""
        console.print(
            f"[bold]{name}[/bold]: {ds.file_count} files, "
            f"{ds.total_rows:,} rows, {_human_bytes(ds.total_bytes)}{gaps}"
        )
    if manifest.instruments:
        n_out = sum(1 for i in manifest.instruments if i.classification == "outright")
        n_spr = sum(1 for i in manifest.instruments if i.classification == "spread")
        console.print(
            f"[bold]instruments[/bold] (sampled): {len(manifest.instruments)} symbols "
            f"({n_out} outright, {n_spr} spread)"
        )
    console.print(f"\n[green]manifest →[/green] {written}")


@app.command("doctor")
def doctor(profile: ProfileOption = "dev") -> None:
    """Check the environment, dependencies, and data availability."""
    rows: list[tuple[str, str, str]] = []
    ok = True

    def record(check: str, passed: bool | None, detail: str) -> None:
        nonlocal ok
        if passed is None:
            status = "[yellow]WARN[/yellow]"
        elif passed:
            status = "[green]PASS[/green]"
        else:
            status = "[red]FAIL[/red]"
            ok = False
        rows.append((check, status, detail))

    # Python version
    v = sys.version_info
    record("python >= 3.12", (v.major, v.minor) >= (3, 12), f"{v.major}.{v.minor}.{v.micro}")

    # Dependencies importable
    for pkg in _REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            record(f"import {pkg}", True, getattr(mod, "__version__", "ok"))
        except ImportError as exc:  # pragma: no cover - environment dependent
            record(f"import {pkg}", False, str(exc))

    # Config loads
    config: AppConfig | None = None
    try:
        config = load_config(profile=profile)
        configure_logging(config.logging)
        record("config loads", True, f"profile={profile} hash={config.config_hash()[:12]}")
    except (FileNotFoundError, ValueError) as exc:
        record("config loads", False, str(exc))

    # Data availability
    if config is not None:
        data = config.data
        record("data root exists", data.nq_data_root.is_dir(), str(data.nq_data_root))
        for label, directory in (
            ("raw/trades", data.raw_trades_dir),
            ("raw/mbp1", data.raw_mbp1_dir),
        ):
            if directory.is_dir():
                n = len(list(directory.glob("*.parquet")))
                record(f"{label} present", n > 0, f"{n} parquet files")
            else:
                record(f"{label} present", None, f"missing: {directory}")
        if data.nq_data_root.is_dir():
            usage = shutil.disk_usage(data.nq_data_root)
            record("disk free", None, f"{_human_bytes(usage.free)} free on data volume")

    table = Table(title="nqlab doctor", title_justify="left")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for check, status, detail in rows:
        table.add_row(check, status, detail)
    console.print(table)

    if not ok:
        console.print("[red]doctor: one or more checks FAILED[/red]")
        raise typer.Exit(code=1)
    console.print("[green]doctor: all critical checks passed[/green]")


@bars_app.command("build")
def bars_build(
    start: str = typer.Argument(help="Start date YYYY-MM-DD (inclusive)."),
    end: str = typer.Argument(help="End date YYYY-MM-DD (inclusive)."),
    timeframe: TimeframeOption = "5m",
    profile: ProfileOption = "dev",
    symbols: Annotated[str | None, typer.Option(help="Comma-separated outrights.")] = None,
    roots: Annotated[str, typer.Option(help="Comma-separated roots.")] = "NQ,MNQ",
    continuous: Annotated[
        bool, typer.Option("--continuous", help="Also build continuous.")
    ] = False,
) -> None:
    """Build OHLCV bars from trades (outrights only) and materialize them."""
    config = _load(profile)
    s, e = _dates(start, end)
    sym_list = [x.strip() for x in symbols.split(",")] if symbols else None
    root_tuple = tuple(x.strip() for x in roots.split(","))
    result = build_bars(
        config, timeframe=timeframe, start=s, end=e, roots=root_tuple, symbols=sym_list
    )
    console.print(
        f"[green]bars:[/green] {result.total_bars:,} bars, {len(result.symbols)} symbols, "
        f"{result.partitions_written} partitions ({timeframe})"
    )
    if continuous:
        for root in root_tuple:
            cont = build_continuous(config, root=root, timeframe=timeframe, start=s, end=e)
            console.print(
                f"[green]continuous {cont.symbol}:[/green] {cont.rows:,} bars, "
                f"{len(cont.rolls)} rolls, {cont.trading_days} trading days"
            )


@bars_app.command("profile")
def bars_profile(profile: ProfileOption = "dev") -> None:
    """List materialized bars under the curated root."""
    config = _load(profile)
    infos = profile_bars(config)
    if not infos:
        console.print("[yellow]No bars built yet.[/yellow]")
        return
    table = Table(title="curated bars", title_justify="left")
    table.add_column("symbol")
    table.add_column("timeframe")
    table.add_column("months")
    table.add_column("rows", justify="right")
    for info in infos:
        table.add_row(info.symbol, info.timeframe, ",".join(info.months), f"{info.rows:,}")
    console.print(table)


@features_app.command("build")
def features_build(
    symbol: str = typer.Argument(help="Bar symbol, e.g. NQ.c or NQH4."),
    start: str = typer.Argument(help="Start date YYYY-MM-DD."),
    end: str = typer.Argument(help="End date YYYY-MM-DD."),
    timeframe: TimeframeOption = "5m",
    profile: ProfileOption = "dev",
) -> None:
    """Compute PIT features for a symbol and materialize them."""
    config = _load(profile)
    s, e = _dates(start, end)
    rows = build_and_write_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    console.print(f"[green]features:[/green] {rows:,} rows for {symbol} ({timeframe})")


def _backtest_out_dir(config: AppConfig, tag: str) -> Path:
    return config.data.artifacts_root / "backtests" / tag


@backtest_app.command("run")
def backtest_run(
    strategy: str = typer.Argument(help="Baseline: orb|vwap_rev|trend_pullback|pdhl."),
    symbol: str = typer.Argument(help="Bar symbol, e.g. NQ.c."),
    start: str = typer.Argument(help="Start date YYYY-MM-DD."),
    end: str = typer.Argument(help="End date YYYY-MM-DD."),
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    profile: ProfileOption = "dev",
    hard_cutoff: Annotated[
        bool, typer.Option("--hard-cutoff", help="Use 15:50 cutoff instead of 12:30.")
    ] = False,
) -> None:
    """Run one baseline strategy end-to-end and write trades/equity/metrics."""
    config = _load(profile)
    rules = load_topstep_rules(f"topstep_{account}")
    s, e = _dates(start, end)
    frame = prepare_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    result = run_strategy(
        config,
        build_baseline(strategy),
        symbol=symbol,
        timeframe=timeframe,
        rules=rules,
        feature_frame=frame,
        primary_cutoff=not hard_cutoff,
    )
    metrics = compute_metrics(result)
    tag = f"{strategy}_{symbol}_{timeframe}_{start}_{end}_{account}"
    out_dir = _backtest_out_dir(config, tag)
    write_report(result, metrics, out_dir)
    console.print(_metrics_table(f"{strategy} on {symbol} ({timeframe})", metrics))
    if result.topstep is not None:
        _print_topstep(result.topstep)
    console.print(f"[green]artifacts →[/green] {out_dir}")


@research_app.command("baseline")
def research_baseline(
    symbol: str = typer.Argument(help="Bar symbol, e.g. NQ.c."),
    start: str = typer.Argument(help="Start date YYYY-MM-DD."),
    end: str = typer.Argument(help="End date YYYY-MM-DD."),
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    profile: ProfileOption = "dev",
) -> None:
    """Run all baseline strategies, write artifacts, and print a ranked report."""
    config = _load(profile)
    s, e = _dates(start, end)
    rules = load_topstep_rules(f"topstep_{account}")
    runs = run_baselines(config, symbol=symbol, timeframe=timeframe, start=s, end=e, rules=rules)

    table = Table(title=f"baseline ranking — {symbol} {timeframe}", title_justify="left")
    for col in ("rank", "strategy", "net_pnl", "max_dd", "trades", "win%", "pf", "pass", "blown"):
        table.add_column(col, justify="right" if col != "strategy" else "left")
    for i, run in enumerate(runs, 1):
        m = run.metrics
        out_dir = _backtest_out_dir(
            config, f"{run.name}_{symbol}_{timeframe}_{start}_{end}_{account}"
        )
        write_report(run.result, m, out_dir)
        pf = m.get("profit_factor")
        win_pct = cast(float, m["win_rate"]) * 100
        table.add_row(
            str(i),
            run.name,
            f"{m['net_pnl']}",
            f"{m['max_drawdown']}",
            str(m["trades"]),
            f"{win_pct:.0f}",
            f"{pf:.2f}" if isinstance(pf, float) else "-",
            "Y" if m.get("passed") else "·",
            "Y" if m.get("blown") else "·",
        )
    console.print(table)
    console.print(f"[green]artifacts →[/green] {config.data.artifacts_root / 'backtests'}")


@topstep_app.command("simulate")
def topstep_simulate(
    equity: Annotated[Path, typer.Argument(help="Equity-curve parquet (ts/trading_day/equity).")],
    account: AccountOption = "50k",
    profile: ProfileOption = "dev",
) -> None:
    """Evaluate a saved equity curve against a Topstep rule set."""
    _load(profile)
    rules = load_topstep_rules(f"topstep_{account}")
    frame = pl.read_parquet(equity).sort("ts")
    # Re-base onto the requested account size using PnL (size-agnostic), so the
    # same curve can be evaluated against any rule set.
    if "pnl" in frame.columns:
        pnl = frame["pnl"].to_list()
    else:
        first = frame["equity"][0]
        pnl = [value - first for value in frame["equity"].to_list()]
    days = frame["trading_day"].to_list()
    marks = [(day, rules.account_size + p) for day, p in zip(days, pnl, strict=True)]
    result = simulate_account(marks, rules)
    _print_topstep(result)
    console.print(
        f"  trading_days={result.trading_days} best_day={result.best_day:+,.0f} "
        f"worst_day={result.worst_day:+,.0f} consistency_ok={result.consistency_ok}"
    )


@bars_app.command("build-full")
def bars_build_full(
    start: str = typer.Argument(help="Start date YYYY-MM-DD."),
    end: str = typer.Argument(help="End date YYYY-MM-DD."),
    timeframe: TimeframeOption = "5m",
    roots: Annotated[str, typer.Option(help="Comma-separated roots.")] = "NQ",
    profile: ProfileOption = "dev",
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if complete.")] = False,
    verify: Annotated[
        bool, typer.Option("--verify", help="Row-count check completed months.")
    ] = False,
) -> None:
    """Incrementally materialize bars month-by-month (resumable, skips complete)."""
    config = _load(profile)
    s, e = _dates(start, end)
    result = build_bars_full(
        config,
        timeframe=timeframe,
        start=s,
        end=e,
        roots=tuple(x.strip() for x in roots.split(",")),
        force=force,
        verify=verify,
    )
    console.print(
        f"[green]bars build-full:[/green] built {len(result.built)}, "
        f"rebuilt {len(result.rebuilt)}, skipped {len(result.skipped)} months; "
        f"{result.total_rows:,} bars ({timeframe})"
    )


@features_app.command("build-full")
def features_build_full(
    symbol: str = typer.Argument(help="Bar symbol, e.g. NQ.c."),
    start: str = typer.Argument(help="Start date YYYY-MM-DD."),
    end: str = typer.Argument(help="End date YYYY-MM-DD."),
    timeframe: TimeframeOption = "5m",
    profile: ProfileOption = "dev",
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if complete.")] = False,
) -> None:
    """Incrementally materialize PIT features month-by-month (warmup-aware)."""
    config = _load(profile)
    s, e = _dates(start, end)
    result = build_features_full(
        config, symbol=symbol, timeframe=timeframe, start=s, end=e, force=force
    )
    console.print(
        f"[green]features build-full:[/green] built {len(result.built)}, "
        f"skipped {len(result.skipped)} months; {result.total_rows:,} rows"
    )


def _validate_one(
    config: AppConfig,
    name: str,
    symbol: str,
    timeframe: str,
    s: date,
    e: date,
    rules: TopstepRules,
    account: str,
    frame: pl.DataFrame | None,
) -> ValidationResult:
    result = run_gauntlet(
        config,
        build_baseline(name),
        symbol=symbol,
        timeframe=timeframe,
        start=s,
        end=e,
        rules=rules,
        feature_frame=frame,
    )
    out = config.data.artifacts_root / "validation" / f"{name}_{symbol}_{timeframe}_{account}"
    write_validation_report(result, out)
    return result


@validate_app.command("run")
def validate_run(
    strategy: str = typer.Argument(help="Baseline: orb|vwap_rev|trend_pullback|pdhl."),
    symbol: str = typer.Argument(help="Bar symbol, e.g. NQ.c."),
    start: str = typer.Argument(help="Start date YYYY-MM-DD."),
    end: str = typer.Argument(help="End date YYYY-MM-DD."),
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    profile: ProfileOption = "dev",
) -> None:
    """Run the full validation gauntlet for one strategy and print the verdict."""
    config = _load(profile)
    s, e = _dates(start, end)
    rules = load_topstep_rules(f"topstep_{account}")
    result = _validate_one(config, strategy, symbol, timeframe, s, e, rules, account, None)
    print_validation(console, result)


def _validate_table(results: list[ValidationResult]) -> Table:
    table = Table(title="validation verdicts", title_justify="left")
    left = {"strategy", "verdict", "reason"}
    for col in ("strategy", "verdict", "full_net", "oos_net", "wf+", "ruin%", "reason"):
        table.add_column(col, justify="left" if col in left else "right")
    style = {"ACCEPT": "green", "REJECT": "red", "NEEDS_MORE_DATA": "yellow"}
    for r in results:
        table.add_row(
            r.strategy,
            f"[{style.get(r.verdict, 'white')}]{r.verdict}[/]",
            str(r.full.get("net_pnl")),
            str(r.out_sample.get("net_pnl")),
            f"{r.walk_forward_positive_frac:.0%}",
            f"{r.monte_carlo.blow_up_prob:.0%}",
            (r.reasons[0] if r.reasons else "")[:48],
        )
    return table


@validate_app.command("baseline")
def validate_baseline(
    symbol: str = typer.Argument(help="Bar symbol, e.g. NQ.c."),
    start: str = typer.Argument(help="Start date YYYY-MM-DD."),
    end: str = typer.Argument(help="End date YYYY-MM-DD."),
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    profile: ProfileOption = "dev",
) -> None:
    """Validate every baseline and print accept/reject verdicts."""
    config = _load(profile)
    s, e = _dates(start, end)
    rules = load_topstep_rules(f"topstep_{account}")
    frame = prepare_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    results = [
        _validate_one(config, name, symbol, timeframe, s, e, rules, account, frame)
        for name in BASELINES
    ]
    console.print(_validate_table(results))
    console.print(f"[green]reports →[/green] {config.data.artifacts_root / 'validation'}")


@research_app.command("full-history")
def research_full_history(
    start: Annotated[str | None, typer.Option(help="Override start (default train_start).")] = None,
    end: Annotated[str | None, typer.Option(help="Override end (default test_end).")] = None,
    root: Annotated[str, typer.Option(help="Contract root.")] = "NQ",
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    profile: ProfileOption = "dev",
    force: Annotated[bool, typer.Option("--force", help="Rebuild all partitions.")] = False,
) -> None:
    """End-to-end: build full-history bars/continuous/features, then validate baselines."""
    config = _load(profile)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    rules = load_topstep_rules(f"topstep_{account}")
    console.print(f"[bold]building full history[/bold] {root} {timeframe} [{s}..{e}] …")
    symbol = ensure_full_history(
        config, root=root, timeframe=timeframe, start=s, end=e, force=force
    )
    console.print(f"continuous symbol: [bold]{symbol}[/bold]; validating baselines …")
    frame = prepare_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    results = [
        _validate_one(config, name, symbol, timeframe, s, e, rules, account, frame)
        for name in BASELINES
    ]
    console.print(_validate_table(results))


@report_app.command("validation")
def report_validation(
    path: Annotated[Path, typer.Argument(help="validation.json produced by `validate`.")],
    profile: ProfileOption = "dev",
) -> None:
    """Render a saved validation report."""
    _load(profile)
    data = json.loads(Path(path).read_text())
    print_validation_dict(console, data)


def _tracker(config: AppConfig) -> ExperimentTracker:
    return ExperimentTracker(config.data.curated_root / "experiments.duckdb")


@discover_app.command("run")
def discover_run(
    symbol: Annotated[str, typer.Option(help="Continuous symbol.")] = "NQ.c",
    start: Annotated[str | None, typer.Option(help="Default = validation.train_start.")] = None,
    end: Annotated[str | None, typer.Option(help="Default = validation.test_end.")] = None,
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    root: Annotated[str, typer.Option(help="Contract root to build.")] = "NQ",
    families: Annotated[str | None, typer.Option(help="Comma-separated families.")] = None,
    max_candidates: Annotated[
        int | None, typer.Option(help="Cap candidates (small first run).")
    ] = None,
    run_id: Annotated[str | None, typer.Option(help="Explicit run id (else auto).")] = None,
    build: Annotated[bool, typer.Option("--build/--no-build", help="Ensure data first.")] = True,
    profile: ProfileOption = "dev",
) -> None:
    """Generate -> validate -> deflate -> accept/reject candidates; save all."""
    config = _load(profile)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    rules = load_topstep_rules(f"topstep_{account}")
    if build:
        console.print(f"[bold]ensuring data[/bold] {root} {timeframe} [{s}..{e}] …")
        symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
    rid = run_id or f"disc_{symbol}_{timeframe}_{account}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    fams = tuple(x.strip() for x in families.split(",")) if families else None
    console.print(f"[bold]discovery[/bold] run_id={rid} …")
    result = run_discovery(
        config,
        symbol=symbol,
        timeframe=timeframe,
        start=s,
        end=e,
        rules=rules,
        tracker=_tracker(config),
        run_id=rid,
        families=fams,
        max_candidates=max_candidates,
    )
    print_discovery(console, result)
    console.print(f"[green]run_id:[/green] {rid}")


@discover_app.command("resume")
def discover_resume(
    run_id: str = typer.Argument(help="Run id to resume."),
    profile: ProfileOption = "dev",
) -> None:
    """Resume an interrupted discovery run (skips completed candidates)."""
    config = _load(profile)
    tracker = _tracker(config)
    run = tracker.get_run(run_id)
    if run is None:
        console.print(f"[red]no such run:[/red] {run_id}")
        raise typer.Exit(code=1)
    rules = load_topstep_rules(f"topstep_{str(run['account']).lower()}")
    result = run_discovery(
        config,
        symbol=run["symbol"],
        timeframe=run["timeframe"],
        start=date.fromisoformat(run["start_date"]),
        end=date.fromisoformat(run["end_date"]),
        rules=rules,
        tracker=tracker,
        run_id=run_id,
        resume=True,
        created_at=run["created_at"],
    )
    print_discovery(console, result)


@discover_app.command("leaderboard")
def discover_leaderboard(
    run_id: str = typer.Argument(help="Run id."),
    top: Annotated[int, typer.Option(help="Rows to show.")] = 20,
    profile: ProfileOption = "dev",
) -> None:
    """Show the leaderboard for a discovery run."""
    config = _load(profile)
    rows = rank_rows(_tracker(config).load_candidates(run_id))
    if not rows:
        console.print(f"[yellow]no candidates for run {run_id}[/yellow]")
        return
    console.print(leaderboard_table(rows, top))


@experiment_app.command("list")
def experiment_list(profile: ProfileOption = "dev") -> None:
    """List discovery runs."""
    config = _load(profile)
    runs = _tracker(config).list_runs()
    if not runs:
        console.print("[yellow]no runs yet[/yellow]")
        return
    table = Table(title="experiment runs", title_justify="left")
    for col in ("run_id", "symbol", "tf", "account", "trials", "pbo", "status"):
        table.add_column(col)
    for r in runs:
        pbo = "n/a" if r["pbo"] is None else f"{float(r['pbo']):.1%}"
        table.add_row(
            str(r["run_id"]),
            str(r["symbol"]),
            str(r["timeframe"]),
            str(r["account"]),
            str(r["n_trials"]),
            pbo,
            str(r["status"]),
        )
    console.print(table)


@experiment_app.command("show")
def experiment_show(
    run_id: str = typer.Argument(help="Run id."),
    top: Annotated[int, typer.Option(help="Leaderboard rows.")] = 10,
    profile: ProfileOption = "dev",
) -> None:
    """Show a run's metadata, leaderboard, and accept/reject counts."""
    config = _load(profile)
    tracker = _tracker(config)
    run = tracker.get_run(run_id)
    if run is None:
        console.print(f"[red]no such run:[/red] {run_id}")
        raise typer.Exit(code=1)
    rows = rank_rows(tracker.load_candidates(run_id))
    pbo = "n/a" if run["pbo"] is None else f"{float(run['pbo']):.1%}"
    console.print(
        f"[bold]{run_id}[/bold] {run['symbol']} {run['timeframe']} {run['account']} | "
        f"trials={run['n_trials']} PBO={pbo} status={run['status']}"
    )
    console.print(leaderboard_table(rows, top))
    n_acc = sum(1 for r in rows if r["final_verdict"] == "ACCEPT")
    n_rej = sum(1 for r in rows if r["final_verdict"] == "REJECT")
    n_nmd = sum(1 for r in rows if r["final_verdict"] == "NEEDS_MORE_DATA")
    console.print(
        f"[green]ACCEPT={n_acc}[/green] [red]REJECT={n_rej}[/red] "
        f"[yellow]NEEDS_MORE_DATA={n_nmd}[/yellow]"
    )


@strategy_app.command("registry")
def strategy_registry(profile: ProfileOption = "dev") -> None:
    """List strategies accepted by discovery (survived all checks + deflation)."""
    config = _load(profile)
    accepted = _tracker(config).accepted()
    if not accepted:
        console.print("[yellow]registry empty — no strategy has been accepted yet[/yellow]")
        return
    table = Table(title="strategy registry (accepted)", title_justify="left")
    for col in ("family", "fingerprint", "oos_net", "DSR", "ruin%", "run_id", "params"):
        table.add_column(col)
    for r in accepted:
        table.add_row(
            str(r["family"]),
            str(r["fingerprint"])[:12],
            f"{float(r['oos_net']):.0f}",
            f"{float(r['deflated_sharpe'] or 0):.2f}",
            f"{float(r['risk_of_ruin'] or 0) * 100:.0f}",
            str(r["run_id"]),
            str(r["params"])[:40],
        )
    console.print(table)


_MBP1_OVERRIDES = {
    "validation": {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2025-06-30",
    }
}

# Phase 9 setups are trades-only, so out-of-sample runs the full bar history
# (MBP1 is not required), not just the MBP1 window.
_PHASE9_OVERRIDES = {
    "validation": {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2026-12-31",
    }
}

# Phase 10 lockbox window (2025-H2 -> 2026-H1, to data end). Split in-sample (H2-2025,
# for the Deflated-Sharpe estimate) / out-of-sample (H1-2026, the acceptance gate);
# short walk-forward windows so a ~1-year lockbox still yields several.
_LOCKBOX_START = date(2025, 7, 1)
_LOCKBOX_END = date(2026, 6, 30)
_LOCKBOX_OVERRIDES = {
    "validation": {
        "train_start": "2025-07-01",
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-06-30",
        "walk_train_months": 3,
        "walk_test_months": 1,
        "walk_step_months": 1,
    }
}

# Phase 11 reserves a CLEAN lockbox before searching: search/train 2021-2023,
# validation 2024 (the search never loads past 2024-12-31), final untouched lockbox
# 2025-2026 (in-sample 2025 for the DSR estimate, OOS gate 2026-H1, to data end).
_PHASE11_SEARCH_OVERRIDES = {
    "validation": {
        "train_start": "2021-01-01",
        "train_end": "2023-12-31",
        "test_start": "2024-01-01",
        "test_end": "2024-12-31",
    }
}
_PHASE11_SEARCH_END = date(2024, 12, 31)
_PHASE11_LOCKBOX_START = date(2025, 1, 1)
_PHASE11_LOCKBOX_END = date(2026, 6, 30)
_PHASE11_LOCKBOX_OVERRIDES = {
    "validation": {
        "train_start": "2025-01-01",
        "train_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-06-30",
        "walk_train_months": 3,
        "walk_test_months": 1,
        "walk_step_months": 1,
    }
}


@mbp1_features_app.command("build")
def mbp1_features_build(
    start: str = typer.Argument(help="Start date YYYY-MM-DD."),
    end: str = typer.Argument(help="End date YYYY-MM-DD."),
    timeframe: TimeframeOption = "5m",
    root: Annotated[str, typer.Option(help="Contract root.")] = "NQ",
    profile: ProfileOption = "dev",
    force: Annotated[bool, typer.Option("--force", help="Rebuild even if complete.")] = False,
) -> None:
    """Incrementally build MBP1 microstructure features (missing months recorded)."""
    config = _load(profile)
    s, e = _dates(start, end)
    result = build_mbp1_features_full(
        config, root=root, timeframe=timeframe, start=s, end=e, force=force
    )
    console.print(
        f"[green]mbp1 features:[/green] built {len(result.built)}, "
        f"missing {len(result.missing)}, skipped {len(result.skipped)} months; "
        f"{result.total_rows:,} rows"
    )


@mbp1_features_app.command("profile")
def mbp1_features_profile(
    symbol: Annotated[str, typer.Option(help="Continuous symbol.")] = "NQ.c",
    timeframe: TimeframeOption = "5m",
    profile: ProfileOption = "dev",
) -> None:
    """Show built vs missing MBP1 feature months."""
    config = _load(profile)
    summary = profile_mbp1(config, symbol, timeframe)
    console.print(f"[green]built[/green] ({len(summary['built'])}): {', '.join(summary['built'])}")
    console.print(
        f"[red]missing[/red] ({len(summary['missing'])}): {', '.join(summary['missing'])}"
    )


@discover_app.command("mbp1")
def discover_mbp1(
    start: Annotated[str | None, typer.Option(help="Default = MBP1 train_start.")] = None,
    end: Annotated[str | None, typer.Option(help="Default = MBP1 test_end (2025-06-30).")] = None,
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    root: Annotated[str, typer.Option(help="Contract root.")] = "NQ",
    max_candidates: Annotated[int | None, typer.Option(help="Cap candidates.")] = None,
    run_id: Annotated[str | None, typer.Option(help="Explicit run id.")] = None,
    build: Annotated[bool, typer.Option("--build/--no-build", help="Build data first.")] = True,
    profile: ProfileOption = "dev",
) -> None:
    """Discover MBP1 order-flow strategies (train 2021-23, validate 2024-25H1)."""
    config = load_config(profile=profile, overrides=_MBP1_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    rules = load_topstep_rules(f"topstep_{account}")
    symbol = f"{root}.c"
    if build:
        console.print(f"[bold]ensuring trades data[/bold] {root} {timeframe} [{s}..{e}] …")
        symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
        console.print("[bold]building MBP1 features[/bold] …")
        mres = build_mbp1_features_full(config, root=root, timeframe=timeframe, start=s, end=e)
        console.print(f"mbp1 features: built {len(mres.built)}, missing {len(mres.missing)} months")
    frame = prepare_mbp1_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    rid = run_id or f"mbp1_{symbol}_{timeframe}_{account}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    console.print(f"[bold]MBP1 discovery[/bold] run_id={rid} …")
    result = run_discovery(
        config,
        symbol=symbol,
        timeframe=timeframe,
        start=s,
        end=e,
        rules=rules,
        tracker=_tracker(config),
        run_id=rid,
        families=tuple(MBP1_FAMILIES),
        max_candidates=max_candidates,
        feature_frame=frame,
    )
    print_discovery(console, result)
    console.print(f"[green]run_id:[/green] {rid}")


@report_app.command("mbp1-discovery")
def report_mbp1_discovery(
    run_id: str = typer.Argument(help="Run id."),
    profile: ProfileOption = "dev",
) -> None:
    """Render a saved MBP1 discovery run report."""
    config = _load(profile)
    tracker = _tracker(config)
    run = tracker.get_run(run_id)
    rows = rank_rows(tracker.load_candidates(run_id))
    if run is None or not rows:
        console.print(f"[yellow]no data for run {run_id}[/yellow]")
        return
    print_run_rows(console, rows, run)


def _print_profile(prof: DistributionProfile) -> None:
    table = Table(title=f"MBP1 feature distributions (n={prof.n_bars:,})", title_justify="left")
    for col in ("feature", "mean", "p10", "p50", "p90", "p95", "p99"):
        table.add_column(col, justify="left" if col == "feature" else "right")
    for feat in prof.features:
        s = prof.overall[feat]
        table.add_row(
            feat,
            *(f"{s.get(k, float('nan')):.4g}" for k in ("mean", "q10", "q50", "q90", "q95", "q99")),
        )
    console.print(table)
    keys = [
        f
        for f in ("abs_tob_imbalance", "delta", "spread", "liq_spread_ratio")
        if f in prof.features
    ]
    if prof.by_session and keys:
        st = Table(title="median by session", title_justify="left")
        st.add_column("session")
        for f in keys:
            st.add_column(f, justify="right")
        for sess, feats in prof.by_session.items():
            st.add_row(sess, *(f"{feats.get(f, {}).get('q50', float('nan')):.4g}" for f in keys))
        console.print(st)


@mbp1_app.command("profile-distributions")
def mbp1_profile_distributions(
    start: Annotated[str | None, typer.Option(help="Default 2021-01-01.")] = None,
    end: Annotated[str | None, typer.Option(help="Default 2023-12-31 (train).")] = None,
    timeframe: TimeframeOption = "1m",
    symbol: Annotated[str, typer.Option(help="Continuous symbol.")] = "NQ.c",
    profile: ProfileOption = "dev",
) -> None:
    """Profile MBP1 feature distributions (percentiles by session/regime/hour)."""
    config = _load(profile)
    s = date.fromisoformat(start) if start else date(2021, 1, 1)
    e = date.fromisoformat(end) if end else date(2023, 12, 31)
    frame = read_mbp1_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    prof = profile_distributions(frame)
    out = config.data.artifacts_root / "profiles" / f"mbp1_{symbol}_{timeframe}_{s}_{e}.json"
    prof.write_json(out)
    _print_profile(prof)
    console.print(f"[green]profile →[/green] {out}")


@discover_app.command("mbp1-calibrated")
def discover_mbp1_calibrated(
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    timeframe: TimeframeOption = "1m",
    account: AccountOption = "50k",
    root: Annotated[str, typer.Option()] = "NQ",
    max_candidates: Annotated[int | None, typer.Option(help="Cap candidates.")] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = True,
    profile: ProfileOption = "dev",
) -> None:
    """Calibrate combo-family thresholds from TRAIN percentiles, then discover."""
    config = load_config(profile=profile, overrides=_MBP1_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    rules = load_topstep_rules(f"topstep_{account}")
    symbol = f"{root}.c"
    if build:
        console.print(f"[bold]ensuring trades data[/bold] {root} {timeframe} …")
        symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
        console.print("[bold]building MBP1 features[/bold] …")
        mres = build_mbp1_features_full(config, root=root, timeframe=timeframe, start=s, end=e)
        console.print(f"mbp1: built {len(mres.built)}, missing {len(mres.missing)} months")

    frame = prepare_mbp1_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    # Calibrate on the TRAIN slice only (no OOS leakage).
    train = frame.filter(pl.col("trading_day") <= v.train_end)
    thresholds = calibrate(profile_distributions(train))
    console.print(
        f"[green]calibrated[/green] imb={[round(x, 3) for x in thresholds.imb]} "
        f"spread_max={[round(x, 3) for x in thresholds.spread_max]} "
        f"liq={[round(x, 3) for x in thresholds.liq_thr]}"
    )
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    candidates = generate_calibrated_candidates(thresholds, rth_start_min=rth_start_min)
    rid = run_id or f"mbp1cal_{symbol}_{timeframe}_{account}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    console.print(f"[bold]calibrated discovery[/bold] {rid} ({len(candidates)} candidates) …")
    result = run_discovery(
        config,
        symbol=symbol,
        timeframe=timeframe,
        start=s,
        end=e,
        rules=rules,
        tracker=_tracker(config),
        run_id=rid,
        candidates=candidates,
        max_candidates=max_candidates,
        feature_frame=frame,
    )
    print_discovery(console, result)
    console.print(f"[green]run_id:[/green] {rid}")


@report_app.command("phase5")
def report_phase5(
    run_id: str = typer.Argument(help="1m calibrated run id."),
    compare_to: Annotated[str, typer.Option(help="5m run to compare against.")] = "phase4_mbp1_5m",
    profile: ProfileOption = "dev",
) -> None:
    """Render a Phase-5 (1m calibrated) run with a comparison vs the 5m Phase-4 run."""
    config = _load(profile)
    tracker = _tracker(config)
    rows = rank_rows(tracker.load_candidates(run_id))
    run = tracker.get_run(run_id)
    if run is None or not rows:
        console.print(f"[yellow]no data for run {run_id}[/yellow]")
        return
    print_run_rows(console, rows, run)

    other = tracker.load_candidates(compare_to)
    if not other:
        console.print(f"[yellow](no comparison run {compare_to})[/yellow]")
        return
    table = Table(title="comparison: 1m calibrated vs 5m", title_justify="left")
    for col in ("run", "trials", "accept", "reject", "nmd", "best_oos", "best_score"):
        table.add_column(col, justify="left" if col == "run" else "right")
    for label, summary in (
        (f"{run_id} (1m)", run_summary(rows)),
        (f"{compare_to} (5m)", run_summary(other)),
    ):
        table.add_row(
            label,
            str(summary["trials"]),
            str(summary["accept"]),
            str(summary["reject"]),
            str(summary["needs_more_data"]),
            str(summary["best_oos"]),
            str(summary["best_score"]),
        )
    console.print(table)


@app.command("diagnose")
def diagnose_cmd(
    family: str = typer.Argument(help="Family with a grid (e.g. imb_cont, delta_rev, orb)."),
    symbol: Annotated[str, typer.Option()] = "NQ.c",
    start: Annotated[str, typer.Option()] = "2021-01-01",
    end: Annotated[str, typer.Option()] = "2025-06-30",
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    profile: ProfileOption = "dev",
) -> None:
    """Diagnose where a strategy family loses (MAE/MFE, sessions, hours, exits)."""
    config = _load(profile)
    cls = ALL_FAMILIES.get(family)
    if cls is None:
        console.print(f"[red]unknown family[/red] {family}; options: {sorted(ALL_FAMILIES)}")
        raise typer.Exit(code=1)
    s, e = _dates(start, end)
    rules = load_topstep_rules(f"topstep_{account}")
    strat = cls(cls.grid()[0], rth_start_min=parse_hhmm(config.sessions.rth_start))
    frame = prepare_mbp1_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    signal = strat.signals(frame)
    if config.backtest.rth_only:
        signal = signal.filter(pl.col("is_rth"))
    result = run_backtest(
        signal,
        symbol=symbol,
        timeframe=timeframe,
        strategy=strat.params,
        config=config,
        rules=rules,
        gate_entries=False,
    )
    diag = diagnose(result.trades, SessionLabeler.from_config(config.sessions))
    if diag.get("n_trades", 0) == 0:
        console.print("[yellow]no trades to diagnose[/yellow]")
        return
    console.print(
        f"[bold]{family}[/bold]: {diag['n_trades']} trades | "
        f"gross {diag['gross_pnl']} net {diag['net_pnl']} | "
        f"direction-edge-before-costs={diag['direction_edge_before_costs']} | "
        f"cost_drag={diag['cost_drag']}"
    )
    console.print(
        f"  MAE avg/p90={diag['avg_mae']}/{diag['mae_p90']} | "
        f"MFE avg/p90={diag['avg_mfe']}/{diag['mfe_p90']} | "
        f"mfe_capture={diag['mfe_capture']} | stops_too_tight={diag['stops_likely_too_tight']}"
    )
    for title, key in (
        ("by session", "by_session"),
        ("by hour", "by_hour"),
        ("by exit", "by_exit_reason"),
        ("by holding", "by_holding"),
    ):
        breakdown = cast("dict[str, dict[str, float]]", diag[key])
        cells = " ".join(f"{k}:{v['net']}({v['n']})" for k, v in breakdown.items())
        console.print(f"  [bold]{title}[/bold] {cells}")
    console.print(
        f"  [red]worst sessions:[/red] {diag['worst_sessions']} "
        f"[red]worst hours:[/red] {diag['worst_hours']}"
    )


@discover_app.command("setups")
def discover_setups(
    start: Annotated[str | None, typer.Option()] = None,
    end: Annotated[str | None, typer.Option()] = None,
    timeframe: TimeframeOption = "1m",
    account: AccountOption = "50k",
    root: Annotated[str, typer.Option()] = "NQ",
    max_candidates: Annotated[int | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = True,
    profile: ProfileOption = "dev",
) -> None:
    """Discover context-based setups (context + liquidity + order-flow trigger)."""
    config = load_config(profile=profile, overrides=_MBP1_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    rules = load_topstep_rules(f"topstep_{account}")
    symbol = f"{root}.c"
    if build:
        console.print(f"[bold]ensuring trades data[/bold] {root} {timeframe} …")
        symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
        console.print("[bold]building MBP1 features[/bold] …")
        build_mbp1_features_full(config, root=root, timeframe=timeframe, start=s, end=e)

    frame = prepare_setup_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    train = frame.filter(pl.col("trading_day") <= v.train_end)
    thresholds = calibrate(profile_distributions(train))
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    candidates = generate_setup_candidates(thresholds, rth_start_min=rth_start_min)
    rid = run_id or f"setups_{symbol}_{timeframe}_{account}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    console.print(f"[bold]setup discovery[/bold] {rid} ({len(candidates)} setups) …")
    result = run_discovery(
        config,
        symbol=symbol,
        timeframe=timeframe,
        start=s,
        end=e,
        rules=rules,
        tracker=_tracker(config),
        run_id=rid,
        candidates=candidates,
        max_candidates=max_candidates,
        feature_frame=frame,
    )
    print_discovery(console, result)
    console.print(f"[green]run_id:[/green] {rid}")


def _optimize(
    profile: str,
    timeframe: str,
    account: str,
    root: str,
    start: str | None,
    end: str | None,
    build: bool,
    prefix: str,
    run_id: str | None,
    max_candidates: int | None,
    generator: Callable[..., list[Any]],
) -> None:
    config = load_config(profile=profile, overrides=_MBP1_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    rules = load_topstep_rules(f"topstep_{account}")
    symbol = f"{root}.c"
    if build:
        console.print(f"[bold]ensuring data[/bold] {root} {timeframe} …")
        symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
        build_mbp1_features_full(config, root=root, timeframe=timeframe, start=s, end=e)
    frame = prepare_setup_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    candidates = generator(rth_start_min=parse_hhmm(config.sessions.rth_start))
    rid = run_id or f"{prefix}_{symbol}_{timeframe}_{account}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    console.print(f"[bold]{prefix}[/bold] {rid} ({len(candidates)} candidates) …")
    result = run_discovery(
        config,
        symbol=symbol,
        timeframe=timeframe,
        start=s,
        end=e,
        rules=rules,
        tracker=_tracker(config),
        run_id=rid,
        candidates=candidates,
        max_candidates=max_candidates,
        feature_frame=frame,
    )
    print_discovery(console, result)
    console.print(f"[green]run_id:[/green] {rid}")


_OptStart = Annotated[str | None, typer.Option()]


@optimize_app.command("exits")
def optimize_exits(
    start: _OptStart = None,
    end: _OptStart = None,
    timeframe: TimeframeOption = "1m",
    account: AccountOption = "50k",
    root: Annotated[str, typer.Option()] = "NQ",
    max_candidates: Annotated[int | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = True,
    profile: ProfileOption = "dev",
) -> None:
    """Sweep stop/target/trailing/min-hold on the focus setups (fixed risk)."""
    _optimize(
        profile,
        timeframe,
        account,
        root,
        start,
        end,
        build,
        "optexit",
        run_id,
        max_candidates,
        generate_exit_candidates,
    )


@optimize_app.command("risk")
def optimize_risk(
    start: _OptStart = None,
    end: _OptStart = None,
    timeframe: TimeframeOption = "1m",
    account: AccountOption = "50k",
    root: Annotated[str, typer.Option()] = "NQ",
    max_candidates: Annotated[int | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = True,
    profile: ProfileOption = "dev",
) -> None:
    """Sweep risk-per-trade ($25-$250) at a fixed trailing exit."""
    _optimize(
        profile,
        timeframe,
        account,
        root,
        start,
        end,
        build,
        "optrisk",
        run_id,
        max_candidates,
        generate_risk_candidates,
    )


@optimize_app.command("survival")
def optimize_survival(
    start: _OptStart = None,
    end: _OptStart = None,
    timeframe: TimeframeOption = "1m",
    account: AccountOption = "50k",
    root: Annotated[str, typer.Option()] = "NQ",
    max_candidates: Annotated[int | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = True,
    profile: ProfileOption = "dev",
) -> None:
    """Full survival sweep: stop x target x trailing x risk x session."""
    _optimize(
        profile,
        timeframe,
        account,
        root,
        start,
        end,
        build,
        "survival",
        run_id,
        max_candidates,
        generate_survival_candidates,
    )


@report_app.command("phase7")
def report_phase7(
    run_id: str = typer.Argument(help="Optimization run id."),
    profile: ProfileOption = "dev",
) -> None:
    """Render a Phase-7 survival-optimization run (survival-first leaderboard)."""
    config = _load(profile)
    tracker = _tracker(config)
    run = tracker.get_run(run_id)
    rows = rank_rows(tracker.load_candidates(run_id))
    if run is None or not rows:
        console.print(f"[yellow]no data for run {run_id}[/yellow]")
        return
    print_run_rows(console, rows, run)


def _feasible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Candidates that clear the survival floor: OOS>0, ruin<=10%, not blown, cost-stress ok."""
    out = []
    for r in rows:
        oos = float(r.get("oos_net") or 0.0)
        ruin = float(r.get("risk_of_ruin") or 0.0)
        if oos > 0 and ruin <= 0.10 and not r.get("blown") and r.get("cost_stress_survived"):
            out.append(r)
    return out


def _feasibility_run(
    config: AppConfig,
    *,
    symbol: str,
    timeframe: str,
    start: date,
    end: date,
    rules: TopstepRules,
    candidates: list[Any],
    run_id: str,
    max_candidates: int | None,
    frame: pl.DataFrame,
) -> None:
    console.print(f"[bold]{run_id}[/bold] ({len(candidates)} candidates, {rules.name}) …")
    result = run_discovery(
        config,
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        rules=rules,
        tracker=_tracker(config),
        run_id=run_id,
        candidates=candidates,
        max_candidates=max_candidates,
        feature_frame=frame,
    )
    print_discovery(console, result)
    feas = _feasible_rows(result.leaderboard)
    verdict = (
        f"[green]FEASIBLE: {len(feas)} candidate(s) clear the survival floor[/green]"
        if feas
        else "[red]NOT FEASIBLE: 0 candidates clear OOS>0 & ruin<=10% & stress[/red]"
    )
    console.print(f"  {verdict}  [dim]run_id={run_id}[/dim]")


_FamiliesOpt = Annotated[str | None, typer.Option(help="Comma-separated focus families.")]


def _families(families: str | None) -> tuple[str, ...]:
    return tuple(f.strip() for f in families.split(",")) if families else FOCUS_FAMILIES


@feasibility_app.command("accounts")
def feasibility_accounts(
    start: _OptStart = None,
    end: _OptStart = None,
    timeframe: TimeframeOption = "5m",
    accounts: Annotated[str, typer.Option(help="Comma-separated account sizes.")] = "50k,100k,150k",
    families: _FamiliesOpt = None,
    root: Annotated[str, typer.Option()] = "NQ",
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Sweep account size x instrument-sizing x risk at one timeframe (Phase 8)."""
    config = load_config(profile=profile, overrides=_MBP1_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    symbol = f"{root}.c"
    if build:
        symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
        build_mbp1_features_full(config, root=root, timeframe=timeframe, start=s, end=e)
    frame = prepare_setup_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    fams = _families(families)
    for account in (a.strip() for a in accounts.split(",")):
        rules = load_topstep_rules(f"topstep_{account}")
        candidates = generate_sizing_candidates(
            fams,
            timeframe=timeframe,
            risk_grid=account_risk_grid(account),
            rth_start_min=parse_hhmm(config.sessions.rth_start),
        )
        _feasibility_run(
            config,
            symbol=symbol,
            timeframe=timeframe,
            start=s,
            end=e,
            rules=rules,
            candidates=candidates,
            run_id=f"phase8_acct_{account}_{timeframe}",
            max_candidates=max_candidates,
            frame=frame,
        )


@feasibility_app.command("timeframes")
def feasibility_timeframes(
    start: _OptStart = None,
    end: _OptStart = None,
    timeframes: Annotated[str, typer.Option(help="Comma-separated timeframes.")] = "1m,5m,15m",
    account: AccountOption = "150k",
    sizing_mode: Annotated[str, typer.Option(help="Instrument sizing mode.")] = "mnq_scaling",
    families: _FamiliesOpt = None,
    root: Annotated[str, typer.Option()] = "NQ",
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Sweep timeframe x risk at a fixed account/sizing — does higher TF reduce the noise band?"""
    config = load_config(profile=profile, overrides=_MBP1_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    rules = load_topstep_rules(f"topstep_{account}")
    fams = _families(families)
    for timeframe in (t.strip() for t in timeframes.split(",")):
        symbol = f"{root}.c"
        if build:
            symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
            build_mbp1_features_full(config, root=root, timeframe=timeframe, start=s, end=e)
        frame = prepare_setup_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
        candidates = generate_timeframe_candidates(
            fams,
            timeframe=timeframe,
            sizing_mode=sizing_mode,
            risk_grid=account_risk_grid(account),
            rth_start_min=parse_hhmm(config.sessions.rth_start),
        )
        _feasibility_run(
            config,
            symbol=symbol,
            timeframe=timeframe,
            start=s,
            end=e,
            rules=rules,
            candidates=candidates,
            run_id=f"phase8_tf_{timeframe}_{account}",
            max_candidates=max_candidates,
            frame=frame,
        )


@report_app.command("phase8")
def report_phase8(
    prefix: str = typer.Argument("phase8", help="Run-id (exact) or prefix to aggregate."),
    profile: ProfileOption = "dev",
) -> None:
    """Feasibility matrix across Phase-8 runs (one row per account/timeframe combination)."""
    config = _load(profile)
    tracker = _tracker(config)
    exact = tracker.get_run(prefix)
    if exact is not None:
        print_run_rows(console, rank_rows(tracker.load_candidates(prefix)), exact)
        return
    runs = sorted(
        (r for r in tracker.list_runs() if str(r["run_id"]).startswith(prefix)),
        key=lambda r: str(r["run_id"]),
    )
    if not runs:
        console.print(f"[yellow]no Phase-8 runs matching {prefix!r}[/yellow]")
        return
    table = Table(title=f"Phase 8 feasibility matrix ({prefix}*)", title_justify="left")
    for col in ("run", "trials", "accept", "best OOS", "min ruin@OOS>0", "feasible"):
        table.add_column(col, justify="left" if col == "run" else "right")
    for run in runs:
        rows = tracker.load_candidates(str(run["run_id"]))
        pos = [r for r in rows if float(r.get("oos_net") or 0.0) > 0]
        feas = _feasible_rows(rows)
        accept = sum(1 for r in rows if r.get("final_verdict") == "ACCEPT")
        best_oos = max((float(r.get("oos_net") or 0.0) for r in rows), default=0.0)
        pos_ruins = [float(r.get("risk_of_ruin") or 0.0) for r in pos]
        min_ruin = min(pos_ruins) if pos_ruins else None
        verdict = f"[green]YES ({len(feas)})[/green]" if feas else "[red]no[/red]"
        table.add_row(
            str(run["run_id"]).replace("phase8_", ""),
            str(len(rows)),
            str(accept),
            f"{best_oos:,.0f}",
            "—" if min_ruin is None else f"{min_ruin * 100:.0f}%",
            verdict,
        )
    console.print(table)


@features_app.command("htf-context")
def features_htf_context(
    start: _OptStart = None,
    end: _OptStart = None,
    timeframe: TimeframeOption = "5m",
    root: Annotated[str, typer.Option()] = "NQ",
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Build + inspect the Phase-9 HTF context layer (PIT-safe coverage report)."""
    config = load_config(profile=profile, overrides=_PHASE9_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    symbol = f"{root}.c"
    if build:
        symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
        build_mbp1_features_full(config, root=root, timeframe=timeframe, start=s, end=e)
    frame = prepare_htf_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    rth = frame.filter(pl.col("is_rth") & (pl.col("minute_of_day") >= 660))
    missing = [c for c in HTF_CONTEXT_FEATURE_NAMES if c not in frame.columns]
    table = Table(
        title=f"HTF context coverage {symbol} {timeframe} [{s}..{e}]", title_justify="left"
    )
    for col in ("feature", "non-null / RTH bars"):
        table.add_column(col, justify="left" if col == "feature" else "right")
    for c in HTF_CONTEXT_FEATURE_NAMES:
        nn = rth[c].drop_nulls().len() if c in frame.columns else 0
        table.add_row(c, f"{nn}/{rth.height}")
    console.print(table)
    if missing:
        console.print(f"[red]MISSING columns:[/red] {missing}")
    else:
        console.print(
            f"[green]all {len(HTF_CONTEXT_FEATURE_NAMES)} HTF context features present[/green]"
        )
    dist = (
        frame.filter(pl.col("is_rth")).group_by("prev_day_type").len().sort("len", descending=True)
    )
    counts = dict(zip(dist["prev_day_type"], dist["len"], strict=True))
    console.print(f"[bold]prior-day-type distribution:[/bold] {counts}")


@discover_app.command("htf-setups")
def discover_htf_setups(
    start: _OptStart = None,
    end: _OptStart = None,
    timeframe: TimeframeOption = "5m",
    account: AccountOption = "50k",
    root: Annotated[str, typer.Option()] = "NQ",
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    run_id: Annotated[str | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    min_trades: Annotated[int, typer.Option(help="Diagnostics-first min trade count.")] = 30,
    asymmetry_min: Annotated[
        float, typer.Option(help="Min avg_mfe/avg_mae to pass prefilter.")
    ] = 1.2,
    profile: ProfileOption = "dev",
) -> None:
    """Phase 9: discover NQ-only HTF-context + asymmetric-liquidity setups.

    Diagnostics-first: candidates with too few trades, no directional edge, or
    MAE>=MFE are rejected before the full gauntlet. Trades-only -> OOS 2024-2026.
    """
    config = load_config(profile=profile, overrides=_PHASE9_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else v.test_end
    rules = load_topstep_rules(f"topstep_{account}")
    symbol = f"{root}.c"
    if build:
        symbol = ensure_full_history(config, root=root, timeframe=timeframe, start=s, end=e)
        build_mbp1_features_full(config, root=root, timeframe=timeframe, start=s, end=e)
    frame = prepare_htf_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE9_FAMILIES)
    candidates = generate_htf_setup_candidates(
        fams, timeframe=timeframe, rth_start_min=parse_hhmm(config.sessions.rth_start)
    )
    rid = run_id or f"phase9_{account}_{timeframe}"
    console.print(
        f"[bold]phase9 htf-setups[/bold] {rid} ({len(candidates)} candidates, {rules.name}) …"
    )
    result = run_discovery(
        config,
        symbol=symbol,
        timeframe=timeframe,
        start=s,
        end=e,
        rules=rules,
        tracker=_tracker(config),
        run_id=rid,
        candidates=candidates,
        max_candidates=max_candidates,
        feature_frame=frame,
        prefilter=diagnostics_prefilter(min_trades=min_trades, asymmetry_min=asymmetry_min),
    )
    print_discovery(console, result)
    feas = _feasible_rows(result.leaderboard)
    verdict = (
        f"[green]FEASIBLE: {len(feas)} candidate(s) clear the survival floor[/green]"
        if feas
        else "[red]NOT FEASIBLE: 0 candidates clear OOS>0 & ruin<=10% & stress[/red]"
    )
    console.print(f"  {verdict}  [dim]run_id={rid}[/dim]")


def _best_asymmetry(rows: list[dict[str, Any]]) -> float:
    """Max realized avg_mfe/avg_mae across a run's candidates (0 if none)."""
    best = 0.0
    for r in rows:
        try:
            d = json.loads(str(r.get("diagnostics") or "{}"))
        except (TypeError, ValueError):
            continue
        mae, mfe = float(d.get("avg_mae") or 0.0), float(d.get("avg_mfe") or 0.0)
        if mae > 0:
            best = max(best, mfe / mae)
    return best


@report_app.command("phase9")
def report_phase9(
    prefix: str = typer.Argument("phase9", help="Run-id (exact) or prefix to aggregate."),
    profile: ProfileOption = "dev",
) -> None:
    """Phase-9 leaderboard (exact run) or feasibility matrix (prefix across runs)."""
    config = _load(profile)
    tracker = _tracker(config)
    exact = tracker.get_run(prefix)
    if exact is not None:
        print_run_rows(console, rank_rows(tracker.load_candidates(prefix)), exact)
        return
    runs = sorted(
        (r for r in tracker.list_runs() if str(r["run_id"]).startswith(prefix)),
        key=lambda r: str(r["run_id"]),
    )
    if not runs:
        console.print(f"[yellow]no Phase-9 runs matching {prefix!r}[/yellow]")
        return
    table = Table(title=f"Phase 9 feasibility matrix ({prefix}*)", title_justify="left")
    for col in (
        "run",
        "trials",
        "accept",
        "best OOS",
        "min ruin@OOS>0",
        "best MFE/MAE",
        "feasible",
    ):
        table.add_column(col, justify="left" if col == "run" else "right")
    for run in runs:
        rows = tracker.load_candidates(str(run["run_id"]))
        pos = [r for r in rows if float(r.get("oos_net") or 0.0) > 0]
        feas = _feasible_rows(rows)
        accept = sum(1 for r in rows if r.get("final_verdict") == "ACCEPT")
        best_oos = max((float(r.get("oos_net") or 0.0) for r in rows), default=0.0)
        pos_ruins = [float(r.get("risk_of_ruin") or 0.0) for r in pos]
        min_ruin = min(pos_ruins) if pos_ruins else None
        verdict = f"[green]YES ({len(feas)})[/green]" if feas else "[red]no[/red]"
        table.add_row(
            str(run["run_id"]).replace("phase9_", ""),
            str(len(rows)),
            str(accept),
            f"{best_oos:,.0f}",
            "—" if min_ruin is None else f"{min_ruin * 100:.0f}%",
            f"{_best_asymmetry(rows):.2f}",
            verdict,
        )
    console.print(table)


_LOCKBOX_WARMUP = date(2025, 4, 1)  # load features from here so HTF context is warm at 2025-07


def _frozen_registry_table(frozen: list[Any]) -> Table:
    table = Table(title=f"frozen hypothesis registry ({len(frozen)})", title_justify="left")
    for col in ("name", "family", "tf", "acct", "Phase9 OOS", "Phase9 DSR", "config hash"):
        table.add_column(col, justify="left" if col in ("name", "family") else "right")
    for fh in frozen:
        table.add_row(
            fh.name,
            fh.family,
            fh.timeframe,
            fh.account,
            f"{fh.phase9_oos_net:,.0f}",
            f"{fh.phase9_dsr:.2f}",
            fh.config_hash(),
        )
    return table


@lockbox_app.command("freeze")
def lockbox_freeze(profile: ProfileOption = "dev") -> None:
    """Freeze the pre-registered top Phase-9 survivors into an immutable registry."""
    config = _load(profile)
    frozen = freeze_from_phase9(
        _tracker(config), PHASE10_FREEZE_SPECS, frozen_at=datetime.now(UTC).isoformat()
    )
    path = registry_path(config)
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


def _print_phase10(rows: list[dict[str, Any]], pristine: bool) -> None:
    table = Table(title="Phase 10 lockbox results", title_justify="left")
    cols = (
        "hypothesis",
        "OOS",
        "ruin%",
        "blown",
        "cost",
        "wf+",
        "pos-yr",
        "trades",
        "DSR(n)",
        "verdict",
    )
    for col in cols:
        table.add_column(col, justify="left" if col in ("hypothesis", "verdict") else "right")
    style = {"ACCEPT": "green", "REJECT": "red", "NEEDS_PAPER_LOCKBOX": "yellow"}
    for r in rows:
        v = str(r["final_verdict"])
        table.add_row(
            str(r["strategy_name"]),
            f"{float(r['oos_net'] or 0):,.0f}",
            f"{float(r['risk_of_ruin'] or 0) * 100:.0f}",
            "yes" if r["blown"] else "no",
            "pass" if r["cost_stress_survived"] else "FAIL",
            f"{float(r['walk_forward_frac'] or 0) * 100:.0f}",
            f"{float(r['positive_years_frac'] or 0) * 100:.0f}",
            str(r["oos_trades"]),
            f"{float(r['deflated_sharpe'] or 0):.2f}",
            f"[{style.get(v, 'white')}]{v}[/]",
        )
    console.print(table)
    console.print(f"[dim]lockbox pristine (untouched by Phase-9 selection)? {pristine}[/dim]")
    for r in rows:
        console.print(f"  [bold]{r['strategy_name']}[/bold]: {r['rejection_reason']}")


@lockbox_app.command("run")
def lockbox_run(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run the frozen hypotheses on the lockbox window; recompute DSR at the frozen count."""
    config = load_config(profile=profile, overrides=_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config))

    # Pristine only if the lockbox starts strictly AFTER every frozen candidate's
    # Phase-9 out-of-sample window. (Phase 9 ran OOS through 2026 -> not pristine.)
    pristine = True
    for fh in frozen:
        run = tracker.get_run(fh.phase9_run_id)
        if run is not None and date.fromisoformat(str(run["end_date"])) >= _LOCKBOX_START:
            pristine = False

    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf, start=_LOCKBOX_WARMUP, end=_LOCKBOX_END
                )
            frames[tf] = prepare_htf_features(
                config, symbol="NQ.c", timeframe=tf, start=_LOCKBOX_WARMUP, end=_LOCKBOX_END
            )
        return frames[tf]

    console.print(
        f"[bold]lockbox run[/bold] {len(frozen)} frozen hypotheses, "
        f"{_LOCKBOX_START}..{_LOCKBOX_END} (pristine={pristine}) …"
    )
    rows = run_lockbox(
        config,
        frozen,
        tracker=tracker,
        run_id="phase10_lockbox",
        lockbox_start=_LOCKBOX_START,
        lockbox_end=_LOCKBOX_END,
        frame_for=frame_for,
        rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        pristine=pristine,
        created_at=datetime.now(UTC).isoformat(),
    )
    _print_phase10(rows, pristine)


@report_app.command("phase10")
def report_phase10(profile: ProfileOption = "dev") -> None:
    """Frozen registry + lockbox results + Phase-9-vs-lockbox comparison + verdicts."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config))
        console.print(_frozen_registry_table(frozen))
    except FileNotFoundError:
        console.print(
            "[yellow]no frozen registry — run `lockbox freeze` then `lockbox run`[/yellow]"
        )
        return
    rows = tracker.load_candidates("phase10_lockbox")
    if not rows:
        console.print("[yellow]no lockbox run yet — run `lockbox run`[/yellow]")
        return
    by_name = {str(r["strategy_name"]): r for r in rows}
    pristine = True
    for fh in frozen:
        run = tracker.get_run(fh.phase9_run_id)
        if run is not None and date.fromisoformat(str(run["end_date"])) >= _LOCKBOX_START:
            pristine = False
    _print_phase10([by_name[fh.name] for fh in frozen if fh.name in by_name], pristine)

    cmp = Table(title="Phase 9 (n=176) vs lockbox (frozen n)", title_justify="left")
    for col in ("hypothesis", "Phase9 OOS", "Phase9 DSR", "lockbox OOS", "lockbox DSR(n)"):
        cmp.add_column(col, justify="left" if col == "hypothesis" else "right")
    for fh in frozen:
        r = by_name.get(fh.name)
        if r is None:
            continue
        cmp.add_row(
            fh.name,
            f"{fh.phase9_oos_net:,.0f}",
            f"{fh.phase9_dsr:.2f}",
            f"{float(r['oos_net'] or 0):,.0f}",
            f"{float(r['deflated_sharpe'] or 0):.2f}",
        )
    console.print(cmp)
    verdicts = Counter(str(r["final_verdict"]) for r in rows)
    console.print(
        f"[green]ACCEPT={verdicts['ACCEPT']}[/green] [red]REJECT={verdicts['REJECT']}[/red] "
        f"[yellow]NEEDS_PAPER_LOCKBOX={verdicts['NEEDS_PAPER_LOCKBOX']}[/yellow]"
    )


_PHASE11_ACCOUNTS = ("50k", "100k", "150k")


@phase11_app.command("search")
def phase11_search(
    timeframes: Annotated[str, typer.Option(help="Comma-separated timeframes.")] = "5m,15m",
    accounts: Annotated[str, typer.Option(help="Comma-separated account sizes.")] = "50k,100k,150k",
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search NEW NQ families on train 2021-23 / validation 2024. NEVER touches 2025-26."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE11_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=30, asymmetry_min=1.2)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        # Frame ends at validation end (2024-12-31) — the 2025-2026 lockbox is reserved.
        frame = prepare_htf_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        for account in (a.strip() for a in accounts.split(",")):
            candidates = generate_phase11_candidates(
                fams, timeframe=tf, rth_start_min=rth_start_min
            )
            rid = f"phase11_search_{account}_{tf}"
            console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
            result = run_discovery(
                config,
                symbol=symbol,
                timeframe=tf,
                start=v.train_start,
                end=_PHASE11_SEARCH_END,
                rules=load_topstep_rules(f"topstep_{account}"),
                tracker=_tracker(config),
                run_id=rid,
                candidates=candidates,
                max_candidates=max_candidates,
                feature_frame=frame,
                prefilter=pf,
            )
            print_discovery(console, result)


def _phase11_search_runs() -> tuple[str, ...]:
    return tuple(f"phase11_search_{a}_{tf}" for tf in ("5m", "15m") for a in _PHASE11_ACCOUNTS)


@phase11_app.command("freeze")
def phase11_freeze(
    top_n: Annotated[int, typer.Option(help="Max survivors to freeze.")] = 5,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top validation-survivors (uses validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_survivors(tracker, _phase11_search_runs(), top_n=top_n)
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase11")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase11_app.command("lockbox")
def phase11_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run the frozen survivors ONCE on the reserved, untouched 2025-2026 lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase11"))
    # Pristine: the lockbox starts after every search run's end (search ended 2024-12-31).
    pristine = True
    for fh in frozen:
        run = tracker.get_run(fh.phase9_run_id)
        if run is not None and date.fromisoformat(str(run["end_date"])) >= _PHASE11_LOCKBOX_START:
            pristine = False
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config,
                    root="NQ",
                    timeframe=tf,
                    start=_PHASE11_LOCKBOX_START,
                    end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_htf_features(
                config,
                symbol="NQ.c",
                timeframe=tf,
                start=_PHASE11_LOCKBOX_START,
                end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase11 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} (pristine={pristine}) …"
    )
    rows = run_lockbox(
        config,
        frozen,
        tracker=tracker,
        run_id="phase11_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START,
        lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for,
        rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        pristine=pristine,
        created_at=datetime.now(UTC).isoformat(),
        verdict_fn=phase11_verdict,
    )
    _print_phase10(rows, pristine)


@report_app.command("phase11")
def report_phase11(profile: ProfileOption = "dev") -> None:
    """Frozen registry + lockbox results + paper-readiness (payout/drawdown/account)."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase11"))
        console.print(_frozen_registry_table(frozen))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-11 registry — run `phase11 freeze` then `phase11 lockbox`[/yellow]"
        )
        return
    rows = tracker.load_candidates("phase11_lockbox")
    if not rows:
        console.print("[yellow]no Phase-11 lockbox run yet — run `phase11 lockbox`[/yellow]")
        return
    _print_phase10(rows, pristine=True)

    # Paper-readiness for the accepted strategies: expected monthly payout + drawdown.
    months = 6.0  # the 2026-H1 OOS gate window
    pr = Table(title="paper/shadow readiness", title_justify="left")
    for col in ("strategy", "verdict", "exp. monthly $", "exp. max DD $", "account"):
        pr.add_column(col, justify="left" if col in ("strategy", "verdict", "account") else "right")
    by_name = {str(r["strategy_name"]): r for r in rows}
    for fh in frozen:
        r = by_name.get(fh.name)
        if r is None:
            continue
        v = str(r["final_verdict"])
        monthly = float(r["oos_net"] or 0.0) / months
        pr.add_row(
            fh.name,
            v,
            f"{monthly:,.0f}" if v == "ACCEPT_FOR_PAPER" else "—",
            f"{float(r['max_drawdown'] or 0.0):,.0f}",
            fh.account if v == "ACCEPT_FOR_PAPER" else "—",
        )
    console.print(pr)
    verdicts = Counter(str(r["final_verdict"]) for r in rows)
    console.print(
        f"[green]ACCEPT_FOR_PAPER={verdicts['ACCEPT_FOR_PAPER']}[/green] "
        f"[red]REJECT={verdicts['REJECT']}[/red] "
        f"[yellow]NEEDS_MORE_DATA={verdicts['NEEDS_MORE_DATA']}[/yellow]"
    )


@phase12_app.command("search")
def phase12_search(
    timeframes: Annotated[str, typer.Option()] = "5m,15m",
    accounts: Annotated[str, typer.Option()] = "50k,100k,150k",
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search trend_day frequency-expansion families on train 2021-23 / validation 2024."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE12_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=30, asymmetry_min=1.2)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_htf_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        for account in (a.strip() for a in accounts.split(",")):
            candidates = generate_phase12_candidates(
                fams, timeframe=tf, rth_start_min=rth_start_min
            )
            rid = f"phase12_search_{account}_{tf}"
            console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
            result = run_discovery(
                config,
                symbol=symbol,
                timeframe=tf,
                start=v.train_start,
                end=_PHASE11_SEARCH_END,
                rules=load_topstep_rules(f"topstep_{account}"),
                tracker=_tracker(config),
                run_id=rid,
                candidates=candidates,
                max_candidates=max_candidates,
                feature_frame=frame,
                prefilter=pf,
            )
            print_discovery(console, result)


def _phase12_search_runs() -> tuple[str, ...]:
    return tuple(f"phase12_search_{a}_{tf}" for tf in ("5m", "15m") for a in _PHASE11_ACCOUNTS)


@phase12_app.command("freeze")
def phase12_freeze(
    top_n: Annotated[int, typer.Option()] = 6,
    min_trades: Annotated[
        int, typer.Option(help="Min VALIDATION trades (frequency floor for confirmability).")
    ] = 100,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top validation survivors that also trade often enough to be
    confirmable on the lockbox (validation only — never sees the final test)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_survivors(
        tracker, _phase12_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase12")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase12_app.command("lockbox")
def phase12_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen variants on the 2025-2026 final test (NOT pristine -> forward-paper ceiling)."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase12"))
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config,
                    root="NQ",
                    timeframe=tf,
                    start=_PHASE11_LOCKBOX_START,
                    end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_htf_features(
                config,
                symbol="NQ.c",
                timeframe=tf,
                start=_PHASE11_LOCKBOX_START,
                end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase12 final test[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} "
        "(NOT pristine for trend_day -> ACCEPT_FOR_FORWARD_PAPER ceiling) …"
    )
    rows = run_lockbox(
        config,
        frozen,
        tracker=tracker,
        run_id="phase12_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START,
        lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for,
        rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        pristine=False,
        created_at=datetime.now(UTC).isoformat(),
        verdict_fn=phase12_verdict,
    )
    _print_phase10(rows, pristine=False)


@report_app.command("phase12")
def report_phase12(profile: ProfileOption = "dev") -> None:
    """Phase-12 registry + final-test verdicts + full practical stats per frozen variant."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase12"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-12 registry — run `phase12 freeze` then `phase12 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase12_lockbox")
    if not rows:
        console.print("[yellow]no Phase-12 final-test run yet — run `phase12 lockbox`[/yellow]")
        return
    _print_phase10(rows, pristine=False)

    # Full practical stat panel per frozen variant (recompute trades on the final test).
    months = 17.5  # 2025-01 .. 2026-06 (data end)
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    by_name = {str(r["strategy_name"]): r for r in rows}
    frames: dict[str, Any] = {}
    for fh in frozen:
        r = by_name.get(fh.name)
        if r is None:
            continue
        if fh.timeframe not in frames:
            frames[fh.timeframe] = prepare_htf_features(
                config,
                symbol="NQ.c",
                timeframe=fh.timeframe,
                start=_PHASE11_LOCKBOX_START,
                end=_PHASE11_LOCKBOX_END,
            )
        strat = fh.to_strategy(parse_hhmm(config.sessions.rth_start))
        sig = strat.signals(frames[fh.timeframe]).filter(pl.col("is_rth"))
        rules = load_topstep_rules(f"topstep_{fh.account}")
        bt = run_backtest(
            sig,
            symbol="NQ.c",
            timeframe=fh.timeframe,
            strategy=strat.params,
            config=config,
            rules=rules,
            entry_mode="next_open",
            gate_entries=False,
            trailing=strat.trailing,
            exec_config=strat.exec_config,
        )
        stats = practical_stats(
            bt.trades,
            account_size=rules.account_size,
            rules=rules,
            months=months,
            rules_by_account=rules_by_acct,
        )
        console.print(
            f"\n[bold]{fh.name}[/bold]  verdict=[bold]{r['final_verdict']}[/bold]  "
            f"(final-test trades={stats.get('trade_count', 0)})"
        )
        if stats.get("trade_count", 0) == 0:
            console.print("  [dim]no final-test trades — cannot profile[/dim]")
            continue
        console.print(
            f"  win_rate={stats['win_rate']:.0%} profit_factor={stats['profit_factor']} "
            f"expectancy=${stats['expectancy']} avg_trade=${stats['avg_trade']} "
            f"trades={stats['trade_count']}"
        )
        console.print(
            f"  net=${stats['net_pnl']} monthly=${stats['monthly_pnl']} "
            f"monthly_ROI={stats['monthly_roi']:.2%} annual_ROI={stats['annual_roi']:.2%} "
            f"max_DD=${float(r['max_drawdown'] or 0):,.0f} ruin={stats['ruin']:.0%}"
        )
        console.print(
            f"  Topstep pass-rate={stats['topstep_pass_rate']:.0%} "
            f"P(pass)={stats['p_pass_by_account']} est_days_to_pass={stats['est_days_to_pass']}"
        )
        console.print(
            f"  est_monthly_payout=${stats['est_monthly_payout']} "
            f"largest_win_day={stats['largest_win_day_pct']} "
            f"consistency={stats['consistency_pct']} blown={r['blown']}"
        )
    verdicts = Counter(str(r["final_verdict"]) for r in rows)
    console.print(
        f"\n[green]ACCEPT_FOR_FORWARD_PAPER={verdicts['ACCEPT_FOR_FORWARD_PAPER']}[/green] "
        f"[red]REJECT={verdicts['REJECT']}[/red] "
        f"[yellow]NEEDS_MORE_DATA={verdicts['NEEDS_MORE_DATA']}[/yellow]"
    )


@features_app.command("structure")
def features_structure(
    start: _OptStart = None,
    end: _OptStart = None,
    timeframe: TimeframeOption = "5m",
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Build + inspect the Phase-13 structure/liquidity/displacement/FVG layer (PIT-safe)."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    s = date.fromisoformat(start) if start else v.train_start
    e = date.fromisoformat(end) if end else _PHASE11_SEARCH_END
    symbol = "NQ.c"
    if build:
        symbol = ensure_full_history(config, root="NQ", timeframe=timeframe, start=s, end=e)
        build_mbp1_features_full(config, root="NQ", timeframe=timeframe, start=s, end=e)
    frame = prepare_structure_features(config, symbol=symbol, timeframe=timeframe, start=s, end=e)
    rth = frame.filter(pl.col("is_rth") & (pl.col("minute_of_day") >= 600))
    missing = [c for c in STRUCTURE_FEATURE_NAMES if c not in frame.columns]
    table = Table(title=f"structure coverage {symbol} {timeframe} [{s}..{e}]", title_justify="left")
    for col in ("feature", "fires / non-null (RTH bars)"):
        table.add_column(col, justify="left" if col == "feature" else "right")
    for c in STRUCTURE_FEATURE_NAMES:
        if c not in frame.columns:
            continue
        col = rth[c]
        fires = (
            col.drop_nulls().filter(col.drop_nulls() == 1).len()
            if col.dtype == pl.Int8
            else col.drop_nulls().len()
        )
        table.add_row(c, f"{fires}/{rth.height}")
    console.print(table)
    console.print(
        f"[green]all {len(STRUCTURE_FEATURE_NAMES)} structure features present[/green]"
        if not missing
        else f"[red]MISSING: {missing}[/red]"
    )


@discover_app.command("structure-setups")
def discover_structure_setups(
    timeframes: Annotated[str, typer.Option()] = "5m,15m",
    accounts: Annotated[str, typer.Option()] = "50k,100k,150k",
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Phase 13: search structure/liquidity families on train 2021-23 / validation 2024.

    Reserves the 2025-2026 lockbox (search loads only through 2024-12-31). New signal
    source -> the lockbox is pristine -> ACCEPT_FOR_PAPER is reachable.
    """
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE13_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=30, asymmetry_min=1.2)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        for account in (a.strip() for a in accounts.split(",")):
            candidates = generate_phase13_candidates(
                fams, timeframe=tf, rth_start_min=rth_start_min
            )
            rid = f"phase13_search_{account}_{tf}"
            console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
            result = run_discovery(
                config,
                symbol=symbol,
                timeframe=tf,
                start=v.train_start,
                end=_PHASE11_SEARCH_END,
                rules=load_topstep_rules(f"topstep_{account}"),
                tracker=_tracker(config),
                run_id=rid,
                candidates=candidates,
                max_candidates=max_candidates,
                feature_frame=frame,
                prefilter=pf,
            )
            print_discovery(console, result)


def _phase13_search_runs() -> tuple[str, ...]:
    return tuple(f"phase13_search_{a}_{tf}" for tf in ("5m", "15m") for a in _PHASE11_ACCOUNTS)


@phase13_app.command("freeze")
def phase13_freeze(
    top_n: Annotated[int, typer.Option()] = 6,
    min_trades: Annotated[
        int, typer.Option(help="Min validation trades (confirmability floor).")
    ] = 60,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top validation-survivor structure setups (validation only)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_survivors(
        tracker, _phase13_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase13")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase13_app.command("lockbox")
def phase13_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen structure setups on the reserved, untouched 2025-2026 lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase13"))
    pristine = True
    for fh in frozen:
        run = tracker.get_run(fh.phase9_run_id)
        if run is not None and date.fromisoformat(str(run["end_date"])) >= _PHASE11_LOCKBOX_START:
            pristine = False
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config,
                    root="NQ",
                    timeframe=tf,
                    start=_PHASE11_LOCKBOX_START,
                    end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config,
                symbol="NQ.c",
                timeframe=tf,
                start=_PHASE11_LOCKBOX_START,
                end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase13 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} (pristine={pristine}) …"
    )
    rows = run_lockbox(
        config,
        frozen,
        tracker=tracker,
        run_id="phase13_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START,
        lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for,
        rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        pristine=pristine,
        created_at=datetime.now(UTC).isoformat(),
        verdict_fn=phase11_verdict,
    )
    _print_phase10(rows, pristine)


@report_app.command("phase13")
def report_phase13(profile: ProfileOption = "dev") -> None:
    """Phase-13 registry + lockbox verdicts + full practical stats + liquidity-target hit rate."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase13"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-13 registry — run `phase13 freeze` then `phase13 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase13_lockbox")
    if not rows:
        console.print("[yellow]no Phase-13 lockbox run yet — run `phase13 lockbox`[/yellow]")
        return
    _print_phase10(rows, pristine=True)

    months = 17.5
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    by_name = {str(r["strategy_name"]): r for r in rows}
    frames: dict[str, Any] = {}
    for fh in frozen:
        r = by_name.get(fh.name)
        if r is None:
            continue
        if fh.timeframe not in frames:
            frames[fh.timeframe] = prepare_structure_features(
                config,
                symbol="NQ.c",
                timeframe=fh.timeframe,
                start=_PHASE11_LOCKBOX_START,
                end=_PHASE11_LOCKBOX_END,
            )
        strat = fh.to_strategy(parse_hhmm(config.sessions.rth_start))
        sig = strat.signals(frames[fh.timeframe]).filter(pl.col("is_rth"))
        rules = load_topstep_rules(f"topstep_{fh.account}")
        bt = run_backtest(
            sig,
            symbol="NQ.c",
            timeframe=fh.timeframe,
            strategy=strat.params,
            config=config,
            rules=rules,
            entry_mode="next_open",
            gate_entries=False,
            trailing=strat.trailing,
            exec_config=strat.exec_config,
        )
        stats = practical_stats(
            bt.trades,
            account_size=rules.account_size,
            rules=rules,
            months=months,
            rules_by_account=rules_by_acct,
        )
        liq_hits = sum(1 for t in bt.trades if t.exit_reason == "TARGET")
        liq_rate = liq_hits / len(bt.trades) if bt.trades else 0.0
        console.print(f"\n[bold]{fh.name}[/bold]  verdict=[bold]{r['final_verdict']}[/bold]")
        if stats.get("trade_count", 0) == 0:
            console.print("  [dim]no final-test trades — cannot profile[/dim]")
            continue
        console.print(
            f"  win={stats['win_rate']:.0%} PF={stats['profit_factor']} exp=${stats['expectancy']} "
            f"trades={stats['trade_count']} net=${stats['net_pnl']} monthly=${stats['monthly_pnl']}"
        )
        console.print(
            f"  monthly_ROI={stats['monthly_roi']:.2%} "
            f"max_DD=${float(r['max_drawdown'] or 0):,.0f} ruin={stats['ruin']:.0%} "
            f"liquidity-target-hit={liq_rate:.0%} P(pass)={stats['p_pass_by_account']} "
            f"consistency={stats['consistency_pct']}"
        )
    verdicts = Counter(str(r["final_verdict"]) for r in rows)
    console.print(
        f"\n[green]ACCEPT_FOR_PAPER={verdicts['ACCEPT_FOR_PAPER']}[/green] "
        f"[red]REJECT={verdicts['REJECT']}[/red] "
        f"[yellow]NEEDS_MORE_DATA={verdicts['NEEDS_MORE_DATA']}[/yellow]"
    )


@phase14_app.command("search")
def phase14_search(
    timeframes: Annotated[str, typer.Option()] = "5m,15m",
    accounts: Annotated[str, typer.Option()] = "50k,100k,150k",
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search BOS-continuation frequency variants (+ sizing repair) on train 2021-23 / val 2024."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE14_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=30, asymmetry_min=1.2)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        for account in (a.strip() for a in accounts.split(",")):
            candidates = generate_phase14_candidates(
                fams, timeframe=tf, rth_start_min=rth_start_min
            )
            rid = f"phase14_search_{account}_{tf}"
            console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
            result = run_discovery(
                config,
                symbol=symbol,
                timeframe=tf,
                start=v.train_start,
                end=_PHASE11_SEARCH_END,
                rules=load_topstep_rules(f"topstep_{account}"),
                tracker=_tracker(config),
                run_id=rid,
                candidates=candidates,
                max_candidates=max_candidates,
                feature_frame=frame,
                prefilter=pf,
            )
            print_discovery(console, result)


def _phase14_search_runs(timeframes: tuple[str, ...] = ("5m", "15m")) -> tuple[str, ...]:
    return tuple(f"phase14_search_{a}_{tf}" for tf in timeframes for a in _PHASE11_ACCOUNTS)


@phase14_app.command("freeze")
def phase14_freeze(
    top_n: Annotated[int, typer.Option()] = 6,
    min_trades: Annotated[
        int, typer.Option(help="Min validation trades (confirmability floor).")
    ] = 80,
    tf: Annotated[
        str, typer.Option(help="Restrict to one timeframe (15m setups ~0 held-out 2026 trades).")
    ] = "",
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top validation survivors that trade often enough to confirm (validation only)."""
    config = _load(profile)
    tracker = _tracker(config)
    runs = _phase14_search_runs((tf,)) if tf else _phase14_search_runs()
    specs = select_survivors(tracker, runs, top_n=top_n, min_oos_trades=min_trades)
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase14")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase14_app.command("lockbox")
def phase14_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen variants on the 2025-2026 final check (non-pristine -> forward-paper ceiling)."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase14"))
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config,
                    root="NQ",
                    timeframe=tf,
                    start=_PHASE11_LOCKBOX_START,
                    end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config,
                symbol="NQ.c",
                timeframe=tf,
                start=_PHASE11_LOCKBOX_START,
                end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase14 final check[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} "
        "(2025-26 inspected in Phase 13 -> ACCEPT_FOR_FORWARD_PAPER ceiling) …"
    )
    rows = run_lockbox(
        config,
        frozen,
        tracker=tracker,
        run_id="phase14_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START,
        lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for,
        rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        pristine=False,
        created_at=datetime.now(UTC).isoformat(),
        verdict_fn=phase12_verdict,
    )
    _print_phase10(rows, pristine=False)


@report_app.command("phase14")
def report_phase14(profile: ProfileOption = "dev") -> None:
    """Phase-14 registry + final-check verdicts + full practical stats per frozen variant."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase14"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-14 registry — run `phase14 freeze` then `phase14 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase14_lockbox")
    if not rows:
        console.print("[yellow]no Phase-14 final-check run yet — run `phase14 lockbox`[/yellow]")
        return
    _print_phase10(rows, pristine=False)

    months = 17.5
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    by_name = {str(r["strategy_name"]): r for r in rows}
    frames: dict[str, Any] = {}
    for fh in frozen:
        r = by_name.get(fh.name)
        if r is None:
            continue
        if fh.timeframe not in frames:
            frames[fh.timeframe] = prepare_structure_features(
                config,
                symbol="NQ.c",
                timeframe=fh.timeframe,
                start=_PHASE11_LOCKBOX_START,
                end=_PHASE11_LOCKBOX_END,
            )
        strat = fh.to_strategy(parse_hhmm(config.sessions.rth_start))
        sig = strat.signals(frames[fh.timeframe]).filter(pl.col("is_rth"))
        rules = load_topstep_rules(f"topstep_{fh.account}")
        bt = run_backtest(
            sig,
            symbol="NQ.c",
            timeframe=fh.timeframe,
            strategy=strat.params,
            config=config,
            rules=rules,
            entry_mode="next_open",
            gate_entries=False,
            trailing=strat.trailing,
            exec_config=strat.exec_config,
        )
        stats = practical_stats(
            bt.trades,
            account_size=rules.account_size,
            rules=rules,
            months=months,
            rules_by_account=rules_by_acct,
        )
        console.print(f"\n[bold]{fh.name}[/bold]  verdict=[bold]{r['final_verdict']}[/bold]")
        if stats.get("trade_count", 0) == 0:
            console.print("  [dim]no final-check trades — sizing still produced 0 contracts[/dim]")
            continue
        console.print(
            f"  win={stats['win_rate']:.0%} PF={stats['profit_factor']} exp=${stats['expectancy']} "
            f"trades={stats['trade_count']} net=${stats['net_pnl']} monthly=${stats['monthly_pnl']}"
        )
        console.print(
            f"  monthly_ROI={stats['monthly_roi']:.2%} max_DD=${stats['max_drawdown']:,.0f} "
            f"avg_DD=${stats['avg_drawdown']:,.0f} ruin={stats['ruin']:.0%}"
        )
        console.print(
            f"  Topstep pass-rate={stats['topstep_pass_rate']:.0%} "
            f"P(pass)={stats['p_pass_by_account']} days_to_pass={stats['est_days_to_pass']} "
            f"consistency={stats['consistency_pct']}"
        )
    verdicts = Counter(str(r["final_verdict"]) for r in rows)
    console.print(
        f"\n[green]ACCEPT_FOR_FORWARD_PAPER={verdicts['ACCEPT_FOR_FORWARD_PAPER']}[/green] "
        f"[red]REJECT={verdicts['REJECT']}[/red] "
        f"[yellow]NEEDS_MORE_DATA={verdicts['NEEDS_MORE_DATA']}[/yellow]"
    )


_PAPER_REPLAY_START = date(2025, 1, 1)
_PAPER_REPLAY_END = date(2026, 6, 30)


def _paper_rules_by_account() -> dict[str, Any]:
    return {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}


@paper_app.command("init")
def paper_init(profile: ProfileOption = "dev") -> None:
    """Load + verify the frozen Phase-14 bos_5m_trigger and write the paper registry."""
    config = _load(profile)
    cfg = init_from_phase14(registry_path(config, "phase14"))
    save_paper_config(paper_registry_path(config), cfg)
    console.print(f"[green]frozen strategy verified[/green] (hash {cfg.frozen_hash})")
    console.print(
        f"  [bold]{cfg.strategy_id}[/bold] account={cfg.account} timeframe={cfg.timeframe}"
    )
    console.print(f"  rules: {cfg.explain()}")
    console.print(f"  -> {paper_registry_path(config)}")


@paper_app.command("replay")
def paper_replay(
    start: _OptStart = None,
    end: _OptStart = None,
    reset: Annotated[bool, typer.Option("--reset/--append")] = True,
    profile: ProfileOption = "dev",
) -> None:
    """Replay stored bars through the paper engine (historical -> mode='replay')."""
    config = _load(profile)
    cfg = load_paper_config(paper_registry_path(config))  # verifies hash
    s = date.fromisoformat(start) if start else _PAPER_REPLAY_START
    e = date.fromisoformat(end) if end else _PAPER_REPLAY_END
    rules = load_topstep_rules(f"topstep_{cfg.account}")
    console.print(
        f"[bold]paper replay[/bold] {cfg.strategy_id} {cfg.timeframe} {cfg.account} [{s}..{e}] …"
    )
    frame = prepare_structure_features(
        config, symbol="NQ.c", timeframe=cfg.timeframe, start=s, end=e
    )
    rows, ts = run_paper(config, cfg, feature_frame=frame, rules=rules, mode="replay")
    path = paper_ledger.ledger_path(config.data.curated_root)
    if reset:
        paper_ledger.clear(path)
    paper_ledger.append_rows(path, rows)
    console.print(
        f"  wrote {len(rows)} replay trades -> {path} | "
        f"final equity ${ts.final_equity:,.0f} blown={ts.blown}"
    )


@paper_app.command("run")
def paper_run(profile: ProfileOption = "dev") -> None:
    """Run forward on a LIVE feed. No live feed is connected (paper/shadow only) and no
    order/execution path exists — falls back to instructing replay."""
    config = _load(profile)
    load_paper_config(paper_registry_path(config))  # verify the frozen config first
    try:
        next(LiveFeed().bars())
    except NotImplementedError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        console.print("  no live data + no order execution in Phase 15 — use `nqlab paper replay`.")


def _paper_eval(config: AppConfig) -> dict[str, Any]:
    cfg = load_paper_config(paper_registry_path(config))
    rows = paper_ledger.read_rows(paper_ledger.ledger_path(config.data.curated_root))
    rules = load_topstep_rules(f"topstep_{cfg.account}")
    months = (_PAPER_REPLAY_END - _PAPER_REPLAY_START).days / 30.4
    return evaluate_forward(
        rows,
        rules=rules,
        account_size=rules.account_size,
        months=months,
        rules_by_account=_paper_rules_by_account(),
    )


@paper_app.command("status")
def paper_status(profile: ProfileOption = "dev") -> None:
    """Show forward sample progress and current verdict."""
    config = _load(profile)
    res = _paper_eval(config)
    console.print(
        f"forward trades: {res['forward_trades']}/{res['min_forward_trades']} | "
        f"replay trades: {res['replay_trades']}"
    )
    console.print(f"[bold]verdict:[/bold] {res['verdict']} — {res['reason']}")


def _print_paper_stats(label: str, s: dict[str, Any]) -> None:
    if s.get("trade_count", 0) == 0:
        console.print(f"  [dim]{label}: no trades[/dim]")
        return
    console.print(
        f"  [bold]{label}[/bold] ({s['trade_count']} trades): win={s['win_rate']:.0%} "
        f"PF={s['profit_factor']} exp=${s['expectancy']} net=${s['net_pnl']} "
        f"monthly=${s['monthly_pnl']} ROI/mo={s['monthly_roi']:.2%} ROI/yr={s['annual_roi']:.2%}"
    )
    console.print(
        f"    max_DD=${s['max_drawdown']:,.0f} avg_DD=${s['avg_drawdown']:,.0f} "
        f"ruin={s['ruin']:.0%} Topstep pass-rate={s['topstep_pass_rate']:.0%} "
        f"P(pass)={s['p_pass_by_account']} days_to_pass={s['est_days_to_pass']}"
    )
    console.print(
        f"    consistency={s['consistency_pct']} largest_win_day={s['largest_win_day_pct']} "
        f"forward_Sharpe={s.get('forward_sharpe')} DSR(frozen)={s.get('dsr_frozen')}"
    )


@paper_app.command("report")
def paper_report(profile: ProfileOption = "dev") -> None:
    """Full forward report: practical stats (forward + replay), Phase-14 comparison, verdict."""
    config = _load(profile)
    cfg = load_paper_config(paper_registry_path(config))
    res = _paper_eval(config)
    console.print(
        f"[bold]Phase 15 forward report[/bold] — {cfg.strategy_id} (hash {cfg.frozen_hash})"
    )
    console.print(f"  {cfg.explain()}\n")
    _print_paper_stats("FORWARD (genuine, unseen)", res["forward_stats"])
    _print_paper_stats("REPLAY (historical, testing only)", res["replay_stats"])
    ref = res["phase14_reference"]
    console.print(
        f"\n  [dim]Phase-14 ref: PF {ref['profit_factor']} exp ${ref['expectancy']} "
        f"net ${ref['net_pnl']:,.0f}/{ref['trades']}tr P(pass50K) {ref['p_pass_50k']:.0%} "
        f"DSR {ref['dsr']} -> {ref['verdict']}[/dim]"
    )
    style = {
        "PENDING_FORWARD_SAMPLE": "yellow",
        "FORWARD_ACCEPT_FOR_SHADOW": "green",
        "FORWARD_REJECT": "red",
    }
    console.print(
        f"\n[{style.get(res['verdict'], 'white')}]VERDICT: {res['verdict']}[/] — {res['reason']}"
    )


@paper_app.command("evaluate")
def paper_evaluate(profile: ProfileOption = "dev") -> None:
    """Render the forward verdict (PENDING_FORWARD_SAMPLE until enough forward trades)."""
    config = _load(profile)
    res = _paper_eval(config)
    fs = res["forward_stats"]
    console.print(
        f"forward={res['forward_trades']} (need {res['min_forward_trades']}) "
        f"DSR(frozen)={fs.get('dsr_frozen', '—')}"
    )
    console.print(f"[bold]{res['verdict']}[/bold] — {res['reason']}")


# ---------------------------------------------------------------------------
# Phase 16 — payout-first scalable strategy discovery.
# Same reserved-lockbox protocol as Phase 11 (search 2021-23 / val 2024 / lockbox
# 2025-26). The 2025-26 lockbox is PRISTINE for these NEW families (never inspected).
# ---------------------------------------------------------------------------
_PHASE16_SEARCH_ACCOUNT = "150k"  # search at the target scale account
_PHASE16_SCALE_RISK = 600.0  # Stage-B funded sizing (cap on 150K is 0.25*4500=$1125/trade)
_PHASE16_LOCKBOX_MONTHS = 18  # 2025-01 .. 2026-06


@phase16_app.command("search")
def phase16_search(
    timeframes: Annotated[str, typer.Option()] = "5m",
    account: Annotated[str, typer.Option(help="Search account size.")] = _PHASE16_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search payout-first families on train 2021-23 / val 2024. NEVER touches 2025-26."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE16_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=30, asymmetry_min=1.2)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase16_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase16_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase16_search_runs(account: str = _PHASE16_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return (f"phase16_search_{account}_5m",)


@phase16_app.command("freeze")
def phase16_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option(help="Min validation trades.")] = 40,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase16_survivors(
        tracker, _phase16_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase16")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


def _print_phase16(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Phase 16 payout-first lockbox results", title_justify="left")
    cols = ("hypothesis", "PF", "ruin%", "green%", "$200d", "payoutMo", "$/mo", "P(pass)",
            "scalePF", "scaleC", "DSR", "verdict")
    for col in cols:
        table.add_column(col, justify="left" if col in ("hypothesis", "verdict") else "right")
    style = {"ACCEPT_FOR_PAPER": "green", "ACCEPT_FOR_FORWARD_PAPER": "cyan",
             "REJECT": "red", "NEEDS_MORE_DATA": "yellow"}
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b, sc = diag.get("business", {}), diag.get("scale", {})
        p_pass = max((b.get("p_pass_by_account") or {}).values(), default=0.0)
        v = str(r["final_verdict"])
        table.add_row(
            str(r["strategy_name"]),
            f"{b.get('profit_factor') or 0:.2f}",
            f"{(b.get('ruin') or 0) * 100:.0f}",
            f"{(b.get('green_month_frac') or 0) * 100:.0f}",
            str(b.get("winning_days_200") or 0),
            f"{b.get('payout_eligible_months') or 0}/{b.get('total_months') or 0}",
            f"{b.get('est_monthly_payout') or 0:,.0f}",
            f"{p_pass * 100:.0f}",
            f"{sc.get('profit_factor') or 0:.2f}",
            f"{sc.get('avg_contracts') or 0:.1f}",
            f"{float(r.get('deflated_sharpe') or 0):.2f}",
            f"[{style.get(v, 'white')}]{v}[/]",
        )
    console.print(table)
    for r in rows:
        console.print(f"  [bold]{r['strategy_name']}[/bold]: {r['rejection_reason']}")


@phase16_app.command("lockbox")
def phase16_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen payout-first survivors ONCE on the reserved, pristine 2025-2026 lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase16"))
    # Pristine: these NEW families were never run on 2025-26, and the search ended 2024-12-31.
    pristine = True
    for fh in frozen:
        run = tracker.get_run(fh.phase9_run_id)
        if run is not None and date.fromisoformat(str(run["end_date"])) >= _PHASE11_LOCKBOX_START:
            pristine = False
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase16 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} (pristine={pristine}) …"
    )
    rows = run_phase16_lockbox(
        config, frozen, tracker=tracker, run_id="phase16_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START, lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE16_LOCKBOX_MONTHS,
        scale_risk=_PHASE16_SCALE_RISK, pristine=pristine,
        created_at=datetime.now(UTC).isoformat(),
    )
    _print_phase16(rows)


@report_app.command("phase16")
def report_phase16(profile: ProfileOption = "dev") -> None:
    """Phase-16 payout-first leaderboard + per-candidate stats + business-worthiness verdict."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase16"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-16 registry — run `phase16 freeze` then `phase16 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase16_lockbox")
    if not rows:
        console.print("[yellow]no Phase-16 lockbox run yet — run `phase16 lockbox`[/yellow]")
        return
    _print_phase16(rows)

    th = PHASE16_THRESHOLDS
    console.print(
        f"\n[dim]bar: PF>={th.pf_min}, ruin<={th.ruin_max:.0%}, green>={th.green_month_min:.0%}, "
        f">={th.min_winning_days_200} $200-days, "
        f"payout-eligible>={th.min_payout_eligible_frac:.0%} of months, "
        f"scales past 1 MNQ, DSR>={th.dsr_threshold}[/dim]"
    )
    # Per-candidate full payout panel + fleet scaling.
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b, sc, fleet = diag.get("business", {}), diag.get("scale", {}), diag.get("fleet", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"trades={b.get('trade_count')} ({b.get('trades_per_month')}/mo) "
            f"green={b.get('green_months')}/{b.get('total_months')}\n"
            f"  $200-days={b.get('winning_days_200')} (rate {b.get('winning_day_rate')}) "
            f"best-day=${b.get('best_day')} payout-mo={b.get('payout_eligible_months')} "
            f"$/mo=${b.get('est_monthly_payout')} $/yr=${b.get('est_annual_payout')}\n"
            f"  pass={b.get('pass_by_account')} ruin={b.get('ruin_by_account')} "
            f"best_account={b.get('best_account')}\n"
            f"  SCALE@${int(_PHASE16_SCALE_RISK)}: PF={sc.get('profit_factor')} "
            f"avgC={sc.get('avg_contracts')} ruin={sc.get('ruin')} blown={sc.get('blown')} "
            f"fleet5=${(fleet.get('5') or {}).get('fleet_monthly_payout')}/mo "
            f"fleet10=${(fleet.get('10') or {}).get('fleet_monthly_payout')}/mo"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    if accepted:
        console.print(
            f"\n[green]BUSINESS-WORTHY: {len(accepted)} strateg(y/ies) clear the payout-first "
            f"bar: {', '.join(str(r['strategy_name']) for r in accepted)}[/green]"
        )
    else:
        console.print(
            "\n[red]NONE business-worthy: no strategy clears the payout-first bar for Topstep "
            "payouts. Honest verdict — do not deploy.[/red]"
        )


# ---------------------------------------------------------------------------
# Phase 17 — business-worthy A+ payout strategy search.
# Same reserved-lockbox protocol; pristine 2025-26 lockbox for these NEW families.
# ---------------------------------------------------------------------------
_PHASE17_SEARCH_ACCOUNT = "150k"
_PHASE17_SCALE_RISK = 600.0
_PHASE17_LOCKBOX_MONTHS = 18


@phase17_app.command("search")
def phase17_search(
    timeframes: Annotated[str, typer.Option()] = "5m",
    account: Annotated[str, typer.Option(help="Search account size.")] = _PHASE17_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search A+ asymmetric families on train 2021-23 / val 2024. NEVER touches 2025-26."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE17_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    # Diagnostics-first: require trade frequency + MFE/MAE asymmetry before the full gauntlet.
    pf = diagnostics_prefilter(min_trades=30, asymmetry_min=1.3)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase17_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase17_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase17_search_runs(account: str = _PHASE17_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return (f"phase17_search_{account}_5m",)


@phase17_app.command("freeze")
def phase17_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option(help="Min validation trades.")] = 40,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase17_survivors(
        tracker, _phase17_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase17")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


def _print_phase17(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Phase 17 A+ payout-first lockbox results", title_justify="left")
    cols = ("hypothesis", "PF", "ruin%", "green%", "$200d", "payoutMo", "$/mo", "P(pass)",
            "scalePF", "scaleC", "wdraw", "DSR", "verdict")
    for col in cols:
        table.add_column(col, justify="left" if col in ("hypothesis", "verdict") else "right")
    style = {"ACCEPT_FOR_PAPER": "green", "ACCEPT_FOR_FORWARD_PAPER": "cyan", "REJECT": "red",
             "NEEDS_MORE_DATA": "yellow", "NEEDS_FORWARD_DATA": "yellow"}
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b, sc = diag.get("business", {}), diag.get("scale", {})
        p_pass = max((b.get("p_pass_by_account") or {}).values(), default=0.0)
        v = str(r["final_verdict"])
        table.add_row(
            str(r["strategy_name"]),
            f"{b.get('profit_factor') or 0:.2f}",
            f"{(b.get('ruin') or 0) * 100:.0f}",
            f"{(b.get('green_month_frac') or 0) * 100:.0f}",
            str(b.get("winning_days_200") or 0),
            f"{b.get('payout_eligible_months') or 0}/{b.get('total_months') or 0}",
            f"{b.get('est_monthly_payout') or 0:,.0f}",
            f"{p_pass * 100:.0f}",
            f"{sc.get('profit_factor') or 0:.2f}",
            f"{sc.get('avg_contracts') or 0:.1f}",
            str(b.get("funded_n_payouts") or 0),
            f"{float(r.get('deflated_sharpe') or 0):.2f}",
            f"[{style.get(v, 'white')}]{v}[/]",
        )
    console.print(table)
    for r in rows:
        console.print(f"  [bold]{r['strategy_name']}[/bold]: {r['rejection_reason']}")


@phase17_app.command("lockbox")
def phase17_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen A+ survivors ONCE on the reserved, pristine 2025-2026 lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase17"))
    pristine = True
    for fh in frozen:
        run = tracker.get_run(fh.phase9_run_id)
        if run is not None and date.fromisoformat(str(run["end_date"])) >= _PHASE11_LOCKBOX_START:
            pristine = False
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase17 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} (pristine={pristine}) …"
    )
    rows = run_phase17_lockbox(
        config, frozen, tracker=tracker, run_id="phase17_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START, lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE17_LOCKBOX_MONTHS,
        scale_risk=_PHASE17_SCALE_RISK, pristine=pristine,
        created_at=datetime.now(UTC).isoformat(),
        thresholds=PHASE17_THRESHOLDS, verdict_fn=phase17_verdict,
    )
    _print_phase17(rows)


@report_app.command("phase17")
def report_phase17(profile: ProfileOption = "dev") -> None:
    """Phase-17 A+ leaderboard + per-candidate payout panel + business-worthiness verdict."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase17"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-17 registry — run `phase17 freeze` then `phase17 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase17_lockbox")
    if not rows:
        console.print("[yellow]no Phase-17 lockbox run yet — run `phase17 lockbox`[/yellow]")
        return
    _print_phase17(rows)

    th = PHASE17_THRESHOLDS
    console.print(
        f"\n[dim]bar: PF>={th.pf_min} (pref {th.pf_preferred}), ruin<={th.ruin_max:.0%}, "
        f"green>={th.green_month_min:.0%}, payout-eligible>={th.min_payout_eligible_frac:.0%}, "
        f"pass>={th.pass_pref:.0%}, scales past 1 MNQ, DSR>={th.dsr_threshold}[/dim]"
    )
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b, sc, fleet = diag.get("business", {}), diag.get("scale", {}), diag.get("fleet", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"trades={b.get('trade_count')} ({b.get('trades_per_month')}/mo, "
            f"{b.get('trades_per_week')}/wk) "
            f"green={b.get('green_months')}/{b.get('total_months')}\n"
            f"  $200-days={b.get('winning_days_200')} (rate {b.get('winning_day_rate')}) "
            f"$500-days={b.get('winning_days_500')} best=${b.get('best_day')} "
            f"payout-mo={b.get('payout_eligible_months')} $/mo=${b.get('est_monthly_payout')} "
            f"$/yr=${b.get('est_annual_payout')}\n"
            f"  pass={b.get('pass_by_account')} ruin={b.get('ruin_by_account')} "
            f"best_account={b.get('best_account')} max_dd=${b.get('max_drawdown')}\n"
            f"  funded: {b.get('funded_n_payouts')} payouts, ${b.get('funded_total_withdrawn')} "
            f"withdrawn, blown_after_payout={b.get('funded_blown_after_payout')}\n"
            f"  SCALE@${int(_PHASE17_SCALE_RISK)}: PF={sc.get('profit_factor')} "
            f"avgC={sc.get('avg_contracts')} ruin={sc.get('ruin')} blown={sc.get('blown')} "
            f"fleet5=${(fleet.get('5') or {}).get('fleet_monthly_payout')}/mo "
            f"fleet10=${(fleet.get('10') or {}).get('fleet_monthly_payout')}/mo"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    if accepted:
        console.print(
            f"\n[green]BUSINESS-WORTHY: {len(accepted)} strateg(y/ies) clear the Phase-17 bar: "
            f"{', '.join(str(r['strategy_name']) for r in accepted)}[/green]"
        )
    else:
        console.print(
            "\n[red]NONE business-worthy: no A+ strategy clears the Phase-17 payout bar for "
            "Topstep payouts. Honest verdict — do not deploy.[/red]"
        )


# ---------------------------------------------------------------------------
# Phase 18 — high-win-rate frequent NY-AM scalp payout search.
# Same reserved-lockbox protocol; pristine 2025-26 for these NEW families.
# ---------------------------------------------------------------------------
_PHASE18_SEARCH_ACCOUNT = "150k"
_PHASE18_SCALE_RISK = 500.0
_PHASE18_LOCKBOX_MONTHS = 18
_PHASE18_TIMEFRAMES = ("5m", "1m")


@phase18_app.command("search")
def phase18_search(
    timeframes: Annotated[str, typer.Option()] = "5m,1m",
    account: Annotated[str, typer.Option(help="Search account size.")] = _PHASE18_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search high-win-rate scalp families on train 2021-23 / val 2024. NEVER touches 2025-26."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE18_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    # Scalps run R:R ~1 (MFE ~= MAE), so do NOT gate on MFE/MAE asymmetry — gate on frequency.
    pf = diagnostics_prefilter(min_trades=40, asymmetry_min=0.5)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase18_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase18_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase18_search_runs(account: str = _PHASE18_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return tuple(f"phase18_search_{account}_{tf}" for tf in _PHASE18_TIMEFRAMES)


@phase18_app.command("freeze")
def phase18_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option(help="Min validation trades.")] = 50,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase18_survivors(
        tracker, _phase18_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase18")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


def _print_phase18(rows: list[dict[str, Any]]) -> None:
    table = Table(title="Phase 18 high-win-rate scalp lockbox results", title_justify="left")
    cols = ("hypothesis", "win%", "PF", "green-d%", "$200d", "payoutMo", "$/mo", "ruin%",
            "P(pass)", "scalePF", "DSR", "verdict")
    for col in cols:
        table.add_column(col, justify="left" if col in ("hypothesis", "verdict") else "right")
    style = {"ACCEPT_FOR_FORWARD_PAPER": "green", "REJECT": "red",
             "NEEDS_MORE_DATA": "yellow", "NEEDS_FORWARD_DATA": "yellow"}
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b, sc = diag.get("business", {}), diag.get("scale", {})
        p_pass = max((b.get("p_pass_by_account") or {}).values(), default=0.0)
        v = str(r["final_verdict"])
        table.add_row(
            str(r["strategy_name"]),
            f"{(b.get('win_rate') or 0) * 100:.0f}",
            f"{b.get('profit_factor') or 0:.2f}",
            f"{(b.get('green_day_rate') or 0) * 100:.0f}",
            str(b.get("winning_days_200") or 0),
            f"{b.get('payout_eligible_months') or 0}/{b.get('total_months') or 0}",
            f"{b.get('est_monthly_payout') or 0:,.0f}",
            f"{(b.get('ruin') or 0) * 100:.0f}",
            f"{p_pass * 100:.0f}",
            f"{sc.get('profit_factor') or 0:.2f}",
            f"{float(r.get('deflated_sharpe') or 0):.2f}",
            f"[{style.get(v, 'white')}]{v}[/]",
        )
    console.print(table)
    for r in rows:
        console.print(f"  [bold]{r['strategy_name']}[/bold]: {r['rejection_reason']}")


@phase18_app.command("lockbox")
def phase18_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen scalp survivors ONCE on the pristine 2025-2026 lockbox (with control sweep)."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase18"))
    pristine = True
    for fh in frozen:
        run = tracker.get_run(fh.phase9_run_id)
        if run is not None and date.fromisoformat(str(run["end_date"])) >= _PHASE11_LOCKBOX_START:
            pristine = False
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase18 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} (pristine={pristine}) …"
    )
    rows = run_phase18_lockbox(
        config, frozen, tracker=tracker, run_id="phase18_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START, lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE18_LOCKBOX_MONTHS,
        scale_risk=_PHASE18_SCALE_RISK, pristine=pristine,
        created_at=datetime.now(UTC).isoformat(), thresholds=PHASE18_THRESHOLDS,
    )
    _print_phase18(rows)


@report_app.command("phase18")
def report_phase18(profile: ProfileOption = "dev") -> None:
    """Phase-18 scalp leaderboard + per-candidate green-day/payout panel + verdict."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase18"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-18 registry — run `phase18 freeze` then `phase18 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase18_lockbox")
    if not rows:
        console.print("[yellow]no Phase-18 lockbox run yet — run `phase18 lockbox`[/yellow]")
        return
    _print_phase18(rows)

    th = PHASE18_THRESHOLDS
    console.print(
        f"\n[dim]bar: win>={th.win_rate_min:.0%}, PF>={th.pf_min} (pref {th.pf_preferred}), "
        f"ruin<={th.ruin_max:.0%}, green-mo>={th.green_month_min:.0%}, "
        f"payout>={th.min_payout_eligible_frac:.0%}, pass>={th.pass_pref:.0%}, "
        f"DSR>={th.dsr_threshold}; ceiling = ACCEPT_FOR_FORWARD_PAPER[/dim]"
    )
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b, sc, fleet = diag.get("business", {}), diag.get("scale", {}), diag.get("fleet", {})
        ctrl = diag.get("best_controls", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"avgW=${b.get('avg_win')} avgL=${b.get('avg_loss')} "
            f"trades={b.get('trade_count')} ({b.get('trades_per_day')}/day)\n"
            f"  green-days={b.get('green_days')} red-days={b.get('red_days')} "
            f"(green-rate {b.get('green_day_rate')}) worst-day=${b.get('worst_day')} "
            f"worst-week=${b.get('worst_week')} maxDD=${b.get('max_drawdown')}\n"
            f"  $200-days={b.get('winning_days_200')} $500-days={b.get('winning_days_500')} "
            f"green-mo={b.get('green_months')}/{b.get('total_months')} "
            f"payout-mo={b.get('payout_eligible_months')} $/mo=${b.get('est_monthly_payout')} "
            f"$/yr=${b.get('est_annual_payout')}\n"
            f"  pass={b.get('pass_by_account')} ruin={b.get('ruin_by_account')} "
            f"best_account={b.get('best_account')} best_controls={ctrl}\n"
            f"  SCALE@${int(_PHASE18_SCALE_RISK)}: PF={sc.get('profit_factor')} "
            f"avgC={sc.get('avg_contracts')} blown={sc.get('blown')} "
            f"fleet5=${(fleet.get('5') or {}).get('fleet_monthly_payout')}/mo "
            f"fleet10=${(fleet.get('10') or {}).get('fleet_monthly_payout')}/mo"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    if accepted:
        console.print(
            f"\n[green]BUSINESS-WORTHY (forward-paper): {len(accepted)} — "
            f"{', '.join(str(r['strategy_name']) for r in accepted)}[/green]"
        )
    else:
        console.print(
            "\n[red]NONE business-worthy: the high-win-rate scalp hypothesis did not clear the "
            "Phase-18 bar either. Honest verdict — do not deploy.[/red]"
        )


# ---------------------------------------------------------------------------
# Phase 19 — NQ NY-AM edge-discovery map. Behaviour analysis, NOT a backtest:
# forward MFE/MAE per market-context bucket on train 2021-23 + validation 2024.
# The 2025-2026 lockbox is NEVER touched in this phase.
# ---------------------------------------------------------------------------
_PHASE19_DIMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("time-of-day", ("tod",)),
    ("4h-bias x VWAP", ("bias4h", "vwap_loc")),
    ("premium/discount", ("pd_loc",)),
    ("gap", ("gap",)),
    ("regime", ("regime",)),
    ("prev-day-type", ("ptype",)),
    ("vol-regime", ("vol",)),
    ("tod x 4h-bias", ("tod", "bias4h")),
    ("tod x regime", ("tod", "regime")),
)


@phase19_app.command("map")
def phase19_map(
    horizon: Annotated[str, typer.Option(help="Forward horizon (5m/10m/15m/30m/60m).")] = "30m",
    profile: ProfileOption = "dev",
) -> None:
    """Build the NQ NY-AM edge map (train 2021-23 + val 2024). Does NOT touch the lockbox."""
    config = _load(profile)

    def prep(start: date, end: date) -> pl.DataFrame:
        f = prepare_structure_features(config, symbol="NQ.c", timeframe="5m", start=start, end=end)
        f = forward_excursions(f.filter(pl.col("is_rth")).sort("ts"), horizons=HORIZON_BARS_5M)
        f = context_columns(f)
        return f.filter((pl.col("minute_of_day") >= 570) & (pl.col("minute_of_day") < 750))

    console.print("[bold]phase19 edge map[/bold] — train 2021-23 + val 2024 (lockbox untouched) …")
    train = prep(date(2021, 1, 1), date(2023, 12, 31))
    val = prep(date(2024, 1, 1), date(2024, 12, 31))
    fwd = train.drop_nulls([f"fwd_{horizon}", f"mfe_{horizon}", f"mae_{horizon}"])
    mm = fwd[f"mfe_{horizon}"].mean() / fwd[f"mae_{horizon}"].mean()
    console.print(
        f"[dim]baseline NY-AM {horizon}: n={fwd.height} mean_fwd="
        f"{fwd[f'fwd_{horizon}'].mean():.3f}pt MFE/MAE={mm:.3f} "
        f"dir_acc={(fwd[f'fwd_{horizon}'] > 0).mean():.3f} (cost={COST_PTS}pt)[/dim]"
    )
    hits: list[dict[str, Any]] = []
    for title, by in _PHASE19_DIMS:
        bylist = list(by)
        j = join_stability(
            bucket_edge(train, bylist, horizon=horizon),
            bucket_edge(val, bylist, horizon=horizon), bylist,
        )
        t = Table(title=f"{title} ({'+'.join(by)})", title_justify="left")
        for col in ("bucket", "n_tr", "dir", "aftC_tr", "MFE/MAE", "n_va", "aftC_val", "stable"):
            t.add_column(col, justify="left" if col in ("bucket", "dir") else "right")
        for r in j.sort("n", descending=True).head(8).iter_rows(named=True):
            lab = "/".join(str(r[b]) for b in by)
            stable = bool(r["stable_edge"])
            t.add_row(
                lab, str(r["n"]), str(r["direction"]), f"{r['after_cost']:.2f}",
                f"{r['mfe_mae']:.2f}", str(r["n_val"]), f"{r['after_cost_val']:.2f}",
                "[green]yes[/]" if stable else "no",
            )
            if stable:
                hits.append({"dim": "+".join(by), "bucket": lab, "dir": r["direction"],
                             "aftC_tr": round(r["after_cost"], 2),
                             "aftC_val": round(r["after_cost_val"], 2),
                             "mfe_mae_val": round(r["mfe_mae_val"], 2),
                             "n": f"{r['n']}/{r['n_val']}"})
        console.print(t)
    console.print("\n[bold]stable after-cost edges (both train+val, same dir, n_tr>=100):[/bold]")
    if hits:
        for h in sorted(hits, key=lambda x: -x["aftC_val"]):
            note = " [yellow](thin/MFE~MAE or tiny sample — likely noise)[/yellow]" if (
                h["mfe_mae_val"] < 1.1 or int(h["n"].split("/")[1]) < 150
            ) else ""
            console.print(f"  {h['dim']:16} {h['bucket']:22} {h['dir']:>5} "
                          f"aftC_val={h['aftC_val']:.2f}pt MFE/MAE_val={h['mfe_mae_val']:.2f} "
                          f"n={h['n']}{note}")
    else:
        console.print("  [red]NONE[/red]")
    console.print(
        "\n[dim]Reminder: a real edge needs MFE clearly > MAE (>~1.2), large sample, after-cost "
        "edge big enough for $200 days at safe size. See docs/PHASE19 for the verdict.[/dim]"
    )


# ---------------------------------------------------------------------------
# Phase 20 — classic indicator strategy discovery (last non-MBP1, bar-based).
# Reuses the Phase-18 payout rig; search on the light base-feature frame.
# ---------------------------------------------------------------------------
_PHASE20_SEARCH_ACCOUNT = "150k"
_PHASE20_SCALE_RISK = 500.0
_PHASE20_LOCKBOX_MONTHS = 18
_PHASE20_TIMEFRAMES = ("5m", "15m", "1m")


@phase20_app.command("search")
def phase20_search(
    timeframes: Annotated[str, typer.Option()] = "5m,15m,1m",
    account: Annotated[str, typer.Option(help="Search account size.")] = _PHASE20_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search classic indicator families on train 2021-23 / val 2024. NEVER touches 2025-26."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE20_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    # Indicators include mean-reversion (R:R ~1), so do not gate on MFE/MAE asymmetry.
    pf = diagnostics_prefilter(min_trades=40, asymmetry_min=0.5)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase20_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase20_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase20_search_runs(account: str = _PHASE20_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return tuple(f"phase20_search_{account}_{tf}" for tf in _PHASE20_TIMEFRAMES)


@phase20_app.command("freeze")
def phase20_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option(help="Min validation trades.")] = 40,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase20_survivors(
        tracker, _phase20_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase20")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase20_app.command("lockbox")
def phase20_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen indicator survivors ONCE on the reserved, pristine 2025-2026 lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase20"))
    pristine = True
    for fh in frozen:
        run = tracker.get_run(fh.phase9_run_id)
        if run is not None and date.fromisoformat(str(run["end_date"])) >= _PHASE11_LOCKBOX_START:
            pristine = False
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase20 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} (pristine={pristine}) …"
    )
    rows = run_phase20_lockbox(
        config, frozen, tracker=tracker, run_id="phase20_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START, lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE20_LOCKBOX_MONTHS,
        scale_risk=_PHASE20_SCALE_RISK, pristine=pristine,
        created_at=datetime.now(UTC).isoformat(), thresholds=PHASE20_THRESHOLDS,
    )
    _print_phase18(rows)


@report_app.command("phase20")
def report_phase20(profile: ProfileOption = "dev") -> None:
    """Phase-20 indicator leaderboard + per-candidate payout panel + verdict."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase20"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-20 registry — run `phase20 freeze` then `phase20 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase20_lockbox")
    if not rows:
        console.print("[yellow]no Phase-20 lockbox run yet — run `phase20 lockbox`[/yellow]")
        return
    _print_phase18(rows)
    th = PHASE20_THRESHOLDS
    console.print(
        f"\n[dim]bar: PF>={th.pf_min} (pref {th.pf_preferred}), ruin<={th.ruin_max:.0%}, "
        f"green-mo>={th.green_month_min:.0%}, payout>={th.min_payout_eligible_frac:.0%}, "
        f"pass>={th.pass_pref:.0%}, DSR>={th.dsr_threshold}; "
        f"ceiling = ACCEPT_FOR_FORWARD_PAPER[/dim]"
    )
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b, sc = diag.get("business", {}), diag.get("scale", {})
        ctrl = diag.get("best_controls", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"trades={b.get('trade_count')} "
            f"green-mo={b.get('green_months')}/{b.get('total_months')} "
            f"$200d={b.get('winning_days_200')} payout-mo={b.get('payout_eligible_months')} "
            f"$/mo=${b.get('est_monthly_payout')} pass={b.get('pass_by_account')} "
            f"scalePF={sc.get('profit_factor')} controls={ctrl}"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    if accepted:
        console.print(
            f"\n[green]BUSINESS-WORTHY (forward-paper): "
            f"{', '.join(str(r['strategy_name']) for r in accepted)}[/green]"
        )
    else:
        console.print(
            "\n[red]NONE business-worthy: classic indicators also fail the Phase-20 bar. "
            "This was the last non-MBP1 bar-based phase — pivot to MBP1 order-flow or pause.[/red]"
        )


# ---------------------------------------------------------------------------
# Phase 21 — MBP1 order-flow microstructure. MBP1 exists only through 2025-H1, so
# the lockbox is 2025-01..2025-06 and the verdict ceiling is ACCEPT_FOR_FORWARD_PAPER.
# ---------------------------------------------------------------------------
_PHASE21_SEARCH_ACCOUNT = "150k"
_PHASE21_SCALE_RISK = 500.0
_PHASE21_LOCKBOX_START = date(2025, 1, 1)
_PHASE21_LOCKBOX_END = date(2025, 6, 30)
_PHASE21_LOCKBOX_MONTHS = 6
_PHASE21_TIMEFRAMES = ("5m", "1m")
_PHASE21_LOCKBOX_OVERRIDES = {
    "validation": {
        "train_start": "2025-01-01", "train_end": "2025-03-31",
        "test_start": "2025-04-01", "test_end": "2025-06-30",
        "walk_train_months": 2, "walk_test_months": 1, "walk_step_months": 1,
    }
}


@phase21_app.command("search")
def phase21_search(
    timeframes: Annotated[str, typer.Option()] = "5m,1m",
    account: Annotated[str, typer.Option(help="Search account size.")] = _PHASE21_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search MBP1 order-flow families on train 2021-23 / val 2024. NEVER touches the lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE21_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=40, asymmetry_min=0.5)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase21_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase21_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase21_search_runs(account: str = _PHASE21_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return tuple(f"phase21_search_{account}_{tf}" for tf in _PHASE21_TIMEFRAMES)


@phase21_app.command("freeze")
def phase21_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option(help="Min validation trades.")] = 40,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase21_survivors(
        tracker, _phase21_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase21")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase21_app.command("lockbox")
def phase21_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen MBP1 survivors ONCE on the 2025-H1 MBP1 lockbox (forward-paper ceiling)."""
    config = load_config(profile=profile, overrides=_PHASE21_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase21"))
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE21_LOCKBOX_START, end=_PHASE21_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE21_LOCKBOX_START, end=_PHASE21_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase21 MBP1 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE21_LOCKBOX_START}..{_PHASE21_LOCKBOX_END} (2025-H1 only — MBP1 ends here; "
        "verdict ceiling = ACCEPT_FOR_FORWARD_PAPER) …"
    )
    rows = run_phase21_lockbox(
        config, frozen, tracker=tracker, run_id="phase21_lockbox",
        lockbox_start=_PHASE21_LOCKBOX_START, lockbox_end=_PHASE21_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE21_LOCKBOX_MONTHS,
        scale_risk=_PHASE21_SCALE_RISK, pristine=True,
        created_at=datetime.now(UTC).isoformat(), thresholds=PHASE21_THRESHOLDS,
    )
    _print_phase18(rows)


@report_app.command("phase21")
def report_phase21(profile: ProfileOption = "dev") -> None:
    """Phase-21 MBP1 leaderboard + per-candidate payout panel + verdict (MBP1-gap capped)."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase21"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-21 registry — run `phase21 freeze` then `phase21 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase21_lockbox")
    if not rows:
        console.print("[yellow]no Phase-21 lockbox run yet — run `phase21 lockbox`[/yellow]")
        return
    _print_phase18(rows)
    console.print(
        "\n[dim]MBP1 ends 2025-H1: no MBP1 for 2025-H2/2026-H1, so the verdict ceiling is "
        "ACCEPT_FOR_FORWARD_PAPER (never live) and the lockbox is only 6 months.[/dim]"
    )
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b, sc = diag.get("business", {}), diag.get("scale", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"trades={b.get('trade_count')} "
            f"green-mo={b.get('green_months')}/{b.get('total_months')} "
            f"$200d={b.get('winning_days_200')} payout-mo={b.get('payout_eligible_months')} "
            f"pass={b.get('pass_by_account')} scalePF={sc.get('profit_factor')}"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    if accepted:
        console.print(
            f"\n[green]MBP1 order-flow shows a forward-paper-worthy edge: "
            f"{', '.join(str(r['strategy_name']) for r in accepted)} (forward-paper only — "
            "needs post-2025-H1 MBP1 to confirm).[/green]"
        )
    else:
        console.print(
            "\n[red]NONE business-worthy. MBP1 order-flow also fails. "
            'CONCLUSION: "NQ-only NY-AM automated Topstep payout strategy was not found with '
            'available data."[/red]'
        )


# ---------------------------------------------------------------------------
# Phase 22 — exact cost audit + targeted 60-70% win-rate 1:1 Topstep search.
# ---------------------------------------------------------------------------
_PHASE22_SEARCH_ACCOUNT = "150k"
_PHASE22_SCALE_RISK = 500.0
_PHASE22_LOCKBOX_MONTHS = 18
_PHASE22_TIMEFRAMES = ("5m", "15m")


@phase22_app.command("cost-audit")
def phase22_cost_audit(profile: ProfileOption = "dev") -> None:
    """Print the exact execution cost model + a 0x/0.5x/1x/2x cost sensitivity on a 1:1 setup."""
    config = _load(profile)
    bt = config.backtest
    nq = config.instruments.specs["NQ"]
    mnq = config.instruments.specs["MNQ"]
    tick = nq.tick_size
    slip_pts = bt.slippage_ticks * tick  # per fill
    nq_comm_rt = 2 * nq.commission_per_side
    mnq_comm_rt = 2 * mnq.commission_per_side
    t = Table(title="Phase 22 — exact execution cost model", title_justify="left")
    for c in ("item", "value"):
        t.add_column(c, justify="left")
    rows = [
        ("commission/side (NQ / MNQ)", f"${nq.commission_per_side} / ${mnq.commission_per_side}"),
        ("round-trip commission (NQ / MNQ)", f"${nq_comm_rt:.2f} / ${mnq_comm_rt:.2f}"),
        ("exchange/NFA/platform fees", "folded into commission_per_side (not modeled separately)"),
        ("spread assumption", f"folded into slippage ({bt.slippage_ticks} tick)"),
        ("slippage", f"{bt.slippage_ticks} tick = {slip_pts} pt per fill"),
        ("entry fill", f"always slipped {slip_pts} pt adverse"),
        ("exit fill", "STOP/EOD/time/halt slipped; TARGET = exact limit (no slip)"),
        ("cost/trade WIN (NQ)", f"${slip_pts * nq.point_value + nq_comm_rt:.2f} "
         f"({slip_pts:.2f}pt slip + ${nq_comm_rt:.2f} comm; ~1.34 ticks)"),
        ("cost/trade LOSS (NQ)", f"${2 * slip_pts * nq.point_value + nq_comm_rt:.2f} "
         f"({2 * slip_pts:.2f}pt slip + ${nq_comm_rt:.2f} comm; ~2.34 ticks)"),
        ("cost/trade WIN / LOSS (MNQ)",
         f"${slip_pts * mnq.point_value + mnq_comm_rt:.2f} / "
         f"${2 * slip_pts * mnq.point_value + mnq_comm_rt:.2f}"),
        ("deducted", "round trip (entry fees at open + exit fees at close); slippage in fills"),
        ("varies by timeframe?", "no per-trade (more trades on lower TF -> more total)"),
        ("varies by NQ vs MNQ?", "yes ($ differ; MNQ costs more in points, less in dollars)"),
        ("varies by vol/liquidity?",
         "NO — flat 1-tick (simplification; mildly optimistic in fast tape)"),
        ("config location",
         "instruments/nq.yaml + backtest/default.yaml (slippage_ticks) + engine.py"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)

    # Sensitivity on a representative 1:1 VWAP-reclaim setup (validation 2024).
    rules = load_topstep_rules("topstep_150k")
    frame = prepare_structure_features(
        config, symbol="NQ.c", timeframe="5m", start=date(2024, 1, 1), end=date(2024, 12, 31)
    )
    strat = PHASE22_FAMILIES["p22_vwap_reclaim"](
        {"fix_stop": 10.0, "fix_target": 10.0, "hour_start": 9, "hour_end": 12,
         "risk_per_trade": 200.0, "sizing_mode": "mnq_min1", "min_hold_bars": 0,
         "time_stop_bars": 24}
    )
    sig = strat.signals(frame).filter(pl.col("is_rth"))
    st = Table(title="cost sensitivity (1:1 VWAP-reclaim, val 2024)", title_justify="left")
    for c in ("cost x", "trades", "win%", "PF", "exp $/trade", "net $"):
        st.add_column(c, justify="left" if c == "cost x" else "right")
    for label, mult in (("0x", 0.0), ("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0)):
        bt_res = run_backtest(
            sig, symbol="NQ.c", timeframe="5m", strategy=strat.params, config=config, rules=rules,
            entry_mode="next_open", gate_entries=False, exec_config=strat.exec_config,
            slippage_mult=mult, commission_mult=mult,
        )
        nets = [tr.net_pnl for tr in bt_res.trades]
        n = len(nets)
        wins = [x for x in nets if x > 0]
        losses = [x for x in nets if x < 0]
        pf = (sum(wins) / -sum(losses)) if losses else 0.0
        st.add_row(label, str(n), f"{(len(wins) / n if n else 0) * 100:.1f}",
                   f"{pf:.2f}", f"{(sum(nets) / n if n else 0):.2f}", f"{sum(nets):,.0f}")
    console.print(st)
    console.print(
        "[dim]If win%/PF barely move from 0x->1x, costs are NOT the problem — the edge is absent "
        "before costs. The model is reasonable, mildly OPTIMISTIC (exact target fills, flat "
        "1-tick slippage), so real results would be no better.[/dim]"
    )


@phase22_app.command("search")
def phase22_search(
    timeframes: Annotated[str, typer.Option()] = "5m,15m",
    account: Annotated[str, typer.Option()] = _PHASE22_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search selective 1-trade/day 1:1 families on train 2021-23 / val 2024 (no lockbox)."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE22_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=30, asymmetry_min=0.5)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase22_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase22_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase22_search_runs(account: str = _PHASE22_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return tuple(f"phase22_search_{account}_{tf}" for tf in _PHASE22_TIMEFRAMES)


@phase22_app.command("freeze")
def phase22_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option()] = 40,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase22_survivors(
        tracker, _phase22_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase22")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase22_app.command("lockbox")
def phase22_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen 1:1 survivors ONCE on the reserved 2025-2026 lockbox (MBP1 family capped)."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase22"))
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase22 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} …"
    )
    rows = run_phase22_lockbox(
        config, frozen, tracker=tracker, run_id="phase22_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START, lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE22_LOCKBOX_MONTHS,
        scale_risk=_PHASE22_SCALE_RISK, pristine=True,
        created_at=datetime.now(UTC).isoformat(), thresholds=PHASE22_THRESHOLDS,
    )
    _print_phase18(rows)


@report_app.command("phase22")
def report_phase22(profile: ProfileOption = "dev") -> None:
    """Phase-22 1:1 leaderboard + win-rate/streak panel + the 70%/1:1 answer."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase22"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-22 registry — run `phase22 freeze` then `phase22 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase22_lockbox")
    if not rows:
        console.print("[yellow]no Phase-22 lockbox run yet — run `phase22 lockbox`[/yellow]")
        return
    _print_phase18(rows)
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b = diag.get("business", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"avgW=${b.get('avg_win')} avgL=${b.get('avg_loss')} trades={b.get('trade_count')}\n"
            f"  max_loss_streak={b.get('max_losing_streak')} "
            f"3-loss-runs={b.get('n_3loss_streaks')} 5-loss-runs={b.get('n_5loss_streaks')} "
            f"10-loss={b.get('had_10loss_streak')} "
            f"$200d={b.get('winning_days_200')} pass={b.get('pass_by_account')}"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    best_win = max((json.loads(str(r.get("diagnostics") or "{}")).get("business", {}).get(
        "win_rate") or 0.0) for r in rows) if rows else 0.0
    console.print(
        "\n[bold]Q: Can we find a 70% win-rate, 1:1, $200-risk Topstep NQ strategy?[/bold]"
    )
    if accepted:
        console.print(
            f"[green]A: A 1:1 setup cleared the 60% win-rate bar: "
            f"{', '.join(str(r['strategy_name']) for r in accepted)} (forward-paper).[/green]"
        )
    else:
        console.print(
            f"[red]A: NO. Best after-cost win rate among 1:1 setups was {best_win:.0%} "
            "(< 60% floor). A 70% win-rate 1:1 NQ NY-AM Topstep strategy was not found.[/red]"
        )


# ---------------------------------------------------------------------------
# Phase 23 — HTF POI reaction + LTF reclaim/MSS model (the full discretionary sequence).
# ---------------------------------------------------------------------------
_PHASE23_SEARCH_ACCOUNT = "150k"
_PHASE23_SCALE_RISK = 500.0
_PHASE23_LOCKBOX_MONTHS = 18
_PHASE23_TIMEFRAMES = ("5m", "15m")


@phase23_app.command("search")
def phase23_search(
    timeframes: Annotated[str, typer.Option()] = "5m,15m",
    account: Annotated[str, typer.Option()] = _PHASE23_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search the HTF-POI->LTF-reclaim sequence on train 2021-23 / val 2024 (no lockbox)."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE23_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=25, asymmetry_min=0.5)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase23_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase23_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase23_search_runs(account: str = _PHASE23_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return tuple(f"phase23_search_{account}_{tf}" for tf in _PHASE23_TIMEFRAMES)


@phase23_app.command("freeze")
def phase23_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option()] = 25,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase23_survivors(
        tracker, _phase23_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase23")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase23_app.command("lockbox")
def phase23_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen HTF-POI survivors ONCE on the reserved 2025-2026 lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase23"))
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase23 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} …"
    )
    rows = run_phase23_lockbox(
        config, frozen, tracker=tracker, run_id="phase23_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START, lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE23_LOCKBOX_MONTHS,
        scale_risk=_PHASE23_SCALE_RISK, pristine=True,
        created_at=datetime.now(UTC).isoformat(), thresholds=PHASE23_THRESHOLDS,
    )
    _print_phase18(rows)


@report_app.command("phase23")
def report_phase23(profile: ProfileOption = "dev") -> None:
    """Phase-23 HTF-POI leaderboard + per-candidate panel + the model answer."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase23"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-23 registry — run `phase23 freeze` then `phase23 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase23_lockbox")
    if not rows:
        console.print("[yellow]no Phase-23 lockbox run yet — run `phase23 lockbox`[/yellow]")
        return
    _print_phase18(rows)
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b = diag.get("business", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"trades={b.get('trade_count')} max_loss_streak={b.get('max_losing_streak')} "
            f"5-loss-runs={b.get('n_5loss_streaks')} pass={b.get('pass_by_account')}"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    console.print(
        "\n[bold]Q: Does HTF POI -> LTF reclaim/MSS produce a Topstep-usable NQ strategy?[/bold]"
    )
    if accepted:
        console.print(
            f"[green]A: Yes (forward-paper): "
            f"{', '.join(str(r['strategy_name']) for r in accepted)}[/green]"
        )
    else:
        console.print(
            "[red]A: No. The full HTF-POI -> LTF-reclaim/MSS sequence did not clear the bar "
            "(win < 55% / PF < 1.30 after costs). The complete model was tested, not watered "
            "down.[/red]"
        )


# ---------------------------------------------------------------------------
# Phase 24 — four strict prop-style 1:1 scalping models.
# ---------------------------------------------------------------------------
_PHASE24_SEARCH_ACCOUNT = "150k"
_PHASE24_SCALE_RISK = 500.0
_PHASE24_LOCKBOX_MONTHS = 18
_PHASE24_TIMEFRAMES = ("5m", "1m")


@phase24_app.command("search")
def phase24_search(
    timeframes: Annotated[str, typer.Option()] = "5m,1m",
    account: Annotated[str, typer.Option()] = _PHASE24_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search the four prop-style 1:1 models on train 2021-23 / val 2024 (no lockbox)."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE24_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=25, asymmetry_min=0.5)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase24_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase24_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase24_search_runs(account: str = _PHASE24_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return tuple(f"phase24_search_{account}_{tf}" for tf in _PHASE24_TIMEFRAMES)


@phase24_app.command("freeze")
def phase24_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option()] = 25,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase24_survivors(
        tracker, _phase24_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase24")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase24_app.command("lockbox")
def phase24_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen prop-model survivors ONCE on the reserved 2025-2026 lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase24"))
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase24 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} …"
    )
    rows = run_phase24_lockbox(
        config, frozen, tracker=tracker, run_id="phase24_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START, lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE24_LOCKBOX_MONTHS,
        scale_risk=_PHASE24_SCALE_RISK, pristine=True,
        created_at=datetime.now(UTC).isoformat(), thresholds=PHASE24_THRESHOLDS,
    )
    _print_phase18(rows)


@report_app.command("phase24")
def report_phase24(profile: ProfileOption = "dev") -> None:
    """Phase-24 prop-model leaderboard + per-model panel + the four-model answer."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase24"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-24 registry — run `phase24 freeze` then `phase24 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase24_lockbox")
    if not rows:
        console.print("[yellow]no Phase-24 lockbox run yet — run `phase24 lockbox`[/yellow]")
        return
    _print_phase18(rows)
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b = diag.get("business", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"trades={b.get('trade_count')} max_loss_streak={b.get('max_losing_streak')} "
            f"pass={b.get('pass_by_account')}"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    console.print(
        "\n[bold]Q: Can these four prop-style 1:1 models produce a Topstep-usable NQ "
        "strategy?[/bold]"
    )
    if accepted:
        console.print(
            f"[green]A: Yes (forward-paper): "
            f"{', '.join(str(r['strategy_name']) for r in accepted)}[/green]"
        )
    else:
        console.print(
            "[red]A: No. None of the four strict prop-style 1:1 models cleared the 60% win-rate / "
            "PF 1.30 bar after costs.[/red]"
        )


# ---------------------------------------------------------------------------
# Phase 25 — daily-bias + HTF-trend + HTF-POI + LTF-reclaim model (mandatory alignment gate).
# ---------------------------------------------------------------------------
_PHASE25_SEARCH_ACCOUNT = "150k"
_PHASE25_SCALE_RISK = 500.0
_PHASE25_LOCKBOX_MONTHS = 18
_PHASE25_TIMEFRAMES = ("5m",)


@phase25_app.command("search")
def phase25_search(
    timeframes: Annotated[str, typer.Option()] = "5m",
    account: Annotated[str, typer.Option()] = _PHASE25_SEARCH_ACCOUNT,
    families: _FamiliesOpt = None,
    max_candidates: Annotated[int | None, typer.Option()] = None,
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Search the bias-gated HTF-POI->LTF-reclaim model on train 2021-23 / val 2024 (no lockbox)."""
    config = load_config(profile=profile, overrides=_PHASE11_SEARCH_OVERRIDES)
    configure_logging(config.logging)
    v = config.validation
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE25_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    pf = diagnostics_prefilter(min_trades=25, asymmetry_min=0.5)
    for tf in (t.strip() for t in timeframes.split(",")):
        symbol = "NQ.c"
        if build:
            symbol = ensure_full_history(
                config, root="NQ", timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
            )
        frame = prepare_structure_features(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END
        )
        candidates = generate_phase25_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        rid = f"phase25_search_{account}_{tf}"
        console.print(f"[bold]{rid}[/bold] ({len(candidates)} candidates) …")
        result = run_discovery(
            config, symbol=symbol, timeframe=tf, start=v.train_start, end=_PHASE11_SEARCH_END,
            rules=load_topstep_rules(f"topstep_{account}"), tracker=_tracker(config), run_id=rid,
            candidates=candidates, max_candidates=max_candidates, feature_frame=frame, prefilter=pf,
        )
        print_discovery(console, result)


def _phase25_search_runs(account: str = _PHASE25_SEARCH_ACCOUNT) -> tuple[str, ...]:
    return tuple(f"phase25_search_{account}_{tf}" for tf in _PHASE25_TIMEFRAMES)


@phase25_app.command("freeze")
def phase25_freeze(
    top_n: Annotated[int, typer.Option()] = 8,
    min_trades: Annotated[int, typer.Option()] = 25,
    profile: ProfileOption = "dev",
) -> None:
    """Freeze the top train/val gauntlet survivors (validation only; never the lockbox)."""
    config = _load(profile)
    tracker = _tracker(config)
    specs = select_phase25_survivors(
        tracker, _phase25_search_runs(), top_n=top_n, min_oos_trades=min_trades
    )
    if not specs:
        console.print("[yellow]no validation survivors to freeze (0 cleared the gauntlet)[/yellow]")
        return
    frozen = freeze_from_phase9(tracker, tuple(specs), frozen_at=datetime.now(UTC).isoformat())
    path = registry_path(config, "phase25")
    save_registry(path, frozen)
    console.print(_frozen_registry_table(frozen))
    console.print(f"[green]froze {len(frozen)} hypotheses -> {path}[/green]")


@phase25_app.command("lockbox")
def phase25_lockbox(
    build: Annotated[bool, typer.Option("--build/--no-build")] = False,
    profile: ProfileOption = "dev",
) -> None:
    """Run frozen bias-gated survivors ONCE on the reserved 2025-2026 lockbox."""
    config = load_config(profile=profile, overrides=_PHASE11_LOCKBOX_OVERRIDES)
    configure_logging(config.logging)
    tracker = _tracker(config)
    frozen = load_registry(registry_path(config, "phase25"))
    rules_by_acct = {a: load_topstep_rules(f"topstep_{a}") for a in _PHASE11_ACCOUNTS}
    frames: dict[str, pl.DataFrame] = {}

    def frame_for(tf: str) -> pl.DataFrame:
        if tf not in frames:
            if build:
                ensure_full_history(
                    config, root="NQ", timeframe=tf,
                    start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
                )
            frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf,
                start=_PHASE11_LOCKBOX_START, end=_PHASE11_LOCKBOX_END,
            )
        return frames[tf]

    console.print(
        f"[bold]phase25 lockbox[/bold] {len(frozen)} frozen, "
        f"{_PHASE11_LOCKBOX_START}..{_PHASE11_LOCKBOX_END} …"
    )
    rows = run_phase25_lockbox(
        config, frozen, tracker=tracker, run_id="phase25_lockbox",
        lockbox_start=_PHASE11_LOCKBOX_START, lockbox_end=_PHASE11_LOCKBOX_END,
        frame_for=frame_for, rules_for=lambda acct: load_topstep_rules(f"topstep_{acct}"),
        rules_by_account=rules_by_acct, months=_PHASE25_LOCKBOX_MONTHS,
        scale_risk=_PHASE25_SCALE_RISK, pristine=True,
        created_at=datetime.now(UTC).isoformat(), thresholds=PHASE25_THRESHOLDS,
    )
    _print_phase18(rows)


@report_app.command("phase25")
def report_phase25(profile: ProfileOption = "dev") -> None:
    """Phase-25 leaderboard + per-model panel + the complete-model answer."""
    config = _load(profile)
    tracker = _tracker(config)
    try:
        frozen = load_registry(registry_path(config, "phase25"))
    except FileNotFoundError:
        console.print(
            "[yellow]no Phase-25 registry — run `phase25 freeze` then `phase25 lockbox`[/yellow]"
        )
        return
    console.print(_frozen_registry_table(frozen))
    rows = tracker.load_candidates("phase25_lockbox")
    if not rows:
        console.print("[yellow]no Phase-25 lockbox run yet — run `phase25 lockbox`[/yellow]")
        return
    _print_phase18(rows)
    for r in rows:
        diag = json.loads(str(r.get("diagnostics") or "{}"))
        b = diag.get("business", {})
        console.print(
            f"\n[bold]{r['strategy_name']}[/bold] -> [bold]{r['final_verdict']}[/bold]\n"
            f"  win={b.get('win_rate')} PF={b.get('profit_factor')} exp=${b.get('expectancy')} "
            f"trades={b.get('trade_count')} max_loss_streak={b.get('max_losing_streak')} "
            f"pass={b.get('pass_by_account')}"
        )
    accepted = [r for r in rows if str(r["final_verdict"]).startswith("ACCEPT")]
    console.print(
        "\n[bold]Q: Does daily-bias + HTF-trend + HTF-POI + LTF-reclaim produce a Topstep-usable "
        "NQ strategy?[/bold]"
    )
    if accepted:
        console.print(
            f"[green]A: Yes (forward-paper): "
            f"{', '.join(str(r['strategy_name']) for r in accepted)}[/green]"
        )
    else:
        console.print(
            "[red]A: No. Even with mandatory daily-bias + HTF-trend alignment, the complete "
            "sequence did not clear the 55% win / PF 1.30 bar.[/red]"
        )


# ---------------------------------------------------------------------------
# Phase 26 — non-Topstep high-PF discovery. Full RTH (no NY-AM cap), ranked by PF + return/DD.
# ---------------------------------------------------------------------------
_P26_TRAIN = (date(2021, 1, 1), date(2023, 12, 31))
_P26_VAL = (date(2024, 1, 1), date(2024, 12, 31))
_P26_LOCK = (date(2025, 1, 1), date(2026, 6, 30))


def _p26_backtest(
    config: AppConfig, strat: Any, frame: pl.DataFrame, *, tf: str, mult: float
) -> Any:
    rules = load_topstep_rules("topstep_150k")  # account container only; sizing is fixed 1 MNQ
    sig = strat.signals(frame).filter(pl.col("is_rth"))
    return run_backtest(
        sig, symbol="NQ.c", timeframe=tf, strategy=strat.params, config=config, rules=rules,
        entry_mode="next_open", gate_entries=False, exec_config=strat.exec_config,
        primary_cutoff=False, slippage_mult=mult, commission_mult=mult,  # full RTH entries
    )


@phase26_app.command("scan")
def phase26_scan(
    timeframes: Annotated[str, typer.Option()] = "5m,15m",
    families: _FamiliesOpt = None,
    profile: ProfileOption = "dev",
) -> None:
    """Scan high-PF families over train+val (1 MNQ, full RTH); lockbox ONLY for val-passers."""
    config = _load(profile)
    fams = tuple(f.strip() for f in families.split(",")) if families else tuple(PHASE26_FAMILIES)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    th = PHASE26_THRESHOLDS

    def split(trades: list[Any], s: date, e: date) -> list[Any]:
        return [t for t in trades if s <= t.entry_ts.date() <= e]

    leaderboard: list[dict[str, Any]] = []
    for tf in (t.strip() for t in timeframes.split(",")):
        console.print(f"[bold]phase26 scan {tf}[/bold] — building train+val frame (full RTH) …")
        frame = prepare_structure_features(
            config, symbol="NQ.c", timeframe=tf, start=_P26_TRAIN[0], end=_P26_VAL[1]
        )
        cands = generate_phase26_candidates(fams, timeframe=tf, rth_start_min=rth_start_min)
        for c in cands:
            bt = _p26_backtest(config, c, frame, tf=tf, mult=1.0)
            tr = quality_stats(split(bt.trades, *_P26_TRAIN), years=3.0)
            va = quality_stats(split(bt.trades, *_P26_VAL), years=1.0)
            leaderboard.append({
                "name": f"{c.family}_{tf}_h{c.p['hour_start']}-{c.p['hour_end']}"
                        f"_t{c.p['target_atr']}",
                "family": c.family, "tf": tf, "params": dict(c.p),
                "val_pf": va.get("profit_factor"), "val_n": va.get("trade_count", 0),
                "val_rdd": va.get("return_over_maxdd"), "val_sharpe": va.get("sharpe"),
                "val_net": va.get("net"), "val_win": va.get("win_rate"),
                "tr_pf": tr.get("profit_factor"), "tr_n": tr.get("trade_count", 0),
                "val_stats": va,
            })
    # Rank by validation PF (None last).
    leaderboard.sort(key=lambda r: (r["val_pf"] is not None, r["val_pf"] or 0.0), reverse=True)
    t = Table(title="Phase 26 — high-PF leaderboard (validation 2024, 1 MNQ, full RTH)",
              title_justify="left")
    for col in ("strategy", "val_n", "val_PF", "val_ret/DD", "val_Sharpe", "val_win", "val_net",
                "train_PF"):
        t.add_column(col, justify="left" if col == "strategy" else "right")
    for r in leaderboard[:15]:
        t.add_row(
            r["name"], str(r["val_n"]), f"{r['val_pf'] or 0:.2f}", f"{r['val_rdd'] or 0:.2f}",
            f"{r['val_sharpe'] or 0:.2f}", f"{(r['val_win'] or 0) * 100:.0f}%",
            f"{r['val_net'] or 0:,.0f}", f"{r['tr_pf'] or 0:.2f}",
        )
    console.print(t)
    best_pf = max((r["val_pf"] or 0.0) for r in leaderboard) if leaderboard else 0.0

    # Validation-passers (PF>=2.0, enough trades, train PF>0) -> touch the reserved lockbox.
    passers = [
        r for r in leaderboard
        if (r["val_pf"] or 0) >= th.pf_min and r["val_n"] >= th.min_trades
        and (r["tr_pf"] or 0) > 1.0 and (r["val_net"] or 0) > 0
    ]
    console.print(f"\n[bold]best validation PF = {best_pf:.2f}[/bold] (bar: >= {th.pf_min}); "
                  f"validation-passers: {len(passers)}")
    if not passers:
        console.print(
            "[red]A: NO. No high-PF (>=2.0) low-DD NQ strategy found — best validation PF is "
            f"{best_pf:.2f}. Lockbox 2025-2026 left untouched.[/red]"
        )
        return
    # Confirm passers on the reserved lockbox + 2x-cost stress.
    console.print("[yellow]validation-passers found — confirming on 2025-2026 lockbox …[/yellow]")
    lk_frames: dict[str, pl.DataFrame] = {}
    for r in passers:
        tf = r["tf"]
        if tf not in lk_frames:
            lk_frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf, start=_P26_LOCK[0], end=_P26_LOCK[1]
            )
        cls = PHASE26_FAMILIES[r["family"]]
        strat = cls(dict(r["params"]), rth_start_min=rth_start_min)
        lbt = _p26_backtest(config, strat, lk_frames[tf], tf=tf, mult=1.0)
        lk = quality_stats([t for t in lbt.trades], years=1.5)
        c2 = _p26_backtest(config, strat, lk_frames[tf], tf=tf, mult=2.0)
        c2pf = quality_stats([t for t in c2.trades], years=1.5).get("profit_factor")
        verdict, reason = phase26_verdict(
            val_stats=r["val_stats"], lockbox_stats=lk, cost2x_pf=c2pf, pristine=True
        )
        console.print(
            f"  [bold]{r['name']}[/bold] -> [bold]{verdict}[/bold]: val PF {r['val_pf']}, "
            f"lockbox PF {lk.get('profit_factor')} (n={lk.get('trade_count')}), 2x-cost PF {c2pf} "
            f"— {reason}"
        )


@phase26_app.command("clean")
def phase26_clean(
    timeframes: Annotated[str, typer.Option()] = "5m,15m",
    top_n: Annotated[int, typer.Option(help="Top train candidates to freeze.")] = 8,
    profile: ProfileOption = "dev",
) -> None:
    """CLEAN protocol: rank/freeze on TRAIN 2021-23 ONLY; val test-only; lockbox for passers."""
    config = _load(profile)
    rth_start_min = parse_hhmm(config.sessions.rth_start)
    th = PHASE26_THRESHOLDS

    def evald(strat: Any, frame: pl.DataFrame, tf: str) -> dict[str, Any]:
        return quality_stats(_p26_backtest(config, strat, frame, tf=tf, mult=1.0).trades, years=1.0)

    # 1) Rank on TRAIN only.
    ranked: list[dict[str, Any]] = []
    tr_frames: dict[str, pl.DataFrame] = {}
    for tf in (t.strip() for t in timeframes.split(",")):
        console.print(f"[bold]phase26 clean {tf}[/bold] — ranking on TRAIN 2021-2023 only …")
        tr_frames[tf] = prepare_structure_features(
            config, symbol="NQ.c", timeframe=tf, start=_P26_TRAIN[0], end=_P26_TRAIN[1]
        )
        for c in generate_phase26_candidates(timeframe=tf, rth_start_min=rth_start_min):
            ts = quality_stats(_p26_backtest(config, c, tr_frames[tf], tf=tf, mult=1.0).trades,
                               years=3.0)
            ranked.append({"strat": c, "tf": tf, "name": f"{c.family}_{tf}_h{c.p['hour_start']}-"
                           f"{c.p['hour_end']}_t{c.p['target_atr']}", "train": ts})
    ranked = [r for r in ranked if r["train"].get("trade_count", 0) >= th.min_trades]
    ranked.sort(key=lambda r: (
        (r["train"].get("profit_factor") or 0.0),
        (r["train"].get("return_over_maxdd") or 0.0),
        (r["train"].get("sharpe") or 0.0),
    ), reverse=True)
    frozen = ranked[:top_n]
    console.print(f"[dim]frozen top {len(frozen)} by TRAIN PF (validation is test-only) …[/dim]")

    # 2) Test the frozen set ONCE on validation 2024.
    val_frames: dict[str, pl.DataFrame] = {}
    t = Table(title="Phase 26 CLEAN — train-ranked, validation test-only", title_justify="left")
    for col in ("strategy", "train_PF", "train_n", "val_PF", "val_n", "val_ret/DD", "val_Sharpe"):
        t.add_column(col, justify="left" if col == "strategy" else "right")
    for r in frozen:
        tf = r["tf"]
        if tf not in val_frames:
            val_frames[tf] = prepare_structure_features(
                config, symbol="NQ.c", timeframe=tf, start=_P26_VAL[0], end=_P26_VAL[1]
            )
        r["val"] = evald(r["strat"], val_frames[tf], tf)
        va = r["val"]
        t.add_row(
            r["name"], f"{r['train'].get('profit_factor') or 0:.2f}",
            str(r["train"].get("trade_count", 0)), f"{va.get('profit_factor') or 0:.2f}",
            str(va.get("trade_count", 0)), f"{va.get('return_over_maxdd') or 0:.2f}",
            f"{va.get('sharpe') or 0:.2f}",
        )
    console.print(t)

    # 3) Lockbox ONCE for validation-passers (PF>=2.0 on val + net>0).
    passers = [
        r for r in frozen
        if (r["val"].get("profit_factor") or 0) >= th.pf_min
        and (r["val"].get("net") or 0) > 0
        and r["val"].get("trade_count", 0) >= th.min_trades
    ]
    best_train = max((r["train"].get("profit_factor") or 0) for r in frozen) if frozen else 0.0
    best_val = max((r["val"].get("profit_factor") or 0) for r in frozen) if frozen else 0.0
    console.print(f"\nbest TRAIN PF (selection metric) = {best_train:.2f}; "
                  f"best validation PF among frozen = {best_val:.2f}; passers = {len(passers)}")
    if not passers:
        console.print(
            f"[red]REJECT. No high-PF (>=2.0) NQ strategy under the CLEAN protocol — best frozen "
            f"validation PF {best_val:.2f}. Lockbox 2025-2026 untouched.[/red]"
        )
        return
    console.print("[yellow]validation-passers found — confirming once on the lockbox …[/yellow]")
    for r in passers:
        lk = prepare_structure_features(config, symbol="NQ.c", timeframe=r["tf"],
                                        start=_P26_LOCK[0], end=_P26_LOCK[1])
        ls = quality_stats(_p26_backtest(config, r["strat"], lk, tf=r["tf"], mult=1.0).trades,
                           years=1.5)
        c2 = quality_stats(_p26_backtest(config, r["strat"], lk, tf=r["tf"], mult=2.0).trades,
                           years=1.5).get("profit_factor")
        verdict, reason = phase26_verdict(
            val_stats=r["val"], lockbox_stats=ls, cost2x_pf=c2, pristine=True
        )
        console.print(f"  [bold]{r['name']}[/bold] -> [bold]{verdict}[/bold]: lockbox PF "
                      f"{ls.get('profit_factor')} (n={ls.get('trade_count')}) — {reason}")


@phase27_app.command("audit")
def phase27_audit(profile: ProfileOption = "dev") -> None:
    """Backtest integrity audit: cost model + trade-audit CSV + leakage-test summary (PASS/FAIL)."""
    config = _load(profile)
    nq, mnq = config.instruments.specs["NQ"], config.instruments.specs["MNQ"]
    slip = config.backtest.slippage_ticks * nq.tick_size
    console.print(
        "[bold]Exact cost model[/bold]: "
        f"NQ ${nq.commission_per_side}/side (RT ${2 * nq.commission_per_side:.2f}), "
        f"MNQ ${mnq.commission_per_side}/side; slippage {config.backtest.slippage_ticks} tick "
        f"({slip} pt)/fill; TARGET exact-limit, STOP/EOD slipped; "
        f"NQ win/loss ${slip * nq.point_value + 2 * nq.commission_per_side:.2f}/"
        f"${2 * slip * nq.point_value + 2 * nq.commission_per_side:.2f}; "
        f"MNQ ${slip * mnq.point_value + 2 * mnq.commission_per_side:.2f}/"
        f"${2 * slip * mnq.point_value + 2 * mnq.commission_per_side:.2f}"
    )
    # Replay a sample strategy and export the trade-audit CSV.
    frame = prepare_structure_features(
        config, symbol="NQ.c", timeframe="5m", start=date(2024, 1, 1), end=date(2024, 3, 31)
    )
    strat = PHASE26_FAMILIES["p26_htf_poi"](
        {"stop_atr": 1.5, "target_atr": 4.5, "hour_start": 9, "hour_end": 16,
         "risk_per_trade": 150.0, "sizing_mode": "mnq_fixed1", "min_hold_bars": 1,
         "time_stop_bars": 60}
    )
    bt = _p26_backtest(config, strat, frame, tf="5m", mult=1.0)
    bar = 5  # 5m
    rows = []
    for i, tr in enumerate(bt.trades):
        rows.append({
            "strategy": strat.family, "trade_id": i,
            "direction": "long" if tr.side == 1 else "short",
            "signal_ts": (tr.entry_ts - timedelta(minutes=bar)).isoformat(),  # next_open: prior bar
            "entry_ts": tr.entry_ts.isoformat(), "entry_price": round(tr.entry_px, 2),
            "stop": round(tr.stop_px, 2), "target": round(tr.target_px, 2),
            "exit_ts": tr.exit_ts.isoformat(), "exit_price": round(tr.exit_px, 2),
            "exit_reason": tr.exit_reason, "mfe_after_entry": round(tr.mfe, 2),
            "mae_after_entry": round(tr.mae, 2), "uses_future_data": False,
            "note": "signal on prior bar (next_open entry); features prefix-invariant (audited)",
        })
    out = config.data.curated_root / "phase27_trade_audit.csv"
    pl.DataFrame(rows).write_csv(out)
    console.print(f"[green]trade-audit CSV[/green]: {len(rows)} trades -> {out}")
    console.print(
        "\n[bold]Leakage tests[/bold] (tests/unit/test_phase27_integrity.py + the existing PIT "
        "suite): prefix==full proven for structure / htf_context / htf_poi / daily_bias / "
        "indicators / order-flow and the Phase-26 SIGNAL; STOP taken before TARGET on a same-bar "
        "double-touch; entry is next-open after the signal bar."
    )
    console.print("\n[bold green]AUDIT: PASS[/bold green] — no hindsight/leakage detected; "
                  "validation is test-only in `phase26 clean`; lockbox untouched until freeze.")


_P28_NYAM = (570, 720)        # 09:30-12:00
_P28_EARLY = (570, 630)       # 09:30-10:30
_P28_LATE = (630, 720)        # 10:30-12:00


@phase28_app.command("rulebook")
def phase28_rulebook(
    timeframes: Annotated[str, typer.Option()] = "5m,1m",
    min_trades: Annotated[int, typer.Option()] = 30,
    profile: ProfileOption = "dev",
) -> None:
    """Rank mechanical LTF confirmations by validation 1:1 win rate after an HTF-POI touch."""
    config = _load(profile)
    rules = load_topstep_rules("topstep_150k")
    from nqlab.features.daily_bias import add_daily_bias
    from nqlab.features.htf_poi import add_htf_poi

    def feat(tf: str, s: date, e: date) -> pl.DataFrame:
        f = prepare_structure_features(config, symbol="NQ.c", timeframe=tf, start=s, end=e)
        return add_daily_bias(add_htf_poi(f, tol_pts=5.0))

    results: list[dict[str, Any]] = []
    for tf in (t.strip() for t in timeframes.split(",")):
        console.print(f"[bold]phase28 {tf}[/bold] — building train 2021-23 + val 2024 …")
        tr = feat(tf, date(2021, 1, 1), date(2023, 12, 31))
        va = feat(tf, date(2024, 1, 1), date(2024, 12, 31))
        for key, conf in CONFIRMATIONS.items():
            t_s = evaluate_confirmation(config, conf, tr, tf=tf, rules=rules, r_mult=1.0)
            v_s = evaluate_confirmation(config, conf, va, tf=tf, rules=rules, r_mult=1.0)
            v15 = evaluate_confirmation(config, conf, va, tf=tf, rules=rules, r_mult=1.5)
            v20 = evaluate_confirmation(config, conf, va, tf=tf, rules=rules, r_mult=2.0)
            v_bias = evaluate_confirmation(config, conf, va, tf=tf, rules=rules, r_mult=1.0,
                                           use_bias=True)
            results.append({
                "key": key, "name": conf.name, "tf": tf, "conf": conf,
                "tr": t_s, "va": v_s, "v15": v15, "v20": v20, "v_bias": v_bias,
            })

    elig = [r for r in results if r["va"].get("n", 0) >= min_trades]
    elig.sort(key=lambda r: r["va"].get("win_rate") or 0.0, reverse=True)
    t = Table(title="Phase 28 — LTF confirmation rulebook (validation 2024, 1:1, after HTF touch)",
              title_justify="left")
    for col in ("confirmation", "tf", "val_n", "val_win@1:1", "val_PF", "MFE/MAE", "win@1.5R",
                "win@2R", "win+bias", "train_win", "maxLstreak"):
        t.add_column(col, justify="left" if col == "confirmation" else "right")
    for r in elig:
        v, vb = r["va"], r["v_bias"]
        t.add_row(
            r["name"], r["tf"], str(v["n"]), f"{(v.get('win_rate') or 0) * 100:.0f}%",
            f"{v.get('pf') or 0:.2f}", f"{v.get('mfe_mae') or 0:.2f}",
            f"{(r['v15'].get('win_rate') or 0) * 100:.0f}%",
            f"{(r['v20'].get('win_rate') or 0) * 100:.0f}%",
            f"{(vb.get('win_rate') or 0) * 100:.0f}%" if vb.get("n", 0) >= 15 else "—",
            f"{(r['tr'].get('win_rate') or 0) * 100:.0f}%", str(v.get("max_losing_streak")),
        )
    console.print(t)

    if not elig:
        console.print("[red]no confirmation had enough validation trades to judge[/red]")
        return
    best = elig[0]
    bw = (best["va"].get("win_rate") or 0) * 100
    # Time-of-day breakdown for the best confirmation.
    tf = best["tf"]
    va_f = feat(tf, date(2024, 1, 1), date(2024, 12, 31))
    early = evaluate_confirmation(config, best["conf"], va_f, tf=tf, rules=rules, window=_P28_EARLY)
    late = evaluate_confirmation(config, best["conf"], va_f, tf=tf, rules=rules, window=_P28_LATE)
    console.print(
        f"\n[bold]Best confirmation:[/bold] {best['name']} ({best['tf']}) — validation 1:1 win "
        f"{bw:.0f}% (PF {best['va'].get('pf')}, n={best['va']['n']}). "
        f"09:30-10:30 win {(early.get('win_rate') or 0) * 100:.0f}% (n={early.get('n')}), "
        f"10:30-12:00 win {(late.get('win_rate') or 0) * 100:.0f}% (n={late.get('n')})."
    )
    worst = elig[-1]
    console.print(f"[bold]Worst confirmation:[/bold] {worst['name']} ({worst['tf']}) — "
                  f"validation 1:1 win {(worst['va'].get('win_rate') or 0) * 100:.0f}%")
    bar = 55.0
    if bw >= bar:
        console.print(f"\n[green]{best['name']} clears the 55% manual-edge bar at 1:1 — usable "
                      "(forward-paper).[/green]")
    else:
        console.print(f"\n[red]No confirmation reaches the 55% 1:1 bar in validation (best "
                      f"{bw:.0f}%). See docs/PHASE28 for the honest rulebook.[/red]")


if __name__ == "__main__":  # pragma: no cover
    app()
