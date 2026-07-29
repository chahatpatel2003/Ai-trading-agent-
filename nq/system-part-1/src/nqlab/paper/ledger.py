"""Forward paper/shadow ledger (Phase 15): one row per simulated trade.

Append-only JSONL on disk. Each row records the full provenance needed to audit a
forward trade against the frozen rules — including the frozen hash, the setup reason,
the Topstep account state at exit, and whether the trade followed the frozen config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nqlab.backtest.result import Trade

LEDGER_COLUMNS: tuple[str, ...] = (
    "entry_ts",
    "exit_ts",
    "trading_day",
    "strategy_id",
    "frozen_hash",
    "mode",
    "setup_reason",
    "followed_frozen_rules",
    "side",
    "instrument",
    "contracts",
    "entry_px",
    "stop_px",
    "target_px",
    "exit_px",
    "exit_reason",
    "pnl",
    "mae",
    "mfe",
    "account_equity",
    "account_blown",
    "rule_violations",
)


def trade_to_row(
    trade: Trade,
    *,
    strategy_id: str,
    frozen_hash: str,
    mode: str,
    setup_reason: str,
    account_equity: float,
    account_blown: bool,
    rule_violations: list[str],
    followed_frozen_rules: bool,
) -> dict[str, Any]:
    return {
        "entry_ts": trade.entry_ts.isoformat(),
        "exit_ts": trade.exit_ts.isoformat(),
        "trading_day": trade.trading_day.isoformat(),
        "strategy_id": strategy_id,
        "frozen_hash": frozen_hash,
        "mode": mode,
        "setup_reason": setup_reason,
        "followed_frozen_rules": followed_frozen_rules,
        "side": "long" if trade.side == 1 else "short",
        "instrument": trade.instrument,
        "contracts": trade.contracts,
        "entry_px": round(trade.entry_px, 2),
        "stop_px": round(trade.stop_px, 2),
        "target_px": round(trade.target_px, 2),
        "exit_px": round(trade.exit_px, 2),
        "exit_reason": trade.exit_reason,
        "pnl": round(trade.net_pnl, 2),
        "mae": round(trade.mae, 2),
        "mfe": round(trade.mfe, 2),
        "account_equity": round(account_equity, 2),
        "account_blown": account_blown,
        "rule_violations": rule_violations,
    }


def ledger_path(curated_root: Path) -> Path:
    return curated_root / "paper_ledger.jsonl"


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clear(path: Path) -> None:
    if path.exists():
        path.unlink()
