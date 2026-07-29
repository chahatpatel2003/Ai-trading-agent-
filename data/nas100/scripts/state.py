"""state.py - crash-safe month-by-month progress in download_state.json (atomic writes)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import config

STATE_VERSION = 1


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_state() -> dict[str, Any]:
    now = _utc()
    return {
        "version": STATE_VERSION,
        "symbol": config.SYMBOL,
        "created_utc": now,
        "updated_utc": now,
        "price_factor": None,        # filled once auto-detected
        "data_start": None,          # first month with data (YYYY-MM)
        "months": {},                # "YYYY_MM" -> record
    }


def load_state() -> dict[str, Any]:
    p = config.STATE_FILE
    if not p.exists():
        return new_state()
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        try:
            p.replace(p.with_suffix(".json.corrupt"))
        except OSError:
            pass
        return new_state()
    s.setdefault("months", {})
    s.setdefault("symbol", config.SYMBOL)
    return s


def save_state(state: dict[str, Any]) -> None:
    state["updated_utc"] = _utc()
    p = config.STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, p)


def get_month(state: dict[str, Any], key: str) -> dict[str, Any]:
    return state["months"].get(key, {})


def update_month(state: dict[str, Any], key: str, **fields: Any) -> dict[str, Any]:
    rec = state["months"].setdefault(key, {})
    rec.update(fields)
    save_state(state)
    return rec
