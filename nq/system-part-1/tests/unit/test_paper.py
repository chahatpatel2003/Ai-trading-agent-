from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from nqlab.backtest.result import Trade
from nqlab.config.models import AppConfig, TopstepRules
from nqlab.features.build import build_features
from nqlab.features.context import build_context_features
from nqlab.features.htf_context import build_htf_context_features
from nqlab.features.structure import build_structure_features
from nqlab.paper.engine import run_paper
from nqlab.paper.evaluate import PENDING_FORWARD_SAMPLE, evaluate_forward
from nqlab.paper.feed import LiveFeed, build_5m_from_1m
from nqlab.paper.frozen import PaperConfig
from nqlab.paper.ledger import LEDGER_COLUMNS, append_rows, read_rows, trade_to_row
from tests.synth import globex_bars

_FROZEN_PARAMS = {
    "stop_atr": 1.0,
    "target_atr": 4.0,
    "min_hold_bars": 3,
    "time_stop_bars": 48,
    "risk_per_trade": 150.0,
    "sizing_mode": "mnq_fixed1",
    "hour_start": 9,
    "hour_end": 12,
}


def _paper_cfg() -> PaperConfig:
    from nqlab.discovery.phase14_families import PHASE14_FAMILIES

    fp = str(
        PHASE14_FAMILIES["bos_5m_trigger"](
            dict(_FROZEN_PARAMS), rth_start_min=570
        ).params.fingerprint()
    )
    return PaperConfig(
        strategy_id="bos_5m_trigger",
        family="bos_5m_trigger",
        timeframe="5m",
        account="50k",
        params=tuple(sorted(_FROZEN_PARAMS.items())),
        frozen_hash=fp,
        source="phase14",
    )


def _structure_frame() -> pl.DataFrame:
    feats = build_features(globex_bars(days=20), rth_start_min=570)
    ctx = build_context_features(feats, rth_start_min=570)
    htf = build_htf_context_features(ctx, rth_start_min=570, timeframe="5m")
    return build_structure_features(htf)


# --- frozen config enforcement -------------------------------------------------


def test_frozen_config_verifies_and_explains() -> None:
    cfg = _paper_cfg()
    cfg.verify()  # exact frozen config -> no error
    assert "bos_5m_trigger" in cfg.explain()
    assert cfg.frozen_hash in cfg.explain()


def test_modified_config_is_rejected() -> None:
    cfg = _paper_cfg()
    tampered = PaperConfig(
        cfg.strategy_id,
        cfg.family,
        cfg.timeframe,
        cfg.account,
        tuple(sorted({**_FROZEN_PARAMS, "target_atr": 9.0}.items())),  # tuned target
        cfg.frozen_hash,
        cfg.source,
    )
    with pytest.raises(ValueError, match="MODIFIED"):
        tampered.verify()
    with pytest.raises(ValueError):
        PaperConfig.from_json(tampered.to_json() | {"frozen_hash": cfg.frozen_hash})


# --- 5m bar building -----------------------------------------------------------


def test_build_5m_from_1m_aggregates_ohlcv() -> None:
    base = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    rows = [
        {
            "ts": base,
            "symbol": "NQ.c",
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
            "vwap": 100.0,
            "trades": 5,
        },
        {
            "ts": base.replace(minute=31),
            "symbol": "NQ.c",
            "open": 100.5,
            "high": 103.0,
            "low": 100.0,
            "close": 102.0,
            "volume": 20.0,
            "vwap": 102.0,
            "trades": 7,
        },
        {
            "ts": base.replace(minute=34),
            "symbol": "NQ.c",
            "open": 102.0,
            "high": 102.5,
            "low": 101.0,
            "close": 101.5,
            "volume": 30.0,
            "vwap": 101.0,
            "trades": 8,
        },
    ]
    out = build_5m_from_1m(pl.DataFrame(rows))
    assert out.height == 1
    r = out.row(0, named=True)
    assert r["open"] == 100.0 and r["high"] == 103.0 and r["low"] == 99.0 and r["close"] == 101.5
    assert r["volume"] == 60.0 and r["trades"] == 20


# --- ledger schema + round trip ------------------------------------------------


def _trade() -> Trade:
    return Trade(
        entry_ts=datetime(2025, 1, 2, 15, 0, tzinfo=UTC),
        exit_ts=datetime(2025, 1, 2, 15, 30, tzinfo=UTC),
        trading_day=date(2025, 1, 2),
        side=1,
        instrument="MNQ",
        contracts=1,
        entry_px=20000.0,
        exit_px=20040.0,
        stop_px=19980.0,
        target_px=20080.0,
        gross_pnl=80.0,
        fees=1.0,
        net_pnl=79.0,
        exit_reason="TARGET",
        bars_held=10,
        entry_minute=600,
        mae=5.0,
        mfe=42.0,
    )


def test_ledger_schema_and_roundtrip(tmp_path: Path) -> None:
    row = trade_to_row(
        _trade(),
        strategy_id="bos_5m_trigger",
        frozen_hash="abc",
        mode="replay",
        setup_reason="r",
        account_equity=50079.0,
        account_blown=False,
        rule_violations=[],
        followed_frozen_rules=True,
    )
    assert set(row.keys()) == set(LEDGER_COLUMNS)
    path = tmp_path / "ledger.jsonl"
    append_rows(path, [row, row])
    back = read_rows(path)
    assert len(back) == 2 and back[0]["pnl"] == 79.0 and back[0]["mode"] == "replay"


# --- no live order path --------------------------------------------------------


def test_live_feed_refuses_no_order_path() -> None:
    with pytest.raises(NotImplementedError):
        next(LiveFeed().bars())


def test_paper_package_has_no_order_execution() -> None:
    # Execution-call identifiers that would only appear in real order-placement code
    # (not in descriptive docstrings, which may say "no broker orders").
    pkg = Path(__file__).resolve().parents[2] / "src" / "nqlab" / "paper"
    forbidden = ("place_order", "submit_order", "send_order", "create_order", "orderclient")
    for py in pkg.glob("*.py"):
        text = py.read_text().lower()
        for tok in forbidden:
            assert tok not in text, f"{py.name} references order-execution identifier {tok!r}"


# --- paper engine: accounting, locked-buffer state, determinism ----------------


def test_paper_engine_accounting_and_determinism(
    app_config: AppConfig, rules_50k: TopstepRules
) -> None:
    cfg = _paper_cfg()
    frame = _structure_frame()
    rows1, ts1 = run_paper(app_config, cfg, feature_frame=frame, rules=rules_50k, mode="replay")
    rows2, _ = run_paper(app_config, cfg, feature_frame=frame, rules=rules_50k, mode="replay")
    assert rows1 == rows2  # replay determinism
    assert ts1 is not None  # locked-buffer Topstep state produced
    for r in rows1:
        assert set(r.keys()) == set(LEDGER_COLUMNS)
        assert r["mode"] == "replay" and r["followed_frozen_rules"] is True
        assert "account_equity" in r and "account_blown" in r
    if rows1:  # equity accounting: running equity = account_size + cumulative pnl
        equity = rules_50k.account_size
        for r in rows1:
            equity += r["pnl"]
            assert r["account_equity"] == pytest.approx(equity, abs=0.01)


# --- pending verdict until enough forward trades -------------------------------


def test_verdict_pending_until_min_forward_trades(rules_50k: TopstepRules) -> None:
    # 300 REPLAY trades + a handful of FORWARD trades -> still PENDING (replay doesn't count).
    rows = [{"pnl": 10.0, "trading_day": "2025-01-02", "mode": "replay"} for _ in range(300)] + [
        {"pnl": 10.0, "trading_day": "2026-07-01", "mode": "forward"} for _ in range(5)
    ]
    res = evaluate_forward(rows, rules=rules_50k, account_size=rules_50k.account_size, months=12.0)
    assert res["verdict"] == PENDING_FORWARD_SAMPLE
    assert res["forward_trades"] == 5 and res["replay_trades"] == 300
