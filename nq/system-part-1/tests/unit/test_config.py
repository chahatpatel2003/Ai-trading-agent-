from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nqlab.config.loader import load_config
from nqlab.config.models import LoggingConfig


def test_loads_nq_and_mnq(config_dir: Path) -> None:
    cfg = load_config(profile="dev", config_dir=config_dir, apply_env=False)
    assert {"NQ", "MNQ"} <= set(cfg.instruments.specs)
    nq = cfg.instruments.specs["NQ"]
    assert nq.point_value == 20.0
    assert nq.tick_size == 0.25
    assert nq.tick_value == 5.0
    mnq = cfg.instruments.specs["MNQ"]
    assert mnq.point_value == 2.0
    assert mnq.tick_value == 0.5


def test_approved_defaults(config_dir: Path) -> None:
    cfg = load_config(profile="dev", config_dir=config_dir, apply_env=False)
    assert cfg.clock.canonical.value == "ts_recv"
    assert cfg.clock.ordering == ("ts_event", "sequence")
    assert cfg.roll.method.value == "volume"
    assert cfg.roll.adjustment.value == "back_adjusted"


def test_config_hash_deterministic(config_dir: Path) -> None:
    a = load_config(profile="dev", config_dir=config_dir, apply_env=False)
    b = load_config(profile="dev", config_dir=config_dir, apply_env=False)
    assert a.config_hash() == b.config_hash()
    assert len(a.config_hash()) == 64


def test_override_changes_hash(config_dir: Path) -> None:
    a = load_config(profile="dev", config_dir=config_dir, apply_env=False)
    b = load_config(
        profile="dev",
        config_dir=config_dir,
        overrides={"logging": {"level": "WARNING"}},
        apply_env=False,
    )
    assert b.logging.level == "WARNING"
    assert a.config_hash() != b.config_hash()


def test_extra_keys_forbidden() -> None:
    with pytest.raises(ValidationError):
        LoggingConfig.model_validate({"level": "INFO", "bogus": 1})


def test_models_are_frozen() -> None:
    cfg = LoggingConfig()
    with pytest.raises(ValidationError):
        cfg.level = "DEBUG"  # type: ignore[misc]


def test_env_override(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    monkeypatch.setenv("NQLAB_LOG_LEVEL", "ERROR")
    cfg = load_config(profile="dev", config_dir=config_dir, apply_env=True)
    assert cfg.logging.level == "ERROR"


def test_data_paths(config_dir: Path) -> None:
    cfg = load_config(profile="dev", config_dir=config_dir, apply_env=False)
    assert cfg.data.raw_trades_dir.name == "trades"
    assert cfg.data.raw_trades_dir.parent.name == "raw"
    assert cfg.data.raw_mbp1_dir.name == "mbp1"
