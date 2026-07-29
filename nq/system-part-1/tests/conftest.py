from __future__ import annotations

from pathlib import Path

import pytest

from nqlab.config.loader import load_config, load_topstep_rules
from nqlab.config.models import AppConfig, TopstepRules
from tests.fixtures import write_mbp1_parquet, write_trades_parquet

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    """A temp data tree: trades for 3 periods, mbp1 for 2 (missing 2025_H1)."""
    root = tmp_path / "nqdata"
    trades_dir = root / "raw" / "trades"
    mbp1_dir = root / "raw" / "mbp1"
    for label in ("2024_H1", "2024_H2", "2025_H1"):
        write_trades_parquet(trades_dir / f"{label}.parquet")
    for label in ("2024_H1", "2024_H2"):
        write_mbp1_parquet(mbp1_dir / f"{label}.parquet")
    return root


@pytest.fixture
def config(tmp_data_root: Path, config_dir: Path) -> AppConfig:
    overrides = {
        "data": {
            "nq_data_root": str(tmp_data_root),
            "curated_root": str(tmp_data_root / "curated"),
            "catalog_path": str(tmp_data_root / "curated" / "catalog.duckdb"),
            "artifacts_root": str(tmp_data_root / "curated" / "artifacts"),
        }
    }
    return load_config(
        profile="ci", config_dir=config_dir, overrides=overrides, apply_env=False
    )


@pytest.fixture
def app_config(config_dir: Path) -> AppConfig:
    """Config with real instrument/session/backtest settings (no data access)."""
    return load_config(profile="dev", config_dir=config_dir, apply_env=False)


@pytest.fixture
def rules_50k(config_dir: Path) -> TopstepRules:
    return load_topstep_rules("topstep_50k", config_dir=config_dir)
