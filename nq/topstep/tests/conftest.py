from __future__ import annotations

from pathlib import Path

import pytest

from tslab.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(PROJECT_ROOT / "configs")
