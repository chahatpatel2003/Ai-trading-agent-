from __future__ import annotations

import json
from pathlib import Path

from nqlab.config.models import AppConfig
from nqlab.data.manifest import build_manifest


def test_manifest_datasets_and_matrix(config: AppConfig) -> None:
    manifest = build_manifest(config, sample_instruments=True)

    assert manifest.datasets["trades"].file_count == 3
    assert manifest.datasets["mbp1"].file_count == 2
    assert manifest.datasets["trades"].total_rows == 36  # 3 files * 12 rows

    # Cross-dataset trailing gap: trades has 2025_H1, mbp1 does not.
    row = next(r for r in manifest.availability_matrix if r["period"] == "2025_H1")
    assert row["trades"] is True
    assert row["mbp1"] is False


def test_manifest_instruments_sampled(config: AppConfig) -> None:
    manifest = build_manifest(config, sample_instruments=True)
    classes = {obs.classification for obs in manifest.instruments}
    assert "outright" in classes
    assert "spread" in classes
    outright = next(o for o in manifest.instruments if o.classification == "outright")
    assert outright.root == "NQ"


def test_manifest_metadata(config: AppConfig) -> None:
    manifest = build_manifest(config, sample_instruments=False)
    assert manifest.canonical_clock == "ts_recv"
    assert manifest.config_hash == config.config_hash()
    assert manifest.sampled_instruments is False
    assert manifest.instruments == []


def test_manifest_write_json(config: AppConfig, tmp_path: Path) -> None:
    manifest = build_manifest(config, sample_instruments=False)
    out = manifest.write_json(tmp_path / "manifest.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert "availability_matrix" in data
    assert data["sampled_instruments"] is False
    assert set(data["datasets"]) == {"trades", "mbp1"}
