from __future__ import annotations

from pathlib import Path

from nqlab.data.ledger import PartitionLedger


def test_mark_persist_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = PartitionLedger.load(path)
    assert not ledger.is_complete("202401", "h1")

    ledger.mark_complete("202401", rows=42, config_hash="h1", symbols=["NQH4"])
    ledger.save()

    reloaded = PartitionLedger.load(path)
    assert reloaded.is_complete("202401", "h1")
    assert reloaded.get("202401").rows == 42  # type: ignore[union-attr]
    assert reloaded.completed_keys() == ["202401"]


def test_config_hash_change_invalidates(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = PartitionLedger.load(path)
    ledger.mark_complete("202401", rows=1, config_hash="h1")
    assert ledger.is_complete("202401", "h1")
    assert not ledger.is_complete("202401", "h2")  # different config -> stale


def test_unreadable_ledger_is_treated_empty(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("{ not valid json")
    ledger = PartitionLedger.load(path)
    assert ledger.completed_keys() == []
