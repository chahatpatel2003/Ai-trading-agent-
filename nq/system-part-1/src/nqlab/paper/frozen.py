"""Frozen-strategy loader with hash enforcement (Phase 15).

Loads the EXACT frozen Phase-14 ``bos_5m_trigger`` config, verifies its strategy
fingerprint against the stored hash, and REFUSES to run if anything was modified. No
parameter is tuned here — this is a runner, not a search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nqlab.config.models import AppConfig
from nqlab.discovery.phase14_families import PHASE14_FAMILIES

_RTH_OPEN_DEFAULT = 570

# The exact frozen Phase-14 candidate (see docs/PHASE14_STRUCTURE_SIZING.md).
PHASE15_STRATEGY_ID = "bos_5m_trigger"
PHASE15_FROZEN_HASH = "bedfb200abf675b9"
PHASE15_ACCOUNT = "50k"  # best account by Topstep pass probability


@dataclass(frozen=True)
class PaperConfig:
    """An immutable, hash-verified frozen strategy bound to a paper account."""

    strategy_id: str
    family: str
    timeframe: str
    account: str
    params: tuple[tuple[str, Any], ...]  # sorted (key, value)
    frozen_hash: str
    source: str

    def param_dict(self) -> dict[str, Any]:
        return dict(self.params)

    def to_strategy(self, rth_start_min: int = _RTH_OPEN_DEFAULT) -> Any:
        return PHASE14_FAMILIES[self.family](self.param_dict(), rth_start_min=rth_start_min)

    def computed_hash(self) -> str:
        return str(self.to_strategy().params.fingerprint())

    def verify(self) -> None:
        """Raise if the config no longer reproduces its frozen hash (modified config)."""
        got = self.computed_hash()
        if got != self.frozen_hash:
            msg = (
                f"frozen {self.strategy_id!r} config has been MODIFIED "
                f"({got} != {self.frozen_hash}); refusing to run a changed strategy"
            )
            raise ValueError(msg)

    def explain(self) -> str:
        """Human-readable, exact explanation of the frozen signal logic."""
        p = self.param_dict()
        return (
            f"{self.strategy_id} ({self.timeframe}, {self.account}, MNQ/{p.get('sizing_mode')}): "
            "LONG when the 4h bias is bullish AND a break of structure prints with a "
            "displacement candle (SHORT mirror: 4h bias bearish + bearish BOS + displacement); "
            f"NY-AM only ({p.get('hour_start')}:30-{p.get('hour_end')}:00 ET); "
            f"stop {p.get('stop_atr')}xATR, target {p.get('target_atr')}xATR; "
            f"min-hold {p.get('min_hold_bars')} bars, time-stop {p.get('time_stop_bars')} bars; "
            f"risk ${p.get('risk_per_trade')}; locked-buffer Topstep {self.account}. "
            f"[frozen hash {self.frozen_hash}, source {self.source}]"
        )

    def to_json(self) -> dict[str, Any]:
        d = {
            "strategy_id": self.strategy_id,
            "family": self.family,
            "timeframe": self.timeframe,
            "account": self.account,
            "params": self.param_dict(),
            "frozen_hash": self.frozen_hash,
            "source": self.source,
        }
        d["computed_hash"] = self.computed_hash()
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> PaperConfig:
        cfg = PaperConfig(
            strategy_id=d["strategy_id"],
            family=d["family"],
            timeframe=d["timeframe"],
            account=d["account"],
            params=tuple(sorted(d["params"].items())),
            frozen_hash=d["frozen_hash"],
            source=d["source"],
        )
        cfg.verify()  # refuse to load a tampered registry
        return cfg


def paper_registry_path(config: AppConfig) -> Path:
    return config.data.curated_root / "paper_frozen.json"


def save_paper_config(path: Path, cfg: PaperConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_json(), indent=2, sort_keys=True))


def load_paper_config(path: Path) -> PaperConfig:
    if not path.exists():
        msg = f"no paper registry at {path} — run `paper init` first"
        raise FileNotFoundError(msg)
    return PaperConfig.from_json(json.loads(path.read_text()))


def init_from_phase14(
    phase14_registry_path: Path,
    *,
    strategy_id: str = PHASE15_STRATEGY_ID,
    frozen_hash: str = PHASE15_FROZEN_HASH,
    account: str = PHASE15_ACCOUNT,
) -> PaperConfig:
    """Build the paper config from the EXACT frozen Phase-14 entry (matched by hash)."""
    entries = json.loads(phase14_registry_path.read_text())
    match = next(
        (e for e in entries if e["family"] == strategy_id and e.get("config_hash") == frozen_hash),
        None,
    )
    if match is None:
        msg = f"frozen Phase-14 entry {strategy_id}/{frozen_hash} not found in the registry"
        raise ValueError(msg)
    cfg = PaperConfig(
        strategy_id=strategy_id,
        family=strategy_id,
        timeframe=str(match["timeframe"]),
        account=account,
        params=tuple(sorted(match["params"].items())),
        frozen_hash=frozen_hash,
        source="phase14",
    )
    cfg.verify()  # the loaded params must reproduce the frozen hash
    return cfg
