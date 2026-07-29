"""
Configuration loader — merges YAML config with environment overrides.
"""
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = None) -> dict[str, Any]:
    """Load configuration from YAML file and apply .env overrides."""
    load_dotenv()

    if config_path is None:
        config_path = Path(__file__).parent / "default.yaml"

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Environment overrides for sensitive / deployment-specific values
    if os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true":
        cfg["general"]["live_trading_enabled"] = True
    else:
        cfg["general"]["live_trading_enabled"] = False

    if os.getenv("PAPER_TRADE_INITIAL_EQUITY"):
        cfg["paper_trade"]["initial_equity"] = float(
            os.getenv("PAPER_TRADE_INITIAL_EQUITY")
        )

    if os.getenv("DATA_PROVIDER"):
        cfg["data"]["provider"] = os.getenv("DATA_PROVIDER")

    if os.getenv("TV_USERNAME"):
        cfg["data"].setdefault("tradingview_username", os.getenv("TV_USERNAME"))
    if os.getenv("TV_PASSWORD"):
        cfg["data"].setdefault("tradingview_password", os.getenv("TV_PASSWORD"))

    if os.getenv("OANDA_TOKEN"):
        cfg["data"]["oanda_token"] = os.getenv("OANDA_TOKEN")
    if os.getenv("OANDA_ACCOUNT_ID"):
        cfg["data"]["oanda_account_id"] = os.getenv("OANDA_ACCOUNT_ID")

    if os.getenv("LOG_LEVEL"):
        cfg.setdefault("logging", {})["level"] = os.getenv("LOG_LEVEL")

    return cfg


def is_live_trading_enabled(cfg: dict) -> bool:
    """Explicit check — live trading requires BOTH config AND env flag."""
    config_flag = cfg.get("general", {}).get("live_trading_enabled", False)
    env_flag = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
    return config_flag and env_flag
