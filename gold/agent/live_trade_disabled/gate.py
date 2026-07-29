"""
LIVE TRADING GATE
=================
This module exists solely to PREVENT accidental live trading.
Live trading is disabled by default and requires passing ALL deployment gates.

To enable live trading, you must:
1. Set ENABLE_LIVE_TRADING=true in .env
2. Set live_trading_enabled: true in config YAML
3. Pass ALL deployment gate checks (see reports/deployment_gate.py)
4. Acknowledge the risks by calling acknowledge_live_risks()
"""
import os
import sys

from logs.logger import get_logger

log = get_logger("live_trade.gate")

# Global flag — starts as False, can never be set to True without explicit gates
_LIVE_ACKNOWLEDGED = False


class LiveTradingDisabledError(Exception):
    """Raised when live trading is attempted but not enabled."""
    pass


def check_live_trading_gate(cfg: dict) -> tuple[bool, list[str]]:
    """
    Check ALL gates required for live trading.
    Returns (is_allowed, list_of_failures).
    ALL checks must pass.
    """
    failures = []

    # Gate 1: Environment variable
    env_flag = os.getenv("ENABLE_LIVE_TRADING", "false").lower()
    if env_flag != "true":
        failures.append("ENABLE_LIVE_TRADING env var is not 'true'")

    # Gate 2: Config file flag
    config_flag = cfg.get("general", {}).get("live_trading_enabled", False)
    if not config_flag:
        failures.append("live_trading_enabled is false in config YAML")

    # Gate 3: Explicit acknowledgment
    if not _LIVE_ACKNOWLEDGED:
        failures.append("Live trading risks not acknowledged (call acknowledge_live_risks())")

    is_allowed = len(failures) == 0

    if not is_allowed:
        log.warning("live_trading_blocked", failures=failures)
    else:
        log.warning(
            "LIVE_TRADING_GATE_PASSED",
            message="Live trading is enabled. Real money is at risk.",
        )

    return is_allowed, failures


def acknowledge_live_risks() -> None:
    """
    Must be called explicitly to acknowledge live trading risks.
    This is an intentional friction point.
    """
    global _LIVE_ACKNOWLEDGED
    log.warning(
        "live_risk_acknowledgment",
        message=(
            "You are acknowledging that live trading involves real financial risk. "
            "Past backtest and paper trading results do NOT guarantee future performance. "
            "You may lose money."
        ),
    )
    _LIVE_ACKNOWLEDGED = True


def block_live_trading(reason: str = "Live trading is disabled") -> None:
    """Hard block — raises an exception to prevent any live execution."""
    log.error("live_trading_blocked_hard", reason=reason)
    raise LiveTradingDisabledError(
        f"\n{'='*60}\n"
        f"LIVE TRADING IS DISABLED\n"
        f"Reason: {reason}\n"
        f"\n"
        f"To enable live trading, ALL of the following must be true:\n"
        f"  1. ENABLE_LIVE_TRADING=true in .env\n"
        f"  2. live_trading_enabled: true in config/default.yaml\n"
        f"  3. All deployment gates passed\n"
        f"  4. Risks explicitly acknowledged in code\n"
        f"\n"
        f"Do NOT enable live trading unless you have:\n"
        f"  - Run paper trading for 4+ weeks\n"
        f"  - Validated out-of-sample performance\n"
        f"  - Confirmed results match backtest within tolerance\n"
        f"  - Accepted that you may lose real money\n"
        f"{'='*60}"
    )
