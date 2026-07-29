"""
Gold Trading Bot — Dashboard API

Read-only layer over the bot's existing data files.
Zero changes to the bot backend.
"""

import json
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware


class NoCacheMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app = FastAPI(title="Gold Trading Bot Dashboard")
app.add_middleware(NoCacheMiddleware)

BOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BOT_DIR / "learn" / "data"
LOG_DIR = BOT_DIR / "logs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | list:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _bot_pids() -> dict:
    """Return PIDs of running bot processes."""
    import shutil
    pgrep = shutil.which("pgrep")
    result = {"learn_deep": None, "learn_live": None, "launcher": None}
    try:
        out = subprocess.check_output(
            ["ps", "aux"], text=True, timeout=5
        )
        for line in out.splitlines():
            if "main.py learn-deep" in line and "grep" not in line:
                result["learn_deep"] = int(line.split()[1])
            elif "main.py learn-live" in line and "grep" not in line:
                result["learn_live"] = int(line.split()[1])
            elif "run_learner.sh" in line and "grep" not in line and result["launcher"] is None:
                result["launcher"] = int(line.split()[1])
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    pids = _bot_pids()
    running = pids["learn_deep"] is not None or pids["learn_live"] is not None

    # Tick count and latest price
    tick_file = DATA_DIR / "live_ticks.jsonl"
    tick_count = 0
    latest_tick = None
    if tick_file.exists():
        size = tick_file.stat().st_size
        tick_count = size // 75  # approximate line count from file size
        try:
            with open(tick_file, "rb") as f:
                f.seek(max(0, size - 200))
                lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                if lines:
                    latest_tick = json.loads(lines[-1])
                    # Get accurate count from wc -l if small enough, else estimate
                    if size < 100_000_000:
                        result = subprocess.run(
                            ["wc", "-l", str(tick_file)],
                            capture_output=True, text=True, timeout=5
                        )
                        tick_count = int(result.stdout.strip().split()[0])
        except Exception:
            pass

    # Last cycle info from log
    last_cycle = None
    log_path = LOG_DIR / "learner_tradingview.log"
    if log_path.exists():
        try:
            with open(log_path, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 5000))
                tail = f.read().decode("utf-8", errors="ignore")
            for line in reversed(tail.splitlines()):
                if "Cycle" in line and "done" in line:
                    last_cycle = line.strip()
                    break
        except Exception:
            pass

    # OANDA practice account live data
    oanda_account = None
    try:
        import oandapyV20
        from oandapyV20.endpoints import accounts as oanda_accounts

        env_path = BOT_DIR / ".env"
        token = account_id = ""
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OANDA_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                elif line.startswith("OANDA_ACCOUNT_ID="):
                    account_id = line.split("=", 1)[1].strip()
        if token and account_id:
            client = oandapyV20.API(access_token=token, environment="practice")
            r = oanda_accounts.AccountSummary(accountID=account_id)
            client.request(r)
            acct = r.response["account"]
            oanda_account = {
                "balance": float(acct["balance"]),
                "unrealized_pnl": float(acct.get("unrealizedPL", 0)),
                "nav": float(acct.get("NAV", acct["balance"])),
                "margin_used": float(acct.get("marginUsed", 0)),
                "open_trades": int(acct.get("openTradeCount", 0)),
                "open_positions": int(acct.get("openPositionCount", 0)),
                "pl": float(acct.get("pl", 0)),
                "currency": acct.get("currency", "USD"),
                "account_id": account_id,
            }
    except Exception:
        pass

    return {
        "running": running,
        "pids": pids,
        "tick_count": tick_count,
        "latest_tick": latest_tick,
        "last_cycle": last_cycle,
        "oanda": oanda_account,
    }


@app.get("/api/knowledge")
def get_knowledge():
    kb = _read_json(DATA_DIR / "knowledge_base.json")
    return {
        "total_observations": kb.get("total_observations", 0),
        "last_updated": kb.get("last_updated", ""),
        "pattern_counts": kb.get("pattern_counts", {}),
        "support_resistance": kb.get("support_resistance", []),
        "session_stats": kb.get("session_stats", {}),
    }


@app.get("/api/strategies")
def get_strategies():
    lab = _read_json(DATA_DIR / "strategy_lab.json")
    hyps = lab.get("hypotheses", [])
    return {
        "last_updated": lab.get("last_updated", ""),
        "hypotheses": [
            {
                "name": h.get("name", ""),
                "direction": h.get("direction", ""),
                "timeframe": h.get("timeframe", ""),
                "status": h.get("status", ""),
                "signals_generated": h.get("signals_generated", 0),
                "correct": h.get("correct", 0),
                "wrong": h.get("wrong", 0),
                "win_rate": h.get("win_rate", 0),
                "total_pnl": h.get("total_pnl", 0),
                "confidence_score": h.get("confidence_score", 0),
                "rules": h.get("rules", []),
                "origin": h.get("origin", ""),
            }
            for h in hyps
        ],
    }


@app.get("/api/predictions")
def get_predictions():
    pred = _read_json(DATA_DIR / "predictions.json")
    return {
        "active": pred.get("active", []),
        "resolved": pred.get("resolved", []),
        "pattern_accuracy": pred.get("pattern_accuracy", {}),
    }


@app.get("/api/trades")
def get_trades():
    db_path = DATA_DIR / "trade_journal.db"
    if not db_path.exists():
        return {"trades": [], "summary": {}}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        raw_trades = cur.execute(
            "SELECT * FROM trade_log ORDER BY timestamp DESC LIMIT 100"
        ).fetchall()
        trades = []
        for r in raw_trades:
            t = dict(r)
            # Parse JSON string columns
            for col in ("reasons", "market_context"):
                if isinstance(t.get(col), str):
                    try:
                        t[col] = json.loads(t[col])
                    except (json.JSONDecodeError, TypeError):
                        pass
            trades.append(t)

        # Summary by strategy
        summary = {}
        for r in cur.execute(
            "SELECT strategy, direction, COUNT(*) as cnt, AVG(confidence) as avg_conf "
            "FROM trade_log GROUP BY strategy, direction ORDER BY cnt DESC"
        ).fetchall():
            summary[f"{r['strategy']}_{r['direction']}"] = {
                "strategy": r["strategy"],
                "direction": r["direction"],
                "count": r["cnt"],
                "avg_confidence": round(r["avg_conf"], 3),
            }

        # Daily summaries
        daily = [
            dict(r)
            for r in cur.execute(
                "SELECT * FROM daily_summary ORDER BY rowid DESC LIMIT 10"
            ).fetchall()
        ]

        conn.close()
        return {"trades": trades, "summary": summary, "daily": daily}
    except Exception as e:
        return {"trades": [], "summary": {}, "error": str(e)}


@app.get("/api/market")
def get_market():
    mu = _read_json(DATA_DIR / "market_universe_state.json")

    # Extract gold correlations from the matrix
    corr_matrix = mu.get("correlation_matrix", {})
    gold_corr = {}
    if "gold" in corr_matrix:
        gold_row = corr_matrix["gold"]
        for market, val in sorted(
            gold_row.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True
        ):
            if market != "gold" and isinstance(val, (int, float)):
                gold_corr[market] = round(val, 3)

    # Lead-lag
    lead_lag = mu.get("lead_lag", {})
    top_leads = {}
    for mkt, data in list(lead_lag.items())[:10]:
        if isinstance(data, dict):
            top_leads[mkt] = data

    # Regime
    regime_history = mu.get("regime_history", [])
    latest_regime = regime_history[-1] if regime_history else {}

    return {
        "last_updated": mu.get("last_updated", ""),
        "rmt_diagnostics": mu.get("rmt_diagnostics", {}),
        "gold_correlations": gold_corr,
        "lead_lag": top_leads,
        "latest_regime": latest_regime,
        "regime_history": regime_history[-20:],
    }


@app.get("/api/risk")
def get_risk():
    return _read_json(DATA_DIR / "risk_manager_state.json")


@app.get("/api/thinker")
def get_thinker():
    return _read_json(DATA_DIR / "thinker_state.json")


@app.get("/api/evolver")
def get_evolver():
    return _read_json(DATA_DIR / "evolver_state.json")


@app.get("/api/optimizer")
def get_optimizer():
    return _read_json(DATA_DIR / "optimizer_state.json")


@app.get("/api/paper")
def get_paper():
    return _read_json(DATA_DIR / "paper_trader_state.json")


@app.get("/api/mental")
def get_mental():
    data = _read_json(DATA_DIR / "mental_trades_state.json")
    return {
        "open_trades": list((data.get("open_trades") or {}).values()),
        "resolved_trades": data.get("resolved_trades", []),
    }


@app.get("/api/candles")
def get_candles(bars: int = 60, tf: str = "5m"):
    """Build candles from live tick data for mini-charts. No backend changes."""
    tick_file = DATA_DIR / "live_ticks.jsonl"
    if not tick_file.exists():
        return {"candles": [], "tf": tf}

    try:
        # Read last chunk of ticks
        size = tick_file.stat().st_size
        read_bytes = min(size, 500_000)  # last ~500KB
        with open(tick_file, "rb") as f:
            f.seek(max(0, size - read_bytes))
            raw = f.read().decode("utf-8", errors="ignore")

        lines = raw.strip().splitlines()
        if lines and not lines[0].startswith("{"):
            lines = lines[1:]  # skip partial first line

        ticks = []
        for line in lines:
            try:
                t = json.loads(line)
                ticks.append(t)
            except Exception:
                continue

        if not ticks:
            return {"candles": [], "tf": tf}

        # Build candles from ticks
        tf_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(tf, 300)
        candles = []
        bucket = None
        bucket_start = None

        for tick in ticks:
            ts = tick.get("t", "")
            mid = tick.get("m", 0)
            if not ts or not mid:
                continue

            # Parse timestamp to bucket
            try:
                from datetime import datetime as dt
                parsed = dt.fromisoformat(ts)
                epoch = int(parsed.timestamp())
                bucket_key = (epoch // tf_seconds) * tf_seconds
            except Exception:
                continue

            if bucket_key != bucket_start:
                if bucket:
                    candles.append(bucket)
                bucket = {"t": bucket_key, "o": mid, "h": mid, "l": mid, "c": mid}
                bucket_start = bucket_key
            else:
                bucket["h"] = max(bucket["h"], mid)
                bucket["l"] = min(bucket["l"], mid)
                bucket["c"] = mid

        if bucket:
            candles.append(bucket)

        return {"candles": candles[-bars:], "tf": tf}
    except Exception as e:
        return {"candles": [], "tf": tf, "error": str(e)}


@app.get("/api/intelligence")
def get_intelligence():
    result = {}

    # COT
    cot_history = _read_json(DATA_DIR / "cot_gold_history.json")
    if isinstance(cot_history, list) and cot_history:
        latest = cot_history[-1]
        result["cot"] = latest
    else:
        result["cot"] = None

    # Correlations (indicator lead-lag, pre-move patterns, round numbers)
    corr = _read_json(DATA_DIR / "correlations.json")
    result["indicator_lead_lag"] = corr.get("indicator_lead_lag", {})
    rn = corr.get("round_number_reactions", [])
    result["round_number"] = rn[-1] if rn else {}
    result["pre_move_pattern_count"] = len(corr.get("pre_move_patterns", []))

    # Lifecycle
    lifecycle = _read_json(DATA_DIR / "lifecycle_state.json")
    result["lifecycle"] = lifecycle

    # Learning journal
    journal = _read_json(DATA_DIR / "learning_journal.json")
    if isinstance(journal, list):
        result["journal_entries"] = len(journal)
        result["latest_journal"] = journal[-1] if journal else {}
    else:
        result["journal_entries"] = 0
        result["latest_journal"] = {}

    return result


@app.get("/api/logs")
def get_logs(lines: int = 80):
    log_path = LOG_DIR / "learner_tradingview.log"
    if not log_path.exists():
        return {"lines": []}
    try:
        with open(log_path, "rb") as f:
            size = f.seek(0, 2)
            # Read last ~20KB
            f.seek(max(0, size - 20000))
            tail = f.read().decode("utf-8", errors="ignore")
        all_lines = tail.strip().splitlines()
        # Strip ANSI codes
        import re
        ansi_re = re.compile(r"\x1b\[[0-9;]*m")
        cleaned = [ansi_re.sub("", l) for l in all_lines[-lines:]]
        return {"lines": cleaned}
    except Exception as e:
        return {"lines": [str(e)]}


@app.post("/api/bot/start")
def start_bot():
    pids = _bot_pids()
    if pids["learn_deep"] or pids["learn_live"]:
        return {"status": "already_running", "pids": pids}

    try:
        subprocess.Popen(
            ["nohup", "bash", str(BOT_DIR / "run_learner.sh")],
            cwd=str(BOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(3)
        return {"status": "started", "pids": _bot_pids()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/api/bot/stop")
def stop_bot():
    try:
        subprocess.run(["pkill", "-9", "-f", "caffeinate.*run_learner"], timeout=3, capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "run_learner.sh"], timeout=3, capture_output=True)
        time.sleep(1)
        subprocess.run(["pkill", "-9", "-f", "main.py learn"], timeout=3, capture_output=True)
        time.sleep(2)
        return {"status": "stopped", "pids": _bot_pids()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Static files & SPA
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.get("/")
def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
