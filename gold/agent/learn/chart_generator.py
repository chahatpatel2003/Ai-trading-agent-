"""
Technical Analysis Chart Generator for XAUUSD.
Generates candlestick charts with SMA, Bollinger Bands, RSI, MACD, and volume.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from datetime import datetime

from data.indicators import sma, ema, bollinger_bands, rsi, macd, atr


def generate_chart(df: pd.DataFrame, output_dir: str = "./learn/charts", days: int = 30) -> str:
    """Generate a full technical analysis chart and save as PNG."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Use last N days of data
    bars_per_day = 24  # 1H data
    n_bars = min(days * bars_per_day, len(df))
    data = df.tail(n_bars).copy()

    fig, axes = plt.subplots(4, 1, figsize=(18, 14), height_ratios=[4, 1, 1, 1],
                              gridspec_kw={"hspace": 0.05})
    fig.patch.set_facecolor("#1a1a2e")

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=8)
        ax.spines["top"].set_color("#333")
        ax.spines["bottom"].set_color("#333")
        ax.spines["left"].set_color("#333")
        ax.spines["right"].set_color("#333")

    # ═══════════════════════ Panel 1: Candlesticks + MAs + BBands ═══════════════════════
    ax1 = axes[0]

    # Plot candlesticks manually
    up = data[data["close"] >= data["open"]]
    down = data[data["close"] < data["open"]]

    # Wicks
    ax1.vlines(up.index, up["low"], up["high"], color="#00d4aa", linewidth=0.8)
    ax1.vlines(down.index, down["low"], down["high"], color="#ff6b6b", linewidth=0.8)

    # Bodies
    bar_width = pd.Timedelta(minutes=40)
    for _, row in up.iterrows():
        ax1.add_patch(plt.Rectangle(
            (mdates.date2num(row.name) - 0.015, row["open"]),
            0.03, row["close"] - row["open"],
            facecolor="#00d4aa", edgecolor="#00d4aa", linewidth=0.5
        ))
    for _, row in down.iterrows():
        ax1.add_patch(plt.Rectangle(
            (mdates.date2num(row.name) - 0.015, row["close"]),
            0.03, row["open"] - row["close"],
            facecolor="#ff6b6b", edgecolor="#ff6b6b", linewidth=0.5
        ))

    # SMAs
    sma20 = sma(data["close"], 20)
    sma50 = sma(data["close"], 50)
    ax1.plot(data.index, sma20, color="#ffd700", linewidth=1.2, label="SMA 20", alpha=0.9)
    ax1.plot(data.index, sma50, color="#ff69b4", linewidth=1.2, label="SMA 50", alpha=0.9)

    # Bollinger Bands
    bb_upper, bb_mid, bb_lower = bollinger_bands(data["close"], 20, 2.0)
    ax1.fill_between(data.index, bb_upper, bb_lower, alpha=0.08, color="#4fc3f7")
    ax1.plot(data.index, bb_upper, color="#4fc3f7", linewidth=0.6, alpha=0.5)
    ax1.plot(data.index, bb_lower, color="#4fc3f7", linewidth=0.6, alpha=0.5)

    # Current price line
    current_price = data["close"].iloc[-1]
    ax1.axhline(y=current_price, color="#ffd700", linewidth=0.5, linestyle="--", alpha=0.5)
    ax1.text(data.index[-1], current_price, f"  ${current_price:.2f}",
             color="#ffd700", fontsize=9, va="center", fontweight="bold")

    ax1.legend(loc="upper left", fontsize=8, facecolor="#16213e", edgecolor="#333",
               labelcolor="white")
    ax1.set_ylabel("Price (USD)", color="white", fontsize=9)
    ax1.set_title(f"XAUUSD Technical Analysis — {data.index[0].strftime('%b %d')} to {data.index[-1].strftime('%b %d, %Y')}",
                  color="white", fontsize=13, fontweight="bold", pad=10)
    ax1.xaxis.set_visible(False)

    # ═══════════════════════ Panel 2: Volume ═══════════════════════
    ax2 = axes[1]
    vol_colors = ["#00d4aa" if c >= o else "#ff6b6b"
                  for c, o in zip(data["close"], data["open"])]
    ax2.bar(data.index, data["volume"], color=vol_colors, alpha=0.6, width=0.03)
    vol_avg = data["volume"].rolling(20, min_periods=1).mean()
    ax2.plot(data.index, vol_avg, color="#ffd700", linewidth=1, alpha=0.7, label="Vol MA(20)")
    ax2.set_ylabel("Volume", color="white", fontsize=8)
    ax2.legend(loc="upper left", fontsize=7, facecolor="#16213e", edgecolor="#333",
               labelcolor="white")
    ax2.xaxis.set_visible(False)

    # ═══════════════════════ Panel 3: RSI ═══════════════════════
    ax3 = axes[2]
    rsi_vals = rsi(data["close"], 14)
    ax3.plot(data.index, rsi_vals, color="#e040fb", linewidth=1.2)
    ax3.axhline(y=70, color="#ff6b6b", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.axhline(y=30, color="#00d4aa", linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.axhline(y=50, color="gray", linewidth=0.5, linestyle=":", alpha=0.3)
    ax3.fill_between(data.index, rsi_vals, 70, where=rsi_vals > 70, alpha=0.2, color="#ff6b6b")
    ax3.fill_between(data.index, rsi_vals, 30, where=rsi_vals < 30, alpha=0.2, color="#00d4aa")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("RSI(14)", color="white", fontsize=8)
    ax3.xaxis.set_visible(False)

    # ═══════════════════════ Panel 4: MACD ═══════════════════════
    ax4 = axes[3]
    macd_line, signal_line, macd_hist = macd(data["close"], 12, 26, 9)
    hist_colors = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in macd_hist]
    ax4.bar(data.index, macd_hist, color=hist_colors, alpha=0.5, width=0.03)
    ax4.plot(data.index, macd_line, color="#4fc3f7", linewidth=1, label="MACD")
    ax4.plot(data.index, signal_line, color="#ff9800", linewidth=1, label="Signal")
    ax4.axhline(y=0, color="gray", linewidth=0.5, alpha=0.3)
    ax4.set_ylabel("MACD", color="white", fontsize=8)
    ax4.legend(loc="upper left", fontsize=7, facecolor="#16213e", edgecolor="#333",
               labelcolor="white")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax4.tick_params(axis="x", rotation=30)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"xauusd_analysis_{timestamp}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return str(path)
