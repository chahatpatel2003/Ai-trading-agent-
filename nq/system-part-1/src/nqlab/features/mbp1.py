"""Bar-level microstructure features from Databento MBP-1 (top-of-book).

One scan of the MBP-1 event stream for a contract yields both quote features
(spread, mid, microprice, sizes, top-of-book imbalance, OFI, depletion/
replenishment, realized-spread proxy, mid volatility) and order-flow features
(signed volume, buy/sell delta, trade aggression, trade/quote counts). Bucketed
by ``ts_recv`` (the availability clock); CVD and the liquidity-regime ratio are
added downstream (they need cross-bar context).

Sizes are uint32 in the raw file and MUST be cast before arithmetic — unsigned
subtraction underflows. Crossed/empty quotes are excluded from quote features.

Trade sign convention (Databento): ``side == 'B'`` is a buy aggressor (+),
``'A'`` is a sell aggressor (-), ``'N'`` is unknown (0).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import polars as pl

from nqlab.config.models import AppConfig
from nqlab.data.bars_build import _trade_files_for_range
from nqlab.domain.calendar.sessions import SessionLabeler
from nqlab.domain.market.bars import polars_every

MBP1_FEATURE_VERSION = "1"

MBP1_FEATURE_NAMES: tuple[str, ...] = (
    "spread", "mid", "microprice", "bid_size", "ask_size", "tob_imbalance", "ofi",
    "signed_volume", "buy_volume", "sell_volume", "delta", "quote_updates",
    "trade_count", "avg_trade_size", "trade_aggression", "bid_depletion",
    "bid_replenishment", "ask_depletion", "ask_replenishment",
    "realized_spread_proxy", "mid_vol",
)
# Added downstream (need cross-bar context): cvd, liq_spread_ratio, liquidity_regime.


def trade_sign(side: str) -> int:
    """+1 buy aggressor, -1 sell aggressor, 0 unknown (Databento ``side``)."""
    if side == "B":
        return 1
    if side == "A":
        return -1
    return 0


def _utc(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def build_mbp1_bar_features(
    config: AppConfig, *, symbol: str, timeframe: str, start: date, end: date
) -> pl.DataFrame:
    """Aggregate MBP-1 events for one outright into bar-level features (+calendar)."""
    files = _trade_files_for_range(config.data.raw_mbp1_dir, start, end)
    if not files:
        return pl.DataFrame()
    every = polars_every(timeframe)
    lo, hi = _utc(start), _utc(end + timedelta(days=1))

    lf = (
        pl.scan_parquet([str(f) for f in files])
        .filter((pl.col("ts_recv") >= lo) & (pl.col("ts_recv") < hi))
        .filter(pl.col("symbol") == symbol)
        .select([
            "ts_recv", "sequence", "action", "side", "price",
            "size", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00",
        ])
        .with_columns(
            pl.col("bid_sz_00").cast(pl.Float64).alias("bsz"),
            pl.col("ask_sz_00").cast(pl.Float64).alias("asz"),
            pl.col("bid_px_00").cast(pl.Float64).alias("bpx"),
            pl.col("ask_px_00").cast(pl.Float64).alias("apx"),
            pl.col("size").cast(pl.Float64).alias("sz"),
        )
        .sort(["ts_recv", "sequence"])
    )

    is_trade = pl.col("action") == "T"
    valid = (
        pl.col("bpx").is_not_null() & pl.col("apx").is_not_null()
        & (pl.col("apx") >= pl.col("bpx")) & (pl.col("bpx") > 0) & (pl.col("apx") > 0)
    )
    book = pl.col("bsz") + pl.col("asz")
    lf = lf.with_columns(
        valid.alias("v"),
        is_trade.alias("t"),
        ((pl.col("bpx") + pl.col("apx")) / 2).alias("mid_e"),
        (pl.col("apx") - pl.col("bpx")).alias("spr_e"),
        pl.when(book > 0)
        .then((pl.col("bpx") * pl.col("asz") + pl.col("apx") * pl.col("bsz")) / book)
        .alias("micro_e"),
        pl.when(book > 0).then((pl.col("bsz") - pl.col("asz")) / book).alias("imb_e"),
        pl.col("bsz").diff().alias("dbid"),
        pl.col("asz").diff().alias("dask"),
    ).with_columns(
        pl.when(is_trade & (pl.col("side") == "B")).then(pl.col("sz"))
        .when(is_trade & (pl.col("side") == "A")).then(-pl.col("sz"))
        .otherwise(0.0).alias("signed"),
        pl.when(is_trade & (pl.col("side") == "B")).then(pl.col("sz")).otherwise(0.0).alias("buyv"),
        pl.when(is_trade & (pl.col("side") == "A")).then(pl.col("sz"))
        .otherwise(0.0).alias("sellv"),
        pl.col("ts_recv").dt.truncate(every).alias("ts"),
    )

    pos_d_bid = pl.when(pl.col("dbid") > 0).then(pl.col("dbid")).otherwise(0.0)
    neg_d_bid = pl.when(pl.col("dbid") < 0).then(-pl.col("dbid")).otherwise(0.0)
    pos_d_ask = pl.when(pl.col("dask") > 0).then(pl.col("dask")).otherwise(0.0)
    neg_d_ask = pl.when(pl.col("dask") < 0).then(-pl.col("dask")).otherwise(0.0)
    eff_spread = (2.0 * (pl.col("price") - pl.col("mid_e")).abs()).filter(pl.col("t") & pl.col("v"))

    agg = lf.group_by("ts").agg(
        pl.col("spr_e").filter(pl.col("v")).mean().alias("spread"),
        pl.col("mid_e").filter(pl.col("v")).last().alias("mid"),
        pl.col("micro_e").filter(pl.col("v")).last().alias("microprice"),
        pl.col("bsz").filter(pl.col("v")).mean().alias("bid_size"),
        pl.col("asz").filter(pl.col("v")).mean().alias("ask_size"),
        pl.col("imb_e").filter(pl.col("v")).mean().alias("tob_imbalance"),
        (pl.col("dbid") - pl.col("dask")).sum().alias("ofi"),
        pl.col("signed").sum().alias("signed_volume"),
        pl.col("buyv").sum().alias("buy_volume"),
        pl.col("sellv").sum().alias("sell_volume"),
        (~pl.col("t")).sum().alias("quote_updates"),
        pl.col("t").sum().alias("trade_count"),
        pl.col("sz").filter(pl.col("t")).sum().alias("_tradesz"),
        neg_d_bid.sum().alias("bid_depletion"),
        pos_d_bid.sum().alias("bid_replenishment"),
        neg_d_ask.sum().alias("ask_depletion"),
        pos_d_ask.sum().alias("ask_replenishment"),
        eff_spread.mean().alias("realized_spread_proxy"),
        pl.col("mid_e").filter(pl.col("v")).std().alias("mid_vol"),
    ).sort("ts")

    df = agg.collect()
    if df.height == 0:
        return df
    df = df.with_columns(
        (pl.col("buy_volume") - pl.col("sell_volume")).alias("delta"),
        (pl.col("_tradesz") / pl.col("trade_count")).alias("avg_trade_size"),
        (pl.col("signed_volume") / (pl.col("_tradesz") + 1.0)).alias("trade_aggression"),
    ).drop("_tradesz")
    return SessionLabeler.from_config(config.sessions).with_calendar_columns(df, "ts")
