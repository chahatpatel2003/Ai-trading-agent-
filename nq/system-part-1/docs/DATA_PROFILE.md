# DATA_PROFILE — Empirical ground truth of `/Volumes/T9/nq data`

> Generated from direct inspection of the dataset on 2026-06-26. This is the
> factual substrate the architecture is designed around. Re-run the profiler in
> Phase 0 to keep this current; do not edit by hand.

## 1. Layout on disk

```
/Volumes/T9/nq data/
├── raw/                         # IMMUTABLE source of truth — never modified
│   ├── mbp1/    2021_H1 … 2025_H1   (9 files)   ~206 GB
│   └── trades/  2021_H1 … 2026_H1   (11 files)  ~13 GB
└── tmp/         mbp1_2025_H1.dbn.zst (26 GB)     # raw Databento DBN, mid-conversion
```

Files are **half-yearly Parquet** (`YYYY_H{1,2}.parquet`), zstd-compressed,
pandas-indexed on `ts_recv`.

## 2. Provenance

- **Vendor**: Databento. `tmp/` holds the original `.dbn.zst` (Databento Binary
  Encoding); `raw/` holds the decoded Parquet. The DBN→Parquet ingestion is the
  pipeline's first (already partly complete) stage.
- **Venue**: CME Globex, NASDAQ-100 futures complex (`publisher_id = 1` = GLBX).
- **Schemas**: `trades` = Databento `trades`; `mbp1` = Databento `mbp-1`
  (top-of-book MBP, one level, with the full add/modify/cancel/trade event stream).

## 3. Coverage & scale

| Dataset | Periods | Files | Rows / half-year | Total rows (est.) | Size |
|---|---|---|---|---|---|
| trades | 2021_H1 → 2026_H1 | 11 | ~51 M | ~0.55 B | ~13 GB |
| mbp1   | 2021_H1 → 2025_H1 | 9  | ~875 M | ~7–8 B | ~206 GB |

**Availability matrix (the single most important data fact):**

| Period | trades | mbp1 |
|---|:--:|:--:|
| 2021 H1 … 2024 H2 | ✅ | ✅ |
| 2025 H1 | ✅ | ✅ |
| 2025 H2 | ✅ | ❌ missing |
| 2026 H1 | ✅ | ❌ missing |

→ Any strategy/feature that **requires order-book microstructure** can only be
researched/validated through **2025-H1**. Trades-derived features span the full
history. The platform must encode this as a first-class *data availability
contract* and refuse to silently backtest a microstructure strategy on a period
where MBP1 is absent.

## 4. Schemas (confirmed)

### `mbp1` (20 columns)
```
ts_event   timestamp[ns, UTC]   # exchange matching-engine time
ts_recv    timestamp[ns, UTC]   # Databento gateway receive time  (current index)
rtype      uint8                # record type (1 = mbp-1)
publisher_id uint16             # 1 = GLBX
instrument_id uint32            # numeric id — the join key to symbology
action     str  {A,M,C,T,F,…}   # Add / Modify / Cancel / Trade / Fill
side       str  {A,B,N}         # Ask / Bid / None
depth      uint8                # book level (0 for mbp-1)
price      double               # decoded price (real units, NOT 1e-9 fixed point)
size       uint32
flags      uint8                # bitfield (last, snapshot, bad-ts, etc.)
ts_in_delta int32               # ts_recv − matching-engine send (ns)
sequence   uint32               # venue sequence number (per-instrument ordering)
bid_px_00  double               # top-of-book quote AFTER this event
ask_px_00  double
bid_sz_00  uint32
ask_sz_00  uint32
bid_ct_00  uint32               # # resting orders at best bid
ask_ct_00  uint32               # # resting orders at best ask
symbol     str                  # e.g. "NQH4", "NQH4-NQM4" (spread)
```

### `trades` (14 columns)
Same core fields, no book columns. `action = T`, `side` = **aggressor** side.

## 5. Symbology — multiple instruments per file

A single file interleaves **all listed NQ instruments** at that time:

- **Outrights**: front month + deferred (`NQH4`, `NQM4`, `NQU4`, `NQZ4`, `NQH5`, …).
  Quarterly cycle **H=Mar, M=Jun, U=Sep, Z=Dec**, expiry 3rd Friday.
- **Calendar spreads**: `NQH4-NQM4`, `NQM4-NQU4`, … (priced as differentials, e.g.
  `207.3` — NOT outright price levels; must never be mixed into a price series).

Implication: the curated layer must (a) join `instrument_id → contract metadata`,
(b) **separate outrights from spreads**, (c) determine the **front month** over
time, and (d) build a **roll calendar + continuous (back-adjusted) series**.
None of this is optional — every downstream bar/feature/backtest depends on it.

## 6. Action mix (sampled, mbp1)

`A` (add) ≈ 47%, `C` (cancel) ≈ 45%, `M` (modify) ≈ 7%, `T` (trade) ≈ 6%.
The MBP1 stream therefore contains the **trade prints too**, enabling
quote-at-trade reconstruction, signed order flow, and queue dynamics — without
joining the separate `trades` file (which remains the canonical, gap-free
time-&-sales source).

## 7. Instrument economics (reference constants → config, not code)

| | NQ (E-mini) | MNQ (Micro) |
|---|---|---|
| Point value | $20 | $2 |
| Tick | 0.25 pt | 0.25 pt |
| Tick value | $5.00 | $0.50 |
| Cycle | Mar/Jun/Sep/Dec | same |

## 8. Engineering consequences (carried into ARCHITECTURE.md)

1. **Out-of-core by default.** 7–8 B rows / 220 GB on a USB-attached SSD: every
   transform is partition-streamed (DuckDB/Polars-lazy/PyArrow datasets), never
   `pandas.read_parquet(wholefile)`.
2. **Don't duplicate 200 GB.** Treat `raw/` as the immutable L0 source; avoid
   materializing a full standardized L1 copy of MBP1. Materialize only the much
   smaller derived layers (bars, order-flow aggregates, features).
3. **Pick the canonical clock.** Default research/availability clock = `ts_recv`
   (information-arrival time → look-ahead-safe); `ts_event`+`sequence` used only
   for intra-instrument event ordering. Configurable, documented, tested.
4. **Roll method is a global, versioned decision.** Recommend volume-based roll
   (OI is unavailable in this feed) with a calendar guard; back-adjusted for
   research, raw per-contract retained for execution realism.
5. **MBP1 gap is structural**, not a bug to fix — encode as availability metadata.
