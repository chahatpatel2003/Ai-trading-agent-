# NQ Futures Market Data

High-resolution Nasdaq-100 futures data divided into market-by-price level-one updates and executed trades.

## Coverage

The external collection contains half-year files covering 2021 through 2026.

## Repository Content

- `samples/nq_mbp1_sample.parquet` — representative MBP-1 sample
- `samples/nq_trades_sample.parquet` — representative trade sample
- `metadata/parquet_manifest.csv` — file sizes, row counts and schemas

## Data Types

### MBP-1

Includes event timestamps, actions, sides, prices, sizes, best bid/ask prices, queue sizes, order counts, symbols and receive timestamps.

### Trades

Includes event timestamps, trade prices, sizes, sides, symbols, sequence information and receive timestamps.

The complete dataset is approximately 219.84 GB and remains on the T9 external drive.
