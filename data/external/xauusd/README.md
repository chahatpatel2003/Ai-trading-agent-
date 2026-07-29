# XAU/USD One-Minute Dataset

Cleaned one-minute XAU/USD OHLCV data produced from Dukascopy tick data.

## Coverage

- First bar: 2021-06-29 00:00 UTC
- Last bar: 2026-06-28 23:59 UTC
- Rows: 1,772,172
- Duplicate timestamps: 0

## Repository Content

- `dataset/XAUUSD_1min.parquet` — complete compressed one-minute dataset
- `metadata/SOURCE_README.txt` — original detailed dataset documentation
- `metadata/daily_counts.csv` — daily bar counts
- `metadata/gaps_report.csv` — identified gaps and classifications
- `metadata/parquet_manifest.csv` — file metadata and schema
- `scripts/xauusd_dukascopy.py` — dataset-generation script

## Columns

- `timestamp_utc`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `source`

The raw Dukascopy `.bi5` tick archive remains on the T9 external drive.
