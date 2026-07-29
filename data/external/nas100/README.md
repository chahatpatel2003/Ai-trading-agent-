# NAS100 Tick Dataset

Historical NAS100 bid/ask tick data stored in monthly Parquet files, together with processed monthly datasets.

## Repository Content

- `samples/nas100_tick_sample.parquet` — representative raw tick sample
- `samples/nas100_processed_sample.parquet` — representative processed sample
- `metadata/parquet_manifest.csv` — file sizes, row counts and schemas
- `metadata/reports/` — source data-processing reports
- `metadata/download_state.json` — download state metadata
- `scripts/` — scripts used to download and process the dataset

## Raw Tick Schema

- `timestamp`
- `bid`
- `ask`
- `bid_volume`
- `ask_volume`

The complete raw, monthly and processed datasets remain on the T9 external drive.
