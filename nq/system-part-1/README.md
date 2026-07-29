# nq-research-platform (`nqlab`)

A quantitative **research laboratory** for NQ/MNQ futures. Its purpose is to
discover, validate, rank, and manage strategies under a strict **survival-first**
objective. This is **not** a trading bot — execution and Topstep integration are
the last phase.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full blueprint,
[`docs/DATA_PROFILE.md`](docs/DATA_PROFILE.md) for the dataset ground truth, and
[`docs/ROADMAP.md`](docs/ROADMAP.md) for the build plan.

## Status

**Phase 1 — first vertical slice to executable strategy testing.** On top of the
Phase 0 foundation: a bars engine (trades → OHLCV, session-labelled), volume-roll
continuous contracts, a PIT-safe feature store, a strategy contract with four
baseline *test-vehicle* strategies, an event-driven backtester (stops/targets/
time-stop/EOD-flatten, NQ→MNQ sizing), a configurable Topstep account simulator
(50K/100K/150K), ranked reports, and CLI commands for the whole loop.

> Data note: this dataset is **NQ E-mini only** (no MNQ micro contracts). Since
> MNQ and NQ share the same Nasdaq-100 price, the NQ→MNQ sizing fallback runs on
> the NQ price series using MNQ's point value — no separate MNQ bars needed.

Baselines are test vehicles to prove the engine, **not** validated strategies.
Still no live trading, replay, ML, or Topstep API.

## CLI

```bash
nqlab doctor                                   # environment / data health
nqlab data profile | data manifest             # raw data (read-only)
nqlab bars build 2024-01-02 2024-01-15 --continuous   # trades -> bars (+ continuous)
nqlab bars profile                             # list curated bars
nqlab features build NQ.c 2024-01-02 2024-01-15
nqlab backtest run pdhl NQ.c 2024-01-02 2024-01-15 --account 50k
nqlab research baseline NQ.c 2024-01-02 2024-01-15 --account 50k
nqlab topstep simulate <equity.parquet> --account 150k
```

## Setup

```bash
make install        # uv venv (3.12) + editable install + importability shim
```

`make install` is the supported path. It creates `.venv`, installs `nqlab` plus
dev tooling, and copies `scripts/sitecustomize.py` into the venv. The shim is
needed because the `uv`/`python-build-standalone` interpreter on this machine
intermittently fails to apply the editable-install `.pth` file; `sitecustomize`
puts `src/` on the path reliably (see that file's docstring). After it, the bare
`nqlab` command works without activating the venv. All commands are read-only
with respect to `raw/` market data; derived bars/features/artifacts go under
`curated_root`.

## Development

```bash
make test     # pytest
make lint     # ruff
make type     # mypy --strict
make check    # lint + type + test
```

## Data

Raw market data lives at `/Volumes/T9/nq data` (configurable). It is **immutable**;
nothing in this project writes to `raw/`. Derived layers go under a separate,
configurable `curated_root` (default on the T9 SSD, movable to local NVMe).
