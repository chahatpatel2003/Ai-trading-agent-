# AI Trading Agent: Development Archive (2026)

This is where my systematic-trading work started. In July 2026 I collected my first NQ futures and gold projects here in one repository. They have since grown into eight dedicated, fully documented repositories, each with its own tests, results and write-up.

This repository is a **historical archive** of that starting point, created in October 2026 from the original `Ai-trading-agent-` repository:

- its commit history is a cleaned copy of the original, showing how those projects first came together;
- its current tree holds the one piece of work that lives nowhere else: the **market-data engineering** used to build the datasets behind the research.

## Where the projects live now

| Project | Repository | Started here as |
|---|---|---|
| NQ research platform: 59 research phases, Topstep account simulator | [nq-trading-system](https://github.com/chahatpatel2003/nq-trading-system) | `nq/system-part-1/` |
| Leakage-proof Topstep strategy discovery for NQ and GC | [nq-topstep-system](https://github.com/chahatpatel2003/nq-topstep-system) | `nq/topstep/` |
| Risk-first XAUUSD paper-trading agent with a learning loop | [gold-trading-agent](https://github.com/chahatpatel2003/gold-trading-agent) | `gold/agent/` |
| Seven pre-registered NQ strategy tests | [nq-strategy-research](https://github.com/chahatpatel2003/nq-strategy-research) | n/a |
| Pre-open NQ price-zone research with sealed holdouts | [nq-daily-levels](https://github.com/chahatpatel2003/nq-daily-levels) | n/a |
| Leak-safe NQ/GC framework that re-audits earlier results | [nqgc-research](https://github.com/chahatpatel2003/nqgc-research) | n/a |
| 44 pre-registered supply/demand zone tests in gold futures | [gc-intraday-zones](https://github.com/chahatpatel2003/gc-intraday-zones) | n/a |
| Strategy discovery ranked by payout-before-blow-up | [topstep-quant](https://github.com/chahatpatel2003/topstep-quant) | n/a |

## What is still here: market-data engineering ([`data/`](data/README.md))

- **[NAS100 tick pipeline](data/nas100/README.md)** (`data/nas100/scripts/`), an autonomous, resumable downloader for Dukascopy NAS100 tick data since 2012:
  - decodes Dukascopy's LZMA-compressed `.bi5` hourly files (20-byte big-endian records);
  - writes monthly Parquet files;
  - validates every month against 18 structural checks, re-downloading on failure with back-off;
  - keeps crash-safe progress in `download_state.json`;
  - builds bid/ask/mid candles for 11 timeframes plus tick bars;
  - runs under a self-healing PowerShell watchdog.
- **[XAU/USD pipeline](data/xauusd/README.md)** (`data/xauusd/scripts/xauusd_dukascopy.py`) builds a cleaned 5-year 1-minute OHLCV dataset from Dukascopy ticks, with daily bar counts and a classified gap report (`data/xauusd/metadata/`).
- **[NQ futures data notes](data/nq/README.md)** document the Databento MBP-1 and trades schemas, with a manifest of the half-year files (2021 → 2026).
- **Quality evidence:**
  - one validation report per NAS100 month (`data/nas100/metadata/reports/`);
  - Parquet manifests with row counts and schemas.

The datasets themselves are **not** included. They are licensed or redistribution-restricted, and far too large: about 11 GB of NAS100 ticks, 220 GB of NQ data and 1.3 GB of XAU/USD.

## History

The commits from 28–29 July 2026 show the projects being assembled:

1. the gold trading agent;
2. the NQ Topstep system;
3. the NQ research platform;
4. Git LFS storage;
5. the project README;
6. the documented external datasets (pull requests #1 and #2 in the original repository).

Browse any older commit to see those early snapshots.

**October 2026 cleanup.** This archive's history is a cleaned copy of the original repository's history. Material that should not be public was removed from every commit:

- third-party course PDFs and images;
- licensed market data and broker captures;
- a third-party MetaTrader expert advisor;
- Pine scripts whose authorship could not be verified;
- private trade logs.

Commit e-mails were changed to GitHub's no-reply address. Commit messages, authors and dates are unchanged. Four early commits that contained only removed material do not appear here. Files that now live in the dedicated repositories were removed from the current tree, but remain in this history. The complete original history is preserved privately.

## What the research found

These outcomes are reported in full, including the failures, in each repository's own README:

- **[nq-strategy-research](https://github.com/chahatpatel2003/nq-strategy-research):** seven pre-registered NQ hypotheses; all seven were rejected by their own pre-registered gates.
- **[nq-trading-system](https://github.com/chahatpatel2003/nq-trading-system):** across 59 research phases, one strategy passed forward-paper conditions (out-of-sample profit factor 1.17–1.20). The robust income ceiling on a Topstep account was about $400 per month, short of the $1,000 target.
- **[nq-daily-levels](https://github.com/chahatpatel2003/nq-daily-levels):** the central hypothesis was rejected and both sealed holdouts failed. One real resistance-location effect survived, but it did not convert into a tradable edge.
- **[nq-topstep-system](https://github.com/chahatpatel2003/nq-topstep-system):** 13 discovery passes; no candidate met the win-rate, frequency and profitability requirements at the same time.
- **[nqgc-research](https://github.com/chahatpatel2003/nqgc-research):** showed that an earlier +0.19 to +0.31 R per-trade result came from a chronology leak; the corrected strategy had no edge.
- **[gc-intraday-zones](https://github.com/chahatpatel2003/gc-intraday-zones):** 44 pre-registered zone hypotheses; 32 failed outright, and one finding (ZON-028) survived a selection-bias null but has not been validated.
- **[topstep-quant](https://github.com/chahatpatel2003/topstep-quant):** nothing survived out-of-sample validation; the strongest candidate failed its forward holdout.
- **[gold-trading-agent](https://github.com/chahatpatel2003/gold-trading-agent):** a paper-trading system with live trading gated off; its learning loop graded 27 of its own predictions (11 correct, 4 partial, 12 wrong) and showed no usable edge.

## Third-party material

- **Dukascopy** and **Databento** are the data sources described in `data/`. None of their data is included, and their terms apply to anyone who downloads it.
- Topstep, TradingView and MetaTrader are trademarks of their respective owners. This project is not affiliated with them.
- Open-source libraries are used under their own licences.

© Chahat Patel. All rights reserved. This notice covers the original code and documentation in this repository. It does not extend to the third-party material listed above. No license to reuse this code is granted; it is published for review.
