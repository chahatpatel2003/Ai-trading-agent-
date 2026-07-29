# NQ Research Platform — Architectural Blueprint (Phase 1)

> **Status:** design only. No strategy logic, no indicators, no execution, no
> broker connection in this phase. This document is the blueprint for a system
> expected to exceed 100,000 lines of production-quality code.
>
> **Working name of the package:** `nqlab` (NQ research laboratory).
>
> Read alongside [`DATA_PROFILE.md`](./DATA_PROFILE.md) (empirical ground truth)
> and [`ROADMAP.md`](./ROADMAP.md) (milestones & build order).

---

## 0. North Star — what "good" means here

The platform is a **quantitative research laboratory**, not a trading bot. Its
job is to *discover, validate, rank, and manage* NQ/MNQ strategies, and to be
ruthlessly honest about whether an edge exists.

**Optimization objective, in strict priority order:**

1. Long-term funded-account **survival**
2. **Consistent** monthly payouts
3. **Low probability** of account failure
4. **Stable** equity curve
5. **Low drawdown**
6. **Risk-adjusted** returns

Yearly return, win rate, R:R, biggest day → **reporting metrics only**. They are
never objective-function terms and never selection criteria.

**Two non-negotiable design axioms:**

- **A1 — No assumed edge.** The system never presumes any methodology works.
  Everything is objectively measured and statistically validated before it earns
  any weight. The default verdict on any candidate is *"no edge until proven."*
- **A2 — Survival is the objective, not a constraint.** The funded-account
  lifecycle (eval → funded → payouts, under Topstep-style trailing drawdown and
  daily-loss rules) is modeled explicitly, and *probability of survival* is the
  number we maximize. P&L is an input to that, not the target.

These axioms are why the **Risk Engine and the Validation Gauntlet are the
center of gravity** of the whole platform — not the strategies.

---

## 1. System Architecture (Deliverable 1)

### 1.1 Architectural style

**Hexagonal (ports & adapters) + layered, with a medallion data lake.** Pure
domain logic at the core; I/O, storage, vendors, and (later) brokers are
adapters behind interfaces. This is what makes "every component independently
testable" and "no research/live skew" achievable.

```
┌──────────────────────────────────────────────────────────────────────┐
│  INTERFACES        CLI (Typer) · notebooks · (later) dashboards/API     │
├──────────────────────────────────────────────────────────────────────┤
│  ORCHESTRATION     Pipelines / DAG runner  (ingest→curate→feature→      │
│                    backtest→validate→rank→portfolio→deploy)             │
├──────────────────────────────────────────────────────────────────────┤
│  APPLICATION       Services that compose domain + adapters:             │
│   SERVICES         DataService · FeatureService · BacktestService ·     │
│                    ValidationService · DiscoveryService ·               │
│                    PortfolioService · ExperimentService                 │
├──────────────────────────────────────────────────────────────────────┤
│  DOMAIN (pure)     instruments · symbology · sessions/calendar · bars · │
│   no I/O, no time  events · strategy contract · risk engine · portfolio │
│   no globals       cost model · metrics · statistics                    │
├──────────────────────────────────────────────────────────────────────┤
│  ADAPTERS / INFRA  storage (Parquet/DuckDB) · catalog · experiment      │
│                    tracker · artifact store · config · logging ·        │
│                    compute executor · (Phase 9) broker/exec · market feed│
└──────────────────────────────────────────────────────────────────────┘
        Dependency rule: arrows point DOWN only. Domain depends on NOTHING.
```

### 1.2 Core invariants enforced by the architecture

- **Determinism.** Same code + same config + same data version ⇒ byte-identical
  results. No wall-clock, `random`, or unordered iteration in logic; clocks and
  RNG are injected; seeds are config. Verified by golden tests.
- **Point-in-time correctness.** Every feature/label is a pure function of data
  available *at or before* its timestamp (`ts_recv`). Look-ahead is structurally
  prevented (as-of joins, availability stamps) and tested with controls.
- **Single source of truth for execution semantics.** The same `Strategy`
  interface, `RiskEngine`, and `CostModel` run in backtest, paper, and (Phase 9)
  live. No second "live" implementation can drift from research.
- **Immutability of raw + provenance everywhere.** `raw/` is read-only; every
  derived artifact carries (code SHA, config hash, data-version hash, seed).

---

## 2. Folder Structure (Deliverable 2)

`src/` layout, single installable package `nqlab`, bounded contexts as
sub-packages. (Phase 1 creates the **shape**; modules fill in over later phases.)

```
project-root/
├── pyproject.toml            # build, deps (pinned), tool config (ruff, mypy, pytest)
├── uv.lock / poetry.lock     # reproducible env lock
├── README.md
├── Makefile                  # task entry points (lint, test, profile, pipelines)
├── docs/                     # ARCHITECTURE.md, DATA_PROFILE.md, ROADMAP.md, ADRs/
│   └── adr/                  # Architecture Decision Records (one file per decision)
├── config/                   # configuration-as-data (NO logic)
│   ├── base.yaml             # defaults
│   ├── instruments/nq.yaml   # tick/point value/cycle/fees
│   ├── sessions/cme.yaml     # Globex hours, holidays, half-days, maint break
│   ├── roll/volume.yaml      # roll rules
│   ├── costs/default.yaml    # commissions, slippage, spread model
│   ├── risk/topstep_*.yaml   # account programs/sizes (eval, funded)
│   ├── validation/gauntlet.yaml  # CV scheme + pre-registered thresholds
│   ├── compute/local.yaml    # parallelism, memory budget, paths
│   └── profiles/{dev,research,ci}.yaml
├── src/nqlab/
│   ├── domain/               # PURE. no I/O, no time, no globals
│   │   ├── instruments/      # contract specs, symbology, roll, continuous
│   │   ├── calendar/         # sessions, holidays, session-relative time
│   │   ├── market/           # Event, Quote, Trade, Bar, Book value objects
│   │   ├── bars/             # time/tick/volume/dollar/imbalance bar logic
│   │   ├── strategy/         # Strategy/Signal contracts (interfaces only)
│   │   ├── risk/             # RiskEngine rules, AccountState, lifecycle
│   │   ├── portfolio/        # allocation, correlation, combined-risk
│   │   ├── costs/            # cost & slippage model interfaces + impls
│   │   ├── execution/        # order/fill value objects, matching semantics
│   │   ├── metrics/          # performance & survival metrics (pure)
│   │   └── stats/            # DSR, PBO, CPCV math, bootstraps, SPA
│   ├── data/                 # ADAPTERS for data
│   │   ├── catalog/          # dataset/partition/version registry (DuckDB/SQLite)
│   │   ├── ingest/           # DBN→Parquet (L0), schema contracts
│   │   ├── standardize/      # L0→L1 typed/clean views (mostly virtual)
│   │   ├── curate/           # L1→L2 continuous + bars + order-flow
│   │   ├── store/            # Parquet/DuckDB IO, partitioning, as-of joins
│   │   └── quality/          # data-contract checks, gap/anomaly reports
│   ├── features/             # feature store
│   │   ├── registry/         # versioned feature definitions + metadata
│   │   ├── compute/          # PIT feature computation
│   │   └── labels/           # triple-barrier / horizon labels + sample weights
│   ├── backtest/             # engines
│   │   ├── vectorized/       # fast screener
│   │   ├── event/            # event-driven high-fidelity simulator
│   │   ├── fills/            # fill/latency/queue models
│   │   └── result/           # BacktestResult schema (trades, equity, account)
│   ├── validation/           # the gauntlet
│   │   ├── splits/           # walk-forward, purged k-fold, CPCV
│   │   ├── tests/            # DSR, PBO, MinTRL, SPA, MC, regime, cost-stress
│   │   ├── survival/         # risk-of-ruin, P(pass eval), P(payout)
│   │   └── controls/         # positive/negative synthetic-strategy controls
│   ├── discovery/            # methodology-agnostic candidate search
│   │   ├── generators/       # manual, grid, GP, ML — all behind one interface
│   │   ├── harness/          # screen→backtest→validate→rank loop
│   │   └── registry/         # strategy registry + lifecycle states
│   ├── experiment/           # tracking & reproducibility
│   │   ├── tracker/          # run ledger (DuckDB) + lineage + trial accounting
│   │   └── artifacts/        # content-addressed artifact store
│   ├── services/             # application services (compose domain+adapters)
│   ├── pipelines/            # orchestrated DAGs (one module per pipeline)
│   ├── config/               # pydantic models + layered loader + hashing
│   ├── compute/              # executor abstraction (serial/process-pool/→dask)
│   ├── observability/        # structured logging, run context, metrics, timers
│   ├── cli/                  # Typer commands
│   └── deploy/               # Phase 9 ONLY: broker/exec adapters, live risk, monitor
├── tests/
│   ├── unit/                 # pure-domain + property-based (Hypothesis)
│   ├── integration/          # pipelines on a committed tiny fixture slice
│   ├── golden/               # regression hashes of pipeline/backtest outputs
│   ├── contracts/            # data-schema/quality contracts
│   └── controls/             # validator self-tests (positive/negative)
├── fixtures/                 # tiny committed data slices (1 day, few instruments)
└── scripts/                  # one-off operational scripts (profiling, backfills)
```

---

## 3. Module Breakdown (Deliverable 3)

Bounded contexts and their single responsibility:

| Module | Responsibility | Depends on |
|---|---|---|
| `domain.instruments` | Contract specs; `instrument_id↔symbol`; roll calendar; continuous (back-adjusted) construction | nothing |
| `domain.calendar` | Globex sessions, holidays, half-days, maintenance break; session-relative time; **daily-loss reset boundary** | nothing |
| `domain.market` | Value objects: Event, Quote, Trade, Bar, Book, Order, Fill | nothing |
| `domain.bars` | Aggregate events→bars (time/tick/volume/dollar/imbalance/run) with conservation invariants | market, calendar |
| `domain.strategy` | `Strategy` contract: `features(PIT) → target positions/orders` + param space + data-requirement declaration | market, instruments |
| `domain.risk` | RiskEngine: pre-trade checks, post-fill state, kill-switch, **funded-account lifecycle**; Topstep-style rules as composable predicates | market, calendar, instruments |
| `domain.portfolio` | Cross-strategy correlation, risk allocation, combined-risk vs account limits | risk, metrics, stats |
| `domain.costs` | Commission + slippage + spread-crossing + (optional) impact | market, instruments |
| `domain.metrics` | Sharpe/Sortino/Calmar/MAR/Ulcer/CVaR/drawdown/time-to-recovery (pure) | nothing |
| `domain.stats` | DSR, PBO, CPCV, Min-TRL, haircut Sharpe, White RC / Hansen SPA, bootstraps | nothing |
| `data.catalog` | Registry of datasets, partitions, versions, roll calendars, instrument & session reference | store |
| `data.ingest` | DBN→Parquet (L0); enforce raw schema contract; never mutate raw | store, quality |
| `data.standardize` | L0→L1 canonical typed view: decode enums, split outright/spread, attach metadata (mostly **virtual/lazy**, not materialized) | catalog, instruments |
| `data.curate` | L1→L2: continuous series + bar datasets + order-flow aggregates | standardize, bars, instruments |
| `data.store` | Partitioned Parquet/DuckDB IO; predicate pushdown; **as-of joins** | — |
| `data.quality` | Schema/dtype/null/monotonicity/gap/anomaly contracts; data-quality reports | store |
| `features.registry` | Versioned, declarative feature definitions (+inputs, lookback, data-req, PIT guarantees) | — |
| `features.compute` | PIT-correct feature panels (L3) | curate, registry, store |
| `features.labels` | Triple-barrier/horizon labels + sample weights (L4) | curate, store |
| `backtest.vectorized` | Fast approximate screen over many candidates | strategy, costs, metrics |
| `backtest.event` | High-fidelity event-driven sim; routes every order through RiskEngine | strategy, risk, costs, fills, market |
| `backtest.result` | Canonical `BacktestResult` (trades, fills, equity, account-state timeline, costs) | — |
| `validation.*` | Splits + statistical gauntlet + survival + controls | stats, metrics, backtest |
| `discovery.*` | Candidate generators + harness + strategy registry | backtest, validation, experiment |
| `experiment.*` | Run ledger, lineage, trial accounting, artifact store | config, store |
| `services.*` | Use-case orchestration combining domain + adapters | all of the above |
| `pipelines.*` | DAGs wiring services into reproducible runs | services |
| `config` | Pydantic models + layered YAML loader + canonical hashing | — |
| `compute` | Executor: map a transform over (instrument,date) partitions; swappable backend | — |
| `observability` | Structured logging, run-context propagation, timers, counters | — |
| `cli` | Operator entry points | services, pipelines |
| `deploy` | **Phase 9 only**: broker/exec adapter, live data feed, live risk loop, monitoring | strategy, risk, portfolio |

---

## 4. Dependency Graph (Deliverable 4)

Strict acyclic, top→down. Domain is a sink (depends on nothing); interfaces are
sources.

```
cli / notebooks
      │
      ▼
  pipelines ──────────────► services
      │                        │
      │            ┌───────────┼──────────────┬───────────────┐
      ▼            ▼           ▼               ▼               ▼
  experiment   validation   backtest      features          data
      │            │           │               │               │
      └────────────┴─────┬─────┴───────┬───────┘               │
                         ▼             ▼                       ▼
                       domain ◄──────── (pure)            store / catalog
                         ▲                                      │
                         └──────────────────────────────────────┘
   config · compute · observability  =  cross-cutting (used everywhere,
                                        depend on nothing in the app)
   deploy (Phase 9) sits beside services, reuses domain.strategy/risk/portfolio
```

Rules enforced in CI (import-linter): no module imports a layer above it; nothing
imports `deploy` except `cli`; `domain.*` imports only stdlib + `domain.*`.

---

## 5. Development Roadmap & Milestones (Deliverable 5)

Each milestone ships **working, tested, documented** capability. "Done" = code +
unit/integration tests + golden tests where applicable + docs + a CLI entry.

| Phase | Milestone | Exit criteria (gate) |
|---|---|---|
| **P0** | Foundations | Repo skeleton, config system, logging, CI (lint/type/test), data **catalog**, read-only L0 access, automated **DATA_PROFILE** regeneration. |
| **P1** | Data standardization | L0→L1 canonical typed view; symbology (`instrument_id↔contract`); outright/spread split; **sessions/calendar**; data-quality reports. Contract tests green on full history. |
| **P2** | Continuous & bars | Volume-based **roll calendar** + back-adjusted continuous; time/tick/volume/dollar/imbalance bars (L2); order-flow aggregates. Validated vs independent reference; conservation invariants tested. |
| **P3** | Feature store + labels | Versioned feature registry; PIT feature panels (L3); triple-barrier labels + sample weights (L4); **leakage controls** pass; availability matrix enforced. |
| **P4** | Backtest + cost + risk | Vectorized screener **and** event-driven simulator sharing the domain; cost model; **Risk Engine** with Topstep lifecycle; golden tests; same-engine sim/live guarantee proven by tests. |
| **P5** | Validation gauntlet | Walk-forward/purged-CV/CPCV; DSR/PBO/Min-TRL/SPA; MC & cost-stress; survival metrics; **positive/negative controls** pass; experiment tracker + reproducibility. |
| **P6** | Discovery harness | Strategy contract + generators (manual/grid/GP/ML) + screen→backtest→validate→rank loop + strategy registry + **trial accounting / DSR deflation**. (Still no claimed edge.) |
| **P7** | Portfolio layer | Correlation/independence analysis; drawdown-constrained allocation; combined-portfolio risk vs account limits; portfolio re-validated as a unit. |
| **P8** | Paper / replay | Deterministic market replay harness; "live-shaped" loop on historical data; monitoring + kill-switch dry-run. |
| **P9** | Topstep integration | Broker/exec adapter behind existing interfaces; live data feed; live risk loop = backtest risk loop; shadow → micro-size MNQ → scaled, governed by consistency rules. |

---

## 6. Build Order (Deliverable 6)

Strict bottom-up; nothing is built before what it depends on exists and is tested.

1. **Cross-cutting first:** `config` → `observability` → `compute` → `data.store` →
   `data.catalog`. (Everything needs these.)
2. **Domain primitives:** `instruments` → `calendar` → `market` → `metrics` → `stats`.
   (Pure, fully unit-tested in isolation.)
3. **Data layers:** `ingest`/contracts → `standardize` → `curate` (roll, continuous, bars).
4. **Features & labels:** `features.registry` → `features.compute` → `labels`.
5. **Execution semantics:** `costs` → `backtest.result` → `risk` → `backtest.event`
   → `backtest.vectorized`. (Risk before the event engine — the engine routes
   through it.)
6. **Validation:** `validation.splits` → `validation.tests` → `validation.survival`
   → `validation.controls`. (Build the judge before judging anything.)
7. **Reproducibility:** `experiment.tracker` + `artifacts` (wired into every run).
8. **Discovery & portfolio:** `strategy` contract → `discovery.harness` →
   `discovery.registry` → `portfolio`.
9. **Deployment:** `paper/replay` → `deploy` (Topstep) — **last**.

Guiding rule: **build the measuring instruments before the things being measured.**
The Risk Engine and Validation Gauntlet precede any strategy work.

---

## 7. Data Flow (Deliverable 7)

```
 Databento DBN (tmp/)                                      [vendor raw]
        │  ingest: decode, schema-contract, partition
        ▼
 L0 RAW Parquet (raw/)  ── IMMUTABLE ──────────────────────[source of truth]
        │  standardize (mostly VIRTUAL/lazy — not a 200GB copy):
        │   decode enums · attach contract meta · split outright/spread ·
        │   choose canonical clock (ts_recv) · validate sequence
        ▼
 L1 Standardized event view  (per instrument_id × date)
        │  curate: roll calendar → back-adjusted CONTINUOUS series;
        │          sample BARS (time/tick/volume/dollar/imbalance);
        │          order-flow aggregates (signed vol, OFI, quote imbalance)
        ▼
 L2 Curated: continuous + bars + flow   (materialized; ~GB not ~TB)
        │  features.compute (PIT, as-of on ts_recv)
        ▼
 L3 Feature panels   (versioned: feature-set hash × data version)
        │  labels: triple-barrier / horizon returns + sample weights
        ▼
 L4 Research datasets  →  walk-forward / purged splits
        │  strategy.signals → backtest (event-driven) + costs + RISK ENGINE
        ▼
 BacktestResult (trades, equity, account-state timeline, costs)
        │  VALIDATION GAUNTLET (CPCV, DSR, PBO, MC, cost-stress, survival)
        ▼
 Validation report  →  EXPERIMENT TRACKER (run ledger + lineage + trial count)
        │  ranking by SURVIVAL objective
        ▼
 STRATEGY REGISTRY (versioned, lifecycle states)
        │  portfolio construction (independence + combined risk)
        ▼
 PAPER / REPLAY  →  (Phase 9) TOPSTEP execution
```

Every arrow is a pure, deterministic, versioned transform. Re-running with the
same inputs reproduces identical outputs (golden-tested).

---

## 8. Storage Design (Deliverable 8)

**Medallion lake on Parquet, queried by DuckDB/Polars-lazy/PyArrow — no server DB
for bulk data.** A small DuckDB/SQLite **catalog** holds metadata only.

| Layer | Content | Partitioning | Mutability | Approx size |
|---|---|---|---|---|
| **L0 raw** | vendor Parquet (existing) | by dataset/half-year (as delivered) | **immutable, read-only** | ~220 GB |
| **L1 standardized** | canonical typed event view | `instrument_id` × `date` (Hive) | **mostly virtual** (lazy view); materialize selectively | ≈ raw if materialized → avoid |
| **L2 curated** | continuous series, bars, order-flow | `symbol`/`bar_type`/`date` | versioned by roll+sample config hash | ~GB–tens of GB |
| **L3 features** | PIT feature panels | `date` (+ feature-set) | versioned by feature-set × data version | ~GB |
| **L4 datasets** | labels, weights, splits | `date` | versioned | ~GB |
| **artifacts** | backtests, equity, reports | content-addressed (hash) | append-only | grows w/ research |
| **catalog** | manifests, versions, roll cal, instrument & session ref, run ledger | single DuckDB file | transactional | MB |

**Key storage decisions**

- **Do not duplicate the 200 GB MBP1 as a standardized L1.** Standardization is a
  lazy view over L0; only the small derived layers (L2+) are materialized.
  Materialize a slice of L1 only if repeated full-event scans become the
  bottleneck (decision recorded as an ADR when/if it happens).
- **Parquet tuning:** zstd, row-group size tuned for predicate pushdown on
  `(instrument_id, ts_recv)`; column projection everywhere; sorted within
  partition by `ts_recv, sequence` for as-of joins and event replay.
- **As-of joins** are a first-class store primitive (feature/label/quote-at-trade
  alignment) — implemented once, used everywhere, PIT-safe by construction.
- **Curated-layer physical location** is a config choice (T9 vs local NVMe). Full-
  history MBP1 passes are I/O-bound over USB → cache aggressively; compute is
  partition-incremental so re-runs touch only changed partitions.
- **Data versioning** = content hash of (inputs + transform code SHA + config
  hash) recorded in the catalog; layers are addressable by version so any
  experiment pins exactly which bytes it used.

---

## 9. Configuration System (Deliverable 9)

**Configuration-as-data; zero magic numbers in code.**

- **Models:** Pydantic v2 models for every config domain (paths, instruments,
  sessions, roll, costs, risk/Topstep, validation thresholds, compute). Validation
  at load time; illegal configs fail fast.
- **Layering:** `base.yaml` → `profiles/{env}.yaml` → per-experiment override,
  merged deterministically. Secrets (Phase 9 broker creds) via env/keychain,
  never in repo.
- **Canonical hashing:** the fully-resolved config is serialized canonically and
  hashed; the hash is stamped on every run/artifact → reproducibility + cache keys.
- **What lives in config (examples):** tick/point value/fees; session hours,
  holidays, **daily-loss reset boundary**; roll rule + offsets; slippage/cost
  params; **Topstep program rules** (trailing DD, daily loss, profit target,
  consistency %, min days) — *kept in config because these evolve and must never
  be hardcoded*; CV scheme + **pre-registered validation thresholds**;
  parallelism + memory budget + data roots.

---

## 10. Testing Strategy (Deliverable 10)

A pyramid, plus domain-specific test types that most trading codebases lack.

- **Unit (pure domain):** fast, no I/O. Plus **property-based (Hypothesis)** for
  invariants:
  - bar aggregation conserves volume/notional; OHLC ordering holds;
  - roll/continuous adjustment preserves *returns* (not levels) across the seam;
  - **Risk Engine never permits an account state that breaches a limit without a
    halt** (the single most important property test in the system);
  - PIT: no feature value depends on data with `ts_recv` > its own timestamp.
- **Data-contract tests:** schema/dtype/null/range/monotonic-sequence/gap checks
  on L0 (raw) and every derived layer (pandera/pydantic-style); CI fails on drift.
- **Integration:** full pipelines on a committed **tiny fixture** (1 day, a few
  instruments) — end-to-end ingest→features→backtest→validate.
- **Golden / regression:** hash canonical pipeline & `BacktestResult` outputs;
  unexplained change fails CI and must be acknowledged with an ADR.
- **Validator self-tests (controls):** the gauntlet must **reject** a synthetic
  random/overfit strategy and **accept** a synthetic known-edge strategy. If the
  judge can't tell, nothing it says is trustworthy.
- **Determinism tests:** same seed/config ⇒ identical output hash.
- **Performance tests:** throughput on a benchmark partition; guard regressions
  (this matters at 7–8 B rows).
- **Tooling:** `ruff` (lint+format), `mypy --strict`, `pytest` + coverage,
  `import-linter` (layer rules), pre-commit, CI matrix. Type hints are mandatory.

---

## 11. Validation Strategy (Deliverable 11) — the Gauntlet

This is where most "profitable" strategies die, and that is the point. Every
candidate runs the **same fixed gauntlet**, with thresholds **pre-registered**
before the lockbox test set is touched.

1. **Time-series splits (no shuffling, ever):** anchored + rolling walk-forward.
2. **Leakage control:** **purged k-fold + embargo** around label horizons
   (López de Prado) so overlapping labels can't leak.
3. **Robust generalization:** **Combinatorial Purged CV (CPCV)** → a *distribution*
   of OOS performance and the **Probability of Backtest Overfitting (PBO)**.
4. **Multiple-testing correction (because discovery tests thousands of candidates):**
   **Deflated Sharpe Ratio** (using the trial count from the experiment tracker),
   **Minimum Track-Record Length**, **haircut Sharpe** (Harvey–Liu), and
   **White's Reality Check / Hansen's SPA** for data-snooping over the search.
5. **Robustness:** parameter-sensitivity surfaces (reward *plateaus*, punish
   spikes); Monte-Carlo trade-order & block bootstrap of returns; perturbation
   (jitter fills, shift latency); **regime-conditional** performance (vol/trend/
   session).
6. **Cost stress:** re-run with pessimistic commissions/slippage/spread; edge must
   survive realistic and adverse costs.
7. **Survival-centric metrics (the objective):** drawdown distribution, time-to-
   recovery, CVaR/Ulcer, **risk-of-ruin / P(account failure)**, **P(pass eval)**,
   **P(monthly payout)**, MAR/Calmar — all under the Topstep rule set.
8. **Gate:** a candidate is "validated" only if it clears the pre-registered
   thresholds on a **lockbox test set touched exactly once**. No goalpost moving.

---

## 12. Experiment Tracking & Reproducibility (Deliverable 12)

Every run is an immutable, queryable record.

- **Run record:** `run_id`, git SHA, **config hash**, **data-version hashes**
  (which L1/L2/L3/L4 snapshots), **seed**, env-lock hash, timestamps, status,
  metrics, artifact pointers, **parent lineage**, and **trial registration**
  (so DSR/SPA know how many strategies were tried — central to honest stats).
- **Store:** local DuckDB run ledger + **content-addressed artifact store** on
  disk (equity curves, trade ledgers, validation reports, plots). Optional
  MLflow-compatible export. No external service required.
- **Leaderboard:** query runs by survival objective; full lineage from raw data
  version → features → strategy version → validation report.
- **Reproducibility guarantee:** re-executing a `run_id` reproduces identical
  result hashes (enforced by a golden test). This is what makes research auditable.

---

## 13. Feature Store Design (Deliverable 13)

- **Feature = a versioned, pure, declarative definition:** `f(past window) → value`
  registered with metadata: inputs, lookback, **data requirement** (e.g. "needs
  MBP1 ⇒ only valid ≤2025-H1"), compute cost, and **PIT guarantee**.
- **Offline store (now):** partitioned Parquet panels keyed by
  `(continuous_symbol, ts_recv)`, computed deterministically, PIT-correct via
  as-of joins on availability time. Versioned by feature-set hash × data version.
- **Leakage guards:** enforced by timestamp contracts + the PIT property test; a
  feature physically cannot read a row whose `ts_recv` exceeds its own.
- **Availability matrix:** every feature tagged by data dependency; the system
  refuses to build a microstructure feature on a period without MBP1 (see
  DATA_PROFILE gap), and the dataset assembler propagates availability so a
  strategy's usable history is computed, not assumed.
- **Online store (interface now, impl Phase 9):** same feature definitions served
  in real time for live trading — guaranteeing train/serve parity.

---

## 14. Strategy Versioning Design (Deliverable 14)

- **Strategy spec = immutable record:** name, **semver**, code-module hash, param
  values, feature-set version, data version it was validated on, and the
  validation-report id that justified its state.
- **Registry & lifecycle states:** `proposed → screened → validated → candidate →
  portfolio → paper → live → retired`. Transitions require passing the relevant
  gate and are logged with lineage.
- **Semver rules:** param-only change ⇒ minor/patch; logic change ⇒ major (and
  full re-validation). A live strategy is *pinned* to an exact spec hash; nothing
  ambiguous can reach execution.
- **Decay monitoring (Phase 8+):** live performance is continuously compared to
  the validated distribution; drift triggers re-validation or retirement.

---

## 15. Research Pipeline (Deliverable 15)

The end-to-end DAG, orchestrated and reproducible:

```
profile-data → standardize → curate(roll,bars,flow) → features → labels →
assemble-datasets(walk-forward) → [generate candidates] → screen(vectorized) →
backtest(event-driven + risk + costs) → validate(gauntlet) →
track(experiment + trial count) → rank(survival objective) →
register(strategy) → [portfolio construct] → report
```

- Each node is a **service** with typed inputs/outputs, cached by version hash
  (unchanged upstream ⇒ skip recompute).
- **Partition-incremental:** only changed `(instrument, date)` shards recompute.
- Runs are launched via CLI or pipeline modules; every run lands in the tracker
  with full lineage. A "research question" maps to one configured pipeline run.

---

## 16. Backtesting Pipeline (Deliverable 16)

**Two engines, one domain — no sim/live skew.**

- **Vectorized screener:** bar-resolution, approximate fills, simple costs. Sweeps
  thousands of candidates cheaply to triage before expensive validation. Used for
  *ranking-to-shortlist only*, never for final verdicts.
- **Event-driven simulator (the truth engine):** replays the L1/L2 event stream in
  strict `(ts_recv, sequence)` order; models latency, queue position, partial
  fills, spread-crossing, slippage, fees; **routes every order through the same
  Risk Engine used live**; emits the canonical `BacktestResult` (trades, fills,
  positions, equity, **account-state timeline**, costs).
- **Walk-forward orchestration:** fit/select on in-sample, evaluate OOS, roll,
  with purge+embargo at boundaries. Deterministic (seeded, no wall clock).
- **Outputs feed validation directly;** the `BacktestResult` schema is the single
  contract between backtest and the gauntlet, the tracker, and (later) live
  reconciliation.

---

## 17. Risk Engine Architecture (Deliverable 17) — the heart

The Risk Engine is a **pure, deterministic rules engine over an `AccountState`**,
shared identically by backtest, paper, and live. It encodes the *objective*.

- **Pre-trade gate:** every prospective order is checked against position/contract
  limits, max order size, and projected limit breaches → allow / resize / reject.
- **Post-fill update:** maintains realized+unrealized equity, peak equity, and all
  program counters; raises **kill-switch events** (halt-for-day / flatten) when a
  rule trips.
- **Topstep-style funded-account lifecycle (config-driven, because rules evolve):**
  - **Trailing max drawdown** (peak-equity-based; intraday vs EOD per program),
  - **Daily loss limit** (resets at the configured session boundary),
  - **Profit target** (eval), **minimum trading days**,
  - **Consistency rule** (no single day > X% of total profit),
  - **scaling plan** / max contracts by stage.
- **Lifecycle simulation = the objective-function generator:** the engine can
  simulate the full path *eval → funded → payouts* over many Monte-Carlo
  realizations, yielding **P(pass eval)**, **P(account failure)**, **expected
  months survived**, and **payout cadence** — the exact quantities the platform
  maximizes (per axiom A2).
- **Composable, independently testable predicates:** each rule is a small pure
  predicate with its own property tests; the invariant "no breach without a halt"
  is proven, not hoped for. Rule sets are config so different programs/account
  sizes are modeled without code changes.

---

## 18. Strategy Discovery Architecture (Deliverable 18)

**Methodology-agnostic by construction (axiom A1).** The platform never assumes
*how* an edge arises.

- **Strategy contract (the only thing discovery must satisfy):**
  `features(PIT) → target positions/orders`, plus declared parameter space and
  **data requirements**. Rules, ML, genetic programming, ensembles — all are just
  different implementations of this one interface.
- **Candidate generators (pluggable):** manual hypotheses, parameter grids,
  genetic/symbolic search, feature→label ML pipelines, stacked ensembles.
- **Harness loop:** `generate → vectorized screen → event-driven backtest →
  validation gauntlet → rank by survival objective → register`. Every candidate,
  no matter its source, runs the *identical* gauntlet.
- **Anti-overfitting governance (first-class, not bolted on):**
  - the **experiment tracker counts every trial**, feeding **DSR / SPA** deflation;
  - **hypotheses are pre-registered**; thresholds fixed before the lockbox test;
  - the **lockbox test set is touched exactly once**.
- **Portfolio assembly (with `domain.portfolio`):** select **statistically
  independent** validated strategies (low correlation, distinct regimes), allocate
  under **combined** Topstep risk, and **re-validate the portfolio as a single
  unit** — survival of the *book*, not of any one strategy, is what matters.

---

## 19. Deployment Roadmap — ending at Topstep (Deliverable 19)

Deployment is the **last** phase, reached only after the research lab is trusted.

1. **Deterministic replay / paper (P8):** run validated strategies through the
   *same* event engine + Risk Engine on historical and then near-real-time data;
   dry-run monitoring, alerting, and kill-switch.
2. **Broker/execution adapter (P9):** implemented behind the existing
   `domain.execution` + `RiskEngine` interfaces, so live reuses research code
   paths verbatim (no second implementation can drift).
3. **Shadow mode:** live market data, simulated fills, live Risk Engine — compare
   against backtest expectations (reconciliation report).
4. **Micro-size live (MNQ):** smallest size, single strategy, **live risk loop =
   backtest risk loop**, full monitoring + automated flatten/halt.
5. **Governed scale-up:** increase size/strategies only as the **consistency rule**
   and survival metrics permit; add independent strategies as a re-validated
   portfolio.
6. **Continuous re-validation & decay detection:** live results compared to the
   validated distribution; drift ⇒ auto-retire/re-validate. The lab keeps running;
   deployment is just one consumer of it.

---

## Open decisions to confirm before Phase 0 build

These genuinely gate the build (defaults proposed; override as you like):

1. **Stack:** Python 3.12 + Polars(lazy) + DuckDB + PyArrow for data; Pydantic v2
   config; Typer CLI; pytest/Hypothesis/mypy-strict/ruff. Numba/Rust reserved for
   the event-engine hot loop if profiling demands. *Default: yes.*
2. **Canonical clock:** `ts_recv` for information-availability (look-ahead-safe),
   `ts_event`+`sequence` for intra-instrument ordering. *Default: yes.*
3. **Roll method:** volume-based roll with calendar guard, back-adjusted continuous
   for research, raw per-contract kept for execution realism (OI unavailable in
   this feed). *Confirm — it touches every downstream layer.*
4. **Don't materialize a 200 GB L1.** Keep standardization virtual; materialize
   only L2+ (bars/flow/features). *Confirm storage philosophy.*
5. **Curated storage location/budget:** same T9 SSD vs local NVMe (affects I/O
   throughput on full-history passes). *Need your preference + free-space target.*
6. **Directory naming:** project lives in `~/Desktop/project usd ` —
   trailing space + "usd" misnomer for an NQ project. *Recommend renaming to
   `nq-research-platform/` (package `nqlab`).*
7. **MBP1 gap (2025-H2, 2026-H1):** confirmed handled as a data-availability
   contract, not backfilled. *Confirm acceptable; microstructure research caps at
   2025-H1.*
```
