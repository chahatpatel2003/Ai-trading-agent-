# ROADMAP — NQ Research Platform

> Companion to [`ARCHITECTURE.md`](./ARCHITECTURE.md). Milestones ship working,
> tested, documented capability. The governing rule: **build the measuring
> instruments (Risk Engine, Validation Gauntlet) before the things measured
> (strategies).** Topstep integration is dead last.

## Phase gates

| Phase | Theme | Ships | Gate to pass |
|---|---|---|---|
| **P0** | Foundations | config, logging, compute executor, store, catalog, CI, data-profiler | CI green; DATA_PROFILE auto-regenerates; read-only L0 verified |
| **P1** | Standardization | L0→L1 view, symbology, outright/spread split, sessions/calendar, quality reports | data contracts green on full history |
| **P2** | Continuous & bars | volume roll + back-adjusted continuous, time/tick/volume/dollar/imbalance bars, order-flow | conservation + roll-continuity property tests; validated vs reference |
| **P3** | Features & labels | feature registry, PIT panels (L3), triple-barrier labels + weights (L4) | leakage/PIT controls pass; availability matrix enforced |
| **P4** | Backtest + risk | vectorized screener, event-driven sim, cost model, **Risk Engine + Topstep lifecycle** | golden tests; "no breach without halt" property test; sim==live engine |
| **P5** | Validation | walk-forward/purged/CPCV, DSR/PBO/Min-TRL/SPA, MC, cost-stress, survival, controls, experiment tracker | positive & negative controls behave; reproducibility golden test |
| **P6** | Discovery | strategy contract, generators, harness, registry, trial accounting | end-to-end candidate→ranked, deflation wired |
| **P7** | Portfolio | independence analysis, drawdown-constrained allocation, combined risk | portfolio re-validated as a unit |
| **P8** | Paper/replay | deterministic replay, live-shaped loop, monitoring + kill-switch dry-run | reconciliation vs backtest within tolerance |
| **P9** | Topstep | broker/exec adapter, live data, live risk loop, shadow→micro→scale | live risk loop == backtest risk loop; shadow reconciles |

## Build order (dependency-sorted checklist)

1. Cross-cutting: `config` → `observability` → `compute` → `data.store` → `data.catalog`
2. Domain primitives: `instruments` → `calendar` → `market` → `metrics` → `stats`
3. Data: `ingest`/contracts → `standardize` → `curate` (roll, continuous, bars, flow)
4. Features: `features.registry` → `features.compute` → `labels`
5. Execution semantics: `costs` → `backtest.result` → `risk` → `backtest.event` → `backtest.vectorized`
6. Validation: `splits` → `tests` → `survival` → `controls`
7. Reproducibility: `experiment.tracker` + `artifacts`
8. Discovery & portfolio: `strategy` → `discovery.harness` → `registry` → `portfolio`
9. Deployment: `paper/replay` → `deploy` (Topstep) — last

## Explicit non-goals for Phase 1

- No strategy logic, no indicators, no signals.
- No execution code, no broker/Topstep connection.
- No assumption that any methodology has an edge.
- No modification of `raw/` data, ever.
```
