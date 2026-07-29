# Phase 15 — Forward Paper/Shadow Infrastructure for the Frozen bos_5m_trigger

**Goal:** build the forward paper/shadow system around the single best candidate
(`bos_5m_trigger`, Phase 14) so it can collect new, unseen trades and *prove whether the
edge is real* — **without changing its rules, without live orders, without real money.**
This phase does not claim the strategy works; it builds the rig to find out.

**Status: built, verified, and exercised — verdict PENDING_FORWARD_SAMPLE (by design).**
The frozen strategy loads and hash-verifies, the paper engine replays cleanly, the
ledger writes the full schema, there is no live-order path, and the report carries every
practical stat. The forward verdict stays `PENDING_FORWARD_SAMPLE` because there are 0
genuinely-forward trades yet (no live feed, no post-2026 data) — replay trades are
explicitly excluded from confirmation.

---

## The frozen candidate (exact, hash-enforced)

Loaded from the Phase-14 registry and verified by strategy fingerprint:

```
bos_5m_trigger (5m, 50K, MNQ/mnq_fixed1): LONG when the 4h bias is bullish AND a
break of structure prints with a displacement candle (SHORT mirror); NY-AM only
(09:30-12:00 ET); stop 1.0xATR, target 4.0xATR; min-hold 3 bars, time-stop 48 bars;
risk $150; locked-buffer Topstep 50K.  [frozen hash bedfb200abf675b9, source phase14]
```

`PaperConfig.verify()` recomputes the fingerprint and **refuses to run if a single
parameter changed** (tested). Account = 50K (best Topstep pass odds). Note the frozen
sizing is `mnq_fixed1` (one MNQ contract) — the sizing-repair mode from Phase 14; it is
loaded *exactly as frozen*, not modified to the prose "mnq_min1".

## What was built

1. **Frozen runner** (`paper/frozen.py`) — load + fingerprint-verify the Phase-14 config,
   refuse on any modification, and emit an exact human-readable signal explanation.
2. **Paper/shadow engine** (`paper/engine.py`) — runs the frozen strategy through the
   *same* locked-buffer Topstep backtester (entries, stops, targets, EOD-flatten,
   conservative slippage/commission, real-account `gate_entries=True`). **No orders are
   ever placed.**
3. **Market-data adapter interface** (`paper/feed.py`) — a `MarketDataFeed` protocol, a
   real 1m→5m bar builder (`build_5m_from_1m`, `ts_recv` clock), a `ReplayFeed`
   (historical), and a `LiveFeed` stub that **refuses** (no broker, no execution API).
4. **Forward ledger** (`paper/ledger.py`) — append-only JSONL; every trade records ts,
   strategy id, frozen hash, mode, setup reason, entry/stop/target/exit, exit reason,
   PnL, MAE/MFE, account equity + blown state, rule violations, and a
   `followed_frozen_rules` flag.
5. **Forward evaluator** (`paper/evaluate.py`) — `PENDING_FORWARD_SAMPLE` /
   `FORWARD_ACCEPT_FOR_SHADOW` / `FORWARD_REJECT`; DSR at the frozen candidate count
   (n=1); full practical stats; comparison vs Phase-14. Replay trades never count.
6. **CLI** — `paper init`, `paper replay`, `paper run`, `paper status`, `paper report`,
   `paper evaluate`.

## Replay (system test — historical, NOT forward confirmation)

`paper replay` over 2025-2026 wrote **478 trades** to the ledger; the engine + ledger +
report all work end-to-end. Replay stats (50K, locked-buffer, conservative costs):

| | replay (2025-26) | Phase-14 ref |
|---|---|---|
| trades | 478 | 479 |
| win rate | 24% | — |
| profit factor | **1.177** | 1.155 |
| expectancy | $10.03 | $8.94 |
| net | +$4,795 | +$4,280 |
| monthly / ROI | $267 / 0.53% mo, 6.4% yr | — |
| max DD / avg DD | $1,807 / $771 | — |
| P(pass 50K) | **67.6%** (100K 31%, 150K 0%) | 62% |
| Topstep pass-rate | 68% | — |
| ruin (50K reshuffle) | 32% | 1% (150K) |
| DSR (frozen count n=1) | **0.913** | 0.44 (n=6) |

A note on that **DSR 0.913**: as a *single pre-registered* hypothesis (n=1, so the
multiple-testing penalty vanishes), the candidate's historical Sharpe clears the 0.90
bar. That is exactly why a forward test is warranted — but this is **replay on
already-seen data**, so it is **not** confirmation. The 32% reshuffle ruin at 50K is the
honest counterweight: this is a thin, wide-target edge, fine for shadow study, not for
real capital.

## Verdict

```
$ nqlab paper status
forward trades: 0/150 | replay trades: 478
verdict: PENDING_FORWARD_SAMPLE — only 0 genuine forward trades (need >= 150);
         478 replay trades do not count toward confirmation
```

`PENDING_FORWARD_SAMPLE` is the correct and only honest verdict: no live feed is
connected and there is no post-2026 data, so the genuine forward sample is empty. The
system will render `FORWARD_ACCEPT_FOR_SHADOW` / `FORWARD_REJECT` only once ≥150 real
forward trades accumulate.

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **215 tests**, incl. 9 paper tests).
- ✅ Frozen strategy loads and verifies (hash `bedfb200abf675b9`); modified config is
  rejected.
- ✅ Paper replay works (478 trades); ledger writes the full schema correctly.
- ✅ **No live-order capability exists** — `LiveFeed` refuses, and a source scan asserts
  no order-execution identifiers anywhere in the `paper` package.
- ✅ Paper report includes all practical stats (win/PF/expectancy/ROI/DD/ruin/Topstep
  pass + P(pass 50/100/150K)/days-to-pass/consistency/largest-day/DSR + Phase-14 compare).
- ✅ Verdict remains `PENDING_FORWARD_SAMPLE` until enough genuine forward trades exist.

## Honest framing

> This phase did **not** claim the strategy works. It built the forward paper/shadow
> rig around the one best candidate, exactly as frozen, with no order execution. The
> replay confirms the rig and shows the candidate is promising in isolation (single-
> hypothesis DSR 0.913, P(pass 50K) ~68%) — but promising-on-seen-data is not proof. The
> only path to `FORWARD_ACCEPT_FOR_SHADOW` is genuine forward trades from a live/shadow
> data feed (to be connected via the `MarketDataFeed` interface), and even that earns
> *shadow* status, not live capital. Until then: **PENDING_FORWARD_SAMPLE.**

---

## Reproduction

```
nqlab paper init                 # load + hash-verify the frozen bos_5m_trigger
nqlab paper run                  # (no live feed connected; refuses, points to replay)
nqlab paper replay               # replay 2025-2026 through the paper engine -> ledger
nqlab paper status               # forward-sample progress + verdict
nqlab paper report               # full practical stats (forward + replay) + Phase-14 compare
nqlab paper evaluate             # forward verdict
```
