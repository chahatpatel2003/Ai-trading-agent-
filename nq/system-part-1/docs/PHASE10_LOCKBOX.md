# Phase 10 — Pre-registered Lockbox Confirmation of the Phase-9 Survivors

**Question:** Phase 9's HTF asymmetric setups cleared the economic gauntlet (129
candidates passed OOS + ruin + cost-stress + Topstep + walk-forward) but **0 ACCEPT**
— they failed the Deflated-Sharpe (DSR) correction for searching 176 variants. Were
they a **real edge** or **post-search winners**? Phase 10 answers this the only honest
way: **freeze the hypotheses first, then test on held-out data, then recompute
significance with the frozen trial count.**

**Verdict: not confirmed. 0 ACCEPT, 1 REJECT, 2 NEEDS_PAPER_LOCKBOX.** The one frozen
survivor that traded enough on the lockbox to be judged (`opening_drive` London-range
break, 93 trades) is **positive but statistically insignificant even at the frozen
count of 3** (DSR 0.30 ≪ 0.90) → REJECT. The other two fire too rarely on a one-year
lockbox to judge → NEEDS_PAPER_LOCKBOX. NQ only; no ES; no parameter tuning after
freeze; nothing force-accepted.

---

## Method (and one decisive honesty caveat)

1. **Freeze** the top Phase-9 gauntlet-survivors into an immutable registry
   (`phase10_frozen.json`). Each frozen hypothesis must reproduce its Phase-9 strategy
   fingerprint exactly — the "no tuning after freeze" guarantee (a tampered parameter
   is detected by a hash mismatch on load).
2. **Test** on the lockbox window (2025-H2 → 2026-H1, to data end 2026-06-17), split
   in-sample (H2-2025, for the Sharpe estimate) / out-of-sample (H1-2026, the gate).
3. **Re-deflate** the Sharpe with `n_trials = 3` (the frozen count), not 176.

> **The lockbox is not pristine.** Phase 9 ran its out-of-sample window through 2026,
> so 2025-H2 / 2026-H1 were *already observed by the Phase-9 selection*. By the Phase-10
> acceptance rule ("no candidate is accepted unless it passes the *untouched* lockbox"),
> ACCEPT is therefore impossible here — the strongest a pass can earn is
> NEEDS_PAPER_LOCKBOX (confirm on genuinely forward data). The code makes ACCEPT
> reachable only when `pristine=True`; it is `False` for this run.

**Freeze selection (fixed rule, applied once, pre-registration):** the gauntlet-
surviving variant of each headline setup at the smallest risk that still *sizes to a
tradeable position in the higher-ATR 2026 regime*. At risk \$25–\$50 a 1.5×ATR stop
rounds to ~0 MNQ contracts in 2026 (the \$50 `opening_drive` London variant takes only
**3** trades in all of 2026-H1), so those variants cannot be tested at all; risk \$100
trades normally (**93** trades) and is used.

## Frozen hypothesis registry

| Name | Family | TF | Acct | Phase-9 OOS | Phase-9 DSR (n=176) | config hash |
|---|---|---|---|---:|---:|---|
| opening_drive_london_5m | opening_drive (London break) | 5m | 150k | +4,592 | 0.00 | ea0e38697b… |
| compression_expansion_or_5m | compression_expansion (OR break) | 5m | 50k | +2,899 | 0.17 | 983deaed73… |
| opening_drive_on_5m | opening_drive (overnight break) | 5m | 100k | +379 | 0.34 | 20993de318… |

---

## Result 1 — the lockbox (held-out, frozen n=3)

| Frozen hypothesis | OOS | Ruin | Blown | Cost | WF+ | Pos-yr | Trades | DSR(n=3) | Verdict |
|---|---:|---:|:--:|:--:|---:|---:|---:|---:|---|
| opening_drive_london | +730 | 0% | no | pass | 56% | 100% | **93** | **0.30** | **REJECT** |
| compression_expansion_or | −162 | 0% | no | pass | 50% | 50% | 6 | 0.58 | NEEDS_PAPER_LOCKBOX |
| opening_drive_on | −46 | 0% | no | pass | 50% | 50% | 1 | 0.94 | NEEDS_PAPER_LOCKBOX |

- **opening_drive London** is the only candidate that traded enough to judge (93
  lockbox trades). It *passes every economic gate* on held-out data — positive
  (+\$730), 0% ruin, not blown, survives cost stress, walk-forward 56%, 100% positive
  years — and is then **rejected because its lockbox Sharpe is not significant even at
  a trial count of 3** (DSR 0.30). A real edge would clear DSR easily at n=3; this one
  does not. **The edge does not replicate.**
- **compression_expansion** (6 trades) and **opening_drive overnight** (1 trade) fire
  too rarely on a one-year lockbox to be judged → NEEDS_PAPER_LOCKBOX.

## Result 2 — post-discovery adjustment (Phase-9 in-sample, frozen n=3)

The same Phase-9 *in-sample* Sharpes, re-deflated with `n=3` instead of `n=176`:

| Hypothesis | DSR (n=176) | DSR (n=3) |
|---|---:|---:|
| opening_drive_london | 0.00 | 0.44 |
| compression_expansion_or | 0.17 | 0.69 |
| opening_drive_on | 0.34 | **0.94** |

Reducing the trial count lifts DSR substantially — confirming that **multiple-testing
was the dominant Phase-9 killer** — and one candidate (`opening_drive` overnight)
crosses 0.90 *on Phase-9's own in-sample data*.

## The diagnosis: in-sample-significant, out-of-sample-insignificant

Put Results 1 and 2 side by side. With only three pre-registered hypotheses, the
Phase-9 *in-sample* numbers look defensible (DSR up to 0.94). But on *held-out* data
the picture inverts: the only candidate with enough held-out trades drops to DSR 0.30,
and the in-sample 0.94 candidate produced a single lockbox trade. **Significant in
sample, not out of sample — the textbook signature of a post-search winner.** The
Phase-9 HTF asymmetric setups are best explained as overfit to the 2021-2026 search,
not as a durable edge.

---

## Acceptance gate

- ✅ `make check` green (ruff + mypy + **188 tests**, incl. frozen-immutability,
  no-tuning-after-freeze, lockbox date-isolation, frozen-trial-count, report-schema,
  and ACCEPT/REJECT/NEEDS_PAPER_LOCKBOX verdict-reachability tests).
- ✅ Frozen hypotheses saved to an immutable, hash-verified registry.
- ✅ Lockbox run completes; every frozen candidate gets a verdict.
- ✅ No candidate accepted — no DSR pass on held-out data, and the lockbox was not
  pristine. Nothing was force-accepted.

## Honest answer & what comes next

> **Did Phase 9 find something real?** On this evidence, **no.** The strongest, most-
> tradeable survivor (`opening_drive` London-range break) is positive on held-out data
> but statistically insignificant even at the frozen trial count, and the others trade
> too rarely to confirm. The Phase-9 edge does not survive a held-out significance test
> — it is most consistent with post-search overfitting.

The only path that could still rescue any of these candidates is a **genuinely forward
paper lockbox**: deploy the frozen configs to paper trading on data generated *after*
this analysis (which the platform does not yet have), accumulate enough forward trades
(especially for the low-frequency compression / overnight setups), and re-test. Until
then they are **not** accepted. Building live/paper-trading infrastructure was out of
scope for this phase.

**Retired:** nothing re-optimised. Per the standing rule — *no edge until proven;
reject honestly, do not force an accept* — Phase 10 ends with **zero strategies
promoted**, the headline Phase-9 candidate **rejected on held-out significance**, and
the remaining two parked at **NEEDS_PAPER_LOCKBOX** pending forward confirmation.

---

## Reproduction

```
nqlab lockbox freeze          # write the immutable frozen registry from Phase-9 survivors
nqlab lockbox run --no-build  # test the frozen set on 2025-H2..2026-H1; DSR at frozen count
nqlab report phase10          # registry + lockbox metrics + Phase-9-vs-lockbox + verdicts
```
