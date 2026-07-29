# Phase 34B — High-Win-Rate Hypothesis Lab (retail pattern stacks, not single labels)

Test COMPLETE retail hypotheses — *context + trap/failure + confirmation + target-path + risk
control* — instead of isolated labels, aiming for a **genuinely** high win rate (not fake high WR from
tiny targets). The honesty metric is **edge over the breakeven win rate** (actual WR minus the
avg-win/avg-loss breakeven): it strips out win rate that is merely target/stop geometry. Audited
engine, conservative fills, EOD-flat, all years, default retail bans (skip first 15 min, last hour,
low-vol quintiles).

**Headline.**

```
NO high-win-rate retail stack met the bar (WR>=60% AND PF>=1.4 AND edge>=5pp): 0 of 240 candidates.
High win rate IS achievable but FAKE: 87 stacks hit WR>=60%, but the 90-95%-WR ones (tiny 5-10pt
  targets) have edge-over-breakeven ~0 (-0.5 to +0.7pp) and PF 0.91-1.21 — pure geometry, the exact
  trap. Their high WR is the target/stop, not an edge.
BUT the hypothesis-stack method DID find the best REAL edge in the whole program:
  breakout_acceptance long (break VAH + developing-POC migrating up = real acceptance -> continuation
  to POC): edge-over-breakeven +7.6pp, PF 1.36, 50% WR, 5/6 years, survives 2x cost (PF 1.28),
  $3K-survivable at 1 MNQ (MC breach 3.4%). Ablation proves the "acceptance" confirmation IS the edge
  (remove it -> edge 7.6pp->3.3pp, PF 1.36->1.15).
So: stacks beat isolated labels (real edge concentrated), but it's a ~50%-WR CONTINUATION edge of
modest size (PF 1.36, ~$89/mo at 1 MNQ) — NOT the high-win-rate, PF>=1.4 setup the phase sought.
```

Verdict: **no HIGH_WIN_FORWARD_PAPER candidate.** Best find = `breakout_acceptance` → **RESEARCH_ONLY**
(real, modest continuation edge for Phase 35; not high-WR). High-WR tiny-target stacks → REJECT_NO_EDGE.

---

## 1. Hypothesis universe
- **9 retail stacks** (each context+trap/accept+confirmation+path): failed-breakout trap (PDH/ONH),
  breakout-acceptance, stop-hunt reversion, VWAP rubber-band, POC magnet, OR trap, gap trap, trend
  pullback, range rotation. × **2 directions** × **6 target paths** (to-VWAP / to-POC / to-OR-mid / 1:1
  / fixed-5 / fixed-10) × **2 ATR stops** = **240 candidates**, 5m, 2021→2026-H1, default bans on.
- Calendar bucket (J): **economic-calendar data does not exist in this NQ-only feed → not tested**
  (import spec: a CPI/FOMC/NFP timestamp table joined on session date).

## 2. High-win-rate leaderboard (the fake vs the real)

**By REAL edge over breakeven (n≥120):**
| rank | stack | dir | target | n | win% | breakeven% | **edge** | PF | $/mo | ret/DD | yrs+ |
|--:|---|--|--|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | breakout_acceptance | long | OR-mid | 245 | 53.1% | 45.3% | **+7.8pp** | 1.36 | 83 | 5.3 | 4/6 |
| 2 | breakout_acceptance | long | POC | 244 | 50.0% | 42.4% | **+7.6pp** | 1.36 | 89 | 4.7 | 5/6 |
| 3 | breakout_acceptance | long | VWAP | 244 | 52.5% | 45.2% | +7.2pp | 1.34 | 79 | 4.2 | 5/6 |
| 4 | breakout_acceptance | long | 1:1 | 247 | 54.3% | 48.6% | +5.7pp | 1.26 | 60 | 3.3 | 4/6 |
| 5 | gap_trap | long | POC | 760 | 54.5% | 50.7% | +3.8pp | 1.16 | 135 | 1.8 | 4/6 |
| 6 | gap_trap | short | POC | 883 | 52.0% | 49.0% | +3.0pp | 1.13 | 121 | 2.7 | 3/6 |

**The fake high-win-rate (tiny targets) — what to ignore:**
| stack | dir | target | win% | breakeven% | edge | PF |
|---|--|--|--:|--:|--:|--:|
| breakout_acceptance | long | fixed-5 | **94.8%** | 94.3% | +0.5pp | 1.10 |
| gap_trap | short | fixed-5 | 94.3% | 94.4% | **−0.1pp** | **0.98** |
| trend_pullback | long | fixed-5 | 93.8% | 93.3% | +0.4pp | 1.08 |
| stophunt_reversion | long | fixed-5 | 93.2% | 93.5% | −0.3pp | 0.95 |

> 90-95% win rates exist everywhere with 5-pt targets — and they are worthless: the breakeven win
> rate is *also* ~94%, edge ≈ 0, and after cost PF ≤ 1.1 (several < 1.0). **High win rate ≠ edge.**
> Max real edge anywhere = **+7.8pp** (breakout_acceptance); max PF = **1.36**. Bar (WR≥60% & PF≥1.4 &
> edge≥5pp): **0 setups.**

## 3. Component ablation — breakout_acceptance long → POC (the standout)

| variant | n | win% | edge | PF | $/mo |
|---|--:|--:|--:|--:|--:|
| **FULL** (VAH-break + above-value + **dev-POC rising**) | 244 | 50.0% | **+7.6pp** | **1.36** | 89 |
| − context (drop "above value") | 244 | 50.0% | +7.6pp | 1.36 | 89 | ← redundant (VAH-break ⇒ above value) |
| − **confirmation** (drop dev-POC rising) | 680 | 45.6% | **+3.3pp** | **1.15** | 110 | ← edge halves |
| trigger only (VAH-break) | 680 | 45.6% | +3.3pp | 1.15 | 110 |

> **The "acceptance" confirmation (developing POC migrating up with the break) IS the edge** — it
> roughly doubles edge (3.3→7.6pp) and lifts PF (1.15→1.36) while halving trade count (the raw
> breakout is the well-known fail-prone retail trade; acceptance separates the ~1/3 that continue).
> This is exactly the Phase-34B thesis: a *stack* (trigger + confirmation) concentrates real edge a
> *label* (breakout) cannot. Bans add a little (+0.6pp); survives 2× cost (PF 1.28, edge +6.1pp).

## 4. Do-not-trade filters
The Phase-32 bans (low-vol quintile, first 15 min, last hour) were ON by default and help marginally
(breakout_acceptance edge +7.6pp with bans vs +7.0pp without; fewer, cleaner trades). The dominant
"filter" is not a ban but the **acceptance confirmation** above — the difference between a fake and a
real breakout.

## 5. Account feasibility — breakout_acceptance long → POC, 1 MNQ
| account | result |
|---|---|
| max closed DD | $1,257 ; net $5,867 over 66 mo (**$89/mo**) |
| 100K / **$3K** trailing | MC breach **3.4%** (< 5%), p99 DD **$3,372** (≈ $3K) — borderline-pass |
| 100K / $5K | MC breach 0%, p99 $3,372 — comfortable |
| overfit | remove best 5 trades → $3,421 (best-5 share **42%**, vs idb's 81%); best-month share 17% |
| yearly | 2021 +86 · 2022 +1,320 · **2023 −545** · 2024 +1,484 · 2025 +2,324 · 2026 +1,198 (recent-weighted; 5/6 positive) |
| make $500/mo? | only at ~5-6 MNQ → needs ~$15-18K DD; **no on $3K**. make $1,000/mo? no on any tight account |

> Cleaner profile than idb (less trade-concentration, $3K-survivable by MC), but the same wall:
> ~$89/mo at the only $3K-safe size. It is recent-weighted (2023 negative) — a stability caveat.

## 6. Final answers
- **Did any high-win-rate retail hypothesis stack work?** **No.** 0 setups cleared WR≥60% + PF≥1.4 +
  edge≥5pp. Every genuinely-high-WR setup (90-95%) is fake (edge ≈ 0, PF ≤ 1.1).
- **Best win rate?** ~95% (tiny fixed-5 targets) — but worthless (edge ≈ 0).
- **Best PF?** **breakout_acceptance long → OR-mid/POC, PF 1.36.**
- **Best real edge (over breakeven)?** **breakout_acceptance long, +7.8pp** — the best in the program.
- **Best drawdown / safest money?** breakout_acceptance (DD $1,257, $3K-survivable, $89/mo) and swd
  (Phase 30/34A) are comparable; neither makes meaningful money on $3K.
- **Which filters actually improved results?** The **acceptance confirmation (developing-POC rising)**
  — it is the edge, not a cosmetic filter. Phase-32 bans help marginally. "Above value" was redundant.
- **Which retail ideas are fake?** Tiny-target "high win rate" (all of it). Raw failed-breakout,
  VWAP-rubber-band, POC-magnet, range-rotation, trend-pullback as tested here did **not** clear edge≥5pp
  at PF≥1.4 — their apparent edge is mostly geometry/concentration.
- **Forward-paper candidate?** Not at the high-WR bar. `breakout_acceptance` is the only one worth
  forward-paper *as a continuation idea* (not high-WR) — **RESEARCH_ONLY**, lean toward forward-paper
  on a ≥$5K account if pursued.
- **Live candidate?** No.
- **Phase 35 focus?** Develop **acceptance-confirmed continuation** (breakout/trend continuation gated
  by developing-POC migration / real participation) — it carries the only +5pp real edge found.
  Stop chasing high win rate (it is geometry). And/or import cross-market (ES/VIX) + economic-calendar
  data to test the two untested buckets.

## Verdicts
| candidate | verdict |
|---|---|
| **breakout_acceptance long → POC/OR-mid (dev-POC accept)** | **RESEARCH_ONLY** — best real edge (+7.6pp), PF 1.36, survives 2× cost, $3K-survivable @1 MNQ, but 50% WR (not high), PF<1.4, ~$89/mo, recent-weighted |
| gap_trap → POC | **RESEARCH_ONLY / borderline REJECT** — edge +3-4pp, PF 1.13-1.16, concentrated, 2-4/6 yrs |
| all tiny-target (fixed-5/10) "high win rate" stacks | **REJECT_NO_EDGE** — edge ≈ 0, PF ≤ 1.1 (fake WR from geometry) |
| failed-breakout / VWAP-rubber-band / POC-magnet / range-rotation / trend-pullback / stop-hunt | **REJECT_NO_EDGE** (at edge≥5pp & PF≥1.4 bar) — did not separate from breakeven |
| **HIGH_WIN_FORWARD_PAPER** | **NONE** |

### Bottom line
The honest answer to "find a high-win-rate retail stack": **there isn't one** — high win rate in NQ is
target/stop geometry with ~zero edge over breakeven, exactly the fake the brief warned about. What the
hypothesis-stack method *did* prove is that **stacks beat labels**: the `breakout_acceptance`
continuation (gated by developing-POC acceptance) carries the largest genuine edge over breakeven
found anywhere in the program (+7.6pp, PF 1.36) — but it is a modest ~50%-WR continuation edge, not a
high-win-rate machine, and on a $3K account still only ~$89/mo. Phase 35 should build the
acceptance-continuation idea (and import ES/VIX + calendar), not hunt high win rate.

```
Reproduction:
python scripts/phase34b_run.py    # 240-candidate hypothesis-stack grid -> docs/phase34b/phase34b_leaderboard.csv
# ablation/overfit/account numbers reproduced inline (see report)
```
