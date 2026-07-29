# Phase 59 — GC Overnight-Without-Overnight-Hold (Topstep-legal)

**Verdict: the overnight move is mechanically capturable inside Topstep rules, but it is
long-gold BETA, not a robust edge. It does NOT beat NQ.** Only "always-long overnight"
is positive; it loses in 5 of 10 years, concentrates all profit in 2024–25, fails the
train/val/test gate, and carries overnight-gap blow risk that breaches small accounts.

Setup: reopen 18:00 ET (smin 0) → flatten 16:10 ET (smin 1330) sit in one session_date,
so an overnight hold is modelled exactly. One round-turn MGC cost ($3.48) per overnight
trade (cost-efficient). Long/short split run to separate beta from edge.

## Results (1 MGC, 2016–2025, full cost)
| strategy | n | exp$ | PF | total$ | maxDD$ | train | val | test | +yrs | robust |
|---|---|---|---|---|---|---|---|---|---|---|
| S1a LONG reopen→COMEX | 2579 | +3.4 | 1.08 | +8,658 | 7,631 | +0.3 | **−9.5** | +17.0 | 5 | ✗ |
| S1b SHORT reopen→COMEX (beta chk) | 2579 | −10.3 | 0.78 | −26,608 | 27,113 | −7.3 | +2.6 | −24.0 | 1 | ✗ |
| S2a LONG reopen→flatten | 2498 | +4.1 | 1.07 | +10,324 | 10,179 | −0.3 | **−10.1** | +21.1 | 7 | ✗ |
| S2b SHORT reopen→flatten (beta chk) | 2498 | −11.1 | 0.84 | −27,710 | 29,879 | −6.6 | +3.1 | −28.1 | 3 | ✗ |
| S1a + 2×ATR stop | 2579 | +3.4 | 1.08 | +8,797 | 7,826 | +0.3 | −9.9 | +17.5 | 5 | ✗ |
| S3 Asia-trend (eve dir→03:00) | 2579 | −4.3 | 0.86 | −11,150 | 12,077 | −3.7 | −2.8 | −6.3 | 1 | ✗ |
| S4 London-trend (ON dir→COMEX) | 2578 | −2.5 | 0.91 | −6,462 | 9,056 | −3.4 | −2.3 | −1.1 | 4 | ✗ |
| S5 ON breakout (eve range→COMEX) | 2558 | −0.3 | 0.99 | −728 | 13,267 | −4.4 | −6.0 | +10.6 | 1 | ✗ |
| S6 ON mean-rev (fade eve→COMEX) | 2563 | −2.5 | 0.94 | −6,515 | 8,767 | −2.3 | +1.5 | −5.6 | 3 | ✗ |
| S7 prior-day cont reopen→COMEX | 2578 | −8.6 | 0.82 | −22,034 | 27,965 | −4.4 | −11.6 | −13.3 | 1 | ✗ |
| S8 LONG Mon-only reopen→COMEX | 514 | −4.8 | 0.90 | −2,492 | 7,676 | −11.3 | −20.0 | +16.1 | 1 | ✗ |

**Beta proof:** LONG overnight is mildly +, SHORT overnight is heavily − (PF 0.78/0.84) — a symmetric *edge* would profit both ways. This is gold's upward drift, full stop.

**Yearly $ (S1a LONG reopen→COMEX, 1 MGC):**
`2016 +1687 · 2017 −304 · 2018 −1040 · 2019 −151 · 2020 +231 · 2021 −1703 · 2022 −3219 · 2023 +596 · 2024 +3656 · 2025 +8905`
→ negative in **5/10 years**; 2021–22 (flat gold) lost **−$4,922**; remove 2024–25 and the strategy is net negative. Not regime-robust.

## Topstep EV (best funded sizing)
| strategy | acct | pass% | EV/mo$ | timeout% | worst month (3c) |
|---|---|---|---|---|---|
| S1a LONG→COMEX | 50K | 0.17 | 13 | 32% | **−4,386** |
| S1a LONG→COMEX | 100K | 0.35 | 36 | 67% | −4,386 |
| S1a LONG→COMEX | 150K | 1.00 | 150 | **89%** | −4,386 |
| S2a LONG→flatten | 50K | 0.18 | 25 | 9% | −4,574 |
| S2a LONG→flatten | 100K | 0.10 | 7 | 34% | −4,574 |
| S5 ON breakout | any | ≤0.11 | ≤1 | 33–94% | −2,931 |
| S7 prior-day cont | any | 0.00 | −7 to −19 | — | −16,319 |

- The eye-catching **150K EV/mo $150** is an artifact: **89% timeout** (combine rarely resolves) and the rare resolutions land in the 2024–25 bull → it is beta + small-sample noise, not a repeatable edge.
- **Overnight gap risk is real:** worst month −$4,386 at just 3 MGC **exceeds the 50K ($2,000) and 100K ($3,000) trailing MLL → blows the account.** Even at 1 contract the cumulative drawdown ($7,631) breaches a 50K MLL during the 2021–22 gold drought.

## Reported answers
- **Net profit:** best = +$10,324/MGC over 10y (S2a long→flatten); all non-long and all signal strategies negative.
- **PF:** 1.07–1.08 (long only); <1.0 for every signal-based and short variant.
- **Drawdown:** $7,600–$10,200 per 1 MGC for the long variants (large; blows small Topstep accounts).
- **Yearly:** negative 5/10 years; profit concentrated in 2024–25.
- **Topstep pass probability:** low and timeout-dominated; resolvable EV/month ≈ $13–36 (50K/100K), and not robust.
- **EV/month/account:** ~$25–36 best *resolvable*; the $150 (150K) is a beta/timeout artifact.
- **Beats NQ?** **No.** NQ best is a genuine, symmetric, regime-robust edge (PF 1.22, profitable 5/6 years incl. choppy ones, ~$400/mo robust). GC overnight is one-directional beta that fails val (2021–22), loses in flat/down gold years, and risks blow-ups from gaps.

## Bottom line
Topstep *does* allow capturing the overnight session, and gold's overnight drift is mildly
positive and cost-efficient — but it is **beta, not edge**: regime-dependent (would have
lost in 2013–15 gold bear, and did lose 2017–19/2021–22), asymmetric (longs win, shorts
lose), concentrated in two bull years, and exposed to overnight-gap MLL breaches. **It does
not beat NQ and is not a robust ≥$400/mo Topstep engine.** The only honest use is a
*regime-conditional long-gold overlay* (trade it only in confirmed uptrends), which is a
macro timing bet, not a mechanical edge.

Code: `scripts/phase59_overnight.py`.
