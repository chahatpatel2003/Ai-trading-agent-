# Phase 31 Addendum — Profit-Buffer / Growth-Gated Sizing

Correction applied: do **not** assume a strategy starts at 2/5 MNQ from trade one. Start at 1 MNQ,
build cushion, scale only with buffer, de-scale when buffer shrinks. Tested Plans A–E on the FROZEN
finalists under **both** a pure-trailing and a locked $3K drawdown, with Monte-Carlo sequence risk
(reshuffle, week/month block bootstrap, slippage, **bad-start** = first 20 trades from the worst
quartile). No rule changes. Data: `docs/phase31/phase31_scaling.csv` (48 rows) + `_scaling.json`.

**Headline result (exact).**

```
The "build cushion then scale up in tiers" idea (Plans A/C/D/E) FAILS on a $3K account:
  • Pure trailing: it only survived HISTORICALLY by lucky order — reshuffled breach 60-68%,
    bad-start breach 42-54%. (A trailing floor = peak-3000 caps buffer at $3K forever, so a larger
    size always risks the same $3K from the latest peak — profit cushion does NOT add room.)
  • Locked floor: $/month rises ($272 swd / $857 portfolio / $981 hybrid) BUT bad-start breaches
    39-50% and p99 drawdown is $12K-22K — fragile to the opening sequence.

ONE plan is genuinely safe: Plan B (risk a % of the CURRENT buffer) under pure trailing.
It auto-de-scales as buffer shrinks: reshuffle breach 0%, bad-start breach 0%, p99 DD < $3K.
Its cost is low profit: idb ~$114-165/mo (best), swd ~breakeven. It does NOT reach $500/mo.
```

So growth-gating **does** help — but it fixes *safety*, not *profit*. It cannot manufacture $500/mo on a $3K trailing account.

---

## 1. The mechanic that decides everything

With a **pure trailing** floor = `peak − $3,000`:
`buffer = equity − floor = $3,000 − (peak − equity) = $3,000 − drawdown_from_peak ≤ $3,000` **always**.
At a new equity high the buffer resets to exactly $3,000 — the floor trailed up with you. Therefore
**accumulated profit never increases your room**, and a larger position always risks the same $3,000
from the most recent peak. Tier-scaling (A/C/D/E) just places bigger bets against the same $3K wall.

Durable cushion exists only under a **locked** floor (`min(peak−3000, 100000)` → stops at the
starting balance once the $3K buffer is banked). There, profit above $100K is real room — but you
must first survive the un-cushioned start, which is where it breaks.

---

## 2. Results by plan (swd & portfolio shown; full 48-row table in CSV)

`hist` = historical-order breach; `resh` = reshuffle breach; `bad` = bad-start breach; p99 = reshuffle p99 intraday DD.

### swd_clean_path
| plan | model | max sz | avg sz | net $ | $/mo | intraday DD | hist | resh | bad | p99 DD | class |
|---|---|--:|--:|--:|--:|--:|:--:|--:|--:|--:|---|
| **B 10%** | trailing | 7 | 1.9 | −152 | −2 | 2,433 | ✅ | **0.00** | **0.00** | **2,692** | START_SMALL (too low $) |
| **B 20%** | trailing | 14 | 4.1 | 503 | 8 | 2,892 | ✅ | **0.00** | **0.00** | **2,930** | START_SMALL (too low $) |
| A / C | trailing | 5 | 1.5 | 11,835 | 179 | 2,744 | ✅ | 0.61–0.64 | 0.42–0.44 | 3,480–3,556 | REJECT_TOO_MUCH_SEQUENCE_RISK |
| A / C | locked | 5 | 1.5 | 17,967 | 272 | 4,512 | ✅ | 0.004 | **0.41** | 12,927 | REJECT_TOO_MUCH_SEQUENCE_RISK |
| E (monthly) | trailing | 2 | 1.0 | 7,668 | 116 | 1,223 | ✅ | 0.004 | **0.39** | 2,983 | REJECT_TOO_MUCH_SEQUENCE_RISK |

### portfolio (idb + swd)
| plan | model | max sz | avg sz | net $ | $/mo | intraday DD | hist | resh | bad | p99 DD | class |
|---|---|--:|--:|--:|--:|--:|:--:|--:|--:|--:|---|
| **B 10/20%** | trailing | 13–15 | 2.2–3.8 | 0.3–1.8k | 5–28 | 2,556–2,840 | ✅ | **0.00** | **0.00** | **2,706–2,930** | START_SMALL (too low $) |
| A / C | trailing | 5 | 1.5 | 8,770 | 133 | 3,019 | ❌ | 0.67–0.68 | 0.51 | 3,454–3,494 | REJECT_NOT_SCALABLE |
| A / C | locked | 5 | 3.7 | 56,559 | **857** | 10,341 | ✅ | 0.05–0.06 | **0.46** | 21,200–21,939 | REJECT_TOO_MUCH_SEQUENCE_RISK |
| E | trailing | 3 | 1.05 | 13,160 | 199 | 2,573 | ✅ | 0.10 | **0.49** | 3,080 | REJECT_TOO_MUCH_SEQUENCE_RISK |

**idb** and **hybrid** behave like the portfolio: tier plans reach $251–981/mo under locked but bad-start
breach 45–50% / p99 $20K+; **Plan B is the only safe one** (idb B 20% = **$165/mo, 0% breach, p99
$2,838** — the best safe config found). hybrid/idb locked Plan B blow up (p99 $70K–4.5M) because Plan
B risks a % of an *unbounded* locked buffer → never use Plan B with a locked floor.

---

## 3. Answers to the 10 questions (exact)

1. **Can it start at 1 MNQ and scale after cushion?** Safely, **only via Plan B (buffer-risk %)** —
   it scales up when buffer is full and de-scales automatically (reshuffle/bad-start breach **0%**,
   p99 DD < $3K). Tier-based scaling (A/C/D/E) does **not** scale safely (45–68% breach on
   reshuffle/bad-start).
2. **Cushion needed before 2 MNQ is safe?** Under **pure trailing: none works** — buffer is capped at
   $3K, so 2 MNQ always risks ~2× a single trade's MAE against $3K (reshuffle breach jumps to ~60%).
   Under **locked: ~$3–5K cushion** makes 2 MNQ *normal-sequence* safe — but it still fails bad-start.
3. **Before 3 MNQ?** Locked ~$7–9K cushion (normal-safe; p99 DD ~$15–20K; bad-start fails). Pure
   trailing: never.
4. **Before 5 MNQ?** Locked ~$12K+ cushion (reached only in a strong run like 2025; p99 DD ~$20K;
   bad-start fails). Pure trailing: never.
5. **Does scaling improve $/mo without too much breach risk?** **No.** Tiers raise $/mo to
   $272–981 (locked) but bad-start breach 39–50% and p99 DD $12–22K. Plan B keeps breach ~0% but
   $/mo stays ≤ $165. There is no plan with both meaningful $/mo **and** < 5% breach incl. bad-start.
6. **Best return/DD plan?** **Plan B (buffer-risk ~20%) under pure trailing** — only plan with p99 DD
   < $3K; best safe risk-adjusted (idb: $165/mo at p99 $2,838).
7. **Lowest failure probability?** **Plan B, pure trailing** — 0% reshuffle and 0% bad-start breach,
   by construction (never risks more than a fraction of current buffer).
8. **Does a bad start kill it?** **Yes for every tier plan** (bad-start breach 39–54%). **No for Plan
   B** (0% — it stands down when the buffer is small). This is the decisive differentiator.
9. **Safest practical growth plan?** **Plan B: risk ~10–20% of the current trailing buffer per trade,
   min skip if < 1 MNQ, auto-de-scale** — on a *pure-trailing* $3K account. idb is the best vehicle
   (~$114–165/mo, ~0% breach). swd is ~breakeven under risk-based sizing (its edge needs fixed size).
10. **Still a 1-MNQ edge, or scalable after buffer?** **Effectively still small-size.** Plan B can
    average ~2–3 contracts (transiently more when buffer is full) at ~0% breach, but it **cannot be
    grown into a $500+/mo business on a $3K trailing account** without sequence risk that breaches
    40%+ of the time. It only becomes a meaningful scalable business on a **locked or ≥$5–6K-DD
    account** — and even then a bad opening sequence is a real (~45%) failure risk.

---

## 4. Classifications

| config | classification |
|---|---|
| **idb_base — Plan B buffer-risk 10–20%, pure trailing** | **START_SMALL_FORWARD_PAPER** (safe, auto-scaling, ~$114–165/mo) |
| swd / portfolio / hybrid — Plan B, pure trailing | START_SMALL_FORWARD_PAPER (safe) but **$/mo ≤ $28 → RESEARCH_ONLY** in practice |
| any strategy — Plan A / C / D / E, pure trailing | REJECT_TOO_MUCH_SEQUENCE_RISK (portfolio/hybrid A/C/D: REJECT_NOT_SCALABLE — historical breach) |
| any strategy — Plan A / C / D, locked floor | REJECT_TOO_MUCH_SEQUENCE_RISK (bad-start 39–50%, p99 DD $12–22K) |
| any strategy — Plan B, locked floor | REJECT_NOT_SCALABLE (risks a % of unbounded buffer → p99 $70K–4.5M blow-ups) |
| **SCALABLE_AFTER_BUFFER** | **NONE** (no plan reaches ≥$300/mo with < 5% breach incl. bad-start) |

### Bottom line
Growth-gating was the right instinct and it **does** change the answer — but in the safety dimension,
not profit. **Plan B (buffer-risk %) makes these edges genuinely $3K-survivable** (0% breach, p99 <
$3K, survives bad starts) by auto-de-scaling — the first sizing model that truly respects the trailing
floor. The price is low income (idb best, ~$114–165/mo). Tier-based "scale to 2/3/5 after +$X"
**does not work on a pure-trailing $3K account** (the floor trails up, so cushion never buys room) and
is fragile to bad starts even under a locked floor. To run these as a real ≥$500/mo business you need
a **locked or ≥$5–6K-drawdown account**; on a strict $3K trailing account the honest plan is
**idb via Plan B ~20%, START_SMALL_FORWARD_PAPER, expect ~$150/mo**, not $500.

```
Reproduction: python scripts/phase31_scaling.py   # -> docs/phase31/phase31_scaling.csv (+_scaling.json)
```
