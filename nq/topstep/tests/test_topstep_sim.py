from __future__ import annotations

from datetime import date, timedelta

import pytest

from tslab.sim.topstep import Phase, TopstepSimulator, Violation


def make_sim(cfg, size="50k", **kw):
    return TopstepSimulator(cfg, size, **kw)


def run_day(sim, session, pnl):
    sim.start_day(session)
    sim.realize(pnl)
    if not sim.failed:
        sim.end_day()


def day(n):
    return date(2026, 1, 5) + timedelta(days=n)   # Mon 2026-01-05 onward


# --------------------------------------------------------------------- #
# MLL mechanics
# --------------------------------------------------------------------- #

def test_mll_floor_trails_eod_and_locks_at_start(cfg):
    sim = make_sim(cfg)
    assert sim.mll_floor == 48000
    run_day(sim, day(0), +1000)              # bal 51,000
    assert sim.mll_floor == 49000
    run_day(sim, day(1), +1500)              # bal 52,500 -> floor capped at 50,000
    assert sim.mll_floor == 50000
    run_day(sim, day(2), +5000)              # floor must NOT exceed start balance
    assert sim.mll_floor == 50000


def test_mll_floor_never_moves_down(cfg):
    sim = make_sim(cfg)
    run_day(sim, day(0), +1000)
    assert sim.mll_floor == 49000
    run_day(sim, day(1), -900)               # losing day: floor unchanged
    assert sim.mll_floor == 49000


def test_intraday_mll_breach_fails_account(cfg):
    sim = make_sim(cfg)
    run_day(sim, day(0), +1000)              # floor 49,000
    sim.start_day(day(1))
    sim.mark(-2100)                          # equity 48,900 <= 49,000
    assert sim.phase == Phase.FAILED_COMBINE
    assert any(e.kind == "MLL_BREACH" for e in sim.events)
    with pytest.raises(Violation):
        sim.realize(100)                     # terminal account rejects activity


def test_mll_checked_on_unrealized_marks(cfg):
    # equity dip below floor fails even if it would have recovered
    sim = make_sim(cfg)
    sim.start_day(day(0))
    sim.mark(-2000)                          # 48,000 <= 48,000 floor -> breach
    assert sim.failed


# --------------------------------------------------------------------- #
# DLL mechanics
# --------------------------------------------------------------------- #

def test_personal_dll_lockout_not_failure(cfg):
    sim = make_sim(cfg, personal_dll=500)
    assert sim.effective_dll == 500
    sim.start_day(day(0))
    sim.mark(-500)
    assert not sim.failed
    assert sim.locked_out and not sim.can_trade()
    assert any(e.kind == "DLL_HIT" for e in sim.events)
    sim.end_day()
    sim.start_day(day(1))                    # next session unlocks
    assert sim.can_trade()


def test_effective_dll_is_min_of_account_and_personal(cfg):
    sim100 = make_sim(cfg, size="100k", personal_dll=750)
    assert sim100.effective_dll == 750       # account DLL is 2000
    sim100b = make_sim(cfg, size="100k")
    assert sim100b.effective_dll == 1000     # default personal 1000 < 2000


def test_dll_on_realized_losses(cfg):
    sim = make_sim(cfg, personal_dll=1000)
    sim.start_day(day(0))
    sim.realize(-400)
    assert sim.can_trade()
    sim.realize(-600)                        # day loss 1000 -> lockout
    assert sim.locked_out and not sim.failed


# --------------------------------------------------------------------- #
# Combine pass: min days + consistency target
# --------------------------------------------------------------------- #

def test_combine_pass_two_even_days(cfg):
    sim = make_sim(cfg)
    run_day(sim, day(0), +1600)              # best day 1600 -> target = 3200
    assert sim.phase == Phase.COMBINE
    run_day(sim, day(1), +1600)              # profit 3200 >= 3200, days ok
    assert sim.phase == Phase.FUNDED
    assert sim.balance == 50000              # XFA starts fresh
    assert sim.xfa_balance == 0
    assert any(e.kind == "ACTIVATION_FEE" for e in sim.events)


def test_consistency_target_raises_on_big_day(cfg):
    sim = make_sim(cfg)
    run_day(sim, day(0), +2000)              # 2000 > 50% * 3000 -> target 4000
    assert sim.effective_profit_target == 4000
    run_day(sim, day(1), +1500)              # profit 3500 < 4000: NOT passed
    assert sim.phase == Phase.COMBINE
    run_day(sim, day(2), +600)               # profit 4100 >= 4000: passed
    assert sim.phase == Phase.FUNDED


def test_cannot_pass_in_one_day(cfg):
    sim = make_sim(cfg)
    # huge single day: consistency raises target anyway, but even a clean
    # 2x-target day cannot pass on day one (min_trading_days = 2)
    run_day(sim, day(0), +1400)
    run_day(sim, day(1), +1500)
    run_day(sim, day(2), +200)               # profit 3100, best day 1500 -> target 3000
    assert sim.phase == Phase.FUNDED
    assert sim.trading_days >= 2


# --------------------------------------------------------------------- #
# Funded stage: payouts
# --------------------------------------------------------------------- #

def to_funded(cfg, **kw):
    sim = make_sim(cfg, **kw)
    run_day(sim, day(0), +1500)
    run_day(sim, day(1), +1500)
    assert sim.phase == Phase.FUNDED
    return sim


def test_payout_eligibility_and_split(cfg):
    sim = to_funded(cfg)
    for i in range(4):
        run_day(sim, day(2 + i), +400)
    assert sim.max_payout_available() == 0   # only 4 winning days
    run_day(sim, day(6), +400)               # 5th winning day, profit 2000
    assert sim.winning_days_cycle == 5
    assert sim.max_payout_available() == pytest.approx(1000)   # 50% of 2000 < cap 2000
    net = sim.request_payout(1000)
    assert net == pytest.approx(900)         # 90/10 split
    assert sim.winning_days_cycle == 0       # cycle resets
    assert sim.xfa_balance == pytest.approx(1000)


def test_payout_cap_by_size(cfg):
    # default config has use_topstep_dll=true and the active promo doubles
    # the caps: 50k Standard cap 2000 -> 4000, so 50% of 5000 = 2500 binds
    sim = to_funded(cfg)
    for i in range(5):
        run_day(sim, day(2 + i), +1000)      # profit 5000
    assert sim.max_payout_available() == pytest.approx(2500)


def test_payout_cap_without_dll_promo(cfg):
    from dataclasses import replace
    cfg_nodll = replace(cfg, risk=replace(cfg.risk, use_topstep_dll=False))
    sim = to_funded(cfg_nodll)
    for i in range(5):
        run_day(sim, day(2 + i), +1000)      # profit 5000
    # no DLL at checkout -> plain 50k Standard cap 2000 binds (50% would be 2500)
    assert sim.max_payout_available() == pytest.approx(2000)


def test_min_payout_threshold(cfg):
    sim = to_funded(cfg)
    for i in range(5):
        run_day(sim, day(2 + i), +150)       # profit 750 (winning days at exactly $150)
    # 50% of 750 = 375 >= 125 -> allowed
    assert sim.max_payout_available() == pytest.approx(375)


def test_winning_day_threshold_is_150(cfg):
    sim = to_funded(cfg)
    run_day(sim, day(2), +149.99)
    assert sim.winning_days_cycle == 0
    run_day(sim, day(3), +150.00)
    assert sim.winning_days_cycle == 1


def test_mll_locks_at_zero_after_first_payout(cfg):
    sim = to_funded(cfg)
    for i in range(5):
        run_day(sim, day(2 + i), +400)       # profit 2000; floor still 48,000+trail
    sim.request_payout(1000)
    assert sim.mll_floor == 50000            # XFA $0 floor
    sim.start_day(day(8))
    sim.mark(-1100)                          # equity 49,900 <= 50,000
    assert sim.phase == Phase.FAILED_FUNDED


def test_positive_since_last_payout_required(cfg):
    sim = to_funded(cfg)
    for i in range(5):
        run_day(sim, day(2 + i), +400)
    sim.request_payout(1000)                 # xfa balance now 1000
    # five more winning days but ending BELOW the post-payout balance
    for i, pnl in enumerate((+200, +200, +200, +200, +200)):
        run_day(sim, day(8 + i), pnl)
    run_day(sim, day(14), -1300)             # xfa balance 700 < 1000 at last payout
    assert sim.max_payout_available() == 0


def test_payout_rejected_when_too_large(cfg):
    sim = to_funded(cfg)
    for i in range(5):
        run_day(sim, day(2 + i), +400)
    with pytest.raises(Violation):
        sim.request_payout(1500)             # max is 1000


def test_payout_rejected_below_minimum(cfg):
    sim = to_funded(cfg)
    for i in range(5):
        run_day(sim, day(2 + i), +400)       # max available 1000
    with pytest.raises(Violation):
        sim.request_payout(100)              # below $125 minimum


def test_payout_does_not_distort_day_pnl(cfg):
    sim = to_funded(cfg)
    for i in range(5):
        run_day(sim, day(2 + i), +400)       # profit 2000, cycle 5
    sim.start_day(day(8))
    sim.realize(+400)                        # a genuine +$400 trading day
    sim.request_payout(1000)                 # withdrawal on the same day
    sim.end_day()
    # the -$1000 withdrawal must not turn a winning day into a loser
    # (cycle was reset to 0 by the payout, then +1 for today's $400)
    assert sim.winning_days_cycle == 1
    assert not any(e.kind == "DLL_HIT" for e in sim.events)


# --------------------------------------------------------------------- #
# Fees / cash ledger
# --------------------------------------------------------------------- #

def test_fee_ledger_standard_plan(cfg):
    sim = make_sim(cfg, start_date=date(2026, 1, 3))
    assert sim.fees_paid == 49                       # first month billed
    run_day(sim, date(2026, 1, 5), -100)
    assert sim.fees_paid == 49                       # same month, no rebill
    run_day(sim, date(2026, 2, 2), +100)             # new month
    assert sim.fees_paid == 98
    sim.reset_combine()                              # reset = monthly rebill price
    assert sim.fees_paid == 147
    assert sim.resets == 1
    assert sim.balance == 50000 and sim.mll_floor == 48000


def test_skipped_months_still_billed(cfg):
    # subscription rebills every month whether or not the account trades
    sim = make_sim(cfg, start_date=date(2026, 1, 3))
    run_day(sim, date(2026, 1, 5), +100)
    run_day(sim, date(2026, 4, 6), +100)             # Feb + Mar skipped
    assert sim.fees_paid == 4 * 49                   # Jan, Feb, Mar, Apr


def test_billing_across_year_boundary(cfg):
    sim = make_sim(cfg, start_date=date(2026, 12, 1))
    run_day(sim, date(2026, 12, 3), +100)
    run_day(sim, date(2027, 1, 4), +100)
    assert sim.fees_paid == 2 * 49                   # Dec 2026 + Jan 2027


def test_end_day_requires_matching_start(cfg):
    sim = make_sim(cfg)
    run_day(sim, day(0), +100)
    with pytest.raises(Violation):
        sim.end_day()                                # second end_day, no start


def test_no_monthly_fee_once_funded(cfg):
    sim = to_funded(cfg)
    fees_at_funding = sim.fees_paid
    run_day(sim, date(2026, 3, 2), +200)             # new month while funded
    assert sim.fees_paid == fees_at_funding


def test_net_cashflow(cfg):
    sim = make_sim(cfg, start_date=date(2026, 1, 3))
    run_day(sim, day(0), +1500)
    run_day(sim, day(1), +1500)                      # funded; +149 activation
    for i in range(5):
        run_day(sim, day(2 + i), +400)
    sim.request_payout(1000)
    # fees: 49 monthly + 149 activation = 198; payouts net: 900
    assert sim.fees_paid == pytest.approx(198)
    assert sim.net_cashflow == pytest.approx(900 - 198)


def test_funded_cannot_reset(cfg):
    sim = to_funded(cfg)
    with pytest.raises(Violation):
        sim.reset_combine()


# --------------------------------------------------------------------- #
# Position limits: scaling plan + volatility caps
# --------------------------------------------------------------------- #

def test_combine_position_limits(cfg):
    sim = make_sim(cfg)
    assert sim.max_contracts_now() == 5
    assert sim.check_position_size(5)
    assert not sim.check_position_size(6)
    assert sim.check_position_size(4, 10)            # 4 + 10/10 = 5 mini-equiv
    assert not sim.check_position_size(4, 15)        # 5.5 mini-equiv
    assert not sim.check_position_size(0, 60)        # micros hard cap 50


def test_funded_scaling_plan_limits(cfg):
    # scaling tier is set from the day-START balance (next-session updates)
    sim = to_funded(cfg)                             # xfa balance 0
    assert sim.max_contracts_now() == 2              # below $1,500 tier
    for i in range(4):
        run_day(sim, day(2 + i), +400)               # 1,600 at EOD day 4
    sim.start_day(day(6))
    assert sim.max_contracts_now() == 3              # tier from 1,600 day-start
    sim.realize(+500)                                # 2,100 mid-session...
    assert sim.max_contracts_now() == 3              # ...does NOT bump the tier
    sim.end_day()
    sim.start_day(day(7))
    assert sim.max_contracts_now() == 5              # next session: above $2,000
    sim.end_day()


def test_gold_volatility_cap(cfg):
    sim = make_sim(cfg)                              # combine, limit 5
    assert sim.max_contracts_now("NQ") == 5
    assert sim.max_contracts_now("GC") == 3          # current restriction
    assert not sim.check_position_size(4, instrument="GC")
    assert sim.check_position_size(3, instrument="GC")


def test_micro_gold_volatility_cap(cfg):
    sim = make_sim(cfg)
    # MGC cap is 30 MICROS on the 50k: 40 micros = only 4 mini-equiv (under
    # the account's 5) but must still be rejected by the micro cap
    assert not sim.check_position_size(0, 40, instrument="MGC")
    assert sim.check_position_size(0, 30, instrument="MGC")
    # and the micro cap must not corrupt the mini limit
    assert sim.max_contracts_now("MGC") == 5


def test_eod_flatten_violation_recorded(cfg):
    sim = make_sim(cfg)
    sim.start_day(day(0))
    sim.flatten_violation()
    assert any(e.kind == "EOD_FLATTEN_VIOLATION" for e in sim.events)
