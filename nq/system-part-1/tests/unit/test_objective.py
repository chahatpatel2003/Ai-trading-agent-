from __future__ import annotations

from nqlab.discovery.objective import objective_score, rank_rows


def _row(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "final_verdict": "REJECT", "oos_net": 100.0, "risk_of_ruin": 0.0,
        "blown": False, "max_drawdown": 50.0, "deflated_sharpe": 0.5,
        "walk_forward_frac": 0.6, "positive_years_frac": 1.0, "cost_stress_survived": True,
        "objective_score": 0.0,
    }
    base.update(over)
    base["objective_score"] = objective_score(base)
    return base


def test_blown_scores_below_survivor() -> None:
    good = _row(oos_net=200.0)
    blown = _row(oos_net=200.0, blown=True)
    assert objective_score(blown) < objective_score(good)


def test_negative_oos_scores_below_survivor() -> None:
    assert objective_score(_row(oos_net=-100.0)) < objective_score(_row(oos_net=100.0))


def test_leaderboard_ranks_accept_above_reject() -> None:
    accept = _row(final_verdict="ACCEPT", oos_net=150.0)
    reject = _row(final_verdict="REJECT", oos_net=150.0)
    nmd = _row(final_verdict="NEEDS_MORE_DATA", oos_net=150.0)
    ranked = rank_rows([reject, accept, nmd])
    assert [r["final_verdict"] for r in ranked] == ["ACCEPT", "NEEDS_MORE_DATA", "REJECT"]


def test_lower_ruin_ranks_higher_among_survivors() -> None:
    low = _row(final_verdict="ACCEPT", risk_of_ruin=0.0)
    high = _row(final_verdict="ACCEPT", risk_of_ruin=0.5, oos_net=100.0)
    ranked = rank_rows([high, low])
    assert ranked[0] is low
