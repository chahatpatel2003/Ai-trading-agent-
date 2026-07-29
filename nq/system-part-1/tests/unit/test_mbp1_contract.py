from __future__ import annotations

from nqlab.discovery.candidates import CvdTrendConfirm, ImbalanceContinuation
from nqlab.domain.strategy.base import MBP1_REQUIRED
from nqlab.features.build import build_features
from tests.synth import add_synth_mbp1_columns, feature_bars

_IMB_PARAMS = {
    "imb_thr": 0.2, "stop_atr": 1.0, "target_atr": 2.0,
    "trend_filter": "none", "vol_regime": "any",
}


def _trades_only_frame():
    return build_features(feature_bars(days=2), rth_start_min=570)


def test_mbp1_family_declares_requirement() -> None:
    strat = ImbalanceContinuation(_IMB_PARAMS)
    assert strat.data_requirement == MBP1_REQUIRED
    assert "tob_imbalance" in strat.required_columns


def test_mbp1_strategy_skips_when_columns_absent() -> None:
    frame = _trades_only_frame()  # trades-only: no tob_imbalance
    out = ImbalanceContinuation(_IMB_PARAMS).signals(frame)
    assert int(out["signal"].abs().sum()) == 0  # explicit skip, not a crash


def test_mbp1_strategy_active_when_columns_present() -> None:
    frame = add_synth_mbp1_columns(_trades_only_frame())
    out = ImbalanceContinuation(_IMB_PARAMS).signals(frame)
    assert int(out["signal"].abs().sum()) > 0


def test_cvd_family_trailing_attached() -> None:
    strat = CvdTrendConfirm(
        {"ema_span": 20, "pullback": "none", "stop_atr": 1.5, "target_atr": 2.5, "trail": "atr"}
    )
    assert strat.trailing is not None
    assert strat.trailing.mode == "atr"
