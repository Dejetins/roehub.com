from __future__ import annotations

from datetime import datetime, timedelta, timezone

from trading.contexts.backtest.application.services import (
    BacktestJobSnapshotCadenceV1,
    BacktestJobTopKBufferV1,
    BacktestJobTopVariantCandidateV1,
)
from trading.contexts.indicators.application.dto import IndicatorVariantSelection


def _candidate(*, variant_key: str, total_return_pct: float) -> BacktestJobTopVariantCandidateV1:
    """
    Build deterministic Stage-B candidate fixture for top-k buffer tests.

    Args:
        variant_key: Canonical variant key.
        total_return_pct: Candidate total return metric.
    Returns:
        BacktestJobTopVariantCandidateV1: Prepared candidate fixture.
    Assumptions:
        Variant keys are pre-normalized lowercase sha256-like literals.
    Raises:
        None.
    Side Effects:
        None.
    """
    return BacktestJobTopVariantCandidateV1(
        variant_index=0,
        variant_key=variant_key,
        indicator_variant_key="f" * 64,
        total_return_pct=total_return_pct,
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ema",
                inputs={"source": "close"},
                params={"length": 10},
            ),
        ),
        signal_params={"ema": {"threshold": 1}},
        risk_params={
            "sl_enabled": False,
            "sl_pct": None,
            "tp_enabled": False,
            "tp_pct": None,
        },
    )


def test_snapshot_cadence_should_persist_uses_or_semantics() -> None:
    """
    Verify snapshot cadence persists when either time or processed-step condition is met.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage-B counters are monotonic and timestamps are UTC-aware.
    Raises:
        AssertionError: If OR trigger contract is violated.
    Side Effects:
        None.
    """
    cadence = BacktestJobSnapshotCadenceV1(
        snapshot_seconds=30,
        snapshot_variants_step=100,
    )
    last_persist_at = datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc)

    assert (
        cadence.should_persist(
            now=last_persist_at + timedelta(seconds=31),
            last_persist_at=last_persist_at,
            processed_variants=5,
            last_persist_processed_variants=5,
        )
        is True
    )
    assert (
        cadence.should_persist(
            now=last_persist_at + timedelta(seconds=10),
            last_persist_at=last_persist_at,
            processed_variants=105,
            last_persist_processed_variants=5,
        )
        is True
    )
    assert (
        cadence.should_persist(
            now=last_persist_at + timedelta(seconds=10),
            last_persist_at=last_persist_at,
            processed_variants=50,
            last_persist_processed_variants=5,
        )
        is False
    )


def test_top_k_buffer_keeps_deterministic_rank_order() -> None:
    """
    Verify streaming top-k buffer keeps deterministic rank ordering and bounded capacity.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Ranking key is `total_return_pct DESC, variant_key ASC`.
    Raises:
        AssertionError: If retained rows violate deterministic order or capacity.
    Side Effects:
        None.
    """
    buffer = BacktestJobTopKBufferV1(limit=2)
    buffer.include(candidate=_candidate(variant_key="b" * 64, total_return_pct=10.0))
    buffer.include(candidate=_candidate(variant_key="a" * 64, total_return_pct=10.0))
    buffer.include(candidate=_candidate(variant_key="c" * 64, total_return_pct=11.0))

    ranked = buffer.ranked()
    assert len(ranked) == 2
    assert ranked[0].variant_key == "c" * 64
    assert ranked[1].variant_key == "a" * 64
