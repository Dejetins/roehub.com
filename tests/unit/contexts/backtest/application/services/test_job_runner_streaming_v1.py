from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

from trading.contexts.backtest.application.services import (
    BacktestJobSnapshotCadenceV1,
    BacktestJobTopKBufferV1,
    BacktestJobTopVariantCandidateV1,
    build_finalized_snapshot_rows,
    build_running_snapshot_rows,
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


def _variant_key_from_int(*, value: int) -> str:
    """
    Build deterministic fixed-length lowercase hex variant key from integer fixture value.

    Args:
        value: Non-negative integer used as key fixture source.
    Returns:
        str: Deterministic 64-char lowercase hex key.
    Assumptions:
        Fixed key width keeps lexical order aligned with numeric order.
    Raises:
        ValueError: If value is negative.
    Side Effects:
        None.
    """
    if value < 0:
        raise ValueError("value must be >= 0")
    return f"{value:064x}"


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


def test_top_k_buffer_matches_reference_full_sort_policy_per_insert() -> None:
    """
    Verify heap-based top-k buffer matches full-sort reference policy after each insert.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Reference policy sorts all seen candidates by `total_return_pct DESC, variant_key ASC`.
    Raises:
        AssertionError: If heap buffer differs from deterministic full-sort policy.
    Side Effects:
        None.
    """
    limit = 5
    buffer = BacktestJobTopKBufferV1(limit=limit)
    seen: list[BacktestJobTopVariantCandidateV1] = []
    stream = (
        (7, 11.0),
        (2, 11.0),
        (5, 9.5),
        (1, 11.0),
        (8, 12.0),
        (3, 12.0),
        (0, 11.0),
        (9, 8.0),
        (4, 12.0),
        (6, 11.0),
        (10, 12.0),
    )

    for value, total_return_pct in stream:
        candidate = _candidate(
            variant_key=_variant_key_from_int(value=value),
            total_return_pct=total_return_pct,
        )
        buffer.include(candidate=candidate)
        seen.append(candidate)

        expected = tuple(
            sorted(
                seen,
                key=lambda item: (-item.total_return_pct, item.variant_key),
            )[:limit]
        )
        assert buffer.ranked() == expected


def test_build_running_snapshot_rows_preserves_explicit_source_in_payload_json() -> None:
    """
    Verify persisted running snapshot rows keep explicit `inputs.source` in variant payloads.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Jobs `/top` summary reads `payload_json` directly from persisted snapshot rows.
    Raises:
        AssertionError: If persisted payload drops or mutates explicit source selection.
    Side Effects:
        None.
    """
    candidate = BacktestJobTopVariantCandidateV1(
        variant_index=3,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        total_return_pct=12.5,
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "hlc3"},
                params={"window": 20},
            ),
        ),
        signal_params={"ma.sma": {"cross_up": 0.5}},
        risk_params={
            "sl_enabled": False,
            "sl_pct": None,
            "tp_enabled": False,
            "tp_pct": None,
        },
    )

    rows = build_running_snapshot_rows(
        job_id=UUID("00000000-0000-0000-0000-000000000111"),
        now=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        ranked_candidates=(candidate,),
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"fee_pct": 0.1, "fixed_quote": 100.0},
    )

    assert rows[0].payload_json["indicator_selections"] == [
        {
            "indicator_id": "ma.sma",
            "inputs": {"source": "hlc3"},
            "params": {"window": 20},
        }
    ]


def test_build_finalized_snapshot_rows_remains_summary_only() -> None:
    """
    Verify finalized snapshot builder keeps `report_table_md` and `trades_json` null.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R6-04 persists summary-only top-N rows even for succeeded job snapshots.
    Raises:
        AssertionError: If finalized builder materializes eager report/trades payloads.
    Side Effects:
        None.
    """
    rows = build_finalized_snapshot_rows(
        job_id=UUID("00000000-0000-0000-0000-000000000222"),
        now=datetime(2026, 2, 23, 12, 30, tzinfo=timezone.utc),
        ranked_candidates=(_candidate(variant_key="c" * 64, total_return_pct=11.0),),
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={"fee_pct": 0.1, "fixed_quote": 100.0},
        reports_by_variant_key={"c" * 64: "|Metric|Value|\n|---|---|"},
        trades_by_variant_key={"c" * 64: None},
    )

    assert rows[0].report_table_md is None
    assert rows[0].trades_json is None
