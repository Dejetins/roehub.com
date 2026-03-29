from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from apps.api.dto import (
    BacktestRunVariantReportPostRequest,
    build_backtest_run_status_response,
    build_backtest_run_top_response,
    decode_backtest_runs_cursor,
    decode_backtest_runs_state,
    encode_backtest_runs_cursor,
)
from trading.contexts.backtest.application.use_cases import BacktestRunTopReadResult
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.shared_kernel.primitives import UserId


def test_backtest_runs_cursor_codec_roundtrip_is_deterministic() -> None:
    """
    Verify public runs cursor codec preserves payload and emits stable opaque transport.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public runs reuse the canonical `base64url(json)` cursor contract.
    Raises:
        AssertionError: If encoded cursor drifts or cannot be decoded back.
    Side Effects:
        None.
    """
    cursor = BacktestJobListCursor(
        created_at=datetime(2026, 3, 29, 11, 45, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000991"),
    )

    encoded = encode_backtest_runs_cursor(cursor=cursor)
    assert encoded is not None
    assert "=" not in encoded
    assert decode_backtest_runs_cursor(cursor=encoded) == cursor


def test_backtest_runs_state_decode_accepts_blank_and_valid_values() -> None:
    """
    Verify public runs state decoder accepts blank values as `None` and normalizes literals.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public runs preserve blank-state compatibility during migration from legacy jobs UI.
    Raises:
        AssertionError: If blank or valid values are not normalized correctly.
    Side Effects:
        None.
    """
    assert decode_backtest_runs_state(state=None) is None
    assert decode_backtest_runs_state(state="") is None
    assert decode_backtest_runs_state(state="   ") is None
    assert decode_backtest_runs_state(state="RUNNING") == "running"


def test_backtest_runs_state_decode_rejects_unknown_values() -> None:
    """
    Verify public runs state decoder rejects unknown non-empty values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed state literals are shared with the legacy jobs contract.
    Raises:
        AssertionError: If unknown state value is accepted.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        decode_backtest_runs_state(state="done")

    assert error_info.value.errors == (
        {
            "path": "query.state",
            "code": "invalid_value",
            "message": "state must be one of: queued, running, succeeded, failed, cancelled",
        },
    )


def test_build_backtest_run_status_response_uses_run_vocabulary_without_hashes() -> None:
    """
    Verify public status mapper exposes `run_id` metadata and hides legacy hash fields.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public runs payload omits internal reproducibility hashes from the legacy jobs API.
    Raises:
        AssertionError: If mapped fields drift from the public contract.
    Side Effects:
        None.
    """
    run = _failed_run(run_id=UUID("00000000-0000-0000-0000-000000000992"))

    response = build_backtest_run_status_response(run=run)
    dumped = response.model_dump(mode="json")

    assert dumped["run_id"] == "00000000-0000-0000-0000-000000000992"
    assert dumped["execution_mode"] == "sync_inline"
    assert dumped["market_id"] == 1
    assert dumped["artifact_slot"] == "slot_b"
    assert dumped["last_error_json"]["code"] == "unexpected_error"
    assert "request_hash" not in dumped


def test_build_backtest_run_top_response_includes_summary_metrics_fields() -> None:
    """
    Verify public top mapper exposes persisted summary metrics and TP/SL columns.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public `/top` rows stay summary-only and deterministic.
    Raises:
        AssertionError: If persisted summary fields are dropped.
    Side Effects:
        None.
    """
    run = _queued_run(run_id=UUID("00000000-0000-0000-0000-000000000993"))
    row = BacktestJobTopVariant(
        job_id=run.job_id,
        rank=1,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        variant_index=0,
        total_return_pct=12.34,
        payload_json={"schema_version": 1},
        summary_metrics_json={"total_return_pct": 12.34, "profit_factor": 1.23},
        best_tp_pct=4.0,
        best_sl_pct=2.0,
        report_table_md=None,
        trades_json=None,
        updated_at=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    )

    response = build_backtest_run_top_response(
        result=BacktestRunTopReadResult(job=run, rows=(row,))
    )

    assert response.model_dump(mode="json") == {
        "run_id": "00000000-0000-0000-0000-000000000993",
        "state": "queued",
        "execution_mode": "sync_inline",
        "items": [
            {
                "rank": 1,
                "variant_key": "a" * 64,
                "indicator_variant_key": "b" * 64,
                "variant_index": 0,
                "total_return_pct": 12.34,
                "payload": {"schema_version": 1},
                "summary_metrics_json": {
                    "total_return_pct": 12.34,
                    "profit_factor": 1.23,
                },
                "best_tp_pct": 4.0,
                "best_sl_pct": 2.0,
            }
        ],
    }


def test_backtest_run_variant_report_request_accepts_minimal_run_scoped_payload() -> None:
    """
    Verify run-scoped variant-report request DTO accepts only selected variant + flag payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `run_id` is carried by route path and must not be duplicated in body contract.
    Raises:
        AssertionError: If parsed payload drifts from strict request shape.
    Side Effects:
        None.
    """
    request = BacktestRunVariantReportPostRequest.model_validate(
        {
            "include_trades": True,
            "variant": {
                "indicator_selections": [
                    {
                        "indicator_id": "ma.sma",
                        "inputs": {"source": "close"},
                        "params": {"window": 20},
                    }
                ],
                "signal_params": {"ma.sma": {"cross_up": 0.5}},
                "risk_params": {"sl_enabled": True, "sl_pct": 2.0},
                "execution_params": {"fee_pct": 0.075},
                "direction_mode": "long-short",
                "sizing_mode": "all_in",
            },
        }
    )

    assert request.include_trades is True
    assert request.variant.direction_mode == "long-short"


def test_backtest_run_variant_report_request_rejects_full_run_envelope_fields() -> None:
    """
    Verify run-scoped variant-report request DTO forbids legacy full run-context body fields.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        New endpoint reconstructs original run context from persisted storage, not client body.
    Raises:
        AssertionError: If extra legacy body fields are accepted.
    Side Effects:
        None.
    """
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BacktestRunVariantReportPostRequest.model_validate(
            {
                "template": {"instrument_id": {"market_id": 1, "symbol": "BTCUSDT"}},
                "variant": {
                    "indicator_selections": [
                        {
                            "indicator_id": "ma.sma",
                            "inputs": {"source": "close"},
                            "params": {"window": 20},
                        }
                    ],
                    "signal_params": {"ma.sma": {"cross_up": 0.5}},
                    "risk_params": {"sl_enabled": True},
                    "execution_params": {"fee_pct": 0.075},
                    "direction_mode": "long-short",
                    "sizing_mode": "all_in",
                },
            }
        )


def _queued_run(*, run_id: UUID) -> BacktestJob:
    """
    Build deterministic queued persisted run fixture for DTO tests.

    Args:
        run_id: Deterministic persisted run identifier.
    Returns:
        BacktestJob: Queued persisted run fixture.
    Assumptions:
        Additive persisted-run metadata is fully populated.
    Raises:
        ValueError: If fixture violates domain invariants.
    Side Effects:
        None.
    """
    return BacktestJob.create_queued(
        job_id=run_id,
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        created_at=datetime(2026, 3, 29, 11, 30, tzinfo=timezone.utc),
        request_json={"top_k": 25},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=11,
            artifact_manifest_hash="d" * 64,
            artifact_asof_date="2026-03-29",
        ),
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
    )


def _failed_run(*, run_id: UUID) -> BacktestJob:
    """
    Build deterministic failed persisted run fixture with Roehub-like error payload.

    Args:
        run_id: Deterministic persisted run identifier.
    Returns:
        BacktestJob: Failed persisted run fixture.
    Assumptions:
        Failure payload fields are populated for public status mapping.
    Raises:
        ValueError: If fixture violates lifecycle invariants.
    Side Effects:
        None.
    """
    running = _queued_run(run_id=run_id).claim(
        changed_at=datetime(2026, 3, 29, 11, 35, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        lease_expires_at=datetime(2026, 3, 29, 11, 36, tzinfo=timezone.utc),
    )
    return running.finish(
        next_state="failed",
        changed_at=datetime(2026, 3, 29, 11, 37, tzinfo=timezone.utc),
        last_error="Execution failed",
        last_error_json=BacktestJobErrorPayload(
            code="unexpected_error",
            message="Execution failed",
            details={"stage": "stage_b"},
        ),
    )
