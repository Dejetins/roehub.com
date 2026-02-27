from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from apps.api.dto import (
    BacktestsPostRequest,
    build_backtest_run_request,
    decode_backtest_jobs_cursor,
    decode_backtest_jobs_state,
    encode_backtest_jobs_cursor,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor


def test_backtest_jobs_cursor_codec_roundtrip_is_deterministic() -> None:
    """
    Verify cursor codec roundtrip preserves payload and emits stable opaque value.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cursor transport format is canonical `base64url(json)` without padding.
    Raises:
        AssertionError: If encoded cursor drifts or cannot be decoded back.
    Side Effects:
        None.
    """
    cursor = BacktestJobListCursor(
        created_at=datetime(2026, 2, 23, 11, 45, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000991"),
    )

    encoded = encode_backtest_jobs_cursor(cursor=cursor)
    assert encoded is not None
    assert "=" not in encoded
    assert encoded == (
        "eyJjcmVhdGVkX2F0IjoiMjAyNi0wMi0yM1QxMTo0NTowMCswMDowMCIsImpvYl9pZCI6"
        "IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDk5MSJ9"
    )
    assert decode_backtest_jobs_cursor(cursor=encoded) == cursor



def test_backtest_jobs_state_decode_accepts_blank_and_valid_values() -> None:
    """
    Verify state decoder accepts blank values as `None` and normalizes valid literals.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Legacy clients may send blank `state` query values.
    Raises:
        AssertionError: If blank or valid values are not normalized correctly.
    Side Effects:
        None.
    """
    assert decode_backtest_jobs_state(state=None) is None
    assert decode_backtest_jobs_state(state="") is None
    assert decode_backtest_jobs_state(state="   ") is None
    assert decode_backtest_jobs_state(state="RUNNING") == "running"


def test_backtest_jobs_state_decode_rejects_unknown_values() -> None:
    """
    Verify state decoder rejects unknown non-empty values with deterministic error details.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed state literals are fixed by Backtest Jobs API v1 contract.
    Raises:
        AssertionError: If unknown state value is accepted.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        decode_backtest_jobs_state(state="done")

    assert error_info.value.errors == (
        {
            "path": "query.state",
            "code": "invalid_value",
            "message": "state must be one of: queued, running, succeeded, failed, cancelled",
        },
    )


def test_backtest_jobs_cursor_decode_returns_none_for_blank_cursor() -> None:
    """
    Verify cursor decoder treats blank query values as missing cursor for compatibility.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Jobs list endpoint accepts `cursor=` from older UI links.
    Raises:
        AssertionError: If blank cursor still maps to validation error.
    Side Effects:
        None.
    """
    assert decode_backtest_jobs_cursor(cursor="   ") is None


def test_backtest_jobs_cursor_decode_rejects_non_base64_payload() -> None:
    """
    Verify cursor decoder rejects malformed base64 payloads with deterministic error details.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Malformed payload maps to `BacktestValidationError` with `query.cursor` path.
    Raises:
        AssertionError: If decoder accepts malformed cursor.
    Side Effects:
        None.
    """
    with pytest.raises(BacktestValidationError) as error_info:
        decode_backtest_jobs_cursor(cursor="%%%")

    assert error_info.value.errors == (
        {
            "path": "query.cursor",
            "code": "invalid_cursor",
            "message": "cursor must be base64url(json)",
        },
    )



def test_backtest_jobs_cursor_decode_rejects_invalid_payload_shape() -> None:
    """
    Verify decoder rejects valid base64/json cursor when required keys are missing.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Underlying cursor VO parser validates required `created_at` and `job_id` keys.
    Raises:
        AssertionError: If invalid payload shape is accepted.
    Side Effects:
        None.
    """
    malformed = "eyJjcmVhdGVkX2F0IjoiMjAyNi0wMi0yM1QxMTo0NTowMCswMDowMCJ9"

    with pytest.raises(BacktestValidationError) as error_info:
        decode_backtest_jobs_cursor(cursor=malformed)

    assert error_info.value.errors == (
        {
            "path": "query.cursor",
            "code": "invalid_cursor",
            "message": "cursor must be base64url(json)",
        },
    )



def test_backtest_jobs_cursor_encode_returns_none_for_empty_cursor() -> None:
    """
    Verify encoder returns `None` when list page does not have next cursor.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        API list response uses nullable `next_cursor` field.
    Raises:
        AssertionError: If encoder returns non-null value for empty cursor.
    Side Effects:
        None.
    """
    assert encode_backtest_jobs_cursor(cursor=None) is None


def test_backtest_jobs_create_request_accepts_ranking_block() -> None:
    """
    Verify shared jobs-create envelope accepts ranking block and normalizes metric literals.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Jobs create endpoint reuses `BacktestsPostRequest` contract from sync endpoint.
    Raises:
        AssertionError: If ranking override is dropped or not normalized.
    Side Effects:
        None.
    """
    request = BacktestsPostRequest.model_validate(
        {
            "time_range": {
                "start": datetime(2026, 2, 23, 0, 0, tzinfo=timezone.utc),
                "end": datetime(2026, 2, 23, 1, 0, tzinfo=timezone.utc),
            },
            "template": {
                "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
                "timeframe": "1m",
                "indicator_grids": [
                    {
                        "indicator_id": "ma.sma",
                        "params": {
                            "window": {"mode": "explicit", "values": [20]},
                        },
                    }
                ],
            },
            "ranking": {
                "primary_metric": "TOTAL_RETURN_PCT",
                "secondary_metric": "MAX_DRAWDOWN_PCT",
            },
        }
    )

    run_request = build_backtest_run_request(request=request)
    assert run_request.ranking is not None
    assert run_request.ranking.primary_metric == "total_return_pct"
    assert run_request.ranking.secondary_metric == "max_drawdown_pct"
