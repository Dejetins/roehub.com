from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
)
from trading.contexts.backtest.domain.errors import BacktestJobTransitionError
from trading.shared_kernel.primitives import UserId


def test_backtest_job_saved_mode_requires_spec_snapshot() -> None:
    """
    Verify saved-mode aggregate creation requires `spec_hash` and `spec_payload_json`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Saved-mode storage contract requires full strategy snapshot for reproducibility.
    Raises:
        AssertionError: If missing snapshot fields do not raise transition error.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 22, 18, 0, tzinfo=timezone.utc)

    with pytest.raises(BacktestJobTransitionError, match="spec_hash"):
        BacktestJob.create_queued(
            job_id=UUID("00000000-0000-0000-0000-000000000901"),
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000101"),
            mode="saved",
            created_at=now,
            request_json={"mode": "saved"},
            request_hash="a" * 64,
            spec_hash=None,
            spec_payload_json={"schema_version": 1},
            engine_params_hash="b" * 64,
            backtest_runtime_config_hash="c" * 64,
        )



def test_backtest_job_template_mode_rejects_spec_snapshot_fields() -> None:
    """
    Verify template-mode aggregate rejects saved-mode snapshot fields.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Template mode must keep `spec_hash/spec_payload_json` null in storage.
    Raises:
        AssertionError: If template-mode accepts saved snapshot fields.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 22, 18, 0, tzinfo=timezone.utc)

    with pytest.raises(BacktestJobTransitionError, match="template"):
        BacktestJob.create_queued(
            job_id=UUID("00000000-0000-0000-0000-000000000902"),
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000102"),
            mode="template",
            created_at=now,
            request_json={"mode": "template"},
            request_hash="a" * 64,
            spec_hash="d" * 64,
            spec_payload_json={"schema_version": 1},
            engine_params_hash="b" * 64,
            backtest_runtime_config_hash="c" * 64,
        )



def test_backtest_job_forbids_queued_to_failed_transition() -> None:
    """
    Verify state machine forbids lifecycle transition `queued -> failed`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Contract explicitly disallows direct failure before claim/run.
    Raises:
        AssertionError: If forbidden transition is allowed.
    Side Effects:
        None.
    """
    queued = _build_queued_job()

    with pytest.raises(BacktestJobTransitionError, match="invalid transition"):
        queued.finish(
            next_state="failed",
            changed_at=queued.updated_at + timedelta(seconds=5),
            last_error="runner crashed",
            last_error_json=BacktestJobErrorPayload(
                code="internal_error",
                message="Runner crashed",
                details={"stage": "stage_a"},
            ),
        )



def test_backtest_job_claim_sets_running_state_and_lease_fields() -> None:
    """
    Verify claim transition updates running lease fields and increments attempt.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Claim path sets `state=running` and starts lease ownership.
    Raises:
        AssertionError: If claim invariants are not reflected in returned snapshot.
    Side Effects:
        None.
    """
    queued = _build_queued_job()
    claimed_at = queued.updated_at + timedelta(seconds=1)

    claimed = queued.claim(
        changed_at=claimed_at,
        locked_by="worker-a-123",
        lease_expires_at=claimed_at + timedelta(seconds=60),
    )

    assert claimed.state == "running"
    assert claimed.stage == "stage_a"
    assert claimed.started_at == claimed_at
    assert claimed.locked_by == "worker-a-123"
    assert claimed.locked_at == claimed_at
    assert claimed.lease_expires_at == claimed_at + timedelta(seconds=60)
    assert claimed.heartbeat_at == claimed_at
    assert claimed.attempt == 1



def test_backtest_job_request_cancel_converts_queued_to_cancelled() -> None:
    """
    Verify cancel request immediately transitions queued jobs to terminal cancelled state.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `queued` jobs are cancelled immediately without worker execution.
    Raises:
        AssertionError: If cancel transition fields are incorrect.
    Side Effects:
        None.
    """
    queued = _build_queued_job()
    cancelled_at = queued.updated_at + timedelta(seconds=2)

    cancelled = queued.request_cancel(changed_at=cancelled_at)

    assert cancelled.state == "cancelled"
    assert cancelled.finished_at == cancelled_at
    assert cancelled.cancel_requested_at == cancelled_at
    assert cancelled.locked_by is None



def test_backtest_job_finish_failed_requires_error_payload() -> None:
    """
    Verify failed terminal transition requires both `last_error` and `last_error_json`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Failed jobs persist RoehubError-like payload in storage contract.
    Raises:
        AssertionError: If missing payload fields are accepted.
    Side Effects:
        None.
    """
    running = _build_queued_job().claim(
        changed_at=datetime(2026, 2, 22, 18, 0, 1, tzinfo=timezone.utc),
        locked_by="worker-a-123",
        lease_expires_at=datetime(2026, 2, 22, 18, 1, tzinfo=timezone.utc),
    )

    with pytest.raises(BacktestJobTransitionError, match="last_error_json"):
        running.finish(
            next_state="failed",
            changed_at=datetime(2026, 2, 22, 18, 0, 30, tzinfo=timezone.utc),
            last_error="execution failed",
            last_error_json=None,
        )



def test_backtest_job_update_progress_rejects_backward_stage_transition() -> None:
    """
    Verify progress updates reject stage rollback (`stage_b -> stage_a`).

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage progression is monotonic in Backtest jobs state machine.
    Raises:
        AssertionError: If backward stage update is allowed.
    Side Effects:
        None.
    """
    running = _build_queued_job().claim(
        changed_at=datetime(2026, 2, 22, 18, 0, 1, tzinfo=timezone.utc),
        locked_by="worker-a-123",
        lease_expires_at=datetime(2026, 2, 22, 18, 1, tzinfo=timezone.utc),
    )
    stage_b = running.update_progress(
        changed_at=datetime(2026, 2, 22, 18, 0, 10, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=10,
        total_units=100,
    )

    with pytest.raises(BacktestJobTransitionError, match="cannot move stage backward"):
        stage_b.update_progress(
            changed_at=datetime(2026, 2, 22, 18, 0, 11, tzinfo=timezone.utc),
            stage="stage_a",
            processed_units=11,
            total_units=100,
        )



def _build_queued_job() -> BacktestJob:
    """
    Build deterministic queued job fixture for state-machine unit tests.

    Args:
        None.
    Returns:
        BacktestJob: Valid queued Backtest job aggregate.
    Assumptions:
        Fixture uses template mode and canonical hash literals.
    Raises:
        BacktestJobTransitionError: If fixture violates aggregate invariants.
    Side Effects:
        None.
    """
    created_at = datetime(2026, 2, 22, 18, 0, tzinfo=timezone.utc)
    return BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000900"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000100"),
        mode="template",
        created_at=created_at,
        request_json={"mode": "template", "schema_version": 1},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
    )
