from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobStageWeights,
)
from trading.contexts.backtest.domain.errors import (
    BacktestJobLeaseError,
    BacktestJobTransitionError,
)
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


def test_backtest_job_preserves_artifact_pin_across_claim_and_finish() -> None:
    """
    Verify immutable artifact pin metadata survives state transitions for active jobs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Reproducibility requires slot identity to remain fixed once pinned at creation time.
    Raises:
        AssertionError: If claim/finish transitions mutate artifact pin metadata.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 22, 18, 0, tzinfo=timezone.utc)
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000950"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000101"),
        mode="template",
        created_at=now,
        request_json={"mode": "template"},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=9,
            artifact_manifest_hash="d" * 64,
            artifact_asof_date="2026-03-24",
        ),
    )

    claimed = queued.claim(
        changed_at=now + timedelta(seconds=1),
        locked_by="worker-a-123",
        lease_expires_at=now + timedelta(seconds=61),
    )
    finished = claimed.finish(
        next_state="succeeded",
        changed_at=now + timedelta(seconds=5),
    )

    assert claimed.artifact_pin == queued.artifact_pin
    assert finished.artifact_pin == queued.artifact_pin


def test_backtest_job_preserves_normalized_persisted_run_metadata_across_transitions() -> None:
    """
    Verify additive persisted-run metadata is normalized on create and preserved by transitions.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R7-01 keeps one storage family for background and inline runs, so metadata must survive
        state transitions unchanged.
    Raises:
        AssertionError: If normalization or transition preservation drifts.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 22, 18, 0, tzinfo=timezone.utc)
    queued = BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000951"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000101"),
        mode="template",
        created_at=now,
        request_json={"mode": "template"},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        execution_mode="background_manual_legacy",
        execution_profile_mode_hint="hybrid_conservative",
        effective_execution_profile_mode="hybrid_conservative",
        market_id=1,
        symbol="btcusdt",
        timeframe="1H",
        requested_top_n=25,
        ranking_primary_metric=" total_return_pct ",
        ranking_secondary_metric=" profit_factor ",
    )

    claimed = queued.claim(
        changed_at=now + timedelta(seconds=1),
        locked_by="worker-a-123",
        lease_expires_at=now + timedelta(seconds=61),
    )
    finished = claimed.finish(
        next_state="succeeded",
        changed_at=now + timedelta(seconds=5),
    )

    assert queued.execution_mode == "background_manual_legacy"
    assert queued.execution_profile_mode_hint == "hybrid_conservative"
    assert queued.effective_execution_profile_mode == "hybrid_conservative"
    assert queued.market_id == 1
    assert queued.symbol == "BTCUSDT"
    assert queued.timeframe == "1h"
    assert queued.requested_top_n == 25
    assert queued.ranking_primary_metric == "total_return_pct"
    assert queued.ranking_secondary_metric == "profit_factor"
    assert claimed.execution_mode == queued.execution_mode
    assert claimed.execution_profile_mode_hint == queued.execution_profile_mode_hint
    assert (
        claimed.effective_execution_profile_mode
        == queued.effective_execution_profile_mode
    )
    assert claimed.market_id == queued.market_id
    assert claimed.symbol == queued.symbol
    assert claimed.timeframe == queued.timeframe
    assert claimed.requested_top_n == queued.requested_top_n
    assert claimed.ranking_primary_metric == queued.ranking_primary_metric
    assert claimed.ranking_secondary_metric == queued.ranking_secondary_metric
    assert finished.execution_mode == queued.execution_mode
    assert finished.execution_profile_mode_hint == queued.execution_profile_mode_hint
    assert (
        finished.effective_execution_profile_mode
        == queued.effective_execution_profile_mode
    )
    assert finished.market_id == queued.market_id
    assert finished.symbol == queued.symbol
    assert finished.timeframe == queued.timeframe
    assert finished.requested_top_n == queued.requested_top_n
    assert finished.ranking_primary_metric == queued.ranking_primary_metric
    assert finished.ranking_secondary_metric == queued.ranking_secondary_metric


def test_backtest_job_rejects_execution_profile_metadata_without_persisted_run_metadata() -> None:
    """
    Verify additive execution-profile metadata requires the persisted-run metadata contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `execution_profile_mode_hint` and `effective_execution_profile_mode` are stored only for
        persisted jobs/runs that already carry `execution_mode` and denormalized run metadata.
    Raises:
        AssertionError: If profile metadata is accepted on a plain queued snapshot.
    Side Effects:
        None.
    """
    now = datetime(2026, 2, 22, 18, 0, tzinfo=timezone.utc)

    with pytest.raises(
        BacktestJobTransitionError,
        match="execution-profile metadata requires persisted run metadata",
    ):
        BacktestJob.create_queued(
            job_id=UUID("00000000-0000-0000-0000-000000000954"),
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000101"),
            mode="template",
            created_at=now,
            request_json={"mode": "template"},
            request_hash="a" * 64,
            spec_hash=None,
            spec_payload_json=None,
            engine_params_hash="b" * 64,
            backtest_runtime_config_hash="c" * 64,
            execution_profile_mode_hint="exact_parallel",
        )


def test_backtest_job_rejects_lease_fields_outside_running_state() -> None:
    """
    Verify aggregate rejects non-null lease fields when state is not `running`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Lease tuple is valid only for running jobs.
    Raises:
        AssertionError: If queued state accepts lease fields.
    Side Effects:
        None.
    """
    queued = _build_queued_job()

    with pytest.raises(BacktestJobLeaseError, match="lease fields must be null"):
        BacktestJob(
            job_id=queued.job_id,
            user_id=queued.user_id,
            mode=queued.mode,
            state=queued.state,
            created_at=queued.created_at,
            updated_at=queued.updated_at + timedelta(seconds=1),
            started_at=queued.started_at,
            finished_at=queued.finished_at,
            cancel_requested_at=queued.cancel_requested_at,
            request_json=queued.request_json,
            request_hash=queued.request_hash,
            spec_hash=queued.spec_hash,
            spec_payload_json=queued.spec_payload_json,
            engine_params_hash=queued.engine_params_hash,
            backtest_runtime_config_hash=queued.backtest_runtime_config_hash,
            stage=queued.stage,
            processed_units=queued.processed_units,
            total_units=queued.total_units,
            progress_updated_at=queued.progress_updated_at,
            locked_by="worker-a-123",
            locked_at=queued.updated_at + timedelta(seconds=1),
            lease_expires_at=queued.updated_at + timedelta(seconds=61),
            heartbeat_at=queued.updated_at + timedelta(seconds=1),
            attempt=queued.attempt,
            last_error=queued.last_error,
            last_error_json=queued.last_error_json,
        )


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



def test_backtest_job_request_cancel_for_running_marks_cancel_requested_only() -> None:
    """
    Verify running cancel request sets marker and keeps job in running state.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Running cancel is deferred to worker batch boundaries.
    Raises:
        AssertionError: If running cancel transitions to terminal state immediately.
    Side Effects:
        None.
    """
    running = _build_queued_job().claim(
        changed_at=datetime(2026, 2, 22, 18, 0, 1, tzinfo=timezone.utc),
        locked_by="worker-a-123",
        lease_expires_at=datetime(2026, 2, 22, 18, 1, tzinfo=timezone.utc),
    )
    cancel_requested_at = datetime(2026, 2, 22, 18, 0, 30, tzinfo=timezone.utc)

    updated = running.request_cancel(changed_at=cancel_requested_at)

    assert updated.state == "running"
    assert updated.cancel_requested_at == cancel_requested_at
    assert updated.finished_at is None
    assert updated.locked_by == "worker-a-123"


def test_backtest_job_request_cancel_for_running_preserves_first_marker() -> None:
    """
    Verify repeated running cancel requests keep the first persisted marker unchanged.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R8-03 keeps running cancel best-effort but fixes the first `cancel_requested_at`
        timestamp for deterministic lifecycle visibility.
    Raises:
        AssertionError: If repeated cancel mutates the running snapshot.
    Side Effects:
        None.
    """
    running = _build_queued_job().claim(
        changed_at=datetime(2026, 2, 22, 18, 0, 1, tzinfo=timezone.utc),
        locked_by="worker-a-123",
        lease_expires_at=datetime(2026, 2, 22, 18, 1, tzinfo=timezone.utc),
    )
    first_cancel_at = datetime(2026, 2, 22, 18, 0, 30, tzinfo=timezone.utc)
    repeated_cancel_at = datetime(2026, 2, 22, 18, 0, 45, tzinfo=timezone.utc)

    first = running.request_cancel(changed_at=first_cancel_at)
    repeated = first.request_cancel(changed_at=repeated_cancel_at)

    assert repeated is first
    assert repeated.state == "running"
    assert repeated.cancel_requested_at == first_cancel_at
    assert repeated.updated_at == first_cancel_at


def test_backtest_job_request_cancel_is_idempotent_for_terminal_state() -> None:
    """
    Verify repeated cancel requests on terminal job are idempotent.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Terminal jobs must not mutate on repeated cancel.
    Raises:
        AssertionError: If second cancel mutates terminal snapshot.
    Side Effects:
        None.
    """
    cancelled = _build_queued_job().request_cancel(
        changed_at=datetime(2026, 2, 22, 18, 0, 2, tzinfo=timezone.utc)
    )

    repeated = cancelled.request_cancel(
        changed_at=datetime(2026, 2, 22, 18, 0, 3, tzinfo=timezone.utc)
    )

    assert repeated is cancelled
    assert repeated.state == "cancelled"
    assert repeated.finished_at == cancelled.finished_at


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


def test_backtest_job_progress_percent_uses_stage_weights_and_marks_success_as_complete() -> None:
    """
    Verify weighted progress percent follows stage weights and succeeded runs clamp to `100`.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public persisted-run progress is projected onto a deterministic weighted `0..100%` scale.
    Raises:
        AssertionError: If stage-weight projection or succeeded terminal override drifts.
    Side Effects:
        None.
    """
    weights = BacktestJobStageWeights(stage_a=25, stage_b=70, finalizing=5)
    running = _build_queued_job().claim(
        changed_at=datetime(2026, 2, 22, 18, 0, 1, tzinfo=timezone.utc),
        locked_by="worker-a-123",
        lease_expires_at=datetime(2026, 2, 22, 18, 1, tzinfo=timezone.utc),
    )
    stage_b = running.update_progress(
        changed_at=datetime(2026, 2, 22, 18, 0, 30, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=10,
        total_units=20,
    )
    succeeded = stage_b.finish(
        next_state="succeeded",
        changed_at=datetime(2026, 2, 22, 18, 0, 40, tzinfo=timezone.utc),
    )

    assert stage_b.progress_percent(stage_weights=weights) == 60
    assert succeeded.progress_percent(stage_weights=weights) == 100


def test_backtest_job_eta_seconds_requires_signal_and_estimates_from_weighted_progress() -> None:
    """
    Verify ETA returns `None` without signal and uses weighted progress once it is defensible.

    Docs:
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/domain/entities/test_backtest_job_entities.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
      - src/trading/contexts/backtest/application/use_cases/backtest_runs_history_api_v1.py
    Args:
        None.
    Returns:
        None.
    Assumptions:
        ETA uses only the current run timeline and weighted progress, without benchmark fallback.
    Raises:
        AssertionError: If nullability or remaining-seconds projection drifts.
    Side Effects:
        None.
    """
    weights = BacktestJobStageWeights(stage_a=25, stage_b=70, finalizing=5)
    running = _build_queued_job().claim(
        changed_at=datetime(2026, 2, 22, 18, 0, 1, tzinfo=timezone.utc),
        locked_by="worker-a-123",
        lease_expires_at=datetime(2026, 2, 22, 18, 1, tzinfo=timezone.utc),
    )
    progressed = running.update_progress(
        changed_at=datetime(2026, 2, 22, 18, 1, 1, tzinfo=timezone.utc),
        stage="stage_b",
        processed_units=10,
        total_units=20,
    )

    assert running.eta_seconds(
        stage_weights=weights,
        now=datetime(2026, 2, 22, 18, 0, 30, tzinfo=timezone.utc),
    ) is None
    assert progressed.eta_seconds(
        stage_weights=weights,
        now=datetime(2026, 2, 22, 18, 1, 1, tzinfo=timezone.utc),
    ) == 40



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
