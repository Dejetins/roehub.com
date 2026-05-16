from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Mapping
from uuid import UUID

import pytest

from trading.contexts.backtest.application.ai_configurator import (
    BACKTEST_AI_CONFIG_ERROR_IDEMPOTENCY_CONFLICT,
    BACKTEST_AI_CONFIG_ERROR_NOT_FOUND,
    BacktestAiConfigEvent,
    BacktestAiConfigJob,
    BacktestAiConfigJobsUseCase,
    BacktestAiConfigLlmAttempt,
    BacktestAiConfigTerminalState,
    BacktestAiQuotaConfig,
    BacktestAiQuotaEvent,
    BacktestAiQuotaService,
    BacktestAiTierQuota,
    backtest_ai_prompt_profile_for_mode,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import PaidLevel, UserId


def test_create_persists_queued_job_event_and_single_quota_charge() -> None:
    repository = _Repository()
    use_case = _use_case(repository=repository)
    user_id = _user("00000000-0000-0000-0000-000000000401")

    result = use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        mode="create",
        locale="ru",
        user_prompt_text="Собери конфиг для BTCUSDT на RSI",
        idempotency_key="client-request-1",
        current_config={"top_n": 100},
        ui_context={"runtime_defaults_hash": "a" * 64},
    )

    assert result.job is not None
    assert result.job.state == "queued"
    assert result.job.source_page == "backtests"
    assert result.job.idempotency_key == "client-request-1"
    assert result.job.current_config_hash is not None
    assert result.job.system_prompt_version == "backtest-ai-configurator-v2"
    assert (
        result.job.system_prompt_hash
        == backtest_ai_prompt_profile_for_mode("create").system_prompt_hash
    )
    assert result.job.runtime_defaults_hash == "a" * 64
    assert result.job.quota_charged is True
    assert result.quota_charged is True
    assert result.idempotent_replay is False
    assert result.admission.status == "accepted"
    assert [event.event_name for event in repository.events] == ["queued"]
    assert [event.quota_action for event in repository.quota_events] == [
        "request_charged"
    ]
    assert "user_prompt_text" not in result.job.public_snapshot()


def test_idempotency_replay_returns_existing_job_without_duplicate_charge() -> None:
    repository = _Repository()
    use_case = _use_case(repository=repository)
    user_id = _user("00000000-0000-0000-0000-000000000402")

    first = use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        mode="create",
        locale="en",
        user_prompt_text="Create RSI config",
        idempotency_key="stable-key",
    )
    replay = use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        mode="create",
        locale="en",
        user_prompt_text="Create RSI config",
        idempotency_key="stable-key",
    )

    assert first.job is not None
    assert replay.job is not None
    assert replay.job.job_id == first.job.job_id
    assert replay.idempotent_replay is True
    assert replay.quota_charged is False
    assert len(repository.jobs) == 1
    assert [event.quota_action for event in repository.quota_events] == [
        "request_charged"
    ]


def test_idempotency_key_conflict_rejects_different_logical_request() -> None:
    repository = _Repository()
    use_case = _use_case(repository=repository)
    user_id = _user("00000000-0000-0000-0000-000000000403")

    use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        mode="create",
        locale="en",
        user_prompt_text="Create RSI config",
        idempotency_key="stable-key",
    )

    with pytest.raises(RoehubError) as exc_info:
        use_case.create(
            user_id=user_id,
            paid_level=PaidLevel.free(),
            mode="create",
            locale="en",
            user_prompt_text="Create MACD config",
            idempotency_key="stable-key",
        )

    assert exc_info.value.code == BACKTEST_AI_CONFIG_ERROR_IDEMPOTENCY_CONFLICT
    assert len(repository.jobs) == 1
    assert [event.quota_action for event in repository.quota_events] == [
        "request_charged"
    ]


def test_quota_rejection_records_rejection_without_job_or_charge() -> None:
    repository = _Repository()
    use_case = _use_case(
        repository=repository,
        quota_config=BacktestAiQuotaConfig(
            tier_quotas={
                "free": BacktestAiTierQuota(
                    requests_per_5h=1,
                    requests_per_week=10,
                    max_queued_per_user=10,
                    max_active_user_jobs=10,
                ),
                "base": BacktestAiTierQuota(10, 10, 10, 10),
                "pro": BacktestAiTierQuota(10, 10, 10, 10),
                "ultra": BacktestAiTierQuota(10, 10, 10, 10),
            },
            max_queue_size=10,
        ),
    )
    user_id = _user("00000000-0000-0000-0000-000000000404")

    accepted = use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        mode="create",
        locale="en",
        user_prompt_text="Create RSI config",
    )
    rejected = use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        mode="create",
        locale="en",
        user_prompt_text="Create DEMA config",
    )

    assert accepted.job is not None
    assert rejected.job is None
    assert rejected.admission.status == "quota_exceeded"
    assert rejected.quota_charged is False
    assert len(repository.jobs) == 1
    assert [event.quota_action for event in repository.quota_events] == [
        "request_charged",
        "quota_rejected",
    ]
    assert repository.quota_events[-1].units == 0


def test_owner_scope_hides_foreign_jobs() -> None:
    repository = _Repository()
    use_case = _use_case(repository=repository)
    owner_id = _user("00000000-0000-0000-0000-000000000405")
    foreign_id = _user("00000000-0000-0000-0000-000000000406")
    created = use_case.create(
        user_id=owner_id,
        paid_level=PaidLevel.free(),
        mode="create",
        locale="ru",
        user_prompt_text="Собери конфиг",
    )
    assert created.job is not None

    with pytest.raises(RoehubError) as exc_info:
        use_case.get(user_id=foreign_id, job_id=created.job.job_id)

    assert exc_info.value.code == BACKTEST_AI_CONFIG_ERROR_NOT_FOUND


def test_lease_claim_heartbeat_reclaim_and_attempt_limit_contract() -> None:
    repository = _Repository()
    use_case = _use_case(repository=repository)
    user_id = _user("00000000-0000-0000-0000-000000000407")
    created = use_case.create(
        user_id=user_id,
        paid_level=PaidLevel.free(),
        mode="create",
        locale="en",
        user_prompt_text="Create RSI config",
    )
    assert created.job is not None
    start = datetime(2026, 5, 11, tzinfo=UTC)

    first_claim = repository.claim_next(
        now=start,
        locked_by="worker-a",
        lease_seconds=60,
        max_attempts=2,
    )
    assert first_claim is not None
    assert first_claim.state == "running"
    assert first_claim.attempt == 1
    assert repository.heartbeat(
        job_id=first_claim.job_id,
        now=start + timedelta(seconds=30),
        locked_by="worker-b",
        lease_seconds=60,
    ) is None

    second_claim = repository.claim_next(
        now=start + timedelta(seconds=61),
        locked_by="worker-b",
        lease_seconds=60,
        max_attempts=2,
    )
    assert second_claim is not None
    assert second_claim.locked_by == "worker-b"
    assert second_claim.attempt == 2

    exhausted = repository.claim_next(
        now=start + timedelta(seconds=122),
        locked_by="worker-c",
        lease_seconds=60,
        max_attempts=2,
    )
    final_job = repository.jobs[second_claim.job_id]

    assert exhausted is None
    assert final_job.state == "failed"
    assert final_job.last_error == "lease_attempt_limit_exceeded"


def _use_case(
    *,
    repository: "_Repository",
    quota_config: BacktestAiQuotaConfig | None = None,
) -> BacktestAiConfigJobsUseCase:
    return BacktestAiConfigJobsUseCase(
        repository=repository,
        quota_service=BacktestAiQuotaService(config=quota_config or BacktestAiQuotaConfig()),
    )


def _user(value: str) -> UserId:
    return UserId.from_string(value)


@dataclass
class _Repository:
    jobs: dict[UUID, BacktestAiConfigJob] = field(default_factory=dict)
    events: list[BacktestAiConfigEvent] = field(default_factory=list)
    quota_events: list[BacktestAiQuotaEvent] = field(default_factory=list)
    llm_attempts: list[BacktestAiConfigLlmAttempt] = field(default_factory=list)

    def create_with_quota_event(
        self,
        *,
        job: BacktestAiConfigJob,
        event: BacktestAiConfigEvent,
        quota_event: BacktestAiQuotaEvent,
    ) -> BacktestAiConfigJob:
        self.jobs[job.job_id] = job
        self.events.append(event)
        self.quota_events.append(quota_event)
        return job

    def get(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId | None = None,
    ) -> BacktestAiConfigJob | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if owner_user_id is not None and job.owner_user_id != owner_user_id:
            return None
        return job

    def find_by_idempotency_key(
        self,
        *,
        owner_user_id: UserId,
        idempotency_key: str,
    ) -> BacktestAiConfigJob | None:
        for job in self.jobs.values():
            if (
                job.owner_user_id == owner_user_id
                and job.idempotency_key == idempotency_key
            ):
                return job
        return None

    def record_quota_event(self, *, event: BacktestAiQuotaEvent) -> None:
        self.quota_events.append(event)

    def append_event(self, *, event: BacktestAiConfigEvent) -> None:
        self.events.append(event)

    def record_llm_attempt(self, *, attempt: BacktestAiConfigLlmAttempt) -> None:
        self.llm_attempts.append(attempt)

    def list_events(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId,
    ) -> tuple[BacktestAiConfigEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.job_id == job_id and event.owner_user_id == owner_user_id
        )

    def record_feedback(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId,
        applied: bool,
        feedback_json: Mapping[str, object],
        now: datetime,
    ) -> BacktestAiConfigJob | None:
        job = self.get(job_id=job_id, owner_user_id=owner_user_id)
        if job is None:
            return None
        updated = replace(
            job,
            applied_at=now if applied else job.applied_at,
            user_feedback_json=dict(feedback_json),
            updated_at=now,
        )
        self.jobs[job_id] = updated
        return updated

    def count_quota_events(
        self,
        *,
        owner_user_id: UserId,
        occurred_after: datetime,
    ) -> int:
        return sum(
            1
            for event in self.quota_events
            if event.owner_user_id == owner_user_id
            and event.quota_action == "request_charged"
            and event.units > 0
            and event.occurred_at >= occurred_after
        )

    def count_queued_for_user(self, *, owner_user_id: UserId) -> int:
        return sum(
            1
            for job in self.jobs.values()
            if job.owner_user_id == owner_user_id and job.state == "queued"
        )

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        return sum(
            1
            for job in self.jobs.values()
            if job.owner_user_id == owner_user_id
            and job.state in {"running", "repairing"}
        )

    def count_active_global(self) -> int:
        return sum(
            1
            for job in self.jobs.values()
            if job.state in {"queued", "running", "repairing"}
        )

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> BacktestAiConfigJob | None:
        for job_id, job in tuple(self.jobs.items()):
            if (
                job.state in {"running", "repairing"}
                and job.lease_expires_at is not None
                and job.lease_expires_at <= now
                and job.attempt >= max_attempts
            ):
                self.jobs[job_id] = replace(
                    job,
                    state="failed",
                    finished_at=now,
                    updated_at=now,
                    locked_by=None,
                    locked_at=None,
                    lease_expires_at=None,
                    heartbeat_at=None,
                    last_error="lease_attempt_limit_exceeded",
                    last_error_json={"code": "lease_attempt_limit_exceeded"},
                )
        for job in sorted(self.jobs.values(), key=lambda item: (item.queued_at, item.job_id)):
            if job.state == "queued" or (
                job.state in {"running", "repairing"}
                and job.lease_expires_at is not None
                and job.lease_expires_at <= now
                and job.attempt < max_attempts
            ):
                claimed = replace(
                    job,
                    state="repairing" if job.state == "repairing" else "running",
                    started_at=job.started_at or now,
                    updated_at=now,
                    locked_by=locked_by,
                    locked_at=now,
                    lease_expires_at=now + timedelta(seconds=lease_seconds),
                    heartbeat_at=now,
                    attempt=job.attempt + 1,
                )
                self.jobs[job.job_id] = claimed
                return claimed
        return None

    def heartbeat(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestAiConfigJob | None:
        job = self.jobs.get(job_id)
        if (
            job is None
            or job.state not in {"running", "repairing"}
            or job.locked_by != locked_by
            or job.lease_expires_at is None
            or job.lease_expires_at <= now
        ):
            return None
        updated = replace(
            job,
            updated_at=now,
            heartbeat_at=now,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
        )
        self.jobs[job_id] = updated
        return updated

    def finish(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        next_state: BacktestAiConfigTerminalState,
        assistant_message: str | None = None,
        validated_config_json: dict[str, object] | None = None,
        suggestions_json: tuple[dict[str, object], ...] = (),
        validation_errors_json: tuple[dict[str, object], ...] = (),
        model_id: str | None = None,
        model_path_hash: str | None = None,
        last_error: str | None = None,
        last_error_json: dict[str, object] | None = None,
    ) -> BacktestAiConfigJob | None:
        job = self.jobs.get(job_id)
        if (
            job is None
            or job.state not in {"running", "repairing"}
            or job.locked_by != locked_by
            or job.lease_expires_at is None
            or job.lease_expires_at <= now
        ):
            return None
        updated = replace(
            job,
            state=next_state,
            assistant_message=assistant_message,
            validated_config_json=validated_config_json,
            suggestions_json=suggestions_json,
            validation_errors_json=validation_errors_json,
            model_id=model_id,
            model_path_hash=model_path_hash,
            finished_at=now,
            updated_at=now,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            heartbeat_at=None,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.jobs[job_id] = updated
        return updated
