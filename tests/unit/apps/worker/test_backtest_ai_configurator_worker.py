from __future__ import annotations

import threading
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from time import sleep
from types import SimpleNamespace
from typing import Any, Mapping, cast
from uuid import UUID

from prometheus_client import generate_latest

from apps.worker.backtest_ai_configurator.wiring.observability import (
    BacktestAiConfiguratorHealthState,
    BacktestAiConfiguratorMetrics,
    start_backtest_ai_configurator_http_server,
)
from trading.contexts.backtest.application.ai_configurator import (
    BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH,
    BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION,
    BacktestAiConfigEvent,
    BacktestAiConfigGenerationLimiter,
    BacktestAiConfigJob,
    BacktestAiConfigLlmAttempt,
    BacktestAiConfigPipelineResult,
    BacktestAiConfigTerminalState,
    BacktestAiConfigWorkerUseCase,
    BacktestAiQuotaEvent,
    BacktestAiTrainingExportRecord,
    BacktestAiTrainingExportUseCase,
)
from trading.shared_kernel.primitives import UserId


def test_backtest_ai_configurator_worker_processes_queued_job_to_ready() -> None:
    repository = _Repository()
    job = _job()
    repository.jobs[job.job_id] = job
    worker = BacktestAiConfigWorkerUseCase(
        job_repository=repository,
        lease_repository=repository,
        pipeline=cast(Any, _Pipeline()),
        lease_seconds=30,
        max_attempts=2,
        locked_by="worker-a",
        generation_limiter=BacktestAiConfigGenerationLimiter(active_generations=1),
    )

    result = worker.run_next()

    assert result.claimed is True
    assert result.lease_lost is False
    assert result.job is not None
    assert result.job.state == "ready"
    assert result.job.validated_config_json == {"top_n": 100}
    assert result.job.model_id == "gemma-4-e2b-it-4bit"
    assert result.job.model_path_hash == "model-path-hash"
    assert [event.event_name for event in repository.events] == [
        "preparing_catalog",
        "collecting_context",
        "generating",
        "validating_json",
        "validating_business",
        "ready",
    ]
    assert len(result.llm_attempts) == 0


def test_backtest_ai_configurator_worker_heartbeats_during_generation() -> None:
    repository = _Repository()
    job = _job()
    repository.jobs[job.job_id] = job
    heartbeat_seen = threading.Event()
    repository.heartbeat_event = heartbeat_seen
    worker = BacktestAiConfigWorkerUseCase(
        job_repository=repository,
        lease_repository=repository,
        pipeline=cast(Any, _BlockingPipeline(heartbeat_seen=heartbeat_seen)),
        lease_seconds=30,
        max_attempts=2,
        heartbeat_interval_seconds=0.001,
        locked_by="worker-a",
        generation_limiter=BacktestAiConfigGenerationLimiter(active_generations=1),
    )

    result = worker.run_next()

    assert result.claimed is True
    assert repository.heartbeat_calls >= 1
    assert result.job is not None
    assert result.job.state == "ready"


def test_generation_limiter_serializes_active_generations_default_one() -> None:
    active_transitions: list[bool] = []
    limiter = BacktestAiConfigGenerationLimiter(
        active_generations=1,
        active_callback=active_transitions.append,
    )
    lock = threading.Lock()
    active_ref = {"value": 0}
    max_ref = {"value": 0}

    def work() -> None:
        limiter.run_locked(lambda: _tracked_work(lock, active_ref, max_ref))

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_ref["value"] == 1
    assert active_transitions.count(True) == 2
    assert active_transitions.count(False) == 2


def test_worker_skips_non_backtests_source_page_without_running_pipeline() -> None:
    repository = _Repository()
    job = _job(source_page="monitoring")
    repository.jobs[job.job_id] = job
    pipeline = _Pipeline()
    worker = BacktestAiConfigWorkerUseCase(
        job_repository=repository,
        lease_repository=repository,
        pipeline=cast(Any, pipeline),
        lease_seconds=30,
        max_attempts=2,
        locked_by="worker-a",
    )

    result = worker.run_next()

    assert result.claimed is True
    assert result.skipped_source_page is True
    assert result.job is not None
    assert result.job.state == "failed"
    assert result.job.last_error == "unsupported_source_page"
    assert pipeline.calls == 0


def test_worker_metrics_expose_required_names_after_fake_job() -> None:
    job = replace(
        _job(),
        state="ready",
        started_at=datetime.now(UTC) - timedelta(seconds=2),
        finished_at=datetime.now(UTC),
        model_id="gemma-4-e2b-it-4bit",
        validation_errors_json=(
            {"path": "config.symbol", "code": "unsupported_symbol", "message": "bad"},
        ),
    )
    metrics = BacktestAiConfiguratorMetrics()

    metrics.observe_result(
        result=cast(
            Any,
            SimpleNamespace(claimed=True, lease_lost=False, job=job),
        ),
        duration_seconds=0.5,
    )
    metrics.set_queue_depth(queued=3, running=1, repairing=0)
    metrics.set_model_metadata(
        model_id="gemma-4-e2b-it-4bit",
        loaded=True,
        runtime="lm_studio",
        quantization="4bit",
    )
    payload = generate_latest(metrics.registry).decode("utf-8")

    assert "backtest_ai_config_jobs_total" in payload
    assert "backtest_ai_config_jobs_inflight" in payload
    assert "backtest_ai_config_queue_depth" in payload
    assert "backtest_ai_config_active_generations" in payload
    assert "backtest_ai_config_stage_duration_seconds_bucket" in payload
    assert "backtest_ai_config_llm_latency_seconds_bucket" in payload
    assert "backtest_ai_config_total_latency_seconds_bucket" in payload
    assert "backtest_ai_config_validation_failures_total" in payload
    assert "backtest_ai_config_repair_attempts_total" in payload
    assert "backtest_ai_config_security_decisions_total" in payload
    assert "backtest_ai_config_quota_rejections_total" in payload
    assert "backtest_ai_config_capacity_rejections_total" in payload
    assert "backtest_ai_config_applied_total" in payload
    assert "backtest_ai_config_model_reload_total" in payload
    assert "process_resident_memory_bytes" in payload


def test_training_export_redacts_private_fields_and_labels_rows() -> None:
    repository = _Repository()
    job = replace(
        _job(),
        state="ready",
        user_prompt_text=(
            "Собери BTC config password=secret with "
            "http://127.0.0.1:8080 and /opt/roehub/app/debug"
        ),
        assistant_message="Ready from /Users/daniildegtyarev/private",
        validated_config_json={
            "symbol": "BTCUSDT",
            "base_url": "http://127.0.0.1:8080",
            "token": "sk-123456789012345678901234",
        },
        applied_at=datetime.now(UTC),
        model_id="gemma-4-e2b-it-4bit",
        model_path_hash="hash-only",
    )
    repository.training_records = (
        BacktestAiTrainingExportRecord(job=job, attempts=()),
    )

    rows = BacktestAiTrainingExportUseCase(source=repository).export_rows()

    assert rows[0]["quality_label"] == "applied"
    serialized = str(rows[0])
    assert "password=secret" not in serialized
    assert "127.0.0.1" not in serialized
    assert "/opt/roehub" not in serialized
    assert "/Users/daniildegtyarev" not in serialized
    assert "base_url" not in serialized
    assert "token" not in serialized
    assert rows[0]["model_path_hash"] == "hash-only"


def test_ops_http_server_exposes_health_ready_and_metrics() -> None:
    metrics = BacktestAiConfiguratorMetrics()
    health = BacktestAiConfiguratorHealthState(
        config_loaded=True,
        model_id="gemma-4-e2b-it-4bit",
        model_path="/tmp/model",
        drain_mode=False,
        readiness_checks=(lambda: (True, "model_path"),),
    )
    health.mark_loop_started()
    server = start_backtest_ai_configurator_http_server(
        host="127.0.0.1",
        port=0,
        metrics=metrics,
        health=health,
    )
    port = cast(tuple[str, int], server.server_address)[1]
    try:
        live = urllib.request.urlopen(  # noqa: S310
            f"http://127.0.0.1:{port}/health/live",
            timeout=2,
        ).read()
        ready = urllib.request.urlopen(  # noqa: S310
            f"http://127.0.0.1:{port}/health/ready",
            timeout=2,
        ).read()
        exported_metrics = urllib.request.urlopen(  # noqa: S310
            f"http://127.0.0.1:{port}/metrics",
            timeout=2,
        ).read()
    finally:
        server.shutdown()

    assert b'"status": "live"' in live
    assert b'"status": "ready"' in ready
    assert b"backtest_ai_config_jobs_total" in exported_metrics


def _tracked_work(
    lock: threading.Lock,
    active_ref: dict[str, int],
    max_ref: dict[str, int],
) -> None:
    with lock:
        active_ref["value"] += 1
        max_ref["value"] = max(max_ref["value"], active_ref["value"])
    sleep(0.01)
    with lock:
        active_ref["value"] -= 1


class _Pipeline:
    calls: int

    def __init__(self) -> None:
        self.calls = 0

    def run(self, *, job: BacktestAiConfigJob) -> BacktestAiConfigPipelineResult:
        self.calls += 1
        return BacktestAiConfigPipelineResult(
            status="ready",
            assistant_message="Configuration is ready.",
            catalog_snapshot_hash="a" * 64,
            stage="validation",
            validated_config={"top_n": 100},
            model_id="gemma-4-e2b-it-4bit",
            model_path_hash="model-path-hash",
        )


@dataclass
class _BlockingPipeline:
    heartbeat_seen: threading.Event

    def run(self, *, job: BacktestAiConfigJob) -> BacktestAiConfigPipelineResult:
        assert self.heartbeat_seen.wait(timeout=1.0)
        return BacktestAiConfigPipelineResult(
            status="ready",
            assistant_message="Configuration is ready.",
            catalog_snapshot_hash="a" * 64,
            stage="validation",
            validated_config={"top_n": 100},
            model_id="gemma-4-e2b-it-4bit",
            model_path_hash="model-path-hash",
        )


@dataclass
class _Repository:
    jobs: dict[UUID, BacktestAiConfigJob] = field(default_factory=dict)
    events: list[BacktestAiConfigEvent] = field(default_factory=list)
    llm_attempts: list[BacktestAiConfigLlmAttempt] = field(default_factory=list)
    heartbeat_calls: int = 0
    heartbeat_event: threading.Event | None = None
    training_records: tuple[BacktestAiTrainingExportRecord, ...] = ()

    def append_event(self, *, event: BacktestAiConfigEvent) -> None:
        self.events.append(event)

    def record_llm_attempt(self, *, attempt: BacktestAiConfigLlmAttempt) -> None:
        self.llm_attempts.append(attempt)

    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> BacktestAiConfigJob | None:
        del max_attempts
        for job in sorted(self.jobs.values(), key=lambda item: (item.queued_at, item.job_id)):
            if job.state != "queued":
                continue
            claimed = replace(
                job,
                state="running",
                started_at=now,
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
        self.heartbeat_calls += 1
        if self.heartbeat_event is not None:
            self.heartbeat_event.set()
        job = self.jobs[job_id]
        if job.locked_by != locked_by:
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
        job = self.jobs[job_id]
        if job.locked_by != locked_by:
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

    def create_with_quota_event(
        self,
        *,
        job: BacktestAiConfigJob,
        event: BacktestAiConfigEvent,
        quota_event: BacktestAiQuotaEvent,
    ) -> BacktestAiConfigJob:
        del event, quota_event
        self.jobs[job.job_id] = job
        return job

    def get(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId | None = None,
    ) -> BacktestAiConfigJob | None:
        del owner_user_id
        return self.jobs.get(job_id)

    def find_by_idempotency_key(
        self,
        *,
        owner_user_id: UserId,
        idempotency_key: str,
    ) -> BacktestAiConfigJob | None:
        del owner_user_id, idempotency_key
        return None

    def record_quota_event(self, *, event: BacktestAiQuotaEvent) -> None:
        del event

    def list_events(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId,
    ) -> tuple[BacktestAiConfigEvent, ...]:
        del owner_user_id
        return tuple(event for event in self.events if event.job_id == job_id)

    def record_feedback(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId,
        applied: bool,
        feedback_json: Mapping[str, object],
        now: datetime,
    ) -> BacktestAiConfigJob | None:
        del owner_user_id, applied, feedback_json, now
        return self.jobs.get(job_id)

    def count_quota_events(
        self,
        *,
        owner_user_id: UserId,
        occurred_after: datetime,
    ) -> int:
        del owner_user_id, occurred_after
        return 0

    def count_queued_for_user(self, *, owner_user_id: UserId) -> int:
        del owner_user_id
        return 0

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        del owner_user_id
        return 0

    def count_active_global(self) -> int:
        return 0

    def count_jobs_by_state(self, *, state: str) -> int:
        return sum(1 for job in self.jobs.values() if job.state == state)

    def list_training_export_records(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[BacktestAiTrainingExportRecord, ...]:
        if limit is None:
            return self.training_records
        return self.training_records[:limit]


def _job(*, source_page: str = "backtests") -> BacktestAiConfigJob:
    now = datetime(2026, 5, 11, tzinfo=UTC)
    return BacktestAiConfigJob(
        job_id=UUID("00000000-0000-0000-0000-000000000701"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000702"),
        mode="assistant_v1",
        locale="ru",
        state="queued",
        source_page=source_page,
        user_prompt_text="Собери конфиг",
        user_prompt_hash="a" * 64,
        system_prompt_version=BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION,
        system_prompt_hash=BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH,
        catalog_snapshot_hash="b" * 64,
        runtime_defaults_hash="c" * 64,
        queued_at=now,
        updated_at=now,
    )
