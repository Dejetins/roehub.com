from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.routes.backtests import build_backtests_router
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestLazyTradesDetailReadModel,
    BacktestNoRiskTopResult,
)
from trading.contexts.backtest.application.ports import (
    BacktestArtifactContextUnavailable,
    BacktestJobListPage,
    BacktestJobListQuery,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
    BacktestTopResultAssemblyService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId


def test_get_backtest_runtime_defaults_returns_public_contract() -> None:
    client = _build_client()

    response = client.get(
        "/backtests/runtime-defaults",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000201"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported_timeframes"] == ["15m"]
    assert payload["risk_modes"] == ["none", "tp_sl_grid"]
    assert payload["direction_modes"] == ["long_only", "long_short_reversal"]
    assert "fixed_equity_pct_max_quote" in payload["sizing_modes"]
    assert "total_return_pct" in payload["ranking_metrics"]
    assert payload["top_n_default"] == 100
    assert payload["guardrails"]["max_top_n"] == 100


def test_post_backtest_preflight_returns_normalized_result_without_job_creation() -> None:
    resolver = _FakeArtifactResolver()
    client = _build_client(resolver=resolver)

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000202"},
        json=_valid_request(),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_request"]["timeframe"] == "15m"
    assert payload["normalized_request"]["coordinates"]["symbol"] == "BTCUSDT"
    assert len(payload["request_hash"]) == 64
    assert payload["artifact_metadata"]["artifact_slot"] == "slot_a"
    assert payload["cost_estimate"]["candidate_combinations"] == 6
    assert payload["errors"] == []
    assert resolver.coordinates == (BacktestCoordinates("binance", "spot", "BTCUSDT"),)


def test_post_backtest_preflight_invalid_indicator_returns_backtest_422() -> None:
    client = _build_client()
    request = _valid_request()
    request["indicators"][0]["indicator_id"] = "ma.nope"

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000203"},
        json=request,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "backtest.invalid_request"
    assert response.json()["error"]["details"]["errors"][0]["path"] == (
        "indicators.0.indicator_id"
    )


def test_post_backtest_job_invalid_request_rejects_before_repository_create() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    request = _valid_request()
    request["indicators"][0]["indicator_id"] = "ma.nope"

    response = client.post(
        "/backtests/jobs",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000218",
            "Idempotency-Key": "invalid-request-key",
        },
        json=request,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "backtest.invalid_request"
    assert repository.jobs == {}


def test_post_backtest_preflight_artifacts_unavailable_returns_backtest_503() -> None:
    client = _build_client(resolver=_UnavailableArtifactResolver())

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000204"},
        json=_valid_request(),
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backtest.artifacts_unavailable"
    assert response.json()["error"]["details"]["retryable"] is True


def test_post_backtest_job_creates_queued_job_without_inline_top_rows() -> None:
    repository = _FakeJobRepository()
    trigger = _FakeExecutionTrigger()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(repository=repository, execution_trigger=trigger)
    )

    response = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
        json=_valid_request(),
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["state"] == "queued"
    assert payload["progress"]["pipeline_stage"] == "queued"
    assert payload["terminal_summary"] == {}
    assert len(trigger.job_ids) == 1
    assert repository.jobs is not None
    stored = repository.jobs[UUID(payload["job_id"])]
    assert stored.execution_mode == "background_auto"
    top_response = client.get(
        f"/backtests/jobs/{payload['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
    )
    assert top_response.status_code == 200
    assert top_response.json()["items"] == []


def test_get_top_exposes_public_variant_key_for_worker_finished_job() -> None:
    repository = _FakeJobRepository()
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000205")
    job, row = _seed_succeeded_job(repository=repository, user_id=user_id)
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))

    top_response = client.get(
        f"/backtests/jobs/{job.job_id}/top",
        headers={"x-user-id": str(user_id)},
    )

    assert top_response.status_code == 200
    top_row = top_response.json()["items"][0]
    assert top_row["variant_key"] == row.payload_json["public_variant_key"]
    assert top_row["variant_key"].startswith("job_")
    assert len(top_row["variant_hash"]) == 64
    assert top_row["variant_key"] != top_row["variant_hash"]
    assert top_row["links"]["lazy_trades"].endswith("/trades")
    raw_hash_response = client.get(
        f"/backtests/jobs/{job.job_id}/variants/{top_row['variant_hash']}",
        headers={"x-user-id": str(user_id)},
    )
    assert raw_hash_response.status_code == 404
    assert raw_hash_response.json()["error"]["code"] == "backtest.not_found"


def test_post_backtest_variant_trades_uses_public_key_and_returns_detail() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService()
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000215")
    job, _row = _seed_succeeded_job(repository=repository, user_id=user_id)
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )
    top = client.get(
        f"/backtests/jobs/{job.job_id}/top",
        headers={"x-user-id": str(user_id)},
    ).json()["items"][0]

    response = client.post(
        f"/backtests/jobs/{job.job_id}/variants/{top['variant_key']}/trades",
        headers={"x-user-id": str(user_id)},
    )
    raw_hash_response = client.post(
        f"/backtests/jobs/{job.job_id}/variants/{top['variant_hash']}/trades",
        headers={"x-user-id": str(user_id)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["variant_key"] == top["variant_key"]
    assert payload["variant_hash"] == top["variant_hash"]
    assert payload["cache"]["status"] == "miss"
    assert payload["trades"][0]["exit_reason"] == "signal"
    assert raw_hash_response.status_code == 404
    assert raw_hash_response.json()["error"]["code"] == "backtest.not_found"
    assert lazy_service.requests == ((top["variant_key"], top["variant_hash"]),)


def test_get_backtest_result_summary_is_bounded_and_uses_public_variant_key() -> None:
    """
    Contract: GET /api/backtests/jobs/{job_id}/summary maps to backend
    GET /backtests/jobs/{job_id}/summary, owner-scopes the current user, returns
    job + summary-only top rows, has no pagination, has no cache identity impact,
    and is a compatible additive public API response.
    """
    repository = _FakeJobRepository()
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000218")
    job, row = _seed_succeeded_job(repository=repository, user_id=user_id)
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))

    response = client.get(
        f"/backtests/jobs/{job.job_id}/summary",
        headers={"x-user-id": str(user_id)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["job"]["job_id"] == str(job.job_id)
    assert payload["job"]["terminal_summary"]["top_variants_count"] == 1
    assert payload["selected_variant_key"] == row.payload_json["public_variant_key"]
    assert payload["variants"][0]["variant_key"] == row.payload_json["public_variant_key"]
    assert payload["variants"][0]["variant_hash"] == row.variant_key
    assert "trades" not in payload["variants"][0]


def test_get_backtest_result_unknown_variant_returns_404() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService()
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000219")
    job, _row = _seed_succeeded_job(repository=repository, user_id=user_id)
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )

    response = client.get(
        f"/backtests/jobs/{job.job_id}/variants/not-public/equity",
        headers={"x-user-id": str(user_id)},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "backtest.not_found"
    assert lazy_service.requests == ()


def test_get_backtest_result_trades_uses_server_pagination() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService(trade_count=125)
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000220")
    job, row = _seed_succeeded_job(repository=repository, user_id=user_id)
    variant_key = str(row.payload_json["public_variant_key"])
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )

    response = client.get(
        f"/backtests/jobs/{job.job_id}/variants/{variant_key}/trades",
        params={"page": 2, "page_size": 50},
        headers={"x-user-id": str(user_id)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["items"]) == 50
    assert payload["items"][0]["trade_index"] == 50
    assert payload["items"][-1]["trade_index"] == 99
    assert payload["pagination"] == {
        "page": 2,
        "page_size": 50,
        "max_page_size": 100,
        "total": 125,
        "total_pages": 3,
        "has_previous": True,
        "has_next": True,
    }
    assert payload["links"]["csv"].endswith("/trades.csv")
    assert lazy_service.requests == ((variant_key, row.variant_key),)


def test_get_backtest_result_equity_downsamples_to_requested_bounds() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService(trade_count=1600)
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000221")
    job, row = _seed_succeeded_job(repository=repository, user_id=user_id)
    variant_key = str(row.payload_json["public_variant_key"])
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )

    response = client.get(
        f"/backtests/jobs/{job.job_id}/variants/{variant_key}/equity",
        params={"points": 100},
        headers={"x-user-id": str(user_id)},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["series"] == "equity"
    assert payload["total_points"] == 1600
    assert payload["point_limit"] == 100
    assert payload["downsampled"] is True
    assert len(payload["points"]) <= 100
    assert payload["points"][0]["trade_index"] == 0
    assert payload["points"][-1]["trade_index"] == 1599


def test_get_backtest_result_csv_is_separate_and_owner_scoped() -> None:
    repository = _FakeJobRepository()
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000222")
    job, row = _seed_succeeded_job(repository=repository, user_id=user_id)
    variant_key = str(row.payload_json["public_variant_key"])
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(trade_count=3),
        )
    )

    owner_response = client.get(
        f"/backtests/jobs/{job.job_id}/variants/{variant_key}/trades.csv",
        headers={"x-user-id": str(user_id)},
    )
    foreign_response = client.get(
        f"/backtests/jobs/{job.job_id}/variants/{variant_key}/trades.csv",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000223"},
    )

    assert owner_response.status_code == 200
    assert owner_response.headers["content-type"].startswith("text/csv")
    assert "attachment;" in owner_response.headers["content-disposition"]
    assert "trade_index,side,entry_timestamp,exit_timestamp" in owner_response.text
    assert "\n0,long,2026-01-01T00:00:00Z" in owner_response.text
    assert foreign_response.status_code == 403
    assert foreign_response.json()["error"]["code"] == "backtest.forbidden"


def test_post_backtest_variant_trades_foreign_owner_returns_forbidden() -> None:
    repository = _FakeJobRepository()
    owner_id = UserId.from_string("00000000-0000-0000-0000-000000000216")
    foreign_id = UserId.from_string("00000000-0000-0000-0000-000000000217")
    job, _row = _seed_succeeded_job(repository=repository, user_id=owner_id)
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    top = client.get(
        f"/backtests/jobs/{job.job_id}/top",
        headers={"x-user-id": str(owner_id)},
    ).json()["items"][0]

    response = client.post(
        f"/backtests/jobs/{job.job_id}/variants/{top['variant_key']}/trades",
        headers={"x-user-id": str(foreign_id)},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "backtest.forbidden"


def test_post_backtest_job_idempotency_replay_and_conflict() -> None:
    repository = _FakeJobRepository()
    trigger = _FakeExecutionTrigger()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(repository=repository, execution_trigger=trigger)
    )
    headers = {
        "x-user-id": "00000000-0000-0000-0000-000000000206",
        "Idempotency-Key": "stable-key",
    }

    first = client.post("/backtests/jobs", headers=headers, json=_valid_request())
    replay = client.post("/backtests/jobs", headers=headers, json=_valid_request())
    changed_request = _valid_request()
    changed_request["top_n"] = 50
    conflict = client.post("/backtests/jobs", headers=headers, json=changed_request)

    assert first.status_code == 201
    assert first.json()["state"] == "queued"
    assert replay.status_code == 200
    assert replay.json()["job_id"] == first.json()["job_id"]
    assert replay.json()["idempotent_replay"] is True
    assert trigger.job_ids == (UUID(first.json()["job_id"]),)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "backtest.idempotency_key_conflict"


def test_get_backtest_job_foreign_owner_returns_forbidden_code() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000207"},
        json=_valid_request(),
    )

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000208"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "backtest.forbidden"


def test_post_backtest_job_cancel_terminal_job_is_idempotent() -> None:
    repository = _FakeJobRepository()
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000209")
    job, _row = _seed_succeeded_job(repository=repository, user_id=user_id)
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))

    response = client.post(
        f"/backtests/jobs/{job.job_id}/cancel",
        headers={"x-user-id": str(user_id)},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"


def test_post_backtest_job_cancel_queued_job_is_deterministic() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000210"},
        json=_valid_request(),
    )

    response = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/cancel",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000210"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"
    assert response.json()["progress"]["pipeline_stage"] == "cancelled"


def test_backtest_jobs_auth_failure_uses_auth_required_code() -> None:
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=_FakeJobRepository()))

    response = client.get("/backtests/jobs")

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "auth.required"


class _HeaderCurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )
        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


@dataclass
class _FakeArtifactResolver:
    coordinates: tuple[BacktestCoordinates, ...] = ()

    def resolve_context(self, *, coordinates: BacktestCoordinates) -> BacktestArtifactMetadata:
        self.coordinates = (*self.coordinates, coordinates)
        return BacktestArtifactMetadata(
            artifact_slot="slot_a",
            artifact_slot_generation=4,
            artifact_manifest_hash="a" * 64,
            artifact_asof_date="2026-03-25",
            hit_times_manifest_hash="b" * 64,
            published_at_utc="2026-03-25T02:00:00Z",
        )


class _UnavailableArtifactResolver:
    def resolve_context(self, *, coordinates: BacktestCoordinates) -> BacktestArtifactMetadata:
        raise BacktestArtifactContextUnavailable("current pointer missing")


def _build_client(
    *,
    resolver: _FakeArtifactResolver | _UnavailableArtifactResolver | None = None,
    jobs_use_case: BacktestJobsUseCase | None = None,
) -> TestClient:
    defaults_provider = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path="configs/prod/indicators.yaml"
    )
    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
    )
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtests_router(
            runtime_defaults_service=BacktestRuntimeDefaultsService(
                defaults_provider=defaults_provider,
                runtime_config=runtime_config,
            ),
            preflight_service=BacktestPreflightService(
                defaults_provider=defaults_provider,
                artifact_context_resolver=resolver or _FakeArtifactResolver(),
                runtime_config=runtime_config,
            ),
            current_user_dependency=_HeaderCurrentUserDependency(),  # type: ignore[arg-type]
            jobs_use_case=jobs_use_case,
        )
    )
    return TestClient(app)


def _build_jobs_use_case(
    *,
    repository: "_FakeJobRepository",
    execution_trigger: Any | None = None,
    lazy_trades_service: Any | None = None,
) -> BacktestJobsUseCase:
    defaults_provider = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path="configs/prod/indicators.yaml"
    )
    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
    )
    return BacktestJobsUseCase(
        job_repository=repository,
        preflight_service=BacktestPreflightService(
            defaults_provider=defaults_provider,
            artifact_context_resolver=_FakeArtifactResolver(),
            runtime_config=runtime_config,
        ),
        runtime_config=runtime_config,
        execution_trigger=execution_trigger,
        lazy_trades_service=lazy_trades_service,
    )


@dataclass
class _FakeExecutionTrigger:
    job_ids: tuple[UUID, ...] = ()

    def enqueue(self, *, job: BacktestJob) -> None:
        self.job_ids = (*self.job_ids, job.job_id)


def _seed_succeeded_job(
    *,
    repository: "_FakeJobRepository",
    user_id: UserId,
) -> tuple[BacktestJob, BacktestJobTopVariant]:
    job_id = UUID("00000000-0000-0000-0000-00000000f001")
    created_at = datetime.now(UTC) - timedelta(seconds=3)
    request = _valid_request()
    metadata = _artifact_metadata()
    request["artifact_metadata"] = metadata.as_mapping()
    queued = BacktestJob.create_queued(
        job_id=job_id,
        user_id=user_id,
        mode="template",
        created_at=created_at,
        request_json=request,
        request_hash="d" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="a" * 64,
        backtest_runtime_config_hash="a" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_a",
            artifact_slot_generation=metadata.artifact_slot_generation,
            artifact_manifest_hash=metadata.artifact_manifest_hash,
            artifact_asof_date=metadata.artifact_asof_date,
        ),
        execution_mode="background_auto",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="15m",
        requested_top_n=100,
        ranking_primary_metric="total_return_pct",
    )
    running = queued.claim(
        changed_at=created_at + timedelta(seconds=1),
        locked_by="test-worker",
        lease_expires_at=created_at + timedelta(seconds=901),
    )
    succeeded = running.finish(
        next_state="succeeded",
        changed_at=created_at + timedelta(seconds=2),
    )
    top_result = BacktestNoRiskTopResult(
        rank=1,
        score=12.5,
        indicator_rows={"ma.dema": 7},
        metrics={"total_return_pct": 12.5, "trade_count": 2.0},
        metadata={
            "ma.dema.source": "close",
            "ma.dema.window": 5,
            "confirm_count": 1,
            "proxy_score": 0.25,
        },
    )
    row = BacktestTopResultAssemblyService().assemble(
        job_id=job_id,
        normalized_request=request,
        top_results=(top_result,),
        updated_at=succeeded.finished_at or succeeded.updated_at,
    ).top_variants[0]
    repository.create_with_top_variants(job=succeeded, top_variants=(row,))
    return succeeded, row


def _artifact_metadata() -> BacktestArtifactMetadata:
    return BacktestArtifactMetadata(
        artifact_slot="slot_a",
        artifact_slot_generation=4,
        artifact_manifest_hash="a" * 64,
        artifact_asof_date="2026-03-25",
        hit_times_manifest_hash="b" * 64,
        published_at_utc="2026-03-25T02:00:00Z",
    )


@dataclass
class _FakeLazyTradesService:
    requests: tuple[tuple[str, str], ...] = ()
    trade_count: int = 1

    def execute(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        public_variant_key: str,
    ) -> BacktestLazyTradesDetailReadModel:
        variant_hash = str(row.payload_json["variant_hash"])
        self.requests = (*self.requests, (public_variant_key, variant_hash))
        return BacktestLazyTradesDetailReadModel(
            job_id=str(job.job_id),
            variant_key=public_variant_key,
            variant_hash=variant_hash,
            request_hash=job.request_hash,
            engine_params_hash=job.engine_params_hash,
            artifact_manifest_hash=str(
                job.request_json["artifact_metadata"]["artifact_manifest_hash"]
            ),
            summary_metrics=dict(row.summary_metrics_json),
            canonical_variant_params=dict(row.payload_json["canonical_variant_params"]),
            readable_params=dict(row.payload_json["readable_params"]),
            trades=_fake_trade_rows(count=self.trade_count),
            chart_overlay={"schema": "backtest_chart_overlay_v1", "markers": [], "segments": []},
            cache={"status": "miss"},
            timing={"lazy_trades_compute": 0.001},
        )


def _fake_trade_rows(*, count: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    start = datetime(2026, 1, 1, tzinfo=UTC)
    for index in range(count):
        entry = start + timedelta(minutes=15 * index)
        exit_time = entry + timedelta(minutes=15)
        signed_return = 1.0 if index % 2 == 0 else -0.5
        rows.append(
            {
                "trade_index": index,
                "entry_timestamp": entry.isoformat().replace("+00:00", "Z"),
                "exit_timestamp": exit_time.isoformat().replace("+00:00", "Z"),
                "entry_bar_index": index * 2 + 1,
                "exit_bar_index": index * 2 + 2,
                "side": "long" if index % 3 == 0 else "short",
                "direction": "long" if index % 3 == 0 else "short",
                "entry_price": 100.0 + index,
                "exit_price": 101.0 + index,
                "quantity": 1.0,
                "notional_quote": 100.0,
                "gross_pnl_quote": signed_return,
                "net_pnl_quote": signed_return,
                "return_pct": signed_return,
                "fee_quote": 0.0,
                "slippage_quote": 0.0,
                "exit_reason": "signal",
                "equity_after": 10_000.0 + index + signed_return,
                "safe_quote_after": 0.0,
                "timeframe": "15m",
            }
        )
    return tuple(rows)


@dataclass
class _FakeJobRepository:
    jobs: dict[UUID, BacktestJob] | None = None
    top_rows: dict[UUID, tuple[BacktestJobTopVariant, ...]] | None = None

    def __post_init__(self) -> None:
        if self.jobs is None:
            self.jobs = {}
        if self.top_rows is None:
            self.top_rows = {}

    def create(self, *, job: BacktestJob) -> BacktestJob:
        assert self.jobs is not None
        self.jobs[job.job_id] = job
        return job

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
        stage_a_shortlist: BacktestJobStageAShortlist | None = None,
    ) -> BacktestJob:
        _ = stage_a_shortlist
        stored = self.create(job=job)
        assert self.top_rows is not None
        self.top_rows[job.job_id] = top_variants
        return stored

    def find_by_idempotency_key(
        self,
        *,
        user_id: UserId,
        idempotency_key_hash: str,
        created_after: datetime,
    ) -> BacktestJob | None:
        assert self.jobs is not None
        for job in sorted(self.jobs.values(), key=lambda item: item.created_at):
            idempotency = dict(job.request_json).get("idempotency")
            if (
                job.user_id == user_id
                and job.created_at >= created_after
                and isinstance(idempotency, dict)
                and idempotency.get("key_hash") == idempotency_key_hash
            ):
                return job
        return None

    def claim_for_inline_execution(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        lease_expires_at: datetime,
    ) -> BacktestJob | None:
        assert self.jobs is not None
        job = self.jobs.get(job_id)
        if job is None or job.user_id != user_id or job.state != "queued":
            return None
        claimed = job.claim(
            changed_at=now,
            locked_by=locked_by,
            lease_expires_at=lease_expires_at,
        )
        self.jobs[job_id] = claimed
        return claimed

    def finish_with_top_variants(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        next_state: BacktestJobState,
        top_variants: tuple[BacktestJobTopVariant, ...],
        last_error: str | None = None,
        last_error_json: BacktestJobErrorPayload | None = None,
    ) -> BacktestJob | None:
        _ = locked_by
        assert self.jobs is not None
        assert self.top_rows is not None
        job = self.jobs.get(job_id)
        if job is None or job.user_id != user_id or job.state != "running":
            return None
        finished = job.finish(
            next_state=next_state,
            changed_at=now,
            last_error=last_error,
            last_error_json=last_error_json,
        )
        self.jobs[job_id] = finished
        if next_state == "succeeded":
            self.top_rows[job_id] = top_variants
        return finished

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        assert self.jobs is not None
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if user_id is not None and job.user_id != user_id:
            return None
        return job

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        assert self.jobs is not None
        items = [
            job
            for job in self.jobs.values()
            if job.user_id == query.user_id
            and (query.state is None or job.state == query.state)
        ]
        items.sort(key=lambda item: (item.created_at, str(item.job_id)), reverse=True)
        return BacktestJobListPage(items=tuple(items[: query.limit]), next_cursor=None)

    def list_top_variants(self, *, job_id: UUID) -> tuple[BacktestJobTopVariant, ...]:
        assert self.top_rows is not None
        return self.top_rows.get(job_id, ())

    def get_top_variant_by_public_key(
        self,
        *,
        job_id: UUID,
        public_variant_key: str,
    ) -> BacktestJobTopVariant | None:
        for row in self.list_top_variants(job_id=job_id):
            if row.payload_json.get("public_variant_key") == public_variant_key:
                return row
        return None

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        assert self.jobs is not None
        job = self.jobs.get(job_id)
        if job is None or job.user_id != user_id:
            return None
        cancelled = job.request_cancel(changed_at=cancel_requested_at)
        self.jobs[job_id] = cancelled
        return cancelled

    def count_active_for_user(self, *, user_id: UserId) -> int:
        assert self.jobs is not None
        return sum(1 for job in self.jobs.values() if job.user_id == user_id and job.is_active())

    def count_active_global(self) -> int:
        assert self.jobs is not None
        return sum(1 for job in self.jobs.values() if job.is_active())

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        _ = market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


def _valid_request() -> dict[str, Any]:
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "2020-01-11T20:08:00Z",
            "end": "2026-04-11T20:08:00Z",
        },
        "indicators": [
            {
                "indicator_id": "ma.dema",
                "sources": ["close"],
                "window": {"start": 5, "stop": 10, "step": 1},
            }
        ],
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": 0.00075,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {
            "primary_metric": "total_return_pct",
            "direction": "desc",
        },
        "top_n": 100,
    }
