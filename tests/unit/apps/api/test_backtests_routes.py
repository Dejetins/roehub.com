import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

import apps.api.routes.backtests as backtests_routes
from apps.api.common import register_api_error_handlers
from apps.api.routes import build_backtests_router
from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestLazyTradesDetailReadModel,
    BacktestNoRiskTopResult,
    build_backtest_funding_read_model,
)
from trading.contexts.backtest.application.ports import (
    BacktestArtifactContextUnavailable,
    BacktestJobListPage,
    BacktestJobListQuery,
    BacktestLazyTradesMaterializationRequest,
    BacktestLazyTradesMaterializationTask,
    ResearchOrganizationScope,
)
from trading.contexts.backtest.application.services.v2 import (
    BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET,
    SUPPORTED_BACKTEST_TIMEFRAMES_V1,
    BacktestAdmissionConfig,
    BacktestAdmissionService,
    BacktestJobExecutionResult,
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
    BacktestTopResultAssemblyService,
)
from trading.contexts.backtest.application.use_cases import BacktestJobsUseCase
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobStageAShortlist,
    BacktestJobState,
    BacktestJobTopVariant,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.strategy.adapters.outbound.persistence.in_memory import (
    InMemoryStrategyCompatibilityReadinessRepository,
)
from trading.contexts.strategy.application import (
    BacktestVariantLaunchSnapshot,
    CreateStrategyFromBacktestVariantResult,
    CurrentUser,
    StrategyCompatibilityReadinessService,
    StrategyVariantScenarioMatrixService,
)
from trading.contexts.strategy.application.ports.market_data_readiness import (
    MarketDataReadinessSnapshot,
)
from trading.contexts.strategy.domain.entities import (
    Strategy,
    StrategyBacktestVariantProvenance,
    StrategySpecV1,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId, PaidLevel, UserId

_ORGANIZATION_ID = OrganizationId.from_string("00000000-0000-0000-0000-000000000001")


class _ScopeResolver:
    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        return ResearchOrganizationScope(
            organization_id=_ORGANIZATION_ID,
            user_id=user_id,
        )


@dataclass
class _MappingScopeResolver:
    organizations: dict[UserId, OrganizationId]

    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        return ResearchOrganizationScope(
            organization_id=self.organizations[user_id],
            user_id=user_id,
        )


def test_get_backtest_runtime_defaults_returns_public_contract() -> None:
    client = _build_client()

    response = client.get(
        "/backtests/runtime-defaults",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000201"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["supported_timeframes"] == list(SUPPORTED_BACKTEST_TIMEFRAMES_V1)
    assert payload["risk_modes"] == ["none", "tp_sl_grid"]
    assert payload["direction_modes"] == ["long_only", "short", "long_short_reversal"]
    assert "fixed_equity_pct_max_quote" in payload["sizing_modes"]
    assert "total_return_pct" in payload["ranking_metrics"]
    assert payload["top_n_default"] == 10
    assert payload["guardrails"]["max_top_n"] == 50
    assert payload["execution_defaults"]["funding"] == {
        "mode": "include_when_futures",
        "coverage_policy": "degraded_with_warning",
    }
    assert payload["direction_market_compatibility"]["markets"]["spot"][
        "rejected_direction_modes"
    ] == {
        "short": BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET,
        "long_short_reversal": BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET,
    }
    assert payload["indicator_param_specs"]["ma.dema"]["params"]["window"] == {
        "mode": "range",
        "start": 5,
        "stop_incl": 200,
        "step": 1,
    }


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
    assert payload["normalized_request"]["execution"]["funding"] == {
        "mode": "off",
        "coverage_policy": "degraded_with_warning",
    }
    assert payload["funding_readiness"]["status"] == "not_applicable"
    assert payload["direction_market_compatibility"]["compatible"] is True
    assert payload["errors"] == []
    assert resolver.coordinates == (BacktestCoordinates("binance", "spot", "BTCUSDT"),)


def test_post_backtest_preflight_returns_futures_funding_readiness_fixtures() -> None:
    client = _build_client(
        resolver=_FakeArtifactResolver(
            funding_coverage_status="degraded",
            funding_coverage_policy="degraded_with_warning",
            funding_manifest_hash="c" * 64,
            funding_rows_count=3,
            funding_expected_event_count=4,
            funding_missing_event_count=1,
            funding_reason_codes=("missing_trailing_coverage",),
        )
    )
    request = _valid_request()
    request["coordinates"]["market_type"] = "futures"
    request["execution"]["direction_mode"] = "short"

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000202"},
        json=request,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_request"]["execution"]["funding"] == {
        "mode": "include_when_futures",
        "coverage_policy": "degraded_with_warning",
    }
    assert payload["funding_readiness"]["status"] == "degraded"
    assert payload["funding_readiness"]["warning_codes"] == ["missing_trailing_coverage"]
    assert payload["funding_readiness"]["coverage_ratio"] == 0.75
    assert payload["funding_readiness"]["funding_manifest_hash"] == "c" * 64
    assert payload["direction_market_compatibility"]["required_market_type"] == "futures"


@pytest.mark.parametrize(
    ("status", "rows_count", "expected_event_count", "missing_event_count", "warning_codes"),
    [
        ("ready", 4, 4, 0, []),
        ("degraded", 3, 4, 1, ["missing_trailing_coverage"]),
        ("unavailable", 0, 4, 4, ["no_funding_rows"]),
    ],
)
def test_post_backtest_preflight_route_smoke_for_futures_funding_statuses(
    status: str,
    rows_count: int,
    expected_event_count: int,
    missing_event_count: int,
    warning_codes: list[str],
) -> None:
    client = _build_client(
        resolver=_FakeArtifactResolver(
            funding_coverage_status=status,
            funding_coverage_policy=("degraded_with_warning" if status == "degraded" else status),
            funding_manifest_hash="c" * 64,
            funding_rows_count=rows_count,
            funding_expected_event_count=expected_event_count,
            funding_missing_event_count=missing_event_count,
            funding_reason_codes=tuple(warning_codes),
        )
    )
    request = _valid_request()
    request["coordinates"]["market_type"] = "futures"

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000202"},
        json=request,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["funding_readiness"]["status"] == status
    assert payload["funding_readiness"]["coverage_policy"] == "degraded_with_warning"
    assert payload["funding_readiness"]["warning_codes"] == warning_codes


def test_post_backtest_preflight_rejects_spot_short_like_direction() -> None:
    client = _build_client()
    request = _valid_request()
    request["execution"]["direction_mode"] = "long_short_reversal"

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000203"},
        json=request,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "backtest.invalid_request"
    assert response.json()["error"]["details"]["errors"][0]["code"] == (
        BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET
    )


@pytest.mark.parametrize("timeframe", SUPPORTED_BACKTEST_TIMEFRAMES_V1)
def test_post_backtest_preflight_accepts_supported_artifact_timeframes(timeframe: str) -> None:
    resolver = _FakeArtifactResolver()
    client = _build_client(resolver=resolver)
    request = _valid_request()
    request["timeframe"] = timeframe

    response = client.post(
        "/backtests/preflight",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000212"},
        json=request,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["normalized_request"]["timeframe"] == timeframe
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
    assert response.json()["error"]["details"]["errors"][0]["path"] == ("indicators.0.indicator_id")


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


def test_post_backtest_preflight_applies_tier_request_limits() -> None:
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=_FakeJobRepository()))

    response = client.post(
        "/backtests/preflight",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000239",
            "x-paid-level": "free",
        },
        json=_valid_request(),
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"]["code"] == "backtest.request_too_expensive"
    assert payload["error"]["details"]["paid_level"] == "free"
    assert payload["error"]["details"]["limit_scope"] == "full_jobs.top_n"


def test_post_backtest_job_creates_job_and_exposes_public_top_variant_key() -> None:
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
    assert payload["refresh_status"] == "poll"
    assert payload["retry_after_seconds"] == 2
    assert payload["terminal_summary"] == {}
    assert payload["idempotent_replay"] is False
    assert repository.jobs is not None
    stored = repository.jobs[UUID(payload["job_id"])]
    assert stored.execution_mode == "background_auto"
    assert stored.request_json["scheduling"]["scheduling_class"] == "heavy"
    assert stored.request_json["scheduling"]["estimated_combinations_upper_bound"] == 6
    assert trigger.calls == ((stored.job_id, stored.request_hash),)

    _complete_job(repository=repository, job_id=stored.job_id)
    top_response = client.get(
        f"/backtests/jobs/{payload['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
    )
    assert top_response.status_code == 200
    top_row = top_response.json()["items"][0]
    assert top_row["variant_key"].startswith("job_")
    assert len(top_row["variant_hash"]) == 64
    assert top_row["variant_key"] != top_row["variant_hash"]
    assert "trades" not in top_row
    assert top_row["links"]["lazy_trades"].endswith("/trades")
    raw_hash_response = client.get(
        f"/backtests/jobs/{payload['job_id']}/variants/{top_row['variant_hash']}",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
    )
    assert raw_hash_response.status_code == 404
    assert raw_hash_response.json()["error"]["code"] == "backtest.not_found"


def test_post_backtest_job_rejects_spot_short_like_direction() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    request = _valid_request()
    request["execution"]["direction_mode"] = "short"

    response = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
        json=request,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "backtest.invalid_request"
    assert response.json()["error"]["details"]["errors"][0]["code"] == (
        BACKTEST_SHORT_DIRECTION_REQUIRES_FUTURES_MARKET
    )
    assert repository.jobs == {}


def test_post_backtest_variant_strategy_requires_idempotency_and_returns_provenance() -> None:
    use_case = _FakeCreateStrategyFromVariantUseCase()
    client = _build_client(create_strategy_from_variant_use_case=use_case)

    missing = client.post(
        "/backtests/jobs/00000000-0000-0000-0000-00000000b001/variants/job_demo/strategies",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
    )
    created = client.post(
        "/backtests/jobs/00000000-0000-0000-0000-00000000b001/variants/job_demo/strategies",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000205",
            "Idempotency-Key": "launch-1",
        },
    )

    assert missing.status_code == 422
    assert missing.json()["error"]["code"] == "strategy_variant_launch.idempotency_key_required"
    assert created.status_code == 201
    payload = created.json()
    assert payload["status"] == "created"
    assert payload["duplicate"] is False
    assert payload["strategy"]["spec"]["instrument_key"] == "binance:spot:BTCUSDT"
    assert payload["provenance"]["source_job_id"] == "00000000-0000-0000-0000-00000000b001"
    assert payload["provenance"]["source_variant_key"] == "job_demo"
    assert use_case.calls == (("job_demo", "launch-1"),)


def test_post_backtest_variant_strategy_sanitizes_unexpected_metric_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, str]] = []

    def record_strategy_variant_launch(*, result: str, reason: str = "none") -> None:
        captured.append((result, reason))

    monkeypatch.setattr(
        backtests_routes,
        "record_strategy_variant_launch",
        record_strategy_variant_launch,
    )
    use_case = _RaisingCreateStrategyFromVariantUseCase(
        RoehubError(
            code="unexpected_error",
            message="Unexpected error",
            details={"reason": "PostgresBacktestJobRepository cannot map top-variant row"},
        )
    )
    client = _build_client(create_strategy_from_variant_use_case=use_case)

    response = client.post(
        "/backtests/jobs/00000000-0000-0000-0000-00000000b001/variants/job_demo/strategies",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000205",
            "Idempotency-Key": "launch-1",
        },
    )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "unexpected_error"
    assert captured == [("rejected", "unexpected_error")]


def test_get_backtest_variant_scenario_matrix_uses_public_top_variant_key() -> None:
    repository = _FakeJobRepository()
    jobs_use_case = _build_jobs_use_case(repository=repository)
    request = _valid_request()
    clock = _SequenceClock(
        values=(
            datetime(2026, 6, 17, 12, 0, tzinfo=UTC),
            datetime(2026, 6, 17, 12, 1, tzinfo=UTC),
            datetime(2026, 6, 17, 12, 2, tzinfo=UTC),
        )
    )
    compatibility_service = StrategyCompatibilityReadinessService(
        strategy_repository=None,
        compatibility_repository=InMemoryStrategyCompatibilityReadinessRepository(),
        market_data_reader=_StaticMarketDataReader(state="ready"),
        clock=clock,
    )
    matrix_service = StrategyVariantScenarioMatrixService(
        compatibility_readiness_service=compatibility_service,
        clock=clock,
    )
    client = _build_client(
        jobs_use_case=jobs_use_case,
        compatibility_readiness_service=compatibility_service,
        scenario_matrix_service=matrix_service,
        backtest_variant_launch_reader=_RepoVariantLaunchReader(repository=repository),
    )
    request["coordinates"]["market_type"] = "futures"
    request["execution"]["direction_mode"] = "long_short_reversal"
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
        json=request,
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
    ).json()["items"][0]

    response = client.get(
        (
            f"/backtests/jobs/{created.json()['job_id']}/variants/"
            f"{top['variant_key']}/scenario-matrix"
        ),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000205"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_variant_key"] == top["variant_key"]
    assert payload["backtest_risk_mode"] == "none"
    assert payload["backtest_direction_mode"] == "long_short_reversal"
    assert len(payload["rows"]) == 8
    paper_long = _find_api_matrix_row(
        payload=payload,
        mode="paper",
        market_type="futures",
        entry_sizing="fixed_quote",
        direction="long",
    )
    assert paper_long["scenario_state"] == "blocked"
    assert paper_long["launch_blocked"] is True
    assert paper_long["launch_blocked_reason"] == "unsupported_live_evaluator"
    testnet_futures_short = _find_api_matrix_row(
        payload=payload,
        mode="testnet",
        market_type="futures",
        entry_sizing="fixed_quote",
        direction="short",
    )
    assert testnet_futures_short["scenario_state"] == "blocked"
    assert testnet_futures_short["launch_blocked_reason"] == "unsupported_live_evaluator"
    assert testnet_futures_short["order_capability"] == "real_order_capable"


def test_post_backtest_job_rejects_ultra_top_n_above_50() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    request = _valid_request()
    request["top_n"] = 100

    response = client.post(
        "/backtests/jobs",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000243",
            "x-paid-level": "ultra",
        },
        json=request,
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "backtest.request_too_expensive"


def test_post_backtest_job_free_active_quota_returns_429_with_retry_details() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    headers = {
        "x-user-id": "00000000-0000-0000-0000-000000000240",
        "x-paid-level": "free",
    }
    request = _valid_request()
    request["top_n"] = 20
    request["time_range"] = {
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-02-01T00:00:00Z",
    }

    first = client.post(
        "/backtests/jobs",
        headers={**headers, "Idempotency-Key": "a"},
        json=request,
    )
    second = client.post(
        "/backtests/jobs",
        headers={**headers, "Idempotency-Key": "b"},
        json=request,
    )
    limited = client.post(
        "/backtests/jobs",
        headers={**headers, "Idempotency-Key": "c"},
        json=request,
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert limited.status_code == 429
    payload = limited.json()
    assert payload["error"]["code"] == "backtest.rate_limited"
    assert payload["error"]["details"]["paid_level"] == "free"
    assert payload["error"]["details"]["limit_scope"] == "full_jobs.active"
    assert payload["error"]["details"]["retry_after_seconds"] == 60


def test_post_backtest_job_global_queue_saturated_returns_503() -> None:
    admission = BacktestAdmissionService(
        config=BacktestAdmissionConfig(max_active_full_jobs_global=1)
    )
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            admission_service=admission,
        )
    )
    request = _valid_request()

    first = client.post(
        "/backtests/jobs",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000241",
            "x-paid-level": "pro",
            "Idempotency-Key": "global-a",
        },
        json=request,
    )
    saturated = client.post(
        "/backtests/jobs",
        headers={
            "x-user-id": "00000000-0000-0000-0000-000000000242",
            "x-paid-level": "pro",
            "Idempotency-Key": "global-b",
        },
        json=request,
    )

    assert first.status_code == 201
    assert saturated.status_code == 503
    payload = saturated.json()
    assert payload["error"]["code"] == "backtest.queue_saturated"
    assert payload["error"]["details"]["limit_scope"] == "global.full_jobs.active"
    assert payload["error"]["details"]["retry_after_seconds"] == 60


def test_post_backtest_variant_trades_uses_public_key_and_returns_detail() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000215"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000215"},
    ).json()["items"][0]

    response = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000215"},
    )
    raw_hash_response = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_hash']}/trades",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000215"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["variant_key"] == top["variant_key"]
    assert payload["variant_hash"] == top["variant_hash"]
    assert payload["cache"]["status"] == "hit"
    assert payload["trades"][0]["exit_reason"] == "signal"
    assert raw_hash_response.status_code == 404
    assert raw_hash_response.json()["error"]["code"] == "backtest.not_found"
    assert lazy_service.requests == ((top["variant_key"], top["variant_hash"]),)


def test_backtest_results_routes_expose_funding_fields_without_top_trades() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000253"},
        json=_valid_request(),
    )
    _complete_job(
        repository=repository,
        job_id=UUID(created.json()["job_id"]),
        include_funding=True,
    )

    top_response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000253"},
    )
    top = top_response.json()["items"][0]
    variant_response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000253"},
    )
    detail_response = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000253"},
    )

    assert top_response.status_code == 200
    assert variant_response.status_code == 200
    assert detail_response.status_code == 200
    assert "trades" not in top
    assert top["summary_metrics"]["total_return_pct_net_of_funding"] == pytest.approx(13.25)
    assert top["funding_manifest_hash"] == "f" * 64
    assert top["funding"]["metrics"]["funding_pnl_quote"] == pytest.approx(75.0)
    assert top["funding"]["warning_codes"] == ["missing_leading_coverage"]
    variant = variant_response.json()
    assert variant["funding"]["effective_ranking_metric"] == ("total_return_pct_net_of_funding")
    detail = detail_response.json()
    assert detail["funding_manifest_hash"] == "f" * 64
    assert detail["funding"]["metrics"]["total_return_pct_net_of_funding"] == pytest.approx(13.25)
    assert detail["chart_overlay"]["funding_events"][0]["kind"] == "funding_event"


def test_post_backtest_variant_trades_cache_miss_returns_202_materialization_status() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService(cache_hit=False)
    materializations = _FakeMaterializationRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
            materialization_repository=materializations,
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000227"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000227"},
    ).json()["items"][0]

    response = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000227"},
    )
    replay = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000227"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["variant_key"] == top["variant_key"]
    assert payload["variant_hash"] == top["variant_hash"]
    assert payload["request_hash"] == created.json()["request_hash"]
    assert payload["cache"]["status"] == "miss"
    assert payload["cache"]["cache_key"] == "f" * 64
    assert payload["materialization"]["status"] == "queued"
    assert payload["materialization"]["correlation_id"] == payload["materialization"]["task_id"]
    assert (
        payload["materialization"]["request_identity"]["request_hash"]
        == (created.json()["request_hash"])
    )
    assert payload["materialization"]["retry_after_seconds"] == 2
    assert payload["pagination"] == {"mode": "none"}
    assert replay.status_code == 202
    assert replay.json()["materialization"]["task_id"] == payload["materialization"]["task_id"]
    assert len(materializations.tasks) == 1
    assert lazy_service.execute_calls == 0


def test_get_backtest_variant_result_endpoints_return_status_without_cache_miss_compute() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService(cache_hit=False)
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
            materialization_repository=_FakeMaterializationRepository(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000228"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000228"},
    ).json()["items"][0]

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/equity",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000228"},
    )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["cache"]["status"] == "miss"
    assert lazy_service.execute_calls == 0


def test_post_backtest_variant_trades_rejects_oversized_variant_key() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(cache_hit=False),
            materialization_repository=_FakeMaterializationRepository(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000229"},
        json=_valid_request(),
    )

    response = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{'x' * 257}/trades",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000229"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "backtest.invalid_request"


def test_get_backtest_result_summary_is_bounded_without_trades() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000219"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]), top_count=6)

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/summary",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000219"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_variant_key"].startswith("job_")
    assert payload["top_variants"]["items"]
    assert "trades" not in payload["top_variants"]["items"][0]
    assert lazy_service.requests == ()


def test_get_backtest_result_summary_accepts_top_limit() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000219"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]), top_count=12)

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/summary?top_limit=10",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000219"},
    )

    assert response.status_code == 200
    assert len(response.json()["top_variants"]["items"]) == 10


def test_get_backtest_result_series_downsamples_and_rejects_raw_hash() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000220"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000220"},
    ).json()["items"][0]

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/equity?points=2",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000220"},
    )
    raw_hash_response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_hash']}/equity",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000220"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    bounded = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/equity?points=10",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000220"},
    )
    assert bounded.status_code == 200
    payload = bounded.json()
    assert payload["kind"] == "equity"
    assert payload["requested_points"] == 10
    assert payload["max_points"] == 10
    assert payload["returned_points"] <= 10
    assert payload["source_points"] == 12
    assert payload["downsampled"] is True
    assert payload["points"][0]["value"] == 10001.0
    assert raw_hash_response.status_code == 404
    assert raw_hash_response.json()["error"]["code"] == "backtest.not_found"


def test_get_backtest_result_series_accepts_max_points_alias_and_rejects_conflict() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(reverse_trades=True),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000230"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000230"},
    ).json()["items"][0]

    drawdown = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/drawdown?max_points=10",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000230"},
    )
    conflict = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/drawdown?points=10&max_points=11",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000230"},
    )

    assert drawdown.status_code == 200
    payload = drawdown.json()
    assert payload["kind"] == "drawdown"
    assert payload["requested_points"] == 10
    assert payload["max_points"] == 10
    assert payload["source_points"] == 12
    assert payload["points"][0]["trade_index"] == 0
    assert conflict.status_code == 422
    assert conflict.json()["error"]["code"] == "backtest.invalid_request"


def test_get_backtest_result_series_with_missing_equity_returns_empty_points() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(include_equity=False),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000231"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000231"},
    ).json()["items"][0]

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/equity",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000231"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["points"] == []
    assert payload["source_points"] == 0
    assert payload["returned_points"] == 0
    assert payload["downsampled"] is False


def test_get_backtest_variant_trades_is_paginated_and_csv_is_owner_scoped() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(reverse_trades=True),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000221"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000221"},
    ).json()["items"][0]

    page = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades?page=2&page_size=2",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000221"},
    )
    csv_response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades.csv",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000221"},
    )
    bounded_csv_response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades.csv?max_rows=5",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000221"},
    )
    foreign_csv = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades.csv",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000222"},
    )

    assert page.status_code == 200
    payload = page.json()
    assert payload["pagination"] == {
        "mode": "page",
        "page": 2,
        "page_size": 2,
        "max_page_size": 100,
        "total": 12,
        "has_next": True,
        "has_previous": True,
        "next_page": 3,
        "previous_page": 1,
        "sort": "trade_index_asc",
    }
    assert [item["trade_index"] for item in payload["items"]] == [2, 3]
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert csv_response.headers["x-roehub-trades-row-count"] == "12"
    assert csv_response.headers["x-roehub-trades-total-rows"] == "12"
    assert csv_response.headers["x-roehub-trades-max-rows"] == "10000"
    assert csv_response.headers["x-roehub-trades-truncated"] == "false"
    assert "trade_index,entry_timestamp" in csv_response.text
    assert bounded_csv_response.status_code == 200
    assert bounded_csv_response.headers["x-roehub-trades-row-count"] == "5"
    assert bounded_csv_response.headers["x-roehub-trades-total-rows"] == "12"
    assert bounded_csv_response.headers["x-roehub-trades-max-rows"] == "5"
    assert bounded_csv_response.headers["x-roehub-trades-truncated"] == "true"
    assert len(bounded_csv_response.text.splitlines()) == 6
    assert foreign_csv.status_code == 403
    assert foreign_csv.json()["error"]["code"] == "backtest.forbidden"
    assert repository.jobs is not None


def test_get_backtest_variant_views_use_bounded_cache_readers_not_full_detail_loader() -> None:
    source = inspect.getsource(BacktestJobsUseCase)
    for method_name in (
        "variant_series",
        "monthly_stats",
        "symbol_stats",
        "paginated_trades",
        "trades_csv",
    ):
        method_source = inspect.getsource(getattr(BacktestJobsUseCase, method_name))
        assert "self.trades(" not in method_source
    assert ".read_page(" in source
    assert ".read_series(" in source
    assert ".read_monthly_stats(" in source
    assert ".read_symbol_stats(" in source
    assert ".read_csv(" in source


def test_get_backtest_variant_page_denies_owner_before_cache_read() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000251"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000251"},
    ).json()["items"][0]

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000252"},
    )

    assert response.status_code == 403
    assert lazy_service.requests == ()


def test_get_backtest_variant_stats_are_bounded_by_selected_variant() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000223"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000223"},
    ).json()["items"][0]

    monthly = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/monthly-stats",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000223"},
    )
    symbol = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/symbol-stats",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000223"},
    )

    assert monthly.status_code == 200
    assert monthly.json()["kind"] == "monthly"
    assert monthly.json()["items"][0]["month"] == "2026-01"
    assert monthly.json()["bounds"] == {
        "max_items": 600,
        "returned_items": 9,
        "source_items": 9,
        "truncated": False,
        "sort": "month_asc",
    }
    assert symbol.status_code == 200
    assert symbol.json()["kind"] == "symbol"
    assert symbol.json()["bounds"] == {
        "max_items": 1,
        "returned_items": 1,
        "source_items": 1,
        "truncated": False,
        "sort": "symbol_asc",
    }
    assert symbol.json()["items"] == [
        {
            "symbol": "BTCUSDT",
            "trades_count": 12,
            "net_pnl_quote": 4.0,
            "return_pct": 4.0,
            "wins": 8,
            "losses": 4,
            "win_rate_pct": 66.66666666666666,
        }
    ]


def test_post_backtest_variant_trades_foreign_owner_returns_forbidden() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000216"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000216"},
    ).json()["items"][0]

    response = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000217"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "backtest.forbidden"


def test_get_backtest_result_summary_empty_top_is_typed_missing_data() -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000232"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]), top_count=0)

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/summary",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000232"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_variant_key"] is None
    assert payload["top_variants"]["items"] == []
    assert "selected_variant" not in payload["links"]
    assert lazy_service.requests == ()


@pytest.mark.parametrize(
    "path_template",
    [
        "/backtests/jobs/{job_id}/summary",
        "/backtests/jobs/{job_id}/variants/{variant_key}",
        "/backtests/jobs/{job_id}/variants/{variant_key}/equity",
        "/backtests/jobs/{job_id}/variants/{variant_key}/drawdown",
        "/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats",
        "/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats",
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=1&page_size=2",
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv",
    ],
)
def test_get_backtest_result_endpoint_matrix_invalid_job_returns_not_found(
    path_template: str,
) -> None:
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=_FakeJobRepository(),
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    missing_job_id = "00000000-0000-0000-0000-000000009999"

    response = client.get(
        path_template.format(job_id=missing_job_id, variant_key="job_missing_rank_001"),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000233"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "backtest.not_found"


@pytest.mark.parametrize(
    "path_template",
    [
        "/backtests/jobs/{job_id}/summary",
        "/backtests/jobs/{job_id}/variants/{variant_key}",
        "/backtests/jobs/{job_id}/variants/{variant_key}/equity",
        "/backtests/jobs/{job_id}/variants/{variant_key}/drawdown",
        "/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats",
        "/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats",
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=1&page_size=2",
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv",
    ],
)
def test_get_backtest_result_endpoint_matrix_enforces_owner_scope(
    path_template: str,
) -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000234"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000234"},
    ).json()["items"][0]

    response = client.get(
        path_template.format(
            job_id=created.json()["job_id"],
            variant_key=top["variant_key"],
        ),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000235"},
    )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "backtest.forbidden"


@pytest.mark.parametrize(
    "path_template",
    [
        "/backtests/jobs/{job_id}/variants/{variant_key}",
        "/backtests/jobs/{job_id}/variants/{variant_key}/equity",
        "/backtests/jobs/{job_id}/variants/{variant_key}/drawdown",
        "/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats",
        "/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats",
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=1&page_size=2",
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv",
    ],
)
def test_get_backtest_result_endpoint_matrix_rejects_invalid_variant(
    path_template: str,
) -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000236"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))

    response = client.get(
        path_template.format(
            job_id=created.json()["job_id"],
            variant_key="job_missing_rank_999",
        ),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000236"},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "backtest.not_found"


@pytest.mark.parametrize(
    "path_template",
    [
        "/backtests/jobs/{job_id}/variants/{variant_key}/equity",
        "/backtests/jobs/{job_id}/variants/{variant_key}/drawdown",
        "/backtests/jobs/{job_id}/variants/{variant_key}/monthly-stats",
        "/backtests/jobs/{job_id}/variants/{variant_key}/symbol-stats",
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades?page=1&page_size=2",
        "/backtests/jobs/{job_id}/variants/{variant_key}/trades.csv",
    ],
)
def test_get_backtest_result_endpoint_matrix_returns_materialization_status_on_cache_miss(
    path_template: str,
) -> None:
    repository = _FakeJobRepository()
    lazy_service = _FakeLazyTradesService(cache_hit=False)
    materializations = _FakeMaterializationRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=lazy_service,
            materialization_repository=materializations,
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000237"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000237"},
    ).json()["items"][0]

    response = client.get(
        path_template.format(
            job_id=created.json()["job_id"],
            variant_key=top["variant_key"],
        ),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000237"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["cache"]["status"] == "miss"
    assert payload["pagination"] == {"mode": "none"}
    assert lazy_service.execute_calls == 0
    assert len(materializations.tasks) == 1


def test_get_backtest_variant_trades_rejects_page_size_above_bound() -> None:
    repository = _FakeJobRepository()
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            lazy_trades_service=_FakeLazyTradesService(),
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000238"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))
    top = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/top",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000238"},
    ).json()["items"][0]

    response = client.get(
        f"/backtests/jobs/{created.json()['job_id']}/variants/{top['variant_key']}/trades?page=1&page_size=101",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000238"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


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
    changed_request["top_n"] = 49
    conflict = client.post("/backtests/jobs", headers=headers, json=changed_request)

    assert first.status_code == 201
    assert first.json()["state"] == "queued"
    assert replay.status_code == 200
    assert replay.json()["job_id"] == first.json()["job_id"]
    assert replay.json()["idempotent_replay"] is True
    assert trigger.calls == ((UUID(first.json()["job_id"]), first.json()["request_hash"]),)
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "backtest.idempotency_key_conflict"


def test_post_backtest_job_invalid_request_does_not_create_job() -> None:
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


def test_get_backtest_job_cross_organization_returns_not_found() -> None:
    repository = _FakeJobRepository()
    user_a = UserId.from_string("00000000-0000-0000-0000-000000000271")
    user_b = UserId.from_string("00000000-0000-0000-0000-000000000272")
    resolver = _MappingScopeResolver(
        organizations={
            user_a: OrganizationId.from_string("00000000-0000-0000-0000-000000000011"),
            user_b: OrganizationId.from_string("00000000-0000-0000-0000-000000000012"),
        }
    )
    client = _build_client(
        jobs_use_case=_build_jobs_use_case(
            repository=repository,
            organization_scope_resolver=resolver,
        )
    )
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": str(user_a)},
        json=_valid_request(),
    )

    cross_organization = client.get(
        f"/backtests/jobs/{created.json()['job_id']}",
        headers={"x-user-id": str(user_b)},
    )
    foreign_list = client.get(
        "/backtests/jobs",
        headers={"x-user-id": str(user_b)},
    )

    assert created.status_code == 201
    assert created.json()["organization_id"] == str(resolver.organizations[user_a])
    assert cross_organization.status_code == 404
    assert cross_organization.json()["error"]["code"] == "backtest.not_found"
    assert foreign_list.status_code == 200
    assert foreign_list.json()["items"] == []


def test_post_backtest_job_cancel_terminal_job_is_idempotent() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000209"},
        json=_valid_request(),
    )
    _complete_job(repository=repository, job_id=UUID(created.json()["job_id"]))

    response = client.post(
        f"/backtests/jobs/{created.json()['job_id']}/cancel",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000209"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"


def test_delete_backtest_job_removes_terminal_owner_job_and_variants() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000224"},
        json=_valid_request(),
    )
    job_id = UUID(created.json()["job_id"])
    _complete_job(repository=repository, job_id=job_id)
    assert repository.list_top_variants(
        job_id=job_id,
        organization_id=_ORGANIZATION_ID,
    )

    response = client.delete(
        f"/backtests/jobs/{job_id}",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000224"},
    )
    missing = client.get(
        f"/backtests/jobs/{job_id}",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000224"},
    )

    assert response.status_code == 204
    assert missing.status_code == 404
    assert (
        repository.list_top_variants(
            job_id=job_id,
            organization_id=_ORGANIZATION_ID,
        )
        == ()
    )


def test_delete_backtest_job_rejects_active_and_foreign_jobs() -> None:
    repository = _FakeJobRepository()
    client = _build_client(jobs_use_case=_build_jobs_use_case(repository=repository))
    created = client.post(
        "/backtests/jobs",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000225"},
        json=_valid_request(),
    )

    active = client.delete(
        f"/backtests/jobs/{created.json()['job_id']}",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000225"},
    )
    foreign = client.delete(
        f"/backtests/jobs/{created.json()['job_id']}",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000226"},
    )

    assert active.status_code == 409
    assert active.json()["error"]["code"] == "backtest.job_not_deletable"
    assert foreign.status_code == 403
    assert foreign.json()["error"]["code"] == "backtest.forbidden"


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
            paid_level=PaidLevel(request.headers.get("x-paid-level", "pro")),
        )


class _SequenceClock:
    def __init__(self, *, values: tuple[datetime, ...]) -> None:
        self._values = list(values)

    def now(self) -> datetime:
        if not self._values:
            raise ValueError("_SequenceClock exhausted")
        return self._values.pop(0)


class _StaticMarketDataReader:
    def __init__(self, *, state: str) -> None:
        self._state = state

    def check(self, *, instrument_key: str, timeframe: str, observed_at: datetime):
        return MarketDataReadinessSnapshot(
            state=self._state,  # type: ignore[arg-type]
            reason_code=f"market_data_stream_{self._state}",
            stream_name=f"md.candles.1m.{instrument_key}",
            stream_length=1 if self._state == "ready" else 0,
            last_message_id="1790000000000-0" if self._state == "ready" else None,
            last_observed_at=observed_at if self._state == "ready" else None,
            age_seconds=0 if self._state == "ready" else None,
        )


class _RepoVariantLaunchReader:
    def __init__(self, *, repository: "_FakeJobRepository") -> None:
        self._repository = repository

    def get(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        variant_key: str,
    ) -> BacktestVariantLaunchSnapshot:
        job = self._repository.get(
            job_id=job_id,
            organization_id=_ORGANIZATION_ID,
        )
        if job is None:
            raise RoehubError(
                code="strategy_variant_launch.not_found",
                message="Backtest job was not found",
                details={"reason": "not_found"},
            )
        if job.user_id != user_id:
            raise RoehubError(
                code="strategy_variant_launch.forbidden",
                message="Backtest job does not belong to current user",
                details={"reason": "forbidden"},
            )
        row = self._repository.get_top_variant_by_public_key(
            job_id=job_id,
            organization_id=_ORGANIZATION_ID,
            public_variant_key=variant_key,
        )
        if row is None:
            raise RoehubError(
                code="strategy_variant_launch.not_found",
                message="Backtest variant was not found",
                details={"reason": "not_found"},
            )
        coordinates = _mapping(job.request_json.get("coordinates"))
        payload = _mapping(row.payload_json)
        return BacktestVariantLaunchSnapshot(
            job_id=job.job_id,
            owner_user_id=job.user_id,
            job_state=job.state,
            request_hash=job.request_hash,
            result_config_hash=job.engine_params_hash,
            market_id=int(job.market_id or 1),
            exchange=str(coordinates.get("exchange", "binance")),
            market_type=str(coordinates.get("market_type", "spot")),
            symbol=str(coordinates.get("symbol", job.symbol)),
            timeframe=str(job.timeframe),
            variant_key=str(payload.get("public_variant_key") or variant_key),
            variant_hash=str(payload.get("variant_hash") or row.variant_key),
            indicator_variant_hash=(
                str(payload.get("indicator_variant_hash") or row.indicator_variant_key)
                if (payload.get("indicator_variant_hash") or row.indicator_variant_key)
                else None
            ),
            rank=row.rank,
            summary_metrics=dict(row.summary_metrics_json),
            canonical_variant_params=_mapping(payload.get("canonical_variant_params")),
            readable_params=_mapping(payload.get("readable_params")),
        )


def _find_api_matrix_row(
    *,
    payload: dict[str, Any],
    mode: str,
    market_type: str,
    entry_sizing: str,
    direction: str,
) -> dict[str, Any]:
    return next(
        row
        for row in payload["rows"]
        if row["mode"] == mode
        and row["market_type"] == market_type
        and row["entry_sizing"] == entry_sizing
        and row["direction"] == direction
    )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass
class _FakeArtifactResolver:
    coordinates: tuple[BacktestCoordinates, ...] = ()
    funding_manifest_hash: str | None = None
    funding_coverage_status: str | None = None
    funding_coverage_policy: str | None = None
    funding_rows_count: int | None = None
    funding_expected_event_count: int | None = None
    funding_missing_event_count: int | None = None
    funding_reason_codes: tuple[str, ...] = ()

    def resolve_context(self, *, coordinates: BacktestCoordinates) -> BacktestArtifactMetadata:
        self.coordinates = (*self.coordinates, coordinates)
        return BacktestArtifactMetadata(
            artifact_slot="slot_a",
            artifact_slot_generation=4,
            artifact_manifest_hash="a" * 64,
            artifact_asof_date="2026-03-25",
            hit_times_manifest_hash="b" * 64,
            published_at_utc="2026-03-25T02:00:00Z",
            funding_manifest_hash=self.funding_manifest_hash,
            funding_coverage_status=self.funding_coverage_status,
            funding_coverage_policy=self.funding_coverage_policy,
            funding_rows_count=self.funding_rows_count,
            funding_expected_event_count=self.funding_expected_event_count,
            funding_missing_event_count=self.funding_missing_event_count,
            funding_reason_codes=self.funding_reason_codes,
        )


class _UnavailableArtifactResolver:
    def resolve_context(self, *, coordinates: BacktestCoordinates) -> BacktestArtifactMetadata:
        raise BacktestArtifactContextUnavailable("current pointer missing")


def _build_client(
    *,
    resolver: _FakeArtifactResolver | _UnavailableArtifactResolver | None = None,
    jobs_use_case: BacktestJobsUseCase | None = None,
    create_strategy_from_variant_use_case: Any | None = None,
    compatibility_readiness_service: StrategyCompatibilityReadinessService | None = None,
    scenario_matrix_service: StrategyVariantScenarioMatrixService | None = None,
    backtest_variant_launch_reader: Any | None = None,
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
            organization_scope_resolver=_ScopeResolver(),
            jobs_use_case=jobs_use_case,
            create_strategy_from_variant_use_case=create_strategy_from_variant_use_case,
            compatibility_readiness_service=compatibility_readiness_service,
            scenario_matrix_service=scenario_matrix_service,
            backtest_variant_launch_reader=backtest_variant_launch_reader,
        )
    )
    return TestClient(app)


class _FakeCreateStrategyFromVariantUseCase:
    def __init__(self) -> None:
        self.calls: tuple[tuple[str, str | None], ...] = ()

    def execute(
        self,
        *,
        current_user: CurrentUser,
        job_id: UUID,
        variant_key: str,
        idempotency_key: str | None,
    ) -> CreateStrategyFromBacktestVariantResult:
        if not idempotency_key:
            from trading.platform.errors import RoehubError

            raise RoehubError(
                code="strategy_variant_launch.idempotency_key_required",
                message="Idempotency-Key header is required",
                details={"reason": "idempotency_key_required"},
            )
        self.calls = (*self.calls, (variant_key, idempotency_key))
        strategy = Strategy.create(
            organization_id=current_user.organization_id,
            user_id=current_user.user_id,
            spec=StrategySpecV1.from_json(payload=_strategy_spec_payload()),
            created_at=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
            strategy_id=UUID("00000000-0000-0000-0000-00000000c001"),
        )
        provenance = StrategyBacktestVariantProvenance(
            strategy_id=strategy.strategy_id,
            organization_id=current_user.organization_id,
            user_id=current_user.user_id,
            source_job_id=job_id,
            source_variant_key=variant_key,
            source_variant_hash="a" * 64,
            source_indicator_variant_hash="b" * 64,
            backtest_request_hash="d" * 64,
            backtest_result_config_hash="e" * 64,
            strategy_spec_hash="f" * 64,
            launch_request_hash="1" * 64,
            idempotency_key_hash="2" * 64,
            created_at=datetime(2026, 5, 30, 10, 0, tzinfo=UTC),
        )
        return CreateStrategyFromBacktestVariantResult(
            strategy=strategy,
            provenance=provenance,
            duplicate=False,
        )


class _RaisingCreateStrategyFromVariantUseCase:
    def __init__(self, error: RoehubError) -> None:
        self._error = error

    def execute(
        self,
        *,
        current_user: CurrentUser,
        job_id: UUID,
        variant_key: str,
        idempotency_key: str | None,
    ) -> CreateStrategyFromBacktestVariantResult:
        raise self._error


def _strategy_spec_payload() -> dict[str, Any]:
    return {
        "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
        "instrument_key": "binance:spot:BTCUSDT",
        "market_type": "spot",
        "timeframe": "15m",
        "indicators": [{"id": "ma.dema", "params": {"source": "close", "window": 5}}],
    }


def _build_jobs_use_case(
    *,
    repository: "_FakeJobRepository",
    execution_trigger: Any | None = None,
    lazy_trades_service: Any | None = None,
    materialization_repository: Any | None = None,
    admission_service: BacktestAdmissionService | None = None,
    organization_scope_resolver: Any | None = None,
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
        organization_scope_resolver=organization_scope_resolver or _ScopeResolver(),
        admission_service=admission_service or BacktestAdmissionService(),
        execution_trigger=execution_trigger,
        lazy_trades_service=lazy_trades_service,
        lazy_trades_materialization_repository=materialization_repository,
    )


class _FakeExecutor:
    def execute(
        self,
        *,
        job_id: UUID,
        preflight: Any,
        updated_at: datetime,
    ) -> BacktestJobExecutionResult:
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
        assembly = BacktestTopResultAssemblyService().assemble(
            job_id=job_id,
            normalized_request=preflight.normalized_request,
            top_results=(top_result,),
            updated_at=updated_at,
        )
        return BacktestJobExecutionResult(
            top_variants=assembly.top_variants,
            stage_timings=assembly.stage_timings,
            summary_hash=assembly.summary_hash,
            cleanup_evidence={"result_contains_heavy_references": False},
        )


def _complete_job(
    *,
    repository: "_FakeJobRepository",
    job_id: UUID,
    top_count: int = 1,
    include_funding: bool = False,
) -> None:
    assert repository.jobs is not None
    job = repository.jobs[job_id]
    locked_by = "test-worker"
    now = datetime.now(UTC)
    running = repository.claim_for_inline_execution(
        job_id=job_id,
        organization_id=job.organization_id,
        user_id=job.user_id,
        now=now,
        locked_by=locked_by,
        lease_expires_at=now + timedelta(seconds=60),
    )
    assert running is not None
    top_results = tuple(
        BacktestNoRiskTopResult(
            rank=index,
            score=12.5 - index,
            indicator_rows={"ma.dema": 6 + index},
            metrics=_top_metrics(index=index, include_funding=include_funding),
            metadata=_top_metadata(index=index, include_funding=include_funding),
        )
        for index in range(1, top_count + 1)
    )
    assembly = BacktestTopResultAssemblyService().assemble(
        job_id=job_id,
        normalized_request=dict(job.request_json),
        top_results=top_results,
        updated_at=now,
    )
    finished = repository.finish_with_top_variants(
        job_id=job_id,
        organization_id=job.organization_id,
        user_id=job.user_id,
        now=now,
        locked_by=locked_by,
        next_state="succeeded",
        top_variants=assembly.top_variants,
    )
    assert finished is not None


def _top_metrics(*, index: int, include_funding: bool) -> dict[str, float]:
    metrics = {"total_return_pct": 12.5 - index, "trade_count": 2.0 + index}
    if include_funding and index == 1:
        metrics.update(
            {
                "total_return_pct_net_of_funding": 13.25,
                "funding_return_pct": 1.75,
                "funding_pnl_quote": 75.0,
                "funding_events_count": 3.0,
            }
        )
    return metrics


def _top_metadata(*, index: int, include_funding: bool) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "ma.dema.source": "close",
        "ma.dema.window": 4 + index,
        "confirm_count": 1,
        "proxy_score": 0.25,
    }
    if include_funding and index == 1:
        metadata.update(
            {
                "funding_included": True,
                "funding_data_quality": "degraded",
                "funding_warning_codes": ("missing_leading_coverage",),
                "funding_adjustment_scope": "bounded_candidate_pool",
                "funding_adjustment_exact_global_ranking": False,
                "requested_ranking_metric": "total_return_pct",
                "effective_ranking_metric": "total_return_pct_net_of_funding",
                "funding_candidate_pool_size": 120,
                "requested_top_n": 20,
                "funding_manifest_hash": "f" * 64,
            }
        )
    return metadata


@dataclass
class _FakeLazyTradesService:
    cache_hit: bool = True
    include_equity: bool = True
    reverse_trades: bool = False
    trade_count: int = 12
    requests: tuple[tuple[str, str], ...] = ()
    execute_calls: int = 0
    last_detail: BacktestLazyTradesDetailReadModel | None = None

    def __post_init__(self) -> None:
        self.cache = _FakeLazyTradesCache(service=self)

    def read_cached(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        public_variant_key: str,
    ) -> "_Probe":
        variant_hash = str(row.payload_json["variant_hash"])
        self.requests = (*self.requests, (public_variant_key, variant_hash))
        funding = build_backtest_funding_read_model(
            summary_metrics=row.summary_metrics_json,
            payload=row.payload_json,
            artifact_metadata=_mapping(job.request_json.get("artifact_metadata")),
        )
        funding_manifest_hash = funding.get("funding_manifest_hash")
        chart_overlay: dict[str, Any] = {
            "schema": "backtest_chart_overlay_v1",
            "markers": [],
            "segments": [],
        }
        if funding_manifest_hash is not None:
            chart_overlay.update(
                {
                    "funding_manifest_hash": funding_manifest_hash,
                    "funding_events": [
                        {
                            "kind": "funding_event",
                            "trade_index": 0,
                            "timestamp": "2026-01-01T00:08:00Z",
                            "funding_rate": 0.0001,
                        }
                    ],
                    "funding_events_count": 1,
                    "funding_events_truncated": False,
                }
            )
        trade_range = range(self.trade_count)
        if self.reverse_trades:
            trade_range = range(self.trade_count - 1, -1, -1)
        detail = BacktestLazyTradesDetailReadModel(
            job_id=str(job.job_id),
            variant_key=public_variant_key,
            variant_hash=variant_hash,
            request_hash=job.request_hash,
            engine_params_hash=job.engine_params_hash,
            artifact_manifest_hash=str(
                job.request_json["artifact_metadata"]["artifact_manifest_hash"]
            ),
            funding_manifest_hash=str(funding_manifest_hash)
            if funding_manifest_hash is not None
            else None,
            summary_metrics=dict(row.summary_metrics_json),
            canonical_variant_params=dict(row.payload_json["canonical_variant_params"]),
            readable_params=dict(row.payload_json["readable_params"]),
            trades=tuple(
                _fake_trade(index, include_equity=self.include_equity) for index in trade_range
            ),
            chart_overlay=chart_overlay,
            cache={"status": "hit" if self.cache_hit else "miss"},
            timing={"lazy_trades_cache_hit": 0.001} if self.cache_hit else {},
            funding=funding,
        )
        self.last_detail = detail
        return _Probe(
            detail=detail if self.cache_hit else None,
            cache_status="hit" if self.cache_hit else "miss",
        )

    def execute(
        self,
        *,
        job: BacktestJob,
        row: BacktestJobTopVariant,
        public_variant_key: str,
    ) -> BacktestLazyTradesDetailReadModel:
        self.execute_calls += 1
        return self.read_cached(
            job=job,
            row=row,
            public_variant_key=public_variant_key,
        ).detail  # type: ignore[return-value]


@dataclass
class _FakeLazyTradesCache:
    service: _FakeLazyTradesService

    def read_page(
        self,
        *,
        cache_key: Any,
        now: datetime,
        ttl_seconds: int,
        page: int,
        page_size: int,
    ) -> "_CacheRead":
        _ = cache_key, now, ttl_seconds
        detail = self._detail()
        ordered = sorted(detail.trades, key=lambda item: int(item["trade_index"]))
        offset = (page - 1) * page_size
        items = tuple(dict(item) for item in ordered[offset : offset + page_size])
        total = len(ordered)
        return _CacheRead(
            payload={
                "job_id": detail.job_id,
                "variant_key": detail.variant_key,
                "variant_hash": detail.variant_hash,
                "items": items,
                "pagination": {
                    "mode": "page",
                    "page": page,
                    "page_size": page_size,
                    "max_page_size": 100,
                    "total": total,
                    "has_next": offset + page_size < total,
                    "has_previous": page > 1,
                    "next_page": page + 1 if offset + page_size < total else None,
                    "previous_page": page - 1 if page > 1 else None,
                    "sort": "trade_index_asc",
                },
                "summary_metrics": dict(detail.summary_metrics),
                "cache": dict(detail.cache),
                "timing": dict(detail.timing),
            }
        )

    def read_series(
        self,
        *,
        cache_key: Any,
        now: datetime,
        ttl_seconds: int,
        kind: str,
        points: int,
    ) -> "_CacheRead":
        _ = cache_key, now, ttl_seconds
        detail = self._detail()
        source = [
            {
                "x": trade.get("exit_timestamp") or trade.get("trade_index"),
                "trade_index": trade.get("trade_index"),
                "value": trade.get("equity_after"),
                "net_pnl_quote": trade.get("net_pnl_quote"),
            }
            for trade in sorted(detail.trades, key=lambda item: int(item["trade_index"]))
            if trade.get("equity_after") is not None
        ]
        if kind == "drawdown":
            peak = None
            drawdown = []
            for item in source:
                value = item["value"]
                assert value is not None
                equity = float(value)
                peak = equity if peak is None else max(peak, equity)
                drawdown.append(
                    {
                        "x": item["x"],
                        "trade_index": item["trade_index"],
                        "value": 0.0 if peak <= 0 else ((equity - peak) / peak) * 100.0,
                        "equity": equity,
                    }
                )
            source = drawdown
        sampled = source[:points]
        return _CacheRead(
            payload={
                "job_id": detail.job_id,
                "variant_key": detail.variant_key,
                "variant_hash": detail.variant_hash,
                "kind": kind,
                "points": sampled,
                "requested_points": points,
                "max_points": points,
                "returned_points": len(sampled),
                "source_points": len(source),
                "downsampled": len(sampled) < len(source),
                "cache": dict(detail.cache),
                "timing": dict(detail.timing),
            }
        )

    def read_monthly_stats(
        self,
        *,
        cache_key: Any,
        now: datetime,
        ttl_seconds: int,
    ) -> "_CacheRead":
        _ = cache_key, now, ttl_seconds
        detail = self._detail()
        return _CacheRead(
            payload={
                "job_id": detail.job_id,
                "variant_key": detail.variant_key,
                "variant_hash": detail.variant_hash,
                "items": [
                    {
                        "month": "2026-01",
                        "trades_count": 2,
                        "net_pnl_quote": 1.0,
                        "return_pct": 1.0,
                        "wins": 1,
                        "losses": 1,
                        "win_rate_pct": 50.0,
                    }
                    for _ in range(9)
                ],
                "bounds": {
                    "max_items": 600,
                    "returned_items": 9,
                    "source_items": 9,
                    "truncated": False,
                    "sort": "month_asc",
                },
                "cache": dict(detail.cache),
                "timing": dict(detail.timing),
            }
        )

    def read_symbol_stats(
        self,
        *,
        cache_key: Any,
        now: datetime,
        ttl_seconds: int,
        symbol: str | None,
    ) -> "_CacheRead":
        _ = cache_key, now, ttl_seconds
        detail = self._detail()
        return _CacheRead(
            payload={
                "job_id": detail.job_id,
                "variant_key": detail.variant_key,
                "variant_hash": detail.variant_hash,
                "items": [
                    {
                        "symbol": symbol or "unknown",
                        "trades_count": 12,
                        "net_pnl_quote": 4.0,
                        "return_pct": 4.0,
                        "wins": 8,
                        "losses": 4,
                        "win_rate_pct": 66.66666666666666,
                    }
                ],
                "bounds": {
                    "max_items": 1,
                    "returned_items": 1,
                    "source_items": 1,
                    "truncated": False,
                    "sort": "symbol_asc",
                },
                "cache": dict(detail.cache),
                "timing": dict(detail.timing),
            }
        )

    def read_csv(
        self,
        *,
        cache_key: Any,
        now: datetime,
        ttl_seconds: int,
        max_rows: int,
    ) -> "_CacheRead":
        _ = cache_key, now, ttl_seconds
        detail = self._detail()
        rows = sorted(detail.trades, key=lambda item: int(item["trade_index"]))
        header = "trade_index,entry_timestamp,exit_timestamp\n"
        lines = [
            header,
            *(
                f"{row['trade_index']},{row['entry_timestamp']},{row['exit_timestamp']}\n"
                for row in rows[:max_rows]
            ),
        ]
        return _CacheRead(
            payload={
                "content": "".join(lines),
                "row_count": min(len(rows), max_rows),
                "total_rows": len(rows),
                "max_rows": max_rows,
                "truncated": len(rows) > max_rows,
                "sort": "trade_index_asc",
                "cache": dict(detail.cache),
                "timing": dict(detail.timing),
            }
        )

    def _detail(self) -> BacktestLazyTradesDetailReadModel:
        assert self.service.last_detail is not None
        return self.service.last_detail


@dataclass(frozen=True)
class _CacheRead:
    payload: dict[str, Any]
    status: str = "hit"

    @property
    def is_hit(self) -> bool:
        return self.status == "hit" and self.payload is not None


@dataclass(frozen=True)
class _CacheKey:
    engine_params_hash: str = "e" * 64
    artifact_manifest_hash: str = "a" * 64
    digest: str = "f" * 64


@dataclass(frozen=True)
class _Probe:
    detail: BacktestLazyTradesDetailReadModel | None
    cache_status: str
    cache_key: _CacheKey = _CacheKey()
    cache_warning: str | None = None
    ttl_seconds: int = 172_800
    cache_lookup_s: float = 0.0


@dataclass
class _FakeMaterializationRepository:
    tasks: dict[tuple[str, str, UUID, str, str], BacktestLazyTradesMaterializationTask] = field(
        default_factory=dict
    )

    def request_materialization(
        self,
        *,
        request: BacktestLazyTradesMaterializationRequest,
    ) -> BacktestLazyTradesMaterializationTask:
        key = (
            str(request.organization_id),
            str(request.owner_user_id),
            request.job_id,
            request.public_variant_key,
            request.cache_key,
        )
        existing = self.tasks.get(key)
        if existing is not None:
            return existing
        task = BacktestLazyTradesMaterializationTask(
            task_id=UUID(f"00000000-0000-0000-0000-{len(self.tasks) + 1:012d}"),
            organization_id=request.organization_id,
            owner_user_id=request.owner_user_id,
            job_id=request.job_id,
            public_variant_key=request.public_variant_key,
            variant_hash=request.variant_hash,
            request_hash=request.request_hash,
            engine_params_hash=request.engine_params_hash,
            artifact_manifest_hash=request.artifact_manifest_hash,
            cache_key=request.cache_key,
            status="queued",
            priority_class=request.priority_class,
            created_at=request.requested_at,
            updated_at=request.requested_at,
            started_at=None,
            finished_at=None,
            locked_by=None,
            locked_at=None,
            lease_expires_at=None,
            heartbeat_at=None,
            attempt=0,
            last_error=None,
            last_error_json=None,
            cache_status=request.cache_status,
            cache_path=None,
            ttl_seconds=request.ttl_seconds,
        )
        self.tasks[key] = task
        return task

    def find_by_identity(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        job_id: UUID,
        public_variant_key: str,
        cache_key: str,
    ) -> BacktestLazyTradesMaterializationTask | None:
        return self.tasks.get(
            (
                str(organization_id),
                str(owner_user_id),
                job_id,
                public_variant_key,
                cache_key,
            )
        )

    def count_active_for_user(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
    ) -> int:
        return sum(
            1
            for task in self.tasks.values()
            if task.organization_id == organization_id
            and task.owner_user_id == owner_user_id
            and task.status in {"queued", "running"}
        )

    def count_created_for_user_since(
        self,
        *,
        organization_id: OrganizationId,
        owner_user_id: UserId,
        created_after: datetime,
    ) -> int:
        return sum(
            1
            for task in self.tasks.values()
            if task.organization_id == organization_id
            and task.owner_user_id == owner_user_id
            and task.created_at >= created_after
        )

    def count_active_global(self) -> int:
        return sum(1 for task in self.tasks.values() if task.status in {"queued", "running"})


def _fake_trade(index: int, *, include_equity: bool = True) -> dict[str, Any]:
    pnl = 1.0 if index % 3 != 2 else -1.0
    side = "long" if index % 2 == 0 else "short"
    trade = {
        "trade_index": index,
        "entry_timestamp": f"2026-{(index % 9) + 1:02d}-01T00:00:00Z",
        "exit_timestamp": f"2026-{(index % 9) + 1:02d}-01T00:15:00Z",
        "entry_bar_index": index * 2 + 1,
        "exit_bar_index": index * 2 + 2,
        "side": side,
        "direction": side,
        "entry_price": 100.0 + index,
        "exit_price": 100.0 + index + pnl,
        "quantity": 1.0,
        "notional_quote": 100.0 + index,
        "return_pct": pnl,
        "gross_pnl_quote": pnl,
        "net_pnl_quote": pnl,
        "fee_quote": 0.0,
        "slippage_quote": 0.0,
        "exit_reason": "signal" if index == 0 else "take_profit" if pnl > 0 else "stop_loss",
        "safe_quote_after": 0.0,
        "timeframe": "15m",
    }
    if include_equity:
        trade["equity_after"] = 10000.0 + index + pnl
    return trade


@dataclass
class _FakeExecutionTrigger:
    calls: tuple[tuple[UUID, str], ...] = ()

    def enqueue(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId,
        request_hash: str,
    ) -> None:
        _ = organization_id, user_id
        self.calls = (*self.calls, (job_id, request_hash))


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
        organization_id: OrganizationId,
        user_id: UserId,
        idempotency_key_hash: str,
        created_after: datetime,
    ) -> BacktestJob | None:
        assert self.jobs is not None
        for job in sorted(self.jobs.values(), key=lambda item: item.created_at):
            idempotency = dict(job.request_json).get("idempotency")
            if (
                job.organization_id == organization_id
                and job.user_id == user_id
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
        organization_id: OrganizationId,
        user_id: UserId,
        now: datetime,
        locked_by: str,
        lease_expires_at: datetime,
    ) -> BacktestJob | None:
        assert self.jobs is not None
        job = self.jobs.get(job_id)
        if (
            job is None
            or job.organization_id != organization_id
            or job.user_id != user_id
            or job.state != "queued"
        ):
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
        organization_id: OrganizationId,
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
        if (
            job is None
            or job.organization_id != organization_id
            or job.user_id != user_id
            or job.state != "running"
        ):
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

    def get(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId | None = None,
    ) -> BacktestJob | None:
        assert self.jobs is not None
        job = self.jobs.get(job_id)
        if job is None:
            return None
        if job.organization_id != organization_id:
            return None
        if user_id is not None and job.user_id != user_id:
            return None
        return job

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        assert self.jobs is not None
        items = [
            job
            for job in self.jobs.values()
            if job.organization_id == query.organization_id
            and job.user_id == query.user_id
            and (query.state is None or job.state == query.state)
        ]
        items.sort(key=lambda item: (item.created_at, str(item.job_id)), reverse=True)
        return BacktestJobListPage(items=tuple(items[: query.limit]), next_cursor=None)

    def list_top_variants(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        limit: int | None = None,
    ) -> tuple[BacktestJobTopVariant, ...]:
        assert self.jobs is not None
        assert self.top_rows is not None
        job = self.jobs.get(job_id)
        rows = (
            self.top_rows.get(job_id, ())
            if job is not None and job.organization_id == organization_id
            else ()
        )
        return rows if limit is None else rows[:limit]

    def get_top_variant_by_public_key(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        public_variant_key: str,
    ) -> BacktestJobTopVariant | None:
        for row in self.list_top_variants(
            job_id=job_id,
            organization_id=organization_id,
        ):
            if row.payload_json.get("public_variant_key") == public_variant_key:
                return row
        return None

    def cancel(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        assert self.jobs is not None
        job = self.jobs.get(job_id)
        if job is None or job.organization_id != organization_id or job.user_id != user_id:
            return None
        cancelled = job.request_cancel(changed_at=cancel_requested_at)
        self.jobs[job_id] = cancelled
        return cancelled

    def delete_terminal(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId,
    ) -> bool:
        assert self.jobs is not None
        assert self.top_rows is not None
        job = self.jobs.get(job_id)
        if (
            job is None
            or job.organization_id != organization_id
            or job.user_id != user_id
            or job.state
            not in {
                "succeeded",
                "failed",
                "cancelled",
            }
        ):
            return False
        del self.jobs[job_id]
        self.top_rows.pop(job_id, None)
        return True

    def count_active_for_user(self, *, organization_id: OrganizationId, user_id: UserId) -> int:
        assert self.jobs is not None
        return sum(
            1
            for job in self.jobs.values()
            if job.organization_id == organization_id and job.user_id == user_id and job.is_active()
        )

    def count_created_for_user_since(
        self,
        *,
        organization_id: OrganizationId,
        user_id: UserId,
        created_after: datetime,
    ) -> int:
        assert self.jobs is not None
        return sum(
            1
            for job in self.jobs.values()
            if job.organization_id == organization_id
            and job.user_id == user_id
            and job.created_at >= created_after
        )

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
            "end": "2026-03-25T20:08:00Z",
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
            "direction_mode": "long_only",
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
        "top_n": 50,
    }
