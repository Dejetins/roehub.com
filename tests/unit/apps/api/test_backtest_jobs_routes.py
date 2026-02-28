from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.dto import encode_backtest_jobs_cursor
from apps.api.routes import build_backtest_jobs_router
from trading.contexts.backtest.application.ports import BacktestJobListPage
from trading.contexts.backtest.application.use_cases import BacktestJobTopReadResult
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobErrorPayload,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import PaidLevel, UserId


class _HeaderCurrentUserDependency:
    """
    Request dependency resolving authenticated principal from `X-User-Id` header.
    """

    def __call__(self, request: Request):
        """
        Resolve principal or raise deterministic HTTP 401 payload.

        Args:
            request: HTTP request object.
        Returns:
            object: CurrentUserPrincipal-compatible object.
        Assumptions:
            Header contains UUID string when provided.
        Raises:
            HTTPException: If authentication header is missing.
        Side Effects:
            None.
        """
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )

        from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal

        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


@dataclass
class _CreateUseCaseFake:
    """
    Deterministic create use-case fake returning preconfigured job or error.
    """

    job: BacktestJob | None = None
    error: Exception | None = None
    last_command: Any = None

    def execute(self, *, command, current_user):
        """
        Return configured job snapshot or raise configured exception.

        Args:
            command: Create command payload.
            current_user: Authenticated user payload.
        Returns:
            BacktestJob: Configured job snapshot.
        Assumptions:
            Command/current_user are inspected by tests only through side effects.
        Raises:
            Exception: Configured exception.
        Side Effects:
            Stores last command payload for assertions.
        """
        _ = current_user
        self.last_command = command
        if self.error is not None:
            raise self.error
        if self.job is None:  # pragma: no cover - guarded by test fixtures
            raise ValueError("create fake requires job")
        return self.job


@dataclass
class _StatusUseCaseFake:
    """
    Deterministic status use-case fake returning preconfigured job or error.
    """

    job: BacktestJob | None = None
    error: Exception | None = None

    def execute(self, *, job_id: UUID, current_user):
        """
        Return configured job snapshot or raise configured exception.

        Args:
            job_id: Requested job identifier.
            current_user: Authenticated user payload.
        Returns:
            BacktestJob: Configured job snapshot.
        Assumptions:
            `job_id/current_user` are irrelevant for static fake response.
        Raises:
            Exception: Configured exception.
        Side Effects:
            None.
        """
        _ = job_id, current_user
        if self.error is not None:
            raise self.error
        if self.job is None:  # pragma: no cover - guarded by test fixtures
            raise ValueError("status fake requires job")
        return self.job


@dataclass
class _TopUseCaseFake:
    """
    Deterministic top use-case fake returning preconfigured result or error.
    """

    result: BacktestJobTopReadResult | None = None
    error: Exception | None = None

    def execute(self, *, job_id: UUID, current_user, limit: int | None):
        """
        Return configured top result payload or raise configured exception.

        Args:
            job_id: Requested job identifier.
            current_user: Authenticated user payload.
            limit: Optional top rows limit.
        Returns:
            BacktestJobTopReadResult: Configured top payload.
        Assumptions:
            Fake does not enforce limit semantics.
        Raises:
            Exception: Configured exception.
        Side Effects:
            None.
        """
        _ = job_id, current_user, limit
        if self.error is not None:
            raise self.error
        if self.result is None:  # pragma: no cover - guarded by test fixtures
            raise ValueError("top fake requires result")
        return self.result


@dataclass
class _ListUseCaseFake:
    """
    Deterministic list use-case fake returning preconfigured page payload or error.
    """

    page: BacktestJobListPage
    error: Exception | None = None
    last_cursor: BacktestJobListCursor | None = None
    last_state: str | None = None

    def execute(self, *, current_user, state: str | None, limit: int, cursor):
        """
        Return configured list page and record decoded cursor/state arguments.

        Args:
            current_user: Authenticated user payload.
            state: Optional state filter.
            limit: Page size.
            cursor: Optional decoded keyset cursor.
        Returns:
            BacktestJobListPage: Configured page fixture.
        Assumptions:
            Fake does not validate state/limit values.
        Raises:
            Exception: Configured exception.
        Side Effects:
            Stores last cursor and state payloads for assertions.
        """
        _ = current_user, state, limit
        self.last_cursor = cursor
        self.last_state = state
        if self.error is not None:
            raise self.error
        return self.page


@dataclass
class _CancelUseCaseFake:
    """
    Deterministic cancel use-case fake returning preconfigured job or error.
    """

    job: BacktestJob | None = None
    error: Exception | None = None

    def execute(self, *, job_id: UUID, current_user):
        """
        Return configured cancel status payload or raise configured exception.

        Args:
            job_id: Requested job identifier.
            current_user: Authenticated user payload.
        Returns:
            BacktestJob: Configured updated job snapshot.
        Assumptions:
            Fake ignores input ids and always returns fixture job.
        Raises:
            Exception: Configured exception.
        Side Effects:
            None.
        """
        _ = job_id, current_user
        if self.error is not None:
            raise self.error
        if self.job is None:  # pragma: no cover - guarded by test fixtures
            raise ValueError("cancel fake requires job")
        return self.job



def _build_client(
    *,
    create_use_case: _CreateUseCaseFake | None = None,
    status_use_case: _StatusUseCaseFake | None = None,
    top_use_case: _TopUseCaseFake | None = None,
    list_use_case: _ListUseCaseFake | None = None,
    cancel_use_case: _CancelUseCaseFake | None = None,
) -> tuple[TestClient, _ListUseCaseFake]:
    """
    Build minimal FastAPI TestClient with EPIC-11 jobs router and shared error handlers.

    Args:
        create_use_case: Optional create use-case fake.
        status_use_case: Optional status use-case fake.
        top_use_case: Optional top use-case fake.
        list_use_case: Optional list use-case fake.
        cancel_use_case: Optional cancel use-case fake.
    Returns:
        tuple[TestClient, _ListUseCaseFake]: Configured client and resolved list fake.
    Assumptions:
        Shared API error handlers provide deterministic Roehub/422 payloads.
    Raises:
        ValueError: If router dependencies are invalid.
    Side Effects:
        None.
    """
    base_job = _queued_job(job_id=UUID("00000000-0000-0000-0000-000000000910"))
    resolved_list_use_case = list_use_case or _ListUseCaseFake(
        page=BacktestJobListPage(items=(base_job,), next_cursor=None)
    )

    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtest_jobs_router(
            create_use_case=create_use_case or _CreateUseCaseFake(job=base_job),  # type: ignore[arg-type]
            get_status_use_case=status_use_case or _StatusUseCaseFake(job=base_job),  # type: ignore[arg-type]
            get_top_use_case=top_use_case
            or _TopUseCaseFake(result=_top_result(job=base_job, include_details=False)),  # type: ignore[arg-type]
            list_use_case=resolved_list_use_case,  # type: ignore[arg-type]
            cancel_use_case=cancel_use_case or _CancelUseCaseFake(job=base_job),  # type: ignore[arg-type]
            current_user_dependency=_HeaderCurrentUserDependency(),
        )
    )
    return TestClient(app), resolved_list_use_case



def test_post_backtest_jobs_returns_201_with_status_hash_fields() -> None:
    """
    Verify `POST /backtests/jobs` returns created status payload with required hash fields.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Route delegates mode parsing to sync request mapper.
    Raises:
        AssertionError: If status code/payload diverges from EPIC-11 contract.
    Side Effects:
        None.
    """
    created_job = _queued_job(job_id=UUID("00000000-0000-0000-0000-000000000911"))
    create_fake = _CreateUseCaseFake(job=created_job)
    client, _ = _build_client(create_use_case=create_fake)

    response = client.post(
        "/backtests/jobs",
        json=_template_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["job_id"] == "00000000-0000-0000-0000-000000000911"
    assert body["request_hash"] == "a" * 64
    assert body["engine_params_hash"] == "b" * 64
    assert body["backtest_runtime_config_hash"] == "c" * 64
    assert create_fake.last_command is not None



def test_get_backtest_job_status_returns_failed_last_error_payload() -> None:
    """
    Verify `GET /backtests/jobs/{job_id}` includes failed error payload fields.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Domain failed-job invariant guarantees both fields are present.
    Raises:
        AssertionError: If failed payload fields are missing or malformed.
    Side Effects:
        None.
    """
    failed_job = _failed_job(job_id=UUID("00000000-0000-0000-0000-000000000912"))
    client, _ = _build_client(status_use_case=_StatusUseCaseFake(job=failed_job))

    response = client.get(
        "/backtests/jobs/00000000-0000-0000-0000-000000000912",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    assert response.json()["last_error"] == "Execution failed"
    assert response.json()["last_error_json"] == {
        "code": "unexpected_error",
        "message": "Execution failed",
        "details": {"stage": "stage_b"},
    }



def test_get_backtest_job_status_maps_forbidden_to_403_payload() -> None:
    """
    Verify status route maps owner-policy `forbidden` error to stable HTTP 403 payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Status use-case raises RoehubError for foreign existing job.
    Raises:
        AssertionError: If status code or payload diverges from canonical error contract.
    Side Effects:
        None.
    """
    status_fake = _StatusUseCaseFake(
        error=RoehubError(
            code="forbidden",
            message="Backtest job is forbidden",
            details={"job_id": "00000000-0000-0000-0000-000000000912"},
        )
    )
    client, _ = _build_client(status_use_case=status_fake)

    response = client.get(
        "/backtests/jobs/00000000-0000-0000-0000-000000000912",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "forbidden",
            "message": "Backtest job is forbidden",
            "details": {
                "job_id": "00000000-0000-0000-0000-000000000912",
            },
        }
    }


def test_get_backtest_job_status_maps_not_found_to_404_payload() -> None:
    """
    Verify status route maps missing-job `not_found` error to stable HTTP 404 payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Status use-case raises RoehubError when job id is absent.
    Raises:
        AssertionError: If status code or payload diverges from canonical error contract.
    Side Effects:
        None.
    """
    status_fake = _StatusUseCaseFake(
        error=RoehubError(
            code="not_found",
            message="Backtest job not found",
            details={"job_id": "00000000-0000-0000-0000-000000000913"},
        )
    )
    client, _ = _build_client(status_use_case=status_fake)

    response = client.get(
        "/backtests/jobs/00000000-0000-0000-0000-000000000913",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "Backtest job not found",
            "details": {
                "job_id": "00000000-0000-0000-0000-000000000913",
            },
        }
    }


def test_get_backtest_job_top_hides_details_for_non_succeeded_jobs() -> None:
    """
    Verify `/top` omits `report_table_md` and `trades` fields for non-succeeded jobs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        EPIC-11 lazy-details policy is enforced by DTO mapping layer.
    Raises:
        AssertionError: If response leaks report/trades in non-succeeded state.
    Side Effects:
        None.
    """
    running_job = _running_job(job_id=UUID("00000000-0000-0000-0000-000000000913"))
    client, _ = _build_client(
        top_use_case=_TopUseCaseFake(
            result=_top_result(job=running_job, include_details=True),
        ),
    )

    response = client.get(
        "/backtests/jobs/00000000-0000-0000-0000-000000000913/top?limit=1",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    payload = response.json()
    context = payload["report_context"]
    assert context["strategy_id"] is None
    assert context["template"]["timeframe"] == "1m"
    assert context["warmup_bars"] == 200
    assert context["include_trades"] is True
    item = payload["items"][0]
    assert "report_table_md" not in item
    assert "trades" not in item



def test_get_backtest_job_top_omits_details_even_for_succeeded_jobs() -> None:
    """
    Verify `/top` omits `report_table_md` and `trades` even for `succeeded` jobs.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Detailed report data is loaded through variant-report endpoint, not `/top`.
    Raises:
        AssertionError: If succeeded response still includes report/trades details.
    Side Effects:
        None.
    """
    succeeded_job = _succeeded_job(job_id=UUID("00000000-0000-0000-0000-000000000914"))
    client, _ = _build_client(
        top_use_case=_TopUseCaseFake(
            result=_top_result(job=succeeded_job, include_details=True),
        ),
    )

    response = client.get(
        "/backtests/jobs/00000000-0000-0000-0000-000000000914/top?limit=1",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["report_context"] is not None
    item = payload["items"][0]
    assert "report_table_md" not in item
    assert "trades" not in item



def test_list_backtest_jobs_decodes_cursor_and_returns_next_cursor() -> None:
    """
    Verify list endpoint decodes opaque cursor and returns encoded `next_cursor`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cursor transport format is opaque `base64url(json)`.
    Raises:
        AssertionError: If decoded cursor or next cursor payload is inconsistent.
    Side Effects:
        None.
    """
    previous_cursor = BacktestJobListCursor(
        created_at=datetime(2026, 2, 23, 11, 50, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000915"),
    )
    next_cursor = BacktestJobListCursor(
        created_at=datetime(2026, 2, 23, 11, 40, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000916"),
    )
    job = _queued_job(job_id=UUID("00000000-0000-0000-0000-000000000917"))
    list_fake = _ListUseCaseFake(page=BacktestJobListPage(items=(job,), next_cursor=next_cursor))
    client, resolved_list_fake = _build_client(list_use_case=list_fake)

    response = client.get(
        f"/backtests/jobs?limit=25&cursor={encode_backtest_jobs_cursor(cursor=previous_cursor)}",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    assert resolved_list_fake.last_cursor == previous_cursor
    assert response.json()["next_cursor"] == encode_backtest_jobs_cursor(cursor=next_cursor)


def test_list_backtest_jobs_accepts_blank_state_and_cursor_query_values() -> None:
    """
    Verify list endpoint treats blank `state` and blank `cursor` as missing filters.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Backward compatibility requires accepting `state=` and `cursor=` from UI/client links.
    Raises:
        AssertionError: If blank query values still fail validation or are passed as non-empty.
    Side Effects:
        None.
    """
    listed_job = _queued_job(job_id=UUID("00000000-0000-0000-0000-000000000919"))
    list_fake = _ListUseCaseFake(page=BacktestJobListPage(items=(listed_job,), next_cursor=None))
    client, resolved_list_fake = _build_client(list_use_case=list_fake)

    response = client.get(
        "/backtests/jobs?state=&limit=25&cursor=",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    assert resolved_list_fake.last_state is None
    assert resolved_list_fake.last_cursor is None


def test_list_backtest_jobs_rejects_unknown_state_with_deterministic_payload() -> None:
    """
    Verify list endpoint rejects unknown non-empty `state` with canonical validation payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed states are fixed by Backtest Jobs API v1 contract.
    Raises:
        AssertionError: If status code or payload differs from deterministic validation contract.
    Side Effects:
        None.
    """
    client, _ = _build_client()

    response = client.get(
        "/backtests/jobs?state=done",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Invalid jobs state filter",
            "details": {
                "errors": [
                    {
                        "path": "query.state",
                        "code": "invalid_value",
                        "message": (
                            "state must be one of: queued, running, succeeded, failed, cancelled"
                        ),
                    }
                ]
            },
        }
    }


def test_list_backtest_jobs_rejects_invalid_cursor_with_deterministic_payload() -> None:
    """
    Verify list endpoint returns deterministic `validation_error` payload for malformed cursor.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cursor parser raises `BacktestValidationError` with canonical details.
    Raises:
        AssertionError: If payload/status code differs from deterministic contract.
    Side Effects:
        None.
    """
    client, _ = _build_client()

    response = client.get(
        "/backtests/jobs?cursor=%%%",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Invalid jobs cursor",
            "details": {
                "errors": [
                    {
                        "path": "query.cursor",
                        "code": "invalid_cursor",
                        "message": "cursor must be base64url(json)",
                    }
                ]
            },
        }
    }



def test_cancel_backtest_job_returns_status_payload() -> None:
    """
    Verify cancel endpoint returns status payload (not `204`) for idempotent UI polling flow.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cancel use-case returns updated owner snapshot.
    Raises:
        AssertionError: If endpoint status/payload diverges from EPIC-11 contract.
    Side Effects:
        None.
    """
    cancelled_job = _queued_job(job_id=UUID("00000000-0000-0000-0000-000000000918")).request_cancel(
        changed_at=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc)
    )
    client, _ = _build_client(cancel_use_case=_CancelUseCaseFake(job=cancelled_job))

    response = client.post(
        "/backtests/jobs/00000000-0000-0000-0000-000000000918/cancel",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000111"},
    )

    assert response.status_code == 200
    assert response.json()["state"] == "cancelled"



def _top_result(*, job: BacktestJob, include_details: bool) -> BacktestJobTopReadResult:
    """
    Build deterministic top-use-case result fixture for routes tests.

    Args:
        job: Job fixture used for top response state policy.
        include_details: Whether row fixture should include report/trades payload.
    Returns:
        BacktestJobTopReadResult: Top result fixture.
    Assumptions:
        Route mapping enforces details visibility based on job state.
    Raises:
        ValueError: If top-row fixture violates entity invariants.
    Side Effects:
        None.
    """
    row = BacktestJobTopVariant(
        job_id=job.job_id,
        rank=1,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        variant_index=0,
        total_return_pct=10.0,
        payload_json={"schema_version": 1},
        report_table_md="|Metric|Value|" if include_details else None,
        trades_json=({"trade_id": 1, "net_pnl_quote": 12.34},) if include_details else None,
        updated_at=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
    )
    return BacktestJobTopReadResult(job=job, rows=(row,))



def _queued_job(*, job_id: UUID) -> BacktestJob:
    """
    Build deterministic queued job fixture for jobs routes tests.

    Args:
        job_id: Deterministic job identifier.
    Returns:
        BacktestJob: Queued domain job fixture.
    Assumptions:
        Job belongs to request principal used in route tests.
    Raises:
        ValueError: If fixture violates domain invariants.
    Side Effects:
        None.
    """
    return BacktestJob.create_queued(
        job_id=job_id,
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        created_at=datetime(2026, 2, 23, 11, 30, tzinfo=timezone.utc),
        request_json={
            "time_range": {
                "start": "2026-02-21T00:00:00+00:00",
                "end": "2026-02-21T01:00:00+00:00",
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
            "warmup_bars": 200,
            "top_k": 5,
            "preselect": 20000,
            "top_trades_n": 2,
        },
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
    )



def _running_job(*, job_id: UUID) -> BacktestJob:
    """
    Build deterministic running job fixture for top state-policy route tests.

    Args:
        job_id: Deterministic job identifier.
    Returns:
        BacktestJob: Running job fixture.
    Assumptions:
        Lease fields are valid and non-expired for running state.
    Raises:
        ValueError: If fixture violates domain invariants.
    Side Effects:
        None.
    """
    queued = _queued_job(job_id=job_id)
    return queued.claim(
        changed_at=datetime(2026, 2, 23, 11, 35, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        lease_expires_at=datetime(2026, 2, 23, 11, 36, tzinfo=timezone.utc),
    )



def _succeeded_job(*, job_id: UUID) -> BacktestJob:
    """
    Build deterministic succeeded job fixture for top-details route tests.

    Args:
        job_id: Deterministic job identifier.
    Returns:
        BacktestJob: Succeeded job fixture.
    Assumptions:
        Running fixture transitions to terminal succeeded state.
    Raises:
        ValueError: If fixture violates lifecycle invariants.
    Side Effects:
        None.
    """
    running = _running_job(job_id=job_id)
    return running.finish(
        next_state="succeeded",
        changed_at=datetime(2026, 2, 23, 11, 37, tzinfo=timezone.utc),
    )



def _failed_job(*, job_id: UUID) -> BacktestJob:
    """
    Build deterministic failed job fixture with Roehub-like error payload.

    Args:
        job_id: Deterministic job identifier.
    Returns:
        BacktestJob: Failed job fixture.
    Assumptions:
        Failed state includes both short and structured error payload fields.
    Raises:
        ValueError: If fixture violates lifecycle invariants.
    Side Effects:
        None.
    """
    running = _running_job(job_id=job_id)
    return running.finish(
        next_state="failed",
        changed_at=datetime(2026, 2, 23, 11, 37, tzinfo=timezone.utc),
        last_error="Execution failed",
        last_error_json=BacktestJobErrorPayload(
            code="unexpected_error",
            message="Execution failed",
            details={"stage": "stage_b"},
        ),
    )



def _template_payload() -> dict[str, Any]:
    """
    Build minimal valid template-mode request payload for jobs create route tests.

    Args:
        None.
    Returns:
        dict[str, Any]: Request payload fixture.
    Assumptions:
        Payload matches strict `BacktestsPostRequest` envelope.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "time_range": {
            "start": "2026-02-21T00:00:00Z",
            "end": "2026-02-21T01:00:00Z",
        },
        "template": {
            "instrument_id": {
                "market_id": 1,
                "symbol": "BTCUSDT",
            },
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
    }
