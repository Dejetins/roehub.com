"""Authenticated, bounded catalog/coverage reads and durable Market Data commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Callable, Literal, Mapping
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from apps.api.routes.market_data_reference import (
    _catalog_instrument_or_error,
    _resolve_research_scope,
)
from trading.contexts.backtest.application.ports import ResearchOrganizationScopeResolver
from trading.contexts.identity.adapters.inbound.api.csrf import same_origin_rejection_reason
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.market_data.adapters.outbound.persistence.postgres.catalog_snapshot_repository import (  # noqa: E501
    PostgresCatalogSnapshotRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.instrument_selection_repository import (  # noqa: E501
    PostgresInstrumentSelectionRepository,
)
from trading.contexts.market_data.adapters.outbound.persistence.postgres.work_request_repository import (  # noqa: E501
    ActiveWorkRequestError,
    PostgresMarketDataWorkRequestRepository,
)
from trading.contexts.market_data.application.ports.stores.canonical_candle_index_reader import (
    CanonicalCandleIndexReader,
)
from trading.contexts.market_data.application.services.work_requests import WorkRequest
from trading.contexts.market_data.application.use_cases import (
    ListEnabledMarketsUseCase,
    SearchEnabledTradableInstrumentsUseCase,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class WorkRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: UUID
    kind: Literal["catalog_refresh", "candle_ingestion"]
    market_id: int = Field(ge=1)
    symbols: list[str] = Field(default_factory=list, max_length=8)
    timeframe: Literal["1m"] = "1m"
    start_at: datetime | None = None
    end_at: datetime | None = None


class RetryBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt: int = Field(ge=1, le=5)


class WorkBatchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    idempotency_key: UUID
    kind: Literal["candle_ingestion"] = "candle_ingestion"
    market_id: int = Field(ge=1)
    symbols: list[str] = Field(min_length=1, max_length=50)
    timeframe: Literal["1m"] = "1m"
    start_at: datetime
    end_at: datetime


class ControlBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    control_version: int = Field(ge=0)


class WorkBacklogResponse(BaseModel):
    queued: int
    running: int
    cancel_requested: int
    paused: int = 0
    retry_wait: int = 0
    oldest_queued_at: datetime | None
    observed_at: datetime


class WorkAttemptResponse(BaseModel):
    attempt: int
    state: str
    error_code: str | None
    finished_at: datetime
    rows_read: int | None = None
    rows_written: int | None = None


class WorkRequestResponse(BaseModel):
    # Explicit projection prevents worker tokens, actor identifiers, and future
    # storage-only fields from leaking into browser caches or diagnostics.
    job_id: UUID
    kind: str
    market_id: int
    symbols: list[str]
    timeframe: str
    start_at: datetime | None
    end_at: datetime | None
    state: str
    attempt: int
    completed_units: int
    total_units: int
    rows_read: int
    rows_written: int
    error_code: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    attempts: list[WorkAttemptResponse]
    can_cancel: bool
    can_retry: bool
    can_pause: bool = False
    can_resume: bool = False
    control_version: int = 0
    progress_epoch: int = 0
    retry_count: int = 0
    next_retry_at: datetime | None = None
    error_phase: str | None = None
    queue_waiting: bool = False
    storage_retry_at: datetime | None = None


class WorkEventResponse(BaseModel):
    event_id: int
    event_type: str
    occurred_at: datetime
    state: str
    attempt: int
    completed_units: int
    total_units: int
    rows_written: int
    error_code: str | None
    error_phase: str | None
    next_retry_at: datetime | None


def _response(row: Mapping[str, Any]) -> WorkRequestResponse:
    return WorkRequestResponse.model_validate(
        {
            **row,
            # A command acknowledgement does not prove acquisition of the worker.
            "queue_waiting": row.get("queue_waiting", row["state"] == "queued"),
            "can_cancel": row["state"] in {
                "queued", "running", "pause_requested", "paused", "retry_wait"
            },
            "can_pause": row["state"] in {"queued", "running", "retry_wait"},
            "can_resume": row["state"] == "paused",
            "can_retry": row["state"] == "failed" and row["attempt"] < 5,
        }
    )


def require_market_data_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin is not None:
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=403, detail="Mutation origin is not allowed")
    reason = same_origin_rejection_reason(request=request, fail_closed_without_origin=True)
    if reason:
        raise HTTPException(status_code=403, detail="Mutation origin is not allowed")


def build_market_data_workspace_router(
    *,
    current_user_dependency: Callable[[Request], CurrentUserPrincipal],
    scope_resolver: ResearchOrganizationScopeResolver | None,
    selections: PostgresInstrumentSelectionRepository,
    snapshots: PostgresCatalogSnapshotRepository,
    jobs: PostgresMarketDataWorkRequestRepository,
    markets: ListEnabledMarketsUseCase,
    search: SearchEnabledTradableInstrumentsUseCase,
    index: CanonicalCandleIndexReader,
) -> APIRouter:
    router = APIRouter(tags=["market-data"])

    def organization(principal: CurrentUserPrincipal):
        return _resolve_research_scope(resolver=scope_resolver, principal=principal).organization_id

    def require_market(market_id: int) -> None:
        if not any(m.market_id.value == market_id for m in markets.execute()):
            raise HTTPException(404, "Source is not available")

    @router.get("/market-data/workspace/backlog", response_model=WorkBacklogResponse)
    def backlog(
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> WorkBacklogResponse:
        org = organization(principal)
        return WorkBacklogResponse.model_validate({
            **jobs.backlog(organization_id=org.value),
            "observed_at": datetime.now(UTC),
        })

    @router.get("/market-data/workspace/catalog")
    def catalog(
        market_id: int = Query(ge=1),
        q: str = Query(default="", max_length=64),
        snapshot_id: UUID | None = None,
        after: str = Query(default="", max_length=64),
        limit: int = Query(default=50, ge=1, le=100),
        selected_only: bool = False,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> dict[str, Any]:
        org = organization(principal)
        require_market(market_id)
        snapshot = snapshots.latest(market_id=market_id, snapshot_id=snapshot_id)
        state = selections.catalog_state(market_id=MarketId(market_id), now=datetime.now(UTC))
        if snapshot is None:
            if snapshot_id is not None:
                raise HTTPException(409, "Catalog snapshot expired; refresh the list")
            return dict(
                snapshot_id=None,
                refreshed_at=None,
                catalog_state=state,
                items=[],
                next_cursor=None,
                total=0,
            )
        selected = {
            str(r.instrument_id.symbol)
            for r in selections.list_for_organization(organization_id=org)
            if r.instrument_id.market_id.value == market_id
        }
        raw = snapshots.page(
            snapshot_id=snapshot["snapshot_id"],
            prefix=q.strip().upper(),
            after=after,
            limit=limit + 1,
            selected_symbols=sorted(selected) if selected_only else None,
        )
        if not raw and snapshots.latest(
            market_id=market_id, snapshot_id=snapshot["snapshot_id"]
        ) is None:
            raise HTTPException(409, "Catalog snapshot expired; refresh the list")
        if state == "fresh" and snapshot["refreshed_at"] < datetime.now(UTC)-timedelta(minutes=30):
            state = "stale"
        items = []
        for row in raw[:limit]:
            item = row["item"]
            instrument = InstrumentId(MarketId(market_id), Symbol(item["symbol"]))
            pinned = selections.is_strategy_pinned(organization_id=org, instrument_id=instrument)
            items.append(
                {
                    **item,
                    "market_id": market_id,
                    "selected": item["symbol"] in selected,
                    "strategy_pinned": pinned,
                    "effective": pinned or item["symbol"] in selected,
                }
            )
        return dict(
            snapshot_id=snapshot["snapshot_id"],
            refreshed_at=snapshot["refreshed_at"],
            catalog_state=state,
            items=items,
            total=snapshot["total"],
            next_cursor=items[-1]["symbol"] if len(raw) > limit else None,
        )

    @router.get("/market-data/workspace/coverage")
    def coverage(
        market_id: int,
        symbol: str = Query(max_length=64),
        start_at: datetime = Query(),
        end_at: datetime = Query(),
        timeframe: Literal["1m"] = "1m",
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ) -> dict[str, Any]:
        organization(principal)
        request = WorkRequest("candle_ingestion", market_id, (symbol,), start_at, end_at, timeframe)
        try:
            expected = request.validate(now=datetime.now(UTC))
            if expected > 10080:
                raise ValueError("coverage reads are limited to seven days")
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        instrument = _catalog_instrument_or_error(
            search_use_case=search, market_id=market_id, symbol=symbol
        )
        present = {
            t.value
            for t in index.distinct_ts_opens(
                instrument_id=instrument,
                time_range=TimeRange(UtcTimestamp(start_at), UtcTimestamp(end_at)),
            )
        }
        gaps: list[dict[str, Any]] = []
        gap_start = None
        cursor = start_at
        while cursor < end_at:
            if cursor not in present and gap_start is None:
                gap_start = cursor
            elif cursor in present and gap_start is not None:
                gaps.append(
                    dict(
                        start_at=gap_start,
                        end_at=cursor,
                        missing_candles=int((cursor - gap_start).total_seconds() // 60),
                    )
                )
                gap_start = None
            cursor += timedelta(minutes=1)
        if gap_start is not None:
            gaps.append(
                dict(
                    start_at=gap_start,
                    end_at=end_at,
                    missing_candles=int((end_at - gap_start).total_seconds() // 60),
                )
            )
        return dict(
            market_id=market_id,
            symbol=symbol,
            timeframe=timeframe,
            start_at=start_at,
            end_at=end_at,
            expected_candles=expected,
            actual_candles=len(present),
            coverage_percent=round(100 * len(present) / expected, 2),
            state="complete" if len(present) == expected else "empty" if not present else "partial",
            gaps=gaps[:100],
            gap_count=len(gaps),
            gaps_truncated=len(gaps) > 100,
            observed_at=datetime.now(UTC),
        )

    @router.post("/market-data/work-requests", response_model=WorkRequestResponse)
    def submit(
        body: WorkRequestBody,
        request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ):
        require_market_data_origin(request)
        org = organization(principal)
        require_market(body.market_id)
        command = WorkRequest(
            body.kind,
            body.market_id,
            tuple(body.symbols),
            body.start_at,
            body.end_at,
            body.timeframe,
        )
        try:
            command.validate(now=datetime.now(UTC))
            for symbol in command.symbols:
                _catalog_instrument_or_error(
                    search_use_case=search, market_id=command.market_id, symbol=symbol
                )
            row = jobs.submit(
                organization_id=org.value,
                actor_user_id=principal.user_id.value,
                key=body.idempotency_key,
                request=command,
                now=datetime.now(UTC),
            )
        except ActiveWorkRequestError as error:
            raise HTTPException(409, {"code": error.code}) from error
        except PermissionError as error:
            raise HTTPException(403, "Active organization membership is required") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return _response(row)

    @router.post("/market-data/work-batches")
    def submit_batch(body: WorkBatchBody, request: Request,
                     principal: CurrentUserPrincipal = Depends(current_user_dependency)):
        require_market_data_origin(request)
        org = organization(principal)
        require_market(body.market_id)
        now = datetime.now(UTC)
        try:
            if len(set(body.symbols)) != len(body.symbols):
                raise ValueError("select distinct instruments")
            commands = tuple(WorkRequest("candle_ingestion", body.market_id, (symbol,),
                                         body.start_at, body.end_at) for symbol in body.symbols)
            for command in commands:
                command.validate(now=now)
                _catalog_instrument_or_error(search_use_case=search, market_id=body.market_id,
                                              symbol=command.symbols[0])
            rows = jobs.submit_batch(
                organization_id=org.value, actor_user_id=principal.user_id.value,
                                     key=body.idempotency_key, requests=commands, now=now)
        except ActiveWorkRequestError as error:
            raise HTTPException(409, {"code": error.code}) from error
        except PermissionError as error:
            raise HTTPException(403, "Active organization membership is required") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return dict(items=[_response(row) for row in rows], batch_key=body.idempotency_key)

    @router.get("/market-data/work-requests")
    def history(
        limit: int = Query(default=30, ge=1, le=50),
        before: datetime | None = None,
        before_id: UUID | None = None,
        kind: Literal["catalog_refresh", "candle_ingestion"] | None = None,
        state: Literal["queued", "running", "pause_requested", "paused", "retry_wait",
                       "cancel_requested", "cancelled", "succeeded", "failed"]
        | None = None,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ):
        org = organization(principal)
        if (before is None) != (before_id is None) or (before and before.tzinfo is None):
            raise HTTPException(422, "Invalid history cursor")
        rows = jobs.page(
            organization_id=org.value,
            limit=limit + 1,
            before=before,
            before_id=before_id,
            kind=kind,
            state=state,
        )
        return dict(
            items=[_response(r) for r in rows[:limit]],
            next_cursor=(
                dict(before=rows[limit - 1]["created_at"], before_id=rows[limit - 1]["job_id"])
                if len(rows) > limit
                else None
            ),
        )

    @router.get("/market-data/work-requests/lookup", response_model=WorkRequestResponse)
    def lookup(key: UUID, principal: CurrentUserPrincipal = Depends(current_user_dependency)):
        row = jobs.lookup(organization_id=organization(principal).value, key=key)
        if row is None:
            raise HTTPException(404, "No saved request with this key")
        return _response(row)

    @router.get("/market-data/work-requests/{job_id}", response_model=WorkRequestResponse)
    def detail(job_id: UUID, principal: CurrentUserPrincipal = Depends(current_user_dependency)):
        row = jobs.get(organization_id=organization(principal).value, job_id=job_id)
        if row is None:
            raise HTTPException(404, "Request not found")
        return _response(row)

    @router.get("/market-data/work-requests/{job_id}/events")
    def events(job_id: UUID, before: int | None = Query(default=None, ge=1),
               limit: int = Query(default=30, ge=1, le=100),
               principal: CurrentUserPrincipal = Depends(current_user_dependency)):
        org = organization(principal)
        if jobs.get(organization_id=org.value, job_id=job_id) is None:
            raise HTTPException(404, "Request not found")
        rows = jobs.events(organization_id=org.value, job_id=job_id,
                           before=before, limit=limit + 1)
        return dict(items=[WorkEventResponse.model_validate(row) for row in rows[:limit]],
                    next_cursor=rows[limit - 1]["event_id"] if len(rows) > limit else None,
                    retention_days=30, max_events=512)

    @router.post("/market-data/work-requests/{job_id}/cancel", response_model=WorkRequestResponse)
    def cancel(
        job_id: UUID,
        request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ):
        require_market_data_origin(request)
        org = organization(principal)
        row = jobs.cancel(organization_id=org.value, job_id=job_id, now=datetime.now(UTC))
        if row is None:
            row = jobs.get(organization_id=org.value, job_id=job_id)
            if row is None:
                raise HTTPException(404, "Request not found")
            if row["state"] not in {"cancelled", "cancel_requested"}:
                raise HTTPException(409, {"code": "work_not_cancellable"})
        return _response(row)

    def control(job_id: UUID, body: ControlBody, request: Request,
                principal: CurrentUserPrincipal, action: str):
        require_market_data_origin(request)
        org = organization(principal)
        operation = jobs.pause if action == "pause" else jobs.resume
        row = operation(organization_id=org.value, job_id=job_id,
                        version=body.control_version, now=datetime.now(UTC))
        if row is None:
            row = jobs.get(organization_id=org.value, job_id=job_id)
            if row is None:
                raise HTTPException(404, "Request not found")
            # Retrying an acknowledged action is harmless; a delayed action from
            # an earlier pause/resume cycle cannot alter the current intent.
            allowed = {"paused", "pause_requested"} if action == "pause" else {
                "queued", "running", "retry_wait", "succeeded", "failed"}
            if row["control_version"] != body.control_version + 1 or row["state"] not in allowed:
                raise HTTPException(409, {"code": "stale_work_control"})
        return _response(row)

    @router.post("/market-data/work-requests/{job_id}/pause", response_model=WorkRequestResponse)
    def pause(job_id: UUID, body: ControlBody, request: Request,
              principal: CurrentUserPrincipal = Depends(current_user_dependency)):
        return control(job_id, body, request, principal, "pause")

    @router.post("/market-data/work-requests/{job_id}/resume", response_model=WorkRequestResponse)
    def resume(job_id: UUID, body: ControlBody, request: Request,
               principal: CurrentUserPrincipal = Depends(current_user_dependency)):
        return control(job_id, body, request, principal, "resume")

    @router.post("/market-data/work-requests/{job_id}/retry", response_model=WorkRequestResponse)
    def retry(
        job_id: UUID,
        body: RetryBody,
        request: Request,
        principal: CurrentUserPrincipal = Depends(current_user_dependency),
    ):
        require_market_data_origin(request)
        org = organization(principal)
        try:
            row = jobs.retry(
                organization_id=org.value,
                job_id=job_id,
                attempt=body.attempt,
                now=datetime.now(UTC),
            )
        except ActiveWorkRequestError as error:
            raise HTTPException(409, {"code": error.code}) from error
        if row is None:
            row = jobs.get(organization_id=org.value, job_id=job_id)
            if row is None:
                raise HTTPException(404, "Request not found")
            if row["attempt"] != body.attempt + 1:
                raise HTTPException(409, {"code": "work_not_retryable"})
        return _response(row)

    return router
