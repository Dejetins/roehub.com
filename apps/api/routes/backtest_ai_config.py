from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from apps.api.dto import (
    MAX_AI_CONFIG_CURRENT_CONFIG_CHARS,
    BacktestAiConfigCreateRequest,
    BacktestAiConfigCreateResponse,
    BacktestAiConfigFeedbackRequest,
    BacktestAiConfigFeedbackResponse,
    BacktestAiConfigJobResponse,
    build_backtest_ai_config_create_response,
    build_backtest_ai_config_feedback_response,
    build_backtest_ai_config_job_response,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigEvent,
    BacktestAiConfigJobsUseCase,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.platform.errors import RoehubError

CurrentUserDependency = Callable[[Request], CurrentUserPrincipal]

_BLOCKED_SSE_KEYS = {
    "chain_of_thought",
    "reasoning",
    "raw_model_response",
    "user_prompt",
    "user_prompt_text",
}


def build_backtest_ai_config_router(
    *,
    current_user_dependency: CurrentUserDependency,
    jobs_use_case: BacktestAiConfigJobsUseCase | None = None,
) -> APIRouter:
    if current_user_dependency is None:  # type: ignore[truthy-bool]
        raise ValueError("build_backtest_ai_config_router requires current_user_dependency")

    router = APIRouter(tags=["backtest-ai-config"])

    def require_ai_config_user(request: Request) -> CurrentUserPrincipal:
        try:
            return current_user_dependency(request)
        except HTTPException as error:
            if error.status_code == 401:
                raise RoehubError(
                    code="auth.required",
                    message="Authentication is required",
                    details={},
                ) from error
            raise

    def require_jobs_use_case() -> BacktestAiConfigJobsUseCase:
        if jobs_use_case is None:
            raise RoehubError(
                code="backtest.ai_config.unavailable",
                message="AI configurator service is not configured",
                details={"reason": "storage_unavailable"},
            )
        return jobs_use_case

    @router.post(
        "/backtests/ai-config/jobs",
        response_model=BacktestAiConfigCreateResponse,
        status_code=201,
    )
    def post_backtest_ai_config_job(
        payload: BacktestAiConfigCreateRequest,
        response: Response,
        idempotency_key_header: str | None = Header(
            default=None,
            alias="Idempotency-Key",
        ),
        principal: CurrentUserPrincipal = Depends(require_ai_config_user),
        use_case: BacktestAiConfigJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestAiConfigCreateResponse | JSONResponse:
        _validate_current_config_size(current_config=payload.current_config)
        idempotency_key = _resolve_idempotency_key(
            body_value=payload.idempotency_key,
            header_value=idempotency_key_header,
        )
        result = use_case.create(
            user_id=principal.user_id,
            paid_level=principal.paid_level,
            mode=payload.mode,
            locale=payload.locale,
            user_prompt_text=payload.message,
            idempotency_key=idempotency_key,
            current_config=payload.current_config,
            ui_context=payload.ui_context,
        )
        body = build_backtest_ai_config_create_response(result=result)
        if result.job is None:
            headers = {}
            if body.retry_after_seconds is not None:
                headers["Retry-After"] = str(body.retry_after_seconds)
            return JSONResponse(
                status_code=429,
                content=body.model_dump(mode="json"),
                headers=headers,
            )
        if result.idempotent_replay:
            response.status_code = 200
        return body

    @router.get(
        "/backtests/ai-config/jobs/{job_id}",
        response_model=BacktestAiConfigJobResponse,
    )
    def get_backtest_ai_config_job(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_ai_config_user),
        use_case: BacktestAiConfigJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestAiConfigJobResponse:
        job = use_case.get_owned(user_id=principal.user_id, job_id=job_id)
        return build_backtest_ai_config_job_response(job=job)

    @router.get("/backtests/ai-config/jobs/{job_id}/events")
    def get_backtest_ai_config_events(
        job_id: UUID,
        principal: CurrentUserPrincipal = Depends(require_ai_config_user),
        use_case: BacktestAiConfigJobsUseCase = Depends(require_jobs_use_case),
    ) -> StreamingResponse:
        events = use_case.list_events(user_id=principal.user_id, job_id=job_id)
        return StreamingResponse(
            _sse_lines(events=events),
            media_type="text/event-stream",
        )

    @router.post(
        "/backtests/ai-config/jobs/{job_id}/feedback",
        response_model=BacktestAiConfigFeedbackResponse,
    )
    def post_backtest_ai_config_feedback(
        job_id: UUID,
        payload: BacktestAiConfigFeedbackRequest,
        principal: CurrentUserPrincipal = Depends(require_ai_config_user),
        use_case: BacktestAiConfigJobsUseCase = Depends(require_jobs_use_case),
    ) -> BacktestAiConfigFeedbackResponse:
        feedback = payload.model_dump(exclude={"applied"}, exclude_none=True)
        job = use_case.record_feedback(
            user_id=principal.user_id,
            job_id=job_id,
            applied=payload.applied,
            feedback=feedback,
        )
        return build_backtest_ai_config_feedback_response(
            job=job,
            applied=payload.applied,
        )

    return router


def _validate_current_config_size(*, current_config: dict[str, Any] | None) -> None:
    if current_config is None:
        return
    encoded = json.dumps(
        current_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if len(encoded) > MAX_AI_CONFIG_CURRENT_CONFIG_CHARS:
        raise RoehubError(
            code="backtest.ai_config.invalid_request",
            message="AI configurator current_config is too large",
            details={
                "path": "current_config",
                "max_chars": MAX_AI_CONFIG_CURRENT_CONFIG_CHARS,
                "actual_chars": len(encoded),
            },
        )


def _resolve_idempotency_key(
    *,
    body_value: str | None,
    header_value: str | None,
) -> str | None:
    normalized_body = _optional_text(body_value)
    normalized_header = _optional_text(header_value)
    if (
        normalized_body is not None
        and normalized_header is not None
        and normalized_body != normalized_header
    ):
        raise RoehubError(
            code="backtest.ai_config.invalid_request",
            message="AI configurator idempotency key values conflict",
            details={"path": "idempotency_key"},
        )
    return normalized_body or normalized_header


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _sse_lines(*, events: tuple[BacktestAiConfigEvent, ...]) -> Iterator[str]:
    for event in events:
        yield _format_sse_event(event=event)


def _format_sse_event(*, event: BacktestAiConfigEvent) -> str:
    payload = {
        str(key): value
        for key, value in dict(event.payload_json).items()
        if str(key) not in _BLOCKED_SSE_KEYS
    }
    payload.setdefault("job_id", str(event.job_id))
    payload.setdefault("status", event.event_name)
    payload.setdefault("message", event.message)
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.event_name}\ndata: {data}\n\n"


__all__ = ["build_backtest_ai_config_router"]
