from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiAdmissionDecision,
    BacktestAiConfigCreateResult,
    BacktestAiConfigJob,
)

MAX_AI_CONFIG_MESSAGE_CHARS = 16_000
MAX_AI_CONFIG_CURRENT_CONFIG_CHARS = 64_000


class BacktestAiConfigCreateRequest(BaseModel):
    mode: str = Field(min_length=1)
    locale: str = Field(min_length=2, max_length=2)
    message: str = Field(min_length=1, max_length=MAX_AI_CONFIG_MESSAGE_CHARS)
    idempotency_key: str | None = Field(default=None, max_length=128)
    current_config: dict[str, Any] | None = None
    ui_context: dict[str, Any] | None = None


class BacktestAiConfigCreateResponse(BaseModel):
    job_id: str | None
    status: str
    events_url: str | None = None
    estimated_wait_seconds: int | None = None
    retry_after_seconds: int | None = None
    message: str
    idempotent_replay: bool | None = None


class BacktestAiConfigJobResponse(BaseModel):
    job_id: str
    status: str
    mode: str
    locale: str
    assistant_message: str | None
    validated_config: dict[str, Any] | None
    load_action: dict[str, Any]
    warnings: list[dict[str, Any]]
    suggestions: list[dict[str, Any]]
    validation_errors: list[dict[str, Any]]
    quota_charged: bool
    queued_at: str
    started_at: str | None
    finished_at: str | None
    updated_at: str


class BacktestAiConfigFeedbackRequest(BaseModel):
    applied: bool
    message: str | None = Field(default=None, max_length=4_000)
    reason: str | None = Field(default=None, max_length=128)
    client_context: dict[str, Any] | None = None


class BacktestAiConfigFeedbackResponse(BaseModel):
    job_id: str
    status: str
    feedback_recorded: bool
    applied: bool


def build_backtest_ai_config_create_response(
    *,
    result: BacktestAiConfigCreateResult,
) -> BacktestAiConfigCreateResponse:
    if result.job is None:
        return _admission_response(admission=result.admission)
    job_id = str(result.job.job_id)
    return BacktestAiConfigCreateResponse(
        job_id=job_id,
        status=result.job.state,
        events_url=None,
        estimated_wait_seconds=result.admission.estimated_wait_seconds,
        retry_after_seconds=result.admission.retry_after_seconds,
        message=result.admission.message,
        idempotent_replay=result.idempotent_replay,
    )


def build_backtest_ai_config_job_response(
    *,
    job: BacktestAiConfigJob,
) -> BacktestAiConfigJobResponse:
    snapshot = job.public_snapshot()
    validated_config = snapshot["validated_config"]
    load_action: dict[str, Any] = {"enabled": False}
    if job.state == "ready" and isinstance(validated_config, dict):
        load_action = {
            "enabled": True,
            "label": "Загрузить конфигурацию",
        }
    return BacktestAiConfigJobResponse(
        job_id=str(snapshot["job_id"]),
        status=str(snapshot["status"]),
        mode=str(snapshot["mode"]),
        locale=str(snapshot["locale"]),
        assistant_message=_optional_str(snapshot["assistant_message"]),
        validated_config=validated_config if isinstance(validated_config, dict) else None,
        load_action=load_action,
        warnings=_warning_list(snapshot["suggestions"]),
        suggestions=_dict_list(snapshot["suggestions"]),
        validation_errors=_dict_list(snapshot["validation_errors"]),
        quota_charged=bool(snapshot["quota_charged"]),
        queued_at=str(snapshot["queued_at"]),
        started_at=_optional_str(snapshot["started_at"]),
        finished_at=_optional_str(snapshot["finished_at"]),
        updated_at=str(snapshot["updated_at"]),
    )


def build_backtest_ai_config_feedback_response(
    *,
    job: BacktestAiConfigJob,
    applied: bool,
) -> BacktestAiConfigFeedbackResponse:
    return BacktestAiConfigFeedbackResponse(
        job_id=str(job.job_id),
        status=job.state,
        feedback_recorded=job.user_feedback_json is not None,
        applied=applied,
    )


def _admission_response(
    *,
    admission: BacktestAiAdmissionDecision,
) -> BacktestAiConfigCreateResponse:
    return BacktestAiConfigCreateResponse(
        job_id=None,
        status=admission.status,
        estimated_wait_seconds=admission.estimated_wait_seconds,
        retry_after_seconds=admission.retry_after_seconds,
        message=admission.message,
        idempotent_replay=False,
    )


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _warning_list(value: Any) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in _dict_list(value)
        if str(item.get("kind") or "").casefold() == "warning"
    ]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "BacktestAiConfigCreateRequest",
    "BacktestAiConfigCreateResponse",
    "BacktestAiConfigFeedbackRequest",
    "BacktestAiConfigFeedbackResponse",
    "BacktestAiConfigJobResponse",
    "MAX_AI_CONFIG_CURRENT_CONFIG_CHARS",
    "MAX_AI_CONFIG_MESSAGE_CHARS",
    "build_backtest_ai_config_create_response",
    "build_backtest_ai_config_feedback_response",
    "build_backtest_ai_config_job_response",
]
