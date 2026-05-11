from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Mapping
from uuid import UUID

from trading.shared_kernel.primitives import PaidLevel, UserId

JsonMapping = Mapping[str, Any]

BacktestAiConfigMode = Literal[
    "create",
    "edit",
    "explain",
    "repair",
    "suggest_safer",
]
BacktestAiConfigLocale = Literal["ru", "en"]
BacktestAiConfigJobState = Literal[
    "queued",
    "running",
    "repairing",
    "ready",
    "needs_clarification",
    "blocked_by_policy",
    "input_too_large",
    "security_review",
    "failed",
    "cancelled",
]
BacktestAiConfigTerminalState = Literal[
    "ready",
    "needs_clarification",
    "blocked_by_policy",
    "input_too_large",
    "security_review",
    "failed",
    "cancelled",
]
BacktestAiConfigEventName = Literal[
    "queued",
    "preparing_catalog",
    "assembling_prompt",
    "generating",
    "validating_json",
    "validating_business",
    "repairing",
    "ready",
    "needs_clarification",
    "blocked_by_policy",
    "input_too_large",
    "security_review",
    "failed",
    "heartbeat",
]
BacktestAiConfigLlmAttemptKind = Literal["generate", "repair"]
BacktestAiQuotaAction = Literal[
    "request_charged",
    "quota_rejected",
    "capacity_rejected",
]
BacktestAiAdmissionStatus = Literal[
    "accepted",
    "quota_exceeded",
    "capacity_delayed",
]


@dataclass(frozen=True, slots=True)
class BacktestAiConfigJob:
    job_id: UUID
    owner_user_id: UserId
    mode: BacktestAiConfigMode
    locale: BacktestAiConfigLocale
    state: BacktestAiConfigJobState
    source_page: str
    user_prompt_text: str
    user_prompt_hash: str
    system_prompt_version: str
    system_prompt_hash: str
    catalog_snapshot_hash: str
    runtime_defaults_hash: str
    queued_at: datetime
    updated_at: datetime
    idempotency_key: str | None = None
    current_config_hash: str | None = None
    current_config_json: JsonMapping | None = None
    validated_config_json: JsonMapping | None = None
    assistant_message: str | None = None
    suggestions_json: tuple[JsonMapping, ...] = field(default_factory=tuple)
    validation_errors_json: tuple[JsonMapping, ...] = field(default_factory=tuple)
    model_id: str | None = None
    model_path_hash: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    locked_by: str | None = None
    locked_at: datetime | None = None
    lease_expires_at: datetime | None = None
    heartbeat_at: datetime | None = None
    attempt: int = 0
    quota_charged: bool = False
    applied_at: datetime | None = None
    user_feedback_json: JsonMapping | None = None
    last_error: str | None = None
    last_error_json: JsonMapping | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_page", self.source_page.strip())
        object.__setattr__(
            self,
            "current_config_json",
            _freeze_optional_mapping(self.current_config_json),
        )
        object.__setattr__(
            self,
            "validated_config_json",
            _freeze_optional_mapping(self.validated_config_json),
        )
        object.__setattr__(
            self,
            "suggestions_json",
            tuple(MappingProxyType(dict(item)) for item in self.suggestions_json),
        )
        object.__setattr__(
            self,
            "validation_errors_json",
            tuple(MappingProxyType(dict(item)) for item in self.validation_errors_json),
        )
        object.__setattr__(
            self,
            "user_feedback_json",
            _freeze_optional_mapping(self.user_feedback_json),
        )
        object.__setattr__(
            self,
            "last_error_json",
            _freeze_optional_mapping(self.last_error_json),
        )

    def public_snapshot(self) -> dict[str, Any]:
        """
        Return the read-safe job snapshot without raw user prompt or audit-only payloads.
        """
        return {
            "job_id": str(self.job_id),
            "status": self.state,
            "mode": self.mode,
            "locale": self.locale,
            "assistant_message": self.assistant_message,
            "validated_config": None
            if self.validated_config_json is None
            else dict(self.validated_config_json),
            "suggestions": [dict(item) for item in self.suggestions_json],
            "validation_errors": [dict(item) for item in self.validation_errors_json],
            "quota_charged": self.quota_charged,
            "queued_at": _format_datetime(self.queued_at),
            "started_at": _format_datetime(self.started_at),
            "finished_at": _format_datetime(self.finished_at),
            "updated_at": _format_datetime(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class BacktestAiConfigEvent:
    event_id: UUID
    job_id: UUID
    owner_user_id: UserId
    event_name: BacktestAiConfigEventName
    message: str
    payload_json: JsonMapping
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload_json", MappingProxyType(dict(self.payload_json)))


@dataclass(frozen=True, slots=True)
class BacktestAiConfigLlmAttempt:
    attempt_id: UUID
    job_id: UUID
    owner_user_id: UserId
    attempt_no: int
    attempt_kind: BacktestAiConfigLlmAttemptKind
    prompt_profile: str
    system_prompt_version: str
    system_prompt_hash: str
    user_prompt_text: str
    catalog_subset_json: JsonMapping
    raw_model_response: str | None
    parsed_json_draft: JsonMapping | None
    validation_errors_json: tuple[JsonMapping, ...]
    input_tokens_estimate: int | None
    output_tokens_estimate: int | None
    latency_ms: int | None
    finish_reason: str | None
    success: bool
    failure_reason: str | None
    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "catalog_subset_json",
            MappingProxyType(dict(self.catalog_subset_json)),
        )
        object.__setattr__(
            self,
            "parsed_json_draft",
            _freeze_optional_mapping(self.parsed_json_draft),
        )
        object.__setattr__(
            self,
            "validation_errors_json",
            tuple(MappingProxyType(dict(item)) for item in self.validation_errors_json),
        )


@dataclass(frozen=True, slots=True)
class BacktestAiQuotaEvent:
    quota_event_id: UUID
    owner_user_id: UserId
    paid_level: PaidLevel
    quota_action: BacktestAiQuotaAction
    occurred_at: datetime
    job_id: UUID | None = None
    idempotency_key: str | None = None
    units: int = 1
    reason: str | None = None
    metadata_json: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata_json", MappingProxyType(dict(self.metadata_json)))


@dataclass(frozen=True, slots=True)
class BacktestAiQuotaSnapshot:
    requests_5h: int
    requests_week: int
    queued_jobs_for_user: int
    active_jobs_for_user: int
    active_jobs_global: int


@dataclass(frozen=True, slots=True)
class BacktestAiAdmissionDecision:
    accepted: bool
    status: BacktestAiAdmissionStatus
    reason: str
    message: str
    retry_after_seconds: int | None = None
    estimated_wait_seconds: int | None = None
    details: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
            "retry_after_seconds": self.retry_after_seconds,
            "estimated_wait_seconds": self.estimated_wait_seconds,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class BacktestAiConfigCreateResult:
    job: BacktestAiConfigJob | None
    idempotent_replay: bool
    quota_charged: bool
    admission: BacktestAiAdmissionDecision


def _freeze_optional_mapping(value: JsonMapping | None) -> JsonMapping | None:
    if value is None:
        return None
    return MappingProxyType(dict(value))


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


__all__ = [
    "BacktestAiAdmissionDecision",
    "BacktestAiAdmissionStatus",
    "BacktestAiConfigCreateResult",
    "BacktestAiConfigEvent",
    "BacktestAiConfigEventName",
    "BacktestAiConfigJob",
    "BacktestAiConfigJobState",
    "BacktestAiConfigLlmAttempt",
    "BacktestAiConfigLlmAttemptKind",
    "BacktestAiConfigLocale",
    "BacktestAiConfigMode",
    "BacktestAiConfigTerminalState",
    "BacktestAiQuotaAction",
    "BacktestAiQuotaEvent",
    "BacktestAiQuotaSnapshot",
    "JsonMapping",
]
