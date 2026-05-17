from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Mapping
from uuid import UUID

from trading.shared_kernel.primitives import PaidLevel, UserId

JsonMapping = Mapping[str, Any]

BacktestAiConfigMode = Literal["assistant_v1"]
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
    "collecting_context",
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
BacktestAiContextAxisMode = Literal["range", "explicit", "none"]
BacktestAiConversationLocale = Literal["ru", "en"]
BacktestAiConversationStatus = Literal["active", "archived"]
BacktestAiConversationTitleSource = Literal["fallback", "model"]
BacktestAiConversationMessageRole = Literal["system", "assistant", "user"]
BacktestAiConversationRunStatus = Literal[
    "awaiting_model",
    "ready",
    "needs_clarification",
    "blocked_by_policy",
    "failed",
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
class BacktestAiLoadAction:
    enabled: bool
    state: str
    reason: str | None = None
    config: JsonMapping | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", self.state.strip())
        if not self.state:
            raise ValueError("BacktestAiLoadAction.state must be non-empty")
        object.__setattr__(self, "config", _freeze_optional_mapping(self.config))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "reason": self.reason,
            "config": None if self.config is None else dict(self.config),
        }


@dataclass(frozen=True, slots=True)
class BacktestAiConversation:
    conversation_id: UUID
    owner_user_id: UserId
    locale: BacktestAiConversationLocale
    status: BacktestAiConversationStatus
    title: str
    title_source: BacktestAiConversationTitleSource
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime
    expires_at: datetime

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "conversation_id": str(self.conversation_id),
            "conversation_title": self.title,
            "locale": self.locale,
            "status": self.status,
            "created_at": _format_datetime(self.created_at),
            "updated_at": _format_datetime(self.updated_at),
            "last_message_at": _format_datetime(self.last_message_at),
            "expires_at": _format_datetime(self.expires_at),
        }


@dataclass(frozen=True, slots=True)
class BacktestAiConversationMessage:
    message_id: UUID
    conversation_id: UUID
    owner_user_id: UserId
    role: BacktestAiConversationMessageRole
    content: str
    created_at: datetime
    metadata_json: JsonMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content", self.content.strip())
        if not self.content:
            raise ValueError("BacktestAiConversationMessage.content must be non-empty")
        object.__setattr__(self, "metadata_json", MappingProxyType(dict(self.metadata_json)))

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "message_id": str(self.message_id),
            "conversation_id": str(self.conversation_id),
            "role": self.role,
            "content": self.content,
            "created_at": _format_datetime(self.created_at),
            "metadata": dict(self.metadata_json),
        }


@dataclass(frozen=True, slots=True)
class BacktestAiConversationRun:
    run_id: UUID
    conversation_id: UUID
    owner_user_id: UserId
    user_message_id: UUID
    assistant_message_id: UUID
    status: BacktestAiConversationRunStatus
    load_action: BacktestAiLoadAction
    created_at: datetime
    updated_at: datetime
    intent: str | None = None
    current_config_json: JsonMapping | None = None
    validated_config_json: JsonMapping | None = None
    model_id: str | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
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

    def public_snapshot(self) -> dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "conversation_id": str(self.conversation_id),
            "status": self.status,
            "intent": self.intent,
            "load_action": self.load_action.as_mapping(),
            "created_at": _format_datetime(self.created_at),
            "updated_at": _format_datetime(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class BacktestAiConversationModelResponse:
    assistant_message: str
    conversation_title: str | None = None
    status: BacktestAiConversationRunStatus = "awaiting_model"
    intent: str | None = None
    load_action: BacktestAiLoadAction = field(
        default_factory=lambda: BacktestAiLoadAction(
            enabled=False,
            state="unavailable",
            reason="backend_not_ready",
        )
    )
    validated_config_json: JsonMapping | None = None
    model_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "assistant_message", self.assistant_message.strip())
        if not self.assistant_message:
            raise ValueError("assistant_message must be non-empty")
        object.__setattr__(
            self,
            "validated_config_json",
            _freeze_optional_mapping(self.validated_config_json),
        )


@dataclass(frozen=True, slots=True)
class BacktestAiConversationSendResult:
    conversation: BacktestAiConversation
    user_message: BacktestAiConversationMessage
    assistant_message: BacktestAiConversationMessage
    run: BacktestAiConversationRun


@dataclass(frozen=True, slots=True)
class BacktestAiConversationRead:
    conversation: BacktestAiConversation
    messages: tuple[BacktestAiConversationMessage, ...]
    latest_run: BacktestAiConversationRun | None


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
class BacktestAiContextAxis:
    name: str
    mode: BacktestAiContextAxisMode
    values: tuple[int | float | str, ...] = ()
    start: int | float | None = None
    stop_incl: int | float | None = None
    step: int | float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", self.name.strip().lower())
        if not self.name:
            raise ValueError("BacktestAiContextAxis.name must be non-empty")
        if self.mode == "explicit" and len(self.values) == 0:
            raise ValueError("explicit context axis requires values")
        if self.mode == "range" and (
            self.start is None or self.stop_incl is None or self.step is None
        ):
            raise ValueError("range context axis requires start/stop_incl/step")
        if self.mode == "none" and (
            self.values
            or self.start is not None
            or self.stop_incl is not None
            or self.step is not None
        ):
            raise ValueError("none context axis must not carry values or range bounds")

    def as_mapping(self) -> dict[str, Any]:
        if self.mode == "explicit":
            return {"mode": "explicit", "values": list(self.values)}
        if self.mode == "range":
            return {
                "mode": "range",
                "start": self.start,
                "stop_incl": self.stop_incl,
                "step": self.step,
            }
        return {"mode": "none"}


@dataclass(frozen=True, slots=True)
class BacktestAiIndicatorAvailability:
    indicator_id: str
    available: bool
    reason: str
    sources: tuple[str, ...]
    axes: Mapping[str, BacktestAiContextAxis]
    signal_params: Mapping[str, BacktestAiContextAxis]
    coverage_timeframes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "indicator_id", self.indicator_id.strip().lower())
        if not self.indicator_id:
            raise ValueError("indicator_id must be non-empty")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(
            self,
            "axes",
            MappingProxyType(dict(sorted(self.axes.items()))),
        )
        object.__setattr__(
            self,
            "signal_params",
            MappingProxyType(dict(sorted(self.signal_params.items()))),
        )
        object.__setattr__(self, "coverage_timeframes", tuple(self.coverage_timeframes))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "available": self.available,
            "reason": self.reason,
            "sources": list(self.sources),
            "window_axis": self.axes.get(
                "window",
                BacktestAiContextAxis(name="window", mode="none"),
            ).as_mapping(),
            "axes": {name: axis.as_mapping() for name, axis in self.axes.items()},
            "signal_params": {
                name: axis.as_mapping() for name, axis in self.signal_params.items()
            },
            "coverage_timeframes": list(self.coverage_timeframes),
        }


@dataclass(frozen=True, slots=True)
class BacktestAiContextSnapshot:
    schema_version: int
    source: str
    snapshot_hash: str
    summary_hash: str
    summary_generated_at_utc: str
    resolved_symbol: str
    exchange: str
    market_type: str
    instrument_key: str
    ignored_symbols: tuple[str, ...]
    warnings: tuple[str, ...]
    allowed_values: Mapping[str, Any]
    period: Mapping[str, str]
    timeframe_periods: Mapping[str, Mapping[str, Any]]
    indicators: tuple[BacktestAiIndicatorAvailability, ...]
    indicator_audit: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolved_symbol", self.resolved_symbol.strip().upper())
        object.__setattr__(self, "exchange", self.exchange.strip().lower())
        object.__setattr__(self, "market_type", self.market_type.strip().lower())
        object.__setattr__(self, "ignored_symbols", tuple(self.ignored_symbols))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self,
            "allowed_values",
            MappingProxyType(dict(self.allowed_values)),
        )
        object.__setattr__(self, "period", MappingProxyType(dict(self.period)))
        object.__setattr__(
            self,
            "timeframe_periods",
            MappingProxyType(dict(self.timeframe_periods)),
        )
        object.__setattr__(
            self,
            "indicator_audit",
            MappingProxyType(dict(self.indicator_audit)),
        )
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "snapshot_hash": self.snapshot_hash,
            "summary_hash": self.summary_hash,
            "summary_generated_at_utc": self.summary_generated_at_utc,
            "resolved_symbol": self.resolved_symbol,
            "exchange": self.exchange,
            "market_type": self.market_type,
            "instrument_key": self.instrument_key,
            "ignored_symbols": list(self.ignored_symbols),
            "warnings": list(self.warnings),
            "allowed_values": _json_ready(self.allowed_values),
            "period": dict(self.period),
            "timeframe_periods": _json_ready(self.timeframe_periods),
            "indicators": [item.as_mapping() for item in self.indicators],
            "indicator_audit": _json_ready(self.indicator_audit),
            "provenance": _json_ready(self.provenance),
        }

    def model_prompt_context(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_hash": self.snapshot_hash,
            "allowed_values": _json_ready(self.allowed_values),
            "resolved_symbol": {
                "exchange": self.exchange,
                "market_type": self.market_type,
                "symbol": self.resolved_symbol,
            },
            "ignored_symbols": list(self.ignored_symbols),
            "warnings": list(self.warnings),
            "period": dict(self.period),
            "timeframe_periods": _json_ready(self.timeframe_periods),
            "indicators": [
                item.as_mapping() for item in self.indicators if item.available
            ],
        }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


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
    "BacktestAiContextAxis",
    "BacktestAiContextAxisMode",
    "BacktestAiContextSnapshot",
    "BacktestAiConversation",
    "BacktestAiConversationLocale",
    "BacktestAiConversationMessage",
    "BacktestAiConversationMessageRole",
    "BacktestAiConversationModelResponse",
    "BacktestAiConversationRead",
    "BacktestAiConversationRun",
    "BacktestAiConversationRunStatus",
    "BacktestAiConversationSendResult",
    "BacktestAiConversationStatus",
    "BacktestAiConversationTitleSource",
    "BacktestAiConfigEvent",
    "BacktestAiConfigEventName",
    "BacktestAiConfigJob",
    "BacktestAiConfigJobState",
    "BacktestAiConfigLlmAttempt",
    "BacktestAiConfigLlmAttemptKind",
    "BacktestAiConfigLocale",
    "BacktestAiConfigMode",
    "BacktestAiConfigTerminalState",
    "BacktestAiIndicatorAvailability",
    "BacktestAiLoadAction",
    "BacktestAiQuotaAction",
    "BacktestAiQuotaEvent",
    "BacktestAiQuotaSnapshot",
    "JsonMapping",
]
