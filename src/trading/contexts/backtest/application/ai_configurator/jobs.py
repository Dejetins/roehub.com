from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping, cast
from uuid import UUID, uuid4

from trading.contexts.backtest.application.ports.backtest_ai_configurator import (
    BacktestAiConfigJobRepository,
)
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import PaidLevel, UserId

from .dto import (
    BacktestAiAdmissionDecision,
    BacktestAiConfigCreateResult,
    BacktestAiConfigEvent,
    BacktestAiConfigJob,
    BacktestAiConfigLocale,
    BacktestAiConfigMode,
    BacktestAiQuotaEvent,
    BacktestAiQuotaSnapshot,
)
from .quota import BacktestAiQuotaService

BACKTEST_AI_CONFIG_ERROR_IDEMPOTENCY_CONFLICT = (
    "backtest.ai_config.idempotency_key_conflict"
)
BACKTEST_AI_CONFIG_ERROR_INVALID_REQUEST = "backtest.ai_config.invalid_request"
BACKTEST_AI_CONFIG_ERROR_NOT_FOUND = "backtest.ai_config.not_found"
BACKTEST_AI_CONFIG_ERROR_FORBIDDEN = "backtest.ai_config.forbidden"

BACKTEST_AI_CONFIG_SOURCE_PAGE = "backtests"
PENDING_CATALOG_SNAPSHOT_HASH = hashlib.sha256(
    b"backtest-ai-configurator-pending-catalog-v1"
).hexdigest()
PENDING_RUNTIME_DEFAULTS_HASH = hashlib.sha256(
    b"backtest-ai-configurator-pending-runtime-defaults-v1"
).hexdigest()
BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION = (
    "backtest-ai-configurator-tool-agent-pending-v1"
)
BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH = hashlib.sha256(
    BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION.encode("utf-8")
).hexdigest()

_VALID_MODES = {"create", "edit", "explain", "repair", "suggest_safer"}
_VALID_LOCALES = {"ru", "en"}
_MAX_IDEMPOTENCY_KEY_LENGTH = 128
_MAX_USER_PROMPT_CHARS = 16_000
_MAX_FEEDBACK_MESSAGE_CHARS = 4_000


@dataclass(frozen=True, slots=True)
class BacktestAiConfigJobsUseCase:
    repository: BacktestAiConfigJobRepository
    quota_service: BacktestAiQuotaService = BacktestAiQuotaService()

    def create(
        self,
        *,
        user_id: UserId,
        paid_level: PaidLevel,
        mode: str,
        locale: str,
        user_prompt_text: str,
        idempotency_key: str | None = None,
        current_config: Mapping[str, Any] | None = None,
        ui_context: Mapping[str, Any] | None = None,
        catalog_snapshot_hash: str = PENDING_CATALOG_SNAPSHOT_HASH,
        system_prompt_version: str = BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION,
    ) -> BacktestAiConfigCreateResult:
        normalized_mode = _normalize_mode(mode=mode)
        normalized_locale = _normalize_locale(locale=locale)
        normalized_prompt = _normalize_user_prompt(user_prompt_text=user_prompt_text)
        normalized_idempotency_key = _normalize_idempotency_key(
            idempotency_key=idempotency_key
        )
        current_config_hash = _optional_canonical_hash(current_config)
        prompt_hash = _sha256_text(normalized_prompt)
        runtime_defaults_hash = _runtime_defaults_hash(ui_context=ui_context)
        now = datetime.now(UTC)

        if normalized_idempotency_key is not None:
            existing = self.repository.find_by_idempotency_key(
                owner_user_id=user_id,
                idempotency_key=normalized_idempotency_key,
            )
            if existing is not None:
                if not _same_logical_request(
                    existing=existing,
                    mode=normalized_mode,
                    locale=normalized_locale,
                    user_prompt_hash=prompt_hash,
                    current_config_hash=current_config_hash,
                ):
                    raise RoehubError(
                        code=BACKTEST_AI_CONFIG_ERROR_IDEMPOTENCY_CONFLICT,
                        message=(
                            "AI configurator idempotency_key was already used with "
                            "a different request"
                        ),
                        details={"idempotency_key": normalized_idempotency_key},
                    )
                return BacktestAiConfigCreateResult(
                    job=existing,
                    idempotent_replay=True,
                    quota_charged=False,
                    admission=BacktestAiAdmissionDecision(
                        accepted=True,
                        status="accepted",
                        reason="idempotent_replay",
                        message="AI configurator request was replayed.",
                        estimated_wait_seconds=0,
                    ),
                )

        snapshot = BacktestAiQuotaSnapshot(
            requests_5h=self.repository.count_quota_events(
                owner_user_id=user_id,
                occurred_after=now - timedelta(hours=5),
            ),
            requests_week=self.repository.count_quota_events(
                owner_user_id=user_id,
                occurred_after=now - timedelta(days=7),
            ),
            queued_jobs_for_user=self.repository.count_queued_for_user(
                owner_user_id=user_id
            ),
            active_jobs_for_user=self.repository.count_active_for_user(
                owner_user_id=user_id
            ),
            active_jobs_global=self.repository.count_active_global(),
        )
        admission = self.quota_service.evaluate(
            paid_level=paid_level,
            snapshot=snapshot,
        )
        if not admission.accepted:
            self.repository.record_quota_event(
                event=BacktestAiQuotaEvent(
                    quota_event_id=uuid4(),
                    owner_user_id=user_id,
                    paid_level=paid_level,
                    quota_action="quota_rejected"
                    if admission.status == "quota_exceeded"
                    else "capacity_rejected",
                    occurred_at=now,
                    idempotency_key=normalized_idempotency_key,
                    units=0,
                    reason=admission.reason,
                    metadata_json=admission.as_mapping(),
                )
            )
            return BacktestAiConfigCreateResult(
                job=None,
                idempotent_replay=False,
                quota_charged=False,
                admission=admission,
            )

        job_id = uuid4()
        job = BacktestAiConfigJob(
            job_id=job_id,
            owner_user_id=user_id,
            mode=normalized_mode,
            locale=normalized_locale,
            state="queued",
            source_page=BACKTEST_AI_CONFIG_SOURCE_PAGE,
            user_prompt_text=normalized_prompt,
            user_prompt_hash=prompt_hash,
            idempotency_key=normalized_idempotency_key,
            current_config_hash=current_config_hash,
            current_config_json=current_config,
            system_prompt_version=system_prompt_version,
            system_prompt_hash=BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH,
            catalog_snapshot_hash=catalog_snapshot_hash,
            runtime_defaults_hash=runtime_defaults_hash,
            queued_at=now,
            updated_at=now,
            quota_charged=True,
        )
        event = BacktestAiConfigEvent(
            event_id=uuid4(),
            job_id=job_id,
            owner_user_id=user_id,
            event_name="queued",
            message="AI configurator request was queued.",
            payload_json={
                "job_id": str(job_id),
                "status": "queued",
                "mode": normalized_mode,
                "locale": normalized_locale,
            },
            created_at=now,
        )
        quota_event = BacktestAiQuotaEvent(
            quota_event_id=uuid4(),
            owner_user_id=user_id,
            paid_level=paid_level,
            quota_action="request_charged",
            occurred_at=now,
            job_id=job_id,
            idempotency_key=normalized_idempotency_key,
            units=1,
            reason="accepted",
            metadata_json={
                "mode": normalized_mode,
                "locale": normalized_locale,
                "quota_charged": True,
            },
        )
        stored = self.repository.create_with_quota_event(
            job=job,
            event=event,
            quota_event=quota_event,
        )
        return BacktestAiConfigCreateResult(
            job=stored,
            idempotent_replay=False,
            quota_charged=True,
            admission=admission,
        )

    def get(self, *, user_id: UserId, job_id: UUID) -> BacktestAiConfigJob:
        job = self.repository.get(job_id=job_id, owner_user_id=user_id)
        if job is None:
            raise RoehubError(
                code=BACKTEST_AI_CONFIG_ERROR_NOT_FOUND,
                message="AI configurator job was not found",
                details={"job_id": str(job_id)},
            )
        return job

    def get_owned(self, *, user_id: UserId, job_id: UUID) -> BacktestAiConfigJob:
        job = self._get_by_id_or_not_found(job_id=job_id)
        if job.owner_user_id != user_id:
            raise RoehubError(
                code=BACKTEST_AI_CONFIG_ERROR_FORBIDDEN,
                message="AI configurator job belongs to another owner",
                details={"job_id": str(job_id)},
            )
        return job

    def list_events(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
    ) -> tuple[BacktestAiConfigEvent, ...]:
        self.get_owned(user_id=user_id, job_id=job_id)
        return self.repository.list_events(job_id=job_id, owner_user_id=user_id)

    def record_feedback(
        self,
        *,
        user_id: UserId,
        job_id: UUID,
        applied: bool,
        feedback: Mapping[str, Any] | None = None,
    ) -> BacktestAiConfigJob:
        self.get_owned(user_id=user_id, job_id=job_id)
        now = datetime.now(UTC)
        feedback_json = _feedback_payload(applied=applied, feedback=feedback, now=now)
        updated = self.repository.record_feedback(
            job_id=job_id,
            owner_user_id=user_id,
            applied=applied,
            feedback_json=feedback_json,
            now=now,
        )
        if updated is None:
            raise RoehubError(
                code=BACKTEST_AI_CONFIG_ERROR_NOT_FOUND,
                message="AI configurator job was not found",
                details={"job_id": str(job_id)},
            )
        return updated

    def _get_by_id_or_not_found(self, *, job_id: UUID) -> BacktestAiConfigJob:
        job = self.repository.get(job_id=job_id, owner_user_id=None)
        if job is None:
            raise RoehubError(
                code=BACKTEST_AI_CONFIG_ERROR_NOT_FOUND,
                message="AI configurator job was not found",
                details={"job_id": str(job_id)},
            )
        return job


def _normalize_mode(*, mode: str) -> BacktestAiConfigMode:
    normalized = mode.strip().lower()
    if normalized not in _VALID_MODES:
        raise RoehubError(
            code=BACKTEST_AI_CONFIG_ERROR_INVALID_REQUEST,
            message="Unsupported AI configurator mode",
            details={"mode": mode},
        )
    return cast(BacktestAiConfigMode, normalized)


def _normalize_locale(*, locale: str) -> BacktestAiConfigLocale:
    normalized = locale.strip().lower()
    if normalized not in _VALID_LOCALES:
        raise RoehubError(
            code=BACKTEST_AI_CONFIG_ERROR_INVALID_REQUEST,
            message="Unsupported AI configurator locale",
            details={"locale": locale},
        )
    return cast(BacktestAiConfigLocale, normalized)


def _normalize_user_prompt(*, user_prompt_text: str) -> str:
    normalized = user_prompt_text.strip()
    if not normalized:
        raise RoehubError(
            code=BACKTEST_AI_CONFIG_ERROR_INVALID_REQUEST,
            message="AI configurator prompt must be non-empty",
            details={"path": "message"},
        )
    if len(normalized) > _MAX_USER_PROMPT_CHARS:
        raise RoehubError(
            code=BACKTEST_AI_CONFIG_ERROR_INVALID_REQUEST,
            message="AI configurator prompt is too large",
            details={
                "path": "message",
                "max_chars": _MAX_USER_PROMPT_CHARS,
                "actual_chars": len(normalized),
            },
        )
    return normalized


def _normalize_idempotency_key(*, idempotency_key: str | None) -> str | None:
    if idempotency_key is None:
        return None
    normalized = idempotency_key.strip()
    if not normalized:
        return None
    if len(normalized) > _MAX_IDEMPOTENCY_KEY_LENGTH:
        raise RoehubError(
            code=BACKTEST_AI_CONFIG_ERROR_INVALID_REQUEST,
            message="AI configurator idempotency_key is too long",
            details={
                "path": "idempotency_key",
                "max_chars": _MAX_IDEMPOTENCY_KEY_LENGTH,
            },
        )
    return normalized


def _feedback_payload(
    *,
    applied: bool,
    feedback: Mapping[str, Any] | None,
    now: datetime,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "applied": applied,
        "recorded_at": now.isoformat(),
    }
    if feedback is None:
        return payload
    raw_message = feedback.get("message")
    if raw_message is not None:
        message = str(raw_message).strip()
        if len(message) > _MAX_FEEDBACK_MESSAGE_CHARS:
            raise RoehubError(
                code=BACKTEST_AI_CONFIG_ERROR_INVALID_REQUEST,
                message="AI configurator feedback message is too large",
                details={
                    "path": "message",
                    "max_chars": _MAX_FEEDBACK_MESSAGE_CHARS,
                    "actual_chars": len(message),
                },
            )
        if message:
            payload["message"] = message
    raw_reason = feedback.get("reason")
    if raw_reason is not None:
        reason = str(raw_reason).strip()
        if reason:
            payload["reason"] = reason[:128]
    raw_context = feedback.get("client_context")
    if isinstance(raw_context, Mapping):
        client_context: dict[str, object] = {}
        for key, value in raw_context.items():
            normalized_value = _feedback_scalar(value)
            if normalized_value is not None:
                client_context[str(key)] = normalized_value
        payload["client_context"] = client_context
    return payload


def _feedback_scalar(value: Any) -> object | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _runtime_defaults_hash(*, ui_context: Mapping[str, Any] | None) -> str:
    if ui_context is not None:
        raw_value = ui_context.get("runtime_defaults_hash")
        if isinstance(raw_value, str) and _is_sha256(raw_value.strip().lower()):
            return raw_value.strip().lower()
    return PENDING_RUNTIME_DEFAULTS_HASH


def _same_logical_request(
    *,
    existing: BacktestAiConfigJob,
    mode: BacktestAiConfigMode,
    locale: BacktestAiConfigLocale,
    user_prompt_hash: str,
    current_config_hash: str | None,
) -> bool:
    return (
        existing.mode == mode
        and existing.locale == locale
        and existing.user_prompt_hash == user_prompt_hash
        and existing.current_config_hash == current_config_hash
    )


def _optional_canonical_hash(payload: Mapping[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


__all__ = [
    "BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH",
    "BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION",
    "BACKTEST_AI_CONFIG_ERROR_FORBIDDEN",
    "BACKTEST_AI_CONFIG_ERROR_IDEMPOTENCY_CONFLICT",
    "BACKTEST_AI_CONFIG_ERROR_INVALID_REQUEST",
    "BACKTEST_AI_CONFIG_ERROR_NOT_FOUND",
    "BACKTEST_AI_CONFIG_SOURCE_PAGE",
    "BacktestAiConfigJobsUseCase",
    "PENDING_CATALOG_SNAPSHOT_HASH",
    "PENDING_RUNTIME_DEFAULTS_HASH",
]
