from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from .dto import BacktestAiConfigJob, BacktestAiConfigLlmAttempt

BacktestAiTrainingQualityLabel = Literal[
    "applied",
    "repaired",
    "clarification",
    "blocked",
    "attack_attempt",
    "failed_validation",
]

_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(sk|pk)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(password|passwd|secret|token|api[_-]?key|dsn)\s*[:=]\s*\S+", re.I),
)
_PRIVATE_TOPOLOGY_PATTERNS = (
    re.compile(r"/Users/[^\s,;\"']+"),
    re.compile(r"/opt/roehub/[^\s,;\"']+"),
    re.compile(r"\b(?:127\.0\.0\.1|localhost)\b", re.I),
    re.compile(r"\btailscale\b[^\s,;\"']*", re.I),
    re.compile(r"\bmlx_lm\.server\b", re.I),
)
_TRACEBACK_PATTERN = re.compile(r"Traceback \(most recent call last\):.*", re.S)
_REDACTION = "[REDACTED]"


@dataclass(frozen=True, slots=True)
class BacktestAiTrainingExportRecord:
    job: BacktestAiConfigJob
    attempts: tuple[BacktestAiConfigLlmAttempt, ...]


class BacktestAiTrainingExportSource(Protocol):
    def list_training_export_records(
        self,
        *,
        limit: int | None = None,
    ) -> tuple[BacktestAiTrainingExportRecord, ...]:
        ...


@dataclass(frozen=True, slots=True)
class BacktestAiTrainingExportUseCase:
    source: BacktestAiTrainingExportSource

    def export_rows(self, *, limit: int | None = None) -> tuple[dict[str, Any], ...]:
        records = self.source.list_training_export_records(limit=limit)
        return tuple(_row_from_record(record=record) for record in records)

    def export_jsonl(self, *, limit: int | None = None) -> str:
        rows = self.export_rows(limit=limit)
        return "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )


def _row_from_record(*, record: BacktestAiTrainingExportRecord) -> dict[str, Any]:
    job = record.job
    label = _quality_label(job=job, attempts=record.attempts)
    return {
        "schema_version": 1,
        "job_id": str(job.job_id),
        "owner_user_id": str(job.owner_user_id),
        "quality_label": label,
        "status": job.state,
        "mode": job.mode,
        "locale": job.locale,
        "source_page": job.source_page,
        "user_prompt": _redact_text(job.user_prompt_text),
        "current_config": _redact_json(job.current_config_json),
        "validated_config": _redact_json(job.validated_config_json),
        "assistant_message": _redact_text(job.assistant_message),
        "suggestions": _redact_json([dict(item) for item in job.suggestions_json]),
        "validation_errors": _safe_validation_errors(job.validation_errors_json),
        "applied": job.applied_at is not None,
        "model_id": _redact_text(job.model_id),
        "model_path_hash": job.model_path_hash,
        "system_prompt_version": job.system_prompt_version,
        "system_prompt_hash": job.system_prompt_hash,
        "catalog_snapshot_hash": job.catalog_snapshot_hash,
        "runtime_defaults_hash": job.runtime_defaults_hash,
        "attempts": [_safe_attempt(attempt=attempt) for attempt in record.attempts],
    }


def _quality_label(
    *,
    job: BacktestAiConfigJob,
    attempts: tuple[BacktestAiConfigLlmAttempt, ...],
) -> BacktestAiTrainingQualityLabel:
    if job.last_error_json is not None:
        flags = job.last_error_json.get("security_flags")
        if isinstance(flags, list) and flags:
            return "attack_attempt"
    if job.state in {"blocked_by_policy", "security_review"}:
        return "blocked"
    if job.state == "needs_clarification":
        return "clarification"
    if job.applied_at is not None:
        return "applied"
    if any(attempt.attempt_kind == "repair" and attempt.success for attempt in attempts):
        return "repaired"
    return "failed_validation"


def _safe_attempt(*, attempt: BacktestAiConfigLlmAttempt) -> dict[str, Any]:
    return {
        "attempt_no": attempt.attempt_no,
        "attempt_kind": attempt.attempt_kind,
        "prompt_profile": attempt.prompt_profile,
        "system_prompt_version": attempt.system_prompt_version,
        "system_prompt_hash": attempt.system_prompt_hash,
        "catalog_subset": _redact_json(attempt.catalog_subset_json),
        "parsed_json_draft": _redact_json(attempt.parsed_json_draft),
        "validation_errors": _safe_validation_errors(attempt.validation_errors_json),
        "input_tokens_estimate": attempt.input_tokens_estimate,
        "output_tokens_estimate": attempt.output_tokens_estimate,
        "latency_ms": attempt.latency_ms,
        "finish_reason": _redact_text(attempt.finish_reason),
        "success": attempt.success,
        "failure_reason": _redact_text(attempt.failure_reason),
    }


def _safe_validation_errors(errors: tuple[Mapping[str, Any], ...]) -> list[dict[str, Any]]:
    safe_errors: list[dict[str, Any]] = []
    for error in errors:
        safe_errors.append(
            {
                "path": _redact_text(_string_or_none(error.get("path"))),
                "code": _redact_text(_string_or_none(error.get("code"))),
                "message": _redact_text(_string_or_none(error.get("message"))),
            }
        )
    return safe_errors


def _redact_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_forbidden_key(key_text):
                continue
            result[key_text] = _redact_json(item)
        return result
    if isinstance(value, list | tuple):
        return [_redact_json(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def _redact_text(value: str | None) -> str | None:
    if value is None:
        return None
    redacted = _TRACEBACK_PATTERN.sub(_REDACTION, value)
    for pattern in _SECRET_PATTERNS + _PRIVATE_TOPOLOGY_PATTERNS:
        redacted = pattern.sub(_REDACTION, redacted)
    return redacted


def _is_forbidden_key(key: str) -> bool:
    normalized = key.strip().lower()
    return normalized in {
        "raw_model_response",
        "raw_output",
        "debug_dump",
        "traceback",
        "stacktrace",
        "base_url",
        "model_base_url",
        "model_path",
        "dsn",
        "password",
        "secret",
        "token",
        "api_key",
    }


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


__all__ = [
    "BacktestAiTrainingExportRecord",
    "BacktestAiTrainingExportSource",
    "BacktestAiTrainingExportUseCase",
    "BacktestAiTrainingQualityLabel",
]

