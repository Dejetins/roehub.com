from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from trading.contexts.backtest.application.ai_configurator.dto import (
    BacktestAiConfigJob,
    BacktestAiConfigLlmAttempt,
)
from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigAgentGateway,
    BacktestConfigAgentRepairRequest,
    BacktestConfigAgentRequest,
    BacktestConfigAgentResponse,
)

from .catalog import BacktestAiCatalogResolver
from .security import BacktestAiInputGate
from .validator import BacktestAiConfigValidationOutcome, BacktestAiConfigValidator

BacktestAiPipelineStage = Literal["input_gate", "runtime_gateway", "validation"]
_REPAIR_ATTEMPTS = 1
_REPAIR_BLOCKED_CODES = {
    "artifact_indicator_unavailable",
    "artifact_period_unavailable",
    "automatic_backtest_action",
    "multi_symbol_field_not_allowed",
    "private_or_secret_leakage",
    "unsupported_config_field",
    "unsupported_direction_mode",
    "unsupported_exchange",
    "unsupported_indicator",
    "unsupported_market_type",
    "unsupported_ranking_metric",
    "unsupported_risk_mode",
    "unsupported_sizing_mode",
    "unsupported_source",
    "unsupported_symbol",
    "unsupported_timeframe",
    "unsupported_window_axis",
}


@dataclass(frozen=True, slots=True)
class BacktestAiConfigPipelineResult:
    status: Literal[
        "ready",
        "needs_clarification",
        "blocked_by_policy",
        "input_too_large",
        "security_review",
        "failed",
    ]
    assistant_message: str
    catalog_snapshot_hash: str
    stage: BacktestAiPipelineStage
    validated_config: dict[str, Any] | None = None
    warnings: tuple[dict[str, Any], ...] = ()
    suggestions: tuple[dict[str, Any], ...] = ()
    validation_errors: tuple[dict[str, Any], ...] = ()
    model_id: str | None = None
    model_path_hash: str | None = None
    intent: str | None = None
    last_error: str | None = None
    last_error_json: dict[str, Any] | None = None
    llm_attempts: tuple[BacktestAiConfigLlmAttempt, ...] = ()

    @classmethod
    def from_validation(
        cls,
        *,
        outcome: BacktestAiConfigValidationOutcome,
        catalog_snapshot_hash: str,
        model_id: str | None,
        model_path_hash: str | None,
        llm_attempts: tuple[BacktestAiConfigLlmAttempt, ...],
    ) -> BacktestAiConfigPipelineResult:
        return cls(
            status=outcome.status,
            assistant_message=outcome.assistant_message,
            catalog_snapshot_hash=catalog_snapshot_hash,
            stage="validation",
            validated_config=outcome.validated_config,
            warnings=outcome.warnings,
            suggestions=outcome.suggestions,
            validation_errors=outcome.validation_errors,
            model_id=model_id,
            model_path_hash=model_path_hash,
            intent=_intent_from_draft(outcome.parsed_draft),
            last_error=outcome.last_error,
            last_error_json=outcome.last_error_json,
            llm_attempts=llm_attempts,
        )


@dataclass(frozen=True, slots=True)
class BacktestAiConfigPipeline:
    catalog_resolver: BacktestAiCatalogResolver
    validator: BacktestAiConfigValidator
    agent_gateway: BacktestConfigAgentGateway
    input_gate: BacktestAiInputGate = BacktestAiInputGate()

    def run(self, *, job: BacktestAiConfigJob) -> BacktestAiConfigPipelineResult:
        catalog = self.catalog_resolver.resolve()
        input_gate = self.input_gate.evaluate(
            message=job.user_prompt_text,
            locale=job.locale,
            mode=job.mode,
        )
        if not input_gate.allowed:
            return BacktestAiConfigPipelineResult(
                status=input_gate.terminal_status or "blocked_by_policy",
                assistant_message=input_gate.user_message or "Request cannot be processed.",
                catalog_snapshot_hash=catalog.snapshot_hash,
                stage="input_gate",
                validation_errors=tuple(
                    {
                        "path": "message",
                        "code": flag,
                        "message": "Request did not pass deterministic input checks",
                    }
                    for flag in input_gate.flags
                ),
                last_error=input_gate.terminal_status or "blocked_by_policy",
                last_error_json={
                    "security_flags": list(input_gate.flags),
                    "security_risk_score": input_gate.risk_score,
                    "security_decision": input_gate.decision,
                },
            )

        agent_response = _timed_agent_session(
            gateway=self.agent_gateway,
            request=BacktestConfigAgentRequest(job=job, catalog=catalog),
        )
        if agent_response.raw_output is None:
            return BacktestAiConfigPipelineResult(
                status="failed",
                assistant_message=(
                    "AI configurator assistant v1 runtime is not available yet."
                ),
                catalog_snapshot_hash=catalog.snapshot_hash,
                stage="runtime_gateway",
                model_id=agent_response.model_id,
                model_path_hash=agent_response.model_path_hash,
                last_error="assistant_v1_runtime_unavailable",
                last_error_json=dict(agent_response.audit_json or {}),
            )

        outcome = self.validator.validate_model_output(
            raw_output=agent_response.raw_output,
            catalog=catalog,
        )
        attempts = (
            _llm_attempt(
                job=job,
                catalog_snapshot_hash=catalog.snapshot_hash,
                response=agent_response,
                attempt_no=1,
                attempt_kind="generate",
                prompt_profile="assistant_v1",
                outcome=outcome,
            ),
        )
        if _should_repair(outcome=outcome):
            repair_response = _timed_repair_session(
                gateway=self.agent_gateway,
                request=BacktestConfigAgentRepairRequest(
                    job=job,
                    catalog=catalog,
                    previous_draft=outcome.parsed_draft or {},
                    validation_errors=outcome.validation_errors,
                ),
            )
            if repair_response.raw_output is not None:
                repair_outcome = self.validator.validate_model_output(
                    raw_output=repair_response.raw_output,
                    catalog=catalog,
                )
            else:
                repair_outcome = outcome
            attempts = attempts + (
                _llm_attempt(
                    job=job,
                    catalog_snapshot_hash=catalog.snapshot_hash,
                    response=repair_response,
                    attempt_no=2,
                    attempt_kind="repair",
                    prompt_profile="assistant_v1_repair",
                    outcome=repair_outcome,
                ),
            )
            outcome = repair_outcome
        return BacktestAiConfigPipelineResult.from_validation(
            outcome=outcome,
            catalog_snapshot_hash=catalog.snapshot_hash,
            model_id=agent_response.model_id,
            model_path_hash=agent_response.model_path_hash,
            llm_attempts=attempts,
        )


def _timed_agent_session(
    *,
    gateway: BacktestConfigAgentGateway,
    request: BacktestConfigAgentRequest,
) -> BacktestConfigAgentResponse:
    started = time.perf_counter()
    response = gateway.run_config_session(request)
    if response.latency_ms is not None:
        return response
    latency_ms = max(0, round((time.perf_counter() - started) * 1000))
    return BacktestConfigAgentResponse(
        raw_output=response.raw_output,
        model_id=response.model_id,
        model_path_hash=response.model_path_hash,
        latency_ms=latency_ms,
        finish_reason=response.finish_reason,
        audit_json=response.audit_json,
    )


def _timed_repair_session(
    *,
    gateway: BacktestConfigAgentGateway,
    request: BacktestConfigAgentRepairRequest,
) -> BacktestConfigAgentResponse:
    started = time.perf_counter()
    response = gateway.run_repair_config_session(request)
    if response.latency_ms is not None:
        return response
    latency_ms = max(0, round((time.perf_counter() - started) * 1000))
    return BacktestConfigAgentResponse(
        raw_output=response.raw_output,
        model_id=response.model_id,
        model_path_hash=response.model_path_hash,
        latency_ms=latency_ms,
        finish_reason=response.finish_reason,
        audit_json=response.audit_json,
    )


def _should_repair(*, outcome: BacktestAiConfigValidationOutcome) -> bool:
    if outcome.status != "needs_clarification" or outcome.parsed_draft is None:
        return False
    codes = {str(item.get("code") or "") for item in outcome.validation_errors}
    if not codes:
        return False
    if codes & _REPAIR_BLOCKED_CODES:
        return False
    return _REPAIR_ATTEMPTS == 1


def _llm_attempt(
    *,
    job: BacktestAiConfigJob,
    catalog_snapshot_hash: str,
    response: BacktestConfigAgentResponse,
    attempt_no: int,
    attempt_kind: Literal["generate", "repair"],
    prompt_profile: str,
    outcome: BacktestAiConfigValidationOutcome,
) -> BacktestAiConfigLlmAttempt:
    return BacktestAiConfigLlmAttempt(
        attempt_id=uuid4(),
        job_id=job.job_id,
        owner_user_id=job.owner_user_id,
        attempt_no=attempt_no,
        attempt_kind=attempt_kind,
        prompt_profile=prompt_profile,
        system_prompt_version=job.system_prompt_version,
        system_prompt_hash=job.system_prompt_hash,
        user_prompt_text=job.user_prompt_text,
        catalog_subset_json={
            "snapshot_hash": catalog_snapshot_hash,
            "repair_attempts": _REPAIR_ATTEMPTS,
        },
        raw_model_response=response.raw_output,
        parsed_json_draft=parse_llm_json_object_for_audit(response.raw_output or ""),
        validation_errors_json=outcome.validation_errors,
        input_tokens_estimate=None,
        output_tokens_estimate=None,
        latency_ms=response.latency_ms,
        finish_reason=response.finish_reason,
        success=outcome.loadable,
        failure_reason=None if outcome.loadable else outcome.last_error,
        created_at=datetime.now(UTC),
    )


def parse_llm_json_object_for_audit(raw_output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _intent_from_draft(parsed_draft: dict[str, Any] | None) -> str | None:
    if not isinstance(parsed_draft, dict):
        return None
    raw_intent = parsed_draft.get("intent")
    if isinstance(raw_intent, str) and raw_intent.strip():
        return raw_intent.strip()
    return None


__all__ = [
    "BacktestAiConfigPipeline",
    "BacktestAiConfigPipelineResult",
    "BacktestAiPipelineStage",
]
