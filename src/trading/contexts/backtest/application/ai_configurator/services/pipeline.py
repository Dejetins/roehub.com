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
    BacktestConfigLLMGateway,
    BacktestConfigLLMRepairRequest,
    BacktestConfigLLMRequest,
    BacktestConfigLLMResponse,
)

from .catalog import BacktestAiCatalogResolver
from .prompt_profiles import (
    build_generate_prompt_envelope,
    build_repair_prompt_envelope,
)
from .repair import BacktestAiRepairController
from .security import BacktestAiInputGate
from .validator import BacktestAiConfigValidationOutcome, BacktestAiConfigValidator

BacktestAiPipelineStage = Literal["input_gate", "validation"]


@dataclass(frozen=True, slots=True)
class BacktestAiConfigPipelineResult:
    status: Literal[
        "ready",
        "needs_clarification",
        "blocked_by_policy",
        "input_too_large",
        "security_review",
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
            last_error=outcome.last_error,
            last_error_json=outcome.last_error_json,
            llm_attempts=llm_attempts,
        )


@dataclass(frozen=True, slots=True)
class BacktestAiConfigPipeline:
    catalog_resolver: BacktestAiCatalogResolver
    validator: BacktestAiConfigValidator
    llm_gateway: BacktestConfigLLMGateway
    input_gate: BacktestAiInputGate = BacktestAiInputGate()
    repair_controller: BacktestAiRepairController = BacktestAiRepairController()

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

        generate_envelope = build_generate_prompt_envelope(job=job, catalog=catalog)
        generate_response = _timed_generate(
            gateway=self.llm_gateway,
            request=BacktestConfigLLMRequest(
                job=job,
                catalog=catalog,
                prompt_profile=generate_envelope.profile,
                prompt_text=generate_envelope.prompt_text,
                catalog_subset_json=generate_envelope.catalog_subset,
                output_schema_json=generate_envelope.output_schema,
            ),
        )
        outcome = self.validator.validate_model_output(
            raw_output=generate_response.raw_output,
            catalog=catalog,
        )
        attempts = (
            _audit_attempt(
                job=job,
                attempt_no=1,
                attempt_kind="generate",
                profile=generate_envelope.profile,
                envelope_catalog=generate_envelope.catalog_subset,
                response=generate_response,
                outcome=outcome,
            ),
        )
        final_response = generate_response

        if self.repair_controller.should_repair(outcome=outcome, repairs_used=0):
            repair_envelope = build_repair_prompt_envelope(
                job=job,
                catalog=catalog,
                failed_raw_output=generate_response.raw_output,
                parsed_draft=outcome.parsed_draft,
                validation_errors=outcome.validation_errors,
            )
            repair_response = _timed_repair(
                gateway=self.llm_gateway,
                request=BacktestConfigLLMRepairRequest(
                    job=job,
                    catalog=catalog,
                    prompt_profile=repair_envelope.profile,
                    prompt_text=repair_envelope.prompt_text,
                    catalog_subset_json=repair_envelope.catalog_subset,
                    output_schema_json=repair_envelope.output_schema,
                    failed_raw_output=generate_response.raw_output,
                    parsed_draft_json=outcome.parsed_draft,
                    validation_errors_json=outcome.validation_errors,
                ),
            )
            repair_outcome = self.validator.validate_model_output(
                raw_output=repair_response.raw_output,
                catalog=catalog,
            )
            attempts = attempts + (
                _audit_attempt(
                    job=job,
                    attempt_no=2,
                    attempt_kind="repair",
                    profile=repair_envelope.profile,
                    envelope_catalog=repair_envelope.catalog_subset,
                    response=repair_response,
                    outcome=repair_outcome,
                ),
            )
            outcome = repair_outcome
            final_response = repair_response

        return BacktestAiConfigPipelineResult.from_validation(
            outcome=outcome,
            catalog_snapshot_hash=catalog.snapshot_hash,
            model_id=final_response.model_id,
            model_path_hash=final_response.model_path_hash,
            llm_attempts=attempts,
        )


def _timed_generate(
    *,
    gateway: BacktestConfigLLMGateway,
    request: BacktestConfigLLMRequest,
) -> BacktestConfigLLMResponse:
    started = time.perf_counter()
    response = gateway.generate_config(request)
    return _with_latency(response=response, started=started)


def _timed_repair(
    *,
    gateway: BacktestConfigLLMGateway,
    request: BacktestConfigLLMRepairRequest,
) -> BacktestConfigLLMResponse:
    started = time.perf_counter()
    response = gateway.repair_config(request)
    return _with_latency(response=response, started=started)


def _with_latency(
    *,
    response: BacktestConfigLLMResponse,
    started: float,
) -> BacktestConfigLLMResponse:
    if response.latency_ms is not None:
        return response
    latency_ms = max(0, round((time.perf_counter() - started) * 1000))
    return BacktestConfigLLMResponse(
        raw_output=response.raw_output,
        model_id=response.model_id,
        model_path_hash=response.model_path_hash,
        input_tokens_estimate=response.input_tokens_estimate,
        output_tokens_estimate=response.output_tokens_estimate,
        latency_ms=latency_ms,
        finish_reason=response.finish_reason,
    )


def _audit_attempt(
    *,
    job: BacktestAiConfigJob,
    attempt_no: int,
    attempt_kind: Literal["generate", "repair"],
    profile: Any,
    envelope_catalog: Any,
    response: BacktestConfigLLMResponse,
    outcome: BacktestAiConfigValidationOutcome,
) -> BacktestAiConfigLlmAttempt:
    return BacktestAiConfigLlmAttempt(
        attempt_id=uuid4(),
        job_id=job.job_id,
        owner_user_id=job.owner_user_id,
        attempt_no=attempt_no,
        attempt_kind=attempt_kind,
        prompt_profile=profile.name,
        system_prompt_version=profile.system_prompt_version,
        system_prompt_hash=profile.system_prompt_hash,
        user_prompt_text=job.user_prompt_text,
        catalog_subset_json=dict(envelope_catalog),
        raw_model_response=response.raw_output,
        parsed_json_draft=outcome.parsed_draft,
        validation_errors_json=outcome.validation_errors,
        input_tokens_estimate=response.input_tokens_estimate,
        output_tokens_estimate=response.output_tokens_estimate,
        latency_ms=response.latency_ms,
        finish_reason=response.finish_reason,
        success=outcome.loadable,
        failure_reason=None if outcome.loadable else outcome.last_error or outcome.status,
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


__all__ = [
    "BacktestAiConfigPipeline",
    "BacktestAiConfigPipelineResult",
    "BacktestAiPipelineStage",
]
