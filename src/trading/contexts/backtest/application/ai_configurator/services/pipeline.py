from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Literal

from trading.contexts.backtest.application.ai_configurator.dto import (
    BacktestAiConfigJob,
    BacktestAiConfigLlmAttempt,
)
from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigAgentGateway,
    BacktestConfigAgentRequest,
    BacktestConfigAgentResponse,
)

from .catalog import BacktestAiCatalogResolver
from .security import BacktestAiInputGate
from .validator import BacktestAiConfigValidationOutcome, BacktestAiConfigValidator

BacktestAiPipelineStage = Literal["input_gate", "tool_agent", "validation"]


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
                    "AI configurator tool-agent runtime is not available yet."
                ),
                catalog_snapshot_hash=catalog.snapshot_hash,
                stage="tool_agent",
                model_id=agent_response.model_id,
                model_path_hash=agent_response.model_path_hash,
                last_error="tool_agent_unavailable",
                last_error_json=dict(agent_response.audit_json or {}),
            )

        outcome = self.validator.validate_model_output(
            raw_output=agent_response.raw_output,
            catalog=catalog,
        )
        return BacktestAiConfigPipelineResult.from_validation(
            outcome=outcome,
            catalog_snapshot_hash=catalog.snapshot_hash,
            model_id=agent_response.model_id,
            model_path_hash=agent_response.model_path_hash,
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
