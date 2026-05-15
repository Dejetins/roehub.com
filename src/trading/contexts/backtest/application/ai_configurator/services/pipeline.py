from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Mapping, Sequence
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

from .catalog import (
    BacktestAiAllowedCatalog,
    BacktestAiCatalogResolver,
    BacktestAiIndicatorCatalogItem,
)
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

        fallback_output = _fallback_supported_config_output(job=job, catalog=catalog)
        if outcome.status == "needs_clarification" and fallback_output is not None:
            fallback_outcome = self.validator.validate_model_output(
                raw_output=fallback_output,
                catalog=catalog,
            )
            if fallback_outcome.loadable:
                outcome = fallback_outcome

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


def _fallback_supported_config_output(
    *,
    job: BacktestAiConfigJob,
    catalog: BacktestAiAllowedCatalog,
) -> str | None:
    text = job.user_prompt_text.casefold()
    if "/backtests" not in text and "backtest" not in text:
        return None
    recognized_indicator = _recognized_indicator(text=text, catalog=catalog)
    current_config = job.current_config_json if isinstance(job.current_config_json, Mapping) else {}
    if (
        recognized_indicator is None
        and job.mode not in {"edit", "repair"}
        and not isinstance(current_config.get("indicators"), list)
    ):
        return None

    config = catalog.default_config()
    symbol = _recognized_choice(text=text, values=catalog.symbols, upper=True)
    if symbol is not None:
        config["coordinates"]["symbol"] = symbol
    elif isinstance(current_config.get("coordinates"), Mapping):
        current_symbol = current_config["coordinates"].get("symbol")
        if isinstance(current_symbol, str) and current_symbol.upper() in catalog.symbols:
            config["coordinates"]["symbol"] = current_symbol.upper()

    timeframe = _recognized_choice(text=text, values=catalog.timeframes, upper=False)
    if timeframe is not None:
        config["timeframe"] = timeframe

    indicator = recognized_indicator or _current_indicator(
        current_config=current_config,
        catalog=catalog,
    )
    if indicator is not None:
        config["indicators"] = [_default_indicator_config(item=indicator)]

    top_n = _recognized_top_n(text)
    if top_n is not None:
        config["top_n"] = top_n

    payload = {
        "schema_version": 1,
        "mode": job.mode,
        "status": "config_ready",
        "assistant_message": "Configuration is ready to load.",
        "assumptions": [
            "Supported /backtests values were normalized from the trusted catalog."
        ],
        "warnings": [],
        "config": config,
        "suggestions": [],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _recognized_indicator(
    *,
    text: str,
    catalog: BacktestAiAllowedCatalog,
) -> BacktestAiIndicatorCatalogItem | None:
    for item in catalog.indicators:
        aliases = (item.indicator_id, item.indicator_id.rsplit(".", maxsplit=1)[-1])
        if any(_contains_token(text=text, token=alias) for alias in aliases):
            return item
    return None


def _current_indicator(
    *,
    current_config: Mapping[str, Any],
    catalog: BacktestAiAllowedCatalog,
) -> BacktestAiIndicatorCatalogItem | None:
    indicators = current_config.get("indicators")
    if not isinstance(indicators, list) or not indicators:
        return None
    first = indicators[0]
    if not isinstance(first, Mapping):
        return None
    indicator_id = first.get("indicator_id")
    if not isinstance(indicator_id, str):
        return None
    return catalog.indicator_by_id(indicator_id)


def _recognized_choice(
    *,
    text: str,
    values: tuple[str, ...],
    upper: bool,
) -> str | None:
    for value in values:
        if _contains_token(text=text, token=value):
            return value.upper() if upper else value.lower()
    return None


def _contains_token(*, text: str, token: str) -> bool:
    normalized = token.strip().casefold()
    if not normalized:
        return False
    pattern = rf"(?<![a-z0-9_-]){re.escape(normalized)}(?![a-z0-9_-])"
    return re.search(pattern, text) is not None


def _recognized_top_n(text: str) -> int | None:
    match = re.search(r"\btop\s*[-_ ]?\s*(\d{1,3})\b", text)
    if match is None:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _default_indicator_config(*, item: BacktestAiIndicatorCatalogItem) -> dict[str, Any]:
    params = dict(item.param_specs.get("params") or {})
    window_spec = dict(params.get("window") or {})
    start = int(window_spec.get("start") or _first_int(window_spec.get("values"), default=14))
    stop = min(int(window_spec.get("stop_incl") or start), max(start, 28))
    step = int(window_spec.get("step") or 7)
    return {
        "indicator_id": item.indicator_id,
        "sources": list(item.sources[:1]) or ["close"],
        "window": {"start": start, "stop": stop, "step": step},
    }


def _first_int(value: Any, *, default: int) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, int):
                return item
    return default


__all__ = [
    "BacktestAiConfigPipeline",
    "BacktestAiConfigPipelineResult",
    "BacktestAiPipelineStage",
]
