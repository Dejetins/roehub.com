from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigAgentRepairRequest,
    BacktestConfigAgentRequest,
    BacktestConfigAgentResponse,
)
from trading.contexts.backtest.application.ai_configurator.prompts import (
    BacktestAiPromptMessage,
    BacktestAiPromptPackage,
    build_backtest_ai_prompt_package,
    trusted_context_from_catalog,
)
from trading.contexts.backtest.application.ai_configurator.schema import (
    BACKTEST_AI_OUTPUT_SCHEMA_NAME,
)

_CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


class LMStudioChatCompletionsError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LMStudioChatCompletionsSettings:
    base_url: str
    model_id: str
    request_timeout_seconds: float
    max_output_tokens: int
    temperature: float = 0.2
    top_p: float = 0.9

    @classmethod
    def from_runtime_config(
        cls,
        config: Any,
    ) -> "LMStudioChatCompletionsSettings":
        return cls(
            base_url=config.base_url,
            model_id=config.model_id,
            request_timeout_seconds=config.request_timeout_seconds,
            max_output_tokens=config.max_output_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )


@dataclass(frozen=True, slots=True)
class LMStudioChatCompletionsResult:
    content: str
    model_id: str | None
    finish_reason: str | None
    latency_ms: int
    audit_json: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LMStudioOpenAICompatibleAdapter:
    settings: LMStudioChatCompletionsSettings

    def run_config_session(
        self,
        request: BacktestConfigAgentRequest,
    ) -> BacktestConfigAgentResponse:
        package = build_backtest_ai_prompt_package(
            trusted_context=trusted_context_from_catalog(catalog=request.catalog),
            current_form_config=request.job.current_config_json,
            recent_chat_context=(),
            user_message=request.job.user_prompt_text,
        )
        try:
            result = self.complete_prompt_package(package=package)
        except LMStudioChatCompletionsError as error:
            return BacktestConfigAgentResponse(
                raw_output=None,
                model_id=self.settings.model_id,
                finish_reason="error",
                audit_json={
                    "runtime": "lm_studio",
                    "endpoint": f"POST {_CHAT_COMPLETIONS_PATH}",
                    "response_format": "json_schema",
                    "error": str(error),
                },
            )
        return BacktestConfigAgentResponse(
            raw_output=result.content,
            model_id=result.model_id or self.settings.model_id,
            latency_ms=result.latency_ms,
            finish_reason=result.finish_reason,
            audit_json=result.audit_json,
        )

    def run_repair_config_session(
        self,
        request: BacktestConfigAgentRepairRequest,
    ) -> BacktestConfigAgentResponse:
        package = build_backtest_ai_prompt_package(
            trusted_context=trusted_context_from_catalog(catalog=request.catalog),
            current_form_config=request.job.current_config_json,
            recent_chat_context=(),
            user_message=request.job.user_prompt_text,
        )
        try:
            result = self.run_repair_session(
                package=package,
                previous_draft=request.previous_draft,
                validation_errors=request.validation_errors,
            )
        except LMStudioChatCompletionsError as error:
            return BacktestConfigAgentResponse(
                raw_output=None,
                model_id=self.settings.model_id,
                finish_reason="error",
                audit_json={
                    "runtime": "lm_studio",
                    "endpoint": f"POST {_CHAT_COMPLETIONS_PATH}",
                    "response_format": "json_schema",
                    "repair_attempt": True,
                    "error": str(error),
                },
            )
        return BacktestConfigAgentResponse(
            raw_output=result.content,
            model_id=result.model_id or self.settings.model_id,
            latency_ms=result.latency_ms,
            finish_reason=result.finish_reason,
            audit_json={**dict(result.audit_json), "repair_attempt": True},
        )

    def complete_prompt_package(
        self,
        *,
        package: BacktestAiPromptPackage,
    ) -> LMStudioChatCompletionsResult:
        started = time.perf_counter()
        payload = self.build_payload(package=package)
        try:
            with httpx.Client(
                base_url=self.settings.base_url.rstrip("/"),
                timeout=httpx.Timeout(self.settings.request_timeout_seconds),
            ) as client:
                response = client.post(_CHAT_COMPLETIONS_PATH, json=payload)
                response.raise_for_status()
        except httpx.HTTPError as error:
            raise LMStudioChatCompletionsError(
                f"POST {_CHAT_COMPLETIONS_PATH} failed: {error}"
            ) from error

        try:
            body = response.json()
        except json.JSONDecodeError as error:
            raise LMStudioChatCompletionsError(
                "POST /v1/chat/completions returned non-JSON response"
            ) from error
        content = _extract_message_content(body)
        latency_ms = max(0, round((time.perf_counter() - started) * 1000))
        return LMStudioChatCompletionsResult(
            content=content,
            model_id=_optional_str(body.get("model")),
            finish_reason=_finish_reason(body),
            latency_ms=latency_ms,
            audit_json={
                "runtime": "lm_studio",
                "endpoint": f"POST {_CHAT_COMPLETIONS_PATH}",
                "response_format": "json_schema",
                "schema_name": BACKTEST_AI_OUTPUT_SCHEMA_NAME,
                "tools_sent": False,
                "latency_ms": latency_ms,
            },
        )

    def run_repair_session(
        self,
        *,
        package: BacktestAiPromptPackage,
        previous_draft: Mapping[str, Any],
        validation_errors: tuple[Mapping[str, Any], ...],
    ) -> LMStudioChatCompletionsResult:
        repair_package = BacktestAiPromptPackage(
            system_message=package.system_message,
            user_message=BacktestAiPromptMessage(
                role="user",
                content=(
                    f"{package.user_message.content}\n\n"
                    "REPAIR_INSTRUCTION\n"
                    "Repair the previous JSON draft using only the validation errors. "
                    "If a validation error includes a repair_value, apply that exact "
                    "value and do not ask a clarifying question for it. "
                    "For catalog enum fields such as risk.mode, "
                    "execution.direction_mode, execution.sizing.mode, and "
                    "ranking.primary_metric, replace only the invalid value with "
                    "repair_value or an allowed TRUSTED_CONTEXT_JSON value. "
                    "Keep the same user intent. If a value is unavailable, return "
                    "needs_clarification instead of inventing values.\n\n"
                    "PREVIOUS_JSON_DRAFT\n"
                    f"```json\n{json.dumps(dict(previous_draft), sort_keys=True)}\n```\n\n"
                    "VALIDATION_ERRORS_JSON\n"
                    "```json\n"
                    f"{json.dumps([dict(item) for item in validation_errors], sort_keys=True)}\n"
                    "```"
                ),
            ),
            response_format=package.response_format,
            output_schema=package.output_schema,
            output_example=package.output_example,
            system_prompt_id=package.system_prompt_id,
            system_prompt_hash=package.system_prompt_hash,
        )
        return self.complete_prompt_package(package=repair_package)

    def build_payload(self, *, package: BacktestAiPromptPackage) -> dict[str, Any]:
        return {
            "model": self.settings.model_id,
            "messages": package.messages_payload(),
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_output_tokens,
            "stream": False,
            "response_format": dict(package.response_format),
        }


def _extract_message_content(body: object) -> str:
    if not isinstance(body, Mapping):
        raise LMStudioChatCompletionsError("chat completions response must be an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LMStudioChatCompletionsError("chat completions response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise LMStudioChatCompletionsError("chat completions choice must be an object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise LMStudioChatCompletionsError("chat completions choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise LMStudioChatCompletionsError("chat completions message content is empty")
    return content


def _finish_reason(body: Mapping[str, Any]) -> str | None:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    return _optional_str(choices[0].get("finish_reason"))


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


__all__ = [
    "LMStudioChatCompletionsError",
    "LMStudioChatCompletionsResult",
    "LMStudioChatCompletionsSettings",
    "LMStudioOpenAICompatibleAdapter",
]
