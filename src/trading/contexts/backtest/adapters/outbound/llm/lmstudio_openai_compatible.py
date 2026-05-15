from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping
from urllib.parse import urljoin

import httpx
from jsonschema import Draft202012Validator

from trading.contexts.backtest.adapters.outbound.config import (
    BacktestAiConfiguratorModelRuntimeConfig,
)
from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigLLMRepairRequest,
    BacktestConfigLLMRequest,
    BacktestConfigLLMResponse,
)

log = logging.getLogger(__name__)


class LMStudioOpenAICompatibleAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        sanitized_response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.sanitized_response_body = sanitized_response_body


@dataclass(frozen=True, slots=True)
class LMStudioOpenAICompatibleAdapter:
    config: BacktestAiConfiguratorModelRuntimeConfig
    http_client: httpx.Client | None = None

    def generate_config(
        self,
        request: BacktestConfigLLMRequest,
    ) -> BacktestConfigLLMResponse:
        return self._complete(kind="generate", request=request)

    def repair_config(
        self,
        request: BacktestConfigLLMRepairRequest,
    ) -> BacktestConfigLLMResponse:
        return self._complete(kind="repair", request=request)

    def _complete(
        self,
        *,
        kind: Literal["generate", "repair"],
        request: BacktestConfigLLMRequest | BacktestConfigLLMRepairRequest,
    ) -> BacktestConfigLLMResponse:
        messages = _chat_messages(kind=kind, request=request)
        prompt_text = "\n\n".join(message["content"] for message in messages)
        input_tokens_estimate = _estimate_tokens(prompt_text)
        if input_tokens_estimate > self.config.max_input_tokens:
            raise LMStudioOpenAICompatibleAdapterError(
                "AI configurator prompt exceeds max_input_tokens"
            )
        response_format = _response_format(request.output_schema_json)
        payload = {
            "model": self.config.model_id,
            "messages": messages,
            "response_format": response_format,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_output_tokens,
            "stream": False,
        }
        try:
            response = self._post(payload=payload)
        except httpx.TimeoutException as error:
            raise LMStudioOpenAICompatibleAdapterError(
                f"LM Studio {kind} request timed out"
            ) from error
        except httpx.HTTPError as error:
            raise LMStudioOpenAICompatibleAdapterError(
                f"LM Studio {kind} request failed"
            ) from error
        if response.status_code >= 400:
            sanitized_body = _sanitize_response_body(response.text)
            log.warning(
                "event=backtest_ai_lmstudio_http_error kind=%s status_code=%s "
                "response_body=%s",
                kind,
                response.status_code,
                sanitized_body,
            )
            raise LMStudioOpenAICompatibleAdapterError(
                f"LM Studio {kind} request returned HTTP {response.status_code}",
                status_code=response.status_code,
                sanitized_response_body=sanitized_body,
            )
        data = _response_json(response=response, kind=kind)
        raw_output = _choice_content(data=data, kind=kind)
        normalized_output = _validate_structured_content(
            raw_output=raw_output,
            schema=response_format["json_schema"]["schema"],
            application_schema=request.output_schema_json,
            kind=kind,
        )
        usage = data.get("usage")
        usage_mapping = usage if isinstance(usage, Mapping) else {}
        finish_reason = _finish_reason(data=data)
        return BacktestConfigLLMResponse(
            raw_output=normalized_output,
            model_id=self.config.model_id,
            model_path_hash=_model_path_hash(self.config),
            input_tokens_estimate=_optional_int(
                usage_mapping,
                "prompt_tokens",
                fallback=input_tokens_estimate,
            ),
            output_tokens_estimate=_optional_int(
                usage_mapping,
                "completion_tokens",
                fallback=_estimate_tokens(normalized_output),
            ),
            finish_reason=finish_reason,
        )

    def _post(self, *, payload: Mapping[str, Any]) -> httpx.Response:
        url = urljoin(self.config.base_url.rstrip("/") + "/", "v1/chat/completions")
        if self.http_client is not None:
            return self.http_client.post(
                url,
                json=payload,
                timeout=self.config.request_timeout_seconds,
            )
        with httpx.Client(timeout=self.config.request_timeout_seconds) as client:
            return client.post(url, json=payload)


def _response_json(*, response: httpx.Response, kind: str) -> Mapping[str, Any]:
    try:
        data = response.json()
    except ValueError as error:
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} response is not JSON"
        ) from error
    if not isinstance(data, Mapping):
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} response must be object"
        )
    return data


def _choice_content(*, data: Mapping[str, Any], kind: str) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} response has no choices"
        )
    first = choices[0]
    if not isinstance(first, Mapping):
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} choice must be object"
        )
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} choice has no message"
        )
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} message content is empty"
        )
    return content


def _finish_reason(*, data: Mapping[str, Any]) -> str | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    value = first.get("finish_reason")
    return value if isinstance(value, str) else None


def _optional_int(
    payload: Mapping[str, Any],
    key: str,
    *,
    fallback: int,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    return value


def _estimate_tokens(value: str) -> int:
    return max(1, (len(value) + 3) // 4)


def _model_path_hash(config: BacktestAiConfiguratorModelRuntimeConfig) -> str:
    return hashlib.sha256(str(config.model_path).encode("utf-8")).hexdigest()


def _chat_messages(
    *,
    kind: Literal["generate", "repair"],
    request: BacktestConfigLLMRequest | BacktestConfigLLMRepairRequest,
) -> list[dict[str, str]]:
    blocks: list[tuple[str, str]] = [
        ("TRUSTED_ALLOWED_CATALOG", _canonical_json(request.catalog_subset_json)),
        ("UNTRUSTED_USER_REQUEST", request.job.user_prompt_text),
        (
            "UNTRUSTED_CURRENT_CONFIG",
            _canonical_json(request.job.current_config_json or {}),
        ),
    ]
    if kind == "repair" and isinstance(request, BacktestConfigLLMRepairRequest):
        blocks.append(
            (
                "UNTRUSTED_REPAIR_CONTEXT",
                _canonical_json(
                    {
                        "failed_raw_output": request.failed_raw_output,
                        "parsed_draft": request.parsed_draft_json,
                        "validation_errors": list(request.validation_errors_json),
                        "repair_attempts": 1,
                    }
                ),
            )
        )
    blocks.append(("OUTPUT_JSON_SCHEMA", _canonical_json(request.output_schema_json)))
    blocks.append(
        (
            "OUTPUT_REQUIREMENTS",
            (
                "For config_ready, config must include coordinates, timeframe, "
                "time_range, indicators, risk, execution, ranking, and top_n. "
                "Use only values from TRUSTED_ALLOWED_CATALOG. Use one "
                "coordinates.symbol only. Never add scripts, HTML, API calls, "
                "job creation, or auto-run actions."
            ),
        )
    )
    user_content = "\n\n".join(f"<{name}>\n{body}\n</{name}>" for name, body in blocks)
    return [
        {"role": "system", "content": request.prompt_profile.system_policy},
        {"role": "user", "content": user_content},
    ]


def _response_format(schema: Mapping[str, Any]) -> dict[str, Any]:
    lmstudio_schema = _lmstudio_json_schema(schema)
    _assert_schema_type_values_are_strings(lmstudio_schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "roehub_backtest_ai_config",
            "strict": "true",
            "schema": lmstudio_schema,
        },
    }


def _lmstudio_json_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise LMStudioOpenAICompatibleAdapterError(
            "AI configurator output schema must define object properties"
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(schema.get("required", ())),
        "properties": {
            "schema_version": {"type": "integer", "const": 1},
            "mode": {
                "type": "string",
                "enum": ["create", "edit", "repair", "suggest_safer", "explain"],
            },
            "status": {
                "type": "string",
                "enum": ["config_ready", "needs_clarification"],
            },
            "assistant_message": {"type": "string"},
            "assumptions": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "config": {"type": "object", "additionalProperties": True},
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
    }


def _assert_schema_type_values_are_strings(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "type" and not isinstance(item, str):
                raise LMStudioOpenAICompatibleAdapterError(
                    "LM Studio JSON Schema type values must be strings"
                )
            _assert_schema_type_values_are_strings(item)
    elif isinstance(value, list):
        for item in value:
            _assert_schema_type_values_are_strings(item)


def _validate_structured_content(
    *,
    raw_output: str,
    schema: Mapping[str, Any],
    application_schema: Mapping[str, Any],
    kind: str,
) -> str:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as error:
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} message content is not JSON"
        ) from error
    if not isinstance(parsed, dict):
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} message content must be JSON object"
        )
    errors = tuple(Draft202012Validator(schema).iter_errors(parsed))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.absolute_path) or "body"
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} structured content failed schema validation at {path}"
        )
    normalized = _normalize_for_application_schema(parsed)
    application_errors = tuple(
        Draft202012Validator(application_schema).iter_errors(normalized)
    )
    if application_errors:
        first = application_errors[0]
        path = ".".join(str(part) for part in first.absolute_path) or "body"
        raise LMStudioOpenAICompatibleAdapterError(
            f"LM Studio {kind} content failed application schema validation at {path}"
        )
    return _canonical_json(normalized)


def _normalize_for_application_schema(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parsed)
    if normalized.get("status") == "needs_clarification":
        normalized["config"] = None
    return normalized


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,;]+"),
    re.compile(r"(?i)(password['\"]?\s*[:=]\s*['\"]?)[^'\"\s,;]+"),
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"/Users/[^/\\\s]+"),
)


def _sanitize_response_body(value: str) -> str:
    sanitized = value[:2000]
    for pattern in _SECRET_PATTERNS:
        if pattern.pattern.startswith("/Users/"):
            sanitized = pattern.sub("/Users/<redacted>", sanitized)
        elif pattern.pattern.startswith("sk-"):
            sanitized = pattern.sub("sk-<redacted>", sanitized)
        else:
            sanitized = pattern.sub(r"\1<redacted>", sanitized)
    return sanitized


__all__ = [
    "LMStudioOpenAICompatibleAdapter",
    "LMStudioOpenAICompatibleAdapterError",
]
