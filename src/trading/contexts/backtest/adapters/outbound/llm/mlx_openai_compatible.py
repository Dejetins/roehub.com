from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal, Mapping
from urllib.parse import urljoin

import httpx

from trading.contexts.backtest.adapters.outbound.config import (
    BacktestAiConfiguratorModelRuntimeConfig,
)
from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigLLMRepairRequest,
    BacktestConfigLLMRequest,
    BacktestConfigLLMResponse,
)


class MLXOpenAICompatibleAdapterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MLXOpenAICompatibleAdapter:
    config: BacktestAiConfiguratorModelRuntimeConfig
    http_client: httpx.Client | None = None

    def generate_config(
        self,
        request: BacktestConfigLLMRequest,
    ) -> BacktestConfigLLMResponse:
        return self._complete(kind="generate", prompt_text=request.prompt_text)

    def repair_config(
        self,
        request: BacktestConfigLLMRepairRequest,
    ) -> BacktestConfigLLMResponse:
        return self._complete(kind="repair", prompt_text=request.prompt_text)

    def _complete(
        self,
        *,
        kind: Literal["generate", "repair"],
        prompt_text: str,
    ) -> BacktestConfigLLMResponse:
        input_tokens_estimate = _estimate_tokens(prompt_text)
        if input_tokens_estimate > self.config.max_input_tokens:
            raise MLXOpenAICompatibleAdapterError(
                "AI configurator prompt exceeds max_input_tokens"
            )
        payload = {
            "model": self.config.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_output_tokens,
            "stream": False,
        }
        try:
            response = self._post(payload=payload)
        except httpx.TimeoutException as error:
            raise MLXOpenAICompatibleAdapterError(
                f"MLX {kind} request timed out"
            ) from error
        except httpx.HTTPError as error:
            raise MLXOpenAICompatibleAdapterError(
                f"MLX {kind} request failed"
            ) from error
        if response.status_code >= 400:
            raise MLXOpenAICompatibleAdapterError(
                f"MLX {kind} request returned HTTP {response.status_code}"
            )
        data = _response_json(response=response, kind=kind)
        raw_output = _choice_content(data=data, kind=kind)
        usage = data.get("usage")
        usage_mapping = usage if isinstance(usage, Mapping) else {}
        finish_reason = _finish_reason(data=data)
        return BacktestConfigLLMResponse(
            raw_output=raw_output,
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
                fallback=_estimate_tokens(raw_output),
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
        raise MLXOpenAICompatibleAdapterError(
            f"MLX {kind} response is not JSON"
        ) from error
    if not isinstance(data, Mapping):
        raise MLXOpenAICompatibleAdapterError(f"MLX {kind} response must be object")
    return data


def _choice_content(*, data: Mapping[str, Any], kind: str) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise MLXOpenAICompatibleAdapterError(f"MLX {kind} response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise MLXOpenAICompatibleAdapterError(f"MLX {kind} choice must be object")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise MLXOpenAICompatibleAdapterError(f"MLX {kind} choice has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise MLXOpenAICompatibleAdapterError(f"MLX {kind} message content is empty")
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


__all__ = [
    "MLXOpenAICompatibleAdapter",
    "MLXOpenAICompatibleAdapterError",
]
