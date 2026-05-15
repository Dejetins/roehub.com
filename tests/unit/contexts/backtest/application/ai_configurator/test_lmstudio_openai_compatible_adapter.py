from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest

from trading.contexts.backtest.adapters.outbound.config import (
    BacktestAiConfiguratorModelRuntimeConfig,
)
from trading.contexts.backtest.adapters.outbound.llm import (
    LMStudioOpenAICompatibleAdapter,
    LMStudioOpenAICompatibleAdapterError,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigJob,
    BacktestConfigLLMRepairRequest,
    BacktestConfigLLMRequest,
    backtest_ai_model_output_schema,
    backtest_ai_prompt_profile_for_mode,
)
from trading.shared_kernel.primitives import UserId


def test_lmstudio_adapter_posts_structured_chat_completions_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": _valid_model_output()},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    adapter = LMStudioOpenAICompatibleAdapter(
        config=_model_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = adapter.generate_config(_request(prompt_text="trusted prompt envelope"))

    assert captured["path"] == "/v1/chat/completions"
    payload = captured["payload"]
    assert payload["model"] == "gemma-4-e2b-it-4bit"
    assert payload["messages"][0]["role"] == "system"
    assert "Ты Backtest AI Configurator" in payload["messages"][0]["content"]
    assert payload["messages"][1]["role"] == "user"
    assert "trusted prompt envelope" not in payload["messages"][1]["content"]
    assert "Собери конфиг" in payload["messages"][1]["content"]
    assert "TRUSTED_ALLOWED_CATALOG" in payload["messages"][1]["content"]
    assert "OUTPUT_JSON_SCHEMA" in payload["messages"][1]["content"]
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 1024
    assert payload["stream"] is False
    assert payload["response_format"]["type"] == "json_schema"
    json_schema = payload["response_format"]["json_schema"]
    assert json_schema["name"] == "roehub_backtest_ai_config"
    assert json_schema["strict"] == "true"
    assert json_schema["schema"]["properties"]["config"]["type"] == "object"
    assert _all_schema_type_values_are_strings(json_schema["schema"])
    assert json.loads(response.raw_output) == json.loads(_valid_model_output())
    assert response.model_id == "gemma-4-e2b-it-4bit"
    assert response.model_path_hash is not None
    assert response.input_tokens_estimate == 7
    assert response.output_tokens_estimate == 3
    assert response.finish_reason == "stop"


def test_lmstudio_adapter_adds_trusted_request_interpretation() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": _valid_model_output()},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    adapter = LMStudioOpenAICompatibleAdapter(
        config=_model_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    adapter.generate_config(
        _request(
            prompt_text="trusted prompt envelope",
            user_prompt_text="Create BTCUSDT RSI config on 15m and keep top 10 results.",
            catalog_subset_json={
                "symbols": ["BTCUSDT"],
                "timeframes": ["15m"],
                "indicators": [
                    {
                        "indicator_id": "momentum.rsi",
                        "aliases": ["rsi", "RSI"],
                    }
                ],
            },
        )
    )

    content = captured["payload"]["messages"][1]["content"]
    assert "TRUSTED_REQUEST_INTERPRETATION" in content
    assert '"recognized_symbol":"BTCUSDT"' in content
    assert '"recognized_timeframe":"15m"' in content
    assert '"recognized_indicator_id":"momentum.rsi"' in content
    assert '"recognized_top_n":10' in content


def test_lmstudio_adapter_supports_repair_operation() -> None:
    captured: dict[str, Any] = {}

    def handler(_request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(_request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": _valid_model_output()},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    adapter = LMStudioOpenAICompatibleAdapter(
        config=_model_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = adapter.repair_config(
        BacktestConfigLLMRepairRequest(
            job=_request(prompt_text="repair prompt").job,
            catalog=object(),  # type: ignore[arg-type]
            prompt_profile=backtest_ai_prompt_profile_for_mode("create"),
            prompt_text="repair prompt",
            catalog_subset_json={},
            output_schema_json=backtest_ai_model_output_schema(),
            failed_raw_output="{bad",
            parsed_draft_json=None,
            validation_errors_json=({"code": "invalid_json"},),
        )
    )

    assert json.loads(response.raw_output) == json.loads(_valid_model_output())
    assert "UNTRUSTED_REPAIR_CONTEXT" in captured["payload"]["messages"][1]["content"]


def test_lmstudio_adapter_rejects_non_loopback_base_url() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        _model_config(base_url="http://10.0.0.12:8080")


def test_lmstudio_adapter_timeout_is_deterministic_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    adapter = LMStudioOpenAICompatibleAdapter(
        config=_model_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(LMStudioOpenAICompatibleAdapterError, match="timed out"):
        adapter.generate_config(_request(prompt_text="prompt"))


def test_lmstudio_adapter_sanitizes_http_error_body_for_diagnostics() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            text=(
                "bad schema Authorization: Bearer secret-token "
                "api_key=sk-12345678901234567890 "
                "/Users/daniildegtyarev/private/model"
            ),
        )

    adapter = LMStudioOpenAICompatibleAdapter(
        config=_model_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(LMStudioOpenAICompatibleAdapterError) as exc_info:
        adapter.generate_config(_request(prompt_text="prompt"))

    assert exc_info.value.status_code == 400
    assert exc_info.value.sanitized_response_body is not None
    assert "secret-token" not in exc_info.value.sanitized_response_body
    assert "sk-12345678901234567890" not in exc_info.value.sanitized_response_body
    assert "/Users/daniildegtyarev" not in exc_info.value.sanitized_response_body
    assert "<redacted>" in exc_info.value.sanitized_response_body


def test_lmstudio_adapter_normalizes_empty_clarification_config_to_null() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": _needs_clarification_output()},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    adapter = LMStudioOpenAICompatibleAdapter(
        config=_model_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = adapter.generate_config(_request(prompt_text="prompt"))

    assert json.loads(response.raw_output)["config"] is None


def _model_config(
    *,
    base_url: str = "http://127.0.0.1:8080",
) -> BacktestAiConfiguratorModelRuntimeConfig:
    return BacktestAiConfiguratorModelRuntimeConfig(
        runtime="lm_studio",
        model_id="gemma-4-e2b-it-4bit",
        model_path=Path(
            "/Users/daniildegtyarev/.lmstudio/models/mlx-community/gemma-4-e2b-it-4bit"
        ),
        context_window_tokens=8192,
        max_input_tokens=6144,
        max_output_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        base_url=base_url,
        request_timeout_seconds=3.0,
        active_generations=1,
    )


def _request(
    *,
    prompt_text: str,
    user_prompt_text: str = "Собери конфиг",
    catalog_subset_json: dict[str, Any] | None = None,
) -> BacktestConfigLLMRequest:
    profile = backtest_ai_prompt_profile_for_mode("create")
    now = datetime(2026, 5, 11, tzinfo=UTC)
    return BacktestConfigLLMRequest(
        job=BacktestAiConfigJob(
            job_id=UUID("00000000-0000-0000-0000-000000000601"),
            owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000602"),
            mode="create",
            locale="ru",
            state="running",
            source_page="backtests",
            user_prompt_text=user_prompt_text,
            user_prompt_hash="a" * 64,
            system_prompt_version=profile.system_prompt_version,
            system_prompt_hash=profile.system_prompt_hash,
            catalog_snapshot_hash="b" * 64,
            runtime_defaults_hash="c" * 64,
            queued_at=now,
            updated_at=now,
        ),
        catalog=object(),  # type: ignore[arg-type]
        prompt_profile=profile,
        prompt_text=prompt_text,
        catalog_subset_json=catalog_subset_json or {},
        output_schema_json=backtest_ai_model_output_schema(),
    )


def _valid_model_output() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "mode": "create",
            "status": "config_ready",
            "assistant_message": "Готово.",
            "assumptions": [],
            "warnings": [],
            "config": {
                "coordinates": {
                    "exchange": "binance",
                    "market_type": "spot",
                    "symbol": "BTCUSDT",
                },
                "timeframe": "15m",
                "time_range": {"start": "2024-01-01", "end": "2024-02-01"},
                "indicators": [
                    {
                        "indicator_id": "ma.sma",
                        "sources": ["close"],
                        "window": {"start": 10, "stop": 20, "step": 5},
                    }
                ],
                "risk": {"mode": "none"},
                "execution": {},
                "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
                "top_n": 5,
            },
            "suggestions": [],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _needs_clarification_output() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "mode": "create",
            "status": "needs_clarification",
            "assistant_message": "Уточните период.",
            "assumptions": [],
            "warnings": [],
            "config": {},
            "suggestions": ["Добавьте дату начала и конца."],
        },
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _all_schema_type_values_are_strings(value: Any) -> bool:
    if isinstance(value, dict):
        return all(
            isinstance(item, str) and _all_schema_type_values_are_strings(item)
            if key == "type"
            else _all_schema_type_values_are_strings(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return all(_all_schema_type_values_are_strings(item) for item in value)
    return True
