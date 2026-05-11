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
    MLXOpenAICompatibleAdapter,
    MLXOpenAICompatibleAdapterError,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigJob,
    BacktestConfigLLMRepairRequest,
    BacktestConfigLLMRequest,
    backtest_ai_prompt_profile_for_mode,
)
from trading.shared_kernel.primitives import UserId


def test_mlx_adapter_posts_openai_chat_completions_request() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "{\"schema_version\":1}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    adapter = MLXOpenAICompatibleAdapter(
        config=_model_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    response = adapter.generate_config(_request(prompt_text="trusted prompt envelope"))

    assert captured["path"] == "/v1/chat/completions"
    assert captured["payload"] == {
        "model": "gemma-4-e2b-it-4bit",
        "messages": [{"role": "user", "content": "trusted prompt envelope"}],
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 1024,
        "stream": False,
    }
    assert response.raw_output == "{\"schema_version\":1}"
    assert response.model_id == "gemma-4-e2b-it-4bit"
    assert response.model_path_hash is not None
    assert response.input_tokens_estimate == 7
    assert response.output_tokens_estimate == 3
    assert response.finish_reason == "stop"


def test_mlx_adapter_supports_repair_operation() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "{\"schema_version\":1,\"repaired\":true}"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    adapter = MLXOpenAICompatibleAdapter(
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
            output_schema_json={},
            failed_raw_output="{bad",
            parsed_draft_json=None,
            validation_errors_json=({"code": "invalid_json"},),
        )
    )

    assert response.raw_output == "{\"schema_version\":1,\"repaired\":true}"


def test_mlx_adapter_rejects_non_loopback_base_url() -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        _model_config(base_url="http://10.0.0.12:8080")


def test_mlx_adapter_timeout_is_deterministic_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    adapter = MLXOpenAICompatibleAdapter(
        config=_model_config(),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(MLXOpenAICompatibleAdapterError, match="timed out"):
        adapter.generate_config(_request(prompt_text="prompt"))


def _model_config(
    *,
    base_url: str = "http://127.0.0.1:8080",
) -> BacktestAiConfiguratorModelRuntimeConfig:
    return BacktestAiConfiguratorModelRuntimeConfig(
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


def _request(*, prompt_text: str) -> BacktestConfigLLMRequest:
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
            user_prompt_text="Собери конфиг",
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
        catalog_subset_json={},
        output_schema_json={},
    )
