from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from trading.contexts.backtest.adapters.outbound.ai_config_agent import (
    LMStudioChatCompletionsResult,
    LMStudioChatCompletionsSettings,
    LMStudioOpenAICompatibleAdapter,
)
from trading.contexts.backtest.application.ai_configurator import (
    BACKTEST_AI_OUTPUT_SCHEMA_NAME,
    CANONICAL_SYSTEM_PROMPT,
    SYSTEM_PROMPT_ID,
    backtest_ai_lmstudio_response_format,
    backtest_ai_model_output_schema,
    backtest_ai_output_example,
    build_backtest_ai_prompt_package,
)


def test_canonical_system_prompt_has_machine_readable_scope_and_literals() -> None:
    assert SYSTEM_PROMPT_ID == "backtest_ai_configurator_assistant_v1"
    assert "SYSTEM_PROMPT_ID: backtest_ai_configurator_assistant_v1" in (
        CANONICAL_SYSTEM_PROMPT
    )
    assert "TRUSTED_CONTEXT_JSON" in CANONICAL_SYSTEM_PROMPT
    assert "CURRENT_FORM_CONFIG_JSON" in CANONICAL_SYSTEM_PROMPT
    assert "RECENT_CHAT_CONTEXT_JSON" in CANONICAL_SYSTEM_PROMPT
    assert "OUTPUT_JSON_SCHEMA" in CANONICAL_SYSTEM_PROMPT
    assert "never run backtests" in CANONICAL_SYSTEM_PROMPT
    assert "never access files, tools, APIs, terminals" in CANONICAL_SYSTEM_PROMPT
    assert "Produce a config for exactly one symbol" in CANONICAL_SYSTEM_PROMPT
    assert "function calling" in CANONICAL_SYSTEM_PROMPT
    assert CANONICAL_SYSTEM_PROMPT.isascii()


def test_output_schema_is_strict_has_title_and_no_nullable_type_arrays() -> None:
    schema = backtest_ai_model_output_schema()

    assert schema["additionalProperties"] is False
    assert "conversation_title" in schema["required"]
    assert schema["properties"]["conversation_title"]["maxLength"] == 60
    assert "intent" in schema["required"]
    assert "unsupported_items" in schema["required"]
    assert not _contains_type_array(schema)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(backtest_ai_output_example())


def test_lmstudio_response_format_uses_json_schema_contract() -> None:
    response_format = backtest_ai_lmstudio_response_format()

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == BACKTEST_AI_OUTPUT_SCHEMA_NAME
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert "conversation_title" in schema["required"]
    assert schema["additionalProperties"] is False
    assert not _contains_type_array(schema)


def test_prompt_package_separates_trusted_context_current_form_recent_chat_and_user() -> None:
    package = build_backtest_ai_prompt_package(
        trusted_context={
            "context_schema_version": 1,
            "allowed_values": {"symbol": "BTCUSDT"},
        },
        current_form_config={"coordinates": {"symbol": "BTCUSDT"}},
        recent_chat_context=[{"role": "assistant", "content": "Earlier answer"}],
        user_message="Create RSI for BTCUSDT",
    )

    user_content = package.user_message.content
    assert package.system_message.content == CANONICAL_SYSTEM_PROMPT
    assert "TRUSTED_CONTEXT_JSON" in user_content
    assert "CURRENT_FORM_CONFIG_JSON" in user_content
    assert "RECENT_CHAT_CONTEXT_JSON" in user_content
    assert "USER_MESSAGE" in user_content
    assert "OUTPUT_JSON_SCHEMA" in user_content
    assert "OUTPUT_JSON_EXAMPLE" in user_content


def test_lmstudio_adapter_payload_uses_chat_completions_without_tools() -> None:
    package = build_backtest_ai_prompt_package(
        trusted_context={
            "context_schema_version": 1,
            "allowed_values": {"symbol": "BTCUSDT"},
        },
        current_form_config=None,
        recent_chat_context=(),
        user_message="Create RSI for BTCUSDT",
    )
    adapter = LMStudioOpenAICompatibleAdapter(
        settings=LMStudioChatCompletionsSettings(
            base_url="http://127.0.0.1:8080",
            model_id="gemma-4-e2b-it-4bit",
            request_timeout_seconds=90,
            max_output_tokens=1024,
        )
    )

    payload = adapter.build_payload(package=package)

    assert payload["model"] == "gemma-4-e2b-it-4bit"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
    assert payload["response_format"]["type"] == "json_schema"
    assert "tools" not in payload
    assert "tool_choice" not in payload


def test_lmstudio_repair_prompt_applies_explicit_repair_value(monkeypatch: Any) -> None:
    package = build_backtest_ai_prompt_package(
        trusted_context={
            "context_schema_version": 1,
            "allowed_values": {"symbol": "BTCUSDT"},
        },
        current_form_config=None,
        recent_chat_context=(),
        user_message="Create RSI for BTCUSDT",
    )
    adapter = LMStudioOpenAICompatibleAdapter(
        settings=LMStudioChatCompletionsSettings(
            base_url="http://127.0.0.1:8080",
            model_id="gemma-4-e2b-it-4bit",
            request_timeout_seconds=90,
            max_output_tokens=1024,
        )
    )
    captured: dict[str, Any] = {}

    def _fake_complete(
        self: LMStudioOpenAICompatibleAdapter,
        *,
        package: Any,
    ) -> LMStudioChatCompletionsResult:
        captured["content"] = package.user_message.content
        return LMStudioChatCompletionsResult(
            content="{}",
            model_id=self.settings.model_id,
            finish_reason="stop",
            latency_ms=1,
            audit_json={},
        )

    monkeypatch.setattr(
        LMStudioOpenAICompatibleAdapter,
        "complete_prompt_package",
        _fake_complete,
    )

    adapter.run_repair_session(
        package=package,
        previous_draft={"schema_version": 1, "config": {"top_n": 0}},
        validation_errors=(
            {
                "path": "config.top_n",
                "code": "minimum",
                "message": "top_n must be greater than or equal to 1",
                "repair_value": 10,
            },
        ),
    )

    assert "REPAIR_INSTRUCTION" in captured["content"]
    assert "If a validation error includes a repair_value" in captured["content"]
    assert '"repair_value": 10' in captured["content"]


def _contains_type_array(value: Any) -> bool:
    if isinstance(value, dict):
        current_type = value.get("type")
        if isinstance(current_type, list):
            return True
        return any(_contains_type_array(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_type_array(item) for item in value)
    return False
