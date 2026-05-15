from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from trading.contexts.backtest.adapters.outbound import (
    BacktestAiConfiguratorModelRuntimeConfig,
)


def _load_runtime_module() -> ModuleType:
    module_path = Path("scripts/macos/lmstudio_backtest_ai_runtime.py")
    spec = importlib.util.spec_from_file_location(
        "lmstudio_backtest_ai_runtime",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lmstudio_runtime_target_uses_configured_loopback_port() -> None:
    module = _load_runtime_module()
    config = BacktestAiConfiguratorModelRuntimeConfig(
        runtime="lm_studio",
        model_id="gemma-4-e2b-it-4bit",
        model_path=Path("/Users/daniildegtyarev/.lmstudio/models/gemma-4-e2b-it-4bit"),
        context_window_tokens=8192,
        max_input_tokens=6144,
        max_output_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        base_url="http://localhost:18080",
        request_timeout_seconds=240,
        active_generations=1,
    )

    target = module._target_from_model_config(config, model_key="gemma-4-e2b-it")

    assert target.host == "127.0.0.1"
    assert target.port == 18080
    assert target.base_url == "http://127.0.0.1:18080"
    assert target.model_identifier == "gemma-4-e2b-it-4bit"
    assert target.context_length == 8192
    assert target.parallel == 1


def test_lmstudio_runtime_target_rejects_base_url_without_explicit_port() -> None:
    module = _load_runtime_module()
    config = BacktestAiConfiguratorModelRuntimeConfig(
        runtime="lm_studio",
        model_id="gemma-4-e2b-it-4bit",
        model_path=Path("/Users/daniildegtyarev/.lmstudio/models/gemma-4-e2b-it-4bit"),
        context_window_tokens=8192,
        max_input_tokens=6144,
        max_output_tokens=1024,
        temperature=0.2,
        top_p=0.9,
        base_url="http://127.0.0.1",
        request_timeout_seconds=240,
        active_generations=1,
    )

    with pytest.raises(module.RuntimeCheckError, match="explicit port"):
        module._target_from_model_config(config, model_key="gemma-4-e2b-it")


def test_lmstudio_loaded_model_checks_identifier_context_and_parallel() -> None:
    module = _load_runtime_module()
    target = module.RuntimeTarget(
        base_url="http://127.0.0.1:8080",
        host="127.0.0.1",
        port=8080,
        model_key="gemma-4-e2b-it",
        model_identifier="gemma-4-e2b-it-4bit",
        context_length=8192,
        parallel=1,
    )

    assert module._model_loaded(
        [
            {
                "identifier": "gemma-4-e2b-it-4bit",
                "contextLength": 8192,
                "parallel": 1,
            }
        ],
        target,
    )
    assert not module._model_loaded(
        [
            {
                "identifier": "gemma-4-e2b-it-4bit",
                "contextLength": 4096,
                "parallel": 1,
            }
        ],
        target,
    )


def test_lmstudio_api_models_requires_loaded_instance() -> None:
    module = _load_runtime_module()

    assert module._api_models_has_loaded_instance(
        {
            "data": [
                {
                    "model_key": "gemma-4-e2b-it",
                    "loaded_instances": [{"id": "gemma-4-e2b-it-4bit"}],
                }
            ]
        },
        "gemma-4-e2b-it-4bit",
    )
    assert not module._api_models_has_loaded_instance(
        {"data": [{"model_key": "gemma-4-e2b-it", "loaded_instances": []}]},
        "gemma-4-e2b-it-4bit",
    )


def test_lmstudio_schema_type_values_must_be_strings() -> None:
    module = _load_runtime_module()

    with pytest.raises(module.RuntimeCheckError, match="type values must be strings"):
        module._assert_schema_type_values_are_strings(
            {"type": "object", "properties": {"x": {"type": ["string", "null"]}}}
        )
