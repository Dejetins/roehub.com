from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound import (
    load_backtest_ai_configurator_runtime_config,
)


def test_ai_configurator_runtime_config_loads_lmstudio_model_registry() -> None:
    config = load_backtest_ai_configurator_runtime_config(
        Path("configs/prod/backtest_ai_configurator.yaml")
    )

    assert config.model.runtime == "lm_studio_tools"
    assert config.model.model_id == "gemma-4-e2b-it-4bit"
    assert config.model.model_path.name == "gemma-4-e2b-it-4bit"
    assert config.model.base_url == "http://127.0.0.1:8080"
    assert config.model.active_generations == 1
    assert config.model.max_output_tokens == 1024
    assert config.model.request_timeout_seconds == 240
    assert config.queue.lease_seconds == 300


def test_ai_configurator_runtime_config_rejects_public_lmstudio_base_url(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "backtest_ai_configurator.yaml"
    config_path.write_text(
        """
version: 1
backtest_ai_configurator:
  enabled: false
  queue:
    max_queue_size: 1
    lease_seconds: 120
    job_timeout_seconds: 90
    repair_attempts: 1
    max_active_generations: 1
    request_timeout_sec: 90
    queue_timeout_sec: 180
  model:
    runtime: lm_studio_tools
    model_id: gemma-4-e2b-it-4bit
    model_path: /tmp/gemma-4-e2b-it-4bit
    context_window_tokens: 8192
    max_input_tokens: 6144
    max_output_tokens: 1024
    temperature: 0.2
    top_p: 0.9
    base_url: http://192.168.1.10:8080
    request_timeout_seconds: 90
    active_generations: 1
  quotas:
    free:
      requests_per_5h: 1
      requests_per_week: 1
      max_queued_per_user: 1
      max_active_user_jobs: 1
    base:
      requests_per_5h: 1
      requests_per_week: 1
      max_queued_per_user: 1
      max_active_user_jobs: 1
    pro:
      requests_per_5h: 1
      requests_per_week: 1
      max_queued_per_user: 1
      max_active_user_jobs: 1
    ultra:
      requests_per_5h: 1
      requests_per_week: 1
      max_queued_per_user: 1
      max_active_user_jobs: 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="loopback-only"):
        load_backtest_ai_configurator_runtime_config(config_path)


def test_ai_configurator_runtime_config_rejects_stale_runtime(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "backtest_ai_configurator.yaml"
    config_path.write_text(
        """
version: 1
backtest_ai_configurator:
  enabled: false
  queue:
    max_queue_size: 1
    lease_seconds: 120
    job_timeout_seconds: 90
    repair_attempts: 1
    max_active_generations: 1
    request_timeout_sec: 90
    queue_timeout_sec: 180
  model:
    runtime: unsupported_local_runtime
    model_id: gemma-4-e2b-it-4bit
    model_path: /tmp/gemma-4-e2b-it-4bit
    context_window_tokens: 8192
    max_input_tokens: 6144
    max_output_tokens: 1024
    temperature: 0.2
    top_p: 0.9
    base_url: http://127.0.0.1:8080
    request_timeout_seconds: 90
    active_generations: 1
  quotas:
    free:
      requests_per_5h: 1
      requests_per_week: 1
      max_queued_per_user: 1
      max_active_user_jobs: 1
    base:
      requests_per_5h: 1
      requests_per_week: 1
      max_queued_per_user: 1
      max_active_user_jobs: 1
    pro:
      requests_per_5h: 1
      requests_per_week: 1
      max_queued_per_user: 1
      max_active_user_jobs: 1
    ultra:
      requests_per_5h: 1
      requests_per_week: 1
      max_queued_per_user: 1
      max_active_user_jobs: 1
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="runtime.*lm_studio_tools"):
        load_backtest_ai_configurator_runtime_config(config_path)
