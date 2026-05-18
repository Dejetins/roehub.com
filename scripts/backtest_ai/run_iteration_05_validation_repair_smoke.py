#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.backtest.adapters.outbound import (
    LMStudioChatCompletionsSettings,
    LMStudioOpenAICompatibleAdapter,
    YamlBacktestGridDefaultsProvider,
    load_backtest_ai_configurator_runtime_config,
)
from trading.contexts.backtest.application.ai_configurator import (
    BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH,
    BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION,
    BacktestAiCatalogResolver,
    BacktestAiConfigJob,
    BacktestAiConfigPipeline,
    BacktestAiConfigValidator,
    BacktestAiInputGate,
    BacktestAiOutputGate,
)
from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigAgentRepairRequest,
)
from trading.contexts.backtest.application.dto.runtime_preflight import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightService,
    BacktestRuntimeConfig,
    BacktestRuntimeDefaultsService,
)
from trading.shared_kernel.primitives import UserId

DEFAULT_CONFIG_PATH = Path("configs/prod/backtest_ai_configurator.yaml")
DEFAULT_ARTIFACT_PATH = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-05-17_ai_configurator_assistant_v1/"
    "iteration_05_validation_repair_smoke_latest.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke Iteration 05 validation, repair, and load-action gate."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    try:
        payload = _run_smoke(config_path=args.config)
    except Exception as error:  # noqa: BLE001
        payload = {
            "accepted": False,
            "blocking_reason": str(error),
            "next_iteration_allowed": False,
            "duration_seconds": round(time.time() - started, 3),
        }
        _emit(payload, json_output=args.json)
        return 1

    payload["duration_seconds"] = round(time.time() - started, 3)
    _write_artifact(payload, artifact=args.artifact)
    _emit(payload, json_output=args.json)
    return 0 if payload["accepted"] else 1


def _run_smoke(*, config_path: Path) -> dict[str, Any]:
    runtime_config = load_backtest_ai_configurator_runtime_config(config_path)
    adapter = LMStudioOpenAICompatibleAdapter(
        settings=LMStudioChatCompletionsSettings.from_runtime_config(
            runtime_config.model
        )
    )
    pipeline = _pipeline(adapter=adapter)

    supported = pipeline.run(
        job=_sample_job(
            message=(
                "Create a valid /backtests BTCUSDT configuration on 15m with "
                "momentum.rsi. Do not run a backtest."
            )
        )
    )
    catalog = pipeline.catalog_resolver.resolve()
    repair_job = _sample_job(message="Create BTCUSDT RSI config with a valid top_n.")
    previous_draft = _invalid_top_n_draft(config=catalog.default_config())
    repair_response = adapter.run_repair_config_session(
        BacktestConfigAgentRepairRequest(
            job=repair_job,
            catalog=catalog,
            previous_draft=previous_draft,
            validation_errors=(
                {
                    "path": "config.top_n",
                    "code": "minimum",
                    "message": "top_n must be greater than or equal to 1",
                    "repair_value": 10,
                },
            ),
        )
    )
    repair_outcome = (
        pipeline.validator.validate_model_output(
            raw_output=repair_response.raw_output,
            catalog=catalog,
        )
        if repair_response.raw_output is not None
        else None
    )
    accepted = (
        supported.status == "ready"
        and supported.validated_config is not None
        and _load_action_enabled(status=supported.status, config=supported.validated_config)
        and all(attempt.attempt_kind != "repair" for attempt in supported.llm_attempts)
        and repair_outcome is not None
        and repair_outcome.status == "ready"
        and repair_outcome.validated_config is not None
    )
    return {
        "accepted": accepted,
        "blocking_reason": None if accepted else "iteration_05_validation_smoke_failed",
        "next_iteration_allowed": accepted,
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "runtime": runtime_config.model.runtime,
        "model_id": runtime_config.model.model_id,
        "endpoint": "POST /v1/chat/completions",
        "repair_attempts": 1,
        "auto_run_backtest_attempt": False,
        "load_action": {
            "enabled": _load_action_enabled(
                status=supported.status,
                config=supported.validated_config,
            ),
            "source": "backend_ready_validated_config_only",
        },
        "supported_prompt": {
            "status": supported.status,
            "validated_config_present": supported.validated_config is not None,
            "attempt_kinds": [attempt.attempt_kind for attempt in supported.llm_attempts],
            "validation_errors": list(supported.validation_errors),
        },
        "repair_probe": {
            "status": None if repair_outcome is None else repair_outcome.status,
            "attempt_kind": "repair",
            "same_runtime": "lm_studio",
            "validation_errors": []
            if repair_outcome is None
            else list(repair_outcome.validation_errors),
        },
    }


def _pipeline(*, adapter: LMStudioOpenAICompatibleAdapter) -> BacktestAiConfigPipeline:
    defaults_provider = YamlBacktestGridDefaultsProvider.from_yaml(
        config_path="configs/prod/indicators.yaml"
    )
    runtime_config = BacktestRuntimeConfig(
        hit_times_tp_levels_pct=tuple(i / 2 for i in range(1, 101)),
        hit_times_sl_levels_pct=tuple(i / 2 for i in range(1, 51)),
        artifact_config_hash="a" * 64,
    )
    runtime_defaults_service = BacktestRuntimeDefaultsService(
        defaults_provider=defaults_provider,
        runtime_config=runtime_config,
    )
    return BacktestAiConfigPipeline(
        catalog_resolver=BacktestAiCatalogResolver(
            runtime_defaults_service=runtime_defaults_service,
            supported_symbols=("BTCUSDT",),
        ),
        validator=BacktestAiConfigValidator(
            preflight_service=BacktestPreflightService(
                defaults_provider=defaults_provider,
                artifact_context_resolver=_FakeArtifactResolver(),
                runtime_config=runtime_config,
            ),
            output_gate=BacktestAiOutputGate(),
        ),
        input_gate=BacktestAiInputGate(),
        agent_gateway=adapter,
    )


def _sample_job(*, message: str) -> BacktestAiConfigJob:
    now = datetime.now(UTC)
    return BacktestAiConfigJob(
        job_id=UUID("00000000-0000-0000-0000-000000000505"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000506"),
        mode="assistant_v1",
        locale="en",
        state="running",
        source_page="backtests",
        user_prompt_text=message,
        user_prompt_hash="a" * 64,
        current_config_json=None,
        system_prompt_version=BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION,
        system_prompt_hash=BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH,
        catalog_snapshot_hash="b" * 64,
        runtime_defaults_hash="c" * 64,
        queued_at=now,
        updated_at=now,
    )


def _load_action_enabled(*, status: str, config: Mapping[str, Any] | None) -> bool:
    return status == "ready" and isinstance(config, Mapping)


def _invalid_top_n_draft(*, config: dict[str, Any]) -> dict[str, Any]:
    draft = {
        "schema_version": 1,
        "intent": "create_config",
        "status": "config_ready",
        "assistant_message": (
            "I prepared a BTCUSDT configuration, but the validator should repair top_n."
        ),
        "conversation_title": "RSI for BTCUSDT",
        "config": config,
        "unsupported_items": [],
        "clarifying_questions": [],
        "warnings": [],
    }
    draft["config"]["top_n"] = 0
    return draft


class _FakeArtifactResolver:
    def resolve_context(self, *, coordinates: BacktestCoordinates) -> BacktestArtifactMetadata:
        _ = coordinates
        return BacktestArtifactMetadata(
            artifact_slot="slot_a",
            artifact_slot_generation=1,
            artifact_manifest_hash="a" * 64,
            artifact_asof_date="2026-05-18",
            hit_times_manifest_hash="b" * 64,
            published_at_utc="2026-05-18T00:00:00Z",
        )


def _write_artifact(payload: Mapping[str, Any], *, artifact: Path) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _emit(payload: Mapping[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print(
        "accepted={accepted} blocking_reason={blocking_reason} "
        "supported_status={supported_status}".format(
            accepted=payload.get("accepted"),
            blocking_reason=payload.get("blocking_reason"),
            supported_status=(payload.get("supported_prompt") or {}).get("status")
            if isinstance(payload.get("supported_prompt"), Mapping)
            else None,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
