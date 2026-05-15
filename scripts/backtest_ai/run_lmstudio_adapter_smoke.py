from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import UUID

from jsonschema import Draft202012Validator

from trading.contexts.backtest.adapters.outbound import (
    LMStudioOpenAICompatibleAdapter,
    load_backtest_ai_configurator_runtime_config,
)
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigJob,
    BacktestConfigLLMRepairRequest,
    BacktestConfigLLMRequest,
    backtest_ai_model_output_schema,
    backtest_ai_prompt_profile_for_mode,
    backtest_ai_repair_prompt_profile,
)
from trading.shared_kernel.primitives import UserId


def main() -> None:
    args = _parse_args()
    runtime_config = load_backtest_ai_configurator_runtime_config(args.config)
    adapter = LMStudioOpenAICompatibleAdapter(config=runtime_config.model)
    schema = backtest_ai_model_output_schema()
    validator = Draft202012Validator(schema)

    generate_results = [
        _run_generate(adapter=adapter, schema_validator=validator, index=index)
        for index in range(1, args.attempts + 1)
    ]
    repair_results = [
        _run_repair(adapter=adapter, schema_validator=validator, index=index)
        for index in range(1, args.attempts + 1)
    ]
    generate_successes = sum(1 for item in generate_results if item["success"])
    repair_successes = sum(1 for item in repair_results if item["success"])
    accepted = generate_successes == args.attempts and repair_successes == args.attempts
    payload = {
        "accepted": accepted,
        "blocking_reason": None
        if accepted
        else "lmstudio_adapter_smoke_failed_valid_schema_json",
        "next_prompt_allowed": accepted,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config_path": str(args.config),
        "runtime": runtime_config.model.runtime,
        "base_url": _safe_base_url(runtime_config.model.base_url),
        "endpoint": "/v1/chat/completions",
        "model_id": runtime_config.model.model_id,
        "attempts_per_kind": args.attempts,
        "generate": {
            "successes": generate_successes,
            "failures": args.attempts - generate_successes,
            "results": generate_results,
        },
        "repair": {
            "successes": repair_successes,
            "failures": args.attempts - repair_successes,
            "results": repair_results,
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


def _run_generate(
    *,
    adapter: LMStudioOpenAICompatibleAdapter,
    schema_validator: Draft202012Validator,
    index: int,
) -> dict[str, Any]:
    profile = backtest_ai_prompt_profile_for_mode("create")
    request = BacktestConfigLLMRequest(
        job=_job(index=index, profile=profile.system_prompt_version),
        catalog=cast(Any, object()),
        prompt_profile=profile,
        prompt_text="adapter smoke generate",
        catalog_subset_json=_catalog_subset(),
        output_schema_json=backtest_ai_model_output_schema(),
    )
    return _run_call(
        kind="generate",
        index=index,
        schema_validator=schema_validator,
        call=lambda: adapter.generate_config(request).raw_output,
    )


def _run_repair(
    *,
    adapter: LMStudioOpenAICompatibleAdapter,
    schema_validator: Draft202012Validator,
    index: int,
) -> dict[str, Any]:
    profile = backtest_ai_repair_prompt_profile()
    request = BacktestConfigLLMRepairRequest(
        job=_job(index=index, profile=profile.system_prompt_version),
        catalog=cast(Any, object()),
        prompt_profile=profile,
        prompt_text="adapter smoke repair",
        catalog_subset_json=_catalog_subset(),
        output_schema_json=backtest_ai_model_output_schema(),
        failed_raw_output=(
            '{"schema_version":1,"mode":"create","status":"config_ready","config":{}}'
        ),
        parsed_draft_json={
            "schema_version": 1,
            "mode": "create",
            "status": "config_ready",
            "config": {},
        },
        validation_errors_json=(
            {
                "path": "config.coordinates",
                "code": "required",
                "message": "required date range is missing; ask for clarification",
            },
        ),
    )
    return _run_call(
        kind="repair",
        index=index,
        schema_validator=schema_validator,
        call=lambda: adapter.repair_config(request).raw_output,
    )


def _run_call(
    *,
    kind: str,
    index: int,
    schema_validator: Draft202012Validator,
    call: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        raw_output = call()
        parsed = json.loads(raw_output)
        if not isinstance(parsed, dict):
            raise ValueError("model content must be a JSON object")
        errors = tuple(schema_validator.iter_errors(parsed))
        if errors:
            first = errors[0]
            path = ".".join(str(part) for part in first.absolute_path) or "body"
            raise ValueError(f"schema validation failed at {path}: {first.message}")
        return {
            "kind": kind,
            "attempt": index,
            "success": True,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "status": parsed.get("status"),
            "mode": parsed.get("mode"),
        }
    except Exception as error:  # noqa: BLE001
        return {
            "kind": kind,
            "attempt": index,
            "success": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error_type": type(error).__name__,
            "error": _sanitize_error(str(error)),
        }


def _job(*, index: int, profile: str) -> BacktestAiConfigJob:
    now = datetime.now(UTC)
    return BacktestAiConfigJob(
        job_id=UUID(f"00000000-0000-0000-0000-{index:012d}"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000602"),
        mode="create",
        locale="ru",
        state="running",
        source_page="backtests",
        user_prompt_text=(
            "Собери JSON конфигурацию для BTCUSDT 15m на Binance spot, но период "
            "не указан. Если обязательных данных не хватает, верни "
            "status=needs_clarification, config={} и короткий вопрос."
        ),
        user_prompt_hash="a" * 64,
        system_prompt_version=profile,
        system_prompt_hash="b" * 64,
        catalog_snapshot_hash="c" * 64,
        runtime_defaults_hash="d" * 64,
        queued_at=now,
        updated_at=now,
    )


def _catalog_subset() -> Mapping[str, Any]:
    return {
        "exchanges": ["binance"],
        "market_types": ["spot"],
        "symbols": ["BTCUSDT"],
        "timeframes": ["15m"],
        "risk_modes": ["none"],
        "direction_modes": ["long_only"],
        "sizing_modes": ["fixed_notional"],
        "ranking_metrics": ["total_return_pct"],
        "ranking_default": {"primary_metric": "total_return_pct", "direction": "desc"},
        "top_n_default": 5,
        "execution_defaults": {},
        "hit_times_grid": {},
        "indicators": [
            {
                "indicator_id": "ma.sma",
                "sources": ["close"],
                "param_specs": {"window": {"start": 10, "stop": 20, "step": 5}},
            }
        ],
    }


def _safe_base_url(value: str) -> str:
    return value.replace("127.0.0.1", "127.0.0.1")


def _sanitize_error(value: str) -> str:
    return value.replace("/Users/daniildegtyarev", "/Users/<redacted>")[:500]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/prod/backtest_ai_configurator.yaml"),
    )
    parser.add_argument("--attempts", type=int, default=10)
    args = parser.parse_args()
    if args.attempts <= 0:
        parser.error("--attempts must be positive")
    return args


if __name__ == "__main__":
    main()
