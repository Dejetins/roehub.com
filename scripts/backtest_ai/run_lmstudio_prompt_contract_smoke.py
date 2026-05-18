#!/usr/bin/env python3
from __future__ import annotations

# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.backtest.adapters.outbound import (
    LMStudioChatCompletionsSettings,
    LMStudioOpenAICompatibleAdapter,
    load_backtest_ai_configurator_runtime_config,
)
from trading.contexts.backtest.application.ai_configurator import (
    BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH,
    BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION,
    BacktestAiConfigJob,
    build_backtest_ai_prompt_package,
)
from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigAgentRequest,
)
from trading.contexts.backtest.application.ai_configurator.schema import (
    backtest_ai_model_output_schema,
    backtest_ai_output_example,
)
from trading.contexts.backtest.application.ai_configurator.services.catalog import (
    BacktestAiAllowedCatalog,
    BacktestAiIndicatorCatalogItem,
)
from trading.shared_kernel.primitives import UserId

DEFAULT_CONFIG_PATH = Path("configs/prod/backtest_ai_configurator.yaml")
DEFAULT_ARTIFACT_PATH = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-05-17_ai_configurator_assistant_v1/"
    "iteration_04_lmstudio_smoke_latest.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke Iteration 04 LM Studio prompt/schema contract."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--direct-count", type=int, default=10)
    parser.add_argument("--generate-count", type=int, default=10)
    parser.add_argument("--repair-count", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    try:
        payload = _run_smoke(
            config_path=args.config,
            direct_count=args.direct_count,
            generate_count=args.generate_count,
            repair_count=args.repair_count,
        )
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


def _run_smoke(
    *,
    config_path: Path,
    direct_count: int,
    generate_count: int,
    repair_count: int,
) -> dict[str, Any]:
    if min(direct_count, generate_count, repair_count) <= 0:
        raise ValueError("smoke counts must be positive")
    runtime_config = load_backtest_ai_configurator_runtime_config(config_path)
    if runtime_config.model.runtime != "lm_studio":
        raise ValueError("runtime must be lm_studio for Iteration 04 smoke")
    adapter = LMStudioOpenAICompatibleAdapter(
        settings=LMStudioChatCompletionsSettings.from_runtime_config(
            runtime_config.model
        )
    )
    schema = backtest_ai_model_output_schema()
    validator = Draft202012Validator(schema)
    catalog = _sample_catalog()
    prompt_package = build_backtest_ai_prompt_package(
        trusted_context={
            "context_schema_version": 1,
            "allowed_values": {
                "exchanges": ["binance"],
                "markets": ["spot"],
                "symbol": "BTCUSDT",
                "symbol_candidates": ["BTCUSDT"],
                "timeframes": ["15m"],
                "indicators": [
                    {
                        "indicator_id": "momentum.rsi",
                        "sources": ["close"],
                        "params": {"window": {"mode": "explicit", "values": [14]}},
                    }
                ],
                "risk_modes": ["none"],
            },
        },
        current_form_config=None,
        recent_chat_context=(),
        user_message=(
            "Create a valid /backtests BTCUSDT configuration on 15m with "
            "indicator_id momentum.rsi. Do not run a backtest."
        ),
    )
    direct = _run_direct_structured(
        adapter=adapter,
        package=prompt_package,
        validator=validator,
        count=direct_count,
    )
    generate = _run_generate(
        adapter=adapter,
        catalog=catalog,
        validator=validator,
        count=generate_count,
    )
    repair = _run_repair(
        adapter=adapter,
        package=prompt_package,
        validator=validator,
        count=repair_count,
    )
    accepted = (
        direct["passed"] == direct_count
        and generate["passed"] == generate_count
        and repair["passed"] == repair_count
    )
    return {
        "accepted": accepted,
        "blocking_reason": None if accepted else "lmstudio_prompt_contract_smoke_failed",
        "next_iteration_allowed": accepted,
        "timestamp_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config_path": str(config_path),
        "runtime": runtime_config.model.runtime,
        "model_id": runtime_config.model.model_id,
        "endpoint": "POST /v1/chat/completions",
        "response_format": "json_schema",
        "raw_outputs_redacted": True,
        "direct_structured": direct,
        "adapter_generate": generate,
        "adapter_repair": repair,
    }


def _run_direct_structured(
    *,
    adapter: LMStudioOpenAICompatibleAdapter,
    package: Any,
    validator: Draft202012Validator,
    count: int,
) -> dict[str, Any]:
    return _run_loop(
        count=count,
        callback=lambda: adapter.complete_prompt_package(package=package).content,
        validator=validator,
    )


def _run_generate(
    *,
    adapter: LMStudioOpenAICompatibleAdapter,
    catalog: BacktestAiAllowedCatalog,
    validator: Draft202012Validator,
    count: int,
) -> dict[str, Any]:
    request = BacktestConfigAgentRequest(job=_sample_job(), catalog=catalog)

    def _callback() -> str:
        response = adapter.run_config_session(request)
        if response.raw_output is None:
            raise RuntimeError(response.audit_json)
        return response.raw_output

    return _run_loop(count=count, callback=_callback, validator=validator)


def _run_repair(
    *,
    adapter: LMStudioOpenAICompatibleAdapter,
    package: Any,
    validator: Draft202012Validator,
    count: int,
) -> dict[str, Any]:
    previous_draft = backtest_ai_output_example()
    previous_config = previous_draft["config"]
    if not isinstance(previous_config, dict):
        raise TypeError("smoke output example config must be object")
    previous_config["top_n"] = 0
    validation_errors = (
        {
            "path": "config.top_n",
            "code": "minimum",
            "message": "top_n must be greater than or equal to 1",
            "repair_value": 10,
        },
    )
    return _run_loop(
        count=count,
        callback=lambda: adapter.run_repair_session(
            package=package,
            previous_draft=previous_draft,
            validation_errors=validation_errors,
        ).content,
        validator=validator,
    )


def _run_loop(
    *,
    count: int,
    callback: Any,
    validator: Draft202012Validator,
) -> dict[str, Any]:
    passed = 0
    failures: list[dict[str, Any]] = []
    content_hashes: list[str] = []
    for index in range(count):
        try:
            content = callback()
            parsed = _parse_schema_compatible(content=content, validator=validator)
        except Exception as error:  # noqa: BLE001
            failures.append({"index": index, "error": str(error)[:500]})
            continue
        passed += 1
        content_hashes.append(
            hashlib.sha256(
                json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        )
    return {
        "passed": passed,
        "total": count,
        "failed": count - passed,
        "content_hashes": content_hashes,
        "failures": failures[:3],
    }


def _parse_schema_compatible(
    *,
    content: str,
    validator: Draft202012Validator,
) -> Mapping[str, Any]:
    parsed = json.loads(content)
    if not isinstance(parsed, Mapping):
        raise ValueError("model content must be a JSON object")
    errors = sorted(validator.iter_errors(parsed), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        path = ".".join(str(part) for part in first.absolute_path) or "body"
        raise ValueError(f"schema validation failed at {path}: {first.message}")
    return parsed


def _sample_catalog() -> BacktestAiAllowedCatalog:
    return BacktestAiAllowedCatalog(
        exchanges=("binance",),
        market_types=("spot",),
        symbols=("BTCUSDT",),
        timeframes=("15m",),
        risk_modes=("none", "tp_sl_grid"),
        direction_modes=("long_short_reversal",),
        sizing_modes=("fixed_equity_pct",),
        ranking_metrics=("total_return_pct",),
        ranking_default={"primary_metric": "total_return_pct", "direction": "desc"},
        top_n_default=10,
        guardrails={},
        execution_defaults={
            "direction_mode": "long_short_reversal",
            "fee_rate": 0.00075,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        hit_times_grid={},
        indicators=(
            BacktestAiIndicatorCatalogItem(
                indicator_id="momentum.rsi",
                sources=("close",),
                param_specs={"params": {"window": {"mode": "explicit", "values": [14]}}},
            ),
        ),
        artifact_capabilities={},
        source_paths=("smoke-sample",),
    )


def _sample_job() -> BacktestAiConfigJob:
    now = datetime.now(UTC)
    return BacktestAiConfigJob(
        job_id=UUID("00000000-0000-0000-0000-000000000804"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000805"),
        mode="assistant_v1",
        locale="en",
        state="running",
        source_page="backtests",
        user_prompt_text=(
            "Create a valid /backtests BTCUSDT configuration on 15m with "
            "indicator_id momentum.rsi. Do not run a backtest."
        ),
        user_prompt_hash="a" * 64,
        system_prompt_version=BACKTEST_AI_CONFIG_AGENT_CONTRACT_VERSION,
        system_prompt_hash=BACKTEST_AI_CONFIG_AGENT_CONTRACT_HASH,
        catalog_snapshot_hash="b" * 64,
        runtime_defaults_hash="c" * 64,
        queued_at=now,
        updated_at=now,
    )


def _write_artifact(payload: Mapping[str, Any], *, artifact: Path) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit(payload: Mapping[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, sort_keys=True))
        return
    print(
        "accepted={accepted} direct={direct} generate={generate} repair={repair} "
        "blocking_reason={reason}".format(
            accepted=payload.get("accepted"),
            direct=payload.get("direct_structured", {}).get("passed"),
            generate=payload.get("adapter_generate", {}).get("passed"),
            repair=payload.get("adapter_repair", {}).get("passed"),
            reason=payload.get("blocking_reason"),
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
