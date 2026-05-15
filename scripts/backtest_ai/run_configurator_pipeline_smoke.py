from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.backtest_ai.configurator_benchmark_common import (
    PIPELINE_READY_PROMPT_CASES,
    PIPELINE_REPAIR_PROMPT_CASES,
    FakeWorkerAiConfigClient,
    HttpAiConfigClient,
    PromptCase,
    benchmark_identity,
    local_host_identity,
    markdown_table,
    parse_header_values,
    write_json,
)

DEFAULT_OUTPUT_DIR = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-05-13_lmstudio_serving_recovery"
)
DEFAULT_CONFIG_PATH = Path("configs/prod/backtest_ai_configurator.yaml")

UNSUPPORTED_PROMPT_CASES: tuple[PromptCase, ...] = (
    PromptCase(
        case_id="unsupported_off_topic_ru",
        mode="create",
        locale="ru",
        message="Напиши письмо инвесторам про маркетинг.",
        category="off_topic",
        supported=False,
        expected_statuses=("blocked_by_policy", "needs_clarification"),
    ),
    PromptCase(
        case_id="unsupported_doge_bollinger_en",
        mode="create",
        locale="en",
        message="Create DOGEUSDT 1h Bollinger config for /backtests.",
        category="unsupported_catalog",
        supported=False,
        expected_statuses=("needs_clarification", "blocked_by_policy"),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a real /backtests AI configurator readiness smoke."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--user-id-header", default="x-user-id")
    parser.add_argument("--user-id-prefix", default="pipeline-ai-config")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-name", default="pipeline_smoke_results.json")
    parser.add_argument("--markdown-name", default="pipeline_smoke_summary.md")
    parser.add_argument(
        "--strict-acceptance-exit-code",
        action="store_true",
        help="Exit non-zero when accepted is false.",
    )
    parser.add_argument(
        "--fake-worker",
        action="store_true",
        help="Use in-process deterministic API + fake worker; developer smoke only.",
    )
    return parser


async def run_async(args: argparse.Namespace) -> int:
    headers = parse_header_values(args.header)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    client = (
        FakeWorkerAiConfigClient(timeout_seconds=args.http_timeout_seconds)
        if args.fake_worker
        else HttpAiConfigClient(
            base_url=args.base_url,
            headers=headers,
            user_id_header=args.user_id_header,
            user_id_prefix=args.user_id_prefix,
            timeout_seconds=args.http_timeout_seconds,
        )
    )
    ready_observations = []
    repair_observations = []
    unsupported_observations = []
    try:
        ready_observations = await _run_cases(
            client=client,
            scenario=f"pipeline supported ready smoke {run_id}",
            cases=PIPELINE_READY_PROMPT_CASES,
            user_offset=0,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.job_timeout_seconds,
        )
        repair_observations = await _run_cases(
            client=client,
            scenario=f"pipeline repair smoke {run_id}",
            cases=PIPELINE_REPAIR_PROMPT_CASES,
            user_offset=20_000,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.job_timeout_seconds,
        )
        unsupported_observations = await _run_cases(
            client=client,
            scenario=f"pipeline unsupported prompt smoke {run_id}",
            cases=UNSUPPORTED_PROMPT_CASES,
            user_offset=30_000,
            poll_interval_seconds=args.poll_interval_seconds,
            timeout_seconds=args.job_timeout_seconds,
        )
    finally:
        await client.aclose()

    summary = _summary(
        ready_observations=ready_observations,
        repair_observations=repair_observations,
        unsupported_observations=unsupported_observations,
    )
    blockers = _blockers(fake_worker=args.fake_worker, summary=summary)
    accepted = not blockers
    payload = {
        "kind": "backtest_ai_configurator_pipeline_smoke",
        "run_id": run_id,
        "accepted": accepted,
        "blocking_reason": "; ".join(blockers) if blockers else None,
        "next_prompt_allowed": accepted,
        "acceptance_classification": "developer_smoke"
        if args.fake_worker
        else "macstudio_acceptance_candidate",
        "load_generator_host": local_host_identity(),
        "identity": benchmark_identity(config_path=args.config_path),
        "supported_ready_case_ids": [case.case_id for case in PIPELINE_READY_PROMPT_CASES],
        "repair_case_ids": [case.case_id for case in PIPELINE_REPAIR_PROMPT_CASES],
        "unsupported_case_ids": [case.case_id for case in UNSUPPORTED_PROMPT_CASES],
        "supported_ready_observations": [item.as_mapping() for item in ready_observations],
        "repair_observations": [item.as_mapping() for item in repair_observations],
        "unsupported_observations": [item.as_mapping() for item in unsupported_observations],
        "summary": summary,
        "rollout_decision": {
            "accepted": accepted,
            "reason": "accepted pipeline readiness smoke" if accepted else "rollout blocked",
            "blockers": blockers,
            "blocking_reason": "; ".join(blockers) if blockers else None,
            "next_prompt_allowed": accepted,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / args.json_name
    markdown_path = args.out_dir / args.markdown_name
    write_json(json_path, payload)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    if args.strict_acceptance_exit_code and not accepted:
        return 2
    return 0


async def _run_cases(
    *,
    client: Any,
    scenario: str,
    cases: tuple[PromptCase, ...],
    user_offset: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> list[Any]:
    observations = []
    for index, case in enumerate(cases):
        observations.append(
            await client.run_case(
                scenario=scenario,
                case=case,
                user_index=user_offset + index,
                request_index=index,
                poll_interval_seconds=poll_interval_seconds,
                timeout_seconds=timeout_seconds,
            )
        )
    return observations


def _summary(
    *,
    ready_observations: list[Any],
    repair_observations: list[Any],
    unsupported_observations: list[Any],
) -> dict[str, Any]:
    ready_count = sum(1 for item in ready_observations if item.status == "ready")
    repair_ready_count = sum(1 for item in repair_observations if item.status == "ready")
    unsupported_expected = [
        item
        for item in unsupported_observations
        if item.status in item.expected_statuses and not item.load_action_enabled
    ]
    return {
        "supported_ready": ready_count,
        "supported_total": len(ready_observations),
        "supported_ready_literal": f"{ready_count}/{len(ready_observations)} ready",
        "repair_ready": repair_ready_count,
        "repair_total": len(repair_observations),
        "repair_ready_literal": f"{repair_ready_count}/{len(repair_observations)} ready",
        "unsupported_expected": len(unsupported_expected),
        "unsupported_total": len(unsupported_observations),
        "unsupported_expected_literal": (
            f"{len(unsupported_expected)}/{len(unsupported_observations)} expected"
        ),
    }


def _blockers(*, fake_worker: bool, summary: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if fake_worker:
        blockers.append("local fake-worker pipeline smoke is not Mac Studio acceptance evidence")
    if summary["supported_ready"] != 10 or summary["supported_total"] != 10:
        blockers.append(
            "supported prompt readiness failed: "
            f"{summary['supported_ready']}/{summary['supported_total']} ready"
        )
    if summary["repair_ready"] != 5 or summary["repair_total"] != 5:
        blockers.append(
            f"repair readiness failed: {summary['repair_ready']}/{summary['repair_total']} ready"
        )
    if summary["unsupported_expected"] != summary["unsupported_total"]:
        blockers.append(
            "unsupported/off-topic prompt handling failed: "
            f"{summary['unsupported_expected']}/{summary['unsupported_total']} expected"
        )
    return blockers


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    decision = payload["rollout_decision"]
    supported_ready_literal = summary["supported_ready"] == 10 and summary["supported_total"] == 10
    return "\n".join(
        [
            "# Backtest AI Configurator Pipeline Smoke",
            "",
            "Direct API readiness smoke for supported, repair and unsupported prompts.",
            "",
            "## Summary",
            "",
            f"- accepted: {payload['accepted']}",
            f"- blocking_reason: {payload['blocking_reason']}",
            f"- next_prompt_allowed: {payload['next_prompt_allowed']}",
            f"- supported prompts: {summary['supported_ready_literal']}",
            f"- 10/10 ready: {supported_ready_literal}",
            f"- repair prompts: {summary['repair_ready_literal']}",
            f"- unsupported/off-topic prompts: {summary['unsupported_expected_literal']}",
            "",
            "## Supported Ready",
            "",
            markdown_table(
                _rows(payload["supported_ready_observations"]),
                ("case_id", "status", "load_action", "codes"),
            ),
            "",
            "## Repair",
            "",
            markdown_table(
                _rows(payload["repair_observations"]),
                ("case_id", "status", "load_action", "codes"),
            ),
            "",
            "## Unsupported and Off-topic",
            "",
            markdown_table(
                _rows(payload["unsupported_observations"]),
                ("case_id", "status", "load_action", "codes"),
            ),
            "",
            "## Rollout Decision",
            "",
            f"- Accepted: {decision['accepted']}",
            f"- Reason: {decision['reason']}",
            f"- Blockers: {', '.join(decision['blockers']) if decision['blockers'] else 'none'}",
            "",
        ]
    )


def _rows(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": item["case_id"],
            "status": item["status"],
            "load_action": item["load_action_enabled"],
            "codes": ",".join(item["validation_error_codes"]),
        }
        for item in observations
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
