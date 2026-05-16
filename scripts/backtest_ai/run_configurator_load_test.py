from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from scripts.backtest_ai.configurator_benchmark_common import (
    DEFAULT_SCENARIOS,
    SAFE_PROMPT_CASES,
    FakeWorkerAiConfigClient,
    HttpAiConfigClient,
    benchmark_identity,
    collect_macstudio_snapshot,
    fetch_metrics_snapshot,
    local_host_identity,
    markdown_table,
    parse_header_values,
    parse_session_cookie_file,
    redacted_auth_inventory,
    run_load_scenario,
    selected_scenarios,
    write_json,
)

DEFAULT_OUTPUT_DIR = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-05-13_lmstudio_serving_recovery"
)
DEFAULT_CONFIG_PATH = Path("configs/prod/backtest_ai_configurator.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run /backtests AI configurator API load benchmark scenarios."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--user-id-header", default="x-user-id")
    parser.add_argument("--user-id-prefix", default="bench-ai-config")
    parser.add_argument("--scenario", action="append", choices=tuple(DEFAULT_SCENARIOS))
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--max-requests-per-scenario", type=int, default=None)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-name", default="lmstudio_load_benchmark_acceptance.json")
    parser.add_argument("--markdown-name", default="lmstudio_load_benchmark_acceptance.md")
    parser.add_argument("--metrics-url", default=None)
    parser.add_argument("--macstudio-host", default=None)
    parser.add_argument(
        "--session-cookie-file",
        type=Path,
        default=None,
        help=(
            "JSON object with cookie_name and sessions_by_user_index. "
            "Session values are used for requests but redacted from evidence."
        ),
    )
    parser.add_argument(
        "--strict-acceptance-exit-code",
        action="store_true",
        help="Exit non-zero when rollout_decision.accepted is false.",
    )
    parser.add_argument(
        "--fake-worker",
        action="store_true",
        help="Use in-process deterministic API + fake worker; developer smoke only.",
    )
    return parser


async def run_async(args: argparse.Namespace) -> int:
    scenarios = selected_scenarios(
        tuple(DEFAULT_SCENARIOS) if args.all_scenarios else (args.scenario or ["S1"])
    )
    if args.duration_scale <= 0:
        raise SystemExit("--duration-scale must be > 0")
    if args.max_requests_per_scenario is not None and args.max_requests_per_scenario <= 0:
        raise SystemExit("--max-requests-per-scenario must be > 0")
    headers = parse_header_values(args.header)
    session_cookie_name, session_ids_by_user_index = parse_session_cookie_file(
        args.session_cookie_file
    )
    client = (
        FakeWorkerAiConfigClient(timeout_seconds=args.http_timeout_seconds)
        if args.fake_worker
        else HttpAiConfigClient(
            base_url=args.base_url,
            headers=headers,
            user_id_header=args.user_id_header,
            user_id_prefix=args.user_id_prefix,
            timeout_seconds=args.http_timeout_seconds,
            session_cookie_name=session_cookie_name,
            session_ids_by_user_index=session_ids_by_user_index,
        )
    )
    before_metrics = await fetch_metrics_snapshot(args.metrics_url)
    before_host = collect_macstudio_snapshot(args.macstudio_host)
    scenario_results: list[dict[str, Any]] = []
    try:
        for scenario in scenarios:
            scenario_results.append(
                await run_load_scenario(
                    client=client,
                    scenario=scenario,
                    prompt_cases=SAFE_PROMPT_CASES,
                    duration_scale=args.duration_scale,
                    max_requests=args.max_requests_per_scenario,
                    poll_interval_seconds=args.poll_interval_seconds,
                    job_timeout_seconds=args.job_timeout_seconds,
                    seed=args.seed,
                )
            )
    finally:
        await client.aclose()
    after_metrics = await fetch_metrics_snapshot(args.metrics_url)
    after_host = collect_macstudio_snapshot(args.macstudio_host)

    payload = {
        "kind": "backtest_ai_configurator_load_benchmark",
        "acceptance_classification": "developer_smoke"
        if args.fake_worker
        else "macstudio_acceptance_candidate",
        "load_generator_host": local_host_identity(),
        "macstudio_host_status_before": before_host,
        "macstudio_host_status_after": after_host,
        "identity": benchmark_identity(config_path=args.config_path),
        "target": {
            "base_url": "fake-worker" if args.fake_worker else args.base_url,
            "scenarios": [scenario.name for scenario in scenarios],
            "prompt_case_ids": [case.case_id for case in SAFE_PROMPT_CASES],
            "load_generator_on_macstudio": False,
            "auth": redacted_auth_inventory(
                session_cookie_name=session_cookie_name,
                session_ids_by_user_index=session_ids_by_user_index,
            ),
        },
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "scenario_results": scenario_results,
        "rollout_decision": _rollout_decision(
            fake_worker=args.fake_worker,
            scenario_results=scenario_results,
            macstudio_host=args.macstudio_host,
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / args.json_name
    markdown_path = args.out_dir / args.markdown_name
    write_json(json_path, payload)
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    if args.strict_acceptance_exit_code and not payload["rollout_decision"]["accepted"]:
        return 2
    return 0


def render_markdown(payload: dict[str, Any]) -> str:
    identity = payload["identity"]
    rows = []
    for result in payload["scenario_results"]:
        summary = result["summary"]
        total_latency = summary["total_latency_ms"]
        queue_wait = summary["queue_wait_ms"]
        rows.append(
            {
                "scenario": result["scenario"],
                "requests": summary["requests"],
                "p50_total_ms": total_latency["p50"],
                "p95_total_ms": total_latency["p95"],
                "p99_total_ms": total_latency["p99"],
                "p95_queue_ms": queue_wait["p95"],
                "final valid config rate": summary["final_valid_config_rate"],
                "repair_rate": summary["repair_rate"],
                "quota_capacity": summary["quota_or_capacity_responses"],
            }
        )
    decision = payload["rollout_decision"]
    return "\n".join(
        [
            "# Backtest AI Configurator Iteration 08 Load Benchmark",
            "",
            "API pipeline benchmark harness for `/backtests` AI configurator scenarios.",
            "",
            "## Version",
            "",
            f"- Branch: {identity.get('branch')}",
            f"- Commit: {identity.get('commit')}",
            f"- Config: {identity.get('config_path')}",
            f"- Config SHA256: {identity.get('config_sha256')}",
            f"- Model id: {identity.get('model_id')}",
            f"- Model path hash: {identity.get('model_path_hash')}",
            f"- context_window: {identity.get('context_window_tokens')}",
            f"- max_output_tokens: {identity.get('max_output_tokens')}",
            f"- active_generations: {identity.get('active_generations')}",
            f"- queue limits: {identity.get('queue_limits')}",
            "",
            "## Scenario Metrics",
            "",
            markdown_table(
                rows,
                (
                    "scenario",
                    "requests",
                    "p50_total_ms",
                    "p95_total_ms",
                    "p99_total_ms",
                    "p95_queue_ms",
                    "final valid config rate",
                    "repair_rate",
                    "quota_capacity",
                ),
            ),
            "",
            "## Host Evidence",
            "",
            "- Load generator host is recorded in JSON.",
            (
                "- `memory_pressure` and `vm_stat` snapshots are recorded in JSON "
                "when `--macstudio-host` is used."
            ),
            (
                "- `active_generations` is recorded from config identity; live worker "
                "metric snapshots are recorded when `--metrics-url` is used."
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


def _rollout_decision(
    *,
    fake_worker: bool,
    scenario_results: list[dict[str, Any]],
    macstudio_host: str | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    scenario_names = {str(result["scenario"]) for result in scenario_results}
    required = {"S1", "S5", "S10", "S50", "S100"}
    if fake_worker:
        blockers.append("local fake-worker smoke is not Mac Studio acceptance evidence")
    if scenario_names != required:
        blockers.append(f"missing required scenarios: {sorted(required - scenario_names)}")
    if not macstudio_host:
        blockers.append("Mac Studio host snapshots were not collected")
    for result in scenario_results:
        summary = result["summary"]
        if summary["requests"] <= 0:
            blockers.append(f"{result['scenario']} produced no requests")
        if summary["final_valid_config_rate"] is not None and (
            summary["final_valid_config_rate"] < 0.98
        ):
            blockers.append(f"{result['scenario']} final valid config rate below 98%")
    return {
        "accepted": not blockers,
        "reason": "accepted Mac Studio load evidence" if not blockers else "rollout blocked",
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
