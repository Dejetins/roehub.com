from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from scripts.backtest_ai.configurator_benchmark_common import (
    SECURITY_PROMPT_CASES,
    FakeWorkerAiConfigClient,
    HttpAiConfigClient,
    benchmark_identity,
    local_host_identity,
    markdown_table,
    parse_header_values,
    summarize_security_observations,
    write_json,
)

DEFAULT_OUTPUT_DIR = Path(
    "docs/architecture/backtest/benchmark_iterations/"
    "2026-05-12_iteration_08_ai_configurator_load_security"
)
DEFAULT_CONFIG_PATH = Path("configs/prod/backtest_ai_configurator.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run /backtests AI configurator security evaluation pack."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--user-id-header", default="x-user-id")
    parser.add_argument("--user-id-prefix", default="security-ai-config")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument("--job-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json-name", default="security_eval_results.json")
    parser.add_argument("--markdown-name", default="security_eval_summary.md")
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
    headers = parse_header_values(args.header)
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
    observations = []
    try:
        for index, case in enumerate(SECURITY_PROMPT_CASES):
            observations.append(
                await client.run_case(
                    scenario="security eval mix",
                    case=case,
                    user_index=index,
                    request_index=index,
                    poll_interval_seconds=args.poll_interval_seconds,
                    timeout_seconds=args.job_timeout_seconds,
                )
            )
    finally:
        await client.aclose()
    summary = summarize_security_observations(observations)
    payload = {
        "kind": "backtest_ai_configurator_security_eval",
        "acceptance_classification": "developer_smoke"
        if args.fake_worker
        else "macstudio_acceptance_candidate",
        "load_generator_host": local_host_identity(),
        "identity": benchmark_identity(config_path=args.config_path),
        "security eval mix": [case.case_id for case in SECURITY_PROMPT_CASES],
        "observations": [item.as_mapping() for item in observations],
        "summary": summary,
        "rollout_decision": {
            "accepted": bool(summary["pass"]) and not args.fake_worker,
            "reason": (
                "accepted security eval"
                if summary["pass"] and not args.fake_worker
                else "rollout blocked"
            ),
            "blockers": _security_blockers(
                fake_worker=args.fake_worker,
                summary=summary,
            ),
        },
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
    summary = payload["summary"]
    rows = [
        {
            "case_id": item["case_id"],
            "category": item["category"],
            "status": item["status"],
            "load_action": item["load_action_enabled"],
            "friendly": item["friendly_message"],
            "codes": ",".join(item["validation_error_codes"]),
        }
        for item in payload["observations"]
    ]
    decision = payload["rollout_decision"]
    return "\n".join(
        [
            "# Backtest AI Configurator Iteration 08 Security Eval",
            "",
            (
                "Security eval mix for prompt-injection, off-topic, "
                "output-injection and resource-abuse cases."
            ),
            "",
            "## Summary",
            "",
            f"- Cases: {summary['cases']}",
            f"- Unauthorized actions: {summary['unauthorized_actions']}",
            f"- Private/system leakage: {summary['private_or_system_leakage']}",
            f"- Rendered HTML/script: {summary['rendered_html_or_script']}",
            f"- Friendly blocked message rate: {summary['friendly_blocked_message_rate']}",
            f"- Pass: {summary['pass']}",
            "",
            markdown_table(
                rows,
                ("case_id", "category", "status", "load_action", "friendly", "codes"),
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


def _security_blockers(*, fake_worker: bool, summary: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if fake_worker:
        blockers.append("local fake-worker security eval is not Mac Studio acceptance evidence")
    if summary["unauthorized_actions"]:
        blockers.append(f"unauthorized actions observed: {summary['unauthorized_actions']}")
    if summary["private_or_system_leakage"]:
        blockers.append(f"private/system leakage observed: {summary['private_or_system_leakage']}")
    if summary["rendered_html_or_script"]:
        blockers.append(f"HTML/script output observed: {summary['rendered_html_or_script']}")
    return blockers


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
