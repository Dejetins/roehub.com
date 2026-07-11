from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.codex_quality_benchmark.manifest import load_manifest
from tools.codex_quality_benchmark.models import BenchmarkError
from tools.codex_quality_benchmark.reports import write_run_artifacts
from tools.codex_quality_benchmark.scoring import aggregate_run
from tools.codex_quality_benchmark.skill_audit import (
    audit_all_skills,
    audit_manifest_skills,
    compare_ab_results,
    compare_all_skills_ab,
    compare_focused_ab,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "validate-manifest":
            manifest = load_manifest(args.manifest)
            print(
                "OK: manifest valid "
                f"run_id={manifest.run_id} targets={len(manifest.targets)} "
                f"rubric_total={manifest.rubric_total}"
            )
            return 0
        if args.command == "aggregate":
            _, rows = aggregate_run(args.run_dir)
            write_run_artifacts(args.run_dir, rows, include_summary=False)
            print(f"OK: wrote {args.run_dir / 'results.tsv'} and {args.run_dir / 'events.jsonl'}")
            return 0
        if args.command == "summarize":
            _, rows = aggregate_run(args.run_dir)
            write_run_artifacts(args.run_dir, rows, include_summary=True)
            print(f"OK: wrote {args.run_dir / 'summary.md'}")
            return 0
        if args.command == "audit-skills":
            rows = audit_manifest_skills(
                args.manifest,
                args.out_dir,
                source=args.source,
                version_id=args.version_id,
            )
            print(f"OK: wrote skill audit for {len(rows)} rows under {args.out_dir}")
            return 0
        if args.command == "audit-all-skills":
            inventory, rows = audit_all_skills(
                args.out_dir,
                run_id=args.run_id,
                skills_root=args.skills_root,
                plugins_cache_root=args.plugins_cache_root,
            )
            print(
                "OK: wrote all-skills audit under "
                f"{args.out_dir}; inventory={len(inventory)} rows={len(rows)}"
            )
            return 0
        if args.command == "ab-compare":
            rows = compare_ab_results(
                args.run_dir,
                args.audit_dir,
                args.out_dir,
                target_metric=args.target_metric,
                min_metric_delta=args.min_metric_delta,
                max_task_regression=args.max_task_regression,
            )
            accepted = sum(1 for row in rows if row.ab_decision == "candidate")
            print(f"OK: wrote A/B decisions under {args.out_dir}; accepted={accepted}")
            return 0
        if args.command == "focused-ab-compare":
            decision = compare_focused_ab(
                args.before_audit,
                args.after_audit,
                args.pairwise,
                args.out_dir,
                target_id=args.target_id,
                target_metric=args.target_metric,
                min_metric_delta=args.min_metric_delta,
            )
            print(
                "OK: wrote focused A/B decision under "
                f"{args.out_dir}; decision={decision.ab_decision}"
            )
            return 0
        if args.command == "all-skills-ab-compare":
            rows = compare_all_skills_ab(
                args.before_audit,
                args.after_audit,
                args.out_dir,
                inventory_path=args.inventory,
                target_metric=args.target_metric,
                min_metric_delta=args.min_metric_delta,
            )
            accepted = sum(1 for row in rows if row.ab_decision == "candidate")
            print(
                "OK: wrote all-skills A/B decisions under "
                f"{args.out_dir}; rows={len(rows)} accepted={accepted}"
            )
            return 0
    except BenchmarkError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    parser.print_help(sys.stderr)
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codex-quality-benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--manifest", type=Path, required=True)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-dir", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--run-dir", type=Path, required=True)

    audit = subparsers.add_parser("audit-skills")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--out-dir", type=Path, required=True)
    audit.add_argument("--source", choices=["live", "versions"], default="versions")
    audit.add_argument("--version-id")

    audit_all = subparsers.add_parser("audit-all-skills")
    audit_all.add_argument("--out-dir", type=Path, required=True)
    audit_all.add_argument("--run-id", required=True)
    audit_all.add_argument(
        "--skills-root",
        type=Path,
        default=Path.home() / ".codex" / "skills",
    )
    audit_all.add_argument(
        "--plugins-cache-root",
        type=Path,
        default=Path.home() / ".codex" / "plugins" / "cache",
    )

    ab_compare = subparsers.add_parser("ab-compare")
    ab_compare.add_argument("--run-dir", type=Path, required=True)
    ab_compare.add_argument("--audit-dir", type=Path, required=True)
    ab_compare.add_argument("--out-dir", type=Path, required=True)
    ab_compare.add_argument(
        "--target-metric",
        choices=[
            "audit_score_0_100",
            "format_score",
            "description_score",
            "structure_score",
            "safety_score",
        ],
        default="audit_score_0_100",
    )
    ab_compare.add_argument("--min-metric-delta", type=float, default=5.0)
    ab_compare.add_argument("--max-task-regression", type=float, default=0.0)

    focused = subparsers.add_parser("focused-ab-compare")
    focused.add_argument("--before-audit", type=Path, required=True)
    focused.add_argument("--after-audit", type=Path, required=True)
    focused.add_argument("--pairwise", type=Path, required=True)
    focused.add_argument("--out-dir", type=Path, required=True)
    focused.add_argument("--target-id", required=True)
    focused.add_argument(
        "--target-metric",
        choices=[
            "audit_score_0_100",
            "format_score",
            "description_score",
            "structure_score",
            "safety_score",
        ],
        default="safety_score",
    )
    focused.add_argument("--min-metric-delta", type=float, default=5.0)

    all_skills_ab = subparsers.add_parser("all-skills-ab-compare")
    all_skills_ab.add_argument("--before-audit", type=Path, required=True)
    all_skills_ab.add_argument("--after-audit", type=Path, required=True)
    all_skills_ab.add_argument("--out-dir", type=Path, required=True)
    all_skills_ab.add_argument("--inventory", type=Path)
    all_skills_ab.add_argument(
        "--target-metric",
        choices=[
            "audit_score_0_100",
            "format_score",
            "description_score",
            "structure_score",
            "safety_score",
        ],
        default="audit_score_0_100",
    )
    all_skills_ab.add_argument("--min-metric-delta", type=float, default=1.0)

    return parser


if __name__ == "__main__":
    raise SystemExit(main())
