from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def _resolve_repo_root() -> Path:
    candidates = (Path.cwd(), Path(__file__).resolve().parents[2])
    for candidate in candidates:
        if (candidate / ".codex").exists() and (candidate / "src").exists():
            return candidate
    return Path(__file__).resolve().parents[2]


REPO_ROOT = _resolve_repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.rl_trading.domain import (  # noqa: E402
    DEFAULT_STAGE10A_CALIBRATION_PACK_HASH_V1,
    DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1,
    DEFAULT_STAGE10A_OUTPUT_ROOT_V1,
    DEFAULT_STAGE10A_SOURCE_MANIFEST_SHA256_V1,
    RL_TRADING_ARTIFACT_ROOT_V1,
    STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    Stage10ALifecycleError,
    Stage10APromotionCheckConfig,
    Stage10APromotionScorecard,
    Stage10ARetrainTaskConfig,
    Stage10ARollbackConfig,
    run_stage10a_promotion_check_v1,
    run_stage10a_retrain_task_plan_v1,
    run_stage10a_rollback_dry_run_v1,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "promotion-check":
            payload = _promotion_check(args)
        elif args.command == "rollback-dry-run":
            payload = _rollback_dry_run(args)
        else:
            payload = _plan_retrain(args)
    except Stage10ALifecycleError as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload.get("status") == "accepted" else 2


def _plan_retrain(args: argparse.Namespace) -> dict[str, object]:
    generated_at = _generated_at(args.generated_at_utc)
    run_id = args.run_id or f"stage10a_retrain_{generated_at:%Y%m%dt%H%M%Sz}"
    return run_stage10a_retrain_task_plan_v1(
        Stage10ARetrainTaskConfig(
            artifact_root=args.artifact_root,
            output_root=args.output_root,
            run_id=run_id,
            generated_at_utc=generated_at,
            retrain_mode=args.mode,
            trigger=args.trigger,
            base_model_version_id=args.base_model_version_id,
            calibration_pack_id=args.calibration_pack_id,
            calibration_pack_hash=args.calibration_pack_hash,
            source_manifest_sha256=args.source_manifest_sha256,
            schedule_enabled=args.schedule_enabled,
            schedule_id=args.schedule_id,
            drift_signal_id=args.drift_signal_id,
            requested_by_ref_hash=args.requested_by_ref_hash,
            auto_promote_requested=args.auto_promote_requested,
        )
    )


def _promotion_check(args: argparse.Namespace) -> dict[str, object]:
    generated_at = _generated_at(args.generated_at_utc)
    run_id = args.run_id or f"stage10a_promotion_check_{generated_at:%Y%m%dt%H%M%Sz}"
    scorecard = Stage10APromotionScorecard(
        pnl_after_fees_funding_slippage_quote=args.pnl_after_fees_funding_slippage_quote,
        max_drawdown_quote=args.max_drawdown_quote,
        trades_count=args.trades_count,
        ticker_positive_group_ratio=args.ticker_positive_group_ratio,
        out_of_sample_days=args.out_of_sample_days,
        overfit_ratio=args.overfit_ratio,
        latency_p95_ms=args.latency_p95_ms,
        resource_rss_mb=args.resource_rss_mb,
        artifact_integrity_ok=args.artifact_integrity_ok,
        registry_integrity_ok=args.registry_integrity_ok,
    )
    return run_stage10a_promotion_check_v1(
        Stage10APromotionCheckConfig(
            artifact_root=args.artifact_root,
            output_root=args.output_root,
            run_id=run_id,
            generated_at_utc=generated_at,
            candidate_model_version_id=args.candidate_model_version_id,
            current_champion_model_version_id=args.current_champion_model_version_id,
            candidate_manifest_path=args.candidate_manifest_path,
            expected_candidate_manifest_sha256=args.expected_candidate_manifest_sha256,
            calibration_pack_path=args.calibration_pack_path,
            expected_calibration_pack_sha256=args.expected_calibration_pack_sha256,
            calibration_pack_id=args.calibration_pack_id,
            calibration_pack_hash=args.calibration_pack_hash,
            scorecard=scorecard,
            operator_ref_hash=args.operator_ref_hash,
            admin_ref_hash=args.admin_ref_hash,
            approval_reason=args.approval_reason,
            auto_promote_requested=args.auto_promote_requested,
        )
    )


def _rollback_dry_run(args: argparse.Namespace) -> dict[str, object]:
    generated_at = _generated_at(args.generated_at_utc)
    run_id = args.run_id or f"stage10a_rollback_{generated_at:%Y%m%dt%H%M%Sz}"
    return run_stage10a_rollback_dry_run_v1(
        Stage10ARollbackConfig(
            artifact_root=args.artifact_root,
            output_root=args.output_root,
            run_id=run_id,
            generated_at_utc=generated_at,
            current_champion_model_version_id=args.expected_current_model_version_id,
            previous_champion_model_version_id=args.to_model_version_id,
            current_calibration_pack_id=args.expected_current_calibration_pack_id,
            previous_calibration_pack_id=args.to_calibration_pack_id,
            current_registry_metadata_sha256=args.current_registry_metadata_sha256,
            previous_champion_manifest_sha256=args.previous_champion_manifest_sha256,
            previous_calibration_pack_sha256=args.previous_calibration_pack_sha256,
            operator_ref_hash=args.operator_ref_hash,
            reason=args.reason,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 10A RL retraining, promotion, and rollback lifecycle commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan-retrain")
    _add_common_args(plan)
    plan.add_argument("--mode", choices=["full_retrain", "fine_tune"], required=True)
    plan.add_argument("--trigger", choices=["manual", "scheduled", "drift"], required=True)
    plan.add_argument("--base-model-version-id", type=str, default=STAGE09_ACCEPTED_CANDIDATE_ID_V1)
    plan.add_argument(
        "--calibration-pack-id",
        type=str,
        default=DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1,
    )
    plan.add_argument(
        "--calibration-pack-hash",
        type=str,
        default=DEFAULT_STAGE10A_CALIBRATION_PACK_HASH_V1,
    )
    plan.add_argument(
        "--source-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE10A_SOURCE_MANIFEST_SHA256_V1,
    )
    plan.add_argument("--schedule-enabled", action="store_true")
    plan.add_argument("--schedule-id", type=str, default=None)
    plan.add_argument("--drift-signal-id", type=str, default=None)
    plan.add_argument("--requested-by-ref-hash", type=str, default=None)
    plan.add_argument("--auto-promote-requested", action="store_true")

    promotion = subparsers.add_parser("promotion-check")
    _add_common_args(promotion)
    promotion.add_argument("--candidate-model-version-id", type=str, required=True)
    promotion.add_argument(
        "--current-champion-model-version-id",
        type=str,
        default=STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    )
    promotion.add_argument("--candidate-manifest-path", type=Path, required=True)
    promotion.add_argument("--expected-candidate-manifest-sha256", type=str, required=True)
    promotion.add_argument("--calibration-pack-path", type=Path, required=True)
    promotion.add_argument("--expected-calibration-pack-sha256", type=str, required=True)
    promotion.add_argument(
        "--calibration-pack-id",
        type=str,
        default=DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1,
    )
    promotion.add_argument(
        "--calibration-pack-hash",
        type=str,
        default=DEFAULT_STAGE10A_CALIBRATION_PACK_HASH_V1,
    )
    promotion.add_argument("--pnl-after-fees-funding-slippage-quote", type=float, required=True)
    promotion.add_argument("--max-drawdown-quote", type=float, required=True)
    promotion.add_argument("--trades-count", type=int, required=True)
    promotion.add_argument("--ticker-positive-group-ratio", type=float, required=True)
    promotion.add_argument("--out-of-sample-days", type=int, required=True)
    promotion.add_argument("--overfit-ratio", type=float, required=True)
    promotion.add_argument("--latency-p95-ms", type=float, required=True)
    promotion.add_argument("--resource-rss-mb", type=float, required=True)
    promotion.add_argument("--artifact-integrity-ok", action="store_true")
    promotion.add_argument("--registry-integrity-ok", action="store_true")
    promotion.add_argument("--operator-ref-hash", type=str, default=None)
    promotion.add_argument("--admin-ref-hash", type=str, default=None)
    promotion.add_argument("--approval-reason", type=str, default="stage10a_promotion_check")
    promotion.add_argument("--auto-promote-requested", action="store_true")

    rollback = subparsers.add_parser("rollback-dry-run")
    _add_common_args(rollback)
    rollback.add_argument(
        "--expected-current-model-version-id",
        type=str,
        default=STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    )
    rollback.add_argument("--to-model-version-id", type=str, required=True)
    rollback.add_argument(
        "--expected-current-calibration-pack-id",
        type=str,
        default=DEFAULT_STAGE10A_CALIBRATION_PACK_ID_V1,
    )
    rollback.add_argument("--to-calibration-pack-id", type=str, required=True)
    rollback.add_argument("--current-registry-metadata-sha256", type=str, required=True)
    rollback.add_argument("--previous-champion-manifest-sha256", type=str, required=True)
    rollback.add_argument("--previous-calibration-pack-sha256", type=str, required=True)
    rollback.add_argument("--operator-ref-hash", type=str, required=True)
    rollback.add_argument("--reason", type=str, default="stage10a_operator_rollback")
    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", type=Path, default=Path(RL_TRADING_ARTIFACT_ROOT_V1))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_STAGE10A_OUTPUT_ROOT_V1))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--generated-at-utc", type=str, default=None)


def _generated_at(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).replace(
        microsecond=0
    )


def _render_status(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
