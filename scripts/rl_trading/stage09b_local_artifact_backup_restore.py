from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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
    DEFAULT_STAGE09B_BACKUP_ROOT_V1,
    DEFAULT_STAGE09B_RESTORE_ROOT_V1,
    RL_TRADING_ARTIFACT_ROOT_V1,
    STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    STAGE09B_PREVIOUS_CHAMPION_FIXTURE_ID_V1,
    ArtifactBackupError,
    Stage09BDrillConfig,
    compute_file_sha256,
    hash_json_payload_v1,
    run_stage09b_backup_restore_drill_v1,
)

DEFAULT_STAGE08L_SUMMARY_PATH = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08l_reward_warm_start_research_v1/"
    "stage08l_reward_warm_start_99a00ffa43c83b9ac553/"
    "stage08l_reward_warm_start_research_summary.json"
)
DEFAULT_STAGE08L_SUMMARY_SHA256 = (
    "5c25cc9d6a99b549f230a506f61a64563c64da61864127ae0c4c30405941b1a1"
)
DEFAULT_STAGE08M_RUN_DIR = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08m_supervised_warm_start_candidate_scorecard_v1/"
    "stage08m_supervised_warm_start_fe2fe3c5257fd9992c55"
)
DEFAULT_STAGE08M_CANDIDATE_MANIFEST_PATH = (
    DEFAULT_STAGE08M_RUN_DIR / "stage08m_supervised_warm_start_candidate_manifest.json"
)
DEFAULT_STAGE08M_CANDIDATE_MANIFEST_SHA256 = (
    "9e2767ead0b697d0194e501aa7932b44fc1f5d1b180713f1270c81d1c887a69c"
)
DEFAULT_STAGE08M_SCORECARD_PATH = (
    DEFAULT_STAGE08M_RUN_DIR / "stage08m_supervised_warm_start_candidate_scorecard_summary.json"
)
DEFAULT_STAGE08M_SCORECARD_SHA256 = (
    "ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7"
)
DEFAULT_STAGE08J_MANIFEST_PATH = Path(
    "/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/"
    "stage08j_article_sessionized_manifest.json"
)
DEFAULT_STAGE08J_MANIFEST_SHA256 = (
    "fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a"
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "rollback-dry-run":
            payload = _rollback_dry_run(args)
        else:
            payload = _run_drill(args)
    except ArtifactBackupError as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload.get("status") == "accepted" else 2


def _run_drill(args: argparse.Namespace) -> dict[str, object]:
    generated_at = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    run_id = args.run_id or f"stage09b_drill_{generated_at:%Y%m%dt%H%M%Sz}"
    config = Stage09BDrillConfig(
        artifact_root=args.artifact_root,
        backup_root=args.backup_root,
        restore_root=args.restore_root,
        run_id=run_id,
        generated_at_utc=generated_at,
        current_champion_manifest_path=args.current_champion_manifest_path,
        expected_current_champion_manifest_sha256=args.expected_current_champion_manifest_sha256,
        current_champion_scorecard_path=args.current_champion_scorecard_path,
        expected_current_champion_scorecard_sha256=args.expected_current_champion_scorecard_sha256,
        source_manifest_path=args.source_manifest_path,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        research_source_summary_path=args.research_source_summary_path,
        expected_research_source_summary_sha256=args.expected_research_source_summary_sha256,
        current_champion_id=args.current_champion_id,
        previous_champion_id=args.previous_champion_id,
        restore_retention_days=args.restore_retention_days,
        same_physical_disk=not args.separate_physical_disk,
    )
    return run_stage09b_backup_restore_drill_v1(config)


def _rollback_dry_run(args: argparse.Namespace) -> dict[str, object]:
    generated_at = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    registry_dump_hash: str | None = None
    if args.registry_metadata_dump is not None:
        registry_dump_hash = compute_file_sha256(args.registry_metadata_dump)
        registry_dump = _read_json(args.registry_metadata_dump)
        active = _mapping(registry_dump.get("active_champion"), "active_champion")
        previous = _mapping(registry_dump.get("previous_champion"), "previous_champion")
        if active.get("model_version_id") != args.expected_current_model_version_id:
            raise ArtifactBackupError(
                reason="current_model_version_id_mismatch",
                field=str(active.get("model_version_id")),
            )
        if previous.get("model_version_id") != args.to_model_version_id:
            raise ArtifactBackupError(
                reason="previous_model_version_id_mismatch",
                field=str(previous.get("model_version_id")),
            )

    payload: dict[str, object] = {
        "expected_current_model_version_id": args.expected_current_model_version_id,
        "generated_at_utc": _format_utc(generated_at),
        "kind": "rl_trading_stage09b_rollback_dry_run_result_v1",
        "no_artifact_deletion": True,
        "reason": args.reason,
        "registry_metadata_dump": (
            None
            if args.registry_metadata_dump is None
            else {
                "path": str(args.registry_metadata_dump),
                "sha256": registry_dump_hash,
            }
        ),
        "status": "accepted",
        "to_model_version_id": args.to_model_version_id,
    }
    return {**payload, "rollback_dry_run_hash": hash_json_payload_v1(payload)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 09B local RL artifact backup/restore drill.",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run-drill")
    _add_run_args(run_parser)
    rollback_parser = subparsers.add_parser("rollback-dry-run")
    rollback_parser.add_argument("--registry-metadata-dump", type=Path, default=None)
    rollback_parser.add_argument(
        "--to-model-version-id",
        type=str,
        default=STAGE09B_PREVIOUS_CHAMPION_FIXTURE_ID_V1,
    )
    rollback_parser.add_argument(
        "--expected-current-model-version-id",
        type=str,
        default=STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    )
    rollback_parser.add_argument("--reason", type=str, default="stage09b_restore_drill")
    rollback_parser.add_argument("--generated-at-utc", type=str, default=None)
    _add_run_args(parser)
    parser.set_defaults(command="run-drill")
    return parser


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artifact-root", type=Path, default=Path(RL_TRADING_ARTIFACT_ROOT_V1))
    parser.add_argument("--backup-root", type=Path, default=Path(DEFAULT_STAGE09B_BACKUP_ROOT_V1))
    parser.add_argument("--restore-root", type=Path, default=Path(DEFAULT_STAGE09B_RESTORE_ROOT_V1))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    parser.add_argument("--current-champion-id", type=str, default=STAGE09_ACCEPTED_CANDIDATE_ID_V1)
    parser.add_argument(
        "--previous-champion-id",
        type=str,
        default=STAGE09B_PREVIOUS_CHAMPION_FIXTURE_ID_V1,
    )
    parser.add_argument(
        "--current-champion-manifest-path",
        type=Path,
        default=DEFAULT_STAGE08M_CANDIDATE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--expected-current-champion-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE08M_CANDIDATE_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--current-champion-scorecard-path",
        type=Path,
        default=DEFAULT_STAGE08M_SCORECARD_PATH,
    )
    parser.add_argument(
        "--expected-current-champion-scorecard-sha256",
        type=str,
        default=DEFAULT_STAGE08M_SCORECARD_SHA256,
    )
    parser.add_argument("--source-manifest-path", type=Path, default=DEFAULT_STAGE08J_MANIFEST_PATH)
    parser.add_argument(
        "--expected-source-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE08J_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--research-source-summary-path",
        type=Path,
        default=DEFAULT_STAGE08L_SUMMARY_PATH,
    )
    parser.add_argument(
        "--expected-research-source-summary-sha256",
        type=str,
        default=DEFAULT_STAGE08L_SUMMARY_SHA256,
    )
    parser.add_argument("--restore-retention-days", type=int, default=30)
    parser.add_argument("--separate-physical-disk", action="store_true")


def _read_json(path: Path) -> dict[str, object]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _mapping(value: Any, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactBackupError(reason="expected_mapping", field=field)
    return value


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).replace(
        microsecond=0
    )


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _render_status(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
