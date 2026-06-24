from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.rl_trading.domain import (  # noqa: E402
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    UpstreamAlphaConfig,
    UpstreamMethodologyError,
    build_tiny_stage08b_session_features_v1,
    run_stage08b_core_smoke_v1,
)

DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "training_smokes"
    / "stage08b_upstream_methodology_core_port_v1"
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = UpstreamAlphaConfig(
        seed=args.seed,
        initial_balance=args.initial_balance,
        batch_size=args.batch_size,
        train_start=args.train_start,
        target_update_freq=args.target_update_freq,
        replay_capacity=args.replay_capacity,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
    )
    try:
        features = build_tiny_stage08b_session_features_v1(
            session_count=args.session_count,
            config=config,
        )
        report = run_stage08b_core_smoke_v1(
            session_features=features,
            output_root=args.output_root,
            config=config,
            episodes=args.episodes,
            generated_at_utc=(
                _parse_utc(args.generated_at_utc)
                if args.generated_at_utc is not None
                else datetime.now(UTC).replace(microsecond=0)
            ),
        )
    except UpstreamMethodologyError as exc:
        print(
            json.dumps(
                {"field": exc.field, "reason": exc.reason, "status": "blocked"},
                ensure_ascii=True,
                sort_keys=True,
            )
        )
        return 2

    metrics = report["metrics"]
    print(
        json.dumps(
            {
                "architecture_id": report["architecture_id"],
                "learn_update_count": metrics["learn_update_count"],
                "normalization_stats_hash": report["normalization_stats_hash"],
                "report_path": report["report_path"],
                "rss_mb_after": metrics["resource_usage"]["rss_mb_after"],
                "scripted_transition_sequence_used": metrics[
                    "scripted_transition_sequence_used"
                ],
                "selection_mode_counts": metrics["selection_mode_counts"],
                "smoke_report_hash": report["smoke_report_hash"],
                "status": report["status"],
                "target_sync_count": metrics["target_sync_count"],
                "transition_count": metrics["transition_count"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Stage 08B upstream-methodology core smoke.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--session-count", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=240824)
    parser.add_argument("--initial-balance", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-start", type=int, default=2)
    parser.add_argument("--target-update-freq", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=128)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=1.0)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--generated-at-utc", default=None)
    return parser


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


if __name__ == "__main__":
    raise SystemExit(main())
