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
    DEFAULT_STAGE10_DOMINANCE_SHARE_LIMIT_V1,
    DEFAULT_STAGE10_FULL_CONFIDENCE_SESSIONS_V1,
    DEFAULT_STAGE10_MIN_TICKER_NET_PNL_AFTER_COSTS_QUOTE_V1,
    DEFAULT_STAGE10_MIN_TICKER_POSITIVE_RATIO_V1,
    DEFAULT_STAGE10_MIN_TICKER_SESSIONS_V1,
    DEFAULT_STAGE10_OUTPUT_ROOT_V1,
    RL_TRADING_ARTIFACT_ROOT_V1,
    STAGE09_ACCEPTED_CANDIDATE_ID_V1,
    STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1,
    Stage10CalibrationConfig,
    Stage10CalibrationError,
    run_stage10_per_ticker_calibration_v1,
)

DEFAULT_STAGE08M_RUN_DIR = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08m_supervised_warm_start_candidate_scorecard_v1/"
    "stage08m_supervised_warm_start_fe2fe3c5257fd9992c55"
)
DEFAULT_STAGE08M_CANDIDATE_SUMMARY_PATH = (
    DEFAULT_STAGE08M_RUN_DIR / "stage08m_supervised_warm_start_candidate_scorecard_summary.json"
)
DEFAULT_STAGE08M_CANDIDATE_SUMMARY_SHA256 = (
    "ff518bf3134670a0e814db7bfff45a3112e40f2169f9110c0b352cc77f044ab7"
)
DEFAULT_STAGE08M_CANDIDATE_MANIFEST_PATH = (
    DEFAULT_STAGE08M_RUN_DIR / "stage08m_supervised_warm_start_candidate_manifest.json"
)
DEFAULT_STAGE08M_CANDIDATE_MANIFEST_SHA256 = STAGE09_ACCEPTED_CANDIDATE_MANIFEST_SHA256_V1
DEFAULT_STAGE08J_MANIFEST_PATH = Path(
    "/opt/roehub/state/rl_trading/datasets/stage08j_article_sessionized_dataset_v1/"
    "stage08j_article_sessionized_manifest.json"
)
DEFAULT_STAGE08J_MANIFEST_SHA256 = (
    "fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a"
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = _run(args)
    except Stage10CalibrationError as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload.get("status") == "accepted" else 2


def _run(args: argparse.Namespace) -> dict[str, object]:
    generated_at = (
        _parse_utc(args.generated_at_utc)
        if args.generated_at_utc is not None
        else datetime.now(UTC).replace(microsecond=0)
    )
    run_id = args.run_id or f"stage10_per_ticker_calibration_{generated_at:%Y%m%dt%H%M%Sz}"
    config = Stage10CalibrationConfig(
        artifact_root=args.artifact_root,
        output_root=args.output_root,
        run_id=run_id,
        generated_at_utc=generated_at,
        candidate_summary_path=args.candidate_summary_path,
        expected_candidate_summary_sha256=args.expected_candidate_summary_sha256,
        candidate_manifest_path=args.candidate_manifest_path,
        expected_candidate_manifest_sha256=args.expected_candidate_manifest_sha256,
        source_manifest_path=args.source_manifest_path,
        expected_source_manifest_sha256=args.expected_source_manifest_sha256,
        model_version_id=args.model_version_id,
        exchange=args.exchange,
        market_type=args.market_type,
        min_ticker_sessions=args.min_ticker_sessions,
        min_ticker_positive_ratio=args.min_ticker_positive_ratio,
        min_ticker_net_pnl_after_costs_quote=args.min_ticker_net_pnl_after_costs_quote,
        full_confidence_sessions=args.full_confidence_sessions,
        dominance_share_limit=args.dominance_share_limit,
        pnl_weight=args.pnl_weight,
        positive_ratio_weight=args.positive_ratio_weight,
        turnover_weight=args.turnover_weight,
        risk_concentration_weight=args.risk_concentration_weight,
    )
    return run_stage10_per_ticker_calibration_v1(config)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Stage 10 per-ticker RL calibration artifacts.",
    )
    parser.add_argument("--artifact-root", type=Path, default=Path(RL_TRADING_ARTIFACT_ROOT_V1))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_STAGE10_OUTPUT_ROOT_V1))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    parser.add_argument("--model-version-id", type=str, default=STAGE09_ACCEPTED_CANDIDATE_ID_V1)
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--market-type", type=str, default="futures")
    parser.add_argument(
        "--candidate-summary-path",
        type=Path,
        default=DEFAULT_STAGE08M_CANDIDATE_SUMMARY_PATH,
    )
    parser.add_argument(
        "--expected-candidate-summary-sha256",
        type=str,
        default=DEFAULT_STAGE08M_CANDIDATE_SUMMARY_SHA256,
    )
    parser.add_argument(
        "--candidate-manifest-path",
        type=Path,
        default=DEFAULT_STAGE08M_CANDIDATE_MANIFEST_PATH,
    )
    parser.add_argument(
        "--expected-candidate-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE08M_CANDIDATE_MANIFEST_SHA256,
    )
    parser.add_argument("--source-manifest-path", type=Path, default=DEFAULT_STAGE08J_MANIFEST_PATH)
    parser.add_argument(
        "--expected-source-manifest-sha256",
        type=str,
        default=DEFAULT_STAGE08J_MANIFEST_SHA256,
    )
    parser.add_argument(
        "--min-ticker-sessions",
        type=int,
        default=DEFAULT_STAGE10_MIN_TICKER_SESSIONS_V1,
    )
    parser.add_argument(
        "--min-ticker-positive-ratio",
        type=float,
        default=DEFAULT_STAGE10_MIN_TICKER_POSITIVE_RATIO_V1,
    )
    parser.add_argument(
        "--min-ticker-net-pnl-after-costs-quote",
        type=float,
        default=DEFAULT_STAGE10_MIN_TICKER_NET_PNL_AFTER_COSTS_QUOTE_V1,
    )
    parser.add_argument(
        "--full-confidence-sessions",
        type=int,
        default=DEFAULT_STAGE10_FULL_CONFIDENCE_SESSIONS_V1,
    )
    parser.add_argument(
        "--dominance-share-limit",
        type=float,
        default=DEFAULT_STAGE10_DOMINANCE_SHARE_LIMIT_V1,
    )
    parser.add_argument("--pnl-weight", type=float, default=0.45)
    parser.add_argument("--positive-ratio-weight", type=float, default=0.25)
    parser.add_argument("--turnover-weight", type=float, default=0.15)
    parser.add_argument("--risk-concentration-weight", type=float, default=0.15)
    return parser


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).replace(
        microsecond=0
    )


def _render_status(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())
