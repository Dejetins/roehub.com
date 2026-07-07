from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_STAGE08J_MANIFEST = Path(
    "/opt/roehub/state/rl_trading/datasets/"
    "stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json"
)
DEFAULT_FINAL_MANIFEST = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08k_article_demo_profile_training_evaluation_v1/"
    "stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/"
    "final_holdout_b2adb7da3abc/stage08f_evaluation_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "local_artifacts/rl_trading/stage08k_high_volatility_regime"
)
DEFAULT_DATASET_VERSION = "hf_period_rebuild_current_trading"
DEFAULT_SPLIT = "backtest"
TARGET_BUCKET = "high"


class Stage08OHighVolatilityReportError(RuntimeError):
    def __init__(self, reason: str, *, field: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.field = field


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    stage08j_manifest = _load_json(args.stage08j_manifest)
    final_manifest = _load_json(args.final_manifest)
    scorecards_path = Path(final_manifest["artifact_hashes"]["scorecards"]["path"])
    balance_curve_path = Path(final_manifest["artifact_hashes"]["balance_curve"]["path"])
    scorecards = _scorecards(_load_json(scorecards_path))
    candidate = _candidate_scorecard(scorecards)
    balance_rows = _balance_rows(_load_json(balance_curve_path))
    sessions = _session_index(stage08j_manifest, args.dataset_version, args.split)
    scorecard_session_indexes = _scorecard_session_indexes(balance_rows)
    bucket_payload = _bucket_assignments(
        sessions=sessions,
        eligible_session_indexes=scorecard_session_indexes,
    )
    trades = _trades_with_bucket(balance_rows, sessions, bucket_payload["bucket_by_session_index"])
    target_trades = [trade for trade in trades if trade["volatility_bucket"] == args.bucket]
    target_session_indexes = [
        session_index
        for session_index in scorecard_session_indexes
        if bucket_payload["bucket_by_session_index"][session_index] == args.bucket
    ]
    ticker_rows = _ticker_rows(
        sessions=sessions,
        target_session_indexes=target_session_indexes,
        target_trades=target_trades,
    )
    official_bucket = _official_bucket_row(candidate, args.bucket)
    recomputed = _trade_totals(target_trades)
    summary = {
        "artifact_kind": "rl_trading_stage08o_stage08k_high_volatility_regime_report",
        "schema_version": 1,
        "stage": "08O-followup",
        "source_stage": "08K",
        "status": "accepted_for_local_research_report",
        "generated_at_utc": args.generated_at_utc,
        "proof_boundary": "target_host_non_production_forensic_pre_main",
        "regime_control_answer": {
            "can_control_volatility_regime": True,
            "control_surface": "pre-intent volatility-regime gate",
            "live_contract_required": (
                "Use only past/present candle features, freeze thresholds from accepted "
                "calibration, and fail closed when the volatility score cannot be computed."
            ),
            "current_report_scope": (
                "Stage 08K scorecard-session high-volatility decomposition. This report "
                "does not activate runtime, change registry, or prove product/mainnet readiness."
            ),
        },
        "input_artifacts": {
            "stage08j_manifest": {
                "path": str(args.stage08j_manifest),
                "sha256": _sha256_file(args.stage08j_manifest),
            },
            "final_manifest": {
                "path": str(args.final_manifest),
                "sha256": _sha256_file(args.final_manifest),
            },
            "scorecards": {
                "path": str(scorecards_path),
                "sha256": _sha256_file(scorecards_path),
            },
            "balance_curve": {
                "path": str(balance_curve_path),
                "sha256": _sha256_file(balance_curve_path),
            },
        },
        "bucket_method": {
            "bucket": args.bucket,
            "source": "scorecard_session_volatility_score_tertiles",
            "dataset_version": args.dataset_version,
            "split": args.split,
            "eligible_scorecard_sessions": len(scorecard_session_indexes),
            "thresholds": {
                "q33": bucket_payload["q33"],
                "q66": bucket_payload["q66"],
            },
            "bucket_session_counts": bucket_payload["bucket_session_counts"],
        },
        "official_scorecard_bucket": official_bucket,
        "recomputed_from_balance_curve": recomputed,
        "ticker_count_all_high_sessions": len(ticker_rows),
        "ticker_count_traded_high_sessions": sum(
            1 for row in ticker_rows if row["closed_trades"] > 0
        ),
        "ticker_rows": ticker_rows,
        "trade_rows": target_trades,
        "safety": {
            "training_or_tuning_run": False,
            "optuna_rerun": False,
            "model_registry_mutation": False,
            "runtime_config_activation": False,
            "exchange_side_effects": False,
            "contains_secrets_or_raw_provider_payloads": False,
            "post_main_production_runtime_proof_collected": False,
        },
    }
    return {**summary, "summary_hash": _hash_json(summary)}


def write_report(report: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "stage08k_high_volatility_regime_summary.json"
    ticker_csv_path = output_dir / "stage08k_high_volatility_ticker_pnl.csv"
    trades_csv_path = output_dir / "stage08k_high_volatility_trades.csv"
    summary_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(ticker_csv_path, report["ticker_rows"])
    _write_csv(trades_csv_path, report["trade_rows"])
    return {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "ticker_csv_path": str(ticker_csv_path),
        "ticker_csv_sha256": _sha256_file(ticker_csv_path),
        "trades_csv_path": str(trades_csv_path),
        "trades_csv_sha256": _sha256_file(trades_csv_path),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _scorecards(payload: Any) -> list[dict[str, Any]]:
    scorecards = payload.get("scorecards") if isinstance(payload, dict) else payload
    if not isinstance(scorecards, list):
        raise Stage08OHighVolatilityReportError("scorecards_payload_invalid")
    return [item for item in scorecards if isinstance(item, dict)]


def _candidate_scorecard(scorecards: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for scorecard in scorecards:
        if scorecard.get("acceptance_backtest") is True:
            return scorecard
    for scorecard in scorecards:
        if scorecard.get("policy_name") == "roehub_native_candidate_filtered_backtest":
            return scorecard
    raise Stage08OHighVolatilityReportError("candidate_scorecard_not_found")


def _balance_rows(payload: Any) -> list[Mapping[str, Any]]:
    rows = payload.get("balance_curve") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise Stage08OHighVolatilityReportError("balance_curve_payload_invalid")
    return [item for item in rows if isinstance(item, dict)]


def _session_index(
    manifest: Mapping[str, Any],
    dataset_version: str,
    split: str,
) -> list[dict[str, Any]]:
    entries = [
        entry
        for entry in manifest.get("split_artifacts", [])
        if isinstance(entry, Mapping)
        and entry.get("dataset_version") == dataset_version
        and entry.get("split") == split
    ]
    entries.sort(key=lambda item: str(item.get("symbol", "")))
    sessions: list[dict[str, Any]] = []
    for entry in entries:
        metadata_path = Path(str(entry["files"]["metadata"]["path"]))
        metadata = _load_json(metadata_path)
        metadata_sessions = metadata.get("sessions")
        if not isinstance(metadata_sessions, list):
            raise Stage08OHighVolatilityReportError(
                "metadata_sessions_invalid",
                field=str(metadata_path),
            )
        candidate_count = int(entry["candidate_count"])
        symbol = str(entry.get("symbol", "")).upper()
        for item in metadata_sessions[:candidate_count]:
            if not isinstance(item, Mapping):
                continue
            sessions.append(
                {
                    "session_index": len(sessions),
                    "symbol": symbol,
                    "signal_time_utc": item.get("signal_ts_open"),
                    "session_start_utc": item.get("session_start_utc"),
                    "session_end_utc": item.get("session_end_utc"),
                    "volatility_score": _float_or_zero(item.get("volatility_score")),
                }
            )
    if not sessions:
        raise Stage08OHighVolatilityReportError("session_index_empty")
    return sessions


def _scorecard_session_indexes(balance_rows: Sequence[Mapping[str, Any]]) -> list[int]:
    indexes = sorted({int(row["source_session_index"]) for row in balance_rows})
    if not indexes:
        raise Stage08OHighVolatilityReportError("scorecard_session_indexes_empty")
    return indexes


def _bucket_assignments(
    *,
    sessions: Sequence[Mapping[str, Any]],
    eligible_session_indexes: Sequence[int],
) -> dict[str, Any]:
    eligible_scores = [
        float(sessions[session_index]["volatility_score"])
        for session_index in eligible_session_indexes
    ]
    q33 = _quantile(eligible_scores, 0.3333333333)
    q66 = _quantile(eligible_scores, 0.6666666667)
    bucket_by_session_index: dict[int, str] = {}
    counts = {"low": 0, "medium": 0, "high": 0}
    for session_index in eligible_session_indexes:
        score = float(sessions[session_index]["volatility_score"])
        bucket = "low" if score <= q33 else "medium" if score <= q66 else "high"
        bucket_by_session_index[session_index] = bucket
        counts[bucket] += 1
    return {
        "bucket_by_session_index": bucket_by_session_index,
        "bucket_session_counts": counts,
        "q33": q33,
        "q66": q66,
    }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise Stage08OHighVolatilityReportError("quantile_values_empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _trades_with_bucket(
    balance_rows: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
    bucket_by_session_index: Mapping[int, str],
) -> list[dict[str, Any]]:
    if not balance_rows:
        return []
    previous_balance = float(balance_rows[0]["shared_balance_quote"])
    open_side_by_session: dict[tuple[Any, Any, Any], str] = {}
    trades: list[dict[str, Any]] = []
    for row_index, row in enumerate(balance_rows):
        session_index = int(row["source_session_index"])
        key = (row.get("source_session_index"), row.get("signal_time_utc"), row.get("symbol"))
        action = int(row.get("effective_action_id", 0))
        if action == 1:
            open_side_by_session[key] = "long"
        elif action == 2:
            open_side_by_session[key] = "short"
        balance = float(row["shared_balance_quote"])
        if row_index > 0 and not math.isclose(balance, previous_balance, abs_tol=1e-9):
            session = sessions[session_index]
            pnl = balance - previous_balance
            trades.append(
                {
                    "trade_order": len(trades),
                    "row_index": row_index,
                    "source_session_index": session_index,
                    "signal_time_utc": row.get("signal_time_utc"),
                    "date": str(row.get("signal_time_utc", ""))[:10],
                    "month": str(row.get("signal_time_utc", ""))[:7],
                    "symbol": str(row.get("symbol", session["symbol"])).upper(),
                    "side": open_side_by_session.pop(key, "unknown"),
                    "volatility_bucket": bucket_by_session_index[session_index],
                    "volatility_score": session["volatility_score"],
                    "pnl_after_costs_quote": round(pnl, 10),
                    "balance_before_quote": round(previous_balance, 10),
                    "balance_after_quote": round(balance, 10),
                }
            )
        previous_balance = balance
    return trades


def _ticker_rows(
    *,
    sessions: Sequence[Mapping[str, Any]],
    target_session_indexes: Sequence[int],
    target_trades: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    session_stats: dict[str, dict[str, Any]] = {}
    for session_index in target_session_indexes:
        session = sessions[session_index]
        symbol = str(session["symbol"]).upper()
        stats = session_stats.setdefault(
            symbol,
            {
                "symbol": symbol,
                "high_bucket_sessions": 0,
                "volatility_score_sum": 0.0,
                "max_volatility_score": 0.0,
            },
        )
        score = float(session["volatility_score"])
        stats["high_bucket_sessions"] += 1
        stats["volatility_score_sum"] += score
        stats["max_volatility_score"] = max(float(stats["max_volatility_score"]), score)

    trade_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "net_pnl_after_costs_quote": 0.0,
            "closed_trades": 0,
            "profitable_trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "best_trade_quote": None,
            "worst_trade_quote": None,
        }
    )
    for trade in target_trades:
        symbol = str(trade["symbol"]).upper()
        stats = trade_stats[symbol]
        pnl = float(trade["pnl_after_costs_quote"])
        stats["net_pnl_after_costs_quote"] += pnl
        stats["closed_trades"] += 1
        stats["profitable_trades"] += int(pnl > 0.0)
        stats["long_trades"] += int(trade.get("side") == "long")
        stats["short_trades"] += int(trade.get("side") == "short")
        stats["best_trade_quote"] = (
            pnl
            if stats["best_trade_quote"] is None
            else max(float(stats["best_trade_quote"]), pnl)
        )
        stats["worst_trade_quote"] = (
            pnl
            if stats["worst_trade_quote"] is None
            else min(float(stats["worst_trade_quote"]), pnl)
        )

    rows: list[dict[str, Any]] = []
    for symbol, session in session_stats.items():
        trades = trade_stats.get(symbol, {})
        closed_trades = int(trades.get("closed_trades", 0))
        sessions_count = int(session["high_bucket_sessions"])
        rows.append(
            {
                "symbol": symbol,
                "net_pnl_after_costs_quote": round(
                    float(trades.get("net_pnl_after_costs_quote", 0.0)),
                    10,
                ),
                "closed_trades": closed_trades,
                "profitable_trades": int(trades.get("profitable_trades", 0)),
                "win_rate": (
                    round(float(trades.get("profitable_trades", 0)) / closed_trades, 10)
                    if closed_trades
                    else 0.0
                ),
                "high_bucket_sessions": sessions_count,
                "trade_rate_per_high_session": (
                    round(closed_trades / sessions_count, 10) if sessions_count else 0.0
                ),
                "avg_volatility_score": round(
                    float(session["volatility_score_sum"]) / sessions_count,
                    12,
                ),
                "max_volatility_score": round(float(session["max_volatility_score"]), 12),
                "long_trades": int(trades.get("long_trades", 0)),
                "short_trades": int(trades.get("short_trades", 0)),
                "best_trade_quote": _none_or_round(trades.get("best_trade_quote")),
                "worst_trade_quote": _none_or_round(trades.get("worst_trade_quote")),
                "traded_in_high_regime": closed_trades > 0,
            }
        )
    return sorted(rows, key=lambda row: float(row["net_pnl_after_costs_quote"]), reverse=True)


def _official_bucket_row(candidate: Mapping[str, Any], bucket: str) -> Mapping[str, Any]:
    rows = candidate.get("metrics_by_volatility_bucket")
    if not isinstance(rows, list):
        raise Stage08OHighVolatilityReportError("metrics_by_volatility_bucket_missing")
    for row in rows:
        if isinstance(row, Mapping) and row.get("bucket") == bucket:
            return row
    raise Stage08OHighVolatilityReportError("target_bucket_missing", field=bucket)


def _trade_totals(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    pnl = sum(float(trade["pnl_after_costs_quote"]) for trade in trades)
    closed_trades = len(trades)
    profitable_trades = sum(1 for trade in trades if float(trade["pnl_after_costs_quote"]) > 0.0)
    return {
        "net_pnl_after_costs_quote": round(pnl, 10),
        "closed_trades": closed_trades,
        "profitable_trades": profitable_trades,
        "win_rate": round(profitable_trades / closed_trades, 10) if closed_trades else 0.0,
    }


def _write_csv(path: Path, rows: object) -> None:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise Stage08OHighVolatilityReportError("csv_rows_invalid", field=str(path))
    dict_rows = [row for row in rows if isinstance(row, Mapping)]
    if not dict_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def _float_or_zero(value: object) -> float:
    if value is None:
        return 0.0
    parsed = float(value)  # type: ignore[arg-type]
    return parsed if math.isfinite(parsed) else 0.0


def _none_or_round(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)  # type: ignore[arg-type]
    return round(parsed, 10)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage 08K high-volatility regime local report."
    )
    parser.add_argument("--stage08j-manifest", type=Path, default=DEFAULT_STAGE08J_MANIFEST)
    parser.add_argument("--final-manifest", type=Path, default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--bucket", default=TARGET_BUCKET, choices=("low", "medium", "high"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-at-utc", default="2026-07-07T00:00:00Z")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_report(args)
        outputs = write_report(report, args.output_dir)
    except Stage08OHighVolatilityReportError as exc:
        print(
            json.dumps(
                {"field": exc.field, "reason": exc.reason, "status": "blocked"},
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                **outputs,
                "bucket": args.bucket,
                "recomputed": report["recomputed_from_balance_curve"],
                "status": report["status"],
                "ticker_count_all_high_sessions": report["ticker_count_all_high_sessions"],
                "ticker_count_traded_high_sessions": report["ticker_count_traded_high_sessions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
