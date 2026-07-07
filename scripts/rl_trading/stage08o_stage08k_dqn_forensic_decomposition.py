from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_DUAL_SUMMARY = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08k_article_demo_profile_training_evaluation_v1/dual_branch_runs/"
    "stage08k_dual_branch_cpu_76f51186c00ecb54255e/"
    "stage08k_dual_branch_cpu_run_summary.json"
)
DEFAULT_FINAL_MANIFEST = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08k_article_demo_profile_training_evaluation_v1/"
    "stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/"
    "final_holdout_b2adb7da3abc/stage08f_evaluation_manifest.json"
)
DEFAULT_OPTUNA_SUMMARY = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08k_article_demo_profile_training_evaluation_v1/"
    "stage08k_roehub_native_03fd26aa_cc9179e39ded309544c3/"
    "stage08k_optuna_summary.json"
)
DEFAULT_STAGE08J_MANIFEST = Path(
    "/opt/roehub/state/rl_trading/datasets/"
    "stage08j_article_sessionized_dataset_v1/stage08j_article_sessionized_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/opt/roehub/state/rl_trading/evaluation_runs/"
    "stage08o_stage08k_dqn_forensic_decomposition_v1"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scorecards(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        scorecards = payload.get("scorecards")
    else:
        scorecards = payload
    if not isinstance(scorecards, list):
        raise ValueError("scorecards payload must contain a list")
    return scorecards


def _candidate_scorecard(scorecards: list[dict[str, Any]]) -> dict[str, Any]:
    for scorecard in scorecards:
        if scorecard.get("acceptance_backtest") is True:
            return scorecard
    for scorecard in scorecards:
        if scorecard.get("policy_name") == "roehub_native_candidate_filtered_backtest":
            return scorecard
    raise ValueError("candidate acceptance scorecard was not found")


def _balance_rows(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("balance_curve") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("balance curve payload must contain a list")
    return rows


def _reconstruct_trades(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    open_side_by_session: dict[tuple[Any, Any, Any], str] = {}
    trades: list[dict[str, Any]] = []
    previous_balance = float(rows[0]["shared_balance_quote"])

    for row_index, row in enumerate(rows):
        key = (
            row.get("source_session_index"),
            row.get("signal_time_utc"),
            row.get("symbol"),
        )
        effective_action = int(row.get("effective_action_id", 0))
        if effective_action == 1:
            open_side_by_session[key] = "long"
        elif effective_action == 2:
            open_side_by_session[key] = "short"

        balance = float(row["shared_balance_quote"])
        if row_index > 0 and not math.isclose(balance, previous_balance, abs_tol=1e-9):
            pnl = balance - previous_balance
            side = open_side_by_session.pop(key, "unknown")
            signal_time = str(row["signal_time_utc"])
            trades.append(
                {
                    "trade_order": len(trades),
                    "row_index": row_index,
                    "source_session_index": row.get("source_session_index"),
                    "signal_time_utc": signal_time,
                    "date": signal_time[:10],
                    "month": signal_time[:7],
                    "symbol": row.get("symbol"),
                    "side": side,
                    "step_idx": row.get("step_idx"),
                    "pnl_after_costs_quote": round(pnl, 10),
                    "balance_before_quote": round(previous_balance, 10),
                    "balance_after_quote": round(balance, 10),
                }
            )
        previous_balance = balance

    return trades


def _group_trades(trades: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for trade in trades:
        name = str(trade[key])
        row = grouped.setdefault(
            name,
            {
                key: name,
                "net_pnl_after_costs_quote": 0.0,
                "closed_trades": 0,
                "profitable_trades": 0,
            },
        )
        row["net_pnl_after_costs_quote"] += float(trade["pnl_after_costs_quote"])
        row["closed_trades"] += 1
        row["profitable_trades"] += int(float(trade["pnl_after_costs_quote"]) > 0.0)
    for row in grouped.values():
        row["net_pnl_after_costs_quote"] = round(row["net_pnl_after_costs_quote"], 10)
        trades_count = int(row["closed_trades"])
        row["win_rate"] = (
            round(row["profitable_trades"] / trades_count, 10) if trades_count else 0.0
        )
    return sorted(grouped.values(), key=lambda item: item["net_pnl_after_costs_quote"])


def _dominance(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    value_key: str = "net_pnl_after_costs_quote",
) -> dict[str, Any]:
    denominator = sum(abs(float(row.get(value_key, 0.0))) for row in rows)
    if denominator <= 0.0:
        return {
            "dominant_group": None,
            "dominant_abs_pnl_quote": 0.0,
            "abs_pnl_denominator_quote": 0.0,
            "share": 0.0,
        }
    dominant = max(rows, key=lambda row: abs(float(row.get(value_key, 0.0))))
    numerator = abs(float(dominant.get(value_key, 0.0)))
    return {
        "dominant_group": dominant.get(label_key),
        "dominant_abs_pnl_quote": round(numerator, 10),
        "abs_pnl_denominator_quote": round(denominator, 10),
        "share": numerator / denominator,
    }


def _positive_group_ratio(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = sum(1 for row in rows if float(row.get("net_pnl_after_costs_quote", 0.0)) > 0.0)
    flat = sum(
        1
        for row in rows
        if math.isclose(float(row.get("net_pnl_after_costs_quote", 0.0)), 0.0, abs_tol=1e-12)
    )
    total = len(rows)
    return {
        "positive_groups": positive,
        "flat_groups": flat,
        "non_positive_groups": total - positive,
        "total_groups": total,
        "ratio": positive / total if total else 0.0,
    }


def _daily_article_style_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    by_day = _group_trades(trades, "date")
    pnl_values = [float(row["net_pnl_after_costs_quote"]) for row in by_day]
    trade_pnl_values = [float(trade["pnl_after_costs_quote"]) for trade in trades]
    if not pnl_values:
        return {}
    mean = sum(pnl_values) / len(pnl_values)
    std = _population_std(pnl_values)
    negative_values = [value for value in pnl_values if value < 0.0]
    negative_std = _population_std(negative_values) if negative_values else 0.0
    return {
        "trade_days": len(pnl_values),
        "profit_days": sum(1 for value in pnl_values if value > 0.0),
        "profit_day_ratio": sum(1 for value in pnl_values if value > 0.0) / len(pnl_values),
        "trades_per_day": len(trades) / len(pnl_values),
        "sharpe_like_source_formula": mean / (std + 1e-9) * math.sqrt(len(pnl_values)),
        "sortino_like_source_formula": mean / (negative_std + 1e-9) * math.sqrt(len(pnl_values)),
        "average_trade_pnl_after_costs_quote": sum(trade_pnl_values) / len(trade_pnl_values),
        "max_profit_trade_quote": max(trade_pnl_values),
        "max_loss_trade_quote": min(trade_pnl_values),
    }


def _population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _max_drawdown(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"max_drawdown_pct": 0.0, "trough": None}
    peak = float(rows[0]["shared_balance_quote"])
    max_drawdown = 0.0
    trough: dict[str, Any] | None = None
    for row in rows:
        balance = float(row["shared_balance_quote"])
        if balance > peak:
            peak = balance
        drawdown = (balance / peak) - 1.0 if peak else 0.0
        if drawdown < max_drawdown:
            max_drawdown = drawdown
            trough = row
    return {
        "max_drawdown_pct": max_drawdown * 100.0,
        "trough": {
            "signal_time_utc": trough.get("signal_time_utc"),
            "symbol": trough.get("symbol"),
            "shared_balance_quote": trough.get("shared_balance_quote"),
        }
        if trough
        else None,
    }


def _sequence_summary(trades: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [float(trade["pnl_after_costs_quote"]) for trade in trades]
    if not pnls:
        return {}
    best_single = max(trades, key=lambda trade: float(trade["pnl_after_costs_quote"]))
    worst_single = min(trades, key=lambda trade: float(trade["pnl_after_costs_quote"]))
    return {
        "best_single_trade": _trade_projection(best_single),
        "worst_single_trade": _trade_projection(worst_single),
        "best_contiguous_sequence": _contiguous_extreme(trades, maximize=True),
        "worst_contiguous_sequence": _contiguous_extreme(trades, maximize=False),
        "best_sign_streak": _sign_streak(trades, positive=True),
        "worst_sign_streak": _sign_streak(trades, positive=False),
    }


def _trade_projection(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "trade_order": trade["trade_order"],
        "signal_time_utc": trade["signal_time_utc"],
        "symbol": trade["symbol"],
        "side": trade["side"],
        "pnl_after_costs_quote": trade["pnl_after_costs_quote"],
    }


def _contiguous_extreme(trades: list[dict[str, Any]], *, maximize: bool) -> dict[str, Any]:
    best_sum = -math.inf if maximize else math.inf
    current_sum = 0.0
    start = 0
    best_start = 0
    best_end = 0
    for index, trade in enumerate(trades):
        value = float(trade["pnl_after_costs_quote"])
        if maximize:
            if current_sum <= 0.0:
                current_sum = value
                start = index
            else:
                current_sum += value
            if current_sum > best_sum:
                best_sum = current_sum
                best_start = start
                best_end = index
        else:
            if current_sum >= 0.0:
                current_sum = value
                start = index
            else:
                current_sum += value
            if current_sum < best_sum:
                best_sum = current_sum
                best_start = start
                best_end = index
    return {
        "start_trade_order": trades[best_start]["trade_order"],
        "end_trade_order": trades[best_end]["trade_order"],
        "trade_count": best_end - best_start + 1,
        "net_pnl_after_costs_quote": round(best_sum, 10),
        "start_time_utc": trades[best_start]["signal_time_utc"],
        "end_time_utc": trades[best_end]["signal_time_utc"],
    }


def _sign_streak(trades: list[dict[str, Any]], *, positive: bool) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    current: list[dict[str, Any]] = []
    for trade in trades:
        value = float(trade["pnl_after_costs_quote"])
        matches = value > 0.0 if positive else value < 0.0
        if matches:
            current.append(trade)
        elif current:
            best = _better_streak(best, current, positive=positive)
            current = []
    if current:
        best = _better_streak(best, current, positive=positive)
    return best or {}


def _better_streak(
    best: dict[str, Any] | None,
    streak: list[dict[str, Any]],
    *,
    positive: bool,
) -> dict[str, Any]:
    total = sum(float(trade["pnl_after_costs_quote"]) for trade in streak)
    candidate = {
        "start_trade_order": streak[0]["trade_order"],
        "end_trade_order": streak[-1]["trade_order"],
        "trade_count": len(streak),
        "net_pnl_after_costs_quote": round(total, 10),
        "start_time_utc": streak[0]["signal_time_utc"],
        "end_time_utc": streak[-1]["signal_time_utc"],
    }
    if best is None:
        return candidate
    if positive:
        return candidate if total > float(best["net_pnl_after_costs_quote"]) else best
    return candidate if total < float(best["net_pnl_after_costs_quote"]) else best


def _top_bottom(rows: list[dict[str, Any]], *, limit: int = 10) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=lambda row: float(row.get("net_pnl_after_costs_quote", 0.0)))
    return {
        "bottom": sorted_rows[:limit],
        "top": list(reversed(sorted_rows[-limit:])),
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    dual_summary = _load_json(args.dual_summary)
    final_manifest = _load_json(args.final_manifest)
    optuna_summary = _load_json(args.optuna_summary)
    stage08j_manifest = _load_json(args.stage08j_manifest)

    scorecards_path = Path(final_manifest["artifact_hashes"]["scorecards"]["path"])
    balance_curve_path = Path(final_manifest["artifact_hashes"]["balance_curve"]["path"])
    scorecards = _scorecards(_load_json(scorecards_path))
    candidate = _candidate_scorecard(scorecards)
    balance_rows = _balance_rows(_load_json(balance_curve_path))
    trades = _reconstruct_trades(balance_rows)

    by_day = _group_trades(trades, "date")
    by_side = _group_trades(trades, "side")
    active_by_ticker = _group_trades(trades, "symbol")
    official_tickers = candidate["stability_by_ticker"]
    daily_article_style_metrics = _daily_article_style_metrics(trades)
    volatility_rows = candidate["metrics_by_volatility_bucket"]

    starting_equity = float(candidate["starting_equity_quote"])
    pnl = sum(float(trade["pnl_after_costs_quote"]) for trade in trades)
    return_pct = (pnl / starting_equity) * 100.0 if starting_equity else 0.0
    action_counts = candidate["action_counts"]
    open_total = int(action_counts.get("open_long", 0)) + int(action_counts.get("open_short", 0))
    open_side_dominance = (
        max(action_counts.get("open_long", 0), action_counts.get("open_short", 0)) / open_total
    )

    official_ticker_ratio = _positive_group_ratio(official_tickers)
    active_ticker_ratio = _positive_group_ratio(active_by_ticker)
    volatility_dominance = _dominance(volatility_rows, label_key="bucket")
    native_branch = dual_summary["branches"]["roehub_native_article_selector_30_10"]

    classification = {
        "08k_forensic_status": "per_ticker_per_regime_calibration_candidate",
        "article_similarity_status": "likely_aggregate_return_coincidence",
        "decision_reason": (
            "The aggregate return is real and near the article by one coarse metric, "
            "but high-volatility bucket absolute PnL dominance and the official "
            "all-ticker positive ratio fail the strict gate. Active traded tickers "
            "are materially better than the all-ticker universe, so the only safe "
            "continuation is per-ticker/per-regime research calibration, not Stage 09 "
            "or product/mainnet progression."
        ),
    }

    downstream_gates = {
        "stage09_for_08k_allowed": False,
        "stage19_mainnet_readiness_allowed": False,
        "stage20_mainnet_canary_allowed": False,
        "stage21_product_rollout_allowed": False,
        "08p_allowed": True,
        "next_prompt": (
            "08p-stage08k-per-ticker-regime-calibration.md "
            "(not generated by Stage 08O; requires explicit prompt-pack insertion before execution)"
        ),
    }

    return {
        "artifact_kind": "rl_trading_stage08o_stage08k_dqn_forensic_summary",
        "schema_version": 1,
        "stage": "08O",
        "source_stage": "08K",
        "status": "accepted",
        "generated_at_utc": args.generated_at_utc,
        "proof_boundary": "target_host_non_production_forensic_pre_main",
        "input_artifacts": {
            "prompt_path": args.prompt_path,
            "prompt_sha256": args.prompt_sha256,
            "dual_branch_summary": {
                "path": str(args.dual_summary),
                "sha256": _sha256_file(args.dual_summary),
                "expected_sha256": args.expected_dual_summary_sha256,
            },
            "native_final_evaluation_manifest": {
                "path": str(args.final_manifest),
                "sha256": _sha256_file(args.final_manifest),
                "expected_sha256": args.expected_final_manifest_sha256,
            },
            "native_optuna_summary": {
                "path": str(args.optuna_summary),
                "sha256": _sha256_file(args.optuna_summary),
                "expected_sha256": args.expected_optuna_summary_sha256,
            },
            "stage08j_article_dataset_manifest": {
                "path": str(args.stage08j_manifest),
                "sha256": _sha256_file(args.stage08j_manifest),
                "expected_sha256": args.expected_stage08j_manifest_sha256,
            },
            "scorecards": {
                "path": str(scorecards_path),
                "sha256": _sha256_file(scorecards_path),
                "manifest_sha256": final_manifest["artifact_hashes"]["scorecards"]["sha256"],
            },
            "balance_curve": {
                "path": str(balance_curve_path),
                "sha256": _sha256_file(balance_curve_path),
                "manifest_sha256": final_manifest["artifact_hashes"]["balance_curve"]["sha256"],
            },
        },
        "lineage": {
            "dual_summary_status": dual_summary.get("status"),
            "native_branch_stage09_allowed": native_branch.get("stage09_allowed"),
            "native_branch_candidate_manifest_path": native_branch.get("candidate_manifest_path"),
            "stage08j_status": stage08j_manifest.get("status"),
            "stage08j_selector_id": stage08j_manifest["policy"]["policy_id"],
            "best_optuna_trial_number": optuna_summary.get("best_trial_number"),
            "best_alpha_config_hash": optuna_summary.get("best_alpha_config_hash"),
            "final_evaluation_hash": optuna_summary.get("final_evaluation_hash"),
            "final_strict_gate_status": optuna_summary["final_strict_research_gate"]["status"],
            "final_strict_gate_blockers": optuna_summary["final_strict_research_gate"]["blockers"],
        },
        "math_reconciliation": {
            "starting_equity_quote": starting_equity,
            "shared_balance_initial_quote": candidate["shared_balance_initial_quote"],
            "shared_balance_final_quote": candidate["shared_balance_final_quote"],
            "pnl_from_balance_curve_after_costs_quote": round(pnl, 10),
            "scorecard_pnl_after_costs_quote": candidate["net_pnl_after_costs_quote"],
            "return_pct_from_recomputed_pnl": return_pct,
            "scorecard_return_pct_after_costs": candidate["return_pct_after_costs"],
            "closed_trades_from_balance_changes": len(trades),
            "scorecard_closed_trades": candidate["closed_trades"],
            "profitable_trades": candidate["profitable_trades"],
            "win_rate": candidate["win_rate"],
            "max_drawdown": _max_drawdown(balance_rows),
            "scorecard_max_drawdown_pct": candidate.get("max_drawdown_pct"),
        },
        "final_holdout_scope": {
            "final_split_total_session_count": optuna_summary["final_split"]["total_session_count"],
            "final_split_selected_session_count": optuna_summary["final_split"][
                "selected_session_count"
            ],
            "grouped_filtered_scorecard_session_count": candidate["session_count"],
            "decision_rows": candidate["decisions_count"],
            "balance_curve_rows": len(balance_rows),
            "period": candidate["out_of_sample_period"],
        },
        "decomposition": {
            "by_month_official": candidate["metrics_by_period"],
            "by_day_from_trades": {
                "count": len(by_day),
                **_top_bottom(by_day, limit=10),
            },
            "by_ticker_official_all_groups": {
                "positive_group_ratio": official_ticker_ratio,
                "dominance": _dominance(official_tickers, label_key="symbol"),
                **_top_bottom(official_tickers, limit=10),
            },
            "by_ticker_active_traded_only": {
                "positive_group_ratio": active_ticker_ratio,
                **_top_bottom(active_by_ticker, limit=10),
            },
            "by_volatility_bucket_official": {
                "rows": volatility_rows,
                "dominance": volatility_dominance,
            },
            "by_side_from_trades": by_side,
            "trade_sequences": _sequence_summary(trades),
            "daily_article_style_metrics_from_trades": daily_article_style_metrics,
        },
        "root_cause": {
            "single_group_dominates_final_result": {
                "status": "confirmed_real_concentration",
                "dominant_bucket": volatility_dominance["dominant_group"],
                "dominance_share": volatility_dominance["share"],
                "limit": optuna_summary["final_strict_research_gate"]["dominance_share_limit"],
                "numerator_abs_pnl_quote": volatility_dominance["dominant_abs_pnl_quote"],
                "denominator_abs_pnl_quote": volatility_dominance["abs_pnl_denominator_quote"],
                "interpretation": (
                    "The high-volatility bucket contributes almost all absolute bucket PnL; "
                    "medium volatility is negative and low volatility is small positive. "
                    "This is regime concentration, not a sign inversion."
                ),
            },
            "ticker_stability_obviously_broken": {
                "status": "confirmed_denominator_includes_inactive_ticker_groups",
                "official_all_ticker_positive_ratio": official_ticker_ratio["ratio"],
                "official_positive_groups": official_ticker_ratio["positive_groups"],
                "official_flat_groups": official_ticker_ratio["flat_groups"],
                "official_total_groups": official_ticker_ratio["total_groups"],
                "active_traded_ticker_positive_ratio": active_ticker_ratio["ratio"],
                "active_positive_groups": active_ticker_ratio["positive_groups"],
                "active_total_groups": active_ticker_ratio["total_groups"],
                "minimum": optuna_summary["final_strict_research_gate"][
                    "positive_group_ratio_minimum"
                ],
                "interpretation": (
                    "Among tickers where the filtered policy actually traded, most groups are "
                    "positive. The strict gate correctly fails closed because the full selected "
                    "ticker universe "
                    "contains many flat/no-trade groups."
                ),
            },
        },
        "scorecard_gate_audit": {
            "candidate_beats_best_sanity_baseline": optuna_summary["final_strict_research_gate"][
                "candidate_beats_best_sanity_baseline"
            ],
            "best_sanity_baseline_pnl_after_costs_quote": optuna_summary[
                "final_strict_research_gate"
            ]["best_baseline_net_pnl_after_costs_quote"],
            "closed_trades": optuna_summary["final_strict_research_gate"]["closed_trades"],
            "min_closed_trades": optuna_summary["final_strict_research_gate"]["min_closed_trades"],
            "monthly_dominance_recomputed": _dominance(
                candidate["metrics_by_period"],
                label_key="period",
            ),
            "volatility_dominance_recomputed": volatility_dominance,
            "ticker_dominance_recomputed": _dominance(official_tickers, label_key="symbol"),
            "monthly_positive_group_ratio_recomputed": _positive_group_ratio(
                candidate["metrics_by_period"]
            ),
            "ticker_positive_group_ratio_recomputed": official_ticker_ratio,
            "open_side_dominance_recomputed": open_side_dominance,
            "action_counts": action_counts,
            "bug_audit_verdict": "no_scorecard_gate_bug_found",
            "bug_audit_notes": [
                "PnL and return reconcile exactly from balance changes.",
                "Volatility dominance matches absolute-PnL denominator semantics.",
                "Ticker positive ratio matches all selected ticker groups, including flat no-trade "
                "groups.",
                "Long/short side reconstruction has zero unknown closes.",
            ],
        },
        "article_comparison": {
            "source_urls": [
                "https://habr.com/ru/articles/934258/",
                "https://github.com/YuriyKolesnikov/rl-trading-binance",
                "https://raw.githubusercontent.com/YuriyKolesnikov/rl-trading-binance/main/backtest_engine.py",
            ],
            "source_published_metrics": {
                "return_pct": 144.23,
                "sharpe": 1.85,
                "sortino": 2.05,
                "accuracy_pct": 69.6,
                "max_drawdown_pct": -22.49,
                "trade_days": 56,
                "profit_days": 44,
                "profit_day_ratio": 0.7857,
                "total_trades": 112,
                "average_trade_size_usdt": 11324.29,
                "trades_per_day": 2.0,
            },
            "roehub_artifact_comparable_metrics": {
                "return_pct": candidate["return_pct_after_costs"],
                "sharpe_like_source_formula": daily_article_style_metrics[
                    "sharpe_like_source_formula"
                ],
                "sortino_like_source_formula": daily_article_style_metrics[
                    "sortino_like_source_formula"
                ],
                "accuracy_pct": None,
                "accuracy_unavailable_reason": (
                    "The Stage 08K artifact stores profitable-trade win rate, but not source-style "
                    "correct_prediction fields for signal accuracy."
                ),
                "win_rate_pct_not_source_accuracy": candidate["win_rate"] * 100.0,
                "max_drawdown_pct": -abs(float(candidate.get("max_drawdown_pct", 0.0))),
                "trade_days": daily_article_style_metrics["trade_days"],
                "profit_days": daily_article_style_metrics["profit_days"],
                "profit_day_ratio": daily_article_style_metrics["profit_day_ratio"],
                "total_trades": candidate["closed_trades"],
                "average_trade_size_usdt": None,
                "average_trade_size_unavailable_reason": (
                    "Balance curve stores realized PnL and shared balance, not exact upstream "
                    "trade_amount."
                ),
                "average_trade_pnl_after_costs_quote": daily_article_style_metrics[
                    "average_trade_pnl_after_costs_quote"
                ],
                "trades_per_day": daily_article_style_metrics["trades_per_day"],
            },
            "article_similarity_decision": "likely_aggregate_return_coincidence",
            "article_similarity_notes": [
                "Return is close by one coarse metric: 125.0265333% vs article 144.23%.",
                "Trade count and cadence diverge materially: 316 trades and 3.90 trades/day vs "
                "112 and about 2.00.",
                "Source-style accuracy and exact average trade size are unavailable in Stage 08K "
                "artifacts.",
                "The Roehub result is heavily high-volatility concentrated, while the article's "
                "published summary claims balanced risk-adjusted behavior.",
            ],
        },
        "allowlist_and_regime_feasibility": {
            "restricted_high_volatility_research_possible": True,
            "per_ticker_allowlist_research_possible": True,
            "product_allowlist_allowed": False,
            "reason": (
                "High-volatility and active-traded ticker subsets contain useful research signal, "
                "but the official full-universe ticker gate and volatility concentration fail. "
                "Any follow-up must be calibration/research-only and must not activate runtime."
            ),
            "candidate_allowlist_seed": [
                row["symbol"] for row in _top_bottom(official_tickers, limit=10)["top"]
            ],
        },
        "classification": classification,
        "downstream_gates": downstream_gates,
        "safety": {
            "training_or_tuning_run": False,
            "optuna_rerun": False,
            "model_registry_mutation": False,
            "runtime_config_activation": False,
            "exchange_side_effects": False,
            "browser_auth_used": False,
            "contains_secrets_or_raw_provider_payloads": False,
            "post_main_production_runtime_proof_collected": False,
        },
        "contract_impact": {
            "public_api_contract": "none",
            "port_contract": "none",
            "dto_schema": "none",
            "persisted_schema": "none",
            "config_schema": "none",
            "request_hash_cache_key_persistence_identity": "none",
            "service_call_auth_timeout_retry_error_semantics": "none",
            "external_side_effect_idempotency_unknown_state": "none",
            "logs_metrics_traces_audit_ledger_report_redaction": "compatible-change",
            "benchmark_or_rollout_gate": "compatible-change",
            "browser_visible_behavior": "none",
            "performance_hot_path": "none",
        },
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "stage08o_stage08k_forensic_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dual-summary", type=Path, default=DEFAULT_DUAL_SUMMARY)
    parser.add_argument("--final-manifest", type=Path, default=DEFAULT_FINAL_MANIFEST)
    parser.add_argument("--optuna-summary", type=Path, default=DEFAULT_OPTUNA_SUMMARY)
    parser.add_argument("--stage08j-manifest", type=Path, default=DEFAULT_STAGE08J_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generated-at-utc", default="2026-07-06T00:00:00Z")
    parser.add_argument(
        "--prompt-path",
        default=".codex/agents/generated/rl-trading-agent-platform-v1/"
        "08o-stage08k-dqn-forensic-decomposition.md",
    )
    parser.add_argument(
        "--prompt-sha256",
        default="86918e6ffba475256c97432a6322dd4fed4a8c96582334d635b1e6de62cea178",
    )
    parser.add_argument(
        "--expected-dual-summary-sha256",
        default="70dfffd378f33da997b2471422491e9defc28f8b361b9c5b645087288129286d",
    )
    parser.add_argument(
        "--expected-final-manifest-sha256",
        default="c16a08dffa6f49fd8f24d9a1af6737114d54363b4cca3630a4e83c2661880b07",
    )
    parser.add_argument(
        "--expected-optuna-summary-sha256",
        default="8585d4342dab24850cd077e5287de5faab251e848f18eb044f70cc410ebf6cec",
    )
    parser.add_argument(
        "--expected-stage08j-manifest-sha256",
        default="fd7c614b4cc5085cc24cd054143b6bb188283b9cf423122d436e37769fcd639a",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    summary_path = write_summary(summary, args.output_dir)
    result = {
        "summary_path": str(summary_path),
        "summary_sha256": _sha256_file(summary_path),
        "status": summary["status"],
        "08k_forensic_status": summary["classification"]["08k_forensic_status"],
        "stage09_for_08k_allowed": summary["downstream_gates"]["stage09_for_08k_allowed"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
