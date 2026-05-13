from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCostEstimate,
    BacktestPreflightResult,
    BacktestValidationIssue,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestJobHeavyPromotion,
)
from trading.contexts.backtest.domain.entities import BacktestJobTopVariant


@dataclass(frozen=True, slots=True)
class BacktestChildSuccessResult:
    top_variants: tuple[BacktestJobTopVariant, ...]
    stage_timings: Mapping[str, float]
    summary_hash: str
    cleanup_evidence: Mapping[str, Any]


def preflight_to_mapping(*, preflight: BacktestPreflightResult) -> dict[str, Any]:
    return preflight.as_mapping()


def preflight_from_mapping(*, payload: Mapping[str, Any]) -> BacktestPreflightResult:
    artifact_metadata = payload.get("artifact_metadata")
    cost_estimate = payload.get("cost_estimate")
    if not isinstance(artifact_metadata, Mapping):
        raise ValueError("child preflight artifact_metadata must be object")
    if not isinstance(cost_estimate, Mapping):
        raise ValueError("child preflight cost_estimate must be object")
    return BacktestPreflightResult(
        normalized_request=_mapping(payload.get("normalized_request")),
        request_hash=str(payload["request_hash"]),
        result_config_hash=str(payload["result_config_hash"]),
        artifact_metadata=BacktestArtifactMetadata(
            artifact_slot=str(artifact_metadata["artifact_slot"]),
            artifact_slot_generation=int(artifact_metadata["artifact_slot_generation"]),
            artifact_manifest_hash=str(artifact_metadata["artifact_manifest_hash"]),
            artifact_asof_date=str(artifact_metadata["artifact_asof_date"]),
            hit_times_manifest_hash=(
                None
                if artifact_metadata.get("hit_times_manifest_hash") is None
                else str(artifact_metadata["hit_times_manifest_hash"])
            ),
            published_at_utc=str(artifact_metadata["published_at_utc"]),
        ),
        cost_estimate=BacktestCostEstimate(
            indicator_rows=int(cost_estimate["indicator_rows"]),
            candidate_combinations=int(cost_estimate["candidate_combinations"]),
            tp_sl_cells=int(cost_estimate["tp_sl_cells"]),
            cost_class=str(cost_estimate["cost_class"]),
        ),
        warnings=tuple(
            _validation_issue_from_mapping(item)
            for item in _sequence(payload.get("warnings"))
        ),
        errors=tuple(
            _validation_issue_from_mapping(item)
            for item in _sequence(payload.get("errors"))
        ),
    )


def child_success_to_mapping(*, result: Any) -> dict[str, Any]:
    return {
        "status": "succeeded",
        "top_variants": [
            _top_variant_to_mapping(row=row) for row in result.top_variants
        ],
        "stage_timings": dict(result.stage_timings),
        "summary_hash": str(result.summary_hash),
        "cleanup_evidence": dict(result.cleanup_evidence),
    }


def child_promotion_to_mapping(*, promotion: BacktestJobHeavyPromotion) -> dict[str, Any]:
    return {
        "status": "promote_to_heavy",
        "estimated_combinations_upper_bound": (
            promotion.estimated_combinations_upper_bound
        ),
        "actual_combinations": promotion.actual_combinations,
        "reason": promotion.reason,
    }


def child_result_from_mapping(
    *,
    payload: Mapping[str, Any],
) -> BacktestChildSuccessResult | BacktestJobHeavyPromotion:
    status = payload.get("status")
    if status == "promote_to_heavy":
        return BacktestJobHeavyPromotion(
            estimated_combinations_upper_bound=int(
                payload["estimated_combinations_upper_bound"]
            ),
            actual_combinations=int(payload["actual_combinations"]),
            reason=str(payload["reason"]),
        )
    if status != "succeeded":
        raise ValueError(f"unsupported child result status: {status!r}")
    return BacktestChildSuccessResult(
        top_variants=tuple(
            _top_variant_from_mapping(payload=item)
            for item in _sequence(payload.get("top_variants"))
        ),
        stage_timings={
            str(key): float(value)
            for key, value in _mapping(payload.get("stage_timings")).items()
        },
        summary_hash=str(payload["summary_hash"]),
        cleanup_evidence=dict(_mapping(payload.get("cleanup_evidence"))),
    )


def _top_variant_to_mapping(*, row: BacktestJobTopVariant) -> dict[str, Any]:
    return {
        "job_id": str(row.job_id),
        "rank": row.rank,
        "variant_key": row.variant_key,
        "indicator_variant_key": row.indicator_variant_key,
        "variant_index": row.variant_index,
        "total_return_pct": row.total_return_pct,
        "payload_json": dict(row.payload_json),
        "summary_metrics_json": dict(row.summary_metrics_json),
        "best_tp_pct": row.best_tp_pct,
        "best_sl_pct": row.best_sl_pct,
        "updated_at": row.updated_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
    }


def _top_variant_from_mapping(*, payload: Mapping[str, Any]) -> BacktestJobTopVariant:
    return BacktestJobTopVariant(
        job_id=UUID(str(payload["job_id"])),
        rank=int(payload["rank"]),
        variant_key=str(payload["variant_key"]),
        indicator_variant_key=str(payload["indicator_variant_key"]),
        variant_index=int(payload["variant_index"]),
        total_return_pct=float(payload["total_return_pct"]),
        payload_json=dict(_mapping(payload.get("payload_json"))),
        summary_metrics_json=dict(_mapping(payload.get("summary_metrics_json"))),
        best_tp_pct=(
            None if payload.get("best_tp_pct") is None else float(payload["best_tp_pct"])
        ),
        best_sl_pct=(
            None if payload.get("best_sl_pct") is None else float(payload["best_sl_pct"])
        ),
        updated_at=_parse_utc_datetime(value=str(payload["updated_at"])),
    )


def _validation_issue_from_mapping(payload: Any) -> BacktestValidationIssue:
    mapping = _mapping(payload)
    return BacktestValidationIssue(
        path=str(mapping["path"]),
        code=str(mapping["code"]),
        message=str(mapping["message"]),
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected JSON object")
    return value


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if not isinstance(value, list | tuple):
        raise ValueError("expected JSON array")
    return tuple(value)


def _parse_utc_datetime(*, value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)
