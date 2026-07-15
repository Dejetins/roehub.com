from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, Mapping

from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobStageWeights,
    BacktestJobTopVariant,
)

_FUNDING_METRIC_KEYS = (
    "total_return_pct_net_of_funding",
    "funding_return_pct",
    "funding_pnl_quote",
    "funding_events_count",
)

@dataclass(frozen=True, slots=True)
class BacktestJobProgressReadModel:
    pipeline_stage: str
    percent: int
    processed_units: int
    total_units: int
    updated_at: datetime | None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "pipeline_stage": self.pipeline_stage,
            "percent": self.percent,
            "processed_units": self.processed_units,
            "total_units": self.total_units,
            "updated_at": _format_datetime(self.updated_at),
        }


@dataclass(frozen=True, slots=True)
class BacktestJobReadModel:
    job_id: str
    organization_id: str
    state: str
    request_hash: str
    result_config_hash: str
    artifact_metadata: Mapping[str, Any]
    progress: BacktestJobProgressReadModel
    request: Mapping[str, Any]
    requested_top_n: int | None
    ranking: Mapping[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    cancel_requested_at: datetime | None
    updated_at: datetime
    refresh_status: str
    generated_at: datetime
    next_allowed_refresh_at: datetime
    retry_after_seconds: int
    terminal_summary: Mapping[str, Any] = field(default_factory=dict)
    links: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_metadata",
            MappingProxyType(dict(self.artifact_metadata)),
        )
        object.__setattr__(self, "request", MappingProxyType(dict(self.request)))
        object.__setattr__(self, "ranking", MappingProxyType(dict(self.ranking)))
        object.__setattr__(
            self,
            "terminal_summary",
            MappingProxyType(dict(self.terminal_summary)),
        )
        object.__setattr__(self, "links", MappingProxyType(dict(self.links)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "organization_id": self.organization_id,
            "state": self.state,
            "request_hash": self.request_hash,
            "result_config_hash": self.result_config_hash,
            "artifact_metadata": dict(self.artifact_metadata),
            "progress": self.progress.as_mapping(),
            "request": dict(self.request),
            "requested_top_n": self.requested_top_n,
            "ranking": dict(self.ranking),
            "created_at": _format_datetime(self.created_at),
            "started_at": _format_datetime(self.started_at),
            "finished_at": _format_datetime(self.finished_at),
            "cancel_requested_at": _format_datetime(self.cancel_requested_at),
            "updated_at": _format_datetime(self.updated_at),
            "refresh_status": self.refresh_status,
            "generated_at": _format_datetime(self.generated_at),
            "next_allowed_refresh_at": _format_datetime(self.next_allowed_refresh_at),
            "retry_after_seconds": self.retry_after_seconds,
            "terminal_summary": dict(self.terminal_summary),
            "links": dict(self.links),
        }


@dataclass(frozen=True, slots=True)
class BacktestJobTopVariantReadModel:
    rank: int
    variant_key: str
    variant_hash: str
    indicator_variant_hash: str | None
    summary_metrics: Mapping[str, Any]
    best_tp_pct: float | None
    best_sl_pct: float | None
    canonical_variant_params: Mapping[str, Any]
    readable_params: Mapping[str, Any]
    links: Mapping[str, Any]
    actions: Mapping[str, Any]
    funding_manifest_hash: str | None = None
    funding: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_metrics", MappingProxyType(dict(self.summary_metrics)))
        object.__setattr__(
            self,
            "canonical_variant_params",
            MappingProxyType(dict(self.canonical_variant_params)),
        )
        object.__setattr__(self, "readable_params", MappingProxyType(dict(self.readable_params)))
        object.__setattr__(self, "links", MappingProxyType(dict(self.links)))
        object.__setattr__(self, "actions", MappingProxyType(dict(self.actions)))
        object.__setattr__(self, "funding", MappingProxyType(dict(self.funding)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "indicator_variant_hash": self.indicator_variant_hash,
            "summary_metrics": dict(self.summary_metrics),
            "best_tp_pct": self.best_tp_pct,
            "best_sl_pct": self.best_sl_pct,
            "canonical_variant_params": dict(self.canonical_variant_params),
            "readable_params": dict(self.readable_params),
            "links": dict(self.links),
            "actions": dict(self.actions),
            "funding_manifest_hash": self.funding_manifest_hash,
            "funding": dict(self.funding),
        }


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesDetailReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    request_hash: str
    engine_params_hash: str
    artifact_manifest_hash: str
    summary_metrics: Mapping[str, Any]
    canonical_variant_params: Mapping[str, Any]
    readable_params: Mapping[str, Any]
    trades: tuple[Mapping[str, Any], ...]
    chart_overlay: Mapping[str, Any]
    cache: Mapping[str, Any]
    timing: Mapping[str, Any]
    funding_manifest_hash: str | None = None
    funding: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "summary_metrics",
            MappingProxyType(dict(self.summary_metrics)),
        )
        object.__setattr__(
            self,
            "canonical_variant_params",
            MappingProxyType(dict(self.canonical_variant_params)),
        )
        object.__setattr__(self, "readable_params", MappingProxyType(dict(self.readable_params)))
        object.__setattr__(
            self,
            "trades",
            tuple(MappingProxyType(dict(item)) for item in self.trades),
        )
        object.__setattr__(self, "chart_overlay", MappingProxyType(dict(self.chart_overlay)))
        object.__setattr__(self, "cache", MappingProxyType(dict(self.cache)))
        object.__setattr__(self, "timing", MappingProxyType(dict(self.timing)))
        object.__setattr__(self, "funding", MappingProxyType(dict(self.funding)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "request_hash": self.request_hash,
            "engine_params_hash": self.engine_params_hash,
            "artifact_manifest_hash": self.artifact_manifest_hash,
            "summary_metrics": dict(self.summary_metrics),
            "canonical_variant_params": dict(self.canonical_variant_params),
            "readable_params": dict(self.readable_params),
            "trades": [dict(item) for item in self.trades],
            "chart_overlay": dict(self.chart_overlay),
            "cache": dict(self.cache),
            "timing": dict(self.timing),
            "funding_manifest_hash": self.funding_manifest_hash,
            "funding": dict(self.funding),
        }


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesMaterializationReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    request_hash: str
    status: str
    materialization: Mapping[str, Any]
    cache: Mapping[str, Any]
    timing: Mapping[str, Any]
    pagination: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "materialization",
            MappingProxyType(dict(self.materialization)),
        )
        object.__setattr__(self, "cache", MappingProxyType(dict(self.cache)))
        object.__setattr__(self, "timing", MappingProxyType(dict(self.timing)))
        object.__setattr__(self, "pagination", MappingProxyType(dict(self.pagination)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "request_hash": self.request_hash,
            "status": self.status,
            "materialization": dict(self.materialization),
            "cache": dict(self.cache),
            "timing": dict(self.timing),
            "pagination": dict(self.pagination),
        }


BacktestLazyTradesResultReadModel = (
    BacktestLazyTradesDetailReadModel | BacktestLazyTradesMaterializationReadModel
)


@dataclass(frozen=True, slots=True)
class BacktestJobCreateResult:
    job: BacktestJobReadModel
    idempotent_replay: bool

    def as_mapping(self) -> dict[str, Any]:
        payload = self.job.as_mapping()
        payload["idempotent_replay"] = self.idempotent_replay
        return payload


@dataclass(frozen=True, slots=True)
class BacktestJobListResult:
    items: tuple[BacktestJobReadModel, ...]
    next_cursor: str | None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "items": [item.as_mapping() for item in self.items],
            "next_cursor": self.next_cursor,
        }


@dataclass(frozen=True, slots=True)
class BacktestJobTopResult:
    items: tuple[BacktestJobTopVariantReadModel, ...]

    def as_mapping(self) -> dict[str, Any]:
        return {"items": [item.as_mapping() for item in self.items]}


def build_backtest_job_read_model(
    *,
    job: BacktestJob,
    top_variants_count: int | None = None,
    now: datetime | None = None,
) -> BacktestJobReadModel:
    request = dict(job.request_json)
    ranking = _ranking_from_job(job=job, request=request)
    generated_at = now or datetime.now(UTC)
    retry_after_seconds = _retry_after_seconds(job=job)
    return BacktestJobReadModel(
        job_id=str(job.job_id),
        organization_id=str(job.organization_id),
        state=job.state,
        request_hash=job.request_hash,
        result_config_hash=job.backtest_runtime_config_hash,
        artifact_metadata=_artifact_metadata(job=job),
        progress=_progress_from_job(job=job, now=now),
        request=_request_summary(request=request),
        requested_top_n=job.requested_top_n,
        ranking=ranking,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        cancel_requested_at=job.cancel_requested_at,
        updated_at=job.updated_at,
        refresh_status=_refresh_status(job=job),
        generated_at=generated_at,
        next_allowed_refresh_at=generated_at + _retry_delta(seconds=retry_after_seconds),
        retry_after_seconds=retry_after_seconds,
        terminal_summary=_terminal_summary(job=job, top_variants_count=top_variants_count),
        links=_job_links(job_id=str(job.job_id)),
    )


def build_top_variant_read_model(
    *,
    job_id: str,
    row: BacktestJobTopVariant,
) -> BacktestJobTopVariantReadModel:
    payload = dict(row.payload_json)
    public_variant_key = str(
        payload.get("public_variant_key")
        or payload.get("variant_key")
        or _fallback_public_variant_key(job_id=job_id, row=row)
    )
    variant_hash = str(payload.get("variant_hash") or row.variant_key)
    indicator_variant_hash = payload.get("indicator_variant_hash") or row.indicator_variant_key
    funding = build_backtest_funding_read_model(
        summary_metrics=row.summary_metrics_json,
        payload=payload,
    )
    actions = _mapping_payload(payload.get("actions"))
    create_strategy_available = bool(
        _mapping_payload(payload.get("canonical_variant_params")).get("indicators")
    )
    actions.setdefault(
        "create_strategy_from_variant",
        {
            "available": create_strategy_available,
            "status": "launchable" if create_strategy_available else "not_launchable",
        },
    )
    return BacktestJobTopVariantReadModel(
        rank=row.rank,
        variant_key=public_variant_key,
        variant_hash=variant_hash,
        indicator_variant_hash=str(indicator_variant_hash) if indicator_variant_hash else None,
        summary_metrics=dict(row.summary_metrics_json),
        best_tp_pct=row.best_tp_pct,
        best_sl_pct=row.best_sl_pct,
        canonical_variant_params=_mapping_payload(payload.get("canonical_variant_params")),
        readable_params=_mapping_payload(payload.get("readable_params")),
        links=_mapping_payload(payload.get("links")),
        actions=actions,
        funding_manifest_hash=_optional_string(funding.get("funding_manifest_hash")),
        funding=funding,
    )


def build_backtest_funding_read_model(
    *,
    summary_metrics: Mapping[str, Any],
    payload: Mapping[str, Any] | None = None,
    artifact_metadata: Mapping[str, Any] | None = None,
    funding_events_count: int | None = None,
    overlay_status: str | None = None,
    overlay_warning_code: str | None = None,
) -> dict[str, Any]:
    """
    Build the additive funding read-model payload shared by top and lazy detail APIs.
    """

    source_payload = _mapping_payload(payload)
    adjustment = _mapping_payload(source_payload.get("funding_adjustment"))
    artifact = _mapping_payload(artifact_metadata)
    metrics = {
        key: summary_metrics[key]
        for key in _FUNDING_METRIC_KEYS
        if key in summary_metrics
    }
    manifest_hash = _optional_string(
        adjustment.get("funding_manifest_hash")
        or summary_metrics.get("funding_manifest_hash")
        or artifact.get("funding_manifest_hash")
    )
    warning_codes = _string_sequence(adjustment.get("funding_warning_codes"))
    if overlay_warning_code:
        warning_codes = tuple(sorted({*warning_codes, overlay_warning_code}))
    included = _optional_bool(adjustment.get("funding_included"))
    if included is None:
        included = bool(metrics)
    resolved_events_count = funding_events_count
    if resolved_events_count is None and "funding_events_count" in summary_metrics:
        resolved_events_count = int(float(summary_metrics["funding_events_count"]))
    funding: dict[str, Any] = {
        "funding_manifest_hash": manifest_hash,
        "included": included,
        "metrics": metrics,
        "data_quality": _optional_string(adjustment.get("funding_data_quality")),
        "warning_codes": list(warning_codes),
        "adjustment_scope": _optional_string(adjustment.get("funding_adjustment_scope")),
        "adjustment_exact_global_ranking": _optional_bool(
            adjustment.get("funding_adjustment_exact_global_ranking")
        ),
        "requested_ranking_metric": _optional_string(adjustment.get("requested_ranking_metric")),
        "effective_ranking_metric": _optional_string(adjustment.get("effective_ranking_metric")),
        "candidate_pool_size": _optional_int(adjustment.get("funding_candidate_pool_size")),
        "requested_top_n": _optional_int(adjustment.get("requested_top_n")),
        "funding_events_count": resolved_events_count or 0,
    }
    if overlay_status is not None:
        funding["funding_events_status"] = overlay_status
    return funding


def _progress_from_job(
    *,
    job: BacktestJob,
    now: datetime | None,
) -> BacktestJobProgressReadModel:
    stage_weights = BacktestJobStageWeights(stage_a=60, stage_b=30, finalizing=10)
    reference_now = now or datetime.now(UTC)
    return BacktestJobProgressReadModel(
        pipeline_stage=_pipeline_stage(job=job),
        percent=job.progress_percent(stage_weights=stage_weights),
        processed_units=job.processed_units,
        total_units=job.total_units,
        updated_at=job.progress_updated_at or job.updated_at or reference_now,
    )


def _pipeline_stage(*, job: BacktestJob) -> str:
    if job.state in {"succeeded", "failed", "cancelled"}:
        return job.state
    if job.state == "queued":
        return "queued"
    request = dict(job.request_json)
    explicit = request.get("pipeline_stage")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if job.stage == "finalizing":
        return "persist_top_n_io"
    risk = request.get("risk")
    risk_mode = risk.get("mode") if isinstance(risk, Mapping) else None
    if job.stage == "stage_b":
        return "tp_sl_exact_scoring" if risk_mode == "tp_sl_grid" else "exact_scoring"
    return "prepare_pools_core"


def _request_summary(*, request: Mapping[str, Any]) -> dict[str, Any]:
    risk = request.get("risk")
    return {
        "coordinates": _mapping_payload(request.get("coordinates")),
        "timeframe": request.get("timeframe"),
        "time_range": _mapping_payload(request.get("time_range")),
        "indicators": _sequence_payload(request.get("indicators")),
        "risk_mode": risk.get("mode") if isinstance(risk, Mapping) else None,
        "execution": _mapping_payload(request.get("execution")),
        "ui_metadata": _mapping_payload(request.get("ui_metadata")),
        "top_n": request.get("top_n"),
    }


def _ranking_from_job(*, job: BacktestJob, request: Mapping[str, Any]) -> dict[str, Any]:
    ranking = _mapping_payload(request.get("ranking"))
    if job.ranking_primary_metric is not None:
        ranking.setdefault("primary_metric", job.ranking_primary_metric)
    if job.ranking_secondary_metric is not None:
        ranking.setdefault("secondary_metric", job.ranking_secondary_metric)
    ranking.setdefault("direction", "desc")
    return ranking


def _artifact_metadata(*, job: BacktestJob) -> dict[str, Any]:
    if job.artifact_pin is None:
        return {}
    payload = {
        "artifact_slot": job.artifact_pin.artifact_slot,
        "artifact_slot_generation": job.artifact_pin.artifact_slot_generation,
        "artifact_manifest_hash": job.artifact_pin.artifact_manifest_hash,
        "artifact_asof_date": job.artifact_pin.artifact_asof_date,
    }
    request_artifact = _mapping_payload(dict(job.request_json).get("artifact_metadata"))
    if "hit_times_manifest_hash" in request_artifact:
        payload["hit_times_manifest_hash"] = request_artifact["hit_times_manifest_hash"]
    if "published_at_utc" in request_artifact:
        payload["published_at_utc"] = request_artifact["published_at_utc"]
    return payload


def _terminal_summary(
    *,
    job: BacktestJob,
    top_variants_count: int | None,
) -> dict[str, Any]:
    if job.state not in {"succeeded", "failed", "cancelled"}:
        return {}
    summary: dict[str, Any] = {"top_variants_count": top_variants_count}
    if job.last_error_json is not None:
        summary["last_error"] = job.last_error_json.to_mapping()
    return summary


def _refresh_status(*, job: BacktestJob) -> str:
    if job.state in {"queued", "running"}:
        return "poll"
    return "terminal"


def _retry_after_seconds(*, job: BacktestJob) -> int:
    if job.state == "queued":
        return 2
    if job.state == "running":
        return 5
    return 30


def _retry_delta(*, seconds: int) -> timedelta:
    return timedelta(seconds=seconds)


def _job_links(*, job_id: str) -> dict[str, str]:
    return {
        "status": f"/backtests/jobs/{job_id}",
        "top": f"/backtests/jobs/{job_id}/top",
        "cancel": f"/backtests/jobs/{job_id}/cancel",
        "runtime_defaults": "/backtests/runtime-defaults",
    }


def _fallback_public_variant_key(*, job_id: str, row: BacktestJobTopVariant) -> str:
    return f"job_{job_id.replace('-', '')[:8]}__rank_{row.rank}__vh_{row.variant_key[:8]}"


def _mapping_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _sequence_payload(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "BacktestJobCreateResult",
    "BacktestJobListResult",
    "BacktestLazyTradesDetailReadModel",
    "BacktestLazyTradesMaterializationReadModel",
    "BacktestLazyTradesResultReadModel",
    "BacktestJobProgressReadModel",
    "BacktestJobReadModel",
    "BacktestJobTopResult",
    "BacktestJobTopVariantReadModel",
    "build_backtest_funding_read_model",
    "build_backtest_job_read_model",
    "build_top_variant_read_model",
]
