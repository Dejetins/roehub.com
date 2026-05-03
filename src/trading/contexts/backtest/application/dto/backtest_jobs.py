from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Mapping

from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobStageWeights,
    BacktestJobTopVariant,
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
    updated_at: datetime
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
            "updated_at": _format_datetime(self.updated_at),
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
        }


@dataclass(frozen=True, slots=True)
class BacktestResultSummaryReadModel:
    job: BacktestJobReadModel
    variants: tuple[BacktestJobTopVariantReadModel, ...]
    selected_variant_key: str | None
    links: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "links", MappingProxyType(dict(self.links)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job": self.job.as_mapping(),
            "variants": [variant.as_mapping() for variant in self.variants],
            "selected_variant_key": self.selected_variant_key,
            "links": dict(self.links),
        }


@dataclass(frozen=True, slots=True)
class BacktestResultSeriesReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    series: str
    requested_points: int
    point_limit: int
    total_points: int
    downsampled: bool
    points: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "points",
            tuple(MappingProxyType(dict(item)) for item in self.points),
        )
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "series": self.series,
            "requested_points": self.requested_points,
            "point_limit": self.point_limit,
            "total_points": self.total_points,
            "downsampled": self.downsampled,
            "points": [dict(item) for item in self.points],
            "summary": dict(self.summary),
        }


@dataclass(frozen=True, slots=True)
class BacktestResultMonthlyStatsReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    items: tuple[Mapping[str, Any], ...]
    totals: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "items",
            tuple(MappingProxyType(dict(item)) for item in self.items),
        )
        object.__setattr__(self, "totals", MappingProxyType(dict(self.totals)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "items": [dict(item) for item in self.items],
            "totals": dict(self.totals),
        }


@dataclass(frozen=True, slots=True)
class BacktestResultSymbolStatsReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    items: tuple[Mapping[str, Any], ...]
    totals: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "items",
            tuple(MappingProxyType(dict(item)) for item in self.items),
        )
        object.__setattr__(self, "totals", MappingProxyType(dict(self.totals)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "items": [dict(item) for item in self.items],
            "totals": dict(self.totals),
        }


@dataclass(frozen=True, slots=True)
class BacktestResultTradesPageReadModel:
    job_id: str
    variant_key: str
    variant_hash: str
    items: tuple[Mapping[str, Any], ...]
    pagination: Mapping[str, Any]
    summary: Mapping[str, Any]
    links: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "items",
            tuple(MappingProxyType(dict(item)) for item in self.items),
        )
        object.__setattr__(self, "pagination", MappingProxyType(dict(self.pagination)))
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))
        object.__setattr__(self, "links", MappingProxyType(dict(self.links)))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "variant_key": self.variant_key,
            "variant_hash": self.variant_hash,
            "items": [dict(item) for item in self.items],
            "pagination": dict(self.pagination),
            "summary": dict(self.summary),
            "links": dict(self.links),
        }


@dataclass(frozen=True, slots=True)
class BacktestTradesCsvReadModel:
    filename: str
    content: str
    media_type: str = "text/csv; charset=utf-8"


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
class BacktestJobCountersResult:
    active_jobs: int
    max_active_jobs: int
    max_active_jobs_global: int
    can_create: bool

    def as_mapping(self) -> dict[str, Any]:
        return {
            "active_jobs": self.active_jobs,
            "max_active_jobs": self.max_active_jobs,
            "max_active_jobs_global": self.max_active_jobs_global,
            "can_create": self.can_create,
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
    return BacktestJobReadModel(
        job_id=str(job.job_id),
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
        updated_at=job.updated_at,
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
        actions=_mapping_payload(payload.get("actions")),
    )


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
        "risk_mode": risk.get("mode") if isinstance(risk, Mapping) else None,
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


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "BacktestJobCreateResult",
    "BacktestJobListResult",
    "BacktestLazyTradesDetailReadModel",
    "BacktestJobProgressReadModel",
    "BacktestJobReadModel",
    "BacktestJobTopResult",
    "BacktestJobTopVariantReadModel",
    "BacktestResultMonthlyStatsReadModel",
    "BacktestResultSeriesReadModel",
    "BacktestResultSummaryReadModel",
    "BacktestResultSymbolStatsReadModel",
    "BacktestResultTradesPageReadModel",
    "BacktestTradesCsvReadModel",
    "build_backtest_job_read_model",
    "build_top_variant_read_model",
]
