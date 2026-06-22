from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence
from uuid import UUID

from trading.contexts.backtest.application.dto import (
    BacktestNoRiskTopResult,
    BacktestTpSlTopResult,
)
from trading.contexts.backtest.domain.entities import BacktestJobTopVariant

from .no_risk_funding import (
    FUNDING_ADJUSTMENT_EXACT_GLOBAL_RANKING,
    FUNDING_ADJUSTMENT_SCOPE,
    FUNDING_DATA_QUALITY,
    FUNDING_INCLUDED,
    FUNDING_WARNING_CODES,
)

TOP_RESULT_ASSEMBLY_STAGE_NAME = "top_result_assembly"


@dataclass(frozen=True, slots=True)
class BacktestTopResultAssemblyResult:
    top_variants: tuple[BacktestJobTopVariant, ...]
    stage_timings: Mapping[str, float]
    summary_hash: str

    def as_mapping(self) -> dict[str, Any]:
        return {
            "stage_timings": dict(self.stage_timings),
            "summary_hash": self.summary_hash,
            "top_variants_count": len(self.top_variants),
        }


@dataclass(frozen=True, slots=True)
class BacktestTopResultAssemblyService:
    """
    Build summary-only persisted top rows from accepted compact no-risk/TP-SL top results.

    The legacy storage `variant_key` remains the stable SHA identity. The public readable
    route key is stored inside `payload_json.public_variant_key` and mapped back by the API
    read model.
    """

    def assemble(
        self,
        *,
        job_id: UUID,
        normalized_request: Mapping[str, Any],
        top_results: Sequence[BacktestNoRiskTopResult | BacktestTpSlTopResult],
        updated_at: datetime,
    ) -> BacktestTopResultAssemblyResult:
        start = time.perf_counter()
        rows = tuple(
            _build_top_variant(
                job_id=job_id,
                normalized_request=normalized_request,
                top_result=top_result,
                updated_at=updated_at,
            )
            for top_result in sorted(top_results, key=lambda item: item.rank)
        )
        elapsed = time.perf_counter() - start
        summary_payload = [
            {
                "rank": row.rank,
                "variant_hash": row.variant_key,
                "public_variant_key": row.payload_json["public_variant_key"],
                "summary_metrics": dict(row.summary_metrics_json),
                "best_tp_pct": row.best_tp_pct,
                "best_sl_pct": row.best_sl_pct,
            }
            for row in rows
        ]
        return BacktestTopResultAssemblyResult(
            top_variants=rows,
            stage_timings={TOP_RESULT_ASSEMBLY_STAGE_NAME: elapsed},
            summary_hash=_canonical_sha256(summary_payload),
        )


def _build_top_variant(
    *,
    job_id: UUID,
    normalized_request: Mapping[str, Any],
    top_result: BacktestNoRiskTopResult | BacktestTpSlTopResult,
    updated_at: datetime,
) -> BacktestJobTopVariant:
    indicators = _indicator_params(top_result=top_result)
    summary_metrics = dict(top_result.metrics)
    best_tp_pct = _optional_float(summary_metrics.get("best_tp_pct"))
    best_sl_pct = _optional_float(summary_metrics.get("best_sl_pct"))
    risk_payload = _mapping_payload(normalized_request.get("risk"))
    if _risk_side_disabled(risk_payload, "tp"):
        best_tp_pct = None
    if _risk_side_disabled(risk_payload, "sl"):
        best_sl_pct = None
    canonical_params = {
        "schema_version": 1,
        "indicators": indicators,
        "risk": _risk_params(
            normalized_request=normalized_request,
            best_tp_pct=best_tp_pct,
            best_sl_pct=best_sl_pct,
        ),
        "execution": _mapping_payload(normalized_request.get("execution")),
        "ranking": _mapping_payload(normalized_request.get("ranking")),
    }
    indicator_variant_hash = _canonical_sha256(
        {
            "schema_version": 1,
            "indicators": indicators,
        }
    )
    variant_hash = _canonical_sha256(canonical_params)
    readable_slug = _readable_slug(
        indicators=indicators,
        risk_mode=str(risk_payload.get("mode", "none")),
        best_tp_pct=best_tp_pct,
        best_sl_pct=best_sl_pct,
    )
    job_short = f"{job_id.hex[:8]}{job_id.hex[-4:]}"
    public_variant_key = f"job_{job_short}__{readable_slug}__vh_{variant_hash[:8]}"
    readable_params = {
        "slug": readable_slug,
        "indicators": [
            {
                "indicator_id": item["indicator_id"],
                "source": item.get("source"),
                "window": item.get("window"),
            }
            for item in indicators
        ],
        "risk_mode": risk_payload.get("mode"),
        "best_tp_pct": best_tp_pct,
        "best_sl_pct": best_sl_pct,
    }
    funding_adjustment = _funding_adjustment_payload(metadata=top_result.metadata)
    payload = {
        "schema": "backtest_top_variant_summary_v1",
        "public_variant_key": public_variant_key,
        "variant_hash": variant_hash,
        "indicator_variant_hash": indicator_variant_hash,
        "canonical_variant_params": canonical_params,
        "readable_params": readable_params,
        "links": {
            "variant": f"/backtests/jobs/{job_id}/variants/{public_variant_key}",
            "lazy_trades": f"/backtests/jobs/{job_id}/variants/{public_variant_key}/trades",
        },
        "actions": {
            "lazy_trades": {
                "available": False,
                "status": "not_implemented",
                "iteration": 9,
            }
        },
        "source_top_result": _normalize_json(top_result.as_mapping()),
    }
    if funding_adjustment:
        payload["funding_adjustment"] = funding_adjustment
    return BacktestJobTopVariant(
        job_id=job_id,
        rank=top_result.rank,
        variant_key=variant_hash,
        indicator_variant_key=indicator_variant_hash,
        variant_index=top_result.rank - 1,
        total_return_pct=float(summary_metrics["total_return_pct"]),
        payload_json=payload,
        summary_metrics_json=summary_metrics,
        best_tp_pct=best_tp_pct,
        best_sl_pct=best_sl_pct,
        report_table_md=None,
        trades_json=None,
        updated_at=updated_at,
    )


def _indicator_params(
    *,
    top_result: BacktestNoRiskTopResult | BacktestTpSlTopResult,
) -> list[dict[str, Any]]:
    rows = dict(top_result.indicator_rows)
    metadata = dict(top_result.metadata)
    out: list[dict[str, Any]] = []
    for indicator_id, row_id in rows.items():
        item = {
            "indicator_id": indicator_id,
            "row_id": int(row_id),
            "source": metadata.get(f"{indicator_id}.source"),
            "window": metadata.get(f"{indicator_id}.window"),
        }
        out.append(item)
    return out


def _risk_params(
    *,
    normalized_request: Mapping[str, Any],
    best_tp_pct: float | None,
    best_sl_pct: float | None,
) -> dict[str, Any]:
    risk = _mapping_payload(normalized_request.get("risk"))
    if best_tp_pct is not None:
        risk["best_tp_pct"] = best_tp_pct
    if best_sl_pct is not None:
        risk["best_sl_pct"] = best_sl_pct
    return risk


def _readable_slug(
    *,
    indicators: Sequence[Mapping[str, Any]],
    risk_mode: str,
    best_tp_pct: float | None,
    best_sl_pct: float | None,
) -> str:
    parts: list[str] = []
    for item in indicators:
        indicator_name = str(item["indicator_id"]).split(".")[-1]
        source = item.get("source") or "src"
        window = item.get("window")
        parts.append(_slug_token(f"{indicator_name}_{source}_w{window}"))
    parts.append(_slug_token(f"risk_{risk_mode}"))
    if best_tp_pct is not None:
        parts.append(_slug_token(f"tp_{best_tp_pct:g}"))
    if best_sl_pct is not None:
        parts.append(_slug_token(f"sl_{best_sl_pct:g}"))
    slug = "__".join(part for part in parts if part)
    return slug[:96] if slug else "variant"


def _slug_token(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "x"


def _canonical_sha256(payload: Any) -> str:
    rendered = json.dumps(
        _normalize_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _normalize_json(item())
    return str(value)


def _mapping_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _risk_side_disabled(risk: Mapping[str, Any], side: str) -> bool:
    return _mapping_payload(risk.get(side)).get("enabled") is False


def _funding_adjustment_payload(*, metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        FUNDING_INCLUDED,
        FUNDING_DATA_QUALITY,
        FUNDING_WARNING_CODES,
        FUNDING_ADJUSTMENT_SCOPE,
        FUNDING_ADJUSTMENT_EXACT_GLOBAL_RANKING,
        "requested_ranking_metric",
        "effective_ranking_metric",
        "funding_candidate_pool_size",
        "requested_top_n",
        "funding_manifest_hash",
    )
    payload = {key: metadata[key] for key in keys if key in metadata}
    return _normalize_json(payload) if payload else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


__all__ = [
    "TOP_RESULT_ASSEMBLY_STAGE_NAME",
    "BacktestTopResultAssemblyResult",
    "BacktestTopResultAssemblyService",
]
