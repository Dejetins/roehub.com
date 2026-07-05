from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .live_entitlements import resolve_rl_live_ticker_limit
from .monitor_only_inference import (
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_V1,
    STAGE13_SOURCE_TYPE_V1,
    hash_json_payload_v1,
)

STAGE17_RUNTIME_LOAD_KIND_V1 = "rl_trading_stage17_multi_ticker_runtime_load_v1"
STAGE17_SCHEMA_VERSION_V1 = 1
STAGE17_MODE_INFRASTRUCTURE_ONLY_V1 = "infrastructure_only"
STAGE17_DEFAULT_MAX_FEED_LAG_SECONDS_V1 = 300.0
STAGE17_DEFAULT_EXECUTION_STREAMS_V1 = (
    "execution.requests.v1",
    "execution.requests.retry.v1",
    "execution.requests.dlq.v1",
)


@dataclass(frozen=True, slots=True)
class Stage17QuotaScenario:
    label: str
    paid_level: str
    product_label: str
    live_slots_allowed: int

    def __post_init__(self) -> None:
        _non_empty(self.label, "label")
        _non_empty(self.paid_level, "paid_level")
        _non_empty(self.product_label, "product_label")
        if self.live_slots_allowed <= 0:
            raise ValueError("Stage17QuotaScenario.live_slots_allowed must be positive")


@dataclass(frozen=True, slots=True)
class Stage17LoadObservation:
    scenario_label: str
    paid_level: str
    product_label: str
    live_slots_allowed: int
    exchange: str
    market_type: str
    symbol: str
    instrument_key: str
    feed_source: str
    feed_lag_seconds: float
    feature_window_rows: int
    redis_stream_length: int
    action_name: str
    outcome: str
    outcome_reason: str
    feature_hash: str
    source_event_ref: str
    latency_seconds: Mapping[str, float]

    def __post_init__(self) -> None:
        for field in (
            "scenario_label",
            "paid_level",
            "product_label",
            "exchange",
            "market_type",
            "symbol",
            "instrument_key",
            "feed_source",
            "action_name",
            "outcome",
            "outcome_reason",
            "feature_hash",
            "source_event_ref",
        ):
            _non_empty(str(getattr(self, field)), field)
        if self.live_slots_allowed <= 0:
            raise ValueError("Stage17LoadObservation.live_slots_allowed must be positive")
        if self.feature_window_rows <= 0:
            raise ValueError("Stage17LoadObservation.feature_window_rows must be positive")
        if self.redis_stream_length < 0:
            raise ValueError("Stage17LoadObservation.redis_stream_length must be non-negative")
        _non_negative_finite(self.feed_lag_seconds, "feed_lag_seconds")
        _validate_sha256(self.feature_hash, "feature_hash")
        required_segments = (
            "candle_close_to_feature_ready",
            "feature_to_decision",
            "decision_to_source_event",
        )
        missing_segments = [
            segment for segment in required_segments if segment not in self.latency_seconds
        ]
        if missing_segments:
            raise ValueError(f"Stage17LoadObservation.latency_seconds missing {missing_segments}")
        for segment, seconds in self.latency_seconds.items():
            _non_negative_finite(float(seconds), f"latency_seconds.{segment}")

    def to_summary_payload(self) -> dict[str, object]:
        return {
            "action_name": self.action_name,
            "exchange": self.exchange,
            "feature_hash": self.feature_hash,
            "feature_window_rows": self.feature_window_rows,
            "feed_lag_seconds": _round_float(self.feed_lag_seconds),
            "feed_source": self.feed_source,
            "instrument_key": self.instrument_key,
            "latency_seconds": {
                key: _round_float(float(value))
                for key, value in sorted(self.latency_seconds.items())
            },
            "live_slots_allowed": self.live_slots_allowed,
            "market_type": self.market_type,
            "outcome": self.outcome,
            "outcome_reason": self.outcome_reason,
            "paid_level": self.paid_level,
            "product_label": self.product_label,
            "redis_stream_length": self.redis_stream_length,
            "scenario_label": self.scenario_label,
            "source_event_ref": self.source_event_ref,
            "source_type": STAGE13_SOURCE_TYPE_V1,
            "symbol": self.symbol,
        }


def build_stage17_default_quota_scenarios_v1() -> tuple[Stage17QuotaScenario, ...]:
    scenarios: list[Stage17QuotaScenario] = []
    for label, paid_level in (
        ("free", "free"),
        ("pro", "pro"),
        ("premium", "ultra"),
    ):
        limit = resolve_rl_live_ticker_limit(paid_level=paid_level)
        scenarios.append(
            Stage17QuotaScenario(
                label=label,
                paid_level=limit.paid_level,
                product_label=limit.product_label,
                live_slots_allowed=limit.live_slots_allowed,
            )
        )
    return tuple(scenarios)


def summarize_stage17_runtime_load_v1(
    *,
    observations: Sequence[Stage17LoadObservation],
    quota_scenarios: Sequence[Stage17QuotaScenario],
    latency_budget_ms: Mapping[str, int],
    redis_stream_lengths_before: Mapping[str, int],
    redis_stream_lengths_after: Mapping[str, int],
    resource_usage: Mapping[str, object],
    contention: Mapping[str, object],
    max_feed_lag_seconds: float = STAGE17_DEFAULT_MAX_FEED_LAG_SECONDS_V1,
    generated_at_utc: str,
    prompt_path: str,
    prompt_sha256: str,
    git_revision: str,
    config_profile: str,
) -> dict[str, object]:
    if not observations:
        raise ValueError("Stage17 runtime load summary requires observations")
    if not quota_scenarios:
        raise ValueError("Stage17 runtime load summary requires quota_scenarios")
    _validate_sha256(prompt_sha256, "prompt_sha256")
    _non_empty(generated_at_utc, "generated_at_utc")
    _non_empty(prompt_path, "prompt_path")
    _non_empty(git_revision, "git_revision")
    _non_empty(config_profile, "config_profile")
    _non_negative_finite(max_feed_lag_seconds, "max_feed_lag_seconds")

    scenario_summaries = _scenario_summaries(
        observations=observations,
        quota_scenarios=quota_scenarios,
    )
    latency_p95_seconds = _latency_p95_seconds(observations)
    latency_p95_ms = {
        key: _round_float(value * 1000.0) for key, value in latency_p95_seconds.items()
    }
    redis_deltas = _stream_deltas(
        before=redis_stream_lengths_before,
        after=redis_stream_lengths_after,
    )
    feed_lags = [item.feed_lag_seconds for item in observations]
    rss_after = _optional_float(resource_usage.get("rss_mb_after"))
    max_rss = _optional_float(resource_usage.get("max_rss_mb"))
    wall_seconds = max(_optional_float(resource_usage.get("wall_time_seconds")) or 0.0, 0.0)
    throughput = len(observations) / wall_seconds if wall_seconds > 0.0 else 0.0

    checks = {
        "all_scenarios_met_requested_ticker_counts": all(
            _as_int(row["observed_tickers"]) >= _as_int(row["requested_live_tickers"])
            for row in scenario_summaries
        ),
        "monitor_only_source_events_only": all(
            item.outcome == STAGE13_SOURCE_EVENT_OUTCOME_V1
            and item.outcome_reason == STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1
            for item in observations
        ),
        "segment_latency_budget_met": all(
            latency_p95_ms.get(segment, math.inf) <= float(limit_ms)
            for segment, limit_ms in latency_budget_ms.items()
        ),
        "redis_execution_stream_growth_zero": all(delta == 0 for delta in redis_deltas.values()),
        "dlq_growth_zero": redis_deltas.get("execution.requests.dlq.v1", 0) == 0,
        "feed_lag_within_limit": max(feed_lags) <= max_feed_lag_seconds,
        "rss_budget_met": rss_after is None or max_rss is None or rss_after <= max_rss,
        "resource_contention_bounded_or_observed": str(contention.get("status"))
        in {"observed_overlap", "blocked_by_config", "not_applicable"},
    }
    status = "accepted" if all(checks.values()) else "blocked"
    payload = {
        "acceptance_checks": checks,
        "config_profile": config_profile,
        "contention": dict(contention),
        "decision_throughput_per_second": _round_float(throughput),
        "feed_lag": {
            "max_seconds": _round_float(max(feed_lags)),
            "p95_seconds": _round_float(_p95(feed_lags)),
            "threshold_seconds": _round_float(max_feed_lag_seconds),
        },
        "generated_at_utc": generated_at_utc,
        "git_revision": git_revision,
        "kind": STAGE17_RUNTIME_LOAD_KIND_V1,
        "latency_budget_ms": dict(latency_budget_ms),
        "latency_p95_ms": latency_p95_ms,
        "latency_p95_seconds": latency_p95_seconds,
        "mode": STAGE17_MODE_INFRASTRUCTURE_ONLY_V1,
        "observations": [item.to_summary_payload() for item in observations],
        "prompt_path": prompt_path,
        "prompt_sha256": prompt_sha256,
        "quota_scenarios": scenario_summaries,
        "redis_execution_streams": {
            "after": dict(sorted(redis_stream_lengths_after.items())),
            "before": dict(sorted(redis_stream_lengths_before.items())),
            "delta": redis_deltas,
        },
        "resource_usage": dict(resource_usage),
        "schema_version": STAGE17_SCHEMA_VERSION_V1,
        "stage": "17",
        "stage18_handoff": _stage18_handoff(status=status, scenario_summaries=scenario_summaries),
        "status": status,
    }
    payload["summary_hash"] = hash_json_payload_v1(
        {
            key: value
            for key, value in payload.items()
            if key not in {"summary_hash", "observations"}
        }
    )
    return payload


def _scenario_summaries(
    *,
    observations: Sequence[Stage17LoadObservation],
    quota_scenarios: Sequence[Stage17QuotaScenario],
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for scenario in quota_scenarios:
        items = [item for item in observations if item.scenario_label == scenario.label]
        summaries.append(
            {
                "label": scenario.label,
                "paid_level": scenario.paid_level,
                "product_label": scenario.product_label,
                "requested_live_tickers": scenario.live_slots_allowed,
                "observed_tickers": len({item.instrument_key for item in items}),
                "observation_count": len(items),
                "quota_bypass_observed": len({item.instrument_key for item in items})
                > scenario.live_slots_allowed,
            }
        )
    return summaries


def _latency_p95_seconds(
    observations: Sequence[Stage17LoadObservation],
) -> dict[str, float]:
    segments = sorted({segment for item in observations for segment in item.latency_seconds})
    return {
        segment: _round_float(_p95([float(item.latency_seconds[segment]) for item in observations]))
        for segment in segments
    }


def _stream_deltas(
    *,
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    keys = sorted(set(before.keys()) | set(after.keys()))
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in keys}


def _stage18_handoff(
    *,
    status: str,
    scenario_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    max_tickers = max(
        (_as_int(item["observed_tickers"]) for item in scenario_summaries),
        default=0,
    )
    if status != "accepted":
        return {
            "stage18_allowed": False,
            "reason": "stage17_runtime_load_blocked",
            "max_monitor_only_tickers_for_technical_soak": 0,
        }
    return {
        "stage18_allowed": True,
        "allowed_mode": "monitor_only_technical_soak",
        "forbidden_claims": [
            "model_quality",
            "trading_edge",
            "product_readiness",
            "mainnet_readiness",
        ],
        "max_monitor_only_tickers_for_technical_soak": max_tickers,
        "reason": "stage17_infrastructure_only_load_gate_accepted",
    }


def _p95(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("p95 requires at least one value")
    sorted_values = sorted(float(value) for value in values)
    index = max(0, math.ceil(0.95 * len(sorted_values)) - 1)
    return sorted_values[index]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float | str):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean value cannot be converted to int")
    if not isinstance(value, int | str):
        raise ValueError(f"expected int-like value, got {type(value).__name__}")
    return int(value)


def _round_float(value: float) -> float:
    return round(float(value), 12)


def _non_empty(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _non_negative_finite(value: float, field: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase sha256 hex digest")
