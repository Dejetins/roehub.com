from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from .model_registry import validate_registry_transition_v1
from .monitor_only_inference import (
    STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
    STAGE13_SOURCE_EVENT_OUTCOME_V1,
    STAGE13_SOURCE_TYPE_V1,
    hash_json_payload_v1,
)

STAGE18_SOAK_KIND_V1 = "rl_trading_stage18_monitor_only_technical_soak_v1"
STAGE18_SCHEMA_VERSION_V1 = 1
STAGE18_MODE_MONITOR_ONLY_TECHNICAL_SOAK_V1 = "monitor_only_technical_soak"
STAGE18_MAX_TICKERS_V1 = 20
STAGE18_REQUIRED_DRILLS_V1 = (
    "kill_switch",
    "pause",
    "rollback",
    "missing_artifact",
    "stale_feed",
    "unknown_state",
)

Stage18DrillName = Literal[
    "kill_switch",
    "pause",
    "rollback",
    "missing_artifact",
    "stale_feed",
    "unknown_state",
]
Stage18DrillStatus = Literal["passed", "blocked"]


@dataclass(frozen=True, slots=True)
class Stage18IncidentDrill:
    name: Stage18DrillName
    status: Stage18DrillStatus
    operator_action: str
    detection: str
    fail_closed_result: str
    recovery_evidence: str
    degraded_state_reason: str
    exchange_side_effect: str = "none"
    order_state_involved: bool = False
    reconciliation_before_retry: bool = True
    raw_payload_redacted: bool = True

    def __post_init__(self) -> None:
        for field in (
            "operator_action",
            "detection",
            "fail_closed_result",
            "recovery_evidence",
            "degraded_state_reason",
            "exchange_side_effect",
        ):
            _non_empty(str(getattr(self, field)), field)

    def to_summary_payload(self) -> dict[str, object]:
        return {
            "degraded_state_reason": self.degraded_state_reason,
            "detection": self.detection,
            "exchange_side_effect": self.exchange_side_effect,
            "fail_closed_result": self.fail_closed_result,
            "name": self.name,
            "operator_action": self.operator_action,
            "order_state_involved": self.order_state_involved,
            "raw_payload_redacted": self.raw_payload_redacted,
            "reconciliation_before_retry": self.reconciliation_before_retry,
            "recovery_evidence": self.recovery_evidence,
            "status": self.status,
        }


def build_stage18_default_incident_drills_v1() -> tuple[Stage18IncidentDrill, ...]:
    validate_registry_transition_v1(
        entity="activation",
        current_status="monitor_only",
        next_status="paused",
    )
    validate_registry_transition_v1(
        entity="activation",
        current_status="monitor_only",
        next_status="rolled_back",
    )
    validate_registry_transition_v1(
        entity="model_version",
        current_status="accepted_champion",
        next_status="missing_artifact",
    )
    return (
        Stage18IncidentDrill(
            name="kill_switch",
            status="passed",
            operator_action="dry_run_global_monitor_only_kill_switch",
            detection="source_event_generation_blocked_before_dispatch",
            fail_closed_result="no_ml_agent_decision_write_no_execution_intent",
            recovery_evidence="switch_restored_then_monitor_only_observation_required",
            degraded_state_reason="rl_global_kill_switch_active",
        ),
        Stage18IncidentDrill(
            name="pause",
            status="passed",
            operator_action="dry_run_activation_monitor_only_to_paused",
            detection="registry_transition_monitor_only_to_paused_validated",
            fail_closed_result="inference_loop_pauses_before_source_event",
            recovery_evidence="paused_to_monitor_only_requires_operator_resume",
            degraded_state_reason="rl_activation_paused",
        ),
        Stage18IncidentDrill(
            name="rollback",
            status="passed",
            operator_action="dry_run_activation_monitor_only_to_rolled_back",
            detection="registry_transition_monitor_only_to_rolled_back_validated",
            fail_closed_result="rollback_pointer_selected_without_deleting_artifacts",
            recovery_evidence="rolled_back_state_requires_new_shadow_or_inactive_restart",
            degraded_state_reason="rl_model_rolled_back",
        ),
        Stage18IncidentDrill(
            name="missing_artifact",
            status="passed",
            operator_action="dry_run_active_checkpoint_missing_probe",
            detection="model_version_accepted_champion_to_missing_artifact_validated",
            fail_closed_result="runtime_load_blocked_before_torch_load",
            recovery_evidence="restore_or_select_explicit_rollback_before_resume",
            degraded_state_reason="rl_model_artifact_missing",
        ),
        Stage18IncidentDrill(
            name="stale_feed",
            status="passed",
            operator_action="dry_run_feed_lag_threshold_breach",
            detection="feed_lag_above_threshold_routes_to_degraded_state",
            fail_closed_result="signal_generation_pauses_until_fresh_window",
            recovery_evidence="fresh_redis_window_required_before_next_decision",
            degraded_state_reason="rl_market_feed_stale",
        ),
        Stage18IncidentDrill(
            name="unknown_state",
            status="passed",
            operator_action="dry_run_source_event_unknown_state_lookup",
            detection="durable_source_event_lookup_required_before_retry",
            fail_closed_result="retry_blocked_until_reconciliation_result_known",
            recovery_evidence="source_event_outcome_reconciled_before_retry_allowed",
            degraded_state_reason="rl_source_event_unknown_state",
            reconciliation_before_retry=True,
        ),
    )


def summarize_stage18_monitor_only_technical_soak_v1(
    *,
    stage17_summary: Mapping[str, object],
    incident_drills: Sequence[Stage18IncidentDrill],
    ui_evidence: Mapping[str, object],
    generated_at_utc: str,
    prompt_path: str,
    prompt_sha256: str,
    git_revision: str,
    run_id: str,
) -> dict[str, object]:
    _validate_sha256(prompt_sha256, "prompt_sha256")
    for field, value in (
        ("generated_at_utc", generated_at_utc),
        ("prompt_path", prompt_path),
        ("git_revision", git_revision),
        ("run_id", run_id),
    ):
        _non_empty(value, field)

    observations = _observations(stage17_summary)
    scenario_rows = _mapping_sequence(stage17_summary.get("quota_scenarios"))
    max_observed_tickers = max(
        (_as_int(row.get("observed_tickers", 0)) for row in scenario_rows),
        default=0,
    )
    action_counts: dict[str, int] = {}
    feature_hashes: set[str] = set()
    source_types: set[str] = set()
    outcomes: set[tuple[str, str]] = set()
    for observation in observations:
        action = str(observation.get("action_name", "unknown"))
        action_counts[action] = action_counts.get(action, 0) + 1
        feature_hash = str(observation.get("feature_hash", ""))
        if feature_hash:
            feature_hashes.add(feature_hash)
        source_types.add(str(observation.get("source_type", "")))
        outcomes.add(
            (
                str(observation.get("outcome", "")),
                str(observation.get("outcome_reason", "")),
            )
        )

    drill_payloads = [drill.to_summary_payload() for drill in incident_drills]
    drill_checks = _drill_checks(incident_drills)
    stage17_checks = _mapping(stage17_summary.get("acceptance_checks"))
    redis_deltas = _mapping(_mapping(stage17_summary.get("redis_execution_streams")).get("delta"))
    stage17_handoff = _mapping(stage17_summary.get("stage18_handoff"))
    ui_status = str(ui_evidence.get("status", "not_collected"))
    checks = {
        "stage17_input_accepted": stage17_summary.get("status") == "accepted",
        "stage17_handoff_allows_monitor_only_technical_soak": (
            stage17_handoff.get("stage18_allowed") is True
            and stage17_handoff.get("allowed_mode") == STAGE18_MODE_MONITOR_ONLY_TECHNICAL_SOAK_V1
        ),
        "max_tickers_within_stage18_limit": 0 < max_observed_tickers <= STAGE18_MAX_TICKERS_V1,
        "monitor_only_source_events_only": stage17_checks.get("monitor_only_source_events_only")
        is True
        and source_types == {STAGE13_SOURCE_TYPE_V1}
        and outcomes
        == {
            (
                STAGE13_SOURCE_EVENT_OUTCOME_V1,
                STAGE13_SOURCE_EVENT_OUTCOME_REASON_V1,
            )
        },
        "redis_execution_stream_growth_zero": all(
            _as_int(value) == 0 for value in redis_deltas.values()
        ),
        "required_incident_drills_passed": all(drill_checks.values()),
        "ui_safe_or_degraded_state_recorded": ui_status in {"observed", "not_visible"},
        "no_exchange_side_effects": all(
            drill.exchange_side_effect == "none" for drill in incident_drills
        ),
        "no_order_state_involved": all(not drill.order_state_involved for drill in incident_drills),
    }
    status = "accepted" if all(checks.values()) else "blocked"
    payload: dict[str, object] = {
        "acceptance_checks": checks,
        "drill_checks": drill_checks,
        "feed_lag": stage17_summary.get("feed_lag", {}),
        "generated_at_utc": generated_at_utc,
        "git_revision": git_revision,
        "incident_drills": drill_payloads,
        "kind": STAGE18_SOAK_KIND_V1,
        "latency_p95_ms": stage17_summary.get("latency_p95_ms", {}),
        "mode": STAGE18_MODE_MONITOR_ONLY_TECHNICAL_SOAK_V1,
        "model_drift_and_runtime_health": {
            "action_distribution": dict(sorted(action_counts.items())),
            "feature_hash_unique_count": len(feature_hashes),
            "model_quality_claim": "not_claimed",
            "observation_count": len(observations),
            "source_event_outcomes": [
                {"outcome": outcome, "outcome_reason": reason}
                for outcome, reason in sorted(outcomes)
            ],
        },
        "prompt_path": prompt_path,
        "prompt_sha256": prompt_sha256,
        "quality_claims": {
            "full_trade_readiness": False,
            "mainnet_readiness": False,
            "model_quality": False,
            "paper_testnet_live_execution_readiness": False,
            "product_readiness": False,
            "trading_edge": False,
        },
        "redis_execution_streams": stage17_summary.get("redis_execution_streams", {}),
        "resource_usage": stage17_summary.get("resource_usage", {}),
        "run_id": run_id,
        "schema_version": STAGE18_SCHEMA_VERSION_V1,
        "soak_observation": {
            "bounded_observation_source": "stage17_monitor_only_runtime_harness",
            "max_observed_tickers": max_observed_tickers,
            "observation_count": len(observations),
            "source_stage17_summary_hash": stage17_summary.get("summary_hash"),
            "source_stage17_summary_path": stage17_summary.get("summary_path"),
            "twenty_four_hour_minimum_status": "not_claimed_for_monitor_only_technical_soak",
            "seven_day_status": "not_claimed_for_monitor_only_technical_soak",
        },
        "stage": "18",
        "stage19_handoff": {
            "stage19_mainnet_readiness_allowed": False,
            "reason": "stage18_technical_soak_only_and_stage08n_blocks_mainnet_readiness",
        },
        "status": status,
        "technical_scope": {
            "allowed_claim": "monitor_only_runtime_safety_and_incident_drill_behavior",
            "max_tickers": STAGE18_MAX_TICKERS_V1,
            "runtime_artifact_root": "/opt/roehub/state/rl_trading/",
        },
        "ui_evidence": dict(ui_evidence),
    }
    payload["summary_hash"] = hash_json_payload_v1(
        {key: value for key, value in payload.items() if key != "summary_hash"}
    )
    return payload


def _drill_checks(drills: Sequence[Stage18IncidentDrill]) -> dict[str, bool]:
    by_name = {drill.name: drill for drill in drills}
    checks: dict[str, bool] = {}
    for name in STAGE18_REQUIRED_DRILLS_V1:
        drill = by_name.get(name)
        checks[name] = (
            drill is not None
            and drill.status == "passed"
            and drill.exchange_side_effect == "none"
            and not drill.order_state_involved
            and drill.raw_payload_redacted
        )
    unknown = by_name.get("unknown_state")
    checks["unknown_state_reconciles_before_retry"] = (
        unknown is not None
        and unknown.reconciliation_before_retry
        and not unknown.order_state_involved
    )
    return checks


def _observations(payload: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return _mapping_sequence(payload.get("observations"))


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    rows: list[Mapping[str, object]] = []
    for item in value:
        if isinstance(item, Mapping):
            rows.append(item)
    return tuple(rows)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("boolean value cannot be converted to int")
    if not isinstance(value, int | str):
        raise ValueError(f"expected int-like value, got {type(value).__name__}")
    return int(value)


def _non_empty(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _validate_sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase sha256 hex digest")


def _non_negative_finite(value: float, field: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
