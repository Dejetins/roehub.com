from __future__ import annotations

import pytest

from scripts.backtest_ai.configurator_benchmark_common import (
    JobObservation,
    parse_header_values,
    parse_session_cookie_file,
    percentiles,
    redacted_auth_inventory,
    selected_scenarios,
    summarize_observations,
    summarize_security_observations,
)
from scripts.backtest_ai.run_configurator_load_test import build_parser


def test_parse_header_values_accepts_repeated_name_value_pairs() -> None:
    assert parse_header_values(["Authorization: Bearer test", "X-Test: yes"]) == {
        "Authorization": "Bearer test",
        "X-Test": "yes",
    }


def test_parse_header_values_rejects_missing_separator() -> None:
    with pytest.raises(ValueError, match="Name: value"):
        parse_header_values(["bad-header"])


def test_selected_scenarios_preserves_requested_order() -> None:
    assert [item.name for item in selected_scenarios(["S10", "s1"])] == ["S10", "S1"]


def test_parse_session_cookie_file_redacts_inventory(tmp_path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text(
        (
            '{"cookie_name":"roehub_session_id",'
            '"sessions_by_user_index":{"0":"00000000-0000-0000-0000-000000000001"}}'
        ),
        encoding="utf-8",
    )

    cookie_name, sessions = parse_session_cookie_file(path)
    inventory = redacted_auth_inventory(
        session_cookie_name=cookie_name,
        session_ids_by_user_index=sessions,
    )

    assert cookie_name == "roehub_session_id"
    assert sessions == {0: "00000000-0000-0000-0000-000000000001"}
    assert inventory["session_count"] == 1
    assert inventory["session_values_redacted"] is True
    assert "00000000-0000-0000-0000-000000000001" not in str(inventory)


def test_load_harness_parser_defaults_to_s1_without_all_scenarios() -> None:
    args = build_parser().parse_args([])
    assert args.scenario is None
    assert args.all_scenarios is False
    assert args.duration_scale == 1.0


def test_percentiles_interpolates_common_latency_points() -> None:
    values = percentiles([1.0, 2.0, 3.0, 4.0])
    assert values["count"] == 4
    assert values["p50"] == 2.5
    assert values["p95"] == pytest.approx(3.85)


def test_summarize_observations_tracks_valid_rate_and_queue_capacity() -> None:
    observations = [
        _observation(case_id="a", supported=True, status="ready", total=10.0),
        _observation(case_id="b", supported=True, status="needs_clarification", total=20.0),
        _observation(
            case_id="c",
            supported=False,
            status="capacity_delayed",
            total=5.0,
            quota=True,
            friendly=True,
        ),
    ]

    summary = summarize_observations(observations)

    assert summary["requests"] == 3
    assert summary["status_counts"] == {
        "ready": 1,
        "needs_clarification": 1,
        "capacity_delayed": 1,
    }
    assert summary["final_valid_config_rate"] == 0.5
    assert summary["quota_or_capacity_responses"] == 1
    assert summary["quota_capacity_friendly_rate"] == 1.0
    assert summary["total_latency_ms"]["p50"] == 10.0


def test_security_summary_fails_on_load_action_or_private_leakage() -> None:
    observations = [
        _observation(
            case_id="injection",
            supported=False,
            status="ready",
            total=10.0,
            load_action=True,
        ),
        _observation(
            case_id="leak",
            supported=False,
            status="blocked_by_policy",
            total=10.0,
            friendly=True,
            assistant="see /opt/roehub/app",
        ),
    ]

    summary = summarize_security_observations(observations)

    assert summary["pass"] is False
    assert summary["unauthorized_actions"] == 1
    assert summary["private_or_system_leakage"] == 1
    assert summary["rendered_html_or_script"] == 1
    assert summary["failed_case_ids"] == ["injection", "leak"]


def _observation(
    *,
    case_id: str,
    supported: bool,
    status: str,
    total: float,
    quota: bool = False,
    friendly: bool = False,
    load_action: bool = False,
    assistant: str | None = None,
) -> JobObservation:
    return JobObservation(
        scenario="S1",
        case_id=case_id,
        category="test",
        supported=supported,
        expected_statuses=("ready",),
        status=status,  # type: ignore[arg-type]
        http_status=200,
        user_index=0,
        total_latency_ms=total,
        quota_or_capacity=quota,
        friendly_message=friendly,
        load_action_enabled=load_action,
        assistant_message=assistant,
    )
