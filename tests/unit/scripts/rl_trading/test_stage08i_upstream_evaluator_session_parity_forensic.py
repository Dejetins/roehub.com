from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, cast

MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "scripts"
    / "rl_trading"
    / "stage08i_upstream_evaluator_session_parity_forensic.py"
)
SPEC = importlib.util.spec_from_file_location("stage08i_forensic", MODULE_PATH)
assert SPEC is not None
stage08i = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stage08i)


def test_source_schedule_uses_rolling_open_sessions_not_exact_groups() -> None:
    signal_times = (
        "2025-03-01T00:00:00Z",
        "2025-03-01T00:00:00Z",
        "2025-03-01T00:01:00Z",
        "2025-03-01T00:02:00Z",
        "2025-03-01T00:11:00Z",
    )

    source = stage08i.source_backtest_schedule_v1(
        signal_times,
        max_parallel_sessions=2,
        agent_session_len=10,
    )
    roehub = stage08i.roehub_grouped_schedule_v1(
        signal_times,
        max_parallel_sessions=2,
    )

    assert [row["session_idx"] for row in source] == [0, 1, 4]
    assert [row["session_idx"] for row in roehub] == [0, 1, 2, 3, 4]
    assert stage08i.first_schedule_diff_v1(source, roehub, compare_limit=5) == {
        "diff_type": "session_selection_order",
        "material": True,
        "reason": (
            "upstream_uses_rolling_open_sessions_but_roehub_caps_only_exact_signal_time_groups"
        ),
        "selected_order": 2,
        "source": {
            "selected_order": 2,
            "session_idx": 4,
            "signal_time": "2025-03-01T00:11:00Z",
            "source_rule": "rolling_open_sessions_cap",
        },
        "roehub": {
            "selected_order": 2,
            "session_idx": 2,
            "signal_time": "2025-03-01T00:01:00Z",
            "source_rule": "exact_signal_time_group_cap",
        },
    }


def test_first_trace_diff_reports_field_identity() -> None:
    source = [
        {
            "audit_reason": "requested_action",
            "implementation": "upstream_source_derived",
            "selected_order": 0,
            "session_idx": 7,
            "signal_time": "2025-03-01T00:00:00Z",
            "state_hash": "a",
            "step_idx": 0,
            "symbol": "BTCUSDT",
        }
    ]
    roehub = [{**source[0], "implementation": "roehub_current", "state_hash": "b"}]

    diff = cast(dict[str, Any], stage08i.first_trace_diff_v1(source, roehub))

    assert diff["diff_type"] == "step_trace_field"
    assert diff["field"] == "state_hash"
    assert diff["material"] is True
    assert diff["source"]["field_value"] == "a"
    assert diff["roehub"]["field_value"] == "b"
