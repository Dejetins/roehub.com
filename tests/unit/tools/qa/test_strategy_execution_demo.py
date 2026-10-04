"""The local demo must expose one status regardless of dashboard selection."""

import json

import pytest

from tools.qa.strategy_execution_demo import apply_execution_demo


@pytest.mark.parametrize("running,expected", [(True, "live"), (False, "stopped")])
def test_demo_status_in_another_strategy_dashboard(tmp_path, running, expected):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"running": running}))
    payload = {
        "selected_strategy": {"strategy_id": "other"},
        "runtime_status": {"producer_status": "stopped"},
        "strategy_selector": {
            "items": [
                {"strategy_id": "demo", "status": "stopped", "run_state": None},
                {"strategy_id": "other", "status": "stopped", "run_state": None},
            ],
            "totals": {"strategies": 2, "active": 0, "stopped": 2, "degraded": 0},
        },
    }
    result = apply_execution_demo(payload, "demo", state)
    row = result["strategy_selector"]["items"][0]
    assert row["status"] == expected
    assert row["run_state"] == ("running" if running else "stopped")
    assert result["runtime_status"]["producer_status"] == "stopped"
    assert result["strategy_selector"]["items"][1]["status"] == "stopped"
    assert result["strategy_selector"]["totals"]["active"] == int(running)
    assert result["strategy_selector"]["totals"]["stopped"] == 2 - int(running)
    # Reapplying the overlay must not double-count active strategies.
    totals = dict(result["strategy_selector"]["totals"])
    assert apply_execution_demo(result, "demo", state)["strategy_selector"]["totals"] == totals


def test_demo_does_not_add_an_inaccessible_strategy(tmp_path):
    payload = {"selected_strategy": {"strategy_id": "other"}, "strategy_selector": {
        "items": [], "totals": {"active": 0, "stopped": 0},
    }}
    assert apply_execution_demo(payload, "demo", tmp_path / "absent.json") == payload
    assert not (tmp_path / "absent.json").exists()
