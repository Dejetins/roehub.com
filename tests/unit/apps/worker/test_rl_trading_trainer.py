from __future__ import annotations

import json

from apps.worker.rl_trading_trainer.main import main


def test_rl_trading_trainer_entrypoint_fails_closed_for_non_training_source(
    capsys,
) -> None:
    result = main(["--exchange", "bybit", "--market-type", "spot"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["status"] == "blocked"
    assert payload["reason"] == "blocked_not_training_source_v1"


def test_rl_trading_trainer_entrypoint_dispatches_stage07b_status(tmp_path, capsys) -> None:
    result = main(["stage07b", "status", "--run-dir", str(tmp_path / "missing-run")])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 2
    assert payload["status"] == "blocked"
    assert payload["reason"] == "latest_status_missing"
