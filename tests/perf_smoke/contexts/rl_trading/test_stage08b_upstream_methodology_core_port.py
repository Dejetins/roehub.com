from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rl_trading.stage08b_upstream_methodology_core_smoke import main as stage08b_main


def test_stage08b_upstream_methodology_core_smoke_cli(tmp_path: Path, capsys) -> None:
    pytest.importorskip("torch")

    result = stage08b_main(
        [
            "--output-root",
            str(tmp_path),
            "--session-count",
            "3",
            "--episodes",
            "2",
            "--batch-size",
            "2",
            "--train-start",
            "2",
            "--target-update-freq",
            "1",
            "--replay-capacity",
            "64",
            "--torch-num-threads",
            "1",
            "--generated-at-utc",
            "2026-06-24T12:00:00Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["architecture_id"] == "roehub_d3qn_cnn_dueling_v1"
    assert payload["status"] == "accepted_smoke"
    assert payload["scripted_transition_sequence_used"] is False
    assert payload["learn_update_count"] > 0
    assert payload["target_sync_count"] > 0
    assert Path(payload["report_path"]).exists()
