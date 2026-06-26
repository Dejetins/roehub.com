from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence

from scripts.rl_trading import stage08g_dual_branch_cpu_training_evaluation as stage08g


def test_stage08g_dual_branch_dry_run_builds_sequential_cpu_commands(
    tmp_path: Path,
    capsys,
) -> None:
    result = stage08g.main(
        [
            "--dry-run",
            "--output-root",
            str(tmp_path / "runs"),
            "--hf-training-output-root",
            str(tmp_path / "hf_training"),
            "--native-training-output-root",
            str(tmp_path / "native_training"),
            "--evaluation-output-root",
            str(tmp_path / "evaluation"),
            "--generated-at-utc",
            "2026-06-26T12:00:00Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    commands = [step["command"] for step in summary["steps"]]

    assert result == 0
    assert summary["status"] == "planned"
    assert [step["name"] for step in summary["steps"]] == [
        "hf_original_cpu_training",
        "hf_original_cpu_optuna",
        "roehub_native_cpu_training",
        "roehub_native_cpu_optuna",
    ]
    assert "stage08c_original_hf_full_training_run.py run" in commands[0]
    assert "--device-policy cpu_only_deterministic" in commands[0]
    assert "stage08g_cpu_optuna_calibration.py --branch hf_original" in commands[1]
    assert "--trials 100 --jobs 1" in commands[1]
    assert "stage08e_roehub_native_full_training_run.py run" in commands[2]
    assert "--device-policy cpu_only_deterministic" in commands[2]
    assert "stage08g_cpu_optuna_calibration.py --branch roehub_native" in commands[3]
    assert summary["methodology"]["parallel_training"] is False
    assert summary["stage09_allowed"] is False


def test_stage08g_dual_branch_run_passes_fresh_candidate_sha_to_optuna(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    commands: list[Sequence[str]] = []

    def fake_run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        commands.append(tuple(command))
        command_text = " ".join(command)
        if "stage08c_original_hf_full_training_run.py" in command_text:
            return _completed_training(
                tmp_path=tmp_path,
                branch="hf_original",
                manifest_name="hf_original_candidate_manifest.json",
            )
        if "stage08e_roehub_native_full_training_run.py" in command_text:
            return _completed_training(
                tmp_path=tmp_path,
                branch="roehub_native",
                manifest_name="roehub_native_candidate_manifest.json",
            )
        if "stage08g_cpu_optuna_calibration.py" in command_text:
            return _completed_optuna(tmp_path=tmp_path, command=command)
        raise AssertionError(command_text)

    monkeypatch.setattr(stage08g, "_run_command_capture", fake_run)

    result = stage08g.main(
        [
            "--output-root",
            str(tmp_path / "runs"),
            "--hf-training-output-root",
            str(tmp_path / "hf_training"),
            "--native-training-output-root",
            str(tmp_path / "native_training"),
            "--evaluation-output-root",
            str(tmp_path / "evaluation"),
            "--episodes",
            "2",
            "--trials",
            "1",
            "--generated-at-utc",
            "2026-06-26T12:00:00Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    hf_sha = stage08g._file_sha256_hex(  # noqa: SLF001
        tmp_path / "hf_original" / "hf_original_candidate_manifest.json"
    )
    native_sha = stage08g._file_sha256_hex(  # noqa: SLF001
        tmp_path / "roehub_native" / "roehub_native_candidate_manifest.json"
    )
    optuna_commands = [
        " ".join(command)
        for command in commands
        if "stage08g_cpu_optuna_calibration.py" in " ".join(command)
    ]

    assert result == 0
    assert summary["status"] == "accepted_for_research"
    assert summary["stage09_allowed"] is True
    assert f"--expected-candidate-manifest-sha256 {hf_sha}" in optuna_commands[0]
    assert f"--expected-candidate-manifest-sha256 {native_sha}" in optuna_commands[1]
    assert summary["branches"]["hf_original"]["stage09_allowed"] is True
    assert summary["branches"]["roehub_native"]["stage09_allowed"] is True


def test_stage08g_dual_branch_dry_run_can_label_corrective_stage08h(
    tmp_path: Path,
    capsys,
) -> None:
    result = stage08g.main(
        [
            "--dry-run",
            "--stage-label",
            "08H",
            "--output-root",
            str(tmp_path / "runs"),
            "--generated-at-utc",
            "2026-06-26T12:00:00Z",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    summary = json.loads(Path(payload["summary_path"]).read_text(encoding="utf-8"))
    commands = [step["command"] for step in summary["steps"]]

    assert result == 0
    assert summary["artifact_kind"] == "rl_trading_stage08h_dual_branch_cpu_run"
    assert summary["stage"] == "08H"
    assert (
        "stage08g_cpu_optuna_calibration.py --branch hf_original --stage-label 08H"
        in commands[1]
    )
    assert (
        "stage08g_cpu_optuna_calibration.py --branch roehub_native --stage-label 08H"
        in commands[3]
    )


def _completed_training(
    *,
    tmp_path: Path,
    branch: str,
    manifest_name: str,
) -> subprocess.CompletedProcess[str]:
    branch_dir = tmp_path / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = branch_dir / manifest_name
    manifest_path.write_text(json.dumps({"branch": branch}, sort_keys=True), encoding="utf-8")
    payload = {
        "candidate_manifest_path": str(manifest_path),
        "run_dir": str(branch_dir),
        "run_id": branch,
        "status": "completed",
    }
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(payload) + "\n",
        stderr="",
    )


def _completed_optuna(
    *,
    tmp_path: Path,
    command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    branch = command[command.index("--branch") + 1]
    summary_path = tmp_path / f"{branch}_optuna_summary.json"
    summary_path.write_text(json.dumps({"branch": branch}, sort_keys=True), encoding="utf-8")
    payload = {
        "branch": branch,
        "run_dir": str(summary_path.parent),
        "run_id": f"{branch}_optuna",
        "status": "accepted_for_research",
        "summary_path": str(summary_path),
    }
    return subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=json.dumps(payload) + "\n",
        stderr="",
    )
