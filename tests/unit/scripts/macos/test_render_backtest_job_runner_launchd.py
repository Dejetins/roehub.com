from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.macos import render_backtest_job_runner_launchd as render_module


def _write_template(repo_root: Path) -> None:
    """
    Create a minimal launchd template for renderer unit tests.

    Args:
        repo_root: Temporary repository root used by the test.
    Returns:
        None.
    Assumptions:
        Test template only needs placeholders asserted by the test cases.
    Raises:
        None.
    Side Effects:
        Writes a temporary template file under `infra/macos/launchd`.
    """
    template_path = (
        repo_root / "infra" / "macos" / "launchd" / "com.roehub.backtest-job-runner@.plist.template"
    )
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        (
            "label=${label}\n"
            "env=${roehub_env}\n"
            "config=${config_path}\n"
            "metrics=${metrics_port}\n"
            "instance=${instance_index}\n"
            "stdout=${stdout_path}\n"
            "stderr=${stderr_path}\n"
        ),
        encoding="utf-8",
    )


def test_materialize_worker_launchd_plists_renders_prod_fleet(monkeypatch, tmp_path: Path) -> None:
    """
    Verify prod launchd materialization renders one plist per configured worker instance.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary filesystem root.
    Returns:
        None.
    Assumptions:
        Enabled jobs should map directly to `worker_processes` launchd services.
    Raises:
        AssertionError: If rendered filenames or per-instance content are incorrect.
    Side Effects:
        Writes plist files into the temporary LaunchAgents directory.
    """
    repo_root = tmp_path / "repo"
    launch_agents_dir = tmp_path / "LaunchAgents"
    _write_template(repo_root)
    monkeypatch.setattr(
        render_module,
        "load_backtest_runtime_config",
        lambda _path: SimpleNamespace(
            jobs=SimpleNamespace(enabled=True, worker_processes=2),
        ),
    )

    rendered_paths = render_module.materialize_worker_launchd_plists(
        profile="prod",
        repo_root=repo_root,
        launch_agents_dir=launch_agents_dir,
        clean=True,
    )

    assert [path.name for path in rendered_paths] == [
        "com.roehub.backtest-job-runner.0.plist",
        "com.roehub.backtest-job-runner.1.plist",
    ]
    first_content = rendered_paths[0].read_text(encoding="utf-8")
    second_content = rendered_paths[1].read_text(encoding="utf-8")
    assert "label=com.roehub.backtest-job-runner.0" in first_content
    assert "metrics=9204" in first_content
    assert "instance=0" in first_content
    assert "label=com.roehub.backtest-job-runner.1" in second_content
    assert "instance=1" in second_content
    assert "stdout=/Users/daniildegtyarev/Library/Logs/roehub/backtest-job-runner.1.out.log" in (
        second_content
    )


def test_materialize_worker_launchd_plists_cleans_stale_files_when_jobs_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    Verify disabled jobs remove stale managed worker plists during clean materialization.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary filesystem root.
    Returns:
        None.
    Assumptions:
        Disabled jobs should materialize zero launchd worker services.
    Raises:
        AssertionError: If stale managed plists remain after clean render.
    Side Effects:
        Creates and deletes files inside the temporary LaunchAgents directory.
    """
    repo_root = tmp_path / "repo"
    launch_agents_dir = tmp_path / "LaunchAgents"
    _write_template(repo_root)
    launch_agents_dir.mkdir(parents=True, exist_ok=True)
    stale_path = launch_agents_dir / "com.roehub.test.backtest-job-runner.3.plist"
    stale_path.write_text("stale", encoding="utf-8")
    unrelated_path = launch_agents_dir / "com.roehub.api.plist"
    unrelated_path.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        render_module,
        "load_backtest_runtime_config",
        lambda _path: SimpleNamespace(
            jobs=SimpleNamespace(enabled=False, worker_processes=4),
        ),
    )

    rendered_paths = render_module.materialize_worker_launchd_plists(
        profile="test",
        repo_root=repo_root,
        launch_agents_dir=launch_agents_dir,
        clean=True,
    )

    assert rendered_paths == []
    assert not stale_path.exists()
    assert unrelated_path.exists()


def test_materialize_worker_launchd_plists_preserves_shell_parameter_expansions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """
    Verify the committed launchd template keeps shell `${...}` expansions after render.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        tmp_path: Temporary filesystem root for generated plist output.
    Returns:
        None.
    Assumptions:
        Python template substitution must not consume shell-side parameter expansion literals.
    Raises:
        AssertionError: If rendered plist loses shell env expressions or runtime placeholders.
    Side Effects:
        Writes one rendered plist into the temporary LaunchAgents directory.
    """
    repo_root = Path(__file__).resolve().parents[4]
    launch_agents_dir = tmp_path / "LaunchAgents"
    monkeypatch.setattr(
        render_module,
        "load_backtest_runtime_config",
        lambda _path: SimpleNamespace(
            jobs=SimpleNamespace(enabled=True, worker_processes=1),
        ),
    )

    rendered_paths = render_module.materialize_worker_launchd_plists(
        profile="prod",
        repo_root=repo_root,
        launch_agents_dir=launch_agents_dir,
        clean=True,
    )

    assert [path.name for path in rendered_paths] == ["com.roehub.backtest-job-runner.0.plist"]
    rendered_content = rendered_paths[0].read_text(encoding="utf-8")
    assert "export CH_DATABASE=${CH_DATABASE:-market_data}" in rendered_content
    assert "export CH_USER=${CH_USER:-${CLICKHOUSE_USER:-roe}}" in rendered_content
    assert "export CH_PASSWORD=${CH_PASSWORD:-${CLICKHOUSE_PASSWORD:-}}" in rendered_content
    assert "--metrics-port 9204 --instance-index 0" in rendered_content
