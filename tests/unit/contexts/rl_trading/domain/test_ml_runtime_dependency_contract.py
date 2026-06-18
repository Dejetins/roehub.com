from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]


def test_torch_is_only_in_rl_ml_optional_extra() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    default_dependencies = pyproject["project"]["dependencies"]
    assert all(not dependency.startswith("torch") for dependency in default_dependencies)

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    assert "rl-ml" in optional_dependencies
    assert any(dependency.startswith("torch") for dependency in optional_dependencies["rl-ml"])


def test_rl_ml_runtime_configs_are_fail_closed_and_host_local() -> None:
    for profile in ("dev", "test", "prod"):
        path = REPO_ROOT / "configs" / profile / "rl_trading_ml_runtime.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert config["profile"] == profile
        assert config["artifact_root"] == "/opt/roehub/state/rl_trading"
        assert config["runtime_artifacts"]["allowed_root"] == "/opt/roehub/state/rl_trading"
        assert config["runtime_artifacts"]["commit_to_git"] is False
        assert config["trainer"]["enabled"] is False
        assert config["inference"]["enabled"] is False
        assert config["trainer"]["max_concurrent_jobs"] == 1
        assert config["inference"]["max_concurrent_processes"] == 1
        assert config["disk"]["cleanup_may_delete_accepted_artifacts"] is False
        assert config["disk"]["block_training_below_free_gb"] >= config["disk"]["min_free_gb"]
