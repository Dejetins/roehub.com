from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_retired_native_deployment_workflows_are_absent() -> None:
    root = _repo_root()

    assert not (root / ".github/workflows/deploy-backend.yml").exists()
    assert not (root / ".github/workflows/deploy-web.yml").exists()
    assert not (root / ".github/workflows/exchange-connection-cleanup-ops.yml").exists()
