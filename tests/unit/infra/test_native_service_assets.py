from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read(relative_path: str) -> str:
    return (_repo_root() / relative_path).read_text(encoding="utf-8")


def test_exchange_execution_native_service_assets_are_installed_and_reloaded() -> None:
    prod_bootstrap = _read("scripts/macos/bootstrap_native_prod.sh")
    test_bootstrap = _read("scripts/macos/bootstrap_native_test.sh")
    reload_services = _read("scripts/macos/reload_launchd_services.sh")

    assert "roehub-exchange-execution.monitrc" in prod_bootstrap
    assert "com.roehub.exchange-execution.plist" in prod_bootstrap
    assert "com.roehub.exchange-execution.plist" in reload_services
    assert "roehub-strategy-live-runner.monitrc" in prod_bootstrap
    assert "com.roehub.strategy-live-runner.plist" in prod_bootstrap
    assert "com.roehub.strategy-live-runner.plist" in reload_services
    assert "roehub-rl-trading-inference.monitrc" in prod_bootstrap
    assert "com.roehub.rl-trading-inference.plist" in prod_bootstrap
    assert "com.roehub.rl-trading-inference.plist" in reload_services
    assert "com.roehub.test.exchange-execution.plist" in test_bootstrap
    assert "com.roehub.test.exchange-execution.plist" in reload_services


def test_backend_deploy_reloads_monit_for_monit_asset_changes() -> None:
    workflow = _read(".github/workflows/deploy-backend.yml")

    assert "infra/scripts/monit/" in workflow
    assert "brew services restart monit" in workflow


def test_rl_monitor_smoke_checks_ready_loaded_and_safety_metrics() -> None:
    smoke = _read("scripts/macos/smoke_prod.sh")
    deploy_workflow = _read(".github/workflows/deploy-backend.yml")

    assert "127.0.0.1:9213/health/ready" in smoke
    assert "for _attempt in {1..60}" in smoke
    assert "did not become ready within 60 seconds" in smoke
    assert "rl_trading_inference_model_loaded 1.0" in smoke
    assert "rl_trading_inference_safety_breaches_total" in smoke
    assert "bash scripts/macos/smoke_prod.sh" in deploy_workflow
