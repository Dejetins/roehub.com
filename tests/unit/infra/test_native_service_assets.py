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
    assert "com.roehub.test.exchange-execution.plist" in test_bootstrap
    assert "com.roehub.test.exchange-execution.plist" in reload_services
