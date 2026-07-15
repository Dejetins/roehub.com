from __future__ import annotations

import importlib
import logging

import pytest

from apps.common.uvicorn_logging import (
    SensitiveQueryRedactionFilter,
    build_uvicorn_log_config,
)


def test_access_filter_redacts_api_and_web_oidc_callback_queries() -> None:
    for target in (
        "/auth/oidc/callback?code=disposable-code&state=disposable-state",
        "/api/auth/oidc/callback?code=disposable-code&state=disposable-state",
    ):
        record = logging.LogRecord(
            name="uvicorn.access",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg='%s - "%s %s HTTP/%s" %d',
            args=("127.0.0.1", "GET", target, "1.1", 303),
            exc_info=None,
        )

        assert SensitiveQueryRedactionFilter().filter(record) is True
        rendered = record.getMessage()
        assert "disposable-code" not in rendered
        assert "disposable-state" not in rendered
        assert "?redacted" in rendered


@pytest.mark.parametrize(
    ("module_name", "target", "factory"),
    (
        ("apps.api.main.main", "apps.api.main.app:app", False),
        ("apps.web.main.main", "apps.web.main.app:create_app", True),
    ),
)
def test_production_entrypoints_install_access_log_redaction(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    target: str,
    factory: bool,
) -> None:
    module = importlib.import_module(module_name)
    captured: dict[str, object] = {}

    def run(import_target: str, **kwargs: object) -> None:
        captured["target"] = import_target
        captured.update(kwargs)

    monkeypatch.setattr(module.uvicorn, "run", run)

    assert module.main(["--host", "127.0.0.1", "--port", "9876"]) == 0
    assert captured["target"] == target
    assert captured.get("factory", False) is factory
    config = captured["log_config"]
    assert config == build_uvicorn_log_config()
