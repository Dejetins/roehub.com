"""Uvicorn logging configuration that redacts OIDC callback query values."""

from __future__ import annotations

import copy
import logging
from typing import Any, cast

from uvicorn.config import LOGGING_CONFIG

_FILTER_NAME = "roehub_sensitive_query_redaction"
_FILTER_PATH = "apps.common.uvicorn_logging.SensitiveQueryRedactionFilter"
_OIDC_CALLBACK_SUFFIX = "/auth/oidc/callback"


def redact_access_log_target(target: str) -> str:
    """Remove the full query from OIDC callback targets before formatting."""
    path, separator, _ = target.partition("?")
    if separator and path.endswith(_OIDC_CALLBACK_SUFFIX):
        return f"{path}?redacted"
    return target


class SensitiveQueryRedactionFilter(logging.Filter):
    """Redact OIDC codes and state from Uvicorn access-log record arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        arguments = record.args
        if (
            isinstance(arguments, tuple)
            and len(arguments) >= 3
            and isinstance(arguments[2], str)
        ):
            updated = list(arguments)
            updated[2] = redact_access_log_target(arguments[2])
            record.args = tuple(updated)
        return True


def build_uvicorn_log_config() -> dict[str, Any]:
    """Return an isolated Uvicorn config with callback-query redaction enabled."""
    config = cast(dict[str, Any], copy.deepcopy(LOGGING_CONFIG))
    filters = cast(dict[str, Any], config.setdefault("filters", {}))
    filters[_FILTER_NAME] = {"()": _FILTER_PATH}
    handlers = cast(dict[str, Any], config["handlers"])
    access_handler = cast(dict[str, Any], handlers["access"])
    access_handler["filters"] = [_FILTER_NAME]
    return config
