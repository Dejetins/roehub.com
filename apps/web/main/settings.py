from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

_WEB_API_BASE_URL_ENV = "WEB_API_BASE_URL"
_WEB_API_UPSTREAM_URL_ENV = "WEB_API_UPSTREAM_URL"


@dataclass(frozen=True)
class WebRuntimeSettings:
    """
    WebRuntimeSettings stores immutable runtime settings for the web delivery process.

    Docs:
      - docs/architecture/apps/web/web-ui-skeleton-ssr-htmx-auth-v1.md
    Related:
      - apps/web/main/app.py
      - apps/web/main/api_client.py
    """

    api_base_url: str
    api_upstream_url: str


def resolve_web_runtime_settings(*, environ: Mapping[str, str]) -> WebRuntimeSettings:
    """
    Resolve runtime settings for the web SSR process from provided environment mapping.

    Args:
        environ: Environment mapping used for startup configuration.
    Returns:
        WebRuntimeSettings: Validated immutable runtime settings.
    Assumptions:
        `WEB_API_BASE_URL` and `WEB_API_UPSTREAM_URL` are configured in all target
        environments.
    Raises:
        ValueError: If required web API URLs are missing or blank.
    Side Effects:
        None.
    """
    raw_api_base_url = environ.get(_WEB_API_BASE_URL_ENV)
    if raw_api_base_url is None or not raw_api_base_url.strip():
        raise ValueError("resolve_web_runtime_settings requires non-empty WEB_API_BASE_URL")

    raw_api_upstream_url = environ.get(_WEB_API_UPSTREAM_URL_ENV)
    if raw_api_upstream_url is None or not raw_api_upstream_url.strip():
        raise ValueError("resolve_web_runtime_settings requires non-empty WEB_API_UPSTREAM_URL")

    normalized_api_base_url = raw_api_base_url.strip().rstrip("/")
    normalized_api_upstream_url = raw_api_upstream_url.strip().rstrip("/")
    return WebRuntimeSettings(
        api_base_url=normalized_api_base_url,
        api_upstream_url=normalized_api_upstream_url,
    )
