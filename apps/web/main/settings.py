from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

_WEB_API_BASE_URL_ENV = "WEB_API_BASE_URL"


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



def resolve_web_runtime_settings(*, environ: Mapping[str, str]) -> WebRuntimeSettings:
    """
    Resolve runtime settings for the web SSR process from provided environment mapping.

    Args:
        environ: Environment mapping used for startup configuration.
    Returns:
        WebRuntimeSettings: Validated immutable runtime settings.
    Assumptions:
        `WEB_API_BASE_URL` is configured in all target environments.
    Raises:
        ValueError: If `WEB_API_BASE_URL` is missing or blank.
    Side Effects:
        None.
    """
    raw_api_base_url = environ.get(_WEB_API_BASE_URL_ENV)
    if raw_api_base_url is None or not raw_api_base_url.strip():
        raise ValueError(
            "resolve_web_runtime_settings requires non-empty WEB_API_BASE_URL"
        )

    normalized_api_base_url = raw_api_base_url.strip().rstrip("/")
    return WebRuntimeSettings(api_base_url=normalized_api_base_url)
