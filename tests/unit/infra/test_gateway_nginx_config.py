from __future__ import annotations

import re
from pathlib import Path


def _read_nginx_config() -> str:
    """
    Read gateway nginx configuration file from repository tree.

    Args:
        None.
    Returns:
        str: UTF-8 nginx config text.
    Assumptions:
        Test file location is stable under `tests/unit/infra`.
    Raises:
        OSError: If config file is missing or cannot be read.
    Side Effects:
        Reads repository file from disk.
    """
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "infra" / "docker" / "nginx" / "nginx.conf"
    return config_path.read_text(encoding="utf-8")


def test_gateway_nginx_routes_api_assets_and_web() -> None:
    """
    Verify gateway config contains required API, assets, and web routing locations.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Location blocks are configured with deterministic prefixes for v1 gateway contract.
    Raises:
        AssertionError: If required location blocks are missing.
    Side Effects:
        None.
    """
    config_text = _read_nginx_config()

    assert "location ^~ /api/" in config_text
    assert "location ^~ /assets/" in config_text
    assert "location / {" in config_text


def test_gateway_nginx_strips_api_prefix_via_trailing_slash_proxy_pass() -> None:
    """
    Verify API upstream uses trailing slash proxy_pass to strip `/api` prefix.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `location ^~ /api/` + `proxy_pass http://api:8000/;` is required strip semantics.
    Raises:
        AssertionError: If proxy_pass line is missing or malformed.
    Side Effects:
        None.
    """
    config_text = _read_nginx_config()
    match = re.search(
        r"location\s+\^~\s+/api/\s*\{[^}]*proxy_pass\s+http://api:8000/;",
        config_text,
        flags=re.DOTALL,
    )

    assert match is not None
