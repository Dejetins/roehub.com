from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def _load_main_compose() -> dict[str, Any]:
    """
    Load main Docker Compose YAML for deterministic structure assertions.

    Args:
        None.
    Returns:
        dict[str, Any]: Parsed compose file payload.
    Assumptions:
        Compose file remains UTF-8 YAML under `infra/docker/docker-compose.yml`.
    Raises:
        OSError: If compose file cannot be read.
        yaml.YAMLError: If compose YAML is malformed.
    Side Effects:
        Reads repository file from disk.
    """
    repo_root = Path(__file__).resolve().parents[3]
    compose_path = repo_root / "infra" / "docker" / "docker-compose.yml"
    raw_payload = compose_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw_payload)
    if parsed is None:
        raise ValueError("Compose payload is empty")
    if "services" not in parsed:
        raise ValueError("Compose payload must contain top-level services mapping")
    return cast(dict[str, Any], parsed)


def test_ui_profile_contains_api_web_and_db_bootstrap() -> None:
    """
    Verify main compose defines expected `ui` profile services after gateway removal.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Services are declared under top-level `services` mapping.
    Raises:
        AssertionError: If required services or `profiles: [\"ui\"]` are missing.
    Side Effects:
        None.
    """
    compose_payload = _load_main_compose()
    services = compose_payload["services"]

    for service_name in ("api", "web", "db-bootstrap"):
        assert service_name in services
        assert services[service_name]["profiles"] == ["ui"]
    assert "gateway" not in services


def test_ui_profile_publishes_localhost_api_and_web_to_host() -> None:
    """
    Verify UI profile publishes localhost-only API and web host mappings.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        API and web are intentionally published only on localhost for private admin,
        tailnet proxy, and local browser same-origin scenarios.
    Raises:
        AssertionError: If required host publishing drifts from expected contract.
    Side Effects:
        None.
    """
    compose_payload = _load_main_compose()
    services = compose_payload["services"]

    assert services["api"]["ports"] == ["${API_HOST_BIND:-127.0.0.1}:${API_HOST_PORT:-8000}:8000"]
    assert services["web"]["ports"] == ["127.0.0.1:${WEB_HOST_PORT:-8010}:8010"]
    assert "ports" not in services["db-bootstrap"]


def test_ui_profile_web_uses_same_origin_base_and_direct_api_upstream() -> None:
    """
    Verify web service relies on an internal `/api/*` proxy instead of a gateway container.
    """
    compose_payload = _load_main_compose()
    web_env = compose_payload["services"]["web"]["environment"]

    assert web_env["WEB_API_BASE_URL"] == "${WEB_API_BASE_URL:-http://127.0.0.1:8010}"
    assert web_env["WEB_API_UPSTREAM_URL"] == "${WEB_API_UPSTREAM_URL:-http://api:8000}"


def test_ui_profile_uses_conninfo_dsn_defaults_from_postgres_env() -> None:
    """
    Verify UI profile services derive DSNs from POSTGRES_* values in conninfo format.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        UI profile should not require explicit DSN lines in env file.
    Raises:
        AssertionError: If DSN defaults drift from required conninfo template.
    Side Effects:
        None.
    """
    compose_payload = _load_main_compose()
    services = compose_payload["services"]
    expected_conninfo = (
        "host=postgres port=5432 dbname=${POSTGRES_DB:-roehub} "
        "user=${POSTGRES_USER:-roehub} password=${POSTGRES_PASSWORD}"
    )

    db_bootstrap_env = services["db-bootstrap"]["environment"]
    api_env = services["api"]["environment"]

    assert db_bootstrap_env["IDENTITY_PG_DSN"] == expected_conninfo
    assert db_bootstrap_env["POSTGRES_DSN"] == expected_conninfo
    assert api_env["IDENTITY_PG_DSN"] == expected_conninfo
    assert api_env["STRATEGY_PG_DSN"] == expected_conninfo
