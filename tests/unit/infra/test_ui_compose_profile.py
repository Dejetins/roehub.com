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


def test_ui_profile_contains_gateway_api_web_and_db_bootstrap() -> None:
    """
    Verify main compose defines expected `ui` profile services for WEB-EPIC-02.

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

    for service_name in ("api", "web", "gateway", "db-bootstrap"):
        assert service_name in services
        assert services[service_name]["profiles"] == ["ui"]


def test_ui_profile_publishes_only_gateway_to_host() -> None:
    """
    Verify only gateway service publishes host port mapping in `ui` profile.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        API and web are internal-only services on compose network.
    Raises:
        AssertionError: If host publishing is configured for non-gateway ui services.
    Side Effects:
        None.
    """
    compose_payload = _load_main_compose()
    services = compose_payload["services"]

    assert services["gateway"]["ports"] == [
        "${GATEWAY_HOST_BIND:-127.0.0.1}:${GATEWAY_HOST_PORT:-8080}:80"
    ]
    assert "ports" not in services["api"]
    assert "ports" not in services["web"]
    assert "ports" not in services["db-bootstrap"]


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
