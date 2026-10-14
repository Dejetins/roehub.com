from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

import apps.migrations.storage as storage


def _service_config(*, mode: str = "embedded") -> dict[str, object]:
    return {
        "schema": storage.RUNTIME_CONFIG_SCHEMA,
        "profile": "trading",
        "stores": {
            "postgresql": {
                "mode": mode,
                "host": "postgresql",
                "port": 5432,
                "database": "roehub",
                "tls": False,
            },
            "redis": {
                "mode": mode,
                "host": "redis",
                "port": 6379,
                "database": "roehub",
                "tls": False,
            },
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _endpoints() -> storage.StorageEndpoints:
    return storage.StorageEndpoints(
        postgresql_dsn="host=postgresql dbname=roehub user=roehub",
        redis_url="redis://redis:6379/0",
        redis_auth=None,
    )


def test_load_storage_config_accepts_embedded_and_external_profiles(tmp_path: Path) -> None:
    for mode in ("embedded", "external"):
        path = tmp_path / f"{mode}.json"
        _write_json(path, _service_config(mode=mode))
        config = storage.load_storage_config(path)
        assert config.profile == "trading"
        assert {store.mode for store in config.stores.values()} == {mode}


def test_load_storage_config_rejects_localhost_service_addressing(tmp_path: Path) -> None:
    payload = _service_config()
    stores = payload["stores"]
    assert isinstance(stores, dict)
    postgresql = stores["postgresql"]
    assert isinstance(postgresql, dict)
    postgresql["host"] = "localhost"
    path = tmp_path / "service-config.json"
    _write_json(path, payload)

    with pytest.raises(storage.StorageLifecycleError, match="localhost"):
        storage.load_storage_config(path)


def test_load_storage_endpoints_requires_configured_service_hosts(tmp_path: Path) -> None:
    path = tmp_path / "service-config.json"
    _write_json(path, _service_config())
    config = storage.load_storage_config(path)

    endpoints = storage.load_storage_endpoints(
        config=config,
        environ={
            "ROEHUB_STORAGE_POSTGRES_DSN": "host=postgresql dbname=roehub user=roehub",
            "ROEHUB_STORAGE_REDIS_URL": "redis://redis:6379/0",
        },
    )
    assert endpoints == _endpoints()

    with pytest.raises(storage.StorageLifecycleError, match="host does not match"):
        storage.load_storage_endpoints(
            config=config,
            environ={
                "ROEHUB_STORAGE_POSTGRES_DSN": "host=wrong dbname=roehub user=roehub",
                "ROEHUB_STORAGE_REDIS_URL": "redis://redis:6379/0",
            },
        )




def test_migration_manifests_match_immutable_sources() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    phases = storage._load_postgres_phases(  # noqa: SLF001
        repo_root / "migrations" / "postgres" / "manifest.json"
    )
    assert tuple(phases) == (
        "identity-0001-0009",
        "strategy-0010",
        "organization-0011",
        "local-auth-0012",
        "oidc-provider-0013",
        "research-tenancy-0014",
        "trading-tenancy-0015",
        "notification-providers-0016",
        "extensions-plugin-platform-0017",
        "artifact-store-0018",
        "isolated-job-runtime-0019",
        "execution-gateway-safety-0020",
        "control-operation-audit-0021",
        "market-data-selections-0022",
        "market-data-work-requests-0023",
        "market-data-full-history-0024",
        "market-data-work-recovery-0025",
        "market-data-queue-events-0026",
        "market-data-stream-recovery-0027",
        "backtest-input-recipe-0028",
        "backtest-artifact-management-0029",
        "backtest-artifact-dispatch-0030",
        "backtest-artifact-storage-0031",
        "identity-browser-devices-0032",
        "market-data-stream-connections-0033",
        "market-data-provider-admission-0034",
        "market-data-stream-events-0035",
        "market-data-storage-0036",
        "market-data-reference-0037",
        "market-data-verification-0038",
        "market-data-consumers-0039",
        "market-data-append-fence-0040",
    )






def test_bootstrap_runs_capabilities_before_ordered_durable_migrations(
    monkeypatch: Any,
) -> None:
    calls: list[str] = []
    config = storage.StorageConfig(
        profile="trading",
        stores={
            name: storage.StoreConfig(
                name=name,
                mode="embedded",
                host=name,
                port={"postgresql": 5432, "redis": 6379}[name],
                database="roehub",
                tls=False,
            )
            for name in ("postgresql", "redis")
        },
    )
    monkeypatch.setattr(
        storage,
        "check_postgres_capabilities",
        lambda dsn: calls.append("check-postgresql") or {},
    )
    monkeypatch.setattr(
        storage,
        "check_redis_capabilities",
        lambda url, auth=None: calls.append("check-redis") or {},
    )
    monkeypatch.setattr(
        storage,
        "apply_postgres_migrations",
        lambda dsn, repo_root, manifest_path: calls.append("migrate-postgresql"),
    )
    monkeypatch.setattr(
        storage,
        "build_storage_status",
        lambda **kwargs: {"ready": True},
    )

    monkeypatch.setattr(storage, "provision_runtime_roles", lambda dsn: calls.append("grant-roles"))

    result = storage.bootstrap_storage(
        config=config,
        endpoints=_endpoints(),
        repo_root=Path("."),
        postgres_manifest=Path("postgres.json"),
    )

    assert result == {"ready": True}
    assert calls == [
        "check-postgresql",
        "check-redis",
        "migrate-postgresql",
        "grant-roles",
    ]


def test_storage_status_is_secret_free_and_schema_valid(monkeypatch: Any) -> None:
    config = storage.StorageConfig(
        profile="trading",
        stores={
            name: storage.StoreConfig(
                name=name,
                mode="external",
                host=name,
                port={"postgresql": 5432, "redis": 6379}[name],
                database="roehub",
                tls=False,
            )
            for name in ("postgresql", "redis")
        },
    )
    monkeypatch.setattr(
        storage,
        "check_postgres_capabilities",
        lambda dsn: {
            "engine": "PostgreSQL",
            "version": "16.0",
            "capabilities": ["schema-create"],
        },
    )
    monkeypatch.setattr(
        storage,
        "check_redis_capabilities",
        lambda url, auth=None: {
            "engine": "Redis",
            "version": "7.2.1",
            "capabilities": ["aof"],
        },
    )
    monkeypatch.setattr(
        storage,
        "_postgres_schema_status",
        lambda dsn, repo_root, manifest_path: ["alembic:head"],
    )

    result = storage.build_storage_status(
        config=config,
        endpoints=_endpoints(),
        repo_root=Path("."),
        postgres_manifest=Path("postgres.json"),
        require_schema=True,
    )
    schema_path = Path(__file__).resolve().parents[4] / "schemas/config/storage-status.schema.json"
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(result)
    rendered = json.dumps(result)
    assert "postgresql_dsn" not in rendered
    assert "redis_url" not in rendered
    assert result["ready"] is True


def test_installation_schema_rejects_localhost_store_host() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    schema = json.loads((repo_root / "schemas/config/roehub.schema.json").read_text())
    config = json.loads(json.dumps(_service_config()))
    config.update(
        {
            "schema": "io.roehub.installation/v2alpha1",
            "installation_id": "test-installation",
            "domain": "localhost",
            "architecture": "linux/arm64",
            "profiles": ["base", "trading"],
            "paths": {"data": "/data", "config": "/config", "artifacts": "/artifacts"},
            "ports": {"http": 8080, "https": 8443, "api": 8000, "metrics": 9090},
            "artifacts": {"mode": "local_cas", "path": "/artifacts"},
            "resources": {
                "base": {"cpus": 1, "memory_mb": 512},
                "trading": {"cpus": 1, "memory_mb": 512},
                "ml": {"cpus": 1, "memory_mb": 512},
            },
            "tls": {"mode": "disabled"},
            "proxy": {"mode": "embedded", "trusted_proxies": []},
            "update_checks": {"enabled": False},
            "oidc": {"enabled": False},
            "openbao": {
                "mode": "embedded",
                "address": "http://openbao:8200",
                "secret_root": "kv/roehub",
            },
            "notifications": {"telegram": {"enabled": False}},
            "trading": {"mode": "paper"},
        }
    )
    stores = config["stores"]
    stores["postgresql"]["credentials_ref"] = "openbao://kv/roehub/storage/postgresql"
    stores["redis"]["credentials_ref"] = "openbao://kv/roehub/storage/redis"
    stores["postgresql"]["host"] = "localhost"

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(config))
    assert any(tuple(error.path) == ("stores", "postgresql", "host") for error in errors)
