"""Greenfield PostgreSQL, ClickHouse, and Redis lifecycle for self-hosted Roehub."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast
from urllib.parse import urlsplit

import clickhouse_connect
import psycopg
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from redis import Redis

from alembic.script import ScriptDirectory
from apps.migrations.bootstrap import (
    apply_artifact_store_sql,
    apply_control_operation_audit_sql,
    apply_execution_gateway_mainnet_safety_sql,
    apply_extensions_plugin_platform_sql,
    apply_identity_baseline_sql,
    apply_isolated_job_runtime_sql,
    apply_local_auth_sql,
    apply_notification_provider_instances_sql,
    apply_oidc_provider_sql,
    apply_organizations_rbac_audit_sql,
    apply_research_organization_isolation_sql,
    apply_strategy_exchange_bindings_sql,
    apply_trading_organization_isolation_sql,
    normalize_psycopg_dsn,
    run_alembic_upgrade_head,
)
from apps.migrations.main import (
    _build_alembic_config,
)
from apps.migrations.main import (
    main as run_alembic_migrations_main,
)

STORAGE_STATUS_SCHEMA = "io.roehub.storage-status/v1alpha1"
RUNTIME_CONFIG_SCHEMA = "io.roehub.runtime-config/v1alpha1"
POSTGRES_MANIFEST_SCHEMA = "io.roehub.postgres-migrations/v1alpha1"
CLICKHOUSE_MANIFEST_SCHEMA = "io.roehub.clickhouse-migrations/v1alpha1"
POSTGRES_STATE_TABLE = "roehub_storage_migrations"
CLICKHOUSE_STATE_TABLE = "roehub_schema_migrations"
POSTGRES_LOCK_KEY = 56329814721
SUPPORTED_PROFILE_STORES: dict[str, tuple[str, ...]] = {
    "base": ("postgresql", "redis"),
    "trading": ("postgresql", "clickhouse", "redis"),
    "ml": ("postgresql", "clickhouse", "redis"),
}


class StorageLifecycleError(RuntimeError):
    """Raised when a storage capability, migration, or readiness invariant fails."""


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """Non-secret store connection contract from generated service configuration."""

    name: str
    mode: str
    host: str
    port: int
    database: str
    tls: bool


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Storage subset of one generated Roehub runtime profile."""

    profile: str
    stores: Mapping[str, StoreConfig]


@dataclass(frozen=True, slots=True)
class StorageEndpoints:
    """Secret-bearing runtime endpoints kept outside user-edited configuration."""

    postgresql_dsn: str
    clickhouse_dsn: str | None
    redis_url: str
    redis_auth: str | None = None
    clickhouse_auth: str | None = None


@dataclass(frozen=True, slots=True)
class MigrationFile:
    """One immutable, ordered migration source."""

    version: str
    name: str
    path: Path
    sha256: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StorageLifecycleError(f"cannot read {label}") from error
    if not isinstance(payload, dict):
        raise StorageLifecycleError(f"{label} root must be an object")
    return payload


def load_storage_config(path: Path) -> StorageConfig:
    """Load the non-secret storage contract from generated `service-config.json`."""

    payload = _load_json_object(path, label="generated service configuration")
    if payload.get("schema") != RUNTIME_CONFIG_SCHEMA:
        raise StorageLifecycleError("unsupported generated service configuration schema")
    profile = payload.get("profile")
    if not isinstance(profile, str) or profile not in SUPPORTED_PROFILE_STORES:
        raise StorageLifecycleError("unsupported generated service profile")
    raw_stores = payload.get("stores")
    if not isinstance(raw_stores, dict):
        raise StorageLifecycleError("generated service configuration has no stores")

    expected = SUPPORTED_PROFILE_STORES[profile]
    if tuple(sorted(raw_stores)) != tuple(sorted(expected)):
        raise StorageLifecycleError("generated service stores do not match profile contract")

    stores: dict[str, StoreConfig] = {}
    for name in expected:
        raw = raw_stores.get(name)
        if not isinstance(raw, dict):
            raise StorageLifecycleError(f"generated store contract is invalid: {name}")
        mode = raw.get("mode")
        host = raw.get("host")
        port = raw.get("port")
        database = raw.get("database")
        tls = raw.get("tls", False)
        if mode not in {"embedded", "external"}:
            raise StorageLifecycleError(f"unsupported store mode: {name}")
        if not isinstance(host, str) or host.lower() in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise StorageLifecycleError(f"localhost service addressing is forbidden: {name}")
        if not isinstance(port, int) or not 1 <= port <= 65535:
            raise StorageLifecycleError(f"invalid store port: {name}")
        if not isinstance(database, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_-]{0,62}", database
        ):
            raise StorageLifecycleError(f"invalid store database or namespace: {name}")
        if not isinstance(tls, bool):
            raise StorageLifecycleError(f"invalid store TLS flag: {name}")
        stores[name] = StoreConfig(
            name=name,
            mode=mode,
            host=host,
            port=port,
            database=database,
            tls=tls,
        )
    return StorageConfig(profile=profile, stores=stores)


def load_storage_endpoints(
    *,
    config: StorageConfig,
    environ: Mapping[str, str],
) -> StorageEndpoints:
    """Resolve secret-bearing endpoints from the process environment only."""

    postgresql_dsn = environ.get("ROEHUB_STORAGE_POSTGRES_DSN", "").strip()
    redis_url = environ.get("ROEHUB_STORAGE_REDIS_URL", "").strip()
    redis_auth = environ.get("ROEHUB_STORAGE_REDIS_PASSWORD", "").strip()
    clickhouse_dsn = environ.get("ROEHUB_STORAGE_CLICKHOUSE_DSN", "").strip()
    clickhouse_auth = environ.get("ROEHUB_STORAGE_CLICKHOUSE_PASSWORD", "").strip()
    if not postgresql_dsn:
        raise StorageLifecycleError("ROEHUB_STORAGE_POSTGRES_DSN is required")
    if not redis_url:
        raise StorageLifecycleError("ROEHUB_STORAGE_REDIS_URL is required")
    if "clickhouse" in config.stores and not clickhouse_dsn:
        raise StorageLifecycleError("ROEHUB_STORAGE_CLICKHOUSE_DSN is required")

    endpoints = StorageEndpoints(
        postgresql_dsn=postgresql_dsn,
        clickhouse_dsn=clickhouse_dsn or None,
        redis_url=redis_url,
        redis_auth=redis_auth or None,
        clickhouse_auth=clickhouse_auth or None,
    )
    _validate_endpoint_hosts(config=config, endpoints=endpoints)
    return endpoints


def _validate_endpoint_hosts(*, config: StorageConfig, endpoints: StorageEndpoints) -> None:
    postgres_fields = conninfo_to_dict(normalize_psycopg_dsn(dsn=endpoints.postgresql_dsn))
    _require_endpoint_host(
        store=config.stores["postgresql"],
        actual=str(postgres_fields.get("host", "")),
    )
    actual_database = str(postgres_fields.get("dbname", ""))
    if actual_database and actual_database != config.stores["postgresql"].database:
        raise StorageLifecycleError("PostgreSQL endpoint database does not match config")

    redis_parts = urlsplit(endpoints.redis_url)
    if redis_parts.scheme not in {"redis", "rediss"}:
        raise StorageLifecycleError("Redis endpoint must use redis:// or rediss://")
    _require_endpoint_host(
        store=config.stores["redis"],
        actual=redis_parts.hostname or "",
    )

    if endpoints.clickhouse_dsn is not None:
        clickhouse_parts = urlsplit(endpoints.clickhouse_dsn)
        if clickhouse_parts.scheme not in {"http", "https"}:
            raise StorageLifecycleError("ClickHouse endpoint must use http:// or https://")
        _require_endpoint_host(
            store=config.stores["clickhouse"],
            actual=clickhouse_parts.hostname or "",
        )


def _require_endpoint_host(*, store: StoreConfig, actual: str) -> None:
    if actual.lower() != store.host.lower():
        raise StorageLifecycleError(f"{store.name} endpoint host does not match config")


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if match is None:
        raise StorageLifecycleError("store returned an unsupported version format")
    return tuple(int(part or 0) for part in match.groups())


def check_postgres_capabilities(dsn: str) -> dict[str, Any]:
    """Verify the certified PostgreSQL 16 capability profile without durable writes."""

    try:
        with psycopg.connect(
            normalize_psycopg_dsn(dsn=dsn),
            autocommit=True,
            row_factory=cast(Any, dict_row),
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        """
                        SELECT
                            current_setting('server_version_num')::integer AS version_num,
                            current_setting('server_version') AS version,
                            has_schema_privilege(current_user, 'public', 'CREATE') AS schema_create,
                            has_database_privilege(
                                current_user, current_database(), 'TEMP'
                            ) AS temp_create
                        """,
                    )
                )
                raw_row = cursor.fetchone()
                cursor.execute(
                    cast(Any, "SELECT pg_try_advisory_lock(%s) AS acquired"),
                    (POSTGRES_LOCK_KEY,),
                )
                raw_lock_row = cursor.fetchone()
                row = cast(Mapping[str, Any] | None, raw_row)
                lock_row = cast(Mapping[str, Any] | None, raw_lock_row)
                if lock_row and lock_row["acquired"]:
                    cursor.execute(
                        cast(Any, "SELECT pg_advisory_unlock(%s)"),
                        (POSTGRES_LOCK_KEY,),
                    )
            if row is None or lock_row is None:
                raise StorageLifecycleError("PostgreSQL capability query returned no row")
            version_num = int(row["version_num"])
            if not 160000 <= version_num < 170000:
                raise StorageLifecycleError(
                    "PostgreSQL certified profile requires major version 16"
                )
            if not bool(row["schema_create"]) or not bool(row["temp_create"]):
                raise StorageLifecycleError(
                    "PostgreSQL user lacks schema or temporary-table capability"
                )
            if not bool(lock_row["acquired"]):
                raise StorageLifecycleError("PostgreSQL advisory-lock capability is unavailable")

        with psycopg.connect(normalize_psycopg_dsn(dsn=dsn), autocommit=False) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, "CREATE TEMP TABLE roehub_ddl_probe(value integer)"))
            connection.rollback()
    except StorageLifecycleError:
        raise
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("PostgreSQL capability check failed") from error
    return {
        "engine": "PostgreSQL",
        "version": str(row["version"]),
        "capabilities": ["advisory-lock", "schema-create", "transactional-ddl"],
    }


def _quote_clickhouse_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,62}", value):
        raise StorageLifecycleError("invalid ClickHouse database identifier")
    return f"`{value}`"


def _open_clickhouse(dsn: str, *, auth: str | None = None) -> Any:
    try:
        options: dict[str, Any] = {}
        if auth is not None:
            options["password"] = auth
        return clickhouse_connect.get_client(
            dsn=dsn,
            database="default",
            connect_timeout=10,
            **options,
        )
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("ClickHouse connection failed") from error


def check_clickhouse_capabilities(
    dsn: str,
    *,
    database: str,
    auth: str | None = None,
) -> dict[str, Any]:
    """Verify the certified ClickHouse 24.8 profile and transient DDL capability."""

    client = _open_clickhouse(dsn, auth=auth)
    database_sql = _quote_clickhouse_identifier(database)
    probe = _quote_clickhouse_identifier(f"roehub_capability_probe_{uuid.uuid4().hex}")
    try:
        version = str(client.command("SELECT version()"))
        version_parts = _version_tuple(version)
        if version_parts[:2] != (24, 8):
            raise StorageLifecycleError("ClickHouse certified profile requires version 24.8.x")
        client.command(f"CREATE DATABASE IF NOT EXISTS {database_sql}")
        client.command(f"CREATE TABLE {database_sql}.{probe} (value UInt8) ENGINE = Memory")
        client.command(f"DROP TABLE {database_sql}.{probe}")
        system_tables = int(client.command("SELECT count() FROM system.tables"))
        if system_tables <= 0:
            raise StorageLifecycleError("ClickHouse system catalog capability is unavailable")
    except StorageLifecycleError:
        raise
    except Exception as error:  # noqa: BLE001
        try:
            client.command(f"DROP TABLE IF EXISTS {database_sql}.{probe}")
        except Exception:  # noqa: BLE001
            pass
        raise StorageLifecycleError("ClickHouse capability check failed") from error
    finally:
        client.close()
    return {
        "engine": "ClickHouse",
        "version": version,
        "capabilities": ["database-create", "merge-tree", "system-catalog"],
    }


def check_redis_capabilities(url: str, *, auth: str | None = None) -> dict[str, Any]:
    """Verify Redis 7.2, write/read/delete semantics, AOF, and no-eviction policy."""

    client: Redis = Redis.from_url(
        url,
        **({"password": auth} if auth is not None else {}),
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=10,
    )
    probe_key = f"roehub:capability-probe:{uuid.uuid4().hex}"
    try:
        if client.ping() is not True:
            raise StorageLifecycleError("Redis PING failed")
        server = cast(dict[str, Any], client.info(section="server"))
        version = str(server.get("redis_version", ""))
        if _version_tuple(version)[:2] != (7, 2):
            raise StorageLifecycleError("Redis certified profile requires version 7.2.x")
        persistence = cast(dict[str, str], client.config_get("appendonly"))
        policy = cast(dict[str, str], client.config_get("maxmemory-policy"))
        if persistence.get("appendonly") != "yes":
            raise StorageLifecycleError("Redis AOF must be enabled")
        if policy.get("maxmemory-policy") != "noeviction":
            raise StorageLifecycleError("Redis maxmemory-policy must be noeviction")
        if client.set(probe_key, "ready", ex=10, nx=True) is not True:
            raise StorageLifecycleError("Redis write capability is unavailable")
        if client.get(probe_key) != "ready":
            raise StorageLifecycleError("Redis read capability is unavailable")
    except StorageLifecycleError:
        raise
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("Redis capability check failed") from error
    finally:
        try:
            client.delete(probe_key)
        except Exception:  # noqa: BLE001
            pass
        client.close()
    return {
        "engine": "Redis",
        "version": version,
        "capabilities": ["aof", "noeviction", "ping", "read-write-delete"],
    }


def _load_postgres_phases(manifest_path: Path) -> dict[str, tuple[str, tuple[Path, ...]]]:
    payload = _load_json_object(manifest_path, label="PostgreSQL migration manifest")
    if payload.get("schema") != POSTGRES_MANIFEST_SCHEMA:
        raise StorageLifecycleError("unsupported PostgreSQL migration manifest schema")
    raw_phases = payload.get("phases")
    if not isinstance(raw_phases, list) or len(raw_phases) != 13:
        raise StorageLifecycleError("PostgreSQL migration manifest must define thirteen phases")
    phases: dict[str, tuple[str, tuple[Path, ...]]] = {}
    for raw_phase in raw_phases:
        if not isinstance(raw_phase, dict):
            raise StorageLifecycleError("invalid PostgreSQL migration phase")
        version = raw_phase.get("version")
        expected_phase_sha = raw_phase.get("sha256")
        raw_files = raw_phase.get("files")
        if not isinstance(version, str) or not isinstance(expected_phase_sha, str):
            raise StorageLifecycleError("invalid PostgreSQL migration phase identity")
        if not isinstance(raw_files, list) or not raw_files:
            raise StorageLifecycleError("PostgreSQL migration phase has no files")
        paths: list[Path] = []
        digest = hashlib.sha256()
        for raw_file in raw_files:
            if not isinstance(raw_file, dict):
                raise StorageLifecycleError("invalid PostgreSQL migration file entry")
            relative = raw_file.get("path")
            expected_sha = raw_file.get("sha256")
            if not isinstance(relative, str) or Path(relative).name != relative:
                raise StorageLifecycleError("unsafe PostgreSQL migration path")
            path = manifest_path.parent / relative
            try:
                content = path.read_bytes()
            except OSError as error:
                raise StorageLifecycleError("missing PostgreSQL migration source") from error
            if not isinstance(expected_sha, str) or _sha256(content) != expected_sha:
                raise StorageLifecycleError("PostgreSQL migration source checksum mismatch")
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(content)
            digest.update(b"\0")
            paths.append(path)
        if digest.hexdigest() != expected_phase_sha:
            raise StorageLifecycleError("PostgreSQL migration phase checksum mismatch")
        phases[version] = (expected_phase_sha, tuple(paths))
    if set(phases) != {
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
    }:
        raise StorageLifecycleError("PostgreSQL migration phases are incomplete")
    return phases


def _ensure_postgres_state_table(dsn: str) -> None:
    sql = f"""
        CREATE TABLE IF NOT EXISTS {POSTGRES_STATE_TABLE} (
            store TEXT NOT NULL,
            version TEXT NOT NULL,
            checksum TEXT NOT NULL CHECK (char_length(checksum) = 64),
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (store, version)
        )
    """
    try:
        with psycopg.connect(normalize_psycopg_dsn(dsn=dsn), autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, sql))
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("cannot create PostgreSQL migration state") from error


def _read_postgres_markers(dsn: str) -> dict[tuple[str, str], str]:
    try:
        with psycopg.connect(
            normalize_psycopg_dsn(dsn=dsn),
            autocommit=True,
            row_factory=cast(Any, dict_row),
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        f"SELECT store, version, checksum FROM {POSTGRES_STATE_TABLE} "
                        "ORDER BY store, version",
                    )
                )
                rows = cast(list[Mapping[str, Any]], cursor.fetchall())
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("cannot read PostgreSQL migration state") from error
    return {(str(row["store"]), str(row["version"])): str(row["checksum"]) for row in rows}


def _record_postgres_marker(dsn: str, *, store: str, version: str, checksum: str) -> None:
    try:
        with psycopg.connect(normalize_psycopg_dsn(dsn=dsn), autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    cast(
                        Any,
                        f"INSERT INTO {POSTGRES_STATE_TABLE} (store, version, checksum) "
                        "VALUES (%s, %s, %s) ON CONFLICT (store, version) DO NOTHING",
                    ),
                    (store, version, checksum),
                )
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("cannot persist PostgreSQL migration state") from error


def _alembic_revision_files(repo_root: Path) -> tuple[MigrationFile, ...]:
    config = _build_alembic_config(repo_root=repo_root)
    script = ScriptDirectory.from_config(config)
    heads = script.get_heads()
    if len(heads) != 1:
        raise StorageLifecycleError("PostgreSQL Alembic history must have exactly one head")
    revisions: list[MigrationFile] = []
    for revision in reversed(list(script.walk_revisions(base="base", head="heads"))):
        path = Path(revision.path)
        content = path.read_bytes()
        revisions.append(
            MigrationFile(
                version=str(revision.revision),
                name=path.stem,
                path=path,
                sha256=_sha256(content),
            )
        )
    return tuple(revisions)


def apply_postgres_migrations(
    dsn: str,
    *,
    repo_root: Path,
    manifest_path: Path,
) -> None:
    """Apply identity, Alembic, and binding phases with immutable durable markers."""

    phases = _load_postgres_phases(manifest_path)
    revisions = _alembic_revision_files(repo_root)
    _ensure_postgres_state_table(dsn)
    markers = _read_postgres_markers(dsn)

    expected_markers = {
        ("postgres-sql", version): checksum for version, (checksum, _paths) in phases.items()
    }
    expected_markers.update(
        {("postgres-alembic", revision.version): revision.sha256 for revision in revisions}
    )
    for identity, actual_checksum in markers.items():
        expected_checksum = expected_markers.get(identity)
        if expected_checksum is not None and actual_checksum != expected_checksum:
            raise StorageLifecycleError("applied PostgreSQL migration checksum drift detected")

    identity_version = "identity-0001-0009"
    identity_checksum, _identity_paths = phases[identity_version]
    if ("postgres-sql", identity_version) not in markers:
        apply_identity_baseline_sql(
            identity_dsn=dsn,
            migrations_dir=manifest_path.parent,
            include_strategy_bindings=False,
        )
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=identity_version,
            checksum=identity_checksum,
        )

    run_alembic_upgrade_head(
        postgres_dsn=dsn,
        alembic_upgrade_runner=run_alembic_migrations_main,
    )
    for revision in revisions:
        _record_postgres_marker(
            dsn,
            store="postgres-alembic",
            version=revision.version,
            checksum=revision.sha256,
        )

    strategy_version = "strategy-0010"
    strategy_checksum, _strategy_paths = phases[strategy_version]
    if ("postgres-sql", strategy_version) not in markers:
        apply_strategy_exchange_bindings_sql(
            identity_dsn=dsn,
            migrations_dir=manifest_path.parent,
        )
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=strategy_version,
            checksum=strategy_checksum,
        )

    organization_version = "organization-0011"
    organization_checksum, _organization_paths = phases[organization_version]
    if ("postgres-sql", organization_version) not in markers:
        try:
            apply_organizations_rbac_audit_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("organization migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=organization_version,
            checksum=organization_checksum,
        )

    local_auth_version = "local-auth-0012"
    local_auth_checksum, _local_auth_paths = phases[local_auth_version]
    if ("postgres-sql", local_auth_version) not in markers:
        try:
            apply_local_auth_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("local auth migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=local_auth_version,
            checksum=local_auth_checksum,
        )

    oidc_version = "oidc-provider-0013"
    oidc_checksum, _oidc_paths = phases[oidc_version]
    if ("postgres-sql", oidc_version) not in markers:
        try:
            apply_oidc_provider_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("OIDC provider migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=oidc_version,
            checksum=oidc_checksum,
        )

    research_version = "research-tenancy-0014"
    research_checksum, _research_paths = phases[research_version]
    if ("postgres-sql", research_version) not in markers:
        try:
            apply_research_organization_isolation_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("research tenancy migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=research_version,
            checksum=research_checksum,
        )

    trading_version = "trading-tenancy-0015"
    trading_checksum, _trading_paths = phases[trading_version]
    if ("postgres-sql", trading_version) not in markers:
        try:
            apply_trading_organization_isolation_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("trading tenancy migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=trading_version,
            checksum=trading_checksum,
        )

    notifications_version = "notification-providers-0016"
    notifications_checksum, _notification_paths = phases[notifications_version]
    if ("postgres-sql", notifications_version) not in markers:
        try:
            apply_notification_provider_instances_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("notification provider migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=notifications_version,
            checksum=notifications_checksum,
        )

    extensions_version = "extensions-plugin-platform-0017"
    extensions_checksum, _extensions_paths = phases[extensions_version]
    if ("postgres-sql", extensions_version) not in markers:
        try:
            apply_extensions_plugin_platform_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("extensions plugin migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=extensions_version,
            checksum=extensions_checksum,
        )

    artifact_version = "artifact-store-0018"
    artifact_checksum, _artifact_paths = phases[artifact_version]
    if ("postgres-sql", artifact_version) not in markers:
        try:
            apply_artifact_store_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("artifact store migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=artifact_version,
            checksum=artifact_checksum,
        )

    job_runtime_version = "isolated-job-runtime-0019"
    job_runtime_checksum, _job_runtime_paths = phases[job_runtime_version]
    if ("postgres-sql", job_runtime_version) not in markers:
        try:
            apply_isolated_job_runtime_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError("isolated job runtime migration phase failed") from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=job_runtime_version,
            checksum=job_runtime_checksum,
        )

    execution_gateway_version = "execution-gateway-safety-0020"
    execution_gateway_checksum, _execution_gateway_paths = phases[
        execution_gateway_version
    ]
    if ("postgres-sql", execution_gateway_version) not in markers:
        try:
            apply_execution_gateway_mainnet_safety_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError(
                "execution gateway safety migration phase failed"
            ) from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=execution_gateway_version,
            checksum=execution_gateway_checksum,
        )

    control_audit_version = "control-operation-audit-0021"
    control_audit_checksum, _control_audit_paths = phases[control_audit_version]
    if ("postgres-sql", control_audit_version) not in markers:
        try:
            apply_control_operation_audit_sql(
                identity_dsn=dsn,
                migrations_dir=manifest_path.parent,
            )
        except Exception as error:  # noqa: BLE001
            raise StorageLifecycleError(
                "control operation audit migration phase failed"
            ) from error
        _record_postgres_marker(
            dsn,
            store="postgres-sql",
            version=control_audit_version,
            checksum=control_audit_checksum,
        )


def _load_clickhouse_migrations(manifest_path: Path) -> tuple[str, tuple[MigrationFile, ...]]:
    payload = _load_json_object(manifest_path, label="ClickHouse migration manifest")
    if payload.get("schema") != CLICKHOUSE_MANIFEST_SCHEMA:
        raise StorageLifecycleError("unsupported ClickHouse migration manifest schema")
    database_token = payload.get("database_token")
    raw_migrations = payload.get("migrations")
    if not isinstance(database_token, str) or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_-]{0,62}", database_token
    ):
        raise StorageLifecycleError("invalid ClickHouse database token")
    if not isinstance(raw_migrations, list) or not raw_migrations:
        raise StorageLifecycleError("ClickHouse migration manifest is empty")
    migrations: list[MigrationFile] = []
    for raw in raw_migrations:
        if not isinstance(raw, dict):
            raise StorageLifecycleError("invalid ClickHouse migration entry")
        version = raw.get("version")
        name = raw.get("name")
        relative = raw.get("path")
        expected_sha = raw.get("sha256")
        if not all(isinstance(value, str) for value in (version, name, relative, expected_sha)):
            raise StorageLifecycleError("invalid ClickHouse migration identity")
        if Path(cast(str, relative)).name != relative:
            raise StorageLifecycleError("unsafe ClickHouse migration path")
        path = manifest_path.parent / cast(str, relative)
        try:
            content = path.read_bytes()
        except OSError as error:
            raise StorageLifecycleError("missing ClickHouse migration source") from error
        if _sha256(content) != expected_sha:
            raise StorageLifecycleError("ClickHouse migration source checksum mismatch")
        migrations.append(
            MigrationFile(
                version=cast(str, version),
                name=cast(str, name),
                path=path,
                sha256=cast(str, expected_sha),
            )
        )
    versions = [migration.version for migration in migrations]
    if versions != sorted(versions) or len(versions) != len(set(versions)):
        raise StorageLifecycleError("ClickHouse migrations must be uniquely ordered")
    return database_token, tuple(migrations)


def split_clickhouse_sql(sql: str) -> tuple[str, ...]:
    """Split ClickHouse DDL without treating comments or quoted semicolons as boundaries."""

    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    line_comment = False
    block_comment = False
    index = 0
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
                current.append(char)
            index += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote is not None:
            current.append(char)
            if char == "\\" and next_char:
                current.append(next_char)
                index += 2
                continue
            if char == quote:
                if next_char == quote and quote in {"'", '"'}:
                    current.append(next_char)
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if char == "-" and next_char == "-":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char in {"'", '"', "`"}:
            quote = char
            current.append(char)
            index += 1
            continue
        if char == ";":
            statement = "".join(current).strip()
            if statement:
                statements.append(statement)
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    if quote is not None or block_comment:
        raise StorageLifecycleError("unterminated ClickHouse SQL quote or comment")
    final_statement = "".join(current).strip()
    if final_statement:
        statements.append(final_statement)
    return tuple(statements)


def _render_clickhouse_sql(source: bytes, *, database_token: str, database: str) -> bytes:
    text = source.decode("utf-8")
    rendered = re.sub(rf"\b{re.escape(database_token)}\b", database, text)
    if re.search(r"\b(?:DROP\s+DATABASE|DROP\s+TABLE|TRUNCATE\s+TABLE)\b", rendered, re.I):
        raise StorageLifecycleError("destructive ClickHouse DDL is forbidden")
    return rendered.encode()


def _ensure_clickhouse_state(client: Any, *, database: str) -> None:
    database_sql = _quote_clickhouse_identifier(database)
    client.command(f"CREATE DATABASE IF NOT EXISTS {database_sql}")
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {database_sql}.{CLICKHOUSE_STATE_TABLE} (
            store LowCardinality(String),
            version String,
            checksum FixedString(64),
            applied_at DateTime64(3, 'UTC') DEFAULT now64(3)
        )
        ENGINE = ReplacingMergeTree(applied_at)
        ORDER BY (store, version)
        """
    )


def _read_clickhouse_markers(client: Any, *, database: str) -> dict[str, str]:
    database_sql = _quote_clickhouse_identifier(database)
    result = client.query(
        f"""
        SELECT version, checksum
        FROM {database_sql}.{CLICKHOUSE_STATE_TABLE} FINAL
        WHERE store = 'clickhouse'
        ORDER BY version
        """
    )
    return {
        str(row[0]): row[1].decode("ascii") if isinstance(row[1], bytes) else str(row[1])
        for row in result.result_rows
    }


def apply_clickhouse_migrations(
    dsn: str,
    *,
    database: str,
    manifest_path: Path,
    auth: str | None = None,
) -> None:
    """Apply ordered idempotent ClickHouse DDL and persist per-source checksums."""

    database_token, migrations = _load_clickhouse_migrations(manifest_path)
    client = _open_clickhouse(dsn, auth=auth)
    database_sql = _quote_clickhouse_identifier(database)
    try:
        _ensure_clickhouse_state(client, database=database)
        markers = _read_clickhouse_markers(client, database=database)
        expected: dict[str, str] = {}
        rendered_sources: dict[str, bytes] = {}
        for migration in migrations:
            rendered = _render_clickhouse_sql(
                migration.path.read_bytes(),
                database_token=database_token,
                database=database,
            )
            rendered_sources[migration.version] = rendered
            expected[migration.version] = _sha256(rendered)
        for version, checksum in markers.items():
            if version in expected and checksum != expected[version]:
                raise StorageLifecycleError("applied ClickHouse migration checksum drift detected")
        for migration in migrations:
            if migration.version in markers:
                continue
            for statement in split_clickhouse_sql(rendered_sources[migration.version].decode()):
                if re.match(r"^USE\s+", statement, re.I):
                    continue
                client.command(statement)
            client.command(
                f"""
                INSERT INTO {database_sql}.{CLICKHOUSE_STATE_TABLE}
                    (store, version, checksum)
                VALUES
                    ('clickhouse', {{version:String}}, {{checksum:String}})
                """,
                parameters={"version": migration.version, "checksum": expected[migration.version]},
            )
    except StorageLifecycleError:
        raise
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("ClickHouse migration failed") from error
    finally:
        client.close()


def _postgres_schema_status(
    dsn: str,
    *,
    repo_root: Path,
    manifest_path: Path,
) -> list[str]:
    phases = _load_postgres_phases(manifest_path)
    revisions = _alembic_revision_files(repo_root)
    markers = _read_postgres_markers(dsn)
    for version, (checksum, _paths) in phases.items():
        if markers.get(("postgres-sql", version)) != checksum:
            raise StorageLifecycleError("PostgreSQL SQL phase is not ready")
    for revision in revisions:
        if markers.get(("postgres-alembic", revision.version)) != revision.sha256:
            raise StorageLifecycleError("PostgreSQL Alembic history is not ready")
    try:
        with psycopg.connect(
            normalize_psycopg_dsn(dsn=dsn),
            autocommit=True,
            row_factory=cast(Any, dict_row),
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(cast(Any, "SELECT version_num FROM alembic_version"))
                row = cast(Mapping[str, Any] | None, cursor.fetchone())
                cursor.execute(
                    cast(
                        Any,
                        """
                        SELECT
                            to_regclass('public.identity_users') IS NOT NULL AS identity_ready,
                            to_regclass('public.strategy_strategies') IS NOT NULL AS strategy_ready,
                            to_regclass('public.strategy_exchange_bindings') IS NOT NULL
                                AS binding_ready,
                            to_regclass('public.identity_webauthn_credentials') IS NOT NULL
                                AS local_auth_ready,
                            to_regclass('public.identity_external_identities') IS NOT NULL
                                AS oidc_ready,
                            EXISTS (
                                SELECT 1
                                FROM information_schema.columns
                                WHERE table_schema = 'public'
                                  AND table_name = 'backtest_job_top_variants'
                                  AND column_name = 'organization_id'
                            ) AS research_ready,
                            EXISTS (
                                SELECT 1
                                FROM information_schema.columns
                                WHERE table_schema = 'public'
                                  AND table_name = 'execution_orders'
                                  AND column_name = 'organization_id'
                            ) AS trading_ready,
                            to_regclass('public.extensions_plugin_packages') IS NOT NULL
                                AS extensions_ready,
                            to_regclass('public.artifact_store_manifests') IS NOT NULL
                                AS artifact_store_ready,
                            to_regclass('public.job_runtime_attempts') IS NOT NULL
                                AS job_runtime_ready,
                            to_regclass('public.execution_mainnet_approvals') IS NOT NULL
                                AS execution_gateway_ready
                        """,
                    )
                )
                table_row = cast(Mapping[str, Any] | None, cursor.fetchone())
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("cannot read PostgreSQL schema version") from error
    head = revisions[-1].version
    if row is None or str(row["version_num"]) != head:
        raise StorageLifecycleError("PostgreSQL Alembic head is not ready")
    if table_row is None or not all(bool(table_row[key]) for key in table_row):
        raise StorageLifecycleError("PostgreSQL required schema tables are missing")
    return [
        "identity-0001-0009",
        f"alembic:{head}",
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
    ]


def _clickhouse_schema_status(
    dsn: str,
    *,
    database: str,
    manifest_path: Path,
    auth: str | None = None,
) -> list[str]:
    database_token, migrations = _load_clickhouse_migrations(manifest_path)
    client = _open_clickhouse(dsn, auth=auth)
    try:
        _ensure_clickhouse_state(client, database=database)
        markers = _read_clickhouse_markers(client, database=database)
        for migration in migrations:
            rendered = _render_clickhouse_sql(
                migration.path.read_bytes(),
                database_token=database_token,
                database=database,
            )
            if markers.get(migration.version) != _sha256(rendered):
                raise StorageLifecycleError("ClickHouse schema migration is not ready")
        database_sql = _quote_clickhouse_identifier(database)
        count = int(
            client.command(
                """
                SELECT count()
                FROM system.tables
                WHERE database = {database:String}
                  AND name IN ('canonical_candles_1m', 'canonical_funding_rates')
                """,
                parameters={"database": database},
            )
        )
        if count != 2:
            raise StorageLifecycleError("ClickHouse required schema tables are missing")
        client.command(f"SELECT count() FROM {database_sql}.{CLICKHOUSE_STATE_TABLE}")
    except StorageLifecycleError:
        raise
    except Exception as error:  # noqa: BLE001
        raise StorageLifecycleError("cannot read ClickHouse schema version") from error
    finally:
        client.close()
    return [migration.version for migration in migrations]


def build_storage_status(
    *,
    config: StorageConfig,
    endpoints: StorageEndpoints,
    repo_root: Path,
    postgres_manifest: Path,
    clickhouse_manifest: Path,
    require_schema: bool,
) -> dict[str, Any]:
    """Build a secret-free status contract for future `roehubctl` consumers."""

    postgres_capabilities = check_postgres_capabilities(endpoints.postgresql_dsn)
    redis_capabilities = check_redis_capabilities(
        endpoints.redis_url,
        auth=endpoints.redis_auth,
    )
    postgres_versions = (
        _postgres_schema_status(
            endpoints.postgresql_dsn,
            repo_root=repo_root,
            manifest_path=postgres_manifest,
        )
        if require_schema
        else []
    )
    stores: dict[str, Any] = {
        "postgresql": {
            "mode": config.stores["postgresql"].mode,
            **postgres_capabilities,
            "schema_versions": postgres_versions,
            "ready": True,
        },
        "redis": {
            "mode": config.stores["redis"].mode,
            **redis_capabilities,
            "schema_versions": [],
            "ready": True,
        },
    }
    if "clickhouse" in config.stores:
        if endpoints.clickhouse_dsn is None:
            raise StorageLifecycleError("ClickHouse endpoint is required by profile")
        clickhouse_capabilities = check_clickhouse_capabilities(
            endpoints.clickhouse_dsn,
            database=config.stores["clickhouse"].database,
            auth=endpoints.clickhouse_auth,
        )
        clickhouse_versions = (
            _clickhouse_schema_status(
                endpoints.clickhouse_dsn,
                database=config.stores["clickhouse"].database,
                manifest_path=clickhouse_manifest,
                auth=endpoints.clickhouse_auth,
            )
            if require_schema
            else []
        )
        stores["clickhouse"] = {
            "mode": config.stores["clickhouse"].mode,
            **clickhouse_capabilities,
            "schema_versions": clickhouse_versions,
            "ready": True,
        }
    return {
        "schema": STORAGE_STATUS_SCHEMA,
        "profile": config.profile,
        "ready": True,
        "stores": stores,
        "backup_prerequisites": {
            "postgresql": [
                "credentials-restorable-separately",
                "pg-dump-or-consistent-volume-snapshot",
                "schema-status-ready",
            ],
            "clickhouse": [
                "consistent-parts-or-freeze-backup",
                "credentials-restorable-separately",
                "schema-status-ready",
            ],
            "redis": [
                "not-a-durable-source-of-truth",
                "rebuild-path-documented",
            ],
        },
    }


def bootstrap_storage(
    *,
    config: StorageConfig,
    endpoints: StorageEndpoints,
    repo_root: Path,
    postgres_manifest: Path,
    clickhouse_manifest: Path,
) -> dict[str, Any]:
    """Run the single ordered storage bootstrap and return its readiness proof."""

    check_postgres_capabilities(endpoints.postgresql_dsn)
    check_redis_capabilities(
        endpoints.redis_url,
        auth=endpoints.redis_auth,
    )
    if "clickhouse" in config.stores:
        if endpoints.clickhouse_dsn is None:
            raise StorageLifecycleError("ClickHouse endpoint is required by profile")
        check_clickhouse_capabilities(
            endpoints.clickhouse_dsn,
            database=config.stores["clickhouse"].database,
            auth=endpoints.clickhouse_auth,
        )

    apply_postgres_migrations(
        endpoints.postgresql_dsn,
        repo_root=repo_root,
        manifest_path=postgres_manifest,
    )
    if "clickhouse" in config.stores:
        assert endpoints.clickhouse_dsn is not None
        apply_clickhouse_migrations(
            endpoints.clickhouse_dsn,
            database=config.stores["clickhouse"].database,
            manifest_path=clickhouse_manifest,
            auth=endpoints.clickhouse_auth,
        )
    return build_storage_status(
        config=config,
        endpoints=endpoints,
        repo_root=repo_root,
        postgres_manifest=postgres_manifest,
        clickhouse_manifest=clickhouse_manifest,
        require_schema=True,
    )
