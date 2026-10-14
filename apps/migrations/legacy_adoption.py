"""Explicit legacy schema adoption; never infer historical migration execution.

Only a clean, fully migrated reference may produce the schema witness. The target
must match its complete public schema and validated constraints before missing SQL
phases are recorded with the distinct postgres-sql-adopted provenance.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from psycopg.rows import tuple_row

from apps.migrations.market_data import market_data_schema_ready

# Only the specifically audited pre-PG-only compatibility gap may be adopted.
ADOPTABLE_PHASES = frozenset(
    {
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
        "market-data-queue-events-0026",
    }
)


SCHEMA_SQL = """
SELECT 'column', c.relname||'.'||a.attname,
 jsonb_build_array(a.attnum,format_type(a.atttypid,a.atttypmod),a.attnotnull,
 a.attidentity,a.attgenerated,pg_get_expr(d.adbin,d.adrelid))::text
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum
WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m')
 AND c.relname<>'roehub_storage_migrations'
UNION ALL
SELECT 'constraint',c.relname||'.'||k.conname,
 jsonb_build_array(pg_get_constraintdef(k.oid),k.convalidated,k.condeferrable,k.condeferred)::text
FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid
JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='public' AND c.relname<>'roehub_storage_migrations'
UNION ALL
SELECT 'index',c.relname||'.'||i.relname,
 jsonb_build_array(pg_get_indexdef(x.indexrelid),x.indisvalid,x.indisready)::text
FROM pg_index x JOIN pg_class c ON c.oid=x.indrelid JOIN pg_class i ON i.oid=x.indexrelid
JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='public' AND c.relname<>'roehub_storage_migrations'
UNION ALL
SELECT 'function',p.proname||'('||pg_get_function_identity_arguments(p.oid)||')',
 pg_get_functiondef(p.oid) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
WHERE n.nspname='public' AND p.prokind IN ('f','p')
UNION ALL
SELECT 'trigger',c.relname||'.'||t.tgname,
 jsonb_build_array(pg_get_triggerdef(t.oid),t.tgenabled)::text
FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid
JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND NOT t.tgisinternal
UNION ALL
SELECT 'relation',c.relname,
 jsonb_build_array(c.relkind,c.relrowsecurity,c.relforcerowsecurity)::text
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m')
 AND c.relname<>'roehub_storage_migrations'
UNION ALL
SELECT 'policy',c.relname||'.'||p.polname,
 jsonb_build_array(p.polcmd,p.polpermissive,pg_get_expr(p.polqual,p.polrelid),
 pg_get_expr(p.polwithcheck,p.polrelid),
 ARRAY(SELECT rolname FROM pg_roles WHERE oid=ANY(p.polroles)))::text
FROM pg_policy p JOIN pg_class c ON c.oid=p.polrelid
JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public'
ORDER BY 1,2
"""


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def schema_witness(connection: Any) -> list[list[str]]:
    with connection.cursor(row_factory=tuple_row) as cursor:
        cursor.execute("SET LOCAL search_path=public,pg_catalog")
        rows = [list(row) for row in cursor.execute(SCHEMA_SQL).fetchall()]
        # Some legacy bootstraps retained SQL0008's redundant single-column FK
        # alongside SQL0015's validated organization-scoped composite FK. The
        # latter implies the former. Preserve it physically; normalize only this
        # exact, validated pair, never arbitrary extra constraints or weak FKs.
        old = [
            "constraint",
            "exchange_connections.exchange_connections_active_credential_version_id_fk",
            json.dumps(
                [
                    "FOREIGN KEY (active_credential_version_id) REFERENCES "
                    "exchange_credential_versions(credential_version_id) "
                    "DEFERRABLE INITIALLY DEFERRED",
                    True,
                    True,
                    True,
                ]
            ),
        ]
        stronger = [
            "constraint",
            "exchange_connections.exchange_connections_org_active_credential_fk",
            json.dumps(
                [
                    "FOREIGN KEY (organization_id, active_credential_version_id) REFERENCES "
                    "exchange_credential_versions(organization_id, credential_version_id) "
                    "DEFERRABLE INITIALLY DEFERRED",
                    True,
                    True,
                    True,
                ]
            ),
        ]
        if old in rows and stronger in rows:
            rows.remove(old)
        return sorted(rows)


def adopt(
    connection: Any, reference: dict[str, Any], phases: dict[str, Any], *, lock: int
) -> list[str]:
    """Caller owns a counted admin transaction and a quiesced installation."""
    expected = {name: item[0] for name, item in phases.items()}
    if (
        reference.get("schema") != "io.roehub.legacy-schema-witness/v1"
        or reference.get("phases") != expected
    ):
        raise ValueError("adoption reference does not match immutable current phases")
    if reference.get("schema_sha256") != digest(reference.get("objects")):
        raise ValueError("adoption witness digest mismatch")
    connection.execute("SELECT pg_advisory_xact_lock(%s)", (lock,))
    objects = schema_witness(connection)
    if objects != reference["objects"]:
        previous = {(a, b): c for a, b, c in reference["objects"]}
        current = {(a, b): c for a, b, c in objects}
        changed = [
            f"{a}:{b}"
            for a, b in sorted(previous.keys() | current.keys())
            if previous.get((a, b)) != current.get((a, b))
        ]
        raise ValueError("legacy schema differs from reference: " + ",".join(changed[:20]))
    rows = connection.execute(
        "SELECT store,version,checksum FROM roehub_storage_migrations"
    ).fetchall()
    present: dict[str, str] = {}
    for store, version, checksum in rows:
        if store in ("postgres-sql", "postgres-sql-adopted"):
            if version not in expected or expected[version] != checksum:
                raise ValueError("known SQL phase checksum mismatch")
            present[version] = checksum
    actual = {version for store, version, _ in rows if store == "postgres-sql"}
    if any(version not in actual for version in expected.keys() - ADOPTABLE_PHASES):
        raise ValueError("non-adoptable phases require actual execution markers")
    if not market_data_schema_ready(connection):
        raise ValueError("actual market schema and current partitions required")
    verify_seed_postconditions(connection)
    added = []
    for version, checksum in expected.items():
        if version in present:
            continue
        connection.execute(
            "INSERT INTO roehub_storage_migrations(store,version,checksum) VALUES (%s,%s,%s)",
            ("postgres-sql-adopted", version, checksum),
        )
        added.append(version)
    connection.execute(
        "INSERT INTO roehub_storage_migrations(store,version,checksum) "
        "VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
        ("postgres-adoption-reference", digest(expected), reference["schema_sha256"]),
    )
    return added


def verify_seed_postconditions(connection: Any) -> None:
    """Verify durable seed identity, without resetting mutable user configuration."""
    packages = connection.execute("""
        SELECT count(*) FROM notification_provider_packages
        WHERE built_in AND contract_version='NotificationProvider/v1'
          AND (package_id,provider_key) IN (
            ('00000000-0000-4000-8000-000000000101'::uuid,'log_only'),
            ('00000000-0000-4000-8000-000000000102'::uuid,'fake'),
            ('00000000-0000-4000-8000-000000000103'::uuid,'telegram_bot_api'))
    """).fetchone()
    instances = connection.execute("""
        SELECT count(*) FROM notification_provider_instances
        WHERE scope='installation' AND organization_id IS NULL
          AND (instance_id,package_id,provider_key) IN (
            ('00000000-0000-4000-8000-000000000001'::uuid,
             '00000000-0000-4000-8000-000000000101'::uuid,'log_only'),
            ('00000000-0000-4000-8000-000000000002'::uuid,
             '00000000-0000-4000-8000-000000000102'::uuid,'fake'))
    """).fetchone()
    cursor = connection.execute(
        "SELECT count(*) FROM control_operation_audit_cursor WHERE singleton"
    ).fetchone()
    unobserved = connection.execute("""
        SELECT count(*) FROM market_data_work_requests w WHERE NOT EXISTS (
          SELECT 1 FROM market_data_work_events e WHERE e.job_id=w.job_id
          AND e.organization_id=w.organization_id)
    """).fetchone()
    if packages != (3,) or instances != (2,) or cursor != (1,) or unobserved != (0,):
        raise ValueError("legacy seed postconditions are not established")
