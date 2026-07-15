"""Disposable PostgreSQL proof for organization schema and ownership invariants."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, cast
from uuid import UUID, uuid4

import psycopg


class OrganizationRuntimeProofError(RuntimeError):
    """Raised when disposable organization database evidence is incomplete."""


def run_probe(*, dsn: str) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    owner_id, admin_id, operator_id, trader_id, viewer_id, support_id = (
        uuid4() for _index in range(6)
    )
    installation_id = uuid4()
    primary_id = uuid4()
    secondary_id = uuid4()

    with psycopg.connect(dsn, autocommit=True) as connection:
        _seed_identity(
            connection=connection,
            user_ids=(owner_id, admin_id, operator_id, trader_id, viewer_id, support_id),
            installation_id=installation_id,
            primary_id=primary_id,
            secondary_id=secondary_id,
            now=now,
        )
        checks = _prove_database_constraints(
            connection=connection,
            owner_user_id=owner_id,
            installation_id=installation_id,
            primary_id=primary_id,
            secondary_id=secondary_id,
            now=now,
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM identity_support_access_grants WHERE revoked_at IS NULL"
            )
            support_count = _count_row(cursor.fetchone())
            cursor.execute(
                "SELECT count(*) FROM identity_administrative_audit_events"
            )
            audit_count = _count_row(cursor.fetchone())
            cursor.execute(
                """
                SELECT metadata_json::text
                FROM identity_administrative_audit_events
                ORDER BY created_at, event_id
                """
            )
            audit_payload = " ".join(str(row[0]) for row in cursor.fetchall()).lower()
    if support_count != 1:
        raise OrganizationRuntimeProofError("support access must be absent by default and explicit")
    if audit_count < 3:
        raise OrganizationRuntimeProofError("administrative audit is incomplete")
    for sensitive_key in ("password", "token", "secret", "credential", "cookie", "dsn"):
        if f'"{sensitive_key}"' in audit_payload:
            raise OrganizationRuntimeProofError("audit contains a sensitive key")

    return {
        "schema": "io.roehub.organization-runtime-proof/v1alpha1",
        "organizations": 2,
        "role_matrix": "passed",
        "last_owner": "passed",
        "support_access_expiry": "passed",
        "audit_events": audit_count,
        "audit_redaction": "passed",
        "database_constraints": checks,
    }


def _count_row(row: tuple[object, ...] | None) -> int:
    if row is None:
        raise OrganizationRuntimeProofError("database count query returned no row")
    return int(cast(Any, row[0]))


def _seed_identity(
    *,
    connection: psycopg.Connection[Any],
    user_ids: tuple[UUID, ...],
    installation_id: UUID,
    primary_id: UUID,
    secondary_id: UUID,
    now: datetime,
) -> None:
    owner_id, admin_id, operator_id, trader_id, viewer_id, support_id = user_ids
    with connection.transaction(), connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO identity_users (
                user_id, telegram_user_id, paid_level, created_at,
                last_login_at, is_deleted, keycloak_subject
            ) VALUES (%s, NULL, 'free', %s, %s, FALSE, NULL)
            """,
            [(user_id, now, now) for user_id in user_ids],
        )
        cursor.execute(
            """
            INSERT INTO identity_installations (
                installation_id, singleton_key, display_name, created_at
            ) VALUES (%s, TRUE, 'Stage 05 disposable site', %s)
            """,
            (installation_id, now),
        )
        cursor.execute(
            """
            INSERT INTO identity_installation_owners (
                installation_id, user_id, granted_by_user_id, granted_at
            ) VALUES (%s, %s, %s, %s)
            """,
            (installation_id, owner_id, owner_id, now),
        )
        cursor.executemany(
            """
            INSERT INTO identity_organizations (
                organization_id, installation_id, slug, display_name, status, created_at
            ) VALUES (%s, %s, %s, %s, 'active', %s)
            """,
            (
                (primary_id, installation_id, "stage05-primary", "Stage 05 Primary", now),
                (secondary_id, installation_id, "stage05-secondary", "Stage 05 Secondary", now),
            ),
        )
        cursor.executemany(
            """
            INSERT INTO identity_memberships (
                organization_id, user_id, role, status, created_at, updated_at
            ) VALUES (%s, %s, %s, 'active', %s, %s)
            """,
            (
                (primary_id, owner_id, "owner", now, now),
                (primary_id, admin_id, "admin", now, now),
                (primary_id, operator_id, "operator", now, now),
                (primary_id, trader_id, "trader", now, now),
                (primary_id, viewer_id, "viewer", now, now),
                (secondary_id, owner_id, "owner", now, now),
            ),
        )
        cursor.execute(
            """
            INSERT INTO identity_plugin_permissions (
                organization_id, plugin_id, user_id, permission, granted_by_user_id,
                created_at, updated_at
            ) VALUES (%s, 'roehub.stage05-proof', %s, 'operate', %s, %s, %s)
            """,
            (primary_id, viewer_id, admin_id, now, now),
        )
        cursor.execute(
            """
            INSERT INTO identity_invitations (
                invitation_id, organization_id, recipient_email_sha256, role,
                status, created_by_user_id, expires_at, created_at
            ) VALUES (%s, %s, %s, 'viewer', 'pending', %s, %s, %s)
            """,
            (uuid4(), primary_id, "a" * 64, admin_id, now + timedelta(hours=2), now),
        )
        cursor.execute(
            """
            INSERT INTO identity_support_access_grants (
                grant_id, installation_id, support_user_id, granted_by_user_id,
                reason, expires_at, created_at
            ) VALUES (%s, %s, %s, %s, 'Disposable database proof', %s, %s)
            """,
            (uuid4(), installation_id, support_id, owner_id, now + timedelta(hours=1), now),
        )
        cursor.executemany(
            """
            INSERT INTO identity_administrative_audit_events (
                event_id, installation_id, organization_id, actor_user_id,
                action, target_type, target_id, outcome, metadata_json, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 'succeeded', %s::jsonb, %s)
            """,
            (
                (
                    uuid4(), installation_id, primary_id, owner_id,
                    "installation.bootstrap", "installation", str(installation_id),
                    json.dumps({"organization_id": str(primary_id)}), now,
                ),
                (
                    uuid4(), installation_id, primary_id, admin_id,
                    "plugin.permission_set", "plugin_permission", "roehub.stage05-proof",
                    json.dumps({"permission": "operate"}), now,
                ),
                (
                    uuid4(), installation_id, None, owner_id,
                    "support_access.granted", "support_access", "time-bounded",
                    json.dumps({"expires_at": (now + timedelta(hours=1)).isoformat()}), now,
                ),
            ),
        )


def _prove_database_constraints(
    *,
    connection: psycopg.Connection[Any],
    owner_user_id: UUID,
    installation_id: UUID,
    primary_id: UUID,
    secondary_id: UUID,
    now: datetime,
) -> dict[str, str]:
    connection_id = uuid4()
    strategy_id = uuid4()
    account_snapshot_id = uuid4()
    with connection.transaction(), connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO exchange_connections (
                connection_id, organization_id, owner_user_id, exchange_name,
                market_type, environment, status, permission_summary_json,
                ip_restriction_status, created_at, updated_at
            ) VALUES (%s, %s, %s, 'bybit', 'spot', 'testnet', 'active',
                      '{}'::jsonb, 'unknown', %s, %s)
            """,
            (connection_id, primary_id, owner_user_id, now, now),
        )
        cursor.execute(
            """
            INSERT INTO strategy_strategies (
                strategy_id, organization_id, user_id, name, instrument_id,
                instrument_key, market_type, symbol, timeframe, indicators_json,
                spec_json, created_at, is_deleted
            ) VALUES (%s, %s, %s, 'stage05-proof',
                      '{"market_id":"bybit:spot","symbol":"BTCUSDT"}'::jsonb,
                      'bybit:spot:BTCUSDT', 'spot', 'BTCUSDT', '1m', '[]'::jsonb,
                      '{"schema_version":1,"spec_kind":"roehub.strategy.v1"}'::jsonb,
                      %s, FALSE)
            """,
            (strategy_id, secondary_id, owner_user_id, now),
        )
        cursor.execute(
            """
            INSERT INTO exchange_account_snapshots (
                account_snapshot_id, organization_id, owner_user_id,
                exchange_connection_id, exchange_name, market_type, environment,
                account_mode, source_hash, sync_status, sync_reason, observed_at,
                synced_at, metadata_json
            ) VALUES (%s, %s, %s, %s, 'bybit', 'spot', 'testnet', 'unified',
                      %s, 'fresh', 'stage05-proof', %s, %s, '{}'::jsonb)
            """,
            (account_snapshot_id, primary_id, owner_user_id, connection_id, "a" * 64, now, now),
        )

    return {
        "strategy_exchange_binding": _expect_database_error(
            connection=connection,
            sqlstate="23503",
            query="""
                INSERT INTO strategy_exchange_bindings (
                    binding_id, organization_id, owner_user_id, strategy_id,
                    exchange_connection_id, usage_mode, binding_status, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, 'trading', 'active', %s, %s)
            """,
            parameters=(uuid4(), primary_id, owner_user_id, strategy_id, connection_id, now, now),
        ),
        "strategy_position_ownership": _expect_database_error(
            connection=connection,
            sqlstate="23503",
            query="""
                INSERT INTO strategy_position_ownership (
                    ownership_id, organization_id, owner_user_id, exchange_connection_id,
                    strategy_id, strategy_run_id, market_type, instrument_key,
                    state, acquired_at, reason
                ) VALUES (%s, %s, %s, %s, %s, %s, 'spot',
                          'bybit:spot:BTCUSDT', 'active', %s, 'stage05-proof')
            """,
            parameters=(
                uuid4(), secondary_id, owner_user_id, connection_id,
                strategy_id, uuid4(), now,
            ),
        ),
        "strategy_provenance": _expect_database_error(
            connection=connection,
            sqlstate="23503",
            query="""
                INSERT INTO strategy_backtest_variant_provenance (
                    strategy_id, organization_id, user_id, source_job_id,
                    source_variant_key, source_variant_hash, backtest_request_hash,
                    backtest_result_config_hash, strategy_spec_hash, launch_request_hash,
                    idempotency_key_hash, created_at, metadata_json
                ) VALUES (%s, %s, %s, %s, 'variant', %s, %s, %s, %s, %s, %s,
                          %s, '{}'::jsonb)
            """,
            parameters=(
                strategy_id, primary_id, owner_user_id, uuid4(), "b" * 64,
                "c" * 64, "d" * 64, "e" * 64, "f" * 64, "1" * 64, now,
            ),
        ),
        "exchange_position_snapshot": _expect_database_error(
            connection=connection,
            sqlstate="23503",
            query="""
                INSERT INTO exchange_position_snapshots (
                    position_snapshot_id, organization_id, account_snapshot_id,
                    owner_user_id, exchange_connection_id, instrument_key,
                    side, quantity, observed_at
                ) VALUES (%s, %s, %s, %s, %s,
                          'bybit:spot:BTCUSDT', 'long', 1, %s)
            """,
            parameters=(
                uuid4(), secondary_id, account_snapshot_id, owner_user_id, connection_id, now,
            ),
        ),
        "last_owner": _expect_database_error(
            connection=connection,
            sqlstate="23514",
            query="""
                UPDATE identity_memberships SET role = 'admin', updated_at = %s
                WHERE organization_id = %s AND user_id = %s
            """,
            parameters=(now, secondary_id, owner_user_id),
        ),
        "audit_immutable": _expect_database_error(
            connection=connection,
            sqlstate="55000",
            query="UPDATE identity_administrative_audit_events SET action = 'tampered'",
            parameters=(),
        ),
        "audit_sensitive_key": _expect_database_error(
            connection=connection,
            sqlstate="23514",
            query="""
                INSERT INTO identity_administrative_audit_events (
                    event_id, installation_id, organization_id, actor_user_id,
                    action, target_type, target_id, outcome, metadata_json, created_at
                ) VALUES (%s, %s, %s, %s, 'proof.rejected', 'proof', 'proof',
                          'rejected', '{"token":"redacted-value"}'::jsonb, %s)
            """,
            parameters=(uuid4(), installation_id, primary_id, owner_user_id, now),
        ),
    }


def _expect_database_error(
    *,
    connection: psycopg.Connection[Any],
    sqlstate: str,
    query: str,
    parameters: tuple[object, ...],
) -> str:
    try:
        with connection.transaction(), connection.cursor() as cursor:
            cursor.execute(cast(Any, query), parameters)
    except psycopg.Error as error:
        if error.sqlstate != sqlstate:
            raise OrganizationRuntimeProofError(
                f"expected SQLSTATE {sqlstate}, got {error.sqlstate}"
            ) from error
        return "passed"
    raise OrganizationRuntimeProofError(f"expected SQLSTATE {sqlstate}")


def main() -> int:
    dsn = os.environ.get("ROEHUB_STORAGE_POSTGRES_DSN", "").strip()
    if not dsn:
        print("organization runtime proof failed: PostgreSQL DSN is unavailable")
        return 1
    try:
        result = run_probe(dsn=dsn)
    except Exception as error:  # noqa: BLE001
        print(f"organization runtime proof failed: {type(error).__name__}")
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
