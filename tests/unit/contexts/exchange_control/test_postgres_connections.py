from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from trading.contexts.exchange_control.adapters.outbound import postgres_connections
from trading.contexts.exchange_control.application.connections import ExchangeConnectionRecord
from trading.shared_kernel.primitives import UserId


def test_fingerprint_text_accepts_transit_text_values() -> None:
    assert postgres_connections._fingerprint_text("hmac-sha256:abc123") == (
        "hmac-sha256:abc123"
    )
    assert postgres_connections._fingerprint_text(memoryview(b"hmac-sha256:abc123")) == (
        "hmac-sha256:abc123"
    )


def test_fingerprint_text_maps_legacy_binary_hashes_to_stable_hex() -> None:
    legacy_hash = bytes([0, 160, 255]) + (b"x" * 29)

    assert postgres_connections._fingerprint_text(legacy_hash) == (
        "legacy-bytea-sha256:00a0ff" + ("78" * 29)
    )


def test_connection_parameters_preserve_create_validation_metadata() -> None:
    observed_at = datetime(2026, 6, 19, 1, 30, tzinfo=timezone.utc)
    connection = ExchangeConnectionRecord(
        connection_id=UUID("00000000-0000-0000-0000-000000000501"),
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000601"),
        exchange_name="bybit",
        market_type="futures",
        environment="testnet",
        label="bybit_testnet",
        permissions="trade",
        active_credential_version_id=UUID("00000000-0000-0000-0000-000000000701"),
        status="active",
        status_reason=None,
        validation_status="valid_trade_enabled",
        validation_reason="write_permission_detected",
        ip_restriction_status="not_restricted_testnet",
        last_validated_at=observed_at,
        created_at=observed_at,
        updated_at=observed_at,
        permission_summary={
            "permissions": "trade",
            "requested_permissions": "trade",
            "exchange_permissions": "trade",
            "effective_permissions": "trade",
            "validation_status": "valid_trade_enabled",
            "validation_reason": "write_permission_detected",
            "permission_warnings": [],
        },
    )

    parameters = postgres_connections._connection_parameters(connection=connection)

    assert parameters["ip_restriction_status"] == "not_restricted_testnet"
    assert parameters["last_validated_at"] == observed_at
