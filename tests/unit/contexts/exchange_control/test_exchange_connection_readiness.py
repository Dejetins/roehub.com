from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from trading.contexts.exchange_control.application.connections import (
    RECLASSIFIED_NON_TRADING_STATUS_REASON,
    ExchangeConnectionError,
    ExchangeConnectionService,
    InMemoryExchangeConnectionRepository,
)
from trading.contexts.exchange_control.application.secret_cipher import (
    DeterministicInMemoryExchangeSecretCipher,
)
from trading.contexts.exchange_control.application.validation import (
    ExchangeCredentialValidationRequest,
    ExchangeCredentialValidationResult,
)
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True)
class _StaticValidator:
    result: ExchangeCredentialValidationResult
    requires_plaintext: bool = False

    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
        now: datetime,
    ) -> ExchangeCredentialValidationResult:
        _ = request, now
        return self.result


def test_connection_create_exposes_trading_capability_needing_validation() -> None:
    service = _service()
    created = _create_connection(service=service)

    assert created.requested_capability == "trading"
    assert created.effective_capability == "none"
    assert created.connection_readiness == "needs_action"
    assert created.connection_readiness_reason == "validation_required"
    assert created.permissions_deprecated is True
    assert created.permissions == "read"
    assert created.requested_permissions == "read"


@pytest.mark.parametrize(
    (
        "result",
        "expected_effective_capability",
        "expected_readiness",
        "expected_reason",
    ),
    [
        (
            ExchangeCredentialValidationResult(
                status="valid_trade_enabled",
                reason="trade_permission_detected",
                ip_restriction_status="restricted",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "trade",
                    "effective_permissions": "trade",
                    "permission_warnings": [],
                },
            ),
            "trading",
            "ready_for_trading",
            "trading_policy_ok",
        ),
        (
            ExchangeCredentialValidationResult(
                status="valid_readonly",
                reason="readonly_permission_detected",
                ip_restriction_status="not_restricted_testnet",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "read",
                    "effective_permissions": "read",
                    "permission_warnings": [],
                },
            ),
            "none",
            "rejected",
            "read_only_not_supported",
        ),
        (
            ExchangeCredentialValidationResult(
                status="invalid_permissions",
                reason="withdraw_or_transfer_enabled",
                ip_restriction_status="restricted",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "withdraw_or_transfer",
                    "effective_permissions": "none",
                    "permission_warnings": [],
                },
            ),
            "none",
            "rejected",
            "unsafe_permissions",
        ),
        (
            ExchangeCredentialValidationResult(
                status="invalid_credentials",
                reason="exchange_rejected_credentials_400",
                ip_restriction_status="unknown",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "unknown",
                    "effective_permissions": "none",
                    "permission_warnings": [],
                },
            ),
            "none",
            "rejected",
            "invalid_credentials",
        ),
        (
            ExchangeCredentialValidationResult(
                status="invalid_ip_restriction",
                reason="mainnet_ip_restriction_missing",
                ip_restriction_status="missing_mainnet_restriction",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "trade",
                    "effective_permissions": "none",
                    "permission_warnings": [],
                },
            ),
            "none",
            "needs_action",
            "ip_restriction_required",
        ),
        (
            ExchangeCredentialValidationResult(
                status="skipped_external_validation",
                reason="live_validation_disabled",
                ip_restriction_status="not_checked",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "unknown",
                    "effective_permissions": "none",
                    "permission_warnings": [],
                },
            ),
            "none",
            "needs_action",
            "validation_required",
        ),
    ],
)
def test_validation_maps_to_trading_capability_readiness_truth_table(
    result: ExchangeCredentialValidationResult,
    expected_effective_capability: str,
    expected_readiness: str,
    expected_reason: str,
) -> None:
    service = _service()
    created = _create_connection(service=service, permissions="trade")

    validated = service.validate_connection(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        connection_id=created.connection_id,
        validator=_StaticValidator(result=result),
        now=datetime(2026, 5, 26, 12, 1, tzinfo=timezone.utc),
    )

    assert validated.requested_capability == "trading"
    assert validated.effective_capability == expected_effective_capability
    assert validated.connection_readiness == expected_readiness
    assert validated.connection_readiness_reason == expected_reason
    assert validated.permissions_deprecated is True


def test_legacy_permissions_remain_readable_but_not_authoritative() -> None:
    service = _service()
    created = _create_connection(service=service, permissions="trade")

    validated = service.validate_connection(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        connection_id=created.connection_id,
        validator=_StaticValidator(
            result=ExchangeCredentialValidationResult(
                status="valid_readonly",
                reason="readonly_permission_detected",
                ip_restriction_status="not_restricted_testnet",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "read",
                    "effective_permissions": "read",
                    "permission_warnings": [],
                },
            )
        ),
        now=datetime(2026, 5, 26, 12, 2, tzinfo=timezone.utc),
    )

    assert validated.requested_permissions == "trade"
    assert validated.exchange_permissions == "read"
    assert validated.effective_permissions == "read"
    assert validated.effective_capability == "none"
    assert validated.connection_readiness == "rejected"
    assert validated.connection_readiness_reason == "read_only_not_supported"


@pytest.mark.parametrize(
    (
        "result",
        "expected_status",
        "expected_effective_capability",
        "expected_readiness",
        "expected_reason",
    ),
    [
        (
            ExchangeCredentialValidationResult(
                status="valid_trade_enabled",
                reason="trade_permission_detected",
                ip_restriction_status="restricted",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "trade",
                    "effective_permissions": "trade",
                    "permission_warnings": [],
                },
            ),
            "active",
            "trading",
            "ready_for_trading",
            "trading_policy_ok",
        ),
        (
            ExchangeCredentialValidationResult(
                status="valid_readonly",
                reason="readonly_permission_detected",
                ip_restriction_status="not_restricted_testnet",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "read",
                    "effective_permissions": "read",
                    "permission_warnings": [],
                },
            ),
            "disabled",
            "none",
            "rejected",
            "read_only_not_supported",
        ),
        (
            ExchangeCredentialValidationResult(
                status="invalid_permissions",
                reason="withdraw_or_transfer_enabled",
                ip_restriction_status="restricted",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "withdraw_or_transfer",
                    "effective_permissions": "none",
                    "permission_warnings": [],
                },
            ),
            "disabled",
            "none",
            "rejected",
            "unsafe_permissions",
        ),
        (
            ExchangeCredentialValidationResult(
                status="invalid_credentials",
                reason="exchange_rejected_credentials_400",
                ip_restriction_status="unknown",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "unknown",
                    "effective_permissions": "none",
                    "permission_warnings": [],
                },
            ),
            "disabled",
            "none",
            "rejected",
            "invalid_credentials",
        ),
        (
            ExchangeCredentialValidationResult(
                status="invalid_ip_restriction",
                reason="mainnet_ip_restriction_missing",
                ip_restriction_status="missing_mainnet_restriction",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "trade",
                    "effective_permissions": "none",
                    "permission_warnings": [],
                },
            ),
            "disabled",
            "none",
            "needs_action",
            "ip_restriction_required",
        ),
        (
            ExchangeCredentialValidationResult(
                status="skipped_external_validation",
                reason="live_validation_disabled",
                ip_restriction_status="not_checked",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "unknown",
                    "effective_permissions": "none",
                    "permission_warnings": [],
                },
            ),
            "disabled",
            "none",
            "needs_action",
            "validation_unavailable",
        ),
    ],
)
def test_auto_validation_create_only_keeps_trading_ready_active(
    result: ExchangeCredentialValidationResult,
    expected_status: str,
    expected_effective_capability: str,
    expected_readiness: str,
    expected_reason: str,
) -> None:
    service = _service()

    created = service.create_connection_with_validation(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        exchange_name="bybit",
        market_type="spot",
        environment="mainnet",
        label="auto-validation",
        permissions="trade",
        api_key="ACCOUNTKEY1234",
        api_secret="TEST_SECRET",
        passphrase=None,
        validator=_StaticValidator(result=result),
        now=datetime(2026, 5, 26, 13, 0, tzinfo=timezone.utc),
    )

    assert created.status == expected_status
    assert created.status_reason == (
        None if expected_status == "active" else "auto_validation_failed"
    )
    assert created.effective_capability == expected_effective_capability
    assert created.connection_readiness == expected_readiness
    assert created.connection_readiness_reason == expected_reason


def test_auto_validation_rotate_failure_preserves_active_credential_version() -> None:
    service = _service()
    created = service.create_connection_with_validation(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        exchange_name="bybit",
        market_type="spot",
        environment="mainnet",
        label="auto-validation-rotate",
        permissions="trade",
        api_key="ACCOUNTKEY1234",
        api_secret="TEST_SECRET",
        passphrase=None,
        validator=_StaticValidator(
            result=ExchangeCredentialValidationResult(
                status="valid_trade_enabled",
                reason="trade_permission_detected",
                ip_restriction_status="restricted",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "trade",
                    "effective_permissions": "trade",
                    "permission_warnings": [],
                },
            )
        ),
        now=datetime(2026, 5, 26, 14, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(ExchangeConnectionError) as error:
        service.rotate_connection_with_validation(
            owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
            connection_id=created.connection_id,
            api_key="ROTATEDKEY9876",
            api_secret="TEST_SECRET_ROTATED",
            passphrase=None,
            validator=_StaticValidator(
                result=ExchangeCredentialValidationResult(
                    status="valid_readonly",
                    reason="readonly_permission_detected",
                    ip_restriction_status="not_restricted_testnet",
                    permission_summary={
                        "requested_permissions": "trade",
                        "permissions": "trade",
                        "exchange_permissions": "read",
                        "effective_permissions": "read",
                        "permission_warnings": [],
                    },
                )
            ),
            now=datetime(2026, 5, 26, 14, 1, tzinfo=timezone.utc),
        )

    assert error.value.code == "read_only_not_supported"
    listed = service.list_connections(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123")
    )
    assert listed[0].connection_id == created.connection_id
    assert listed[0].credential_version_id == created.credential_version_id
    assert listed[0].api_key == "****1234"
    assert listed[0].status == "active"
    assert listed[0].connection_readiness == "ready_for_trading"


def test_reclassification_moves_active_readonly_to_history_without_losing_reason() -> None:
    service = _service()
    created = _create_connection(service=service, permissions="trade")
    validated = service.validate_connection(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        connection_id=created.connection_id,
        validator=_StaticValidator(
            result=ExchangeCredentialValidationResult(
                status="valid_readonly",
                reason="readonly_permission_detected",
                ip_restriction_status="not_restricted_testnet",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "read",
                    "effective_permissions": "read",
                    "permission_warnings": [],
                },
            )
        ),
        now=datetime(2026, 5, 26, 15, 0, tzinfo=timezone.utc),
    )

    reclassified = service.reclassify_non_trading_active_connection(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        connection_id=validated.connection_id,
        now=datetime(2026, 5, 26, 15, 1, tzinfo=timezone.utc),
    )

    assert reclassified.status == "disabled"
    assert reclassified.status_reason == RECLASSIFIED_NON_TRADING_STATUS_REASON
    assert reclassified.effective_capability == "none"
    assert reclassified.connection_readiness == "rejected"
    assert reclassified.connection_readiness_reason == "read_only_not_supported"

    repeated = service.reclassify_non_trading_active_connection(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        connection_id=validated.connection_id,
        now=datetime(2026, 5, 26, 15, 2, tzinfo=timezone.utc),
    )

    assert repeated.status == "disabled"
    assert repeated.status_reason == RECLASSIFIED_NON_TRADING_STATUS_REASON


def test_reclassification_refuses_active_trading_ready_connection() -> None:
    service = _service()
    created = service.create_connection_with_validation(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        exchange_name="bybit",
        market_type="spot",
        environment="mainnet",
        label="ready",
        permissions="trade",
        api_key="ACCOUNTKEY1234",
        api_secret="TEST_SECRET",
        passphrase=None,
        validator=_StaticValidator(
            result=ExchangeCredentialValidationResult(
                status="valid_trade_enabled",
                reason="trade_permission_detected",
                ip_restriction_status="restricted",
                permission_summary={
                    "requested_permissions": "trade",
                    "permissions": "trade",
                    "exchange_permissions": "trade",
                    "effective_permissions": "trade",
                    "permission_warnings": [],
                },
            )
        ),
        now=datetime(2026, 5, 26, 16, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(ExchangeConnectionError) as error:
        service.reclassify_non_trading_active_connection(
            owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
            connection_id=created.connection_id,
            now=datetime(2026, 5, 26, 16, 1, tzinfo=timezone.utc),
        )

    assert error.value.code == "exchange_connection_trading_ready"


def _service() -> ExchangeConnectionService:
    return ExchangeConnectionService(
        repository=InMemoryExchangeConnectionRepository(),
        secret_cipher=DeterministicInMemoryExchangeSecretCipher(),
    )


def _create_connection(
    *,
    service: ExchangeConnectionService,
    permissions: str = "read",
):
    return service.create_connection(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000123"),
        exchange_name="binance",
        market_type="spot",
        environment="testnet",
        label="readiness",
        permissions=permissions,
        api_key="ACCOUNTKEY1234",
        api_secret="TEST_SECRET",
        passphrase=None,
        now=datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc),
    )
