from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from trading.contexts.exchange_control.adapters.inbound.http.app import (
    EXCHANGE_CONTROL_INTERNAL_CONTRACT_VERSION,
    EXCHANGE_CONTROL_METRICS_PORT,
    ExchangeControlRuntimeConfig,
    create_exchange_control_app,
)
from trading.contexts.exchange_control.adapters.outbound.openbao_transit import (
    OpenBaoTransitExchangeSecretCipher,
)
from trading.contexts.exchange_control.application.secret_cipher import (
    TRANSIT_KEY_NAME,
    DeterministicInMemoryExchangeSecretCipher,
    ExchangeCredentialCiphertext,
    ExchangeCredentialFingerprint,
    ExchangeCredentialSecret,
    ExchangeSecretCipher,
    ExchangeSecretCipherError,
)
from trading.contexts.exchange_control.application.service_identity import (
    EXCHANGE_CONTROL_SERVICE_IDENTITY,
    ExchangeControlServiceIdentity,
)
from trading.contexts.exchange_control.application.validation import (
    ExchangeCredentialValidationRequest,
    ExchangeCredentialValidationResult,
)


def _build_client() -> TestClient:
    config = ExchangeControlRuntimeConfig.from_environ(environ={"ROEHUB_ENV": "dev"})
    return TestClient(create_exchange_control_app(config=config))


class _RuntimeConfigWithValidator(ExchangeControlRuntimeConfig):
    _validator: _SequenceValidator | _StaticValidator | _CapturingValidator
    _secret_cipher: ExchangeSecretCipher | None = None

    @classmethod
    def from_validator(
        cls,
        *,
        validator: "_SequenceValidator | _StaticValidator | _CapturingValidator",
        secret_cipher: ExchangeSecretCipher | None = None,
    ) -> "_RuntimeConfigWithValidator":
        base = ExchangeControlRuntimeConfig.from_environ(
            environ={
                "ROEHUB_ENV": "dev",
                "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
            }
        )
        config = cls(**base.__dict__)
        object.__setattr__(config, "_validator", validator)
        object.__setattr__(config, "_secret_cipher", secret_cipher)
        return config

    def build_credential_validator(
        self,
    ) -> "_SequenceValidator | _StaticValidator | _CapturingValidator":
        return self._validator

    def build_secret_cipher(self) -> ExchangeSecretCipher:
        if self._secret_cipher is not None:
            return self._secret_cipher
        return super().build_secret_cipher()


class _StaticValidator:
    requires_plaintext = False

    def __init__(self, result: ExchangeCredentialValidationResult) -> None:
        self._result = result

    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
        now: datetime,
    ) -> ExchangeCredentialValidationResult:
        _ = request, now
        return self._result


class _SequenceValidator:
    requires_plaintext = False

    def __init__(self, results: tuple[ExchangeCredentialValidationResult, ...]) -> None:
        self._results = list(results)

    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
        now: datetime,
    ) -> ExchangeCredentialValidationResult:
        _ = request, now
        return self._results.pop(0)


class _CapturingValidator:
    requires_plaintext = True

    def __init__(self, result: ExchangeCredentialValidationResult) -> None:
        self._result = result
        self.requests: list[ExchangeCredentialValidationRequest] = []

    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
        now: datetime,
    ) -> ExchangeCredentialValidationResult:
        _ = now
        self.requests.append(request)
        return self._result


class _RoundTripInMemoryExchangeSecretCipher:
    _prefix = "vault:v1:test:"

    def encrypt(self, secret: ExchangeCredentialSecret) -> ExchangeCredentialCiphertext:
        return ExchangeCredentialCiphertext(value=f"{self._prefix}{secret.value}")

    def decrypt(self, ciphertext: ExchangeCredentialCiphertext) -> ExchangeCredentialSecret:
        if not ciphertext.value.startswith(self._prefix):
            raise ExchangeSecretCipherError("test ciphertext is invalid")
        return ExchangeCredentialSecret(value=ciphertext.value.removeprefix(self._prefix))

    def fingerprint(self, secret: ExchangeCredentialSecret) -> ExchangeCredentialFingerprint:
        return DeterministicInMemoryExchangeSecretCipher().fingerprint(secret)


def _internal_headers(request_id: str) -> dict[str, str]:
    return {
        "Authorization": "Bearer internal-token",
        "X-Roehub-Internal-Service": "apps/api",
        "X-Request-Id": request_id,
    }


def _trade_ready_result() -> ExchangeCredentialValidationResult:
    return ExchangeCredentialValidationResult(
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


def _readonly_result() -> ExchangeCredentialValidationResult:
    return ExchangeCredentialValidationResult(
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


def test_service_identity_is_mandatory_exchange_control() -> None:
    identity = ExchangeControlServiceIdentity(name=EXCHANGE_CONTROL_SERVICE_IDENTITY)

    assert identity.name == "exchange-control"

    with pytest.raises(ValueError, match="service identity"):
        ExchangeControlServiceIdentity(name="apps-api")


def test_prod_runtime_requires_localhost_port_9205_and_explicit_validation_flag() -> None:
    environ = {
        "ROEHUB_ENV": "prod",
        "ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER": "openbao_transit_v1",
        "OPENBAO_ADDR": "http://127.0.0.1:8200",
        "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN": "exchange-control-token",
        "ROEHUB_API_TRANSIT_TOKEN": "api-token",
        "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
        "IDENTITY_PG_DSN": "postgresql://roehub:roehub@127.0.0.1:5432/roehub",
    }
    config = ExchangeControlRuntimeConfig.from_environ(environ=environ)

    assert config.service_identity_name == "exchange-control"
    assert config.bind_host == "127.0.0.1"
    assert config.metrics_port == EXCHANGE_CONTROL_METRICS_PORT
    assert not config.real_exchange_validation_enabled
    assert not config.exchange_validation_live_enabled
    assert config.secret_cipher_backend == "openbao_transit_v1"
    assert config.transit_key_name == TRANSIT_KEY_NAME

    with pytest.raises(ValueError, match="port 9205"):
        ExchangeControlRuntimeConfig.from_environ(
            environ=environ,
            metrics_port=9206,
        )

    with pytest.raises(ValueError, match="ROEHUB_EXCHANGE_VALIDATION_LIVE"):
        ExchangeControlRuntimeConfig.from_environ(
            environ={
                "ROEHUB_ENV": "prod",
                "ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED": "true",
            }
        )

    live_config = ExchangeControlRuntimeConfig.from_environ(
        environ={**environ, "ROEHUB_EXCHANGE_VALIDATION_LIVE": "1"}
    )
    assert live_config.exchange_validation_live_enabled


def test_prod_runtime_fails_closed_without_transit_config() -> None:
    with pytest.raises(ValueError, match="requires OpenBao/Vault Transit"):
        ExchangeControlRuntimeConfig.from_environ(environ={"ROEHUB_ENV": "prod"})

    base_environ = {
        "ROEHUB_ENV": "prod",
        "ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER": "openbao_transit_v1",
        "OPENBAO_ADDR": "http://127.0.0.1:8200",
        "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN": "exchange-control-token",
        "ROEHUB_API_TRANSIT_TOKEN": "api-token",
        "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
        "IDENTITY_PG_DSN": "postgresql://roehub:roehub@127.0.0.1:5432/roehub",
    }
    for missing_name, expected in (
        ("OPENBAO_ADDR", "OPENBAO_ADDR"),
        ("ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN", "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN"),
        ("ROEHUB_API_TRANSIT_TOKEN", "ROEHUB_API_TRANSIT_TOKEN"),
        (
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN",
        ),
        ("IDENTITY_PG_DSN", "IDENTITY_PG_DSN"),
    ):
        environ = dict(base_environ)
        del environ[missing_name]
        with pytest.raises(ValueError, match=expected):
            ExchangeControlRuntimeConfig.from_environ(environ=environ)

    with pytest.raises(ValueError, match="roehub-exchange-credentials"):
        ExchangeControlRuntimeConfig.from_environ(
            environ={
                **base_environ,
                "ROEHUB_EXCHANGE_CONTROL_TRANSIT_KEY": "wrong-key",
            }
        )


def test_health_ready_exposes_service_identity_and_disabled_external_validation() -> None:
    client = _build_client()

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "service": "exchange-control",
        "service_identity": "exchange-control",
        "checks": [
            {"name": "service_identity", "status": "ready"},
            {"name": "external_exchange_validation", "status": "ready"},
        ],
    }


def test_health_ready_fails_closed_when_transit_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_transit_error(
        self: OpenBaoTransitExchangeSecretCipher,
        secret: ExchangeCredentialSecret,
    ) -> ExchangeCredentialFingerprint:
        raise ExchangeSecretCipherError("transit request failed with status 503")

    monkeypatch.setattr(
        OpenBaoTransitExchangeSecretCipher,
        "fingerprint",
        raise_transit_error,
    )
    config = ExchangeControlRuntimeConfig.from_environ(
        environ={
            "ROEHUB_ENV": "prod",
            "ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER": "openbao_transit_v1",
            "OPENBAO_ADDR": "http://127.0.0.1:8200",
            "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN": "exchange-control-token",
            "ROEHUB_API_TRANSIT_TOKEN": "api-token",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
            "IDENTITY_PG_DSN": "postgresql://roehub:roehub@127.0.0.1:5432/roehub",
        }
    )
    client = TestClient(create_exchange_control_app(config=config))

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "service": "exchange-control",
        "service_identity": "exchange-control",
        "checks": [
            {"name": "service_identity", "status": "ready"},
            {"name": "external_exchange_validation", "status": "ready"},
            {"name": "secret_cipher_transit", "status": "not_ready"},
        ],
    }


def test_metrics_expose_secret_safe_exchange_control_series() -> None:
    client = _build_client()

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "exchange_control_active 1.0" in response.text
    assert "exchange_connection_validation_total" in response.text
    assert "exchange_connection_archive_total" in response.text
    assert "exchange_connection_cleanup_total" in response.text
    assert "exchange_connection_trading_readiness_total" in response.text
    assert 'exchange="none"' in response.text
    assert "api_key" not in response.text
    assert "connection_id" not in response.text


def test_internal_capabilities_require_service_auth_and_headers() -> None:
    config = _RuntimeConfigWithValidator.from_validator(
        validator=_SequenceValidator(
            results=(
                _trade_ready_result(),
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
            )
        )
    )
    client = TestClient(create_exchange_control_app(config=config))

    missing_auth = client.get(
        "/internal/v1/capabilities",
        headers={
            "X-Roehub-Internal-Service": "apps/api",
            "X-Request-Id": "stage-3c-test",
        },
    )
    invalid_auth = client.get(
        "/internal/v1/capabilities",
        headers={
            "Authorization": "Bearer wrong-token",
            "X-Roehub-Internal-Service": "apps/api",
            "X-Request-Id": "stage-3c-test",
        },
    )
    missing_service = client.get(
        "/internal/v1/capabilities",
        headers={
            "Authorization": "Bearer internal-token",
            "X-Request-Id": "stage-3c-test",
        },
    )
    missing_request_id = client.get(
        "/internal/v1/capabilities",
        headers={
            "Authorization": "Bearer internal-token",
            "X-Roehub-Internal-Service": "apps/api",
        },
    )

    assert missing_auth.status_code == 401
    assert missing_auth.json()["detail"]["error"]["code"] == "internal_auth_required"
    assert invalid_auth.status_code == 403
    assert invalid_auth.json()["detail"]["error"]["code"] == "internal_auth_denied"
    assert missing_service.status_code == 403
    assert missing_service.json()["detail"]["error"]["code"] == "internal_service_denied"
    assert missing_request_id.status_code == 400
    assert missing_request_id.json()["detail"]["error"]["code"] == "request_id_required"


def test_internal_capabilities_are_secret_safe() -> None:
    config = ExchangeControlRuntimeConfig.from_environ(
        environ={
            "ROEHUB_ENV": "dev",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
        }
    )
    client = TestClient(create_exchange_control_app(config=config))

    response = client.get(
        "/internal/v1/capabilities",
        headers={
            "Authorization": "Bearer internal-token",
            "X-Roehub-Internal-Service": "apps/api",
            "X-Request-Id": "stage-3c-test",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "exchange-control"
    assert payload["service_identity"] == "exchange-control"
    assert payload["contract_version"] == EXCHANGE_CONTROL_INTERNAL_CONTRACT_VERSION
    assert payload["request_id"] == "stage-3c-test"
    assert "capabilities.read" in payload["capabilities"]
    assert "exchange_connections.create_from_existing" in payload["capabilities"]
    assert "exchange_connections.archive" in payload["capabilities"]
    assert payload["timeout_policy"]["retry_policy"] == "no_implicit_retry"
    assert "internal-token" not in response.text
    assert "api_secret" not in response.text
    assert "passphrase" not in response.text


def test_internal_exchange_connection_create_rotate_disable_flow_is_secret_safe() -> None:
    config = _RuntimeConfigWithValidator.from_validator(
        validator=_StaticValidator(result=_trade_ready_result())
    )
    client = TestClient(create_exchange_control_app(config=config))
    headers = {
        "Authorization": "Bearer internal-token",
        "X-Roehub-Internal-Service": "apps/api",
        "X-Request-Id": "stage-4-test",
    }
    owner_user_id = "00000000-0000-0000-0000-000000000401"

    created = client.post(
        "/internal/v1/exchange-connections",
        headers=headers,
        json={
            "owner_user_id": owner_user_id,
            "exchange_name": "binance",
            "market_type": "spot",
            "environment": "testnet",
            "label": "readonly",
            "permissions": "trade",
            "api_key": "STAGE4KEY1234",
            "api_secret": "TEST_SECRET_STAGE4",
            "passphrase": "TEST_PASSPHRASE_STAGE4",
        },
    )

    assert created.status_code == 200
    created_payload = created.json()
    connection_id = created_payload["connection_id"]
    first_version_id = created_payload["credential_version_id"]
    assert created_payload["api_key"] == "****1234"
    assert created_payload["permissions"] == "trade"
    assert created_payload["requested_permissions"] == "trade"
    assert created_payload["exchange_permissions"] == "trade"
    assert created_payload["effective_permissions"] == "trade"
    assert created_payload["requested_capability"] == "trading"
    assert created_payload["effective_capability"] == "trading"
    assert created_payload["connection_readiness"] == "ready_for_trading"
    assert created_payload["connection_readiness_reason"] == "trading_policy_ok"
    assert created_payload["permissions_deprecated"] is True
    assert created_payload["permission_warnings"] == []
    assert created_payload["validation_status"] == "valid_trade_enabled"
    assert "TEST_SECRET_STAGE4" not in created.text
    assert "TEST_PASSPHRASE_STAGE4" not in created.text
    assert "vault:v1:" not in created.text

    listed = client.get(
        "/internal/v1/exchange-connections",
        headers=headers,
        params={"owner_user_id": owner_user_id},
    )
    assert listed.status_code == 200
    assert listed.json()["items"][0]["connection_id"] == connection_id

    rotated = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/rotate",
        headers=headers,
        json={
            "owner_user_id": owner_user_id,
            "api_key": "ROTATEDKEY9876",
            "api_secret": "TEST_SECRET_ROTATED",
        },
    )
    assert rotated.status_code == 200
    rotated_payload = rotated.json()
    assert rotated_payload["connection_id"] == connection_id
    assert rotated_payload["credential_version_id"] != first_version_id
    assert rotated_payload["api_key"] == "****9876"
    assert "TEST_SECRET_ROTATED" not in rotated.text

    disabled = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/disable",
        headers=headers,
        json={"owner_user_id": owner_user_id},
    )
    assert disabled.status_code == 200
    assert disabled.json()["connection_id"] == connection_id
    assert disabled.json()["status"] == "disabled"
    assert disabled.json()["disabled_at"] is not None
    assert disabled.json()["archived_at"] is None

    archived = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/archive",
        headers=headers,
        json={"owner_user_id": owner_user_id, "cleanup_source": "stage09d"},
    )
    assert archived.status_code == 200
    archived_payload = archived.json()
    assert archived_payload["connection_id"] == connection_id
    assert archived_payload["credential_version_id"] == rotated_payload["credential_version_id"]
    assert archived_payload["status"] == "archived"
    assert archived_payload["status_reason"] == "user_archived"
    assert archived_payload["disabled_at"] is not None
    assert archived_payload["archived_at"] is not None
    assert "TEST_SECRET_ROTATED" not in archived.text

    archive_again = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/archive",
        headers=headers,
        json={"owner_user_id": owner_user_id},
    )
    assert archive_again.status_code == 200
    assert archive_again.json()["status"] == "archived"

    rotate_archived = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/rotate",
        headers=headers,
        json={
            "owner_user_id": owner_user_id,
            "api_key": "AFTERARCHIVE1234",
            "api_secret": "TEST_SECRET_AFTER_ARCHIVE",
        },
    )
    assert rotate_archived.status_code == 404
    assert (
        rotate_archived.json()["detail"]["error"]["code"]
        == "exchange_connection_not_found"
    )

    validate_archived = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/validate",
        headers=headers,
        json={"owner_user_id": owner_user_id},
    )
    assert validate_archived.status_code == 404
    assert (
        validate_archived.json()["detail"]["error"]["code"]
        == "exchange_connection_not_found"
    )

    metrics = client.get("/metrics")
    assert "exchange_connection_archive_total" in metrics.text
    assert 'result="archived"' in metrics.text
    assert "exchange_connection_cleanup_total" in metrics.text
    assert 'result="archived",source="stage09d"' in metrics.text
    assert "connection_id" not in metrics.text


def test_internal_exchange_connection_create_market_from_existing_is_secret_safe() -> None:
    validator = _CapturingValidator(result=_trade_ready_result())
    config = _RuntimeConfigWithValidator.from_validator(
        validator=validator,
        secret_cipher=_RoundTripInMemoryExchangeSecretCipher(),
    )
    client = TestClient(create_exchange_control_app(config=config))
    headers = _internal_headers("stage-12-create-market-test")
    owner_user_id = "00000000-0000-0000-0000-000000000412"

    source = client.post(
        "/internal/v1/exchange-connections",
        headers=headers,
        json={
            "owner_user_id": owner_user_id,
            "exchange_name": "bybit",
            "market_type": "spot",
            "environment": "testnet",
            "label": "bybit_testnet",
            "permissions": "trade",
            "api_key": "BYBIT_SOURCE_KEY_AUN5",
            "api_secret": "BYBIT_SOURCE_SECRET",
        },
    )
    assert source.status_code == 200
    source_id = source.json()["connection_id"]

    created = client.post(
        f"/internal/v1/exchange-connections/{source_id}/markets",
        headers=headers,
        json={
            "owner_user_id": owner_user_id,
            "market_type": "futures",
            "label": "bybit_testnet",
        },
    )

    assert created.status_code == 200
    payload = created.json()
    assert payload["exchange_name"] == "bybit"
    assert payload["market_type"] == "futures"
    assert payload["environment"] == "testnet"
    assert payload["api_key"] == "****AUN5"
    assert payload["connection_readiness"] == "ready_for_trading"
    assert "BYBIT_SOURCE_KEY_AUN5" not in created.text
    assert "BYBIT_SOURCE_SECRET" not in created.text
    assert [request.market_type for request in validator.requests] == ["spot", "futures"]
    assert validator.requests[1].credential.api_key == "BYBIT_SOURCE_KEY_AUN5"
    assert validator.requests[1].credential.api_secret == "BYBIT_SOURCE_SECRET"

    listed = client.get(
        "/internal/v1/exchange-connections",
        headers=headers,
        params={"owner_user_id": owner_user_id},
    )
    assert listed.status_code == 200
    assert [item["market_type"] for item in listed.json()["items"]] == [
        "spot",
        "futures",
    ]


def test_internal_validate_reclassifies_readonly_connection_and_allows_recreate() -> None:
    config = _RuntimeConfigWithValidator.from_validator(
        validator=_SequenceValidator(
            results=(
                _trade_ready_result(),
                _readonly_result(),
                _trade_ready_result(),
            )
        )
    )
    client = TestClient(create_exchange_control_app(config=config))
    headers = {
        "Authorization": "Bearer internal-token",
        "X-Roehub-Internal-Service": "apps/api",
        "X-Request-Id": "readonly-recheck-regression",
    }
    owner_user_id = "00000000-0000-0000-0000-000000000402"
    create_payload = {
        "owner_user_id": owner_user_id,
        "exchange_name": "bybit",
        "market_type": "spot",
        "environment": "mainnet",
        "label": "bybit-recheck",
        "permissions": "trade",
        "api_key": "BYBITRECHECK1234",
        "api_secret": "TEST_SECRET_BYBIT_RECHECK",
    }

    created = client.post(
        "/internal/v1/exchange-connections",
        headers=headers,
        json=create_payload,
    )
    assert created.status_code == 200
    connection_id = created.json()["connection_id"]

    rechecked = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/validate",
        headers=headers,
        json={"owner_user_id": owner_user_id},
    )

    assert rechecked.status_code == 200
    rechecked_payload = rechecked.json()
    assert rechecked_payload["connection_id"] == connection_id
    assert rechecked_payload["status"] == "disabled"
    assert rechecked_payload["status_reason"] == "reclassified_non_trading_ready"
    assert rechecked_payload["connection_readiness"] == "rejected"
    assert rechecked_payload["connection_readiness_reason"] == "read_only_not_supported"

    recreated = client.post(
        "/internal/v1/exchange-connections",
        headers=headers,
        json={**create_payload, "label": "bybit-recreated"},
    )
    assert recreated.status_code == 200
    recreated_payload = recreated.json()
    assert recreated_payload["connection_id"] != connection_id
    assert recreated_payload["status"] == "active"
    assert recreated_payload["connection_readiness"] == "ready_for_trading"


def test_internal_account_state_read_requires_trading_ready_connection() -> None:
    config = _RuntimeConfigWithValidator.from_validator(
        validator=_StaticValidator(_readonly_result())
    )
    client = TestClient(create_exchange_control_app(config=config))
    created = client.post(
        "/internal/v1/exchange-connections",
        headers=_internal_headers("account-state-readonly"),
        json={
            "owner_user_id": "00000000-0000-0000-0000-000000000123",
            "exchange_name": "bybit",
            "market_type": "spot",
            "environment": "testnet",
            "label": "readonly",
            "permissions": "trade",
            "api_key": "readonly-key",
            "api_secret": "readonly-secret",
        },
    )
    assert created.status_code == 200
    connection_id = created.json()["connection_id"]

    response = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/account-state",
        headers=_internal_headers("account-state-readonly-sync"),
        json={
            "owner_user_id": "00000000-0000-0000-0000-000000000123",
            "instrument_keys": ["bybit:spot:BTCUSDT"],
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error"]["code"] == "exchange_connection_not_found"


def test_internal_account_state_read_returns_secret_safe_sanitized_projection() -> None:
    config = _RuntimeConfigWithValidator.from_validator(
        validator=_StaticValidator(_trade_ready_result())
    )
    client = TestClient(create_exchange_control_app(config=config))
    created = client.post(
        "/internal/v1/exchange-connections",
        headers=_internal_headers("account-state-create"),
        json={
            "owner_user_id": "00000000-0000-0000-0000-000000000124",
            "exchange_name": "bybit",
            "market_type": "spot",
            "environment": "testnet",
            "label": "trade-ready",
            "permissions": "trade",
            "api_key": "trade-key",
            "api_secret": "trade-secret",
        },
    )
    assert created.status_code == 200
    connection_id = created.json()["connection_id"]

    response = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/account-state",
        headers=_internal_headers("account-state-sync"),
        json={
            "owner_user_id": "00000000-0000-0000-0000-000000000124",
            "instrument_keys": ["bybit:spot:BTCUSDT"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sync_status"] == "degraded"
    assert payload["sync_reason"] == "account_state_sync_disabled"
    assert len(payload["source_hash"]) == 64
    dumped = str(payload).lower()
    assert "trade-key" not in dumped
    assert "trade-secret" not in dumped
    assert "authorization" not in dumped


def test_internal_account_state_read_reports_legacy_ciphertext_as_unavailable() -> None:
    config = _RuntimeConfigWithValidator.from_validator(
        validator=_StaticValidator(_trade_ready_result())
    )
    object.__setattr__(config, "account_state_sync_enabled", True)
    app = create_exchange_control_app(config=config)
    client = TestClient(app)
    owner_user_id = "00000000-0000-0000-0000-000000000125"
    created = client.post(
        "/internal/v1/exchange-connections",
        headers=_internal_headers("account-state-legacy-cipher-create"),
        json={
            "owner_user_id": owner_user_id,
            "exchange_name": "binance",
            "market_type": "futures",
            "environment": "testnet",
            "label": "legacy-cipher",
            "permissions": "trade",
            "api_key": "trade-key",
            "api_secret": "trade-secret",
        },
    )
    assert created.status_code == 200
    connection_id = UUID(created.json()["connection_id"])

    repository = app.state.exchange_connection_service._repository
    credential = repository.get_active_credential(connection_id=connection_id)
    assert credential is not None
    repository._credential_versions[credential.credential_version_id] = replace(
        credential,
        api_key_ciphertext="legacy-key-ciphertext",
        api_secret_ciphertext="legacy-secret-ciphertext",
    )

    response = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/account-state",
        headers=_internal_headers("account-state-legacy-cipher-sync"),
        json={
            "owner_user_id": owner_user_id,
            "instrument_keys": ["binance:futures:BTCUSDT"],
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["error"]["code"] == (
        "exchange_connection_account_state_unavailable"
    )
    dumped = str(response.json()).lower()
    assert "trade-key" not in dumped
    assert "trade-secret" not in dumped
    assert "legacy-key-ciphertext" not in dumped


def test_internal_exchange_connection_auto_validation_unavailable_is_not_active() -> None:
    config = ExchangeControlRuntimeConfig.from_environ(
        environ={
            "ROEHUB_ENV": "dev",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
        }
    )
    client = TestClient(create_exchange_control_app(config=config))
    headers = {
        "Authorization": "Bearer internal-token",
        "X-Roehub-Internal-Service": "apps/api",
        "X-Request-Id": "stage-10b-unavailable-test",
    }

    created = client.post(
        "/internal/v1/exchange-connections",
        headers=headers,
        json={
            "owner_user_id": "00000000-0000-0000-0000-000000001010",
            "exchange_name": "bybit",
            "market_type": "spot",
            "environment": "mainnet",
            "label": "validation-unavailable",
            "permissions": "trade",
            "api_key": "STAGE10BKEY1234",
            "api_secret": "TEST_SECRET_STAGE10B",
        },
    )

    assert created.status_code == 200
    payload = created.json()
    assert payload["status"] == "disabled"
    assert payload["status_reason"] == "auto_validation_failed"
    assert payload["effective_capability"] == "none"
    assert payload["connection_readiness"] == "needs_action"
    assert payload["connection_readiness_reason"] == "validation_unavailable"
    assert "TEST_SECRET_STAGE10B" not in created.text

    metrics = client.get("/metrics")
    assert "exchange_connection_auto_validation_total" in metrics.text
    assert 'reason="validation_unavailable"' in metrics.text


def test_internal_exchange_connection_archive_rejects_active_connection() -> None:
    config = _RuntimeConfigWithValidator.from_validator(
        validator=_StaticValidator(result=_trade_ready_result())
    )
    client = TestClient(create_exchange_control_app(config=config))
    headers = {
        "Authorization": "Bearer internal-token",
        "X-Roehub-Internal-Service": "apps/api",
        "X-Request-Id": "stage-09a-test",
    }
    owner_user_id = "00000000-0000-0000-0000-000000000901"

    created = client.post(
        "/internal/v1/exchange-connections",
        headers=headers,
        json={
            "owner_user_id": owner_user_id,
            "exchange_name": "bybit",
            "market_type": "spot",
            "environment": "testnet",
            "label": "active",
            "permissions": "trade",
            "api_key": "STAGE09AKEY1234",
            "api_secret": "TEST_SECRET_STAGE09A",
        },
    )
    assert created.status_code == 200

    archived = client.post(
        f"/internal/v1/exchange-connections/{created.json()['connection_id']}/archive",
        headers=headers,
        json={"owner_user_id": owner_user_id},
    )

    assert archived.status_code == 409
    assert (
        archived.json()["detail"]["error"]["code"]
        == "exchange_connection_not_disabled"
    )
    assert "TEST_SECRET_STAGE09A" not in archived.text


def test_internal_exchange_connection_validate_skips_live_calls_by_default() -> None:
    config = _RuntimeConfigWithValidator.from_validator(
        validator=_SequenceValidator(
            results=(
                _trade_ready_result(),
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
            )
        )
    )
    client = TestClient(create_exchange_control_app(config=config))
    headers = {
        "Authorization": "Bearer internal-token",
        "X-Roehub-Internal-Service": "apps/api",
        "X-Request-Id": "stage-5-test",
    }
    owner_user_id = "00000000-0000-0000-0000-000000000501"

    created = client.post(
        "/internal/v1/exchange-connections",
        headers=headers,
        json={
            "owner_user_id": owner_user_id,
            "exchange_name": "binance",
            "market_type": "spot",
            "environment": "testnet",
            "label": "readonly",
            "permissions": "trade",
            "api_key": "STAGE5KEY1234",
            "api_secret": "TEST_SECRET_STAGE5",
        },
    )
    assert created.status_code == 200
    connection_id = created.json()["connection_id"]

    validated = client.post(
        f"/internal/v1/exchange-connections/{connection_id}/validate",
        headers=headers,
        json={"owner_user_id": owner_user_id},
    )

    assert validated.status_code == 200
    payload = validated.json()
    assert payload["connection_id"] == connection_id
    assert payload["validation_status"] == "skipped_external_validation"
    assert payload["validation_reason"] == "live_validation_disabled"
    assert payload["ip_restriction_status"] == "not_checked"
    assert payload["requested_permissions"] == "trade"
    assert payload["exchange_permissions"] == "unknown"
    assert payload["effective_permissions"] == "none"
    assert payload["requested_capability"] == "trading"
    assert payload["effective_capability"] == "none"
    assert payload["connection_readiness"] == "needs_action"
    assert payload["connection_readiness_reason"] == "validation_required"
    assert payload["permissions_deprecated"] is True
    assert payload["permission_warnings"] == []
    assert "TEST_SECRET_STAGE5" not in validated.text

    metrics = client.get("/metrics")
    assert 'result="skipped_external_validation"' in metrics.text
    assert "exchange_connection_trading_readiness_total" in metrics.text
    assert "exchange_connection_reclassification_total" in metrics.text
    assert 'reason="validation_required"' in metrics.text
    assert "exchange_permission_mismatch_total" in metrics.text
    assert "connection_id" not in metrics.text


def test_internal_exchange_connection_rejects_linear_market_type() -> None:
    config = ExchangeControlRuntimeConfig.from_environ(
        environ={
            "ROEHUB_ENV": "dev",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
        }
    )
    client = TestClient(create_exchange_control_app(config=config))

    response = client.post(
        "/internal/v1/exchange-connections",
        headers={
            "Authorization": "Bearer internal-token",
            "X-Roehub-Internal-Service": "apps/api",
            "X-Request-Id": "stage-4-test",
        },
        json={
            "owner_user_id": "00000000-0000-0000-0000-000000000401",
            "exchange_name": "bybit",
            "market_type": "linear",
            "environment": "testnet",
            "permissions": "read",
            "api_key": "STAGE4KEY1234",
            "api_secret": "TEST_SECRET_STAGE4",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["error"]["code"] == "exchange_connection_invalid"


def test_secret_value_objects_redact_repr() -> None:
    secret = ExchangeCredentialSecret(value="TEST_SECRET")
    ciphertext = ExchangeCredentialCiphertext(value="vault:v1:test-ciphertext")
    fingerprint = DeterministicInMemoryExchangeSecretCipher().fingerprint(secret)

    assert "TEST_SECRET" not in repr(secret)
    assert "test-ciphertext" not in repr(ciphertext)
    assert fingerprint.value not in repr(fingerprint)


def test_deterministic_test_cipher_encrypts_and_fingerprints_without_decrypt_path() -> None:
    cipher = DeterministicInMemoryExchangeSecretCipher()
    secret = ExchangeCredentialSecret(value="TEST_SECRET")

    first = cipher.encrypt(secret)
    second = cipher.encrypt(secret)
    fingerprint = cipher.fingerprint(secret)

    assert first == second
    assert first.value.startswith("vault:v1:deterministic:")
    assert fingerprint.value.startswith("hmac-sha256:")
    with pytest.raises(ExchangeSecretCipherError, match="decrypt is unavailable"):
        cipher.decrypt(first)


def test_openbao_transit_adapter_sanitizes_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    import urllib.request
    from email.message import Message

    def raise_http_error(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> object:
        raise urllib.error.HTTPError(
            url=request.full_url,
            code=403,
            msg="permission denied: TEST_SECRET",
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)
    cipher = OpenBaoTransitExchangeSecretCipher(
        address="http://127.0.0.1:8200",
        token="exchange-control-token",
    )

    with pytest.raises(ExchangeSecretCipherError) as exc_info:
        cipher.encrypt(ExchangeCredentialSecret(value="TEST_SECRET"))

    message = str(exc_info.value)
    assert "403" in message
    assert "TEST_SECRET" not in message
    assert "permission denied" not in message
