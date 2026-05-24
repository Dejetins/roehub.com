from __future__ import annotations

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
    ExchangeCredentialSecret,
    ExchangeSecretCipherError,
)
from trading.contexts.exchange_control.application.service_identity import (
    EXCHANGE_CONTROL_SERVICE_IDENTITY,
    ExchangeControlServiceIdentity,
)


def _build_client() -> TestClient:
    config = ExchangeControlRuntimeConfig.from_environ(environ={"ROEHUB_ENV": "dev"})
    return TestClient(create_exchange_control_app(config=config))


def test_service_identity_is_mandatory_exchange_control() -> None:
    identity = ExchangeControlServiceIdentity(name=EXCHANGE_CONTROL_SERVICE_IDENTITY)

    assert identity.name == "exchange-control"

    with pytest.raises(ValueError, match="service identity"):
        ExchangeControlServiceIdentity(name="apps-api")


def test_prod_runtime_requires_localhost_port_9205_and_disabled_validation() -> None:
    environ = {
        "ROEHUB_ENV": "prod",
        "ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER": "openbao_transit_v1",
        "OPENBAO_ADDR": "http://127.0.0.1:8200",
        "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN": "exchange-control-token",
        "ROEHUB_API_TRANSIT_TOKEN": "api-token",
        "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
    }
    config = ExchangeControlRuntimeConfig.from_environ(environ=environ)

    assert config.service_identity_name == "exchange-control"
    assert config.bind_host == "127.0.0.1"
    assert config.metrics_port == EXCHANGE_CONTROL_METRICS_PORT
    assert not config.real_exchange_validation_enabled
    assert config.secret_cipher_backend == "openbao_transit_v1"
    assert config.transit_key_name == TRANSIT_KEY_NAME

    with pytest.raises(ValueError, match="port 9205"):
        ExchangeControlRuntimeConfig.from_environ(
            environ=environ,
            metrics_port=9206,
        )

    with pytest.raises(ValueError, match="real exchange validation"):
        ExchangeControlRuntimeConfig.from_environ(
            environ={
                "ROEHUB_ENV": "prod",
                "ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED": "true",
            }
        )


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
    }
    for missing_name, expected in (
        ("OPENBAO_ADDR", "OPENBAO_ADDR"),
        ("ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN", "ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN"),
        ("ROEHUB_API_TRANSIT_TOKEN", "ROEHUB_API_TRANSIT_TOKEN"),
        (
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN",
        ),
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


def test_metrics_expose_secret_safe_exchange_control_series() -> None:
    client = _build_client()

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "exchange_control_active 1.0" in response.text
    assert "exchange_connection_validation_total" in response.text
    assert 'exchange="none"' in response.text
    assert "api_key" not in response.text
    assert "connection_id" not in response.text


def test_internal_capabilities_require_service_auth_and_headers() -> None:
    config = ExchangeControlRuntimeConfig.from_environ(
        environ={
            "ROEHUB_ENV": "dev",
            "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN": "internal-token",
        }
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
    assert payload["timeout_policy"]["retry_policy"] == "no_implicit_retry"
    assert "internal-token" not in response.text
    assert "api_secret" not in response.text
    assert "passphrase" not in response.text


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
