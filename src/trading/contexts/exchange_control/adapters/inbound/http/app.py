from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass

from fastapi import FastAPI, Header, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)

from trading.contexts.exchange_control.adapters.outbound import (
    OpenBaoTransitExchangeSecretCipher,
)
from trading.contexts.exchange_control.application.readiness import (
    ExchangeControlReadinessProbe,
)
from trading.contexts.exchange_control.application.secret_cipher import (
    TRANSIT_KEY_NAME,
    DeterministicInMemoryExchangeSecretCipher,
    ExchangeSecretCipher,
)
from trading.contexts.exchange_control.application.service_identity import (
    EXCHANGE_CONTROL_SERVICE_IDENTITY,
    build_exchange_control_service_identity,
)

EXCHANGE_CONTROL_DEFAULT_HOST = "127.0.0.1"
EXCHANGE_CONTROL_METRICS_PORT = 9205
EXCHANGE_CONTROL_INTERNAL_SERVICE = "apps/api"
EXCHANGE_CONTROL_INTERNAL_CONTRACT_VERSION = "internal-v1"
EXCHANGE_CONTROL_INTERNAL_CAPABILITIES = (
    "capabilities.read",
    "exchange_credentials.encrypt",
    "exchange_credentials.decrypt.exchange_control_only",
    "exchange_credentials.fingerprint",
    "exchange_connections.create.stage_4_pending",
    "exchange_connections.rotate.stage_4_pending",
    "exchange_connections.disable.stage_4_pending",
    "exchange_connections.validate.stage_5_pending",
)
SECRET_CIPHER_IN_MEMORY_DEV = "in_memory_dev"
SECRET_CIPHER_OPENBAO_TRANSIT_V1 = "openbao_transit_v1"
SECRET_CIPHER_VAULT_TRANSIT_V1 = "vault_transit_v1"
SUPPORTED_TRANSIT_SECRET_CIPHERS = {
    SECRET_CIPHER_OPENBAO_TRANSIT_V1,
    SECRET_CIPHER_VAULT_TRANSIT_V1,
}


@dataclass(frozen=True)
class ExchangeControlRuntimeConfig:
    service_identity_name: str
    bind_host: str
    metrics_port: int
    real_exchange_validation_enabled: bool = False
    secret_cipher_backend: str = SECRET_CIPHER_IN_MEMORY_DEV
    openbao_addr: str | None = None
    exchange_control_transit_token: str | None = None
    api_transit_token_configured: bool = False
    transit_key_name: str = TRANSIT_KEY_NAME
    internal_api_token: str | None = None

    @classmethod
    def from_environ(
        cls,
        *,
        environ: Mapping[str, str],
        bind_host: str | None = None,
        metrics_port: int | None = None,
    ) -> "ExchangeControlRuntimeConfig":
        environment_name = environ.get("ROEHUB_ENV", "dev").strip().lower()
        service_identity_name = environ.get(
            "ROEHUB_EXCHANGE_CONTROL_SERVICE_IDENTITY",
            EXCHANGE_CONTROL_SERVICE_IDENTITY,
        )
        resolved_host = bind_host or environ.get(
            "ROEHUB_EXCHANGE_CONTROL_BIND_HOST",
            EXCHANGE_CONTROL_DEFAULT_HOST,
        )
        resolved_port = metrics_port or _read_int(
            value=environ.get("ROEHUB_EXCHANGE_CONTROL_METRICS_PORT"),
            default=EXCHANGE_CONTROL_METRICS_PORT,
            name="ROEHUB_EXCHANGE_CONTROL_METRICS_PORT",
        )
        validation_enabled = _read_bool(
            value=environ.get("ROEHUB_EXCHANGE_CONTROL_REAL_EXCHANGE_VALIDATION_ENABLED"),
            default=False,
        )
        secret_cipher_backend = environ.get(
            "ROEHUB_EXCHANGE_CONTROL_SECRET_CIPHER",
            SECRET_CIPHER_IN_MEMORY_DEV,
        ).strip()
        openbao_addr = _read_optional_str(environ.get("OPENBAO_ADDR"))
        exchange_control_transit_token = _read_optional_str(
            environ.get("ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN")
        )
        api_transit_token = _read_optional_str(environ.get("ROEHUB_API_TRANSIT_TOKEN"))
        transit_key_name = environ.get(
            "ROEHUB_EXCHANGE_CONTROL_TRANSIT_KEY",
            TRANSIT_KEY_NAME,
        ).strip()
        internal_api_token = _read_optional_str(
            environ.get("ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN")
        )
        config = cls(
            service_identity_name=service_identity_name,
            bind_host=resolved_host,
            metrics_port=resolved_port,
            real_exchange_validation_enabled=validation_enabled,
            secret_cipher_backend=secret_cipher_backend,
            openbao_addr=openbao_addr,
            exchange_control_transit_token=exchange_control_transit_token,
            api_transit_token_configured=api_transit_token is not None,
            transit_key_name=transit_key_name,
            internal_api_token=internal_api_token,
        )
        config.validate(environment_name=environment_name)
        return config

    def validate(self, *, environment_name: str) -> None:
        build_exchange_control_service_identity(name=self.service_identity_name)
        if self.metrics_port <= 0:
            raise ValueError("exchange-control metrics_port must be > 0")
        if self.real_exchange_validation_enabled:
            raise ValueError("real exchange validation must remain disabled before Stage 5")
        if self.transit_key_name != TRANSIT_KEY_NAME:
            raise ValueError("exchange-control Transit key must be roehub-exchange-credentials")
        if self.secret_cipher_backend not in {
            SECRET_CIPHER_IN_MEMORY_DEV,
            *SUPPORTED_TRANSIT_SECRET_CIPHERS,
        }:
            raise ValueError("unsupported exchange-control secret cipher backend")
        if environment_name == "prod":
            if self.bind_host != EXCHANGE_CONTROL_DEFAULT_HOST:
                raise ValueError("prod exchange-control must bind to 127.0.0.1")
            if self.metrics_port != EXCHANGE_CONTROL_METRICS_PORT:
                raise ValueError("prod exchange-control must use metrics port 9205")
            if self.secret_cipher_backend not in SUPPORTED_TRANSIT_SECRET_CIPHERS:
                raise ValueError(
                    "prod exchange-control requires OpenBao/Vault Transit secret cipher"
                )
            if not self.openbao_addr:
                raise ValueError("prod exchange-control requires OPENBAO_ADDR")
            if not self.exchange_control_transit_token:
                raise ValueError(
                    "prod exchange-control requires ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN"
                )
            if not self.api_transit_token_configured:
                raise ValueError("prod exchange-control requires ROEHUB_API_TRANSIT_TOKEN")
            if not self.internal_api_token:
                raise ValueError(
                    "prod exchange-control requires ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN"
                )

    def build_secret_cipher(self) -> ExchangeSecretCipher:
        if self.secret_cipher_backend == SECRET_CIPHER_IN_MEMORY_DEV:
            return DeterministicInMemoryExchangeSecretCipher(key_name=self.transit_key_name)
        if self.secret_cipher_backend in SUPPORTED_TRANSIT_SECRET_CIPHERS:
            if not self.openbao_addr or not self.exchange_control_transit_token:
                raise ValueError("Transit secret cipher config is incomplete")
            return OpenBaoTransitExchangeSecretCipher(
                address=self.openbao_addr,
                token=self.exchange_control_transit_token,
                key_name=self.transit_key_name,
            )
        raise ValueError("unsupported exchange-control secret cipher backend")


class ExchangeControlMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.active = Gauge(
            "exchange_control_active",
            "Whether exchange-control runtime is active.",
            registry=self.registry,
        )
        self.connection_validation_total = Counter(
            "exchange_connection_validation_total",
            "Exchange connection validation attempts by exchange, result, and reason.",
            ("exchange", "result", "reason"),
            registry=self.registry,
        )
        self.connection_status = Gauge(
            "exchange_connection_status",
            "Current exchange connection status by exchange and status.",
            ("exchange", "status"),
            registry=self.registry,
        )

    def mark_active(self) -> None:
        self.active.set(1)
        self.connection_validation_total.labels(
            exchange="none",
            result="disabled",
            reason="stage_2_no_real_exchange_calls",
        ).inc(0)
        self.connection_status.labels(exchange="none", status="validation_disabled").set(0)


def create_exchange_control_app(*, config: ExchangeControlRuntimeConfig) -> FastAPI:
    service_identity = build_exchange_control_service_identity(name=config.service_identity_name)
    secret_cipher = config.build_secret_cipher()
    readiness_probe = ExchangeControlReadinessProbe(service_identity=service_identity)
    metrics = ExchangeControlMetrics()
    metrics.mark_active()

    app = FastAPI(title="Roehub Exchange Control", version="1.0.0")
    app.state.secret_cipher = secret_cipher

    @app.get("/health/ready")
    def get_readiness() -> dict[str, object]:
        return readiness_probe.check().as_response_payload()

    @app.get("/metrics", include_in_schema=False)
    def get_metrics() -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            media_type=CONTENT_TYPE_LATEST,
        )

    @app.get("/internal/v1/capabilities", include_in_schema=False)
    def get_internal_capabilities(
        request: Request,
        authorization: str | None = Header(default=None),
        x_roehub_internal_service: str | None = Header(
            default=None,
            alias="X-Roehub-Internal-Service",
        ),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    ) -> dict[str, object]:
        _require_local_request(request=request)
        _require_internal_auth(
            authorization=authorization,
            expected_token=config.internal_api_token,
        )
        _require_internal_service_header(value=x_roehub_internal_service)
        request_id = _require_request_id(value=x_request_id)
        return {
            "service": "exchange-control",
            "service_identity": service_identity.name,
            "contract_version": EXCHANGE_CONTROL_INTERNAL_CONTRACT_VERSION,
            "request_id": request_id,
            "capabilities": list(EXCHANGE_CONTROL_INTERNAL_CAPABILITIES),
            "error_model": {
                "shape": "roe_internal_error_v1",
                "secret_safe": True,
            },
            "timeout_policy": {
                "default_timeout_seconds": 2.0,
                "retry_policy": "no_implicit_retry",
                "mutating_commands_require_idempotency_key": True,
            },
        }

    return app


def _require_local_request(*, request: Request) -> None:
    client_host = request.client.host if request.client else ""
    if client_host in {"127.0.0.1", "::1", "testclient"}:
        return
    raise _internal_error(status_code=403, code="internal_local_only")


def _require_internal_auth(*, authorization: str | None, expected_token: str | None) -> None:
    if not expected_token:
        raise _internal_error(status_code=503, code="internal_auth_not_configured")
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise _internal_error(status_code=401, code="internal_auth_required")
    supplied_token = authorization.removeprefix(prefix)
    if not hmac.compare_digest(supplied_token, expected_token):
        raise _internal_error(status_code=403, code="internal_auth_denied")


def _require_internal_service_header(*, value: str | None) -> None:
    if value == EXCHANGE_CONTROL_INTERNAL_SERVICE:
        return
    raise _internal_error(status_code=403, code="internal_service_denied")


def _require_request_id(*, value: str | None) -> str:
    if value is None or value.strip() == "":
        raise _internal_error(status_code=400, code="request_id_required")
    request_id = value.strip()
    if len(request_id) > 128:
        raise _internal_error(status_code=400, code="request_id_invalid")
    return request_id


def _internal_error(*, status_code: int, code: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": code,
                "message": "Internal exchange-control request rejected",
            }
        },
    )


def _read_int(*, value: str | None, default: int, name: str) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _read_bool(*, value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


__all__ = [
    "EXCHANGE_CONTROL_DEFAULT_HOST",
    "EXCHANGE_CONTROL_INTERNAL_CONTRACT_VERSION",
    "EXCHANGE_CONTROL_METRICS_PORT",
    "ExchangeControlRuntimeConfig",
    "create_exchange_control_app",
]
