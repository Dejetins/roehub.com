from __future__ import annotations

import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from fastapi import FastAPI, Header, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from trading.contexts.exchange_control.adapters.outbound import (
    OpenBaoTransitExchangeSecretCipher,
    PostgresExchangeConnectionRepository,
)
from trading.contexts.exchange_control.application.connections import (
    ExchangeConnectionError,
    ExchangeConnectionRepository,
    ExchangeConnectionService,
    ExchangeConnectionView,
    InMemoryExchangeConnectionRepository,
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
from trading.shared_kernel.primitives import UserId

EXCHANGE_CONTROL_DEFAULT_HOST = "127.0.0.1"
EXCHANGE_CONTROL_METRICS_PORT = 9205
EXCHANGE_CONTROL_INTERNAL_SERVICE = "apps/api"
EXCHANGE_CONTROL_INTERNAL_CONTRACT_VERSION = "internal-v1"
EXCHANGE_CONTROL_INTERNAL_CAPABILITIES = (
    "capabilities.read",
    "exchange_credentials.encrypt",
    "exchange_credentials.decrypt.exchange_control_only",
    "exchange_credentials.fingerprint",
    "exchange_connections.create",
    "exchange_connections.list",
    "exchange_connections.rotate",
    "exchange_connections.disable",
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
    identity_postgres_dsn: str | None = None

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
        identity_postgres_dsn = _read_optional_str(environ.get("IDENTITY_PG_DSN"))
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
            identity_postgres_dsn=identity_postgres_dsn,
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
            if not self.identity_postgres_dsn:
                raise ValueError("prod exchange-control requires IDENTITY_PG_DSN")

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

    def build_connection_repository(self) -> ExchangeConnectionRepository:
        if self.identity_postgres_dsn:
            return PostgresExchangeConnectionRepository(dsn=self.identity_postgres_dsn)
        return InMemoryExchangeConnectionRepository()


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


class CreateExchangeConnectionInternalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_user_id: str
    exchange_name: str
    market_type: str
    environment: str = "mainnet"
    label: str | None = Field(default=None, max_length=80)
    permissions: str = "read"
    api_key: str
    api_secret: str
    passphrase: str | None = None


class RotateExchangeConnectionInternalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_user_id: str
    api_key: str
    api_secret: str
    passphrase: str | None = None


class DisableExchangeConnectionInternalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner_user_id: str


def create_exchange_control_app(*, config: ExchangeControlRuntimeConfig) -> FastAPI:
    service_identity = build_exchange_control_service_identity(name=config.service_identity_name)
    secret_cipher = config.build_secret_cipher()
    connection_service = ExchangeConnectionService(
        repository=config.build_connection_repository(),
        secret_cipher=secret_cipher,
    )
    readiness_probe = ExchangeControlReadinessProbe(service_identity=service_identity)
    metrics = ExchangeControlMetrics()
    metrics.mark_active()

    app = FastAPI(title="Roehub Exchange Control", version="1.0.0")
    app.state.secret_cipher = secret_cipher
    app.state.exchange_connection_service = connection_service

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

    @app.get("/internal/v1/exchange-connections", include_in_schema=False)
    def list_internal_exchange_connections(
        request: Request,
        owner_user_id: str,
        authorization: str | None = Header(default=None),
        x_roehub_internal_service: str | None = Header(
            default=None,
            alias="X-Roehub-Internal-Service",
        ),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    ) -> dict[str, object]:
        _require_internal_request(
            request=request,
            authorization=authorization,
            expected_token=config.internal_api_token,
            internal_service=x_roehub_internal_service,
            request_id=x_request_id,
        )
        owner = _parse_user_id(raw_value=owner_user_id)
        return {
            "items": [
                _exchange_connection_response(view=view)
                for view in connection_service.list_connections(owner_user_id=owner)
            ]
        }

    @app.post("/internal/v1/exchange-connections", include_in_schema=False)
    def create_internal_exchange_connection(
        payload: CreateExchangeConnectionInternalRequest,
        request: Request,
        authorization: str | None = Header(default=None),
        x_roehub_internal_service: str | None = Header(
            default=None,
            alias="X-Roehub-Internal-Service",
        ),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    ) -> dict[str, object]:
        _require_internal_request(
            request=request,
            authorization=authorization,
            expected_token=config.internal_api_token,
            internal_service=x_roehub_internal_service,
            request_id=x_request_id,
        )
        try:
            view = connection_service.create_connection(
                owner_user_id=_parse_user_id(raw_value=payload.owner_user_id),
                exchange_name=payload.exchange_name,
                market_type=payload.market_type,
                environment=payload.environment,
                label=payload.label,
                permissions=payload.permissions,
                api_key=payload.api_key,
                api_secret=payload.api_secret,
                passphrase=payload.passphrase,
                now=_utc_now(),
            )
        except ExchangeConnectionError as error:
            raise _exchange_connection_http_error(error=error) from error
        return _exchange_connection_response(view=view)

    @app.post(
        "/internal/v1/exchange-connections/{connection_id}/rotate",
        include_in_schema=False,
    )
    def rotate_internal_exchange_connection(
        connection_id: UUID,
        payload: RotateExchangeConnectionInternalRequest,
        request: Request,
        authorization: str | None = Header(default=None),
        x_roehub_internal_service: str | None = Header(
            default=None,
            alias="X-Roehub-Internal-Service",
        ),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    ) -> dict[str, object]:
        _require_internal_request(
            request=request,
            authorization=authorization,
            expected_token=config.internal_api_token,
            internal_service=x_roehub_internal_service,
            request_id=x_request_id,
        )
        try:
            view = connection_service.rotate_connection(
                owner_user_id=_parse_user_id(raw_value=payload.owner_user_id),
                connection_id=connection_id,
                api_key=payload.api_key,
                api_secret=payload.api_secret,
                passphrase=payload.passphrase,
                now=_utc_now(),
            )
        except ExchangeConnectionError as error:
            raise _exchange_connection_http_error(error=error) from error
        return _exchange_connection_response(view=view)

    @app.post(
        "/internal/v1/exchange-connections/{connection_id}/disable",
        include_in_schema=False,
    )
    def disable_internal_exchange_connection(
        connection_id: UUID,
        payload: DisableExchangeConnectionInternalRequest,
        request: Request,
        authorization: str | None = Header(default=None),
        x_roehub_internal_service: str | None = Header(
            default=None,
            alias="X-Roehub-Internal-Service",
        ),
        x_request_id: str | None = Header(default=None, alias="X-Request-Id"),
    ) -> dict[str, object]:
        _require_internal_request(
            request=request,
            authorization=authorization,
            expected_token=config.internal_api_token,
            internal_service=x_roehub_internal_service,
            request_id=x_request_id,
        )
        try:
            view = connection_service.disable_connection(
                owner_user_id=_parse_user_id(raw_value=payload.owner_user_id),
                connection_id=connection_id,
                now=_utc_now(),
            )
        except ExchangeConnectionError as error:
            raise _exchange_connection_http_error(error=error) from error
        return _exchange_connection_response(view=view)

    return app


def _require_internal_request(
    *,
    request: Request,
    authorization: str | None,
    expected_token: str | None,
    internal_service: str | None,
    request_id: str | None,
) -> str:
    _require_local_request(request=request)
    _require_internal_auth(
        authorization=authorization,
        expected_token=expected_token,
    )
    _require_internal_service_header(value=internal_service)
    return _require_request_id(value=request_id)


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


def _exchange_connection_http_error(*, error: ExchangeConnectionError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"error": error.payload()},
    )


def _parse_user_id(*, raw_value: str) -> UserId:
    try:
        return UserId.from_string(raw_value)
    except ValueError as exc:
        raise _internal_error(status_code=400, code="owner_user_id_invalid") from exc


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _exchange_connection_response(*, view: ExchangeConnectionView) -> dict[str, object]:
    return {
        "connection_id": str(view.connection_id),
        "credential_version_id": str(view.credential_version_id),
        "exchange_name": view.exchange_name,
        "market_type": view.market_type,
        "environment": view.environment,
        "label": view.label,
        "permissions": view.permissions,
        "api_key": view.api_key,
        "status": view.status,
        "status_reason": view.status_reason,
        "created_at": view.created_at.isoformat(),
        "updated_at": view.updated_at.isoformat(),
        "disabled_at": view.disabled_at.isoformat() if view.disabled_at else None,
    }


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
