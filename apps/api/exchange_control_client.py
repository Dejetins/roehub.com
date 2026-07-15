from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol
from uuid import UUID

import httpx

from trading.platform.secrets import SecureCredentialFile

INTERNAL_SERVICE_HEADER = "X-Roehub-Internal-Service"
REQUEST_ID_HEADER = "X-Request-Id"
INTERNAL_SERVICE_NAME = "apps/api"

_BASE_URL_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL"
_CREDENTIAL_PATH_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN_FILE"
_ROUTES_ENABLED_ENV = "ROEHUB_EXCHANGE_CONNECTIONS_PUBLIC_ROUTES_ENABLED"
_TIMEOUT_ENV = "ROEHUB_EXCHANGE_CONTROL_INTERNAL_TIMEOUT_SECONDS"
_DEFAULT_TIMEOUT_SECONDS = 2.0


class ExchangeControlClientError(RuntimeError):
    """Sanitized exchange-control client error safe for API logs."""


@dataclass(frozen=True)
class ExchangeControlCapabilities:
    service: str
    service_identity: str
    contract_version: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class ExchangeConnectionCommandResult:
    connection_id: str
    credential_version_id: str
    exchange_name: str
    market_type: str
    environment: str
    label: str | None
    permissions: str
    requested_permissions: str
    exchange_permissions: str
    effective_permissions: str
    permission_warnings: tuple[str, ...]
    api_key: str
    status: str
    status_reason: str | None
    validation_status: str
    validation_reason: str | None
    ip_restriction_status: str
    last_validated_at: datetime | None
    created_at: datetime
    updated_at: datetime
    disabled_at: datetime | None
    archived_at: datetime | None
    requested_capability: str = "trading"
    effective_capability: str = "none"
    connection_readiness: str = "needs_action"
    connection_readiness_reason: str = "validation_required"
    permissions_deprecated: bool = True
    used_by_strategies_count: int = 0
    active_strategy_bindings_count: int = 0


@dataclass(frozen=True)
class ExchangeControlBalanceSnapshot:
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal | None


@dataclass(frozen=True)
class ExchangeControlPositionSnapshot:
    instrument_key: str
    side: str
    quantity: Decimal
    entry_price: Decimal | None
    leverage: Decimal | None
    margin_mode: str | None
    position_mode: str | None


@dataclass(frozen=True)
class ExchangeControlOpenOrderSnapshot:
    instrument_key: str
    exchange_order_ref: str
    side: str
    order_type: str
    quantity: Decimal
    price: Decimal | None
    status: str


@dataclass(frozen=True)
class ExchangeControlInstrumentFilterSnapshot:
    instrument_key: str
    tick_size: Decimal | None
    step_size: Decimal | None
    min_qty: Decimal | None
    min_notional: Decimal | None
    max_leverage: Decimal | None


@dataclass(frozen=True)
class ExchangeControlAccountStateSnapshot:
    exchange_name: str
    market_type: str
    environment: str
    account_mode: str
    sync_status: str
    sync_reason: str
    source_hash: str
    observed_at: datetime
    balances: tuple[ExchangeControlBalanceSnapshot, ...]
    positions: tuple[ExchangeControlPositionSnapshot, ...]
    open_orders: tuple[ExchangeControlOpenOrderSnapshot, ...]
    instrument_filters: tuple[ExchangeControlInstrumentFilterSnapshot, ...]


@dataclass(frozen=True)
class ExchangeControlAccountConfigResult:
    exchange_name: str
    market_type: str
    environment: str
    instrument_key: str
    target_margin_mode: str
    target_leverage: Decimal
    observed_margin_mode: str | None
    observed_leverage: Decimal | None
    observed_position_mode: str | None
    account_mode: str
    sync_status: str
    sync_reason: str
    source_hash: str
    observed_at: datetime


class ExchangeControlClient(Protocol):
    def get_capabilities(
        self, *, request_id: str | None = None
    ) -> ExchangeControlCapabilities: ...

    def list_connections(
        self, *, owner_user_id: str, request_id: str | None = None
    ) -> tuple[ExchangeConnectionCommandResult, ...]: ...

    def create_connection(
        self,
        *,
        owner_user_id: str,
        exchange_name: str,
        market_type: str,
        environment: str,
        label: str | None,
        permissions: str,
        api_key: str,
        api_secret: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult: ...

    def create_connection_market(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        market_type: str,
        label: str | None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult: ...

    def rotate_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        api_key: str,
        api_secret: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult: ...

    def disable_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        status_reason: str | None = None,
        reclassification_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult: ...

    def archive_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        cleanup_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult: ...

    def validate_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult: ...

    def read_account_state(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        instrument_keys: tuple[str, ...] = (),
        request_id: str | None = None,
    ) -> ExchangeControlAccountStateSnapshot: ...

    def configure_account(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        instrument_key: str,
        margin_mode: str,
        leverage: Decimal,
        request_id: str | None = None,
    ) -> ExchangeControlAccountConfigResult: ...


@dataclass(frozen=True)
class ExchangeControlClientConfig:
    base_url: str | None
    internal_api_credential: SecureCredentialFile | None
    timeout_seconds: float
    public_routes_enabled: bool

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> "ExchangeControlClientConfig":
        config = cls(
            base_url=_read_optional_str(environ.get(_BASE_URL_ENV)),
            internal_api_credential=(
                SecureCredentialFile(Path(raw_path))
                if (raw_path := _read_optional_str(environ.get(_CREDENTIAL_PATH_ENV)))
                else None
            ),
            timeout_seconds=_read_positive_float(
                value=environ.get(_TIMEOUT_ENV),
                default=_DEFAULT_TIMEOUT_SECONDS,
                name=_TIMEOUT_ENV,
            ),
            public_routes_enabled=_read_bool(environ.get(_ROUTES_ENABLED_ENV)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.public_routes_enabled:
            return
        if not self.base_url:
            raise ValueError(
                "exchange connection routes require "
                "ROEHUB_EXCHANGE_CONTROL_INTERNAL_BASE_URL"
            )
        if self.internal_api_credential is None:
            raise ValueError(
                "exchange connection routes require "
                "ROEHUB_EXCHANGE_CONTROL_INTERNAL_API_TOKEN_FILE"
            )

    def build_client(self) -> ExchangeControlClient | None:
        if not self.base_url or self.internal_api_credential is None:
            return None
        return HttpExchangeControlClient(
            base_url=self.base_url,
            internal_api_credential=self.internal_api_credential,
            timeout_seconds=self.timeout_seconds,
        )


@dataclass(frozen=True)
class HttpExchangeControlClient:
    base_url: str
    internal_api_credential: SecureCredentialFile
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    transport: httpx.BaseTransport | None = None

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("exchange-control internal base URL is required")
        self.internal_api_credential.read()
        if self.timeout_seconds <= 0:
            raise ValueError("exchange-control internal timeout must be positive")

    def get_capabilities(self, *, request_id: str | None = None) -> ExchangeControlCapabilities:
        response = self._request("GET", "/internal/v1/capabilities", request_id=request_id)
        payload = response.json()
        return _capabilities_from_payload(payload)

    def list_connections(
        self, *, owner_user_id: str, request_id: str | None = None
    ) -> tuple[ExchangeConnectionCommandResult, ...]:
        response = self._request(
            "GET",
            "/internal/v1/exchange-connections",
            request_id=request_id,
            params={"owner_user_id": owner_user_id},
        )
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise ExchangeControlClientError("exchange-control connections response is invalid")
        return tuple(_connection_from_payload(item) for item in payload["items"])

    def create_connection(
        self,
        *,
        owner_user_id: str,
        exchange_name: str,
        market_type: str,
        environment: str,
        label: str | None,
        permissions: str,
        api_key: str,
        api_secret: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        response = self._request(
            "POST",
            "/internal/v1/exchange-connections",
            request_id=request_id,
            json={
                "owner_user_id": owner_user_id,
                "exchange_name": exchange_name,
                "market_type": market_type,
                "environment": environment,
                "label": label,
                "permissions": permissions,
                "api_key": api_key,
                "api_secret": api_secret,
            },
        )
        return _connection_from_payload(response.json())

    def create_connection_market(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        market_type: str,
        label: str | None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        payload: dict[str, object] = {
            "owner_user_id": owner_user_id,
            "market_type": market_type,
        }
        if label is not None:
            payload["label"] = label
        response = self._request(
            "POST",
            f"/internal/v1/exchange-connections/{connection_id}/markets",
            request_id=request_id,
            json=payload,
        )
        return _connection_from_payload(response.json())

    def rotate_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        api_key: str,
        api_secret: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        response = self._request(
            "POST",
            f"/internal/v1/exchange-connections/{connection_id}/rotate",
            request_id=request_id,
            json={
                "owner_user_id": owner_user_id,
                "api_key": api_key,
                "api_secret": api_secret,
            },
        )
        return _connection_from_payload(response.json())

    def disable_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        status_reason: str | None = None,
        reclassification_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        payload: dict[str, object] = {"owner_user_id": owner_user_id}
        if status_reason is not None:
            payload["status_reason"] = status_reason
        if reclassification_source is not None:
            payload["reclassification_source"] = reclassification_source
        response = self._request(
            "POST",
            f"/internal/v1/exchange-connections/{connection_id}/disable",
            request_id=request_id,
            json=payload,
        )
        return _connection_from_payload(response.json())

    def archive_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        cleanup_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        payload: dict[str, object] = {"owner_user_id": owner_user_id}
        if cleanup_source is not None:
            payload["cleanup_source"] = cleanup_source
        response = self._request(
            "POST",
            f"/internal/v1/exchange-connections/{connection_id}/archive",
            request_id=request_id,
            json=payload,
        )
        return _connection_from_payload(response.json())

    def validate_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        response = self._request(
            "POST",
            f"/internal/v1/exchange-connections/{connection_id}/validate",
            request_id=request_id,
            json={"owner_user_id": owner_user_id},
        )
        return _connection_from_payload(response.json())

    def read_account_state(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        instrument_keys: tuple[str, ...] = (),
        request_id: str | None = None,
    ) -> ExchangeControlAccountStateSnapshot:
        response = self._request(
            "POST",
            f"/internal/v1/exchange-connections/{connection_id}/account-state",
            request_id=request_id,
            json={
                "owner_user_id": owner_user_id,
                "instrument_keys": list(instrument_keys),
            },
        )
        return _account_state_from_payload(response.json())

    def configure_account(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        instrument_key: str,
        margin_mode: str,
        leverage: Decimal,
        request_id: str | None = None,
    ) -> ExchangeControlAccountConfigResult:
        response = self._request(
            "POST",
            f"/internal/v1/exchange-connections/{connection_id}/account-config",
            request_id=request_id,
            json={
                "owner_user_id": owner_user_id,
                "instrument_key": instrument_key,
                "margin_mode": margin_mode,
                "leverage": str(leverage),
            },
        )
        return _account_config_from_payload(response.json())

    def _request(
        self,
        method: str,
        path: str,
        *,
        request_id: str | None,
        params: Mapping[str, str] | None = None,
        json: Mapping[str, object] | None = None,
    ) -> httpx.Response:
        effective_request_id = request_id or f"apps-api-{uuid.uuid4()}"
        try:
            with httpx.Client(
                base_url=self.base_url.rstrip("/"),
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers={
                        "Authorization": f"Bearer {self.internal_api_credential.read()}",
                        INTERNAL_SERVICE_HEADER: INTERNAL_SERVICE_NAME,
                        REQUEST_ID_HEADER: effective_request_id,
                    },
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            error_code = _safe_error_code(response=exc.response)
            code_suffix = f" code {error_code}" if error_code else ""
            raise ExchangeControlClientError(
                "exchange-control internal request failed with status "
                f"{exc.response.status_code}{code_suffix}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ExchangeControlClientError("exchange-control internal request failed") from exc
        return response


@dataclass(frozen=True)
class InMemoryExchangeControlClient:
    capabilities: ExchangeControlCapabilities = ExchangeControlCapabilities(
        service="exchange-control",
        service_identity="exchange-control",
        contract_version="internal-v1",
        capabilities=("capabilities.read", "exchange_account_config.write"),
    )
    _connections: dict[str, ExchangeConnectionCommandResult] | None = None
    _owners: dict[str, str] | None = None
    _next_id: int = 1

    def get_capabilities(self, *, request_id: str | None = None) -> ExchangeControlCapabilities:
        _ = request_id
        return self.capabilities

    def __post_init__(self) -> None:
        if self._connections is None:
            object.__setattr__(self, "_connections", {})
        if self._owners is None:
            object.__setattr__(self, "_owners", {})

    def list_connections(
        self, *, owner_user_id: str, request_id: str | None = None
    ) -> tuple[ExchangeConnectionCommandResult, ...]:
        _ = request_id
        owners = self._owners_dict()
        rows = sorted(
            (
                connection
                for connection_id, connection in self._connections_dict().items()
                if owners.get(connection_id) == owner_user_id
            ),
            key=lambda item: (item.created_at, item.connection_id),
        )
        return tuple(rows)

    def create_connection(
        self,
        *,
        owner_user_id: str,
        exchange_name: str,
        market_type: str,
        environment: str,
        label: str | None,
        permissions: str,
        api_key: str,
        api_secret: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        _ = owner_user_id, api_secret, request_id
        now = datetime.fromisoformat("2026-05-24T12:00:00+00:00")
        connection_id = str(UUID(int=self._next_id))
        credential_version_id = str(UUID(int=self._next_id + 1000))
        object.__setattr__(self, "_next_id", self._next_id + 1)
        readiness = _fake_auto_validation_result(
            permissions=permissions,
            api_key=api_key,
            api_secret=api_secret,
        )
        status = (
            "active"
            if readiness["connection_readiness"] == "ready_for_trading"
            else "disabled"
        )
        status_reason = None if status == "active" else "auto_validation_failed"
        result = ExchangeConnectionCommandResult(
            connection_id=connection_id,
            credential_version_id=credential_version_id,
            exchange_name=exchange_name,
            market_type=market_type,
            environment=environment,
            label=label,
            permissions=permissions,
            requested_permissions=permissions,
            exchange_permissions=readiness["exchange_permissions"],
            effective_permissions=readiness["effective_permissions"],
            permission_warnings=(),
            api_key=f"****{api_key[-4:]}",
            status=status,
            status_reason=status_reason,
            validation_status=readiness["validation_status"],
            validation_reason=readiness["validation_reason"],
            ip_restriction_status=readiness["ip_restriction_status"],
            last_validated_at=now,
            created_at=now,
            updated_at=now,
            disabled_at=now if status == "disabled" else None,
            archived_at=None,
            requested_capability="trading",
            effective_capability=readiness["effective_capability"],
            connection_readiness=readiness["connection_readiness"],
            connection_readiness_reason=readiness["connection_readiness_reason"],
            permissions_deprecated=True,
        )
        self._connections_dict()[connection_id] = result
        self._owners_dict()[connection_id] = owner_user_id
        return result

    def create_connection_market(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        market_type: str,
        label: str | None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        _ = request_id
        existing = self._connections_dict().get(connection_id)
        if existing is None or self._owners_dict().get(connection_id) != owner_user_id:
            raise ExchangeControlClientError("exchange_connection_not_found")
        if existing.status != "active":
            raise ExchangeControlClientError("exchange_connection_not_found")
        return self.create_connection(
            owner_user_id=owner_user_id,
            exchange_name=existing.exchange_name,
            market_type=market_type,
            environment=existing.environment,
            label=label if label is not None else existing.label,
            permissions=existing.permissions,
            api_key=f"FAKE{existing.api_key[-4:]}",
            api_secret="FAKE_SECRET",
            request_id=request_id,
        )

    def rotate_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        api_key: str,
        api_secret: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        _ = owner_user_id, api_secret, request_id
        existing = self._connections_dict().get(connection_id)
        if existing is None or self._owners_dict().get(connection_id) != owner_user_id:
            raise ExchangeControlClientError("exchange_connection_not_found")
        if existing.status != "active":
            raise ExchangeControlClientError("exchange_connection_not_found")
        readiness = _fake_auto_validation_result(
            permissions=existing.permissions,
            api_key=api_key,
            api_secret=api_secret,
        )
        if readiness["connection_readiness"] != "ready_for_trading":
            raise ExchangeControlClientError(
                "exchange-control internal request failed with status 422 code "
                f"{readiness['connection_readiness_reason']}"
            )
        credential_version_id = str(UUID(int=self._next_id + 1000))
        object.__setattr__(self, "_next_id", self._next_id + 1)
        rotated = ExchangeConnectionCommandResult(
            connection_id=existing.connection_id,
            credential_version_id=credential_version_id,
            exchange_name=existing.exchange_name,
            market_type=existing.market_type,
            environment=existing.environment,
            label=existing.label,
            permissions=existing.permissions,
            requested_permissions=existing.requested_permissions,
            exchange_permissions=readiness["exchange_permissions"],
            effective_permissions=readiness["effective_permissions"],
            permission_warnings=(),
            api_key=f"****{api_key[-4:]}",
            status=existing.status,
            status_reason=existing.status_reason,
            validation_status=readiness["validation_status"],
            validation_reason=readiness["validation_reason"],
            ip_restriction_status=readiness["ip_restriction_status"],
            last_validated_at=datetime.fromisoformat("2026-05-24T12:01:00+00:00"),
            created_at=existing.created_at,
            updated_at=datetime.fromisoformat("2026-05-24T12:01:00+00:00"),
            disabled_at=existing.disabled_at,
            archived_at=existing.archived_at,
            requested_capability="trading",
            effective_capability=readiness["effective_capability"],
            connection_readiness=readiness["connection_readiness"],
            connection_readiness_reason=readiness["connection_readiness_reason"],
            permissions_deprecated=True,
        )
        self._connections_dict()[connection_id] = rotated
        return rotated

    def disable_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        status_reason: str | None = None,
        reclassification_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        _ = owner_user_id, reclassification_source, request_id
        existing = self._connections_dict().get(connection_id)
        if existing is None or self._owners_dict().get(connection_id) != owner_user_id:
            raise ExchangeControlClientError("exchange_connection_not_found")
        if existing.status != "active":
            raise ExchangeControlClientError("exchange_connection_not_found")
        disabled_at = datetime.fromisoformat("2026-05-24T12:02:00+00:00")
        disabled = ExchangeConnectionCommandResult(
            connection_id=existing.connection_id,
            credential_version_id=existing.credential_version_id,
            exchange_name=existing.exchange_name,
            market_type=existing.market_type,
            environment=existing.environment,
            label=existing.label,
            permissions=existing.permissions,
            requested_permissions=existing.requested_permissions,
            exchange_permissions=existing.exchange_permissions,
            effective_permissions=existing.effective_permissions,
            permission_warnings=existing.permission_warnings,
            api_key=existing.api_key,
            status="disabled",
            status_reason=status_reason or "user_disabled",
            validation_status=existing.validation_status,
            validation_reason=existing.validation_reason,
            ip_restriction_status=existing.ip_restriction_status,
            last_validated_at=existing.last_validated_at,
            created_at=existing.created_at,
            updated_at=disabled_at,
            disabled_at=disabled_at,
            archived_at=None,
            requested_capability="trading",
            effective_capability="none",
            connection_readiness="disconnected",
            connection_readiness_reason="user_disconnected",
            permissions_deprecated=True,
        )
        self._connections_dict()[connection_id] = disabled
        return disabled

    def archive_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        cleanup_source: str | None = None,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        _ = cleanup_source, request_id
        existing = self._connections_dict().get(connection_id)
        if existing is None or self._owners_dict().get(connection_id) != owner_user_id:
            raise ExchangeControlClientError("exchange_connection_not_found")
        if existing.status == "active":
            raise ExchangeControlClientError("exchange_connection_not_disabled")
        if existing.status == "archived":
            return existing
        archived_at = datetime.fromisoformat("2026-05-24T12:03:00+00:00")
        archived = ExchangeConnectionCommandResult(
            connection_id=existing.connection_id,
            credential_version_id=existing.credential_version_id,
            exchange_name=existing.exchange_name,
            market_type=existing.market_type,
            environment=existing.environment,
            label=existing.label,
            permissions=existing.permissions,
            requested_permissions=existing.requested_permissions,
            exchange_permissions=existing.exchange_permissions,
            effective_permissions=existing.effective_permissions,
            permission_warnings=existing.permission_warnings,
            api_key=existing.api_key,
            status="archived",
            status_reason="user_archived",
            validation_status=existing.validation_status,
            validation_reason=existing.validation_reason,
            ip_restriction_status=existing.ip_restriction_status,
            last_validated_at=existing.last_validated_at,
            created_at=existing.created_at,
            updated_at=archived_at,
            disabled_at=existing.disabled_at,
            archived_at=archived_at,
            requested_capability="trading",
            effective_capability="none",
            connection_readiness="archived",
            connection_readiness_reason="archived",
            permissions_deprecated=True,
        )
        self._connections_dict()[connection_id] = archived
        return archived

    def validate_connection(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        request_id: str | None = None,
    ) -> ExchangeConnectionCommandResult:
        _ = request_id
        existing = self._connections_dict().get(connection_id)
        if existing is None or self._owners_dict().get(connection_id) != owner_user_id:
            raise ExchangeControlClientError("exchange_connection_not_found")
        if existing.status != "active":
            raise ExchangeControlClientError("exchange_connection_not_found")
        validated_at = datetime.fromisoformat("2026-05-24T12:03:00+00:00")
        validated = ExchangeConnectionCommandResult(
            connection_id=existing.connection_id,
            credential_version_id=existing.credential_version_id,
            exchange_name=existing.exchange_name,
            market_type=existing.market_type,
            environment=existing.environment,
            label=existing.label,
            permissions=existing.permissions,
            requested_permissions=existing.requested_permissions,
            exchange_permissions="read",
            effective_permissions="read",
            permission_warnings=(),
            api_key=existing.api_key,
            status=existing.status,
            status_reason=existing.status_reason,
            validation_status="valid_readonly",
            validation_reason="fake_client_readonly",
            ip_restriction_status="not_restricted_testnet",
            last_validated_at=validated_at,
            created_at=existing.created_at,
            updated_at=validated_at,
            disabled_at=existing.disabled_at,
            archived_at=existing.archived_at,
            requested_capability="trading",
            effective_capability="none",
            connection_readiness="rejected",
            connection_readiness_reason="read_only_not_supported",
            permissions_deprecated=True,
        )
        self._connections_dict()[connection_id] = validated
        return validated

    def read_account_state(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        instrument_keys: tuple[str, ...] = (),
        request_id: str | None = None,
    ) -> ExchangeControlAccountStateSnapshot:
        _ = request_id
        existing = self._connections_dict().get(connection_id)
        if existing is None or self._owners_dict().get(connection_id) != owner_user_id:
            raise ExchangeControlClientError("exchange_connection_not_found")
        if existing.connection_readiness != "ready_for_trading":
            raise ExchangeControlClientError("exchange_connection_not_ready")
        observed_at = datetime.fromisoformat("2026-05-24T12:04:00+00:00")
        filters = tuple(
            ExchangeControlInstrumentFilterSnapshot(
                instrument_key=instrument_key,
                tick_size=Decimal("0.01"),
                step_size=Decimal("0.00001"),
                min_qty=Decimal("0.00001"),
                min_notional=Decimal("5"),
                max_leverage=None,
            )
            for instrument_key in instrument_keys
        )
        return ExchangeControlAccountStateSnapshot(
            exchange_name=existing.exchange_name,
            market_type=existing.market_type,
            environment=existing.environment,
            account_mode="unified",
            sync_status="fresh",
            sync_reason="account_state_read_ok",
            source_hash="b" * 64,
            observed_at=observed_at,
            balances=(
                ExchangeControlBalanceSnapshot(
                    asset="USDT",
                    free=Decimal("100"),
                    locked=Decimal("0"),
                    total=Decimal("100"),
                ),
            ),
            positions=(),
            open_orders=(),
            instrument_filters=filters,
        )

    def configure_account(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        instrument_key: str,
        margin_mode: str,
        leverage: Decimal,
        request_id: str | None = None,
    ) -> ExchangeControlAccountConfigResult:
        _ = request_id
        existing = self._connections_dict().get(connection_id)
        if existing is None or self._owners_dict().get(connection_id) != owner_user_id:
            raise ExchangeControlClientError("exchange_connection_not_found")
        if existing.connection_readiness != "ready_for_trading":
            raise ExchangeControlClientError("exchange_connection_not_ready")
        if existing.market_type != "futures":
            raise ExchangeControlClientError(
                "exchange-control internal request failed with status 422 code "
                "exchange_account_config_futures_only"
            )
        observed_at = datetime.fromisoformat("2026-05-24T12:05:00+00:00")
        return ExchangeControlAccountConfigResult(
            exchange_name=existing.exchange_name,
            market_type=existing.market_type,
            environment=existing.environment,
            instrument_key=instrument_key,
            target_margin_mode=margin_mode,
            target_leverage=leverage,
            observed_margin_mode=margin_mode,
            observed_leverage=leverage,
            observed_position_mode="one_way",
            account_mode="unified" if existing.exchange_name == "bybit" else "futures",
            sync_status="fresh",
            sync_reason="account_config_write_ok",
            source_hash="c" * 64,
            observed_at=observed_at,
        )

    def _connections_dict(self) -> dict[str, ExchangeConnectionCommandResult]:
        if self._connections is None:
            raise ExchangeControlClientError("exchange-control fake client is uninitialized")
        return self._connections

    def _owners_dict(self) -> dict[str, str]:
        if self._owners is None:
            raise ExchangeControlClientError("exchange-control fake client is uninitialized")
        return self._owners


def build_exchange_control_client_from_environ(
    *,
    environ: Mapping[str, str],
) -> ExchangeControlClient | None:
    return ExchangeControlClientConfig.from_environ(environ).build_client()


def _capabilities_from_payload(payload: object) -> ExchangeControlCapabilities:
    if not isinstance(payload, dict):
        raise ExchangeControlClientError("exchange-control internal response is invalid")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or not all(
        isinstance(item, str) and item for item in capabilities
    ):
        raise ExchangeControlClientError("exchange-control capabilities response is invalid")
    service = payload.get("service")
    service_identity = payload.get("service_identity")
    contract_version = payload.get("contract_version")
    if not isinstance(service, str) or not isinstance(service_identity, str):
        raise ExchangeControlClientError("exchange-control capabilities response is invalid")
    if not isinstance(contract_version, str):
        raise ExchangeControlClientError("exchange-control capabilities response is invalid")
    return ExchangeControlCapabilities(
        service=service,
        service_identity=service_identity,
        contract_version=contract_version,
        capabilities=tuple(capabilities),
    )


def _connection_from_payload(payload: object) -> ExchangeConnectionCommandResult:
    if not isinstance(payload, dict):
        raise ExchangeControlClientError("exchange-control connection response is invalid")
    try:
        return ExchangeConnectionCommandResult(
            connection_id=str(UUID(str(payload["connection_id"]))),
            credential_version_id=str(UUID(str(payload["credential_version_id"]))),
            exchange_name=str(payload["exchange_name"]),
            market_type=str(payload["market_type"]),
            environment=str(payload["environment"]),
            label=str(payload["label"]) if payload.get("label") is not None else None,
            permissions=str(payload["permissions"]),
            requested_permissions=str(
                payload.get("requested_permissions") or payload["permissions"]
            ),
            exchange_permissions=str(payload.get("exchange_permissions") or "unknown"),
            effective_permissions=str(payload.get("effective_permissions") or "none"),
            permission_warnings=_permission_warnings_from_payload(
                payload.get("permission_warnings")
            ),
            api_key=str(payload["api_key"]),
            status=str(payload["status"]),
            status_reason=(
                str(payload["status_reason"])
                if payload.get("status_reason") is not None
                else None
            ),
            validation_status=str(
                payload.get("validation_status") or "skipped_external_validation"
            ),
            validation_reason=(
                str(payload["validation_reason"])
                if payload.get("validation_reason") is not None
                else None
            ),
            ip_restriction_status=str(payload.get("ip_restriction_status") or "unknown"),
            last_validated_at=(
                datetime.fromisoformat(str(payload["last_validated_at"]))
                if payload.get("last_validated_at") is not None
                else None
            ),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            updated_at=datetime.fromisoformat(str(payload["updated_at"])),
            disabled_at=(
                datetime.fromisoformat(str(payload["disabled_at"]))
                if payload.get("disabled_at") is not None
                else None
            ),
            archived_at=(
                datetime.fromisoformat(str(payload["archived_at"]))
                if payload.get("archived_at") is not None
                else None
            ),
            requested_capability=str(payload.get("requested_capability") or "trading"),
            effective_capability=str(payload.get("effective_capability") or "none"),
            connection_readiness=str(
                payload.get("connection_readiness") or "needs_action"
            ),
            connection_readiness_reason=str(
                payload.get("connection_readiness_reason") or "validation_required"
            ),
            permissions_deprecated=bool(payload.get("permissions_deprecated", True)),
            used_by_strategies_count=int(payload.get("used_by_strategies_count") or 0),
            active_strategy_bindings_count=int(
                payload.get("active_strategy_bindings_count") or 0
            ),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ExchangeControlClientError(
            "exchange-control connection response is invalid"
        ) from exc


def _account_state_from_payload(payload: object) -> ExchangeControlAccountStateSnapshot:
    if not isinstance(payload, dict):
        raise ExchangeControlClientError("exchange-control account-state response is invalid")
    try:
        return ExchangeControlAccountStateSnapshot(
            exchange_name=str(payload["exchange_name"]),
            market_type=str(payload["market_type"]),
            environment=str(payload["environment"]),
            account_mode=str(payload["account_mode"]),
            sync_status=str(payload["sync_status"]),
            sync_reason=str(payload["sync_reason"]),
            source_hash=str(payload["source_hash"]),
            observed_at=datetime.fromisoformat(str(payload["observed_at"])),
            balances=tuple(
                _balance_from_payload(item) for item in _list_payload(payload, "balances")
            ),
            positions=tuple(
                _position_from_payload(item) for item in _list_payload(payload, "positions")
            ),
            open_orders=tuple(
                _order_from_payload(item) for item in _list_payload(payload, "open_orders")
            ),
            instrument_filters=tuple(
                _filter_from_payload(item)
                for item in _list_payload(payload, "instrument_filters")
            ),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ExchangeControlClientError(
            "exchange-control account-state response is invalid"
        ) from exc


def _account_config_from_payload(payload: object) -> ExchangeControlAccountConfigResult:
    if not isinstance(payload, dict):
        raise ExchangeControlClientError("exchange-control account-config response is invalid")
    try:
        return ExchangeControlAccountConfigResult(
            exchange_name=str(payload["exchange_name"]),
            market_type=str(payload["market_type"]),
            environment=str(payload["environment"]),
            instrument_key=str(payload["instrument_key"]),
            target_margin_mode=str(payload["target_margin_mode"]),
            target_leverage=Decimal(str(payload["target_leverage"])),
            observed_margin_mode=(
                str(payload["observed_margin_mode"])
                if payload.get("observed_margin_mode") is not None
                else None
            ),
            observed_leverage=_decimal_or_none(payload.get("observed_leverage")),
            observed_position_mode=(
                str(payload["observed_position_mode"])
                if payload.get("observed_position_mode") is not None
                else None
            ),
            account_mode=str(payload["account_mode"]),
            sync_status=str(payload["sync_status"]),
            sync_reason=str(payload["sync_reason"]),
            source_hash=str(payload["source_hash"]),
            observed_at=datetime.fromisoformat(str(payload["observed_at"])),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise ExchangeControlClientError(
            "exchange-control account-config response is invalid"
        ) from exc


def _list_payload(payload: dict[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _balance_from_payload(payload: object) -> ExchangeControlBalanceSnapshot:
    if not isinstance(payload, dict):
        raise ValueError("balance payload is invalid")
    return ExchangeControlBalanceSnapshot(
        asset=str(payload["asset"]),
        free=Decimal(str(payload["free"])),
        locked=Decimal(str(payload["locked"])),
        total=_decimal_or_none(payload.get("total")),
    )


def _position_from_payload(payload: object) -> ExchangeControlPositionSnapshot:
    if not isinstance(payload, dict):
        raise ValueError("position payload is invalid")
    return ExchangeControlPositionSnapshot(
        instrument_key=str(payload["instrument_key"]),
        side=str(payload["side"]),
        quantity=Decimal(str(payload["quantity"])),
        entry_price=_decimal_or_none(payload.get("entry_price")),
        leverage=_decimal_or_none(payload.get("leverage")),
        margin_mode=str(payload["margin_mode"]) if payload.get("margin_mode") else None,
        position_mode=str(payload["position_mode"]) if payload.get("position_mode") else None,
    )


def _order_from_payload(payload: object) -> ExchangeControlOpenOrderSnapshot:
    if not isinstance(payload, dict):
        raise ValueError("open order payload is invalid")
    return ExchangeControlOpenOrderSnapshot(
        instrument_key=str(payload["instrument_key"]),
        exchange_order_ref=str(payload["exchange_order_ref"]),
        side=str(payload["side"]),
        order_type=str(payload["order_type"]),
        quantity=Decimal(str(payload["quantity"])),
        price=_decimal_or_none(payload.get("price")),
        status=str(payload["status"]),
    )


def _filter_from_payload(payload: object) -> ExchangeControlInstrumentFilterSnapshot:
    if not isinstance(payload, dict):
        raise ValueError("instrument filter payload is invalid")
    return ExchangeControlInstrumentFilterSnapshot(
        instrument_key=str(payload["instrument_key"]),
        tick_size=_decimal_or_none(payload.get("tick_size")),
        step_size=_decimal_or_none(payload.get("step_size")),
        min_qty=_decimal_or_none(payload.get("min_qty")),
        min_notional=_decimal_or_none(payload.get("min_notional")),
        max_leverage=_decimal_or_none(payload.get("max_leverage")),
    )


def _decimal_or_none(value: object) -> Decimal | None:
    if value is None:
        return None
    raw = str(value).strip()
    return Decimal(raw) if raw else None


def _safe_error_code(*, response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if isinstance(detail, dict):
        error = detail.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return error["code"]
    error = payload.get("error")
    if isinstance(error, dict) and isinstance(error.get("code"), str):
        return error["code"]
    return None


def _fake_auto_validation_result(
    *,
    permissions: str,
    api_key: str,
    api_secret: str,
) -> dict[str, str]:
    if api_key == "INVALID" or api_secret == "INVALID":
        return {
            "exchange_permissions": "unknown",
            "effective_permissions": "none",
            "validation_status": "invalid_credentials",
            "validation_reason": "fake_client_invalid_credentials",
            "ip_restriction_status": "unknown",
            "effective_capability": "none",
            "connection_readiness": "rejected",
            "connection_readiness_reason": "invalid_credentials",
        }
    if "UNAVAILABLE" in api_secret:
        return {
            "exchange_permissions": "unknown",
            "effective_permissions": "none",
            "validation_status": "skipped_external_validation",
            "validation_reason": "fake_client_validation_unavailable",
            "ip_restriction_status": "not_checked",
            "effective_capability": "none",
            "connection_readiness": "needs_action",
            "connection_readiness_reason": "validation_unavailable",
        }
    if permissions != "trade" or "READONLY" in api_secret:
        return {
            "exchange_permissions": "read",
            "effective_permissions": "read",
            "validation_status": "valid_readonly",
            "validation_reason": "fake_client_readonly",
            "ip_restriction_status": "not_restricted_testnet",
            "effective_capability": "none",
            "connection_readiness": "rejected",
            "connection_readiness_reason": "read_only_not_supported",
        }
    return {
        "exchange_permissions": "trade",
        "effective_permissions": "trade",
        "validation_status": "valid_trade_enabled",
        "validation_reason": "fake_client_trade_ready",
        "ip_restriction_status": "restricted",
        "effective_capability": "trading",
        "connection_readiness": "ready_for_trading",
        "connection_readiness_reason": "trading_policy_ok",
    }


def _permission_warnings_from_payload(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        warning
        for warning in value
        if isinstance(warning, str)
        and warning in {"exchange_permissions_exceed_requested"}
    )


def _read_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _read_bool(value: str | None) -> bool:
    if value is None or value == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_positive_float(*, value: str | None, default: float, name: str) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


__all__ = [
    "INTERNAL_SERVICE_HEADER",
    "INTERNAL_SERVICE_NAME",
    "REQUEST_ID_HEADER",
    "ExchangeControlCapabilities",
    "ExchangeControlAccountStateSnapshot",
    "ExchangeControlBalanceSnapshot",
    "ExchangeControlClient",
    "ExchangeControlClientConfig",
    "ExchangeControlClientError",
    "ExchangeControlInstrumentFilterSnapshot",
    "ExchangeConnectionCommandResult",
    "ExchangeControlOpenOrderSnapshot",
    "ExchangeControlPositionSnapshot",
    "HttpExchangeControlClient",
    "InMemoryExchangeControlClient",
    "build_exchange_control_client_from_environ",
]
