from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from trading.contexts.exchange_control.application.validation import (
    EffectivePermissions,
    ExchangeCredentialValidationRequest,
    ExchangeCredentialValidationResult,
    ExchangeCredentialValidator,
    ExchangePermissions,
    ExchangeValidationStatus,
    PermissionWarning,
    RequestedPermissions,
)

_BINANCE_MAINNET_URL = "https://api.binance.com"
_BINANCE_SPOT_DEMO_URL = "https://demo-api.binance.com"
_BINANCE_FUTURES_TESTNET_URL = "https://demo-fapi.binance.com"
_BYBIT_MAINNET_URL = "https://api.bybit.com"
_BYBIT_TESTNET_URL = "https://api-testnet.bybit.com"
_RECV_WINDOW = "5000"


@dataclass(frozen=True)
class HttpExchangeCredentialValidator(ExchangeCredentialValidator):
    timeout_seconds: float = 3.0
    requires_plaintext: bool = True

    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
        now: datetime,
    ) -> ExchangeCredentialValidationResult:
        _ = now
        if request.exchange_name == "binance":
            return BinanceExchangeCredentialValidator(
                timeout_seconds=self.timeout_seconds
            ).validate(request=request)
        if request.exchange_name == "bybit":
            return BybitExchangeCredentialValidator(
                timeout_seconds=self.timeout_seconds
            ).validate(request=request)
        return ExchangeCredentialValidationResult(
            status="unsupported_account_mode",
            reason="unsupported_exchange",
            ip_restriction_status="unknown",
            permission_summary={"exchange": request.exchange_name},
        )


@dataclass(frozen=True)
class BinanceExchangeCredentialValidator:
    timeout_seconds: float = 3.0

    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
    ) -> ExchangeCredentialValidationResult:
        if request.market_type == "spot" and request.environment == "testnet":
            return self._validate_spot_demo(request=request)
        if request.market_type == "futures" and request.environment == "testnet":
            return self._validate_futures_testnet(request=request)
        base_url = _BINANCE_MAINNET_URL
        timestamp = str(int(time.time() * 1000))
        query = urllib.parse.urlencode({"recvWindow": _RECV_WINDOW, "timestamp": timestamp})
        signature = hmac.new(
            request.credential.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        url = (
            f"{base_url}/sapi/v1/account/apiRestrictions"
            f"?{query}&signature={signature}"
        )
        try:
            payload = _get_json(
                url=url,
                headers={"X-MBX-APIKEY": request.credential.api_key},
                timeout_seconds=self.timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            return _invalid_credentials_from_http(
                status_code=exc.code,
                exchange="binance",
                requested_permissions=request.requested_permissions,
            )
        except (OSError, ValueError):
            return ExchangeCredentialValidationResult(
                status="skipped_external_validation",
                reason="exchange_request_failed",
                ip_restriction_status="not_checked",
                permission_summary=_permission_summary(
                    base={"exchange": "binance"},
                    requested_permissions=request.requested_permissions,
                    exchange_permissions="unknown",
                    effective_permissions="none",
                    permission_warnings=(),
                ),
            )
        return normalize_binance_api_restrictions(
            payload=payload,
            environment=request.environment,
            requested_permissions=request.requested_permissions,
        )

    def _validate_spot_demo(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
    ) -> ExchangeCredentialValidationResult:
        try:
            account_payload = _binance_signed_request(
                method="GET",
                base_url=_BINANCE_SPOT_DEMO_URL,
                path="/api/v3/account",
                params={},
                api_key=request.credential.api_key,
                api_secret=request.credential.api_secret,
                timeout_seconds=self.timeout_seconds,
            )
        except _ExchangeHttpError as exc:
            return _invalid_credentials_from_http(
                status_code=exc.status_code,
                exchange="binance",
                requested_permissions=request.requested_permissions,
            )
        except (OSError, ValueError):
            return ExchangeCredentialValidationResult(
                status="skipped_external_validation",
                reason="exchange_request_failed",
                ip_restriction_status="not_checked",
                permission_summary=_permission_summary(
                    base={"exchange": "binance", "market_type": "spot"},
                    requested_permissions=request.requested_permissions,
                    exchange_permissions="unknown",
                    effective_permissions="none",
                    permission_warnings=(),
                ),
            )
        exchange_permissions: ExchangePermissions = "read"
        if _requested_permissions(value=request.requested_permissions) == "trade":
            try:
                exchange_permissions = self._probe_spot_demo_trade_permission(
                    request=request,
                )
            except (OSError, ValueError):
                return ExchangeCredentialValidationResult(
                    status="skipped_external_validation",
                    reason="exchange_request_failed",
                    ip_restriction_status="not_checked",
                    permission_summary=_permission_summary(
                        base={"exchange": "binance", "market_type": "spot"},
                        requested_permissions=request.requested_permissions,
                        exchange_permissions="unknown",
                        effective_permissions="none",
                        permission_warnings=(),
                    ),
                )
        return normalize_binance_spot_demo_account(
            payload=account_payload,
            requested_permissions=request.requested_permissions,
            exchange_permissions=exchange_permissions,
        )

    def _probe_spot_demo_trade_permission(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
    ) -> ExchangePermissions:
        params = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": "GTC",
            "quantity": "0.001",
            "price": "10000",
        }
        try:
            _binance_signed_request(
                method="POST",
                base_url=_BINANCE_SPOT_DEMO_URL,
                path="/api/v3/order/test",
                params=params,
                api_key=request.credential.api_key,
                api_secret=request.credential.api_secret,
                timeout_seconds=self.timeout_seconds,
            )
        except _ExchangeHttpError as exc:
            if _binance_auth_or_permission_error(error=exc):
                return "read"
            if exc.status_code == 400:
                return "trade"
            raise ValueError("spot demo trade permission probe failed") from exc
        return "trade"

    def _validate_futures_testnet(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
    ) -> ExchangeCredentialValidationResult:
        timestamp = str(int(time.time() * 1000))
        query = urllib.parse.urlencode({"recvWindow": _RECV_WINDOW, "timestamp": timestamp})
        signature = hmac.new(
            request.credential.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        url = (
            f"{_BINANCE_FUTURES_TESTNET_URL}/fapi/v2/account"
            f"?{query}&signature={signature}"
        )
        try:
            payload = _get_json(
                url=url,
                headers={"X-MBX-APIKEY": request.credential.api_key},
                timeout_seconds=self.timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            return _invalid_credentials_from_http(
                status_code=exc.code,
                exchange="binance",
                requested_permissions=request.requested_permissions,
            )
        except (OSError, ValueError):
            return ExchangeCredentialValidationResult(
                status="skipped_external_validation",
                reason="exchange_request_failed",
                ip_restriction_status="not_checked",
                permission_summary=_permission_summary(
                    base={"exchange": "binance", "market_type": "futures"},
                    requested_permissions=request.requested_permissions,
                    exchange_permissions="unknown",
                    effective_permissions="none",
                    permission_warnings=(),
                ),
            )
        return normalize_binance_futures_testnet_account(
            payload=payload,
            requested_permissions=request.requested_permissions,
        )


@dataclass(frozen=True)
class BybitExchangeCredentialValidator:
    timeout_seconds: float = 3.0

    def validate(
        self,
        *,
        request: ExchangeCredentialValidationRequest,
    ) -> ExchangeCredentialValidationResult:
        base_url = (
            _BYBIT_TESTNET_URL if request.environment == "testnet" else _BYBIT_MAINNET_URL
        )
        timestamp = str(int(time.time() * 1000))
        query = ""
        signing_payload = f"{timestamp}{request.credential.api_key}{_RECV_WINDOW}{query}"
        signature = hmac.new(
            request.credential.api_secret.encode("utf-8"),
            signing_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        try:
            payload = _get_json(
                url=f"{base_url}/v5/user/query-api",
                headers={
                    "X-BAPI-API-KEY": request.credential.api_key,
                    "X-BAPI-TIMESTAMP": timestamp,
                    "X-BAPI-RECV-WINDOW": _RECV_WINDOW,
                    "X-BAPI-SIGN": signature,
                },
                timeout_seconds=self.timeout_seconds,
            )
        except urllib.error.HTTPError as exc:
            return _invalid_credentials_from_http(
                status_code=exc.code,
                exchange="bybit",
                requested_permissions=request.requested_permissions,
            )
        except (OSError, ValueError):
            return ExchangeCredentialValidationResult(
                status="skipped_external_validation",
                reason="exchange_request_failed",
                ip_restriction_status="not_checked",
                permission_summary=_permission_summary(
                    base={"exchange": "bybit"},
                    requested_permissions=request.requested_permissions,
                    exchange_permissions="unknown",
                    effective_permissions="none",
                    permission_warnings=(),
                ),
            )
        return normalize_bybit_api_key_info(
            payload=payload,
            environment=request.environment,
            requested_permissions=request.requested_permissions,
        )


def normalize_binance_api_restrictions(
    *,
    payload: dict[str, Any],
    environment: str,
    requested_permissions: str = "read",
) -> ExchangeCredentialValidationResult:
    requested = _requested_permissions(value=requested_permissions)
    ip_status = _ip_status(payload=payload, environment=environment)
    if not bool(payload.get("enableReading")):
        return _with_permission_policy(
            status="invalid_permissions",
            reason="reading_permission_disabled",
            ip_restriction_status=ip_status,
            base_summary=_binance_summary(payload=payload),
            requested_permissions=requested,
            exchange_permissions="unknown",
        )
    if _binance_withdraw_or_transfer_enabled(payload=payload):
        return _with_permission_policy(
            status="invalid_permissions",
            reason="withdraw_or_transfer_enabled",
            ip_restriction_status=ip_status,
            base_summary=_binance_summary(payload=payload),
            requested_permissions=requested,
            exchange_permissions="withdraw_or_transfer",
        )
    exchange_permissions: ExchangePermissions = (
        "trade" if _binance_trade_enabled(payload=payload) else "read"
    )
    if ip_status == "missing_mainnet_restriction":
        return _with_permission_policy(
            status="invalid_ip_restriction",
            reason="mainnet_ip_restriction_missing",
            ip_restriction_status="missing_mainnet_restriction",
            base_summary=_binance_summary(payload=payload),
            requested_permissions=requested,
            exchange_permissions=exchange_permissions,
        )
    if bool(payload.get("enablePortfolioMarginTrading")):
        return _with_permission_policy(
            status="unsupported_account_mode",
            reason="portfolio_margin_enabled",
            ip_restriction_status=ip_status,
            account_mode="portfolio_margin",
            base_summary=_binance_summary(payload=payload),
            requested_permissions=requested,
            exchange_permissions=exchange_permissions,
        )
    if _binance_trade_enabled(payload=payload):
        return _with_permission_policy(
            status="valid_trade_enabled",
            reason="trade_permission_detected",
            ip_restriction_status=ip_status,
            base_summary=_binance_summary(payload=payload),
            requested_permissions=requested,
            exchange_permissions="trade",
        )
    return _with_permission_policy(
        status="valid_readonly",
        reason="readonly_permission_detected",
        ip_restriction_status=ip_status,
        base_summary=_binance_summary(payload=payload),
        requested_permissions=requested,
        exchange_permissions="read",
    )


def normalize_binance_futures_testnet_account(
    *,
    payload: dict[str, Any],
    requested_permissions: str = "read",
) -> ExchangeCredentialValidationResult:
    requested = _requested_permissions(value=requested_permissions)
    exchange_permissions: ExchangePermissions = (
        "trade" if bool(payload.get("canTrade")) else "read"
    )
    if exchange_permissions == "trade":
        return _with_permission_policy(
            status="valid_trade_enabled",
            reason="trade_permission_detected",
            ip_restriction_status="not_restricted_testnet",
            base_summary={
                "exchange": "binance",
                "market_type": "futures",
                "read": True,
                "trade": True,
                "withdraw_or_transfer": False,
                "multi_assets_margin": bool(payload.get("multiAssetsMargin")),
            },
            requested_permissions=requested,
            exchange_permissions="trade",
        )
    return _with_permission_policy(
        status="valid_readonly",
        reason="readonly_permission_detected",
        ip_restriction_status="not_restricted_testnet",
        base_summary={
            "exchange": "binance",
            "market_type": "futures",
            "read": True,
            "trade": False,
            "withdraw_or_transfer": False,
            "multi_assets_margin": bool(payload.get("multiAssetsMargin")),
        },
        requested_permissions=requested,
        exchange_permissions="read",
    )


def normalize_binance_spot_demo_account(
    *,
    payload: dict[str, Any],
    requested_permissions: str = "read",
    exchange_permissions: ExchangePermissions = "read",
) -> ExchangeCredentialValidationResult:
    requested = _requested_permissions(value=requested_permissions)
    permissions = payload.get("permissions")
    if isinstance(permissions, list) and "SPOT" not in {
        str(item).upper() for item in permissions
    }:
        return _with_permission_policy(
            status="unsupported_account_mode",
            reason="spot_permission_missing",
            ip_restriction_status="not_restricted_testnet",
            account_mode=str(payload.get("accountType") or "unknown").lower(),
            base_summary=_binance_spot_demo_summary(
                payload=payload,
                exchange_permissions="unknown",
            ),
            requested_permissions=requested,
            exchange_permissions="unknown",
        )
    if exchange_permissions == "trade":
        return _with_permission_policy(
            status="valid_trade_enabled",
            reason="trade_permission_detected",
            ip_restriction_status="not_restricted_testnet",
            account_mode=str(payload.get("accountType") or "spot").lower(),
            base_summary=_binance_spot_demo_summary(
                payload=payload,
                exchange_permissions="trade",
            ),
            requested_permissions=requested,
            exchange_permissions="trade",
        )
    return _with_permission_policy(
        status="valid_readonly",
        reason="readonly_permission_detected",
        ip_restriction_status="not_restricted_testnet",
        account_mode=str(payload.get("accountType") or "spot").lower(),
        base_summary=_binance_spot_demo_summary(
            payload=payload,
            exchange_permissions="read",
        ),
        requested_permissions=requested,
        exchange_permissions="read",
    )


def normalize_bybit_api_key_info(
    *,
    payload: dict[str, Any],
    environment: str,
    requested_permissions: str = "read",
) -> ExchangeCredentialValidationResult:
    requested = _requested_permissions(value=requested_permissions)
    if int(payload.get("retCode", 0)) != 0:
        return ExchangeCredentialValidationResult(
            status="invalid_credentials",
            reason="exchange_rejected_credentials",
            ip_restriction_status="unknown",
            permission_summary=_permission_summary(
                base={"exchange": "bybit"},
                requested_permissions=requested,
                exchange_permissions="unknown",
                effective_permissions="none",
                permission_warnings=(),
            ),
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        return ExchangeCredentialValidationResult(
            status="invalid_credentials",
            reason="exchange_response_invalid",
            ip_restriction_status="unknown",
            permission_summary=_permission_summary(
                base={"exchange": "bybit"},
                requested_permissions=requested,
                exchange_permissions="unknown",
                effective_permissions="none",
                permission_warnings=(),
            ),
        )
    account_mode = _bybit_account_mode(result=result)
    read_only = int(result.get("readOnly", 0)) == 1
    ip_status = _bybit_ip_status(result=result, environment=environment)
    exchange_permissions: ExchangePermissions = "read" if read_only else "trade"
    if account_mode == "unsupported":
        return _with_permission_policy(
            status="unsupported_account_mode",
            reason="unsupported_account_mode",
            ip_restriction_status=ip_status,
            account_mode=account_mode,
            base_summary=_bybit_summary(result=result),
            requested_permissions=requested,
            exchange_permissions=exchange_permissions,
        )
    if not read_only and _bybit_transfer_enabled(result=result):
        return _with_permission_policy(
            status="invalid_permissions",
            reason="transfer_permission_enabled",
            ip_restriction_status=ip_status,
            account_mode=account_mode,
            base_summary=_bybit_summary(result=result),
            requested_permissions=requested,
            exchange_permissions="withdraw_or_transfer",
        )
    if ip_status == "missing_mainnet_restriction":
        return _with_permission_policy(
            status="invalid_ip_restriction",
            reason="mainnet_ip_restriction_missing",
            ip_restriction_status=ip_status,
            account_mode=account_mode,
            base_summary=_bybit_summary(result=result),
            requested_permissions=requested,
            exchange_permissions=exchange_permissions,
        )
    if read_only:
        return _with_permission_policy(
            status="valid_readonly",
            reason="readonly_permission_detected",
            ip_restriction_status=ip_status,
            account_mode=account_mode,
            base_summary=_bybit_summary(result=result),
            requested_permissions=requested,
            exchange_permissions="read",
        )
    return _with_permission_policy(
        status="valid_trade_enabled",
        reason="write_permission_detected",
        ip_restriction_status=ip_status,
        account_mode=account_mode,
        base_summary=_bybit_summary(result=result),
        requested_permissions=requested,
        exchange_permissions="trade",
    )


def _get_json(
    *,
    url: str,
    headers: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(url=url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exchange response must be an object")
    return payload


@dataclass(frozen=True)
class _ExchangeHttpError(RuntimeError):
    status_code: int
    payload: dict[str, Any]


def _binance_signed_request(
    *,
    method: str,
    base_url: str,
    path: str,
    params: dict[str, str],
    api_key: str,
    api_secret: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    signed = {
        **params,
        "recvWindow": _RECV_WINDOW,
        "timestamp": str(int(time.time() * 1000)),
    }
    query = urllib.parse.urlencode(signed)
    signature = hmac.new(
        api_secret.encode("utf-8"),
        query.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    request = urllib.request.Request(
        url=f"{base_url}{path}?{query}&signature={signature}",
        headers={"X-MBX-APIKEY": api_key},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        raise _ExchangeHttpError(
            status_code=exc.code,
            payload=_http_error_payload(error=exc),
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("exchange response must be an object")
    return payload


def _http_error_payload(*, error: urllib.error.HTTPError) -> dict[str, Any]:
    try:
        payload = json.loads(error.read().decode("utf-8") or "{}")
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _binance_auth_or_permission_error(*, error: _ExchangeHttpError) -> bool:
    code = error.payload.get("code")
    message = str(error.payload.get("msg") or "").lower()
    return (
        error.status_code in {401, 403}
        or code in {-1022, -2014, -2015}
        or "api-key" in message
        or "signature" in message
        or "permission" in message
    )


def _invalid_credentials_from_http(
    *,
    status_code: int,
    exchange: str,
    requested_permissions: str,
) -> ExchangeCredentialValidationResult:
    requested = _requested_permissions(value=requested_permissions)
    if status_code in {400, 401, 403}:
        return ExchangeCredentialValidationResult(
            status="invalid_credentials",
            reason=f"exchange_rejected_credentials_{status_code}",
            ip_restriction_status="unknown",
            permission_summary=_permission_summary(
                base={"exchange": exchange},
                requested_permissions=requested,
                exchange_permissions="unknown",
                effective_permissions="none",
                permission_warnings=(),
            ),
        )
    return ExchangeCredentialValidationResult(
        status="invalid_credentials",
        reason="exchange_request_failed",
        ip_restriction_status="unknown",
        permission_summary=_permission_summary(
            base={"exchange": exchange},
            requested_permissions=requested,
            exchange_permissions="unknown",
            effective_permissions="none",
            permission_warnings=(),
        ),
    )


def _with_permission_policy(
    *,
    status: ExchangeValidationStatus,
    reason: str,
    ip_restriction_status: str,
    base_summary: dict[str, object],
    requested_permissions: RequestedPermissions,
    exchange_permissions: ExchangePermissions,
    account_mode: str | None = None,
) -> ExchangeCredentialValidationResult:
    effective_permissions: EffectivePermissions = "none"
    permission_warnings: tuple[PermissionWarning, ...] = ()
    resolved_status = status
    resolved_reason = reason
    if status in {"valid_readonly", "valid_trade_enabled"}:
        if requested_permissions == "trade" and exchange_permissions == "read":
            resolved_status = "permission_mismatch"
            resolved_reason = "requested_trade_but_exchange_readonly"
            effective_permissions = "read"
        elif requested_permissions == "read" and exchange_permissions == "trade":
            effective_permissions = "read"
            permission_warnings = ("exchange_permissions_exceed_requested",)
        else:
            effective_permissions = "trade" if exchange_permissions == "trade" else "read"
    return ExchangeCredentialValidationResult(
        status=resolved_status,
        reason=resolved_reason,
        ip_restriction_status=ip_restriction_status,
        account_mode=account_mode,
        permission_summary=_permission_summary(
            base=base_summary,
            requested_permissions=requested_permissions,
            exchange_permissions=exchange_permissions,
            effective_permissions=effective_permissions,
            permission_warnings=permission_warnings,
        ),
    )


def _permission_summary(
    *,
    base: dict[str, object],
    requested_permissions: RequestedPermissions | str,
    exchange_permissions: ExchangePermissions,
    effective_permissions: EffectivePermissions,
    permission_warnings: tuple[PermissionWarning, ...],
) -> dict[str, object]:
    return {
        **base,
        "permissions": requested_permissions,
        "requested_permissions": requested_permissions,
        "exchange_permissions": exchange_permissions,
        "effective_permissions": effective_permissions,
        "permission_warnings": list(permission_warnings),
    }


def _requested_permissions(*, value: str) -> RequestedPermissions:
    return "trade" if value == "trade" else "read"


def _ip_status(*, payload: dict[str, Any], environment: str) -> str:
    restricted = bool(payload.get("ipRestrict"))
    if restricted:
        return "restricted"
    if environment == "mainnet":
        return "missing_mainnet_restriction"
    return "not_restricted_testnet"


def _binance_withdraw_or_transfer_enabled(*, payload: dict[str, Any]) -> bool:
    return any(
        bool(payload.get(field))
        for field in (
            "enableWithdrawals",
            "enableInternalTransfer",
            "permitsUniversalTransfer",
        )
    )


def _binance_trade_enabled(*, payload: dict[str, Any]) -> bool:
    return any(
        bool(payload.get(field))
        for field in (
            "enableMargin",
            "enableFutures",
            "enableVanillaOptions",
            "enableSpotAndMarginTrading",
            "enableFixApiTrade",
        )
    )


def _binance_summary(*, payload: dict[str, Any]) -> dict[str, object]:
    return {
        "exchange": "binance",
        "read": bool(payload.get("enableReading")),
        "trade": _binance_trade_enabled(payload=payload),
        "withdraw_or_transfer": _binance_withdraw_or_transfer_enabled(payload=payload),
    }


def _binance_spot_demo_summary(
    *,
    payload: dict[str, Any],
    exchange_permissions: ExchangePermissions,
) -> dict[str, object]:
    permissions = payload.get("permissions")
    normalized_permissions = (
        sorted({str(item) for item in permissions}) if isinstance(permissions, list) else []
    )
    return {
        "exchange": "binance",
        "market_type": "spot",
        "read": True,
        "trade": exchange_permissions == "trade",
        "withdraw_or_transfer": False,
        "account_can_trade": bool(payload.get("canTrade")),
        "account_type": str(payload.get("accountType") or "spot").lower(),
        "spot_permissions": normalized_permissions,
    }


def _bybit_ip_status(*, result: dict[str, Any], environment: str) -> str:
    ips = result.get("ips")
    has_ips = isinstance(ips, list) and len(ips) > 0
    if has_ips:
        return "restricted"
    if environment == "mainnet":
        return "missing_mainnet_restriction"
    return "not_restricted_testnet"


def _bybit_transfer_enabled(*, result: dict[str, Any]) -> bool:
    permissions = result.get("permissions")
    if not isinstance(permissions, dict):
        return False
    wallet = permissions.get("Wallet")
    if not isinstance(wallet, list):
        return False
    return any(isinstance(item, str) and "Transfer" in item for item in wallet)


def _bybit_account_mode(*, result: dict[str, Any]) -> str | None:
    uta = result.get("uta")
    if uta is None:
        return None
    if int(uta) in {0, 1}:
        return "unified" if int(uta) == 1 else "classic"
    return "unsupported"


def _bybit_summary(*, result: dict[str, Any]) -> dict[str, object]:
    return {
        "exchange": "bybit",
        "readonly": int(result.get("readOnly", 0)) == 1,
        "transfer": _bybit_transfer_enabled(result=result),
    }


__all__ = [
    "BinanceExchangeCredentialValidator",
    "BybitExchangeCredentialValidator",
    "HttpExchangeCredentialValidator",
    "normalize_binance_api_restrictions",
    "normalize_binance_futures_testnet_account",
    "normalize_binance_spot_demo_account",
    "normalize_bybit_api_key_info",
]
