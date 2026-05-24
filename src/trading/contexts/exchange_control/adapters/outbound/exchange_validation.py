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
    ExchangeCredentialValidationRequest,
    ExchangeCredentialValidationResult,
    ExchangeCredentialValidator,
)

_BINANCE_MAINNET_URL = "https://api.binance.com"
_BINANCE_TESTNET_URL = "https://testnet.binance.vision"
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
        base_url = (
            _BINANCE_TESTNET_URL
            if request.environment == "testnet"
            else _BINANCE_MAINNET_URL
        )
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
            return _invalid_credentials_from_http(status_code=exc.code)
        except (OSError, ValueError):
            return ExchangeCredentialValidationResult(
                status="invalid_credentials",
                reason="exchange_request_failed",
                ip_restriction_status="unknown",
                permission_summary={"exchange": "binance"},
            )
        return normalize_binance_api_restrictions(
            payload=payload,
            environment=request.environment,
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
            return _invalid_credentials_from_http(status_code=exc.code)
        except (OSError, ValueError):
            return ExchangeCredentialValidationResult(
                status="invalid_credentials",
                reason="exchange_request_failed",
                ip_restriction_status="unknown",
                permission_summary={"exchange": "bybit"},
            )
        return normalize_bybit_api_key_info(
            payload=payload,
            environment=request.environment,
        )


def normalize_binance_api_restrictions(
    *,
    payload: dict[str, Any],
    environment: str,
) -> ExchangeCredentialValidationResult:
    if not bool(payload.get("enableReading")):
        return ExchangeCredentialValidationResult(
            status="invalid_permissions",
            reason="reading_permission_disabled",
            ip_restriction_status=_ip_status(payload=payload, environment=environment),
            permission_summary=_binance_summary(payload=payload),
        )
    if _binance_withdraw_or_transfer_enabled(payload=payload):
        return ExchangeCredentialValidationResult(
            status="invalid_permissions",
            reason="withdraw_or_transfer_enabled",
            ip_restriction_status=_ip_status(payload=payload, environment=environment),
            permission_summary=_binance_summary(payload=payload),
        )
    if _ip_status(payload=payload, environment=environment) == "missing_mainnet_restriction":
        return ExchangeCredentialValidationResult(
            status="invalid_ip_restriction",
            reason="mainnet_ip_restriction_missing",
            ip_restriction_status="missing_mainnet_restriction",
            permission_summary=_binance_summary(payload=payload),
        )
    if bool(payload.get("enablePortfolioMarginTrading")):
        return ExchangeCredentialValidationResult(
            status="unsupported_account_mode",
            reason="portfolio_margin_enabled",
            ip_restriction_status=_ip_status(payload=payload, environment=environment),
            account_mode="portfolio_margin",
            permission_summary=_binance_summary(payload=payload),
        )
    if _binance_trade_enabled(payload=payload):
        return ExchangeCredentialValidationResult(
            status="valid_trade_enabled",
            reason="trade_permission_detected",
            ip_restriction_status=_ip_status(payload=payload, environment=environment),
            permission_summary=_binance_summary(payload=payload),
        )
    return ExchangeCredentialValidationResult(
        status="valid_readonly",
        reason="readonly_permission_detected",
        ip_restriction_status=_ip_status(payload=payload, environment=environment),
        permission_summary=_binance_summary(payload=payload),
    )


def normalize_bybit_api_key_info(
    *,
    payload: dict[str, Any],
    environment: str,
) -> ExchangeCredentialValidationResult:
    if int(payload.get("retCode", 0)) != 0:
        return ExchangeCredentialValidationResult(
            status="invalid_credentials",
            reason="exchange_rejected_credentials",
            ip_restriction_status="unknown",
            permission_summary={"exchange": "bybit"},
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        return ExchangeCredentialValidationResult(
            status="invalid_credentials",
            reason="exchange_response_invalid",
            ip_restriction_status="unknown",
            permission_summary={"exchange": "bybit"},
        )
    account_mode = _bybit_account_mode(result=result)
    if account_mode == "unsupported":
        return ExchangeCredentialValidationResult(
            status="unsupported_account_mode",
            reason="unsupported_account_mode",
            ip_restriction_status=_bybit_ip_status(result=result, environment=environment),
            account_mode=account_mode,
            permission_summary=_bybit_summary(result=result),
        )
    read_only = int(result.get("readOnly", 0)) == 1
    ip_status = _bybit_ip_status(result=result, environment=environment)
    if not read_only and _bybit_transfer_enabled(result=result):
        return ExchangeCredentialValidationResult(
            status="invalid_permissions",
            reason="transfer_permission_enabled",
            ip_restriction_status=ip_status,
            account_mode=account_mode,
            permission_summary=_bybit_summary(result=result),
        )
    if ip_status == "missing_mainnet_restriction":
        return ExchangeCredentialValidationResult(
            status="invalid_ip_restriction",
            reason="mainnet_ip_restriction_missing",
            ip_restriction_status=ip_status,
            account_mode=account_mode,
            permission_summary=_bybit_summary(result=result),
        )
    if read_only:
        return ExchangeCredentialValidationResult(
            status="valid_readonly",
            reason="readonly_permission_detected",
            ip_restriction_status=ip_status,
            account_mode=account_mode,
            permission_summary=_bybit_summary(result=result),
        )
    return ExchangeCredentialValidationResult(
        status="valid_trade_enabled",
        reason="write_permission_detected",
        ip_restriction_status=ip_status,
        account_mode=account_mode,
        permission_summary=_bybit_summary(result=result),
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


def _invalid_credentials_from_http(*, status_code: int) -> ExchangeCredentialValidationResult:
    if status_code in {400, 401, 403}:
        return ExchangeCredentialValidationResult(
            status="invalid_credentials",
            reason=f"exchange_rejected_credentials_{status_code}",
            ip_restriction_status="unknown",
        )
    return ExchangeCredentialValidationResult(
        status="invalid_credentials",
        reason="exchange_request_failed",
        ip_restriction_status="unknown",
    )


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
    "normalize_bybit_api_key_info",
]
