from __future__ import annotations

import json
import urllib.request

from trading.contexts.exchange_control.adapters.outbound.exchange_validation import (
    BinanceExchangeCredentialValidator,
    normalize_binance_api_restrictions,
    normalize_binance_futures_testnet_account,
    normalize_bybit_api_key_info,
)
from trading.contexts.exchange_control.application.validation import (
    ExchangeCredentialPlaintext,
    ExchangeCredentialValidationRequest,
    ExchangeCredentialValidationResult,
)


def test_exchange_credential_plaintext_redacts_repr() -> None:
    secret = ExchangeCredentialPlaintext(
        api_key="ROEHUB_TEST_BINANCE_READONLY_API_KEY",
        api_secret="TEST_SECRET",
    )

    assert "TEST_SECRET" not in repr(secret)
    assert "ROEHUB_TEST_BINANCE_READONLY_API_KEY" not in repr(secret)


def _permission_summary(
    result: ExchangeCredentialValidationResult,
) -> dict[str, object]:
    assert result.permission_summary is not None
    return result.permission_summary


def test_binance_validation_status_mapping() -> None:
    readonly = normalize_binance_api_restrictions(
        payload={
            "ipRestrict": False,
            "enableReading": True,
            "enableWithdrawals": False,
            "enableInternalTransfer": False,
            "permitsUniversalTransfer": False,
            "enableMargin": False,
            "enableFutures": False,
            "enableVanillaOptions": False,
            "enableSpotAndMarginTrading": False,
            "enableFixApiTrade": False,
            "enablePortfolioMarginTrading": False,
        },
        environment="testnet",
    )
    trade = normalize_binance_api_restrictions(
        payload={
            "ipRestrict": True,
            "enableReading": True,
            "enableWithdrawals": False,
            "enableInternalTransfer": False,
            "permitsUniversalTransfer": False,
            "enableSpotAndMarginTrading": True,
        },
        environment="mainnet",
    )
    withdrawal = normalize_binance_api_restrictions(
        payload={
            "ipRestrict": True,
            "enableReading": True,
            "enableWithdrawals": True,
        },
        environment="mainnet",
    )
    missing_ip = normalize_binance_api_restrictions(
        payload={
            "ipRestrict": False,
            "enableReading": True,
            "enableWithdrawals": False,
            "enableInternalTransfer": False,
            "permitsUniversalTransfer": False,
        },
        environment="mainnet",
    )
    portfolio_margin = normalize_binance_api_restrictions(
        payload={
            "ipRestrict": True,
            "enableReading": True,
            "enableWithdrawals": False,
            "enableInternalTransfer": False,
            "permitsUniversalTransfer": False,
            "enablePortfolioMarginTrading": True,
        },
        environment="mainnet",
    )

    assert readonly.status == "valid_readonly"
    assert trade.status == "valid_trade_enabled"
    assert withdrawal.status == "invalid_permissions"
    assert missing_ip.status == "invalid_ip_restriction"
    assert portfolio_margin.status == "unsupported_account_mode"


def test_binance_permission_truth_table() -> None:
    readonly_payload = {
        "ipRestrict": False,
        "enableReading": True,
        "enableWithdrawals": False,
        "enableInternalTransfer": False,
        "permitsUniversalTransfer": False,
        "enableMargin": False,
        "enableFutures": False,
        "enableVanillaOptions": False,
        "enableSpotAndMarginTrading": False,
        "enableFixApiTrade": False,
        "enablePortfolioMarginTrading": False,
    }
    trade_payload = {
        **readonly_payload,
        "ipRestrict": True,
        "enableSpotAndMarginTrading": True,
    }
    withdrawal_payload = {
        **readonly_payload,
        "ipRestrict": True,
        "enableWithdrawals": True,
    }

    read_readonly = normalize_binance_api_restrictions(
        payload=readonly_payload,
        environment="testnet",
        requested_permissions="read",
    )
    read_trade = normalize_binance_api_restrictions(
        payload=trade_payload,
        environment="mainnet",
        requested_permissions="read",
    )
    trade_trade = normalize_binance_api_restrictions(
        payload=trade_payload,
        environment="mainnet",
        requested_permissions="trade",
    )
    trade_readonly = normalize_binance_api_restrictions(
        payload=readonly_payload,
        environment="testnet",
        requested_permissions="trade",
    )
    withdrawal = normalize_binance_api_restrictions(
        payload=withdrawal_payload,
        environment="mainnet",
        requested_permissions="trade",
    )

    assert read_readonly.status == "valid_readonly"
    assert read_readonly.permission_summary == {
        **_permission_summary(read_readonly),
        "requested_permissions": "read",
        "exchange_permissions": "read",
        "effective_permissions": "read",
        "permission_warnings": [],
    }
    assert read_trade.status == "valid_trade_enabled"
    assert read_trade.permission_summary == {
        **_permission_summary(read_trade),
        "requested_permissions": "read",
        "exchange_permissions": "trade",
        "effective_permissions": "read",
        "permission_warnings": ["exchange_permissions_exceed_requested"],
    }
    assert trade_trade.status == "valid_trade_enabled"
    assert trade_trade.permission_summary == {
        **_permission_summary(trade_trade),
        "requested_permissions": "trade",
        "exchange_permissions": "trade",
        "effective_permissions": "trade",
        "permission_warnings": [],
    }
    assert trade_readonly.status == "permission_mismatch"
    assert trade_readonly.reason == "requested_trade_but_exchange_readonly"
    assert trade_readonly.permission_summary == {
        **_permission_summary(trade_readonly),
        "requested_permissions": "trade",
        "exchange_permissions": "read",
        "effective_permissions": "read",
        "permission_warnings": [],
    }
    assert withdrawal.status == "invalid_permissions"
    assert withdrawal.permission_summary == {
        **_permission_summary(withdrawal),
        "exchange_permissions": "withdraw_or_transfer",
        "effective_permissions": "none",
    }


def test_binance_futures_testnet_validation_status_mapping() -> None:
    trade = normalize_binance_futures_testnet_account(
        payload={
            "canTrade": True,
            "canWithdraw": True,
            "multiAssetsMargin": False,
        },
        requested_permissions="trade",
    )
    readonly = normalize_binance_futures_testnet_account(
        payload={
            "canTrade": False,
            "canWithdraw": True,
            "multiAssetsMargin": False,
        },
        requested_permissions="trade",
    )

    assert trade.status == "valid_trade_enabled"
    assert trade.reason == "trade_permission_detected"
    assert trade.ip_restriction_status == "not_restricted_testnet"
    assert trade.permission_summary == {
        **_permission_summary(trade),
        "market_type": "futures",
        "exchange_permissions": "trade",
        "effective_permissions": "trade",
        "withdraw_or_transfer": False,
    }
    assert readonly.status == "permission_mismatch"
    assert readonly.reason == "requested_trade_but_exchange_readonly"
    assert readonly.permission_summary == {
        **_permission_summary(readonly),
        "exchange_permissions": "read",
        "effective_permissions": "read",
    }


class _JsonResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def __enter__(self) -> "_JsonResponse":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_binance_futures_testnet_validator_uses_fapi_account_endpoint(
    monkeypatch: object,
) -> None:
    calls: list[str] = []

    def fake_urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> _JsonResponse:
        _ = timeout
        calls.append(request.full_url)
        return _JsonResponse(
            {
                "canTrade": True,
                "canWithdraw": True,
                "multiAssetsMargin": False,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)  # type: ignore[attr-defined]

    result = BinanceExchangeCredentialValidator().validate(
        request=ExchangeCredentialValidationRequest(
            exchange_name="binance",
            market_type="futures",
            environment="testnet",
            requested_permissions="trade",
            credential=ExchangeCredentialPlaintext(
                api_key="testnet-futures-key",
                api_secret="testnet-futures-secret",
            ),
        )
    )

    assert result.status == "valid_trade_enabled"
    assert len(calls) == 1
    assert calls[0].startswith("https://testnet.binancefuture.com/fapi/v2/account?")
    assert "/sapi/v1/account/apiRestrictions" not in calls[0]


def test_bybit_validation_status_mapping() -> None:
    readonly = normalize_bybit_api_key_info(
        payload={
            "retCode": 0,
            "result": {"readOnly": 1, "permissions": {}, "ips": [], "uta": 1},
        },
        environment="testnet",
    )
    trade = normalize_bybit_api_key_info(
        payload={
            "retCode": 0,
            "result": {
                "readOnly": 0,
                "permissions": {"Spot": ["SpotTrade"], "Wallet": []},
                "ips": ["203.0.113.10"],
                "uta": 1,
            },
        },
        environment="mainnet",
    )
    transfer = normalize_bybit_api_key_info(
        payload={
            "retCode": 0,
            "result": {
                "readOnly": 0,
                "permissions": {"Wallet": ["AccountTransfer"]},
                "ips": ["203.0.113.10"],
                "uta": 1,
            },
        },
        environment="mainnet",
    )
    invalid = normalize_bybit_api_key_info(
        payload={"retCode": 10003, "retMsg": "raw exchange body with TEST_SECRET"},
        environment="testnet",
    )
    missing_ip = normalize_bybit_api_key_info(
        payload={
            "retCode": 0,
            "result": {"readOnly": 1, "permissions": {}, "ips": [], "uta": 1},
        },
        environment="mainnet",
    )
    unsupported = normalize_bybit_api_key_info(
        payload={
            "retCode": 0,
            "result": {"readOnly": 1, "permissions": {}, "ips": [], "uta": 9},
        },
        environment="testnet",
    )

    assert readonly.status == "valid_readonly"
    assert trade.status == "valid_trade_enabled"
    assert transfer.status == "invalid_permissions"
    assert invalid.status == "invalid_credentials"
    assert "TEST_SECRET" not in invalid.reason
    assert missing_ip.status == "invalid_ip_restriction"
    assert unsupported.status == "unsupported_account_mode"


def test_bybit_permission_truth_table() -> None:
    readonly_payload = {
        "retCode": 0,
        "result": {"readOnly": 1, "permissions": {}, "ips": [], "uta": 1},
    }
    trade_payload = {
        "retCode": 0,
        "result": {
            "readOnly": 0,
            "permissions": {"Spot": ["SpotTrade"], "Wallet": []},
            "ips": ["203.0.113.10"],
            "uta": 1,
        },
    }
    transfer_payload = {
        "retCode": 0,
        "result": {
            "readOnly": 0,
            "permissions": {"Wallet": ["AccountTransfer"]},
            "ips": ["203.0.113.10"],
            "uta": 1,
        },
    }

    read_readonly = normalize_bybit_api_key_info(
        payload=readonly_payload,
        environment="testnet",
        requested_permissions="read",
    )
    read_trade = normalize_bybit_api_key_info(
        payload=trade_payload,
        environment="mainnet",
        requested_permissions="read",
    )
    trade_trade = normalize_bybit_api_key_info(
        payload=trade_payload,
        environment="mainnet",
        requested_permissions="trade",
    )
    trade_readonly = normalize_bybit_api_key_info(
        payload=readonly_payload,
        environment="testnet",
        requested_permissions="trade",
    )
    transfer = normalize_bybit_api_key_info(
        payload=transfer_payload,
        environment="mainnet",
        requested_permissions="trade",
    )
    invalid = normalize_bybit_api_key_info(
        payload={"retCode": 10003, "retMsg": "raw exchange body with TEST_SECRET"},
        environment="testnet",
        requested_permissions="trade",
    )

    assert read_readonly.status == "valid_readonly"
    assert read_readonly.permission_summary == {
        **_permission_summary(read_readonly),
        "requested_permissions": "read",
        "exchange_permissions": "read",
        "effective_permissions": "read",
        "permission_warnings": [],
    }
    assert read_trade.status == "valid_trade_enabled"
    assert read_trade.permission_summary == {
        **_permission_summary(read_trade),
        "requested_permissions": "read",
        "exchange_permissions": "trade",
        "effective_permissions": "read",
        "permission_warnings": ["exchange_permissions_exceed_requested"],
    }
    assert trade_trade.status == "valid_trade_enabled"
    assert trade_trade.permission_summary == {
        **_permission_summary(trade_trade),
        "requested_permissions": "trade",
        "exchange_permissions": "trade",
        "effective_permissions": "trade",
        "permission_warnings": [],
    }
    assert trade_readonly.status == "permission_mismatch"
    assert trade_readonly.reason == "requested_trade_but_exchange_readonly"
    assert trade_readonly.permission_summary == {
        **_permission_summary(trade_readonly),
        "requested_permissions": "trade",
        "exchange_permissions": "read",
        "effective_permissions": "read",
        "permission_warnings": [],
    }
    assert transfer.status == "invalid_permissions"
    assert transfer.permission_summary == {
        **_permission_summary(transfer),
        "exchange_permissions": "withdraw_or_transfer",
        "effective_permissions": "none",
    }
    assert invalid.status == "invalid_credentials"
    assert invalid.permission_summary == {
        **_permission_summary(invalid),
        "exchange_permissions": "unknown",
        "effective_permissions": "none",
    }
    assert "TEST_SECRET" not in invalid.reason
