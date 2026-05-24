from __future__ import annotations

from trading.contexts.exchange_control.adapters.outbound.exchange_validation import (
    normalize_binance_api_restrictions,
    normalize_bybit_api_key_info,
)
from trading.contexts.exchange_control.application.validation import (
    ExchangeCredentialPlaintext,
)


def test_exchange_credential_plaintext_redacts_repr() -> None:
    secret = ExchangeCredentialPlaintext(
        api_key="ROEHUB_TEST_BINANCE_READONLY_API_KEY",
        api_secret="TEST_SECRET",
    )

    assert "TEST_SECRET" not in repr(secret)
    assert "ROEHUB_TEST_BINANCE_READONLY_API_KEY" not in repr(secret)


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
