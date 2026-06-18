from __future__ import annotations

from pathlib import Path

from scripts.rl_trading.resolve_stage04a_binance_futures_universe import (
    EXCLUDED_REASON,
    build_manifest,
    load_train_symbols_json,
    plan_whitelist_additions,
    update_whitelist_csv,
)


def test_stage04a_manifest_accepts_only_current_trading_usdt_perpetuals() -> None:
    manifest = build_manifest(
        train_symbols=[
            "BTCUSDT",
            "ETHUSDC",
            "ETHUSDT_230929",
            "OLDUSDT",
            "MISSINGUSDT",
        ],
        exchange_info={
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "onboardDate": 1577836800000,
                },
                {
                    "symbol": "ETHUSDC",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDC",
                    "onboardDate": 1577836800000,
                },
                {
                    "symbol": "ETHUSDT_230929",
                    "status": "TRADING",
                    "contractType": "CURRENT_QUARTER",
                    "quoteAsset": "USDT",
                    "onboardDate": 1577836800000,
                },
                {
                    "symbol": "OLDUSDT",
                    "status": "BREAK",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "onboardDate": 1577836800000,
                },
            ]
        },
        market_id=2,
    )

    assert manifest["accepted_symbols"] == ["BTCUSDT"]
    assert manifest["accepted_count"] == 1
    assert manifest["excluded_count"] == 4
    assert manifest["excluded_symbols"] == [
        {"symbol": "ETHUSDC", "reason": EXCLUDED_REASON},
        {"symbol": "ETHUSDT_230929", "reason": EXCLUDED_REASON},
        {"symbol": "MISSINGUSDT", "reason": EXCLUDED_REASON},
        {"symbol": "OLDUSDT", "reason": EXCLUDED_REASON},
    ]


def test_stage04a_source_lower_bound_uses_exchange_onboard_date_when_later() -> None:
    manifest = build_manifest(
        train_symbols=["NEWUSDT"],
        exchange_info={
            "symbols": [
                {
                    "symbol": "NEWUSDT",
                    "status": "TRADING",
                    "contractType": "PERPETUAL",
                    "quoteAsset": "USDT",
                    "onboardDate": 1640995200000,
                },
            ]
        },
        market_id=2,
        required_source_window_start_utc="2020-01-13T22:30:00Z",
    )

    assert manifest["accepted_symbol_source_windows"] == [
        {
            "symbol": "NEWUSDT",
            "exchange_onboard_utc": "2022-01-01T00:00:00Z",
            "source_lower_bound_utc": "2022-01-01T00:00:00Z",
        }
    ]


def test_stage04a_whitelist_update_appends_missing_or_disabled_accepted_symbols(
    tmp_path: Path,
) -> None:
    whitelist = tmp_path / "whitelist.csv"
    whitelist.write_text(
        "market_id,symbol,is_enabled\n"
        "1,AAVEUSDT,1\n"
        "2,BTCUSDT,1\n"
        "2,OLDUSDT,0\n"
        "3,AAVEUSDT,1\n",
        encoding="utf-8",
    )
    rows = [
        {"market_id": "1", "symbol": "AAVEUSDT", "is_enabled": "1"},
        {"market_id": "2", "symbol": "BTCUSDT", "is_enabled": "1"},
        {"market_id": "2", "symbol": "OLDUSDT", "is_enabled": "0"},
        {"market_id": "3", "symbol": "AAVEUSDT", "is_enabled": "1"},
    ]

    assert plan_whitelist_additions(
        rows=rows,
        market_id=2,
        accepted_symbols=["OLDUSDT", "AAVEUSDT", "BTCUSDT"],
    ) == ["AAVEUSDT", "OLDUSDT"]

    added = update_whitelist_csv(
        path=whitelist,
        market_id=2,
        accepted_symbols=["OLDUSDT", "AAVEUSDT", "BTCUSDT"],
    )

    assert added == ["AAVEUSDT", "OLDUSDT"]
    assert whitelist.read_text(encoding="utf-8").splitlines() == [
        "market_id,symbol,is_enabled",
        "1,AAVEUSDT,1",
        "2,BTCUSDT,1",
        "2,OLDUSDT,0",
        "3,AAVEUSDT,1",
        "2,AAVEUSDT,1",
        "2,OLDUSDT,1",
    ]


def test_stage04a_can_load_sanitized_train_symbol_list(tmp_path: Path) -> None:
    train_symbols = tmp_path / "train_symbols.json"
    train_symbols.write_text('["btcusdt", " ETHUSDT ", "BTCUSDT"]', encoding="utf-8")

    assert load_train_symbols_json(train_symbols) == ["BTCUSDT", "ETHUSDT"]
