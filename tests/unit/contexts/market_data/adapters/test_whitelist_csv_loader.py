from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.market_data.adapters.outbound.config.whitelist import (
    load_whitelist_rows_from_csv,
)


def test_whitelist_last_win_and_preserves_disabled_rows(tmp_path: Path) -> None:
    p = tmp_path / "whitelist.csv"
    p.write_text(
        "market_id,symbol,is_enabled\n"
        "1,BTCUSDT,1\n"
        "1,BTCUSDT,0\n"
        "1,ETHUSDT,1\n",
        encoding="utf-8",
    )

    rows = load_whitelist_rows_from_csv(p)
    assert [(str(row.instrument_id), row.is_enabled) for row in rows] == [
        ("1:BTCUSDT", False),
        ("1:ETHUSDT", True),
    ]


def test_whitelist_requires_columns(tmp_path: Path) -> None:
    p = tmp_path / "whitelist.csv"
    p.write_text("a,b,c\n1,BTCUSDT,1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_whitelist_rows_from_csv(p)
