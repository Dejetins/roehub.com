from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.market_data.adapters.outbound.config.whitelist import (
    load_enabled_instruments_from_csv,
)


def test_whitelist_last_win_and_filters_disabled(tmp_path: Path) -> None:
    p = tmp_path / "whitelist.csv"
    p.write_text(
        "market_id,symbol,is_enabled\n"
        "1,BTCUSDT,1\n"
        "1,BTCUSDT,0\n"
        "1,ETHUSDT,1\n",
        encoding="utf-8",
    )

    instruments = load_enabled_instruments_from_csv(p)
    # BTCUSDT disabled by last-win, only ETHUSDT remains
    assert len(instruments) == 1
    assert str(instruments[0]) == "1:ETHUSDT"


def test_whitelist_requires_columns(tmp_path: Path) -> None:
    p = tmp_path / "whitelist.csv"
    p.write_text("a,b,c\n1,BTCUSDT,1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_enabled_instruments_from_csv(p)
