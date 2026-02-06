from __future__ import annotations

import csv
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from trading.shared_kernel.primitives.instrument_id import InstrumentId
from trading.shared_kernel.primitives.market_id import MarketId
from trading.shared_kernel.primitives.symbol import Symbol

log = logging.getLogger(__name__)


_REQUIRED_COLUMNS = ("market_id", "symbol", "is_enabled")


@dataclass(frozen=True, slots=True)
class WhitelistRow:
    instrument_id: InstrumentId
    is_enabled: bool


def load_enabled_instruments_from_csv(path: str | Path) -> list[InstrumentId]:
    """
    Load whitelist CSV and return only enabled InstrumentId.

    Contract:
    - required columns: market_id,symbol,is_enabled
    - symbol: strip + upper (via Symbol)
    - is_enabled: 0/1 only
    - duplicates (market_id,symbol): last-win with warning
    - disabled rows are excluded from returned list
    """
    rows = _load_whitelist_rows(path)
    return [r.instrument_id for r in rows if r.is_enabled]


def _load_whitelist_rows(path: str | Path) -> list[WhitelistRow]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"whitelist csv not found: {p}")

    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("whitelist csv must have a header row")

        missing = [c for c in _REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"whitelist csv missing required columns: {missing}; got columns={reader.fieldnames}")  # noqa: E501

        # last-win preserving last occurrence ordering:
        # if duplicate key occurs, remove then reinsert to move it to the end.
        acc: "OrderedDict[tuple[int, str], WhitelistRow]" = OrderedDict()

        for idx, raw in enumerate(reader, start=2):  # header is line 1
            try:
                market_id = MarketId(_parse_int(raw["market_id"], line=idx, field="market_id"))
                symbol = Symbol(_parse_str(raw["symbol"], line=idx, field="symbol"))
                is_enabled = _parse_enabled(raw["is_enabled"], line=idx)
                ins = InstrumentId(market_id=market_id, symbol=symbol)
                key = (market_id.value, str(symbol))
            except Exception as e:
                raise ValueError(f"invalid whitelist row at line {idx}: {e}") from e

            if key in acc:
                log.warning("duplicate whitelist key %s at line %s: last-win applied", key, idx)
                del acc[key]
            acc[key] = WhitelistRow(instrument_id=ins, is_enabled=is_enabled)

        return list(acc.values())


def _parse_str(v: object, *, line: int, field: str) -> str:
    if v is None:
        raise ValueError(f"{field} is required")
    if not isinstance(v, str):
        raise ValueError(f"{field} must be a string, got {type(v).__name__}")
    s = v.strip()
    if not s:
        raise ValueError(f"{field} must be non-empty")
    return s


def _parse_int(v: object, *, line: int, field: str) -> int:
    if v is None:
        raise ValueError(f"{field} is required")
    if isinstance(v, str):
        s = v.strip()
        if not s:
            raise ValueError(f"{field} must be non-empty")
        try:
            return int(s)
        except ValueError as e:
            raise ValueError(f"{field} must be an int, got {v!r}") from e
    if isinstance(v, bool):
        raise ValueError(f"{field} must be an int, got bool")
    if isinstance(v, int):
        return v
    raise ValueError(f"{field} must be an int, got {type(v).__name__}")


def _parse_enabled(v: object, *, line: int) -> bool:
    if v is None:
        raise ValueError("is_enabled is required")
    if isinstance(v, str):
        s = v.strip()
        if s not in ("0", "1"):
            raise ValueError(f"is_enabled must be '0' or '1', got {v!r}")
        return s == "1"
    if isinstance(v, int) and not isinstance(v, bool):
        if v not in (0, 1):
            raise ValueError(f"is_enabled must be 0 or 1, got {v!r}")
        return v == 1
    raise ValueError(f"is_enabled must be 0/1, got {type(v).__name__}")
