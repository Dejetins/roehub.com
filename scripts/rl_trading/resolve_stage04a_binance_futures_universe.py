from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.market_data.adapters.outbound.config.runtime_config import (  # noqa: E402
    load_market_data_runtime_config,
)

BINANCE_FUTURES_MARKET_CODE = "binance:futures"
EXCLUDED_REASON = "excluded_not_currently_trading_or_not_usdt_perpetual"
DEFAULT_TRAIN_NPZ = Path(
    "/opt/roehub/state/rl_trading/hf_reproducibility/dataset/"
    "ResearchRL/open-rl-trading-binance-dataset/train_data.npz"
)
DEFAULT_OUTPUT_JSON = Path(
    "/opt/roehub/state/rl_trading/stage04a_binance_futures_universe/"
    "stage04a_universe_manifest.json"
)
HF_TRAIN_REQUIRED_SOURCE_START_UTC = "2020-01-13T22:30:00Z"


@dataclass(frozen=True, slots=True)
class CurrentFuturesSymbol:
    symbol: str
    onboard_utc: str | None


def load_hf_train_symbols(train_npz: Path) -> list[str]:
    with np.load(train_npz, allow_pickle=True) as data:
        keys_map_raw = data["_keys_map_"].item()
    keys_map = cast(Mapping[str, tuple[Any, Any]], keys_map_raw)
    return sorted({str(value[0]).strip().upper() for value in keys_map.values()})


def load_exchange_info(path: Path | None, *, base_url: str, timeout_s: float) -> Mapping[str, Any]:
    if path is not None:
        return cast(Mapping[str, Any], json.loads(path.read_text(encoding="utf-8")))

    url = base_url.rstrip("/") + "/fapi/v1/exchangeInfo"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "roehub-stage04a-universe-resolver/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return cast(Mapping[str, Any], json.load(response))


def current_trading_usdt_perpetuals(
    exchange_info: Mapping[str, Any],
) -> dict[str, CurrentFuturesSymbol]:
    rows = exchange_info.get("symbols")
    if not isinstance(rows, list):
        raise ValueError("Binance exchangeInfo payload is missing symbols list")

    out: dict[str, CurrentFuturesSymbol] = {}
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        if item.get("status") != "TRADING":
            continue
        if item.get("contractType") != "PERPETUAL":
            continue
        if item.get("quoteAsset") != "USDT":
            continue
        out[symbol] = CurrentFuturesSymbol(
            symbol=symbol,
            onboard_utc=_onboard_ms_to_utc(item.get("onboardDate")),
        )
    return out


def build_manifest(
    *,
    train_symbols: Sequence[str],
    exchange_info: Mapping[str, Any],
    market_id: int,
    required_source_window_start_utc: str = HF_TRAIN_REQUIRED_SOURCE_START_UTC,
) -> dict[str, Any]:
    metadata_rows = exchange_info.get("symbols")
    if not isinstance(metadata_rows, list):
        raise ValueError("Binance exchangeInfo payload is missing symbols list")

    normalized_train_symbols = sorted({symbol.strip().upper() for symbol in train_symbols})
    current_symbols = current_trading_usdt_perpetuals(exchange_info)
    accepted_symbols = sorted(set(normalized_train_symbols) & set(current_symbols))
    excluded_symbols = sorted(set(normalized_train_symbols) - set(accepted_symbols))
    source_windows = [
        {
            "symbol": symbol,
            "exchange_onboard_utc": current_symbols[symbol].onboard_utc,
            "source_lower_bound_utc": _max_iso_utc(
                required_source_window_start_utc,
                current_symbols[symbol].onboard_utc,
            ),
        }
        for symbol in accepted_symbols
    ]

    return {
        "schema_version": 1,
        "stage": "04A",
        "market": BINANCE_FUTURES_MARKET_CODE,
        "market_id": market_id,
        "excluded_reason": EXCLUDED_REASON,
        "hf_candidate_count": len(normalized_train_symbols),
        "binance_metadata_count": len(metadata_rows),
        "binance_current_trading_usdt_perpetual_count": len(current_symbols),
        "accepted_count": len(accepted_symbols),
        "excluded_count": len(excluded_symbols),
        "accepted_symbols": accepted_symbols,
        "excluded_symbols": [
            {"symbol": symbol, "reason": EXCLUDED_REASON} for symbol in excluded_symbols
        ],
        "source_window_lower_bound_policy": {
            "required_source_window_start_utc": required_source_window_start_utc,
            "rule": "max(required_source_window_start_utc, exchangeInfo.onboardDate)",
        },
        "accepted_symbol_source_windows": source_windows,
        "hashes": {
            "accepted_symbols_sha256": _hash_lines(accepted_symbols),
            "excluded_symbols_sha256": _hash_lines(excluded_symbols),
            "current_trading_usdt_perpetual_symbols_sha256": _hash_lines(
                sorted(current_symbols)
            ),
        },
    }


def update_whitelist_csv(
    *,
    path: Path,
    market_id: int,
    accepted_symbols: Sequence[str],
) -> list[str]:
    rows, fieldnames = _read_whitelist_csv(path)
    additions = plan_whitelist_additions(
        rows=rows,
        market_id=market_id,
        accepted_symbols=accepted_symbols,
    )
    if not additions:
        return []

    next_rows = list(rows)
    for symbol in additions:
        next_rows.append(
            {
                "market_id": str(market_id),
                "symbol": symbol,
                "is_enabled": "1",
            }
        )
    _write_whitelist_csv(path=path, rows=next_rows, fieldnames=fieldnames)
    return additions


def plan_whitelist_additions(
    *,
    rows: Sequence[Mapping[str, str]],
    market_id: int,
    accepted_symbols: Sequence[str],
) -> list[str]:
    effective_enabled: dict[tuple[int, str], bool] = {}
    for row in rows:
        row_market_id = int(row["market_id"])
        symbol = row["symbol"].strip().upper()
        effective_enabled[(row_market_id, symbol)] = row["is_enabled"].strip() == "1"

    additions: list[str] = []
    for symbol in sorted({item.strip().upper() for item in accepted_symbols}):
        if effective_enabled.get((market_id, symbol)) is True:
            continue
        additions.append(symbol)
    return additions


def render_json_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_market_data_runtime_config(args.market_config)
    binance_futures_market = next(
        market for market in cfg.markets if market.market_code == BINANCE_FUTURES_MARKET_CODE
    )
    train_symbols = (
        load_train_symbols_json(args.train_symbols_json)
        if args.train_symbols_json is not None
        else load_hf_train_symbols(args.train_npz)
    )
    exchange_info = load_exchange_info(
        args.exchange_info_json,
        base_url=binance_futures_market.rest.base_url,
        timeout_s=binance_futures_market.rest.timeout_s,
    )
    manifest = build_manifest(
        train_symbols=train_symbols,
        exchange_info=exchange_info,
        market_id=binance_futures_market.market_id.value,
    )

    whitelist_added: list[str] = []
    if args.update_whitelist:
        whitelist_added = update_whitelist_csv(
            path=args.whitelist,
            market_id=binance_futures_market.market_id.value,
            accepted_symbols=cast(list[str], manifest["accepted_symbols"]),
        )

    manifest = {
        **manifest,
        "resolved_at_utc": _now_utc(),
        "whitelist_update": {
            "path": str(args.whitelist),
            "updated": bool(args.update_whitelist),
            "added_count": len(whitelist_added),
            "added_symbols": whitelist_added,
            "added_symbols_sha256": _hash_lines(whitelist_added),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(render_json_payload(manifest) + "\n", encoding="utf-8")
    print(render_json_payload(manifest))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve Stage 04A Binance Futures train-compatible universe."
    )
    parser.add_argument("--train-npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument(
        "--train-symbols-json",
        type=Path,
        default=None,
        help="Optional sanitized JSON list of HF train symbols; skips NPZ loading.",
    )
    parser.add_argument(
        "--market-config",
        type=Path,
        default=Path("configs/prod/market_data.yaml"),
    )
    parser.add_argument("--whitelist", type=Path, default=Path("configs/prod/whitelist.csv"))
    parser.add_argument("--exchange-info-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--update-whitelist",
        action="store_true",
        help="Append accepted binance:futures rows missing from the whitelist.",
    )
    return parser


def load_train_symbols_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("train symbols json must be a list")
    symbols: list[str] = []
    for item in payload:
        symbol = str(item).strip().upper()
        if symbol:
            symbols.append(symbol)
    return sorted(set(symbols))


def _read_whitelist_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("whitelist csv must have a header row")
        required = {"market_id", "symbol", "is_enabled"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"whitelist csv missing required columns: {sorted(missing)}")
        return [dict(row) for row in reader], list(reader.fieldnames)


def _write_whitelist_csv(
    *,
    path: Path,
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _hash_lines(items: Iterable[str]) -> str:
    text = "\n".join(items)
    if text:
        text += "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _onboard_ms_to_utc(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC).replace(second=0, microsecond=0)
    return _format_utc(dt)


def _max_iso_utc(left: str, right: str | None) -> str:
    if right is None:
        return left
    left_dt = _parse_utc(left)
    right_dt = _parse_utc(right)
    return _format_utc(max(left_dt, right_dt))


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(UTC)


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now_utc() -> str:
    return _format_utc(datetime.now(tz=UTC))


if __name__ == "__main__":
    raise SystemExit(main())
