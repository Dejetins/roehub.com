from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from trading.contexts.backtest.application.services.signals_from_indicators_v1 import (
    supported_indicator_ids_for_signals_v1,
)
from trading.contexts.backtest.application.services.v2 import BacktestRuntimeDefaultsService

_DEFAULT_EXCHANGE = "binance"
_DEFAULT_MARKET_TYPE = "spot"
_DEFAULT_SYMBOL = "BTCUSDT"
_INDICATORS_CONFIG_SOURCE = "configs/prod/indicators.yaml"


@dataclass(frozen=True, slots=True)
class BacktestAiIndicatorCatalogItem:
    indicator_id: str
    sources: tuple[str, ...]
    param_specs: Mapping[str, Any]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "indicator_id": self.indicator_id,
            "sources": list(self.sources),
            "param_specs": dict(self.param_specs),
        }


@dataclass(frozen=True, slots=True)
class BacktestAiAllowedCatalog:
    exchanges: tuple[str, ...]
    market_types: tuple[str, ...]
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    risk_modes: tuple[str, ...]
    direction_modes: tuple[str, ...]
    sizing_modes: tuple[str, ...]
    ranking_metrics: tuple[str, ...]
    ranking_default: Mapping[str, Any]
    top_n_default: int
    guardrails: Mapping[str, Any]
    execution_defaults: Mapping[str, Any]
    hit_times_grid: Mapping[str, Any]
    indicators: tuple[BacktestAiIndicatorCatalogItem, ...]
    artifact_capabilities: Mapping[str, Any] = field(default_factory=dict)
    source_paths: tuple[str, ...] = (_INDICATORS_CONFIG_SOURCE,)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "exchanges": list(self.exchanges),
            "market_types": list(self.market_types),
            "symbols": list(self.symbols),
            "timeframes": list(self.timeframes),
            "risk_modes": list(self.risk_modes),
            "direction_modes": list(self.direction_modes),
            "sizing_modes": list(self.sizing_modes),
            "ranking_metrics": list(self.ranking_metrics),
            "ranking_default": dict(self.ranking_default),
            "top_n_default": self.top_n_default,
            "guardrails": dict(self.guardrails),
            "execution_defaults": dict(self.execution_defaults),
            "hit_times_grid": dict(self.hit_times_grid),
            "indicators": [item.as_mapping() for item in self.indicators],
            "artifact_capabilities": _json_ready(self.artifact_capabilities),
            "source_paths": list(self.source_paths),
        }

    @property
    def snapshot_hash(self) -> str:
        encoded = json.dumps(
            self.as_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def indicator_ids(self) -> tuple[str, ...]:
        return tuple(item.indicator_id for item in self.indicators)

    def indicator_by_id(self, indicator_id: str) -> BacktestAiIndicatorCatalogItem | None:
        normalized = indicator_id.strip().lower()
        for item in self.indicators:
            if item.indicator_id == normalized:
                return item
        return None

    def default_config(self) -> dict[str, Any]:
        ranking_default = dict(self.ranking_default)
        execution_defaults = dict(self.execution_defaults)
        default_indicator = self.indicator_by_id("momentum.rsi") or self.indicators[0]
        return {
            "coordinates": {
                "exchange": self.exchanges[0],
                "market_type": self.market_types[0],
                "symbol": self.symbols[0],
            },
            "timeframe": self.timeframes[0],
            "time_range": _default_time_range(self),
            "indicators": [_default_indicator_config(item=default_indicator)],
            "risk": {"mode": self.risk_modes[0]},
            "execution": {
                "direction_mode": self.direction_modes[0],
                "fee_rate": execution_defaults.get("fee_rate", 0.00075),
                "slippage_rate": execution_defaults.get("slippage_rate", 0.0001),
                "initial_cash_quote": execution_defaults.get("initial_cash_quote", 10000.0),
                "sizing": execution_defaults.get(
                    "sizing",
                    {"mode": "fixed_equity_pct", "equity_pct": 10.0},
                ),
                "profit_lock": execution_defaults.get("profit_lock", {"enabled": False}),
                "close_on_end": execution_defaults.get("close_on_end", True),
            },
            "ranking": {
                "primary_metric": ranking_default.get("primary_metric", "total_return_pct"),
                "direction": ranking_default.get("direction", "desc"),
            },
            "top_n": self.top_n_default,
        }


@dataclass(frozen=True, slots=True)
class BacktestAiCatalogResolver:
    runtime_defaults_service: BacktestRuntimeDefaultsService
    supported_symbols: Sequence[str] = (_DEFAULT_SYMBOL,)
    exchanges: Sequence[str] = (_DEFAULT_EXCHANGE,)
    market_types: Sequence[str] = (_DEFAULT_MARKET_TYPE,)
    artifact_capabilities: Mapping[str, Any] = field(default_factory=dict)
    source_paths: Sequence[str] = (_INDICATORS_CONFIG_SOURCE,)

    def resolve(self) -> BacktestAiAllowedCatalog:
        runtime_defaults = self.runtime_defaults_service.execute().as_mapping()
        indicator_sources = dict(runtime_defaults.get("indicator_sources") or {})
        indicator_param_specs = dict(runtime_defaults.get("indicator_param_specs") or {})
        executable_indicator_ids = set(supported_indicator_ids_for_signals_v1())
        supported_indicator_ids = tuple(
            indicator_id
            for indicator_id in _normalize_sorted(runtime_defaults.get("supported_indicator_ids"))
            if indicator_id in executable_indicator_ids
        )
        indicators = tuple(
            BacktestAiIndicatorCatalogItem(
                indicator_id=indicator_id,
                sources=tuple(str(value) for value in indicator_sources.get(indicator_id, [])),
                param_specs=dict(indicator_param_specs.get(indicator_id) or {}),
            )
            for indicator_id in supported_indicator_ids
        )
        if not indicators:
            raise ValueError(
                "Backtest AI catalog requires at least one executable supported indicator"
            )
        artifact_capabilities = _filtered_artifact_capabilities(
            artifact_capabilities=self.artifact_capabilities,
            symbols=self.supported_symbols,
            executable_indicator_ids=supported_indicator_ids,
        )
        return BacktestAiAllowedCatalog(
            exchanges=_normalize_preserve_order(self.exchanges, fallback=_DEFAULT_EXCHANGE),
            market_types=_normalize_preserve_order(
                self.market_types,
                fallback=_DEFAULT_MARKET_TYPE,
            ),
            symbols=_normalize_preserve_order(self.supported_symbols, fallback=_DEFAULT_SYMBOL),
            timeframes=_normalize_preserve_order(
                runtime_defaults.get("supported_timeframes"),
                fallback="15m",
            ),
            risk_modes=_normalize_preserve_order(
                runtime_defaults.get("risk_modes"),
                fallback="none",
            ),
            direction_modes=_normalize_preserve_order(
                runtime_defaults.get("direction_modes"),
                fallback="long_short_reversal",
            ),
            sizing_modes=_normalize_preserve_order(
                runtime_defaults.get("sizing_modes"),
                fallback="fixed_equity_pct",
            ),
            ranking_metrics=_normalize_preserve_order(
                runtime_defaults.get("ranking_metrics"),
                fallback="total_return_pct",
            ),
            ranking_default=dict(runtime_defaults.get("ranking_default") or {}),
            top_n_default=int(runtime_defaults.get("top_n_default") or 10),
            guardrails=dict(runtime_defaults.get("guardrails") or {}),
            execution_defaults=dict(runtime_defaults.get("execution_defaults") or {}),
            hit_times_grid=dict(runtime_defaults.get("hit_times_grid") or {}),
            indicators=indicators,
            artifact_capabilities=artifact_capabilities,
            source_paths=tuple(self.source_paths),
        )


def _default_indicator_config(*, item: BacktestAiIndicatorCatalogItem) -> dict[str, Any]:
    params = dict(item.param_specs.get("params") or {})
    window_spec = dict(params.get("window") or {})
    start = int(window_spec.get("start") or _first_int(window_spec.get("values"), default=14))
    stop = min(int(window_spec.get("stop_incl") or start), max(start, 28))
    step = int(window_spec.get("step") or 7)
    return {
        "indicator_id": item.indicator_id,
        "sources": list(item.sources[:1]),
        "window": {"start": start, "stop": stop, "step": step},
    }


def _default_time_range(catalog: BacktestAiAllowedCatalog) -> dict[str, str]:
    capability = _first_timeframe_capability(catalog)
    if capability is not None:
        period = capability.get("available_period")
        if isinstance(period, Mapping):
            start = period.get("start")
            end = period.get("end_exclusive") or period.get("end")
            if isinstance(start, str) and isinstance(end, str):
                return {"start": start, "end": end}
    return {
        "start": "2023-01-01T00:00:00Z",
        "end": "2024-01-01T00:00:00Z",
    }


def _first_timeframe_capability(
    catalog: BacktestAiAllowedCatalog,
) -> Mapping[str, Any] | None:
    symbols = catalog.artifact_capabilities.get("symbols")
    if not isinstance(symbols, Mapping):
        return None
    symbol_payload = symbols.get(catalog.symbols[0])
    if not isinstance(symbol_payload, Mapping):
        return None
    timeframes = symbol_payload.get("timeframes")
    if not isinstance(timeframes, Mapping):
        return None
    timeframe_payload = timeframes.get(catalog.timeframes[0])
    return timeframe_payload if isinstance(timeframe_payload, Mapping) else None


def _filtered_artifact_capabilities(
    *,
    artifact_capabilities: Mapping[str, Any],
    symbols: Sequence[str],
    executable_indicator_ids: tuple[str, ...],
) -> Mapping[str, Any]:
    if not artifact_capabilities:
        return {}
    allowed_symbols = set(_normalize_preserve_order(symbols, fallback=_DEFAULT_SYMBOL))
    allowed_indicators = set(executable_indicator_ids)
    raw_symbols = artifact_capabilities.get("symbols")
    if not isinstance(raw_symbols, Mapping):
        return {}
    filtered_symbols: dict[str, Any] = {}
    for raw_symbol, raw_symbol_payload in raw_symbols.items():
        symbol = str(raw_symbol).strip().upper()
        if symbol not in allowed_symbols or not isinstance(raw_symbol_payload, Mapping):
            continue
        raw_timeframes = raw_symbol_payload.get("timeframes")
        if not isinstance(raw_timeframes, Mapping):
            continue
        filtered_timeframes: dict[str, Any] = {}
        for raw_timeframe, raw_timeframe_payload in raw_timeframes.items():
            timeframe = str(raw_timeframe).strip().lower()
            if not timeframe or not isinstance(raw_timeframe_payload, Mapping):
                continue
            indicators = tuple(
                indicator_id
                for indicator_id in _normalize_sorted(raw_timeframe_payload.get("indicators"))
                if indicator_id in allowed_indicators
            )
            if not indicators:
                continue
            filtered_timeframes[timeframe] = {
                "available_period": _json_ready(
                    raw_timeframe_payload.get("available_period") or {}
                ),
                "indicators": list(indicators),
                "hit_times_available": bool(
                    raw_timeframe_payload.get("hit_times_available", False)
                ),
                "artifact_asof_date": str(raw_timeframe_payload.get("artifact_asof_date", "")),
                "published_at_utc": str(raw_timeframe_payload.get("published_at_utc", "")),
            }
        if filtered_timeframes:
            filtered_symbols[symbol] = {"timeframes": filtered_timeframes}
    if not filtered_symbols:
        return {}
    return {
        "schema_version": 1,
        "symbols": filtered_symbols,
    }


def _normalize_sorted(value: Any) -> tuple[str, ...]:
    return tuple(sorted({str(item).strip().lower() for item in value or [] if str(item).strip()}))


def _normalize_preserve_order(value: Any, *, fallback: str) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value or ():
        text = str(item).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text.upper() if text.upper().endswith("USDT") else text.lower())
    return tuple(normalized) or (fallback,)


def _first_int(value: Any, *, default: int) -> int:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, int):
                return item
    return default


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "BacktestAiAllowedCatalog",
    "BacktestAiCatalogResolver",
    "BacktestAiIndicatorCatalogItem",
]
