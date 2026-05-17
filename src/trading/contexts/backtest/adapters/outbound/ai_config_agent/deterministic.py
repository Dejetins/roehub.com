from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from trading.contexts.backtest.application.ai_configurator.dto import BacktestAiConfigJob
from trading.contexts.backtest.application.ai_configurator.ports import (
    BacktestConfigAgentRequest,
    BacktestConfigAgentResponse,
)
from trading.contexts.backtest.application.ai_configurator.services.catalog import (
    BacktestAiAllowedCatalog,
)

_MODEL_ID = "deterministic-test-tool-agent-v1"
_SYMBOL_ALIASES: tuple[tuple[str, str], ...] = (
    ("биток", "BTCUSDT"),
    ("биткоин", "BTCUSDT"),
    ("bitcoin", "BTCUSDT"),
    ("btc", "BTCUSDT"),
    ("эфир", "ETHUSDT"),
    ("ethereum", "ETHUSDT"),
    ("eth", "ETHUSDT"),
    ("солана", "SOLUSDT"),
    ("solana", "SOLUSDT"),
    ("sol", "SOLUSDT"),
    ("doge", "DOGEUSDT"),
)
_INDICATOR_ALIASES: tuple[tuple[str, str], ...] = (
    ("rsi", "momentum.rsi"),
    ("ema", "ma.ema"),
    ("sma", "ma.sma"),
    ("dema", "ma.dema"),
    ("atr", "volatility.atr"),
)
_UNSUPPORTED_INDICATOR_TERMS = ("bollinger", "bbands", "боллиндж")
_UNSUPPORTED_TIMEFRAME_PATTERNS = (
    re.compile(r"\b(?:1h|4h|1d|1w|hourly|daily)\b", re.I),
    re.compile(r"\bчасов(?:ик|ой|ая)?\b", re.I),
    re.compile(r"\bдневн(?:ой|ая)?\b", re.I),
)
_DIRECT_SYMBOL_RE = re.compile(r"\b[A-Z0-9]{2,12}USDT\b", re.I)

DeterministicToolAgentScenario = Literal[
    "valid",
    "invalid_json",
    "schema_invalid",
    "unsupported_config",
    "business_invalid",
]


@dataclass(frozen=True, slots=True)
class DisabledBacktestConfigAgentGateway:
    reason: str = "tool_agent_not_implemented"

    def run_config_session(
        self,
        request: BacktestConfigAgentRequest,
    ) -> BacktestConfigAgentResponse:
        _ = request
        return BacktestConfigAgentResponse(
            raw_output=None,
            model_id=None,
            finish_reason="blocked",
            audit_json={"reason": self.reason},
        )


@dataclass(frozen=True, slots=True)
class DeterministicBacktestConfigAgentGateway:
    """
    Deterministic backend-agent adapter for tests and local fake execution.

    It simulates the future backend-controlled tool-agent result so
    validator/security behavior can stay covered while the real LM Studio tools
    adapter is rebuilt.
    """

    scenario: DeterministicToolAgentScenario = "valid"
    scripted_outputs: tuple[str, ...] = ()
    _index: int = field(default=0, init=False, compare=False)

    def run_config_session(
        self,
        request: BacktestConfigAgentRequest,
    ) -> BacktestConfigAgentResponse:
        raw_output = self._next_scripted()
        if raw_output is None:
            raw_output = _raw_output_for_scenario(
                scenario=self.scenario,
                job=request.job,
                catalog=request.catalog,
            )
        return BacktestConfigAgentResponse(
            raw_output=raw_output,
            model_id=_MODEL_ID,
            latency_ms=0,
            finish_reason="stop",
            audit_json={"tool_agent": "deterministic"},
        )

    def _next_scripted(self) -> str | None:
        index = self._index
        if index >= len(self.scripted_outputs):
            return None
        object.__setattr__(self, "_index", index + 1)
        return self.scripted_outputs[index]


def _raw_output_for_scenario(
    *,
    scenario: DeterministicToolAgentScenario,
    job: BacktestAiConfigJob,
    catalog: BacktestAiAllowedCatalog,
) -> str:
    if scenario == "invalid_json":
        return "```json\n{\"schema_version\":1}\n```"
    draft = _deterministic_agent_output(job=job, catalog=catalog)
    if scenario == "schema_invalid":
        draft.pop("assistant_message", None)
    elif scenario == "unsupported_config":
        assert isinstance(draft["config"], dict)
        draft["config"]["indicators"][0]["indicator_id"] = "volatility.bollinger"
    elif scenario == "business_invalid":
        assert isinstance(draft["config"], dict)
        draft["config"]["time_range"] = {
            "start": "2024-01-01T00:00:00Z",
            "end": "2023-01-01T00:00:00Z",
        }
    return json.dumps(draft, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _deterministic_agent_output(
    *,
    job: BacktestAiConfigJob,
    catalog: BacktestAiAllowedCatalog,
) -> dict[str, Any]:
    message = job.user_prompt_text
    lowered = message.casefold()
    symbols = _extract_symbols(message=message)
    unsupported_symbols = tuple(symbol for symbol in symbols if symbol not in catalog.symbols)
    supported_symbols = tuple(symbol for symbol in symbols if symbol in catalog.symbols)
    config = catalog.default_config()
    if unsupported_symbols:
        config["coordinates"]["symbol"] = unsupported_symbols[0]
        return _config_ready_output(
            job=job,
            config=config,
            assistant_message=(
                "I could not find supported symbol "
                f"{unsupported_symbols[0]} in the current /backtests catalog."
            ),
            warnings=[],
            suggestions=[f"Use one of the supported symbols: {', '.join(catalog.symbols[:5])}"],
        )

    if any(term in lowered for term in _UNSUPPORTED_INDICATOR_TERMS):
        config["indicators"][0]["indicator_id"] = "volatility.bollinger"
        return _config_ready_output(
            job=job,
            config=config,
            assistant_message=(
                "I could not find Bollinger Bands in the supported indicator catalog. "
                "Choose a supported indicator such as momentum.rsi, ma.ema or volatility.atr."
            ),
            warnings=[],
            suggestions=["Try RSI, EMA or ATR with timeframe 15m."],
        )

    primary_symbol = (
        supported_symbols[0]
        if supported_symbols
        else _current_or_default_symbol(job=job, catalog=catalog)
    )
    config["coordinates"]["symbol"] = primary_symbol
    indicator_id = _select_indicator_id(message=message, catalog=catalog)
    config["indicators"] = [_indicator_config(indicator_id=indicator_id, catalog=catalog)]

    warnings: list[str] = []
    suggestions: list[str] = []
    if _requested_unsupported_timeframe(message):
        config["timeframe"] = "15m"
        warnings.append("Requested timeframe is not supported; using 15m.")
    if len(supported_symbols) > 1:
        suggestions.append(
            f"Single-symbol MVP loaded {primary_symbol}; "
            f"request another config for {', '.join(supported_symbols[1:])}."
        )
    if _requests_tp_sl_grid(lowered):
        config["risk"] = _default_tp_sl_grid(catalog=catalog)
    else:
        suggestions.append("Add tp_sl_grid if you want stop loss / take profit validation.")
    if "2024" in message:
        config["time_range"] = {
            "start": "2024-01-01T00:00:00Z",
            "end": "2025-01-01T00:00:00Z",
        }

    return _config_ready_output(
        job=job,
        config=config,
        assistant_message=_ready_message(symbol=primary_symbol, locale=job.locale),
        warnings=warnings,
        suggestions=suggestions,
    )


def _config_ready_output(
    *,
    job: BacktestAiConfigJob,
    config: dict[str, Any],
    assistant_message: str,
    warnings: list[str],
    suggestions: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": job.mode,
        "status": "config_ready",
        "assistant_message": assistant_message,
        "assumptions": [
            "Default execution settings come from current /backtests runtime defaults."
        ],
        "warnings": warnings,
        "config": config,
        "suggestions": suggestions,
    }


def _extract_symbols(*, message: str) -> tuple[str, ...]:
    found: list[str] = []
    seen: set[str] = set()
    for match in _DIRECT_SYMBOL_RE.finditer(message):
        symbol = match.group(0).upper()
        if symbol not in seen:
            seen.add(symbol)
            found.append(symbol)
    lowered = message.casefold()
    for alias, symbol in _SYMBOL_ALIASES:
        if alias in lowered and symbol not in seen:
            seen.add(symbol)
            found.append(symbol)
    return tuple(found)


def _current_or_default_symbol(
    *,
    job: BacktestAiConfigJob,
    catalog: BacktestAiAllowedCatalog,
) -> str:
    current = job.current_config_json
    if isinstance(current, Mapping):
        coordinates = current.get("coordinates")
        if isinstance(coordinates, Mapping):
            raw_symbol = coordinates.get("symbol")
            if isinstance(raw_symbol, str):
                symbol = raw_symbol.strip().upper()
                if symbol in catalog.symbols:
                    return symbol
    return catalog.symbols[0]


def _select_indicator_id(
    *,
    message: str,
    catalog: BacktestAiAllowedCatalog,
) -> str:
    lowered = message.casefold()
    for alias, indicator_id in _INDICATOR_ALIASES:
        if alias in lowered and indicator_id in catalog.indicator_ids:
            return indicator_id
    if "momentum.rsi" in catalog.indicator_ids:
        return "momentum.rsi"
    return catalog.indicator_ids[0]


def _indicator_config(
    *,
    indicator_id: str,
    catalog: BacktestAiAllowedCatalog,
) -> dict[str, Any]:
    item = catalog.indicator_by_id(indicator_id) or catalog.indicators[0]
    params = dict(item.param_specs.get("params") or {})
    window_spec = dict(params.get("window") or {})
    values = window_spec.get("values")
    if isinstance(values, list) and values:
        start = int(values[0])
        stop = int(values[min(1, len(values) - 1)])
        step = max(1, stop - start) if stop > start else 1
    else:
        start = int(window_spec.get("start") or 7)
        stop = min(int(window_spec.get("stop_incl") or 28), max(start, 28))
        step = int(window_spec.get("step") or 7)
    return {
        "indicator_id": item.indicator_id,
        "sources": list(item.sources[:1]),
        "window": {"start": start, "stop": stop, "step": step},
    }


def _requested_unsupported_timeframe(message: str) -> bool:
    return any(pattern.search(message) is not None for pattern in _UNSUPPORTED_TIMEFRAME_PATTERNS)


def _requests_tp_sl_grid(lowered: str) -> bool:
    return any(
        term in lowered
        for term in (
            "tp_sl_grid",
            "stop loss",
            "take profit",
            "sl/tp",
            "tp/sl",
            "стоп",
            "тейк",
            "безопас",
            "safer",
        )
    )


def _default_tp_sl_grid(*, catalog: BacktestAiAllowedCatalog) -> dict[str, Any]:
    hit_times = dict(catalog.hit_times_grid)
    tp_levels = _levels(hit_times.get("tp_levels_pct"), fallback=(1.0, 1.5))
    sl_levels = _levels(hit_times.get("sl_levels_pct"), fallback=(1.0, 1.5))
    return {
        "mode": "tp_sl_grid",
        "tp": {"start_pct": tp_levels[0], "stop_pct": tp_levels[-1], "step_pct": _step(tp_levels)},
        "sl": {"start_pct": sl_levels[0], "stop_pct": sl_levels[-1], "step_pct": _step(sl_levels)},
    }


def _levels(value: Any, *, fallback: tuple[float, float]) -> tuple[float, ...]:
    if isinstance(value, list | tuple):
        levels = tuple(float(item) for item in value[:2])
        if levels:
            return levels
    return fallback


def _step(levels: tuple[float, ...]) -> float:
    if len(levels) < 2:
        return 1.0
    return round(levels[1] - levels[0], 10)


def _ready_message(*, symbol: str, locale: str) -> str:
    if locale == "ru":
        return f"Я собрал валидный конфиг для {symbol} на 15m."
    return f"I prepared a valid {symbol} configuration on 15m."


__all__ = [
    "DeterministicBacktestConfigAgentGateway",
    "DeterministicToolAgentScenario",
    "DisabledBacktestConfigAgentGateway",
]
