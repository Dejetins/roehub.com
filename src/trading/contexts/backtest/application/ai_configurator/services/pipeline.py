from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from trading.contexts.backtest.application.ai_configurator.dto import BacktestAiConfigJob

from .catalog import BacktestAiAllowedCatalog, BacktestAiCatalogResolver
from .security import BacktestAiInputGate
from .validator import BacktestAiConfigValidationOutcome, BacktestAiConfigValidator

BacktestAiPipelineStage = Literal["input_gate", "validation"]

_FAKE_MODEL_ID = "deterministic-catalog-validator-v1"
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


@dataclass(frozen=True, slots=True)
class BacktestAiConfigPipelineResult:
    status: Literal[
        "ready",
        "needs_clarification",
        "blocked_by_policy",
        "input_too_large",
        "security_review",
    ]
    assistant_message: str
    catalog_snapshot_hash: str
    stage: BacktestAiPipelineStage
    validated_config: dict[str, Any] | None = None
    warnings: tuple[dict[str, Any], ...] = ()
    suggestions: tuple[dict[str, Any], ...] = ()
    validation_errors: tuple[dict[str, Any], ...] = ()
    model_id: str = _FAKE_MODEL_ID
    last_error: str | None = None
    last_error_json: dict[str, Any] | None = None

    @classmethod
    def from_validation(
        cls,
        *,
        outcome: BacktestAiConfigValidationOutcome,
        catalog: BacktestAiAllowedCatalog,
    ) -> BacktestAiConfigPipelineResult:
        return cls(
            status=outcome.status,
            assistant_message=outcome.assistant_message,
            catalog_snapshot_hash=catalog.snapshot_hash,
            stage="validation",
            validated_config=outcome.validated_config,
            warnings=outcome.warnings,
            suggestions=outcome.suggestions,
            validation_errors=outcome.validation_errors,
            last_error=outcome.last_error,
            last_error_json=outcome.last_error_json,
        )


@dataclass(frozen=True, slots=True)
class BacktestAiConfigPipeline:
    catalog_resolver: BacktestAiCatalogResolver
    validator: BacktestAiConfigValidator
    input_gate: BacktestAiInputGate = BacktestAiInputGate()

    def run(self, *, job: BacktestAiConfigJob) -> BacktestAiConfigPipelineResult:
        catalog = self.catalog_resolver.resolve()
        input_gate = self.input_gate.evaluate(
            message=job.user_prompt_text,
            locale=job.locale,
            mode=job.mode,
        )
        if not input_gate.allowed:
            return BacktestAiConfigPipelineResult(
                status=input_gate.terminal_status or "blocked_by_policy",
                assistant_message=input_gate.user_message or "Request cannot be processed.",
                catalog_snapshot_hash=catalog.snapshot_hash,
                stage="input_gate",
                validation_errors=tuple(
                    {
                        "path": "message",
                        "code": flag,
                        "message": "Request did not pass deterministic input checks",
                    }
                    for flag in input_gate.flags
                ),
                last_error=input_gate.terminal_status or "blocked_by_policy",
                last_error_json={
                    "security_flags": list(input_gate.flags),
                    "security_risk_score": input_gate.risk_score,
                    "security_decision": input_gate.decision,
                },
            )

        draft = _deterministic_model_output(job=job, catalog=catalog)
        raw_output = json.dumps(
            draft,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        outcome = self.validator.validate_model_output(
            raw_output=raw_output,
            catalog=catalog,
        )
        return BacktestAiConfigPipelineResult.from_validation(
            outcome=outcome,
            catalog=catalog,
        )


def _deterministic_model_output(
    *,
    job: BacktestAiConfigJob,
    catalog: BacktestAiAllowedCatalog,
) -> dict[str, Any]:
    message = job.user_prompt_text
    lowered = message.casefold()
    symbols = _extract_symbols(message=message, catalog=catalog)
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


def _extract_symbols(
    *,
    message: str,
    catalog: BacktestAiAllowedCatalog,
) -> tuple[str, ...]:
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
    item = catalog.indicator_by_id(indicator_id)
    if item is None:
        item = catalog.indicators[0]
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
    "BacktestAiConfigPipeline",
    "BacktestAiConfigPipelineResult",
    "BacktestAiPipelineStage",
]
