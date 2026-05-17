from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trading.contexts.backtest.application.ai_configurator.dto import (
    BacktestAiContextAxis,
    BacktestAiContextSnapshot,
    BacktestAiIndicatorAvailability,
)
from trading.contexts.backtest.application.ai_configurator.ports.availability_summary import (
    BacktestAiAvailabilitySummaryRepository,
)
from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider
from trading.contexts.backtest.application.services.signals_from_indicators_v1 import (
    supported_indicator_ids_for_signals_v1,
)
from trading.contexts.backtest.application.services.v2 import BacktestRuntimeDefaultsService
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    RangeValuesSpec,
)
from trading.contexts.indicators.domain.specifications.grid_param_spec import GridParamSpec

_SNAPSHOT_SCHEMA_VERSION = 1
_SNAPSHOT_SOURCE = "backtest_ai_context_snapshot_v1"
_SUMMARY_SOURCE = "artifact_publisher_active_slot_scan"
_DIRECT_SYMBOL_RE = re.compile(r"\b[A-Z0-9]{2,12}USDT\b", re.I)
_SYMBOL_ALIASES: tuple[tuple[str, str], ...] = (
    ("биток", "BTCUSDT"),
    ("биткоин", "BTCUSDT"),
    ("bitcoin", "BTCUSDT"),
    ("btc", "BTCUSDT"),
    ("эфир", "ETHUSDT"),
    ("ethereum", "ETHUSDT"),
    ("eth", "ETHUSDT"),
)


class BacktestAiContextSnapshotUnavailable(ValueError):
    """
    Fail-closed error for missing, corrupt, or unsupported context source state.
    """


@dataclass(frozen=True, slots=True)
class BacktestAiContextSnapshotBuilder:
    availability_summary_repository: BacktestAiAvailabilitySummaryRepository
    defaults_provider: BacktestGridDefaultsProvider
    runtime_defaults_service: BacktestRuntimeDefaultsService

    def build(
        self,
        *,
        user_message: str,
        current_config: Mapping[str, Any] | None = None,
    ) -> BacktestAiContextSnapshot:
        summary = self._validated_summary()
        instrument_key, instrument, ignored_symbols, warnings = _resolve_instrument(
            summary=summary,
            user_message=user_message,
            current_config=current_config,
        )
        indicator_items = self._indicator_availability(instrument=instrument)
        runtime_defaults = self.runtime_defaults_service.execute().as_mapping()
        allowed_values = _allowed_values(
            instrument=instrument,
            runtime_defaults=runtime_defaults,
        )
        timeframe_periods = _timeframe_periods(instrument=instrument)
        audit = _indicator_audit(indicators=indicator_items)
        provisional = BacktestAiContextSnapshot(
            schema_version=_SNAPSHOT_SCHEMA_VERSION,
            source=_SNAPSHOT_SOURCE,
            snapshot_hash="",
            summary_hash=str(summary["summary_hash"]),
            summary_generated_at_utc=str(summary["generated_at_utc"]),
            resolved_symbol=str(instrument["symbol"]),
            exchange=str(instrument["exchange"]),
            market_type=str(instrument["market"]),
            instrument_key=instrument_key,
            ignored_symbols=ignored_symbols,
            warnings=warnings,
            allowed_values=allowed_values,
            period={
                "start_date": str(instrument["start_date"]),
                "end_date": str(instrument["end_date"]),
            },
            timeframe_periods=timeframe_periods,
            indicators=indicator_items,
            indicator_audit=audit,
            provenance={
                "availability_summary": "availability_summary.yaml",
                "availability_summary_hash": str(summary["summary_hash"]),
                "indicators_catalog": "configs/prod/indicators.yaml",
                "signal_registry": "supported_indicator_ids_for_signals_v1",
                "hard_definitions": "trading.contexts.indicators.domain.definitions.all_defs",
            },
        )
        payload = provisional.as_mapping()
        payload["snapshot_hash"] = ""
        snapshot_hash = _canonical_sha256(payload)
        return BacktestAiContextSnapshot(
            schema_version=provisional.schema_version,
            source=provisional.source,
            snapshot_hash=snapshot_hash,
            summary_hash=provisional.summary_hash,
            summary_generated_at_utc=provisional.summary_generated_at_utc,
            resolved_symbol=provisional.resolved_symbol,
            exchange=provisional.exchange,
            market_type=provisional.market_type,
            instrument_key=provisional.instrument_key,
            ignored_symbols=provisional.ignored_symbols,
            warnings=provisional.warnings,
            allowed_values=provisional.allowed_values,
            period=provisional.period,
            timeframe_periods=provisional.timeframe_periods,
            indicators=provisional.indicators,
            indicator_audit=provisional.indicator_audit,
            provenance=provisional.provenance,
        )

    def _validated_summary(self) -> Mapping[str, Any]:
        summary = self.availability_summary_repository.load_availability_summary()
        if not isinstance(summary, Mapping):
            raise BacktestAiContextSnapshotUnavailable(
                "availability_summary.yaml must be a mapping"
            )
        if summary.get("schema_version") != 1:
            raise BacktestAiContextSnapshotUnavailable(
                "availability_summary.yaml schema_version must be 1"
            )
        if summary.get("source") != _SUMMARY_SOURCE:
            raise BacktestAiContextSnapshotUnavailable(
                "availability_summary.yaml source is unsupported"
            )
        instruments = summary.get("instruments")
        if not isinstance(instruments, Mapping) or not instruments:
            raise BacktestAiContextSnapshotUnavailable(
                "availability_summary.yaml has no instruments"
            )
        summary_hash = summary.get("summary_hash")
        if not isinstance(summary_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", summary_hash):
            raise BacktestAiContextSnapshotUnavailable(
                "availability_summary.yaml summary_hash is invalid"
            )
        expected_hash = _summary_hash(summary)
        if expected_hash != summary_hash:
            raise BacktestAiContextSnapshotUnavailable(
                "availability_summary.yaml summary_hash mismatch"
            )
        return summary

    def _indicator_availability(
        self,
        *,
        instrument: Mapping[str, Any],
    ) -> tuple[BacktestAiIndicatorAvailability, ...]:
        yaml_indicator_ids = tuple(self.defaults_provider.supported_indicator_ids())
        hard_defs = {item.indicator_id.value: item for item in all_defs()}
        signal_ids = set(supported_indicator_ids_for_signals_v1())
        summary_indicator_ids_by_timeframe = _summary_indicator_ids_by_timeframe(
            instrument=instrument
        )
        items: list[BacktestAiIndicatorAvailability] = []
        for indicator_id in yaml_indicator_ids:
            normalized_id = indicator_id.strip().lower()
            hard_def = hard_defs.get(normalized_id)
            compute_grid = self.defaults_provider.compute_defaults(
                indicator_id=normalized_id
            )
            signal_defaults = self.defaults_provider.signal_param_defaults(
                indicator_id=normalized_id
            )
            coverage_timeframes = tuple(
                timeframe
                for timeframe, ids in summary_indicator_ids_by_timeframe.items()
                if normalized_id in ids
            )
            available, reason = _availability_reason(
                indicator_id=normalized_id,
                hard_def_present=hard_def is not None,
                signal_supported=normalized_id in signal_ids,
                compute_grid_present=compute_grid is not None,
                hard_axes=() if hard_def is None else hard_def.axes,
                coverage_timeframes=coverage_timeframes,
            )
            axes = _context_axes_from_compute_grid(compute_grid)
            if "window" not in axes:
                axes["window"] = BacktestAiContextAxis(name="window", mode="none")
            items.append(
                BacktestAiIndicatorAvailability(
                    indicator_id=normalized_id,
                    available=available,
                    reason=reason,
                    sources=tuple(
                        self.defaults_provider.allowed_source_values(
                            indicator_id=normalized_id
                        )
                    ),
                    axes=axes,
                    signal_params=_context_axes_from_signal_defaults(signal_defaults),
                    coverage_timeframes=coverage_timeframes,
                )
            )
        return tuple(items)


def _resolve_instrument(
    *,
    summary: Mapping[str, Any],
    user_message: str,
    current_config: Mapping[str, Any] | None,
) -> tuple[str, Mapping[str, Any], tuple[str, ...], tuple[str, ...]]:
    instruments = summary["instruments"]
    if not isinstance(instruments, Mapping):
        raise BacktestAiContextSnapshotUnavailable("summary instruments must be mapping")
    by_symbol: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for raw_key, raw_instrument in instruments.items():
        if not isinstance(raw_instrument, Mapping):
            continue
        symbol = _required_text(raw_instrument, "symbol").upper()
        by_symbol[symbol] = (str(raw_key), raw_instrument)
    if not by_symbol:
        raise BacktestAiContextSnapshotUnavailable("summary has no valid instruments")

    requested_symbols = _extract_symbols(user_message)
    if not requested_symbols:
        current_symbol = _current_config_symbol(current_config)
        if current_symbol is not None:
            requested_symbols = (current_symbol,)
    if not requested_symbols:
        requested_symbols = (sorted(by_symbol)[0],)

    resolved_symbol = requested_symbols[0]
    if resolved_symbol not in by_symbol:
        raise BacktestAiContextSnapshotUnavailable(
            f"requested symbol is unavailable in availability_summary.yaml: {resolved_symbol}"
        )
    ignored_symbols = tuple(
        symbol for symbol in requested_symbols[1:] if symbol != resolved_symbol
    )
    warnings: list[str] = []
    if ignored_symbols:
        warnings.append(
            "multiple_symbol_request: using first symbol and recording ignored_symbols"
        )
    instrument_key, instrument = by_symbol[resolved_symbol]
    _validate_instrument_payload(instrument_key=instrument_key, instrument=instrument)
    return instrument_key, instrument, ignored_symbols, tuple(warnings)


def _indicator_audit(
    *,
    indicators: Sequence[BacktestAiIndicatorAvailability],
) -> dict[str, Any]:
    available = [item.indicator_id for item in indicators if item.available]
    excluded = [
        {"indicator_id": item.indicator_id, "reason": item.reason}
        for item in indicators
        if not item.available
    ]
    return {
        "total_indicators": len(indicators),
        "available_count": len(available),
        "excluded_count": len(excluded),
        "available_indicator_ids": available,
        "excluded_indicators": excluded,
    }


def _availability_reason(
    *,
    indicator_id: str,
    hard_def_present: bool,
    signal_supported: bool,
    compute_grid_present: bool,
    hard_axes: Sequence[str],
    coverage_timeframes: Sequence[str],
) -> tuple[bool, str]:
    if not hard_def_present:
        return False, "missing_hard_definition"
    if not signal_supported:
        return False, "missing_signal_registry"
    if not compute_grid_present and len(hard_axes) > 0:
        return False, "missing_compute_defaults"
    if not coverage_timeframes:
        return False, "missing_summary_coverage"
    _ = indicator_id
    return True, "available"


def _context_axes_from_compute_grid(grid: Any) -> dict[str, BacktestAiContextAxis]:
    if grid is None:
        return {}
    axes: dict[str, BacktestAiContextAxis] = {}
    if grid.source is not None:
        axes["source"] = _axis_from_grid_spec(grid.source)
    for name, spec in sorted(grid.params.items()):
        axes[str(name).strip().lower()] = _axis_from_grid_spec(spec)
    return axes


def _context_axes_from_signal_defaults(
    signal_defaults: Mapping[str, GridParamSpec],
) -> dict[str, BacktestAiContextAxis]:
    return {
        str(name).strip().lower(): _axis_from_grid_spec(spec)
        for name, spec in sorted(signal_defaults.items())
    }


def _axis_from_grid_spec(spec: GridParamSpec) -> BacktestAiContextAxis:
    if isinstance(spec, ExplicitValuesSpec):
        return BacktestAiContextAxis(
            name=spec.name,
            mode="explicit",
            values=tuple(spec.values),
        )
    if isinstance(spec, RangeValuesSpec):
        return BacktestAiContextAxis(
            name=spec.name,
            mode="range",
            start=spec.start,
            stop_incl=spec.stop_inclusive,
            step=spec.step,
        )
    return BacktestAiContextAxis(
        name=spec.name,
        mode="explicit",
        values=tuple(spec.materialize()),
    )


def _allowed_values(
    *,
    instrument: Mapping[str, Any],
    runtime_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    timeframes = tuple(str(item) for item in instrument["backtest_timeframes"])
    return {
        "exchange": [str(instrument["exchange"])],
        "market_type": [str(instrument["market"])],
        "symbol": [str(instrument["symbol"]).upper()],
        "timeframe": list(timeframes),
        "risk_mode": list(runtime_defaults.get("risk_modes") or ("none",)),
        "direction_mode": list(runtime_defaults.get("direction_modes") or ()),
        "sizing_mode": list(runtime_defaults.get("sizing_modes") or ()),
        "ranking_metric": list(runtime_defaults.get("ranking_metrics") or ()),
    }


def _timeframe_periods(*, instrument: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    timeframes = instrument["timeframes"]
    if not isinstance(timeframes, Mapping):
        raise BacktestAiContextSnapshotUnavailable("instrument timeframes must be mapping")
    result: dict[str, Mapping[str, Any]] = {}
    for timeframe in instrument["backtest_timeframes"]:
        payload = timeframes.get(timeframe)
        if not isinstance(payload, Mapping):
            continue
        result[str(timeframe)] = {
            "start_date": str(payload["start_date"]),
            "end_date": str(payload["end_date"]),
            "bars": int(payload["bars"]),
        }
    return result


def _summary_indicator_ids_by_timeframe(
    *,
    instrument: Mapping[str, Any],
) -> dict[str, set[str]]:
    timeframes = instrument.get("timeframes")
    if not isinstance(timeframes, Mapping):
        raise BacktestAiContextSnapshotUnavailable("instrument timeframes must be mapping")
    result: dict[str, set[str]] = {}
    for timeframe in instrument.get("backtest_timeframes", ()):
        payload = timeframes.get(timeframe)
        if not isinstance(payload, Mapping):
            continue
        result[str(timeframe)] = {
            str(indicator_id).strip().lower()
            for indicator_id in payload.get("indicator_ids") or ()
            if str(indicator_id).strip()
        }
    return result


def _validate_instrument_payload(
    *,
    instrument_key: str,
    instrument: Mapping[str, Any],
) -> None:
    for key in ("exchange", "market", "symbol", "start_date", "end_date"):
        _required_text(instrument, key)
    backtest_timeframes = instrument.get("backtest_timeframes")
    if not isinstance(backtest_timeframes, list) or not backtest_timeframes:
        raise BacktestAiContextSnapshotUnavailable(
            f"{instrument_key}: backtest_timeframes must be a non-empty list"
        )
    _timeframe_periods(instrument=instrument)


def _extract_symbols(message: str) -> tuple[str, ...]:
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


def _current_config_symbol(current_config: Mapping[str, Any] | None) -> str | None:
    if not isinstance(current_config, Mapping):
        return None
    coordinates = current_config.get("coordinates")
    if not isinstance(coordinates, Mapping):
        return None
    raw_symbol = coordinates.get("symbol")
    if not isinstance(raw_symbol, str) or not raw_symbol.strip():
        return None
    return raw_symbol.strip().upper()


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BacktestAiContextSnapshotUnavailable(f"{key} must be a non-empty string")
    return value.strip()


def _summary_hash(summary: Mapping[str, Any]) -> str:
    payload = dict(summary)
    payload.pop("summary_hash", None)
    payload.pop("generated_at_utc", None)
    return _canonical_sha256(payload)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "BacktestAiContextSnapshotBuilder",
    "BacktestAiContextSnapshotUnavailable",
]
