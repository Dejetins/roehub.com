from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Mapping, Sequence

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestCostEstimate,
    BacktestExecutionDefaults,
    BacktestPreflightResult,
    BacktestRuntimeDefaults,
    BacktestRuntimeGuardrails,
    BacktestValidationIssue,
)
from trading.contexts.backtest.application.ports import (
    BacktestArtifactContextResolver,
    BacktestArtifactContextUnavailable,
    BacktestGridDefaultsProvider,
)
from trading.contexts.backtest.application.ports.staged_runner import (
    BACKTEST_RANKING_DIRECTION_BY_METRIC_LITERAL_V1,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    DEFAULT_LIGHT_ESTIMATED_COMBINATIONS,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
    artifact_market_id_from_coordinates_v2,
)
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    RangeValuesSpec,
)
from trading.contexts.indicators.domain.specifications.grid_param_spec import (
    GridParamSpec,
    GridValue,
)

SUPPORTED_BACKTEST_TIMEFRAMES_V1: tuple[str, ...] = ("15m",)
BACKTEST_RISK_MODES_V1: tuple[str, ...] = ("none", "tp_sl_grid")
BACKTEST_DIRECTION_MODES_V1: tuple[str, ...] = ("long_only", "long_short_reversal")
BACKTEST_SIZING_MODES_V1: tuple[str, ...] = (
    "all_in",
    "fixed_quote",
    "fixed_equity_pct",
    "fixed_equity_pct_min_quote",
    "fixed_equity_pct_max_quote",
)
BACKTEST_RANKING_METRICS_V1: tuple[str, ...] = tuple(
    BACKTEST_RANKING_DIRECTION_BY_METRIC_LITERAL_V1.keys()
)
DEFAULT_BACKTEST_RANKING_V1: Mapping[str, str] = {
    "primary_metric": "total_return_pct",
    "direction": "desc",
}
DEFAULT_BACKTEST_TOP_N_V1 = 50

BACKTEST_ERROR_INVALID_REQUEST = "backtest.invalid_request"
BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED = "backtest.tp_sl_grid_not_covered"
BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE = "backtest.request_too_expensive"
BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE = "backtest.artifacts_unavailable"


@dataclass(frozen=True, slots=True)
class BacktestRuntimeConfig:
    """
    Startup-resolved public backtest runtime contract for Iteration 1 preflight.
    """

    hit_times_tp_levels_pct: tuple[float, ...]
    hit_times_sl_levels_pct: tuple[float, ...]
    artifact_config_hash: str
    guardrails: BacktestRuntimeGuardrails = BacktestRuntimeGuardrails()
    execution_defaults: BacktestExecutionDefaults = BacktestExecutionDefaults()


class BacktestPreflightRejected(ValueError):
    """
    Deterministic preflight rejection carrying public error code and issues.
    """

    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        issues: Sequence[BacktestValidationIssue],
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.issues = tuple(
            sorted(issues, key=lambda issue: (issue.path, issue.code, issue.message))
        )
        self.retryable = retryable

    def details(self) -> dict[str, Any]:
        return {
            "errors": [issue.as_mapping() for issue in self.issues],
            "retryable": self.retryable,
        }


@dataclass(frozen=True, slots=True)
class BacktestRuntimeDefaultsService:
    """
    Application service for `GET /backtests/runtime-defaults`.
    """

    defaults_provider: BacktestGridDefaultsProvider
    runtime_config: BacktestRuntimeConfig

    def execute(self) -> BacktestRuntimeDefaults:
        supported_indicator_ids = self.defaults_provider.supported_indicator_ids()
        indicator_sources = {
            indicator_id: self.defaults_provider.allowed_source_values(
                indicator_id=indicator_id
            )
            for indicator_id in supported_indicator_ids
        }
        return BacktestRuntimeDefaults(
            supported_timeframes=SUPPORTED_BACKTEST_TIMEFRAMES_V1,
            risk_modes=BACKTEST_RISK_MODES_V1,
            direction_modes=BACKTEST_DIRECTION_MODES_V1,
            sizing_modes=BACKTEST_SIZING_MODES_V1,
            ranking_metrics=BACKTEST_RANKING_METRICS_V1,
            ranking_default=DEFAULT_BACKTEST_RANKING_V1,
            top_n_default=DEFAULT_BACKTEST_TOP_N_V1,
            guardrails=self.runtime_config.guardrails,
            execution_defaults=self.runtime_config.execution_defaults,
            supported_indicator_ids=supported_indicator_ids,
            indicator_sources=indicator_sources,
            indicator_param_specs={
                indicator_id: _indicator_param_specs(
                    indicator_id=indicator_id,
                    defaults_provider=self.defaults_provider,
                )
                for indicator_id in supported_indicator_ids
            },
            hit_times_grid={
                "timeframe": "15m",
                "tp_levels_pct": self.runtime_config.hit_times_tp_levels_pct,
                "sl_levels_pct": self.runtime_config.hit_times_sl_levels_pct,
            },
            links={
                "preflight": "/backtests/preflight",
                "jobs": "/backtests/jobs",
            },
        )


def _indicator_param_specs(
    *,
    indicator_id: str,
    defaults_provider: BacktestGridDefaultsProvider,
) -> dict[str, Any]:
    grid = defaults_provider.compute_defaults(indicator_id=indicator_id)
    if grid is None:
        return {"params": {}, "inputs": {}}
    inputs: dict[str, Any] = {}
    if grid.source is not None:
        inputs["source"] = _grid_param_spec_as_mapping(spec=grid.source)
    return {
        "params": {
            name: _grid_param_spec_as_mapping(spec=spec)
            for name, spec in sorted(grid.params.items())
        },
        "inputs": inputs,
    }


def _grid_param_spec_as_mapping(*, spec: GridParamSpec) -> dict[str, Any]:
    if isinstance(spec, RangeValuesSpec):
        return {
            "mode": "range",
            "start": spec.start,
            "stop_incl": spec.stop_inclusive,
            "step": spec.step,
        }
    if isinstance(spec, ExplicitValuesSpec):
        return {"mode": "explicit", "values": list(spec.values)}
    return {"mode": "explicit", "values": list(spec.materialize())}


@dataclass(frozen=True, slots=True)
class BacktestPreflightService:
    """
    Strict Iteration 1 request normalization and cost-estimate service.
    """

    defaults_provider: BacktestGridDefaultsProvider
    artifact_context_resolver: BacktestArtifactContextResolver
    runtime_config: BacktestRuntimeConfig

    def execute(
        self,
        payload: Mapping[str, Any],
        *,
        validation_guardrails: BacktestRuntimeGuardrails | None = None,
    ) -> BacktestPreflightResult:
        guardrails = validation_guardrails or self.runtime_config.guardrails
        if not isinstance(payload, Mapping):
            raise BacktestPreflightRejected(
                error_code=BACKTEST_ERROR_INVALID_REQUEST,
                message="Backtest preflight request must be a JSON object",
                issues=(
                    BacktestValidationIssue(
                        path="body",
                        code="invalid_type",
                        message="Request body must be a JSON object",
                    ),
                ),
            )

        coordinates = self._normalize_coordinates(payload=payload)
        timeframe = self._normalize_timeframe(payload=payload)
        time_range = self._normalize_time_range(payload=payload)
        (
            indicators,
            indicator_rows,
            candidate_combinations,
            row_count_upper_bounds_by_indicator,
        ) = self._normalize_indicators(
            payload=payload,
            guardrails=guardrails,
        )
        execution = self._normalize_execution(payload=payload)
        ranking = self._normalize_ranking(payload=payload)
        top_n = self._normalize_top_n(payload=payload)
        risk, tp_sl_cells = self._normalize_risk(payload=payload, guardrails=guardrails)

        too_expensive_issues = self._cost_guardrail_issues(
            indicators=indicators,
            indicator_rows=indicator_rows,
            candidate_combinations=candidate_combinations,
            tp_sl_cells=tp_sl_cells,
            top_n=top_n,
            guardrails=guardrails,
        )
        if too_expensive_issues:
            raise BacktestPreflightRejected(
                error_code=BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE,
                message="Backtest request exceeds configured preflight guardrails",
                issues=too_expensive_issues,
            )

        normalized_request: dict[str, Any] = {
            "coordinates": coordinates.as_mapping(),
            "timeframe": timeframe,
            "time_range": time_range,
            "indicators": indicators,
            "risk": risk,
            "execution": execution,
            "ranking": ranking,
            "top_n": top_n,
        }
        request_hash = _canonical_json_sha256(normalized_request)
        result_config_hash = self._result_config_hash()
        artifact_metadata = self._resolve_artifact_metadata(coordinates=coordinates)
        cost_estimate = BacktestCostEstimate(
            indicator_rows=indicator_rows,
            candidate_combinations=candidate_combinations,
            tp_sl_cells=tp_sl_cells,
            cost_class=_cost_class(candidate_combinations=candidate_combinations),
            estimated_combinations_upper_bound=candidate_combinations,
            estimated_combinations=candidate_combinations,
            arity=len(indicators),
            row_count_upper_bounds_by_indicator=row_count_upper_bounds_by_indicator,
            risk_mode=str(risk["mode"]),
            requested_range=time_range,
            requested_top_n=top_n,
            scheduling_class=_preflight_scheduling_class(
                estimated_combinations_upper_bound=candidate_combinations,
                arity=len(indicators),
            ),
        )
        warnings = self._guardrail_warnings(
            cost_estimate=cost_estimate,
            guardrails=guardrails,
        )
        return BacktestPreflightResult(
            normalized_request=normalized_request,
            request_hash=request_hash,
            result_config_hash=result_config_hash,
            artifact_metadata=artifact_metadata,
            cost_estimate=cost_estimate,
            warnings=warnings,
            errors=(),
        )

    def _normalize_coordinates(self, *, payload: Mapping[str, Any]) -> BacktestCoordinates:
        raw_coordinates = payload.get("coordinates")
        if not isinstance(raw_coordinates, Mapping):
            raise _invalid_request(
                path="coordinates",
                code="required",
                message="coordinates must be an object",
            )

        exchange = _normalize_token(
            raw_coordinates.get("exchange"),
            path="coordinates.exchange",
            lower=True,
        )
        market_type = _normalize_token(
            raw_coordinates.get("market_type"),
            path="coordinates.market_type",
            lower=True,
        )
        symbol = _normalize_token(
            raw_coordinates.get("symbol"),
            path="coordinates.symbol",
            upper=True,
        )
        try:
            artifact_market_id_from_coordinates_v2(
                ArtifactCoordinatesV2(
                    exchange=exchange,
                    market_type=market_type,
                    symbol=symbol,
                )
            )
        except ValueError as error:
            raise _invalid_request(
                path="coordinates",
                code="unsupported_market",
                message=str(error),
            ) from error
        return BacktestCoordinates(exchange=exchange, market_type=market_type, symbol=symbol)

    def _normalize_timeframe(self, *, payload: Mapping[str, Any]) -> str:
        raw_timeframe = payload.get("timeframe")
        if not isinstance(raw_timeframe, str):
            raise _invalid_request(
                path="timeframe",
                code="required",
                message="timeframe must be string '15m'",
            )
        timeframe = raw_timeframe.strip().lower()
        if timeframe != "15m":
            raise _invalid_request(
                path="timeframe",
                code="unsupported_timeframe",
                message="Only timeframe '15m' is supported for backtest runtime requests",
            )
        return timeframe

    def _normalize_time_range(self, *, payload: Mapping[str, Any]) -> dict[str, str]:
        raw_time_range = payload.get("time_range")
        if not isinstance(raw_time_range, Mapping):
            raise _invalid_request(
                path="time_range",
                code="required",
                message="time_range must be an object with start and end",
            )
        start = _parse_utc_timestamp(raw_time_range.get("start"), path="time_range.start")
        end = _parse_utc_timestamp(raw_time_range.get("end"), path="time_range.end")
        if start >= end:
            raise _invalid_request(
                path="time_range",
                code="invalid_range",
                message="time_range must use non-empty half-open semantics [start, end)",
            )
        return {
            "start": _format_utc_timestamp(start),
            "end": _format_utc_timestamp(end),
        }

    def _normalize_indicators(
        self,
        *,
        payload: Mapping[str, Any],
        guardrails: BacktestRuntimeGuardrails,
    ) -> tuple[list[dict[str, Any]], int, int, dict[str, int]]:
        raw_indicators = payload.get("indicators")
        if not isinstance(raw_indicators, Sequence) or isinstance(
            raw_indicators,
            (str, bytes, bytearray),
        ):
            raise _invalid_request(
                path="indicators",
                code="required",
                message="indicators must be a non-empty list",
            )
        if len(raw_indicators) == 0:
            raise _invalid_request(
                path="indicators",
                code="empty",
                message="indicators must contain at least one item",
            )
        if len(raw_indicators) > guardrails.max_indicator_arity:
            raise BacktestPreflightRejected(
                error_code=BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE,
                message="Backtest request exceeds indicator arity guardrail",
                issues=(
                    BacktestValidationIssue(
                        path="indicators",
                        code="max_indicator_arity",
                        message=(
                            f"indicator arity must be <= {guardrails.max_indicator_arity}"
                        ),
                    ),
                ),
            )

        supported_ids = set(self.defaults_provider.supported_indicator_ids())
        normalized: list[dict[str, Any]] = []
        row_counts: list[int] = []
        row_count_upper_bounds_by_indicator: dict[str, int] = {}
        for index, raw_indicator in enumerate(raw_indicators):
            path = f"indicators.{index}"
            if not isinstance(raw_indicator, Mapping):
                raise _invalid_request(
                    path=path,
                    code="invalid_type",
                    message="indicator item must be an object",
                )
            indicator_id = _normalize_indicator_id(
                raw_indicator.get("indicator_id"),
                path=f"{path}.indicator_id",
            )
            if indicator_id not in supported_ids:
                raise _invalid_request(
                    path=f"{path}.indicator_id",
                    code="unknown_indicator",
                    message=f"Unsupported indicator_id: {indicator_id}",
                )

            sources = self._normalize_sources(
                raw_indicator=raw_indicator,
                indicator_id=indicator_id,
                path=path,
            )
            window = self._normalize_window(
                raw_indicator=raw_indicator,
                indicator_id=indicator_id,
                path=path,
            )
            source_count = len(sources) if len(sources) > 0 else 1
            row_count = source_count * int(window["count"])
            row_counts.append(row_count)
            row_count_key = indicator_id
            if row_count_key in row_count_upper_bounds_by_indicator:
                row_count_key = f"{indicator_id}#{index}"
            row_count_upper_bounds_by_indicator[row_count_key] = row_count
            normalized.append(
                {
                    "indicator_id": indicator_id,
                    "sources": list(sources),
                    "window": {
                        "start": window["start"],
                        "stop": window["stop"],
                        "step": window["step"],
                    },
                }
            )

        indicator_rows = sum(row_counts)
        candidate_combinations = 1
        for row_count in row_counts:
            candidate_combinations *= row_count
        return (
            normalized,
            indicator_rows,
            candidate_combinations,
            row_count_upper_bounds_by_indicator,
        )

    def _normalize_sources(
        self,
        *,
        raw_indicator: Mapping[str, Any],
        indicator_id: str,
        path: str,
    ) -> tuple[str, ...]:
        allowed_sources = self.defaults_provider.allowed_source_values(
            indicator_id=indicator_id
        )
        raw_sources = raw_indicator.get("sources")
        if len(allowed_sources) == 0:
            if raw_sources is None:
                return ()
            if isinstance(raw_sources, Sequence) and not isinstance(
                raw_sources,
                (str, bytes, bytearray),
            ) and len(raw_sources) == 0:
                return ()
            raise _invalid_request(
                path=f"{path}.sources",
                code="unsupported_source_axis",
                message=f"indicator_id {indicator_id} does not support sources",
            )
        if not isinstance(raw_sources, Sequence) or isinstance(
            raw_sources,
            (str, bytes, bytearray),
        ):
            raise _invalid_request(
                path=f"{path}.sources",
                code="required",
                message="sources must be a non-empty list",
            )
        if len(raw_sources) == 0:
            raise _invalid_request(
                path=f"{path}.sources",
                code="empty",
                message="sources must contain at least one value",
            )
        normalized_sources: list[str] = []
        seen: set[str] = set()
        allowed = set(allowed_sources)
        for source_index, raw_source in enumerate(raw_sources):
            if not isinstance(raw_source, str):
                raise _invalid_request(
                    path=f"{path}.sources.{source_index}",
                    code="invalid_type",
                    message="source must be a string",
                )
            source = raw_source.strip().lower()
            if source not in allowed:
                raise _invalid_request(
                    path=f"{path}.sources.{source_index}",
                    code="invalid_source",
                    message=f"Unsupported source {source!r} for indicator_id {indicator_id}",
                )
            if source in seen:
                continue
            seen.add(source)
            normalized_sources.append(source)
        return tuple(normalized_sources)

    def _normalize_window(
        self,
        *,
        raw_indicator: Mapping[str, Any],
        indicator_id: str,
        path: str,
    ) -> dict[str, int]:
        raw_window = raw_indicator.get("window")
        if not isinstance(raw_window, Mapping):
            raise _invalid_request(
                path=f"{path}.window",
                code="required",
                message="window must be an object with start, stop and step",
            )
        start = _positive_int(raw_window.get("start"), path=f"{path}.window.start")
        stop = _positive_int(raw_window.get("stop"), path=f"{path}.window.stop")
        step = _positive_int(raw_window.get("step"), path=f"{path}.window.step")
        if start > stop:
            raise _invalid_request(
                path=f"{path}.window",
                code="invalid_range",
                message="window.start must be <= window.stop",
            )
        values = tuple(range(start, stop + 1, step))
        if len(values) == 0:
            raise _invalid_request(
                path=f"{path}.window",
                code="empty",
                message="window range must materialize at least one value",
            )
        defaults = self.defaults_provider.compute_defaults(indicator_id=indicator_id)
        window_spec = None if defaults is None else defaults.params.get("window")
        if window_spec is None:
            raise _invalid_request(
                path=f"{path}.window",
                code="unsupported_window_axis",
                message=f"indicator_id {indicator_id} does not expose a window axis",
            )
        allowed_values = set(_int_grid_values(window_spec.materialize()))
        unsupported_values = tuple(value for value in values if value not in allowed_values)
        if len(unsupported_values) > 0:
            raise _invalid_request(
                path=f"{path}.window",
                code="invalid_window",
                message=(
                    "window range contains values outside configured catalog: "
                    f"{unsupported_values[:5]}"
                ),
            )
        return {"start": start, "stop": stop, "step": step, "count": len(values)}

    def _normalize_execution(self, *, payload: Mapping[str, Any]) -> dict[str, Any]:
        defaults = self.runtime_config.execution_defaults.as_mapping()
        raw_execution = payload.get("execution", {})
        if raw_execution is None:
            raw_execution = {}
        if not isinstance(raw_execution, Mapping):
            raise _invalid_request(
                path="execution",
                code="invalid_type",
                message="execution must be an object when provided",
            )

        execution: dict[str, Any] = dict(defaults)
        for key in (
            "direction_mode",
            "fee_rate",
            "slippage_rate",
            "initial_cash_quote",
            "sizing",
            "profit_lock",
            "close_on_end",
        ):
            if key in raw_execution:
                execution[key] = raw_execution[key]

        direction_mode = _string_choice(
            execution["direction_mode"],
            path="execution.direction_mode",
            allowed=BACKTEST_DIRECTION_MODES_V1,
        )
        fee_rate = _non_negative_float(execution["fee_rate"], path="execution.fee_rate")
        slippage_rate = _non_negative_float(
            execution["slippage_rate"],
            path="execution.slippage_rate",
        )
        initial_cash_quote = _positive_float(
            execution["initial_cash_quote"],
            path="execution.initial_cash_quote",
        )
        sizing = _normalize_sizing(execution["sizing"])
        profit_lock = _normalize_profit_lock(execution["profit_lock"])
        close_on_end = _strict_bool(execution["close_on_end"], path="execution.close_on_end")
        return {
            "direction_mode": direction_mode,
            "fee_rate": fee_rate,
            "slippage_rate": slippage_rate,
            "initial_cash_quote": initial_cash_quote,
            "sizing": sizing,
            "profit_lock": profit_lock,
            "close_on_end": close_on_end,
        }

    def _normalize_ranking(self, *, payload: Mapping[str, Any]) -> dict[str, str]:
        raw_ranking = payload.get("ranking", {})
        if raw_ranking is None:
            raw_ranking = {}
        if not isinstance(raw_ranking, Mapping):
            raise _invalid_request(
                path="ranking",
                code="invalid_type",
                message="ranking must be an object when provided",
            )
        primary_metric = _string_choice(
            raw_ranking.get("primary_metric", DEFAULT_BACKTEST_RANKING_V1["primary_metric"]),
            path="ranking.primary_metric",
            allowed=BACKTEST_RANKING_METRICS_V1,
        )
        direction = _string_choice(
            raw_ranking.get("direction", DEFAULT_BACKTEST_RANKING_V1["direction"]),
            path="ranking.direction",
            allowed=("asc", "desc"),
        )
        return {"primary_metric": primary_metric, "direction": direction}

    def _normalize_top_n(self, *, payload: Mapping[str, Any]) -> int:
        raw_top_n = payload.get("top_n", DEFAULT_BACKTEST_TOP_N_V1)
        return _positive_int(raw_top_n, path="top_n")

    def _normalize_risk(
        self,
        *,
        payload: Mapping[str, Any],
        guardrails: BacktestRuntimeGuardrails,
    ) -> tuple[dict[str, Any], int]:
        raw_risk = payload.get("risk")
        if not isinstance(raw_risk, Mapping):
            raise _invalid_request(
                path="risk",
                code="required",
                message="risk.mode is required",
            )
        mode = _string_choice(
            raw_risk.get("mode"),
            path="risk.mode",
            allowed=BACKTEST_RISK_MODES_V1,
        )
        if mode == "none":
            return {"mode": "none"}, 0

        tp_levels = _normalize_percent_levels(raw_risk.get("tp"), path="risk.tp")
        sl_levels = _normalize_percent_levels(raw_risk.get("sl"), path="risk.sl")
        cells = len(tp_levels) * len(sl_levels)
        if cells > guardrails.max_tp_sl_cells:
            raise BacktestPreflightRejected(
                error_code=BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE,
                message="Backtest TP/SL grid exceeds configured cell guardrail",
                issues=(
                    BacktestValidationIssue(
                        path="risk",
                        code="max_tp_sl_cells",
                        message=f"tp/sl cells must be <= {guardrails.max_tp_sl_cells}",
                    ),
                ),
            )

        configured_tp = {
            _decimal_level(value) for value in self.runtime_config.hit_times_tp_levels_pct
        }
        configured_sl = {
            _decimal_level(value) for value in self.runtime_config.hit_times_sl_levels_pct
        }
        requested_tp = {_decimal_level(value) for value in tp_levels}
        requested_sl = {_decimal_level(value) for value in sl_levels}
        if not requested_tp.issubset(configured_tp) or not requested_sl.issubset(configured_sl):
            raise BacktestPreflightRejected(
                error_code=BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED,
                message="Requested TP/SL grid is not covered by configured hit_times/15m grid",
                issues=(
                    BacktestValidationIssue(
                        path="risk",
                        code="tp_sl_grid_not_covered",
                        message="Requested TP/SL levels must be covered by hit_times/15m",
                    ),
                ),
            )
        return (
            {
                "mode": "tp_sl_grid",
                "tp": _level_range_from_values(tp_levels),
                "sl": _level_range_from_values(sl_levels),
            },
            cells,
        )

    def _cost_guardrail_issues(
        self,
        *,
        indicators: Sequence[Mapping[str, Any]],
        indicator_rows: int,
        candidate_combinations: int,
        tp_sl_cells: int,
        top_n: int,
        guardrails: BacktestRuntimeGuardrails,
    ) -> tuple[BacktestValidationIssue, ...]:
        issues: list[BacktestValidationIssue] = []
        if len(indicators) > guardrails.max_indicator_arity:
            issues.append(
                BacktestValidationIssue(
                    path="indicators",
                    code="max_indicator_arity",
                    message=f"indicator arity must be <= {guardrails.max_indicator_arity}",
                )
            )
        if indicator_rows > guardrails.max_indicator_rows:
            issues.append(
                BacktestValidationIssue(
                    path="indicators",
                    code="max_indicator_rows",
                    message=f"indicator rows must be <= {guardrails.max_indicator_rows}",
                )
            )
        if candidate_combinations > guardrails.max_candidate_combinations:
            issues.append(
                BacktestValidationIssue(
                    path="indicators",
                    code="max_candidate_combinations",
                    message=(
                        "candidate combinations must be <= "
                        f"{guardrails.max_candidate_combinations}"
                    ),
                )
            )
        if tp_sl_cells > guardrails.max_tp_sl_cells:
            issues.append(
                BacktestValidationIssue(
                    path="risk",
                    code="max_tp_sl_cells",
                    message=f"tp/sl cells must be <= {guardrails.max_tp_sl_cells}",
                )
            )
        if top_n > guardrails.max_top_n:
            issues.append(
                BacktestValidationIssue(
                    path="top_n",
                    code="max_top_n",
                    message=f"top_n must be <= {guardrails.max_top_n}",
                )
            )
        return tuple(issues)

    def _guardrail_warnings(
        self,
        *,
        cost_estimate: BacktestCostEstimate,
        guardrails: BacktestRuntimeGuardrails,
    ) -> tuple[BacktestValidationIssue, ...]:
        warnings: list[BacktestValidationIssue] = []
        if cost_estimate.indicator_rows >= int(guardrails.max_indicator_rows * 0.8):
            warnings.append(
                BacktestValidationIssue(
                    path="indicators",
                    code="near_max_indicator_rows",
                    message="indicator rows are close to the configured preflight limit",
                )
            )
        if cost_estimate.candidate_combinations >= int(
            guardrails.max_candidate_combinations * 0.8
        ):
            warnings.append(
                BacktestValidationIssue(
                    path="indicators",
                    code="near_max_candidate_combinations",
                    message="candidate combinations are close to the configured preflight limit",
                )
            )
        if (
            cost_estimate.tp_sl_cells > 0
            and cost_estimate.tp_sl_cells >= int(guardrails.max_tp_sl_cells * 0.8)
        ):
            warnings.append(
                BacktestValidationIssue(
                    path="risk",
                    code="near_max_tp_sl_cells",
                    message="tp/sl cells are close to the configured preflight limit",
                )
            )
        return tuple(warnings)

    def _result_config_hash(self) -> str:
        payload = {
            "artifact_config_hash": self.runtime_config.artifact_config_hash,
            "execution_defaults": self.runtime_config.execution_defaults.as_mapping(),
            "guardrails": self.runtime_config.guardrails.as_mapping(),
            "ranking_default": DEFAULT_BACKTEST_RANKING_V1,
            "supported_timeframes": SUPPORTED_BACKTEST_TIMEFRAMES_V1,
            "top_n_default": DEFAULT_BACKTEST_TOP_N_V1,
            "hit_times_grid": {
                "tp_levels_pct": self.runtime_config.hit_times_tp_levels_pct,
                "sl_levels_pct": self.runtime_config.hit_times_sl_levels_pct,
            },
        }
        return _canonical_json_sha256(payload)

    def _resolve_artifact_metadata(
        self,
        *,
        coordinates: BacktestCoordinates,
    ) -> BacktestArtifactMetadata:
        try:
            return self.artifact_context_resolver.resolve_context(coordinates=coordinates)
        except (BacktestArtifactContextUnavailable, FileNotFoundError, ValueError) as error:
            raise BacktestPreflightRejected(
                error_code=BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE,
                message="Required backtest artifacts are unavailable",
                issues=(
                    BacktestValidationIssue(
                        path="artifacts",
                        code="artifacts_unavailable",
                        message=str(error),
                    ),
                ),
                retryable=True,
            ) from error


def _invalid_request(*, path: str, code: str, message: str) -> BacktestPreflightRejected:
    return BacktestPreflightRejected(
        error_code=BACKTEST_ERROR_INVALID_REQUEST,
        message="Backtest preflight request is invalid",
        issues=(BacktestValidationIssue(path=path, code=code, message=message),),
    )


def _normalize_token(
    value: Any,
    *,
    path: str,
    lower: bool = False,
    upper: bool = False,
) -> str:
    if not isinstance(value, str):
        raise _invalid_request(path=path, code="required", message=f"{path} must be string")
    normalized = value.strip()
    if lower:
        normalized = normalized.lower()
    if upper:
        normalized = normalized.upper()
    if not normalized or "/" in normalized or "\\" in normalized or ".." in normalized:
        raise _invalid_request(
            path=path,
            code="invalid_token",
            message=f"{path} must be a non-empty safe path token",
        )
    if any(char.isspace() for char in normalized):
        raise _invalid_request(
            path=path,
            code="invalid_token",
            message=f"{path} must not contain whitespace",
        )
    return normalized


def _normalize_indicator_id(value: Any, *, path: str) -> str:
    if not isinstance(value, str):
        raise _invalid_request(path=path, code="required", message="indicator_id is required")
    indicator_id = value.strip().lower()
    if not indicator_id or "/" in indicator_id or "\\" in indicator_id or ".." in indicator_id:
        raise _invalid_request(
            path=path,
            code="invalid_indicator",
            message="indicator_id must be a non-empty safe token",
        )
    return indicator_id


def _parse_utc_timestamp(value: Any, *, path: str) -> datetime:
    if not isinstance(value, str):
        raise _invalid_request(path=path, code="required", message=f"{path} must be string")
    raw_value = value.strip()
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _invalid_request(
            path=path,
            code="invalid_datetime",
            message=f"{path} must be an ISO-8601 UTC timestamp",
        ) from error
    if parsed.tzinfo is None:
        raise _invalid_request(
            path=path,
            code="timezone_required",
            message=f"{path} must include UTC timezone",
        )
    parsed_utc = parsed.astimezone(UTC)
    if parsed_utc.second != 0 or parsed_utc.microsecond != 0:
        raise _invalid_request(
            path=path,
            code="invalid_open_time",
            message=f"{path} must be aligned to minute-level 15m open_time semantics",
        )
    return parsed_utc


def _format_utc_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_request(path=path, code="invalid_type", message=f"{path} must be integer")
    if value <= 0:
        raise _invalid_request(path=path, code="invalid_value", message=f"{path} must be > 0")
    return value


def _positive_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _invalid_request(path=path, code="invalid_type", message=f"{path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise _invalid_request(path=path, code="invalid_value", message=f"{path} must be > 0")
    return numeric


def _non_negative_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _invalid_request(path=path, code="invalid_type", message=f"{path} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise _invalid_request(path=path, code="invalid_value", message=f"{path} must be >= 0")
    return numeric


def _strict_bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise _invalid_request(path=path, code="invalid_type", message=f"{path} must be boolean")
    return value


def _string_choice(value: Any, *, path: str, allowed: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise _invalid_request(path=path, code="invalid_type", message=f"{path} must be string")
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise _invalid_request(
            path=path,
            code="unsupported_value",
            message=f"{path} must be one of {tuple(allowed)}",
        )
    return normalized


def _normalize_sizing(value: Any) -> dict[str, float | str]:
    if not isinstance(value, Mapping):
        raise _invalid_request(
            path="execution.sizing",
            code="invalid_type",
            message="execution.sizing must be an object",
        )
    mode = _string_choice(
        value.get("mode"),
        path="execution.sizing.mode",
        allowed=BACKTEST_SIZING_MODES_V1,
    )
    if mode == "all_in":
        return {"mode": mode}
    if mode == "fixed_quote":
        return {
            "mode": mode,
            "quote_amount": _positive_float(
                value.get("quote_amount"),
                path="execution.sizing.quote_amount",
            ),
        }
    if mode == "fixed_equity_pct":
        return {
            "mode": mode,
            "equity_pct": _equity_pct(value.get("equity_pct")),
        }
    if mode == "fixed_equity_pct_min_quote":
        return {
            "mode": mode,
            "equity_pct": _equity_pct(value.get("equity_pct")),
            "min_quote": _positive_float(value.get("min_quote"), path="execution.sizing.min_quote"),
        }
    return {
        "mode": mode,
        "equity_pct": _equity_pct(value.get("equity_pct")),
        "max_quote": _positive_float(value.get("max_quote"), path="execution.sizing.max_quote"),
    }


def _equity_pct(value: Any) -> float:
    equity_pct = _positive_float(value, path="execution.sizing.equity_pct")
    if equity_pct > 100.0:
        raise _invalid_request(
            path="execution.sizing.equity_pct",
            code="invalid_value",
            message="execution.sizing.equity_pct must be <= 100",
        )
    return equity_pct


def _normalize_profit_lock(value: Any) -> dict[str, bool | float]:
    if not isinstance(value, Mapping):
        raise _invalid_request(
            path="execution.profit_lock",
            code="invalid_type",
            message="execution.profit_lock must be an object",
        )
    enabled = _strict_bool(value.get("enabled", False), path="execution.profit_lock.enabled")
    if not enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "safe_profit_percent": _non_negative_float(
            value.get("safe_profit_percent"),
            path="execution.profit_lock.safe_profit_percent",
        ),
    }


def _normalize_percent_levels(value: Any, *, path: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        start = _positive_decimal(value.get("start_pct"), path=f"{path}.start_pct")
        stop = _positive_decimal(value.get("stop_pct"), path=f"{path}.stop_pct")
        step = _positive_decimal(value.get("step_pct"), path=f"{path}.step_pct")
        if start > stop:
            raise _invalid_request(
                path=path,
                code="invalid_range",
                message=f"{path}.start_pct must be <= {path}.stop_pct",
            )
        levels: list[float] = []
        current = start
        while current <= stop:
            levels.append(float(current))
            current += step
        if len(levels) == 0:
            raise _invalid_request(
                path=path,
                code="empty",
                message=f"{path} must materialize at least one level",
            )
        return tuple(levels)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        explicit_levels = tuple(
            float(_positive_decimal(item, path=f"{path}[]")) for item in value
        )
        if len(explicit_levels) == 0:
            raise _invalid_request(
                path=path,
                code="empty",
                message=f"{path} must contain at least one level",
            )
        return tuple(sorted(set(explicit_levels)))
    raise _invalid_request(
        path=path,
        code="invalid_type",
        message=f"{path} must be a range object or explicit level list",
    )


def _positive_decimal(value: Any, *, path: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise _invalid_request(path=path, code="invalid_type", message=f"{path} must be numeric")
    try:
        numeric = Decimal(str(value))
    except Exception as error:
        raise _invalid_request(
            path=path,
            code="invalid_value",
            message=f"{path} is invalid",
        ) from error
    if numeric <= 0:
        raise _invalid_request(path=path, code="invalid_value", message=f"{path} must be > 0")
    return numeric


def _decimal_level(value: float | int | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.000001"))


def _level_range_from_values(values: Sequence[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"start_pct": values[0], "stop_pct": values[0], "step_pct": values[0]}
    sorted_values = tuple(sorted(values))
    step = sorted_values[1] - sorted_values[0]
    return {"start_pct": sorted_values[0], "stop_pct": sorted_values[-1], "step_pct": step}


def _int_grid_values(values: Sequence[GridValue]) -> tuple[int, ...]:
    int_values: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        int_values.append(value)
    return tuple(int_values)


def _cost_class(*, candidate_combinations: int) -> str:
    if candidate_combinations <= 1_000:
        return "small"
    if candidate_combinations <= 50_000:
        return "medium"
    return "large"


def _preflight_scheduling_class(
    *,
    estimated_combinations_upper_bound: int,
    arity: int,
) -> str:
    if estimated_combinations_upper_bound <= 0:
        return "heavy"
    if arity <= 0:
        return "heavy"
    if estimated_combinations_upper_bound > DEFAULT_LIGHT_ESTIMATED_COMBINATIONS:
        return "heavy"
    return "light_candidate"


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    canonical_json = json.dumps(
        _normalize_json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be hashed")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


__all__ = [
    "BACKTEST_DIRECTION_MODES_V1",
    "BACKTEST_ERROR_ARTIFACTS_UNAVAILABLE",
    "BACKTEST_ERROR_INVALID_REQUEST",
    "BACKTEST_ERROR_REQUEST_TOO_EXPENSIVE",
    "BACKTEST_ERROR_TP_SL_GRID_NOT_COVERED",
    "BACKTEST_RANKING_METRICS_V1",
    "BACKTEST_RISK_MODES_V1",
    "BACKTEST_SIZING_MODES_V1",
    "DEFAULT_BACKTEST_TOP_N_V1",
    "SUPPORTED_BACKTEST_TIMEFRAMES_V1",
    "BacktestPreflightRejected",
    "BacktestPreflightService",
    "BacktestRuntimeConfig",
    "BacktestRuntimeDefaultsService",
]
