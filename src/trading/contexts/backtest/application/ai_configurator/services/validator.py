from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from jsonschema import Draft202012Validator

from trading.contexts.backtest.application.dto.runtime_preflight import (
    BacktestValidationIssue,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestPreflightRejected,
    BacktestPreflightService,
)

from .catalog import BacktestAiAllowedCatalog
from .security import BacktestAiOutputGate

BacktestAiValidationStatus = Literal["ready", "needs_clarification", "blocked_by_policy"]

_MODEL_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "mode",
        "status",
        "assistant_message",
        "assumptions",
        "warnings",
        "config",
        "suggestions",
    ],
    "properties": {
        "schema_version": {"const": 1},
        "mode": {
            "type": "string",
            "enum": ["create", "edit", "repair", "suggest_safer", "explain"],
        },
        "status": {"type": "string", "enum": ["config_ready", "needs_clarification"]},
        "assistant_message": {"type": "string", "minLength": 1, "maxLength": 1200},
        "assumptions": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "warnings": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "config": {
            "anyOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "coordinates",
                        "timeframe",
                        "time_range",
                        "indicators",
                        "risk",
                        "execution",
                        "ranking",
                        "top_n",
                    ],
                    "properties": {
                        "coordinates": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["exchange", "market_type", "symbol"],
                            "properties": {
                                "exchange": {"type": "string"},
                                "market_type": {"type": "string"},
                                "symbol": {"type": "string"},
                            },
                        },
                        "timeframe": {"type": "string"},
                        "time_range": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["start", "end"],
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                            },
                        },
                        "indicators": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 10,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["indicator_id", "sources", "window"],
                                "properties": {
                                    "indicator_id": {"type": "string"},
                                    "sources": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "maxItems": 8,
                                    },
                                    "window": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["start", "stop", "step"],
                                        "properties": {
                                            "start": {"type": "integer", "minimum": 1},
                                            "stop": {"type": "integer", "minimum": 1},
                                            "step": {"type": "integer", "minimum": 1},
                                        },
                                    },
                                },
                            },
                        },
                        "risk": {
                            "type": "object",
                            "additionalProperties": True,
                            "required": ["mode"],
                            "properties": {"mode": {"type": "string"}},
                        },
                        "execution": {"type": "object"},
                        "ranking": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["primary_metric", "direction"],
                            "properties": {
                                "primary_metric": {"type": "string"},
                                "direction": {"type": "string", "enum": ["asc", "desc"]},
                            },
                        },
                        "top_n": {"type": "integer", "minimum": 1},
                    },
                },
                {"type": "null"},
            ]
        },
        "suggestions": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
}
_SCHEMA_VALIDATOR = Draft202012Validator(_MODEL_OUTPUT_SCHEMA)


@dataclass(frozen=True, slots=True)
class BacktestAiConfigValidationOutcome:
    status: BacktestAiValidationStatus
    assistant_message: str
    validated_config: dict[str, Any] | None = None
    warnings: tuple[dict[str, Any], ...] = ()
    suggestions: tuple[dict[str, Any], ...] = ()
    validation_errors: tuple[dict[str, Any], ...] = ()
    last_error: str | None = None
    last_error_json: dict[str, Any] | None = None

    @property
    def loadable(self) -> bool:
        return self.status == "ready" and self.validated_config is not None


@dataclass(frozen=True, slots=True)
class BacktestAiConfigValidator:
    preflight_service: BacktestPreflightService
    output_gate: BacktestAiOutputGate = BacktestAiOutputGate()

    def validate_model_output(
        self,
        *,
        raw_output: str,
        catalog: BacktestAiAllowedCatalog,
    ) -> BacktestAiConfigValidationOutcome:
        parsed_result = _parse_json_object(raw_output)
        if parsed_result is None:
            return _needs_clarification(
                assistant_message="I could not read a valid configuration draft.",
                issues=(
                    {
                        "path": "body",
                        "code": "invalid_json",
                        "message": "Model output must be a single JSON object",
                    },
                ),
            )
        output_gate = self.output_gate.evaluate(
            raw_output=raw_output,
            parsed=parsed_result,
            catalog=catalog,
        )
        if not output_gate.allowed:
            issues = tuple(issue.as_mapping() for issue in output_gate.issues)
            if _only_catalog_output_issues(issues=issues):
                return _needs_clarification(
                    assistant_message=str(
                        parsed_result.get(
                            "assistant_message",
                            "The generated configuration used unsupported values.",
                        )
                    ),
                    issues=issues,
                    last_error="output_catalog_validation_failed",
                )
            return BacktestAiConfigValidationOutcome(
                status="blocked_by_policy",
                assistant_message=(
                    "The generated response did not pass safety checks. "
                    "Please rephrase the backtest request."
                ),
                validation_errors=issues,
                last_error="output_gate_rejected",
                last_error_json={"issues": list(issues)},
            )

        schema_errors = tuple(
            _schema_issue(error) for error in _SCHEMA_VALIDATOR.iter_errors(parsed_result)
        )
        if schema_errors:
            return _needs_clarification(
                assistant_message="The generated configuration did not match the required schema.",
                issues=schema_errors,
            )

        if parsed_result["status"] == "needs_clarification":
            return BacktestAiConfigValidationOutcome(
                status="needs_clarification",
                assistant_message=str(parsed_result["assistant_message"]),
                warnings=_warning_items(parsed_result),
                suggestions=_suggestion_items(parsed_result),
                validation_errors=(),
                last_error="needs_clarification",
                last_error_json=None,
            )

        config = parsed_result.get("config")
        if not isinstance(config, Mapping):
            return _needs_clarification(
                assistant_message="The generated configuration is incomplete.",
                issues=(
                    {
                        "path": "config",
                        "code": "required",
                        "message": "config is required for config_ready status",
                    },
                ),
            )

        catalog_issues = _catalog_issues(config=config, catalog=catalog)
        if catalog_issues:
            return _needs_clarification(
                assistant_message=str(parsed_result["assistant_message"]),
                issues=catalog_issues,
                warnings=_warning_items(parsed_result),
                suggestions=_suggestion_items(parsed_result),
                last_error="catalog_validation_failed",
            )

        try:
            preflight = self.preflight_service.execute(config)
        except BacktestPreflightRejected as error:
            issues = tuple(issue.as_mapping() for issue in error.issues)
            return _needs_clarification(
                assistant_message=str(parsed_result["assistant_message"]),
                issues=issues,
                warnings=_warning_items(parsed_result),
                suggestions=_suggestion_items(parsed_result),
                last_error=error.error_code,
                last_error_json=error.details(),
            )

        normalized = dict(preflight.normalized_request)
        warnings = _warning_items(parsed_result) + tuple(
            _warning_from_issue(issue) for issue in preflight.warnings
        )
        return BacktestAiConfigValidationOutcome(
            status="ready",
            assistant_message=str(parsed_result["assistant_message"]),
            validated_config=normalized,
            warnings=warnings,
            suggestions=_suggestion_items(parsed_result),
            validation_errors=(),
        )


def _parse_json_object(raw_output: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _schema_issue(error: Any) -> dict[str, str]:
    path = ".".join(str(part) for part in error.absolute_path) or "body"
    return {
        "path": path,
        "code": "schema_validation_failed",
        "message": str(error.message),
    }


def _only_catalog_output_issues(*, issues: tuple[dict[str, str], ...]) -> bool:
    catalog_codes = {
        "multi_symbol_field_not_allowed",
        "unsupported_config_field",
        "unsupported_exchange",
        "unsupported_indicator",
        "unsupported_market_type",
        "unsupported_symbol",
        "unsupported_timeframe",
    }
    return bool(issues) and all(issue.get("code") in catalog_codes for issue in issues)


def _catalog_issues(
    *,
    config: Mapping[str, Any],
    catalog: BacktestAiAllowedCatalog,
) -> tuple[dict[str, str], ...]:
    issues: list[dict[str, str]] = []
    coordinates = config.get("coordinates")
    if not isinstance(coordinates, Mapping):
        return (
            {
                "path": "coordinates",
                "code": "required",
                "message": "coordinates must be an object",
            },
        )
    _catalog_choice(
        value=coordinates.get("exchange"),
        allowed=catalog.exchanges,
        path="coordinates.exchange",
        code="unsupported_exchange",
        issues=issues,
    )
    _catalog_choice(
        value=coordinates.get("market_type"),
        allowed=catalog.market_types,
        path="coordinates.market_type",
        code="unsupported_market_type",
        issues=issues,
    )
    _catalog_choice(
        value=coordinates.get("symbol"),
        allowed=catalog.symbols,
        path="coordinates.symbol",
        code="unsupported_symbol",
        issues=issues,
        upper=True,
    )
    _catalog_choice(
        value=config.get("timeframe"),
        allowed=catalog.timeframes,
        path="timeframe",
        code="unsupported_timeframe",
        issues=issues,
    )
    for index, indicator in enumerate(config.get("indicators") or []):
        if not isinstance(indicator, Mapping):
            continue
        indicator_id = indicator.get("indicator_id")
        _catalog_choice(
            value=indicator_id,
            allowed=catalog.indicator_ids,
            path=f"indicators.{index}.indicator_id",
            code="unsupported_indicator",
            issues=issues,
        )
        if not isinstance(indicator_id, str):
            continue
        catalog_item = catalog.indicator_by_id(indicator_id)
        if catalog_item is None:
            continue
        allowed_sources = set(catalog_item.sources)
        for source_index, source in enumerate(indicator.get("sources") or []):
            if (
                isinstance(source, str)
                and allowed_sources
                and source.strip().lower() not in allowed_sources
            ):
                issues.append(
                    _issue(
                        path=f"indicators.{index}.sources.{source_index}",
                        code="unsupported_source",
                        message="source is not supported by the allowed catalog",
                    )
                )
    risk = config.get("risk")
    if isinstance(risk, Mapping):
        _catalog_choice(
            value=risk.get("mode"),
            allowed=catalog.risk_modes,
            path="risk.mode",
            code="unsupported_risk_mode",
            issues=issues,
        )
    execution = config.get("execution")
    if isinstance(execution, Mapping):
        _catalog_choice(
            value=execution.get("direction_mode"),
            allowed=catalog.direction_modes,
            path="execution.direction_mode",
            code="unsupported_direction_mode",
            issues=issues,
        )
        sizing = execution.get("sizing")
        if isinstance(sizing, Mapping):
            _catalog_choice(
                value=sizing.get("mode"),
                allowed=catalog.sizing_modes,
                path="execution.sizing.mode",
                code="unsupported_sizing_mode",
                issues=issues,
            )
    ranking = config.get("ranking")
    if isinstance(ranking, Mapping):
        _catalog_choice(
            value=ranking.get("primary_metric"),
            allowed=catalog.ranking_metrics,
            path="ranking.primary_metric",
            code="unsupported_ranking_metric",
            issues=issues,
        )
    if "symbols" in config:
        issues.append(
            _issue(
                path="symbols",
                code="multi_symbol_field_not_allowed",
                message="MVP accepts one coordinates.symbol value, not symbols[]",
            )
        )
    if "strategy" in config:
        issues.append(
            _issue(
                path="strategy",
                code="unsupported_strategy_field",
                message="strategy is not part of the current /backtests job payload",
            )
        )
    return tuple(issues)


def _catalog_choice(
    *,
    value: Any,
    allowed: tuple[str, ...],
    path: str,
    code: str,
    issues: list[dict[str, str]],
    upper: bool = False,
) -> None:
    if not isinstance(value, str):
        return
    normalized = value.strip().upper() if upper else value.strip().lower()
    allowed_set = {item.upper() if upper else item.lower() for item in allowed}
    if normalized not in allowed_set:
        issues.append(
            _issue(
                path=path,
                code=code,
                message=f"{path} is not supported by the allowed catalog",
            )
        )


def _needs_clarification(
    *,
    assistant_message: str,
    issues: tuple[dict[str, str], ...],
    warnings: tuple[dict[str, Any], ...] = (),
    suggestions: tuple[dict[str, Any], ...] = (),
    last_error: str = "validation_failed",
    last_error_json: dict[str, Any] | None = None,
) -> BacktestAiConfigValidationOutcome:
    return BacktestAiConfigValidationOutcome(
        status="needs_clarification",
        assistant_message=assistant_message,
        warnings=warnings,
        suggestions=suggestions,
        validation_errors=issues,
        last_error=last_error,
        last_error_json=last_error_json or {"issues": list(issues)},
    )


def _warning_items(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"kind": "warning", "message": str(item)}
        for item in payload.get("warnings", [])
        if str(item).strip()
    )


def _suggestion_items(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {"kind": "suggestion", "message": str(item)}
        for item in payload.get("suggestions", [])
        if str(item).strip()
    )


def _warning_from_issue(issue: BacktestValidationIssue) -> dict[str, str]:
    return {"kind": "warning", "path": issue.path, "code": issue.code, "message": issue.message}


def _issue(*, path: str, code: str, message: str) -> dict[str, str]:
    return {"path": path, "code": code, "message": message}


__all__ = [
    "BacktestAiConfigValidationOutcome",
    "BacktestAiConfigValidator",
    "BacktestAiValidationStatus",
]
