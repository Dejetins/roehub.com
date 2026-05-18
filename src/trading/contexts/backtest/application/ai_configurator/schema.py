from __future__ import annotations

from copy import deepcopy
from typing import Any

BACKTEST_AI_OUTPUT_SCHEMA_NAME = "backtest_ai_configurator_assistant_v1_output"
BACKTEST_AI_OUTPUT_SCHEMA_VERSION = 1

_CONFIG_SCHEMA: dict[str, Any] = {
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
                "exchange": {"type": "string", "minLength": 1, "maxLength": 32},
                "market_type": {"type": "string", "minLength": 1, "maxLength": 32},
                "symbol": {"type": "string", "minLength": 1, "maxLength": 32},
            },
        },
        "timeframe": {"type": "string", "minLength": 1, "maxLength": 16},
        "time_range": {
            "type": "object",
            "additionalProperties": False,
            "required": ["start", "end"],
            "properties": {
                "start": {"type": "string", "minLength": 1, "maxLength": 40},
                "end": {"type": "string", "minLength": 1, "maxLength": 40},
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
                    "indicator_id": {"type": "string", "minLength": 1, "maxLength": 80},
                    "sources": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": 32},
                    },
                    "window": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["start", "stop", "step"],
                        "properties": {
                            "start": {"type": "integer", "minimum": 1, "maximum": 10000},
                            "stop": {"type": "integer", "minimum": 1, "maximum": 10000},
                            "step": {"type": "integer", "minimum": 1, "maximum": 10000},
                        },
                    },
                },
            },
        },
        "risk": {
            "type": "object",
            "additionalProperties": True,
            "required": ["mode"],
            "properties": {"mode": {"type": "string", "minLength": 1, "maxLength": 40}},
        },
        "execution": {"type": "object", "additionalProperties": True},
        "ranking": {
            "type": "object",
            "additionalProperties": False,
            "required": ["primary_metric", "direction"],
            "properties": {
                "primary_metric": {"type": "string", "minLength": 1, "maxLength": 80},
                "direction": {"type": "string", "enum": ["asc", "desc"]},
            },
        },
        "top_n": {"type": "integer", "minimum": 1, "maximum": 1000},
    },
}

_MODEL_OUTPUT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "intent",
        "status",
        "assistant_message",
        "conversation_title",
        "config",
        "unsupported_items",
        "clarifying_questions",
        "warnings",
    ],
    "properties": {
        "schema_version": {"const": BACKTEST_AI_OUTPUT_SCHEMA_VERSION},
        "intent": {
            "type": "string",
            "enum": [
                "create_config",
                "edit_current_config",
                "explain_current_config",
                "repair_invalid_config",
                "suggest_safer_config",
                "list_available_indicators",
                "list_available_symbols",
                "list_available_parameters",
                "unsupported_or_offtopic",
            ],
        },
        "status": {
            "type": "string",
            "enum": [
                "config_ready",
                "informational",
                "needs_clarification",
                "unsupported_request",
                "blocked_by_policy",
            ],
        },
        "assistant_message": {"type": "string", "minLength": 1, "maxLength": 1200},
        "conversation_title": {"type": "string", "minLength": 1, "maxLength": 60},
        "config": {"oneOf": [_CONFIG_SCHEMA, {"type": "null"}]},
        "unsupported_items": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "minLength": 1, "maxLength": 120},
        },
        "clarifying_questions": {
            "type": "array",
            "maxItems": 6,
            "items": {"type": "string", "minLength": 1, "maxLength": 240},
        },
        "warnings": {
            "type": "array",
            "maxItems": 8,
            "items": {"type": "string", "minLength": 1, "maxLength": 240},
        },
    },
}

_LMSTUDIO_ENVELOPE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "intent",
        "status",
        "assistant_message",
        "conversation_title",
        "config",
        "unsupported_items",
        "clarifying_questions",
        "warnings",
    ],
    "properties": {
        "schema_version": {"const": BACKTEST_AI_OUTPUT_SCHEMA_VERSION},
        "intent": {"type": "string"},
        "status": {"type": "string"},
        "assistant_message": {"type": "string"},
        "conversation_title": {"type": "string"},
        "config": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": True,
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
                            "additionalProperties": True,
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
                            "additionalProperties": True,
                            "required": ["start", "end"],
                            "properties": {
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                            },
                        },
                        "indicators": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": True,
                                "required": ["indicator_id", "sources", "window"],
                                "properties": {
                                    "indicator_id": {"type": "string"},
                                    "sources": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "window": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "required": ["start", "stop", "step"],
                                        "properties": {
                                            "start": {"type": "integer"},
                                            "stop": {"type": "integer"},
                                            "step": {"type": "integer"},
                                        },
                                    },
                                },
                            },
                        },
                        "risk": {"type": "object", "additionalProperties": True},
                        "execution": {"type": "object", "additionalProperties": True},
                        "ranking": {
                            "type": "object",
                            "additionalProperties": True,
                            "required": ["primary_metric", "direction"],
                            "properties": {
                                "primary_metric": {"type": "string"},
                                "direction": {"type": "string"},
                            },
                        },
                        "top_n": {"type": "integer"},
                    },
                },
                {"type": "null"},
            ]
        },
        "unsupported_items": {"type": "array", "items": {"type": "string"}},
        "clarifying_questions": {"type": "array", "items": {"type": "string"}},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

_OUTPUT_EXAMPLE: dict[str, Any] = {
    "schema_version": BACKTEST_AI_OUTPUT_SCHEMA_VERSION,
    "intent": "create_config",
    "status": "config_ready",
    "assistant_message": (
        "I prepared a BTCUSDT configuration on 15m with RSI. "
        "No backtest has been run."
    ),
    "conversation_title": "RSI for BTCUSDT",
    "config": {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "2023-01-01T00:00:00Z",
            "end": "2024-01-01T00:00:00Z",
        },
        "indicators": [
            {
                "indicator_id": "momentum.rsi",
                "sources": ["close"],
                "window": {"start": 14, "stop": 14, "step": 1},
            }
        ],
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": 0.00075,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000.0,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10.0},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {
            "primary_metric": "total_return_pct",
            "direction": "desc",
        },
        "top_n": 10,
    },
    "unsupported_items": [],
    "clarifying_questions": [],
    "warnings": [],
}


def backtest_ai_model_output_schema() -> dict[str, Any]:
    return deepcopy(_MODEL_OUTPUT_SCHEMA)


def backtest_ai_output_example() -> dict[str, Any]:
    return deepcopy(_OUTPUT_EXAMPLE)


def backtest_ai_lmstudio_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": BACKTEST_AI_OUTPUT_SCHEMA_NAME,
            "strict": True,
            "schema": deepcopy(_LMSTUDIO_ENVELOPE_SCHEMA),
        },
    }


__all__ = [
    "BACKTEST_AI_OUTPUT_SCHEMA_NAME",
    "BACKTEST_AI_OUTPUT_SCHEMA_VERSION",
    "backtest_ai_lmstudio_response_format",
    "backtest_ai_model_output_schema",
    "backtest_ai_output_example",
]
