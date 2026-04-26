"""DTO package for the reset backtest scope."""

from .prepare_pools import (
    BacktestPreparePoolsConfig,
    BacktestPreparePoolsResult,
    PreparedExecutionMapping,
    PreparedIndicatorPool,
    PreparedIndicatorRowMetadata,
    PreparedSignalSegments,
    PreparePoolsTiming,
)
from .runtime_preflight import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestCostEstimate,
    BacktestExecutionDefaults,
    BacktestPreflightResult,
    BacktestRuntimeDefaults,
    BacktestRuntimeGuardrails,
    BacktestValidationIssue,
    JsonMapping,
)

__all__ = [
    "BacktestArtifactMetadata",
    "BacktestCoordinates",
    "BacktestCostEstimate",
    "BacktestExecutionDefaults",
    "BacktestPreparePoolsConfig",
    "BacktestPreparePoolsResult",
    "BacktestPreflightResult",
    "BacktestRuntimeDefaults",
    "BacktestRuntimeGuardrails",
    "BacktestValidationIssue",
    "JsonMapping",
    "PreparedExecutionMapping",
    "PreparedIndicatorPool",
    "PreparedIndicatorRowMetadata",
    "PreparedSignalSegments",
    "PreparePoolsTiming",
]
