"""DTO package for the reset backtest scope."""

from .combo_planning import (
    BacktestComboChunk,
    BacktestComboPlanningConfig,
    BacktestComboPlanningResult,
    BacktestComboPlanningTelemetry,
    BacktestExactContext,
    BacktestProxyContext,
    BacktestProxyFilterResult,
    BacktestSelectedBackend,
)
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
    "BacktestComboChunk",
    "BacktestComboPlanningConfig",
    "BacktestComboPlanningResult",
    "BacktestComboPlanningTelemetry",
    "BacktestCoordinates",
    "BacktestCostEstimate",
    "BacktestExecutionDefaults",
    "BacktestExactContext",
    "BacktestPreparePoolsConfig",
    "BacktestPreparePoolsResult",
    "BacktestPreflightResult",
    "BacktestProxyContext",
    "BacktestProxyFilterResult",
    "BacktestRuntimeDefaults",
    "BacktestRuntimeGuardrails",
    "BacktestSelectedBackend",
    "BacktestValidationIssue",
    "JsonMapping",
    "PreparedExecutionMapping",
    "PreparedIndicatorPool",
    "PreparedIndicatorRowMetadata",
    "PreparedSignalSegments",
    "PreparePoolsTiming",
]
