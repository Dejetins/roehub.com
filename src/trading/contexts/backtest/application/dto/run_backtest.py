from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import UUID

from trading.contexts.backtest.domain.entities import TradeV1
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec
from trading.shared_kernel.primitives import InstrumentId, Timeframe, TimeRange

BacktestRequestScalar = int | float | str | bool | None
BacktestSignalGridMap = Mapping[str, Mapping[str, GridParamSpec]]
BacktestSignalScalarMap = Mapping[str, Mapping[str, BacktestRequestScalar]]
_ALLOWED_DIRECTION_MODES = {"long-only", "short-only", "long-short"}
_ALLOWED_SIZING_MODES = {
    "all_in",
    "fixed_quote",
    "strategy_compound",
    "strategy_compound_profit_lock",
}


@dataclass(frozen=True, slots=True)
class BacktestRiskGridSpec:
    """
    Stage B risk axes specification with explicit enable flags and percent semantics.

    Docs:
      - docs/architecture/backtest/backtest-grid-builder-staged-runner-guards-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/grid_builder_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
    """

    sl_enabled: bool = False
    tp_enabled: bool = False
    sl: GridParamSpec | None = None
    tp: GridParamSpec | None = None

    def __post_init__(self) -> None:
        """
        Validate risk-grid semantic invariants for Stage B expansion.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            SL/TP values are percentages where `3.0 == 3%`.
        Raises:
            ValueError: If enabled axis does not provide a materializable specification.
        Side Effects:
            None.
        """
        if self.sl_enabled and self.sl is None:
            raise ValueError("BacktestRiskGridSpec.sl must be provided when sl_enabled is true")
        if self.tp_enabled and self.tp is None:
            raise ValueError("BacktestRiskGridSpec.tp must be provided when tp_enabled is true")
        if self.sl is not None and len(self.sl.materialize()) == 0:
            raise ValueError("BacktestRiskGridSpec.sl materialized to empty values")
        if self.tp is not None and len(self.tp.materialize()) == 0:
            raise ValueError("BacktestRiskGridSpec.tp materialized to empty values")


@dataclass(frozen=True, slots=True)
class RunBacktestTemplate:
    """
    Ad-hoc backtest template payload (instrument/timeframe/indicator-grid contract).

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/indicators/domain/specifications/grid_spec.py
    """

    instrument_id: InstrumentId
    timeframe: Timeframe
    indicator_grids: tuple[GridSpec, ...]
    indicator_selections: tuple[IndicatorVariantSelection, ...] = ()
    signal_grids: BacktestSignalGridMap | None = None
    risk_grid: BacktestRiskGridSpec | None = None
    direction_mode: str = "long-short"
    sizing_mode: str = "all_in"
    risk_params: Mapping[str, BacktestRequestScalar] | None = None
    execution_params: Mapping[str, BacktestRequestScalar] | None = None

    def __post_init__(self) -> None:
        """
        Validate and normalize ad-hoc template invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            EPIC-01 keeps grid/run settings minimal and defers staged execution details.
        Raises:
            ValueError: If required fields are missing or mode literals are unsupported.
        Side Effects:
            Normalizes mode literals and freezes mapping payloads into immutable mapping proxies.
        """
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestTemplate.instrument_id is required")
        if self.timeframe is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestTemplate.timeframe is required")
        if len(self.indicator_grids) == 0:
            raise ValueError("RunBacktestTemplate.indicator_grids must be non-empty")

        normalized_direction_mode = self.direction_mode.strip().lower()
        object.__setattr__(self, "direction_mode", normalized_direction_mode)
        if normalized_direction_mode not in _ALLOWED_DIRECTION_MODES:
            raise ValueError(
                "RunBacktestTemplate.direction_mode must be one of: "
                f"{sorted(_ALLOWED_DIRECTION_MODES)}"
            )

        normalized_sizing_mode = self.sizing_mode.strip().lower()
        object.__setattr__(self, "sizing_mode", normalized_sizing_mode)
        if normalized_sizing_mode not in _ALLOWED_SIZING_MODES:
            raise ValueError(
                "RunBacktestTemplate.sizing_mode must be one of: "
                f"{sorted(_ALLOWED_SIZING_MODES)}"
            )

        object.__setattr__(
            self,
            "risk_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.risk_params)),
        )
        object.__setattr__(
            self,
            "signal_grids",
            MappingProxyType(_normalize_signal_grid_mapping(values=self.signal_grids)),
        )
        resolved_risk_grid = self.risk_grid
        if resolved_risk_grid is None:
            resolved_risk_grid = BacktestRiskGridSpec()
        object.__setattr__(self, "risk_grid", resolved_risk_grid)
        object.__setattr__(
            self,
            "execution_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.execution_params)),
        )


@dataclass(frozen=True, slots=True)
class RunBacktestSavedOverrides:
    """
    Optional saved-mode override payload applied over loaded strategy snapshot template.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - apps/api/dto/backtests.py
    """

    direction_mode: str | None = None
    sizing_mode: str | None = None
    signal_grids: BacktestSignalGridMap | None = None
    risk_grid: BacktestRiskGridSpec | None = None
    risk_params: Mapping[str, BacktestRequestScalar] | None = None
    execution_params: Mapping[str, BacktestRequestScalar] | None = None

    def __post_init__(self) -> None:
        """
        Validate optional saved-mode overrides and normalize nested payload mappings.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Missing fields keep loaded saved-strategy values unchanged.
        Raises:
            ValueError: If provided mode literal is unsupported.
        Side Effects:
            Freezes mapping fields into deterministic immutable mapping proxies.
        """
        if self.direction_mode is not None:
            normalized_direction_mode = self.direction_mode.strip().lower()
            object.__setattr__(self, "direction_mode", normalized_direction_mode)
            if normalized_direction_mode not in _ALLOWED_DIRECTION_MODES:
                raise ValueError(
                    "RunBacktestSavedOverrides.direction_mode must be one of: "
                    f"{sorted(_ALLOWED_DIRECTION_MODES)}"
                )

        if self.sizing_mode is not None:
            normalized_sizing_mode = self.sizing_mode.strip().lower()
            object.__setattr__(self, "sizing_mode", normalized_sizing_mode)
            if normalized_sizing_mode not in _ALLOWED_SIZING_MODES:
                raise ValueError(
                    "RunBacktestSavedOverrides.sizing_mode must be one of: "
                    f"{sorted(_ALLOWED_SIZING_MODES)}"
                )

        object.__setattr__(
            self,
            "signal_grids",
            MappingProxyType(_normalize_signal_grid_mapping(values=self.signal_grids)),
        )
        object.__setattr__(
            self,
            "risk_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.risk_params)),
        )
        object.__setattr__(
            self,
            "execution_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.execution_params)),
        )


@dataclass(frozen=True, slots=True)
class RunBacktestRequest:
    """
    Backtest use-case request supporting both `saved` and `template` modes.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/ports/current_user.py
      - apps/api/routes
    """

    time_range: TimeRange
    strategy_id: UUID | None = None
    template: RunBacktestTemplate | None = None
    overrides: RunBacktestSavedOverrides | None = None
    warmup_bars: int | None = None
    top_k: int | None = None
    preselect: int | None = None
    top_trades_n: int | None = None

    def __post_init__(self) -> None:
        """
        Validate request-mode exclusivity and scalar override invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Exactly one mode is selected: `strategy_id` (saved) xor `template` (ad-hoc).
        Raises:
            ValueError: If mode selection or override numbers violate v1 contract.
        Side Effects:
            None.
        """
        if self.time_range is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestRequest.time_range is required")

        has_saved_mode = self.strategy_id is not None
        has_template_mode = self.template is not None
        if has_saved_mode == has_template_mode:
            raise ValueError(
                "RunBacktestRequest requires exactly one mode: strategy_id xor template"
            )
        if self.overrides is not None and not has_saved_mode:
            raise ValueError(
                "RunBacktestRequest.overrides is allowed only in saved mode"
            )

        _validate_positive_optional_int(name="warmup_bars", value=self.warmup_bars)
        _validate_positive_optional_int(name="top_k", value=self.top_k)
        _validate_positive_optional_int(name="preselect", value=self.preselect)
        _validate_positive_optional_int(name="top_trades_n", value=self.top_trades_n)

    @property
    def mode(self) -> str:
        """
        Return normalized request mode literal.

        Args:
            None.
        Returns:
            str: `saved` when `strategy_id` is used, otherwise `template`.
        Assumptions:
            Mode exclusivity has been validated during object initialization.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.strategy_id is not None:
            return "saved"
        return "template"


@dataclass(frozen=True, slots=True)
class BacktestMetricRowV1:
    """
    One deterministic reporting row rendered in `|Metric|Value|` table contract.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/table_formatter_v1.py
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    metric: str
    value: str

    def __post_init__(self) -> None:
        """
        Validate one reporting metric row payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Value string is already deterministically formatted by reporting formatter.
        Raises:
            ValueError: If metric name or formatted value is empty.
        Side Effects:
            Normalizes metric/value literals by stripping surrounding whitespace.
        """
        normalized_metric = self.metric.strip()
        object.__setattr__(self, "metric", normalized_metric)
        if not normalized_metric:
            raise ValueError("BacktestMetricRowV1.metric must be non-empty")

        normalized_value = self.value.strip()
        object.__setattr__(self, "value", normalized_value)
        if not normalized_value:
            raise ValueError("BacktestMetricRowV1.value must be non-empty")


@dataclass(frozen=True, slots=True)
class BacktestReportV1:
    """
    Deterministic reporting payload with metric rows, markdown table, and optional trades.

    Docs:
      - docs/architecture/backtest/backtest-reporting-metrics-table-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/reporting_service_v1.py
      - src/trading/contexts/backtest/application/services/metrics_calculator_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
    """

    rows: tuple[BacktestMetricRowV1, ...]
    table_md: str | None = None
    trades: tuple[TradeV1, ...] | None = None

    def __post_init__(self) -> None:
        """
        Validate deterministic report payload shape and markdown-table contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Rows are provided in fixed contract order by reporting formatter.
        Raises:
            ValueError: If rows are empty, table header is invalid, or trades are not sorted.
        Side Effects:
            None.
        """
        if len(self.rows) == 0:
            raise ValueError("BacktestReportV1.rows must be non-empty")

        if self.table_md is not None and not self.table_md.startswith("|Metric|Value|"):
            raise ValueError("BacktestReportV1.table_md must start with '|Metric|Value|'")

        if self.trades is not None:
            previous_trade_id = 0
            for trade in self.trades:
                if trade.trade_id < previous_trade_id:
                    raise ValueError("BacktestReportV1.trades must be ordered by trade_id asc")
                previous_trade_id = trade.trade_id


@dataclass(frozen=True, slots=True)
class BacktestVariantPayloadV1:
    """
    Explicit deterministic variant payload required for saveable API response contract.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/roadmap/base_milestone_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_runner_v1.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - apps/api/dto/backtests.py
    """

    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_params: BacktestSignalScalarMap | None = None
    risk_params: Mapping[str, BacktestRequestScalar] | None = None
    execution_params: Mapping[str, BacktestRequestScalar] | None = None
    direction_mode: str = "long-short"
    sizing_mode: str = "all_in"

    def __post_init__(self) -> None:
        """
        Validate payload fields and freeze nested mappings into deterministic forms.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Payload is assembled from Stage-B deterministic variant task data.
        Raises:
            ValueError: If one mode literal is unsupported.
        Side Effects:
            Normalizes mode literals and replaces mapping payloads with immutable proxies.
        """
        if len(self.indicator_selections) == 0:
            raise ValueError("BacktestVariantPayloadV1.indicator_selections must be non-empty")

        normalized_direction_mode = self.direction_mode.strip().lower()
        object.__setattr__(self, "direction_mode", normalized_direction_mode)
        if normalized_direction_mode not in _ALLOWED_DIRECTION_MODES:
            raise ValueError(
                "BacktestVariantPayloadV1.direction_mode must be one of: "
                f"{sorted(_ALLOWED_DIRECTION_MODES)}"
            )

        normalized_sizing_mode = self.sizing_mode.strip().lower()
        object.__setattr__(self, "sizing_mode", normalized_sizing_mode)
        if normalized_sizing_mode not in _ALLOWED_SIZING_MODES:
            raise ValueError(
                "BacktestVariantPayloadV1.sizing_mode must be one of: "
                f"{sorted(_ALLOWED_SIZING_MODES)}"
            )

        object.__setattr__(
            self,
            "signal_params",
            MappingProxyType(_normalize_nested_scalar_mapping(values=self.signal_params)),
        )
        object.__setattr__(
            self,
            "risk_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.risk_params)),
        )
        object.__setattr__(
            self,
            "execution_params",
            MappingProxyType(_normalize_scalar_mapping(values=self.execution_params)),
        )


@dataclass(frozen=True, slots=True)
class BacktestVariantPreview:
    """
    One deterministic variant preview identity returned by skeleton use-case.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/domain/value_objects/variant_identity.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - apps/api/routes
    """

    variant_index: int
    variant_key: str
    indicator_variant_key: str
    total_return_pct: float = 0.0
    payload: BacktestVariantPayloadV1 | None = None
    report: BacktestReportV1 | None = None

    def __post_init__(self) -> None:
        """
        Validate deterministic variant identity payload shape.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Keys are lowercase hex SHA-256 strings produced by canonical builders.
        Raises:
            ValueError: If index/key invariants are violated.
        Side Effects:
            Normalizes string keys to lowercase stripped format.
        """
        if self.variant_index < 0:
            raise ValueError("BacktestVariantPreview.variant_index must be >= 0")

        normalized_variant_key = self.variant_key.strip().lower()
        object.__setattr__(self, "variant_key", normalized_variant_key)
        if len(normalized_variant_key) != 64:
            raise ValueError("BacktestVariantPreview.variant_key must be 64 hex chars")

        normalized_indicator_key = self.indicator_variant_key.strip().lower()
        object.__setattr__(self, "indicator_variant_key", normalized_indicator_key)
        if len(normalized_indicator_key) != 64:
            raise ValueError("BacktestVariantPreview.indicator_variant_key must be 64 hex chars")

        if (
            isinstance(self.total_return_pct, bool)
            or not isinstance(self.total_return_pct, int | float)
        ):
            raise ValueError("BacktestVariantPreview.total_return_pct must be numeric")
        object.__setattr__(self, "total_return_pct", float(self.total_return_pct))

        if self.payload is None:  # pragma: no cover - guarded by staged runner payload assembly
            raise ValueError("BacktestVariantPreview.payload is required")


@dataclass(frozen=True, slots=True)
class RunBacktestResponse:
    """
    Backtest use-case response skeleton for BKT-EPIC-01.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/roadmap/milestone-4-epics-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/domain/entities/backtest_placeholders.py
      - apps/api/routes
    """

    mode: str
    instrument_id: InstrumentId
    timeframe: Timeframe
    strategy_id: UUID | None
    warmup_bars: int
    top_k: int
    preselect: int
    top_trades_n: int
    variants: tuple[BacktestVariantPreview, ...]
    total_indicator_compute_calls: int

    def __post_init__(self) -> None:
        """
        Validate response-level deterministic ordering and scalar invariants.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Variants are emitted in deterministic order and indexes are unique.
        Raises:
            ValueError: If mode is unknown, scalar bounds are invalid, or variant ordering breaks.
        Side Effects:
            Normalizes mode literal to lowercase stripped representation.
        """
        normalized_mode = self.mode.strip().lower()
        object.__setattr__(self, "mode", normalized_mode)
        if normalized_mode not in {"saved", "template"}:
            raise ValueError("RunBacktestResponse.mode must be saved or template")

        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestResponse.instrument_id is required")
        if self.timeframe is None:  # type: ignore[truthy-bool]
            raise ValueError("RunBacktestResponse.timeframe is required")

        if self.warmup_bars <= 0:
            raise ValueError("RunBacktestResponse.warmup_bars must be > 0")
        if self.top_k <= 0:
            raise ValueError("RunBacktestResponse.top_k must be > 0")
        if self.preselect <= 0:
            raise ValueError("RunBacktestResponse.preselect must be > 0")
        if self.top_trades_n <= 0:
            raise ValueError("RunBacktestResponse.top_trades_n must be > 0")
        if self.top_trades_n > self.top_k:
            raise ValueError("RunBacktestResponse.top_trades_n must be <= top_k")
        if self.total_indicator_compute_calls < 0:
            raise ValueError("RunBacktestResponse.total_indicator_compute_calls must be >= 0")

        variant_indexes = tuple(item.variant_index for item in self.variants)
        if len(set(variant_indexes)) != len(variant_indexes):
            raise ValueError("RunBacktestResponse variants must contain unique variant_index")

        previous_variant: BacktestVariantPreview | None = None
        for current in self.variants:
            if previous_variant is None:
                previous_variant = current
                continue
            if current.total_return_pct > previous_variant.total_return_pct:
                raise ValueError(
                    "RunBacktestResponse variants must be sorted by total_return_pct desc"
                )
            if (
                current.total_return_pct == previous_variant.total_return_pct
                and current.variant_key < previous_variant.variant_key
            ):
                raise ValueError(
                    "RunBacktestResponse variants with equal total_return_pct must be sorted "
                    "by variant_key asc"
                )
            previous_variant = current


def _validate_positive_optional_int(*, name: str, value: int | None) -> None:
    """
    Validate optional positive integer scalar used for request override fields.

    Args:
        name: Field name used in deterministic error message.
        value: Optional integer value.
    Returns:
        None.
    Assumptions:
        `None` means fallback to runtime config default.
    Raises:
        ValueError: If provided value is non-positive.
    Side Effects:
        None.
    """
    if value is not None and value <= 0:
        raise ValueError(f"RunBacktestRequest.{name} must be > 0 when provided")


def _normalize_scalar_mapping(
    *,
    values: Mapping[str, BacktestRequestScalar] | None,
) -> dict[str, BacktestRequestScalar]:
    """
    Normalize optional scalar mapping into deterministic key-sorted plain dict.

    Args:
        values: Optional scalar mapping.
    Returns:
        dict[str, BacktestRequestScalar]: Deterministic mapping.
    Assumptions:
        Values are JSON-compatible scalars.
    Raises:
        ValueError: If one of keys is blank.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    normalized: dict[str, BacktestRequestScalar] = {}
    for key in sorted(values.keys()):
        normalized_key = str(key).strip()
        if not normalized_key:
            raise ValueError("RunBacktestTemplate mapping keys must be non-empty")
        normalized[normalized_key] = values[key]
    return normalized


def _normalize_signal_grid_mapping(
    *,
    values: BacktestSignalGridMap | None,
) -> dict[str, Mapping[str, GridParamSpec]]:
    """
    Normalize nested signal-grid mapping with deterministic lowercase key ordering.

    Args:
        values: Optional `indicator_id -> signal_param_name -> GridParamSpec` mapping.
    Returns:
        dict[str, Mapping[str, GridParamSpec]]: Deterministic normalized nested mapping.
    Assumptions:
        Every `GridParamSpec` materializes non-empty value sequence.
    Raises:
        ValueError: If one indicator or signal parameter key is blank.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    normalized: dict[str, Mapping[str, GridParamSpec]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("RunBacktestTemplate.signal_grids indicator_id keys must be non-empty")
        signal_axes = values[raw_indicator_id]
        signal_axis_map: dict[str, GridParamSpec] = {}
        for raw_param_name in sorted(signal_axes.keys(), key=lambda key: str(key).strip().lower()):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise ValueError(
                    "RunBacktestTemplate.signal_grids param keys must be non-empty"
                )
            if len(signal_axes[raw_param_name].materialize()) == 0:
                raise ValueError(
                    "RunBacktestTemplate.signal_grids parameter materialized to empty values"
                )
            signal_axis_map[param_name] = signal_axes[raw_param_name]
        normalized[indicator_id] = MappingProxyType(signal_axis_map)
    return normalized


def _normalize_nested_scalar_mapping(
    *,
    values: BacktestSignalScalarMap | None,
) -> dict[str, Mapping[str, BacktestRequestScalar]]:
    """
    Normalize nested scalar mapping with deterministic lowercase key ordering.

    Args:
        values: Optional `indicator_id -> parameter -> scalar` payload mapping.
    Returns:
        dict[str, Mapping[str, BacktestRequestScalar]]: Deterministic normalized nested mapping.
    Assumptions:
        Scalar payload values are JSON-compatible by API/use-case contracts.
    Raises:
        ValueError: If one indicator id or nested parameter key is blank.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    normalized: dict[str, Mapping[str, BacktestRequestScalar]] = {}
    for raw_indicator_id in sorted(values.keys(), key=lambda key: str(key).strip().lower()):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise ValueError("nested scalar mapping indicator_id keys must be non-empty")
        normalized[indicator_id] = MappingProxyType(
            _normalize_scalar_mapping(values=values[raw_indicator_id])
        )
    return normalized
