from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from uuid import UUID

from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridSpec
from trading.shared_kernel.primitives import InstrumentId, Timeframe, TimeRange

BacktestRequestScalar = int | float | str | bool | None
_ALLOWED_DIRECTION_MODES = {"long-only", "short-only", "long-short"}
_ALLOWED_SIZING_MODES = {
    "all_in",
    "fixed_quote",
    "strategy_compound",
    "strategy_compound_profit_lock",
}


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
    indicator_selections: tuple[IndicatorVariantSelection, ...]
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
        if len(self.indicator_selections) == 0:
            raise ValueError("RunBacktestTemplate.indicator_selections must be non-empty")

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
    warmup_bars: int | None = None
    top_k: int | None = None
    preselect: int | None = None

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

        _validate_positive_optional_int(name="warmup_bars", value=self.warmup_bars)
        _validate_positive_optional_int(name="top_k", value=self.top_k)
        _validate_positive_optional_int(name="preselect", value=self.preselect)

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
        if self.total_indicator_compute_calls < 0:
            raise ValueError("RunBacktestResponse.total_indicator_compute_calls must be >= 0")

        variant_indexes = tuple(item.variant_index for item in self.variants)
        if len(set(variant_indexes)) != len(variant_indexes):
            raise ValueError("RunBacktestResponse variants must contain unique variant_index")
        if tuple(sorted(variant_indexes)) != variant_indexes:
            raise ValueError("RunBacktestResponse variants must be sorted by variant_index")


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

