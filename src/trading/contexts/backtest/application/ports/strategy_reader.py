from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol
from uuid import UUID

from trading.contexts.backtest.application.dto import BacktestRequestScalar, BacktestRiskGridSpec
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.specifications import GridParamSpec, GridSpec
from trading.shared_kernel.primitives import InstrumentId, Timeframe, UserId


@dataclass(frozen=True, slots=True)
class BacktestStrategySnapshot:
    """
    Saved strategy projection consumed by backtest use-case via ACL port.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
    """

    strategy_id: UUID
    user_id: UserId
    is_deleted: bool
    instrument_id: InstrumentId
    timeframe: Timeframe
    indicator_grids: tuple[GridSpec, ...]
    indicator_selections: tuple[IndicatorVariantSelection, ...]
    signal_grids: dict[str, dict[str, GridParamSpec]] | None = None
    risk_grid: BacktestRiskGridSpec | None = None
    direction_mode: str = "long-short"
    sizing_mode: str = "all_in"
    risk_params: Mapping[str, BacktestRequestScalar] | None = None
    execution_params: Mapping[str, BacktestRequestScalar] | None = None
    spec_payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        Validate minimal saved-strategy snapshot invariants for backtest v1.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Snapshot is loaded without owner filtering and must be checked in use-case.
        Raises:
            ValueError: If required fields are missing or indicator payloads are empty.
        Side Effects:
            None.
        """
        if self.user_id is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStrategySnapshot.user_id is required")
        if self.instrument_id is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStrategySnapshot.instrument_id is required")
        if self.timeframe is None:  # type: ignore[truthy-bool]
            raise ValueError("BacktestStrategySnapshot.timeframe is required")
        if len(self.indicator_grids) == 0:
            raise ValueError("BacktestStrategySnapshot.indicator_grids must be non-empty")
        if len(self.indicator_selections) == 0:
            raise ValueError("BacktestStrategySnapshot.indicator_selections must be non-empty")

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
        object.__setattr__(
            self,
            "spec_payload",
            MappingProxyType(_normalize_object_mapping(values=self.spec_payload)),
        )


class BacktestStrategyReader(Protocol):
    """
    Backtest ACL port for loading one saved strategy by id without owner filtering.

    Docs:
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - src/trading/contexts/backtest/application/ports/current_user.py
    """

    def load_any(self, *, strategy_id: UUID) -> BacktestStrategySnapshot | None:
        """
        Load strategy snapshot by id without owner filtering.

        Args:
            strategy_id: Saved strategy identifier.
        Returns:
            BacktestStrategySnapshot | None: Snapshot for explicit ownership checks or `None`.
        Assumptions:
            Adapter boundary handles translation from strategy storage model to backtest snapshot.
        Raises:
            ValueError: If adapter cannot map stored strategy payload deterministically.
        Side Effects:
            Reads saved strategy from outbound storage.
        """
        ...


def _normalize_scalar_mapping(
    *,
    values: Mapping[str, BacktestRequestScalar] | None,
) -> dict[str, BacktestRequestScalar]:
    """
    Normalize optional scalar mapping into deterministic key-sorted plain dictionary.

    Args:
        values: Optional scalar payload mapping.
    Returns:
        dict[str, BacktestRequestScalar]: Deterministic normalized mapping.
    Assumptions:
        Values are JSON-compatible scalars consumed by template mapping layer.
    Raises:
        ValueError: If one key is blank after normalization.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    normalized: dict[str, BacktestRequestScalar] = {}
    for raw_key in sorted(values.keys()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("BacktestStrategySnapshot scalar mapping keys must be non-empty")
        normalized[key] = values[raw_key]
    return normalized


def _normalize_object_mapping(*, values: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Normalize optional object mapping into deterministic key-sorted plain dictionary.

    Args:
        values: Optional mapping payload.
    Returns:
        dict[str, Any]: Deterministic normalized mapping.
    Assumptions:
        Payload is used for reproducibility-hash source in API layer.
    Raises:
        ValueError: If one key is blank after normalization.
    Side Effects:
        None.
    """
    if values is None:
        return {}

    normalized: dict[str, Any] = {}
    for raw_key in sorted(values.keys()):
        key = str(raw_key).strip()
        if not key:
            raise ValueError("BacktestStrategySnapshot object mapping keys must be non-empty")
        normalized[key] = values[raw_key]
    return normalized
