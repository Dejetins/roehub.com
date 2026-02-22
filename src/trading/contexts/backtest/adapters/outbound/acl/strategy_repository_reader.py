from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.application.dto import BacktestRequestScalar, BacktestRiskGridSpec
from trading.contexts.backtest.application.ports import (
    BacktestStrategyReader,
    BacktestStrategySnapshot,
)
from trading.contexts.backtest.domain.errors import BacktestStorageError
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import (
    ExplicitValuesSpec,
    GridParamSpec,
    GridSpec,
    RangeValuesSpec,
)
from trading.contexts.strategy.application.ports.repositories import StrategyRepository
from trading.contexts.strategy.domain.entities import Strategy


@dataclass(frozen=True, slots=True)
class StrategyRepositoryBacktestStrategyReader(BacktestStrategyReader):
    """
    Backtest strategy reader ACL adapter over StrategyRepository storage port.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-bounded-context-domain-use-case-skeleton-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
      - src/trading/contexts/strategy/application/ports/repositories/strategy_repository.py
      - apps/api/wiring/modules/backtest.py
    """

    repository: StrategyRepository

    def __post_init__(self) -> None:
        """
        Validate required repository dependency.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Repository resolves saved strategies without owner filtering.
        Raises:
            ValueError: If repository dependency is missing.
        Side Effects:
            None.
        """
        if self.repository is None:  # type: ignore[truthy-bool]
            raise ValueError("StrategyRepositoryBacktestStrategyReader requires repository")

    def load_any(self, *, strategy_id: UUID) -> BacktestStrategySnapshot | None:
        """
        Load saved strategy snapshot and map it into backtest template-ready projection.

        Args:
            strategy_id: Saved strategy identifier.
        Returns:
            BacktestStrategySnapshot | None: Snapshot for explicit use-case ownership checks.
        Assumptions:
            Saved StrategySpec indicators represent explicit one-variant selections.
        Raises:
            BacktestStorageError: If stored strategy payload cannot be mapped deterministically.
        Side Effects:
            Reads strategy row through StrategyRepository adapter.
        """
        strategy = self.repository.find_any_by_strategy_id(strategy_id=strategy_id)
        if strategy is None:
            return None

        try:
            spec_payload = strategy.spec.to_json()
            indicator_grids, indicator_selections = _indicator_payloads_from_strategy(
                strategy=strategy
            )
            signal_grids = _signal_grids_from_payload(spec_payload=spec_payload)
            risk_grid = _risk_grid_from_payload(spec_payload=spec_payload)
            risk_params = _risk_params_from_payload(spec_payload=spec_payload)
            execution_params = _execution_params_from_payload(spec_payload=spec_payload)
            direction_mode = _optional_mode_literal(
                payload=spec_payload,
                key="direction_mode",
                default="long-short",
            )
            sizing_mode = _optional_mode_literal(
                payload=spec_payload,
                key="sizing_mode",
                default="all_in",
            )
            return BacktestStrategySnapshot(
                strategy_id=strategy.strategy_id,
                user_id=strategy.user_id,
                is_deleted=strategy.is_deleted,
                instrument_id=strategy.spec.instrument_id,
                timeframe=strategy.spec.timeframe,
                indicator_grids=indicator_grids,
                indicator_selections=indicator_selections,
                signal_grids=signal_grids,
                risk_grid=risk_grid,
                direction_mode=direction_mode,
                sizing_mode=sizing_mode,
                risk_params=risk_params,
                execution_params=execution_params,
                spec_payload=spec_payload,
            )
        except BacktestStorageError:
            raise
        except Exception as error:  # noqa: BLE001
            raise BacktestStorageError(
                f"failed to map StrategySpec for backtest strategy_id={strategy_id}"
            ) from error


def _indicator_payloads_from_strategy(
    *,
    strategy: Strategy,
) -> tuple[tuple[GridSpec, ...], tuple[IndicatorVariantSelection, ...]]:
    """
    Build deterministic indicator grid/selection tuples from saved StrategySpec payload.

    Args:
        strategy: Saved strategy aggregate.
    Returns:
        tuple[tuple[GridSpec, ...], tuple[IndicatorVariantSelection, ...]]:
            Deterministic indicator grids and explicit selections.
    Assumptions:
        Each strategy indicator entry represents one explicit variant selection.
    Raises:
        BacktestStorageError: If one indicator entry has invalid shape or duplicate id.
    Side Effects:
        None.
    """
    grids: list[GridSpec] = []
    selections: list[IndicatorVariantSelection] = []
    seen_indicator_ids: set[str] = set()

    for indicator_payload in strategy.spec.indicators:
        indicator_id = _indicator_id_from_payload(indicator_payload=indicator_payload)
        if indicator_id in seen_indicator_ids:
            raise BacktestStorageError(
                f"duplicate indicator_id in StrategySpec indicators: {indicator_id}"
            )
        seen_indicator_ids.add(indicator_id)

        inputs = _scalar_map_from_payload(
            payload=indicator_payload,
            key="inputs",
            field_path=f"indicators[{indicator_id}].inputs",
            allow_bool=False,
        )
        params = _scalar_map_from_payload(
            payload=indicator_payload,
            key="params",
            field_path=f"indicators[{indicator_id}].params",
            allow_bool=False,
        )

        source_spec: GridParamSpec | None = None
        if "source" in inputs:
            source_spec = ExplicitValuesSpec(
                name="source",
                values=(str(inputs["source"]),),
            )

        grid_params: dict[str, GridParamSpec] = {}
        for param_name in sorted(params.keys()):
            grid_params[param_name] = ExplicitValuesSpec(
                name=param_name,
                values=(params[param_name],),
            )
        for input_name in sorted(inputs.keys()):
            if input_name == "source" or input_name in grid_params:
                continue
            grid_params[input_name] = ExplicitValuesSpec(
                name=input_name,
                values=(inputs[input_name],),
            )

        grids.append(
            GridSpec(
                indicator_id=IndicatorId(indicator_id),
                params=grid_params,
                source=source_spec,
            )
        )
        selections.append(
            IndicatorVariantSelection(
                indicator_id=indicator_id,
                inputs=inputs,
                params=params,
            )
        )

    if len(grids) == 0:
        raise BacktestStorageError("StrategySpec indicators list must be non-empty for backtest")

    ordered_pairs = sorted(
        zip(grids, selections, strict=True),
        key=lambda pair: pair[1].indicator_id,
    )
    return (
        tuple(pair[0] for pair in ordered_pairs),
        tuple(pair[1] for pair in ordered_pairs),
    )


def _indicator_id_from_payload(*, indicator_payload: Mapping[str, Any]) -> str:
    """
    Extract normalized indicator id from one StrategySpec indicator mapping.

    Args:
        indicator_payload: One StrategySpec indicator mapping.
    Returns:
        str: Normalized lowercase indicator id.
    Assumptions:
        Indicator id may be stored under `indicator_id`, `id`, `kind`, or `name` keys.
    Raises:
        BacktestStorageError: If indicator identifier is missing or blank.
    Side Effects:
        None.
    """
    for key in ("indicator_id", "id", "kind", "name"):
        raw_value = indicator_payload.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip().lower()
    raise BacktestStorageError(
        "StrategySpec indicator requires non-empty indicator_id/id/kind/name"
    )


def _scalar_map_from_payload(
    *,
    payload: Mapping[str, Any],
    key: str,
    field_path: str,
    allow_bool: bool,
) -> dict[str, int | float | str]:
    """
    Parse one optional nested scalar mapping from StrategySpec payload.

    Args:
        payload: Parent mapping.
        key: Nested mapping key.
        field_path: Dot-like field path used for deterministic errors.
        allow_bool: Whether boolean values are accepted as scalars.
    Returns:
        dict[str, int | float | str]: Deterministic key-sorted scalar mapping.
    Assumptions:
        Mapping values are scalar literals representing explicit indicator settings.
    Raises:
        BacktestStorageError: If nested payload is non-mapping or contains invalid scalars.
    Side Effects:
        None.
    """
    raw_value = payload.get(key)
    if raw_value is None:
        return {}
    if not isinstance(raw_value, Mapping):
        raise BacktestStorageError(f"{field_path} must be mapping when provided")

    normalized: dict[str, int | float | str] = {}
    for raw_name in sorted(raw_value.keys(), key=lambda item: str(item).strip().lower()):
        name = str(raw_name).strip().lower()
        if not name:
            raise BacktestStorageError(f"{field_path} keys must be non-empty")
        scalar = raw_value[raw_name]
        if isinstance(scalar, bool) and not allow_bool:
            raise BacktestStorageError(f"{field_path}.{name} must not be boolean")
        if isinstance(scalar, bool) or not isinstance(scalar, int | float | str):
            raise BacktestStorageError(f"{field_path}.{name} must be scalar")
        normalized[name] = scalar
    return normalized


def _signal_grids_from_payload(
    *,
    spec_payload: Mapping[str, Any],
) -> dict[str, dict[str, GridParamSpec]]:
    """
    Parse optional saved signal grid payload from StrategySpec JSON.

    Args:
        spec_payload: Strategy spec JSON payload.
    Returns:
        dict[str, dict[str, GridParamSpec]]: Deterministic signal grid mapping.
    Assumptions:
        Signal payload is stored under top-level `signal_grids` mapping when available.
    Raises:
        BacktestStorageError: If payload shape is invalid.
    Side Effects:
        None.
    """
    raw_signal_grids = spec_payload.get("signal_grids")
    if raw_signal_grids is None:
        return {}
    if not isinstance(raw_signal_grids, Mapping):
        raise BacktestStorageError("StrategySpec.signal_grids must be mapping when provided")

    normalized: dict[str, dict[str, GridParamSpec]] = {}
    for raw_indicator_id in sorted(
        raw_signal_grids.keys(),
        key=lambda key: str(key).strip().lower(),
    ):
        indicator_id = str(raw_indicator_id).strip().lower()
        if not indicator_id:
            raise BacktestStorageError(
                "StrategySpec.signal_grids indicator_id keys must be non-empty"
            )
        indicator_payload = raw_signal_grids[raw_indicator_id]
        if not isinstance(indicator_payload, Mapping):
            raise BacktestStorageError(
                f"StrategySpec.signal_grids.{indicator_id} must be mapping"
            )
        indicator_params: dict[str, GridParamSpec] = {}
        for raw_param_name in sorted(
            indicator_payload.keys(),
            key=lambda key: str(key).strip().lower(),
        ):
            param_name = str(raw_param_name).strip().lower()
            if not param_name:
                raise BacktestStorageError(
                    "StrategySpec.signal_grids parameter keys must be non-empty"
                )
            indicator_params[param_name] = _grid_param_spec_from_payload_value(
                name=param_name,
                value=indicator_payload[raw_param_name],
                field_path=f"signal_grids.{indicator_id}.{param_name}",
            )
        normalized[indicator_id] = indicator_params
    return normalized


def _risk_grid_from_payload(*, spec_payload: Mapping[str, Any]) -> BacktestRiskGridSpec:
    """
    Parse optional risk grid payload with fallback to scalar risk settings.

    Args:
        spec_payload: Strategy spec JSON payload.
    Returns:
        BacktestRiskGridSpec: Deterministic risk-grid specification.
    Assumptions:
        Scalar fallback values may live in top-level `risk` mapping.
    Raises:
        BacktestStorageError: If risk payload shape is invalid.
    Side Effects:
        None.
    """
    raw_risk = spec_payload.get("risk")
    risk_payload = raw_risk if isinstance(raw_risk, Mapping) else {}
    if raw_risk is not None and not isinstance(raw_risk, Mapping):
        raise BacktestStorageError("StrategySpec.risk must be mapping when provided")

    raw_risk_grid = spec_payload.get("risk_grid")
    risk_grid_payload = raw_risk_grid if isinstance(raw_risk_grid, Mapping) else {}
    if raw_risk_grid is not None and not isinstance(raw_risk_grid, Mapping):
        raise BacktestStorageError("StrategySpec.risk_grid must be mapping when provided")

    sl_enabled = _bool_with_default(
        payload=risk_grid_payload,
        key="sl_enabled",
        default=_bool_with_default(
            payload=risk_payload,
            key="sl_enabled",
            default=False,
            field_path="risk.sl_enabled",
        ),
        field_path="risk_grid.sl_enabled",
    )
    tp_enabled = _bool_with_default(
        payload=risk_grid_payload,
        key="tp_enabled",
        default=_bool_with_default(
            payload=risk_payload,
            key="tp_enabled",
            default=False,
            field_path="risk.tp_enabled",
        ),
        field_path="risk_grid.tp_enabled",
    )

    sl_spec = _grid_param_spec_optional(
        payload=risk_grid_payload,
        key="sl",
        field_path="risk_grid.sl",
    )
    tp_spec = _grid_param_spec_optional(
        payload=risk_grid_payload,
        key="tp",
        field_path="risk_grid.tp",
    )

    if sl_enabled and sl_spec is None:
        sl_scalar = risk_payload.get("sl_pct")
        if sl_scalar is not None:
            if isinstance(sl_scalar, bool) or not isinstance(sl_scalar, int | float):
                raise BacktestStorageError(
                    "StrategySpec.risk.sl_pct must be numeric when risk_grid.sl_enabled=true"
                )
            sl_spec = ExplicitValuesSpec(name="sl", values=(float(sl_scalar),))
    if tp_enabled and tp_spec is None:
        tp_scalar = risk_payload.get("tp_pct")
        if tp_scalar is not None:
            if isinstance(tp_scalar, bool) or not isinstance(tp_scalar, int | float):
                raise BacktestStorageError(
                    "StrategySpec.risk.tp_pct must be numeric when risk_grid.tp_enabled=true"
                )
            tp_spec = ExplicitValuesSpec(name="tp", values=(float(tp_scalar),))

    return BacktestRiskGridSpec(
        sl_enabled=sl_enabled,
        tp_enabled=tp_enabled,
        sl=sl_spec,
        tp=tp_spec,
    )


def _risk_params_from_payload(
    *,
    spec_payload: Mapping[str, Any],
) -> Mapping[str, BacktestRequestScalar]:
    """
    Parse optional scalar risk payload mapping from StrategySpec JSON.

    Args:
        spec_payload: Strategy spec JSON payload.
    Returns:
        Mapping[str, BacktestRequestScalar]: Deterministic scalar risk payload mapping.
    Assumptions:
        Risk payload keys are reused by backtest grid/risk expansion services.
    Raises:
        BacktestStorageError: If payload shape is invalid.
    Side Effects:
        None.
    """
    raw_risk = spec_payload.get("risk")
    if raw_risk is None:
        return {}
    if not isinstance(raw_risk, Mapping):
        raise BacktestStorageError("StrategySpec.risk must be mapping when provided")

    normalized: dict[str, BacktestRequestScalar] = {}
    for raw_key in sorted(raw_risk.keys(), key=lambda item: str(item).strip()):
        key = str(raw_key).strip()
        if not key:
            raise BacktestStorageError("StrategySpec.risk keys must be non-empty")
        value = raw_risk[raw_key]
        if not _is_scalar_for_backtest(value=value):
            raise BacktestStorageError(f"StrategySpec.risk.{key} must be scalar")
        normalized[key] = value
    return normalized


def _execution_params_from_payload(
    *,
    spec_payload: Mapping[str, Any],
) -> Mapping[str, BacktestRequestScalar]:
    """
    Parse optional scalar execution payload mapping from StrategySpec JSON.

    Args:
        spec_payload: Strategy spec JSON payload.
    Returns:
        Mapping[str, BacktestRequestScalar]: Deterministic scalar execution payload mapping.
    Assumptions:
        Execution payload keys are reused by scorer/execution runtime settings.
    Raises:
        BacktestStorageError: If payload shape is invalid.
    Side Effects:
        None.
    """
    raw_execution = spec_payload.get("execution")
    if raw_execution is None:
        return {}
    if not isinstance(raw_execution, Mapping):
        raise BacktestStorageError("StrategySpec.execution must be mapping when provided")

    normalized: dict[str, BacktestRequestScalar] = {}
    for raw_key in sorted(raw_execution.keys(), key=lambda item: str(item).strip()):
        key = str(raw_key).strip()
        if not key:
            raise BacktestStorageError("StrategySpec.execution keys must be non-empty")
        value = raw_execution[raw_key]
        if not _is_scalar_for_backtest(value=value):
            raise BacktestStorageError(f"StrategySpec.execution.{key} must be scalar")
        normalized[key] = value
    return normalized


def _optional_mode_literal(
    *,
    payload: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    """
    Parse optional non-empty mode literal from payload with deterministic fallback.

    Args:
        payload: Parent payload mapping.
        key: Target mode key.
        default: Default mode literal when key is absent.
    Returns:
        str: Normalized lowercase mode literal.
    Assumptions:
        Final literal is validated by RunBacktestTemplate value-object invariants.
    Raises:
        BacktestStorageError: If provided mode value is not a non-empty string.
    Side Effects:
        None.
    """
    raw_value = payload.get(key)
    if raw_value is None:
        return default
    if not isinstance(raw_value, str):
        raise BacktestStorageError(f"StrategySpec.{key} must be string when provided")
    normalized = raw_value.strip().lower()
    if not normalized:
        raise BacktestStorageError(f"StrategySpec.{key} must be non-empty when provided")
    return normalized


def _grid_param_spec_optional(
    *,
    payload: Mapping[str, Any],
    key: str,
    field_path: str,
) -> GridParamSpec | None:
    """
    Parse optional GridParamSpec node from mapping key.

    Args:
        payload: Parent mapping.
        key: Child key.
        field_path: Dot-like field path used for deterministic errors.
    Returns:
        GridParamSpec | None: Parsed specification or `None` when key absent.
    Assumptions:
        Node value may use explicit/range object shape or scalar shorthand.
    Raises:
        BacktestStorageError: If node value shape is invalid.
    Side Effects:
        None.
    """
    if key not in payload:
        return None
    return _grid_param_spec_from_payload_value(
        name=key,
        value=payload[key],
        field_path=field_path,
    )


def _grid_param_spec_from_payload_value(
    *,
    name: str,
    value: Any,
    field_path: str,
) -> GridParamSpec:
    """
    Parse one GridParamSpec from scalar shorthand or explicit/range object mapping.

    Args:
        name: Axis name.
        value: Raw payload value.
        field_path: Dot-like field path used for deterministic errors.
    Returns:
        GridParamSpec: Parsed grid specification.
    Assumptions:
        Scalar shorthand denotes one explicit value axis.
    Raises:
        BacktestStorageError: If value cannot be converted into deterministic grid spec.
    Side Effects:
        None.
    """
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise BacktestStorageError(f"{field_path} axis name must be non-empty")

    if isinstance(value, Mapping):
        raw_mode = value.get("mode")
        if not isinstance(raw_mode, str):
            raise BacktestStorageError(f"{field_path}.mode must be string")
        mode = raw_mode.strip().lower()
        if mode == "explicit":
            raw_values = value.get("values")
            if not isinstance(raw_values, list) or len(raw_values) == 0:
                raise BacktestStorageError(f"{field_path}.values must be non-empty list")
            parsed_values: list[int | float | str] = []
            for item in raw_values:
                if isinstance(item, bool) or not isinstance(item, int | float | str):
                    raise BacktestStorageError(f"{field_path}.values items must be scalar")
                parsed_values.append(item)
            return ExplicitValuesSpec(name=normalized_name, values=tuple(parsed_values))

        if mode == "range":
            start = _numeric_required(value=value.get("start"), field_path=f"{field_path}.start")
            stop_incl = _numeric_required(
                value=value.get("stop_incl"),
                field_path=f"{field_path}.stop_incl",
            )
            step = _numeric_required(value=value.get("step"), field_path=f"{field_path}.step")
            return RangeValuesSpec(
                name=normalized_name,
                start=start,
                stop_inclusive=stop_incl,
                step=step,
            )

        raise BacktestStorageError(f"{field_path}.mode must be 'explicit' or 'range'")

    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise BacktestStorageError(f"{field_path} must be scalar or mapping")
    return ExplicitValuesSpec(name=normalized_name, values=(value,))


def _numeric_required(*, value: Any, field_path: str) -> int | float:
    """
    Parse required numeric payload field while rejecting booleans.

    Args:
        value: Raw payload value.
        field_path: Dot-like field path used for deterministic errors.
    Returns:
        int | float: Parsed numeric value.
    Assumptions:
        Range-axis numeric values follow indicators grid semantics.
    Raises:
        BacktestStorageError: If value is absent or non-numeric.
    Side Effects:
        None.
    """
    if value is None:
        raise BacktestStorageError(f"{field_path} is required")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BacktestStorageError(f"{field_path} must be numeric")
    return value


def _bool_with_default(
    *,
    payload: Mapping[str, Any],
    key: str,
    default: bool,
    field_path: str,
) -> bool:
    """
    Parse optional boolean payload key with deterministic default fallback.

    Args:
        payload: Parent mapping.
        key: Target key.
        default: Fallback value when key is absent.
        field_path: Dot-like field path used for deterministic errors.
    Returns:
        bool: Parsed boolean value.
    Assumptions:
        Bool values configure risk enable flags for Stage-B expansion.
    Raises:
        BacktestStorageError: If provided value is non-boolean.
    Side Effects:
        None.
    """
    if key not in payload:
        return default
    value = payload[key]
    if not isinstance(value, bool):
        raise BacktestStorageError(f"{field_path} must be boolean")
    return value


def _is_scalar_for_backtest(*, value: Any) -> bool:
    """
    Return whether value is accepted as backtest scalar payload literal.

    Args:
        value: Raw payload value.
    Returns:
        bool: True when value is scalar (`int|float|str|bool|None`).
    Assumptions:
        Scalars are serialized into deterministic JSON payloads later in API layer.
    Raises:
        None.
    Side Effects:
        None.
    """
    return isinstance(value, (int, float, str, bool)) or value is None


__all__ = ["StrategyRepositoryBacktestStrategyReader"]
