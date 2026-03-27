from __future__ import annotations

from typing import Mapping, Sequence

from trading.contexts.backtest.application.dto import RunBacktestTemplate
from trading.contexts.backtest.application.ports import BacktestGridDefaultsProvider
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.domain.specifications import GridParamSpec

_UNSUPPORTED_VALUE_CODE = "unsupported_value"
_FORBIDDEN_OVERRIDE_CODE = "forbidden_override"


def validate_template_runtime_contract(
    *,
    template: RunBacktestTemplate,
    defaults_provider: BacktestGridDefaultsProvider | None,
    allowed_request_timeframes: Sequence[str] | None,
    forbidden_request_timeframes: Sequence[str] | None,
    root_path: str,
) -> None:
    """
    Validate resolved template against R1 runtime contract invariants.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py

    Args:
        template: Resolved template payload for sync/jobs/job-runner flow.
        defaults_provider: Runtime defaults provider used as authoritative indicator catalog.
        allowed_request_timeframes: Allowed request timeframe literals in contract order.
        forbidden_request_timeframes: Forbidden request timeframe literals.
        root_path: Validation root path (`body.template`, `saved_strategy`, etc.).
    Returns:
        None.
    Assumptions:
        Template invariants were normalized by DTO/value-object constructors beforehand.
    Raises:
        BacktestValidationError: If timeframe, indicator ids, or signal overrides violate R1.
    Side Effects:
        None.
    """
    errors = _timeframe_errors(
        timeframe_code=template.timeframe.code,
        allowed_request_timeframes=allowed_request_timeframes,
        forbidden_request_timeframes=forbidden_request_timeframes,
        path=f"{root_path}.timeframe",
    )
    errors.extend(
        _indicator_errors(
            template=template,
            defaults_provider=defaults_provider,
            root_path=root_path,
        )
    )
    errors.extend(
        _source_axis_errors(
            template=template,
            defaults_provider=defaults_provider,
            root_path=root_path,
        )
    )
    errors.extend(
        _signal_grid_errors(
            signal_grids=template.signal_grids or {},
            defaults_provider=defaults_provider,
            root_path=f"{root_path}.signal_grids",
        )
    )
    _raise_if_validation_errors(errors=errors)


def validate_signal_overrides_default_only(
    *,
    signal_grids: Mapping[str, Mapping[str, GridParamSpec]],
    defaults_provider: BacktestGridDefaultsProvider | None,
    root_path: str,
) -> None:
    """
    Validate saved-mode signal overrides against `signals.v1.params = default-only`.

    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-signals-from-indicators-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - configs/prod/indicators.yaml

    Args:
        signal_grids: Saved-mode signal overrides payload.
        defaults_provider: Runtime defaults provider used as source of truth.
        root_path: Validation root path (`body.overrides.signal_grids`, etc.).
    Returns:
        None.
    Assumptions:
        Omitted overrides are allowed and use server defaults.
    Raises:
        BacktestValidationError: If request-level signal overrides differ from defaults.
    Side Effects:
        None.
    """
    errors = _signal_grid_errors(
        signal_grids=signal_grids,
        defaults_provider=defaults_provider,
        root_path=root_path,
    )
    _raise_if_validation_errors(errors=errors)


def _timeframe_errors(
    *,
    timeframe_code: str,
    allowed_request_timeframes: Sequence[str] | None,
    forbidden_request_timeframes: Sequence[str] | None,
    path: str,
) -> list[dict[str, str]]:
    """
    Build deterministic request-timeframe validation items for one resolved code.

    Args:
        timeframe_code: Resolved timeframe literal.
        allowed_request_timeframes: Allowed timeframe literals in contract order.
        forbidden_request_timeframes: Forbidden timeframe literals.
        path: Validation path for emitted items.
    Returns:
        list[dict[str, str]]: Deterministic validation items, empty when timeframe is allowed.
    Assumptions:
        Missing allow/forbid sequences mean caller intentionally disabled this validation.
    Raises:
        None.
    Side Effects:
        None.
    """
    allowed = _normalize_literal_sequence(values=allowed_request_timeframes)
    forbidden = _normalize_literal_sequence(values=forbidden_request_timeframes)
    if len(allowed) == 0 and len(forbidden) == 0:
        return []

    normalized_code = timeframe_code.strip().lower()
    is_forbidden = normalized_code in forbidden
    is_not_allowed = len(allowed) > 0 and normalized_code not in allowed
    if not is_forbidden and not is_not_allowed:
        return []

    allowed_values_literal = ", ".join(allowed)
    return [
        {
            "path": path,
            "code": _UNSUPPORTED_VALUE_CODE,
            "message": f"timeframe must be one of: {allowed_values_literal}",
        }
    ]


def _indicator_errors(
    *,
    template: RunBacktestTemplate,
    defaults_provider: BacktestGridDefaultsProvider | None,
    root_path: str,
) -> list[dict[str, str]]:
    """
    Build deterministic unsupported-indicator validation items for one template.

    Args:
        template: Resolved template payload.
        defaults_provider: Runtime defaults provider used as authoritative support catalog.
        root_path: Validation root path for indicator grids and signal grids.
    Returns:
        list[dict[str, str]]: Deterministic validation items.
    Assumptions:
        Missing defaults provider means support-catalog validation is intentionally disabled.
    Raises:
        None.
    Side Effects:
        None.
    """
    if defaults_provider is None:
        return []

    supported_indicator_ids = set(defaults_provider.supported_indicator_ids())
    errors: list[dict[str, str]] = []

    for index, grid in enumerate(template.indicator_grids):
        indicator_id = grid.indicator_id.value
        if indicator_id in supported_indicator_ids:
            continue
        errors.append(
            {
                "path": f"{root_path}.indicator_grids[{index}].indicator_id",
                "code": _UNSUPPORTED_VALUE_CODE,
                "message": f"indicator_id '{indicator_id}' is not supported",
            }
        )

    for indicator_id in sorted((template.signal_grids or {}).keys()):
        if indicator_id in supported_indicator_ids:
            continue
        errors.append(
            {
                "path": f"{root_path}.signal_grids.{indicator_id}",
                "code": _UNSUPPORTED_VALUE_CODE,
                "message": f"indicator_id '{indicator_id}' is not supported",
            }
        )

    return errors


def _signal_grid_errors(
    *,
    signal_grids: Mapping[str, Mapping[str, GridParamSpec]],
    defaults_provider: BacktestGridDefaultsProvider | None,
    root_path: str,
) -> list[dict[str, str]]:
    """
    Build deterministic `default-only` validation items for nested signal grids payload.

    Args:
        signal_grids: Nested `indicator_id -> param_name -> GridParamSpec` payload.
        defaults_provider: Runtime defaults provider with authoritative signal defaults.
        root_path: Validation root path for emitted items.
    Returns:
        list[dict[str, str]]: Deterministic validation items, empty when payload is allowed.
    Assumptions:
        Missing defaults provider means signal override validation is intentionally disabled.
    Raises:
        None.
    Side Effects:
        None.
    """
    if defaults_provider is None or len(signal_grids) == 0:
        return []

    supported_indicator_ids = set(defaults_provider.supported_indicator_ids())
    errors: list[dict[str, str]] = []
    for indicator_id in sorted(signal_grids.keys()):
        if indicator_id not in supported_indicator_ids:
            continue
        requested_params = signal_grids[indicator_id]
        default_params = defaults_provider.signal_param_defaults(indicator_id=indicator_id)
        for param_name in sorted(requested_params.keys()):
            default_spec = default_params.get(param_name)
            if default_spec is not None and _materialized_values(
                requested_params[param_name]
            ) == _materialized_values(default_spec):
                continue
            errors.append(
                {
                    "path": f"{root_path}.{indicator_id}.{param_name}",
                    "code": _FORBIDDEN_OVERRIDE_CODE,
                    "message": "signals.v1.params is default-only",
                }
            )
    return errors


def _source_axis_errors(
    *,
    template: RunBacktestTemplate,
    defaults_provider: BacktestGridDefaultsProvider | None,
    root_path: str,
) -> list[dict[str, str]]:
    """
    Build deterministic `inputs.source` validation items for explicit template source axes.

    Args:
        template: Resolved template payload.
        defaults_provider: Runtime defaults provider exposing allowed per-indicator source catalogs.
        root_path: Validation root path for emitted items.
    Returns:
        list[dict[str, str]]: Deterministic source-axis validation items.
    Assumptions:
        Missing explicit source axis is allowed because runtime defaults may still provide the
        effective source value during later grid merge stages.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/apps/web/web-backtest-runtime-defaults-endpoint-v1.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/request_runtime_contract_v1.py
      - src/trading/contexts/backtest/adapters/outbound/defaults/
        indicators_yaml_defaults_provider.py
    """
    if defaults_provider is None:
        return []

    supported_indicator_ids = set(defaults_provider.supported_indicator_ids())
    errors: list[dict[str, str]] = []
    for index, grid in enumerate(template.indicator_grids):
        indicator_id = grid.indicator_id.value
        if indicator_id not in supported_indicator_ids or grid.source is None:
            continue

        path = f"{root_path}.indicator_grids[{index}].source"
        allowed_source_values = tuple(
            defaults_provider.allowed_source_values(indicator_id=indicator_id)
        )
        if len(allowed_source_values) == 0:
            errors.append(
                {
                    "path": path,
                    "code": _UNSUPPORTED_VALUE_CODE,
                    "message": f"indicator_id '{indicator_id}' does not support inputs.source",
                }
            )
            continue

        allowed_literal = ", ".join(allowed_source_values)
        for raw_value in grid.source.materialize():
            normalized_value = str(raw_value).strip().lower()
            if normalized_value in allowed_source_values:
                continue
            errors.append(
                {
                    "path": path,
                    "code": _UNSUPPORTED_VALUE_CODE,
                    "message": f"inputs.source must be one of: {allowed_literal}",
                }
            )
            break

    return errors


def _materialized_values(spec: GridParamSpec) -> tuple[int | float | str, ...]:
    """
    Materialize one grid spec into deterministic scalar tuple for equality comparison.

    Args:
        spec: Grid parameter specification object.
    Returns:
        tuple[int | float | str, ...]: Materialized scalar values in canonical order.
    Assumptions:
        Spec was already validated by DTO/defaults loader before comparison.
    Raises:
        ValueError: Propagated when spec materialization fails.
    Side Effects:
        None.
    """
    return tuple(spec.materialize())


def _normalize_literal_sequence(values: Sequence[str] | None) -> tuple[str, ...]:
    """
    Normalize string literal sequence with first-seen order preservation.

    Args:
        values: Optional raw literal sequence.
    Returns:
        tuple[str, ...]: Normalized deduplicated lowercase literals.
    Assumptions:
        Caller owns semantic validation of literals beyond normalization.
    Raises:
        ValueError: If one literal is blank.
    Side Effects:
        None.
    """
    if values is None:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in values:
        value = str(raw_value).strip().lower()
        if not value:
            raise ValueError("runtime contract literal sequences must be non-empty")
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return tuple(normalized)


def _raise_if_validation_errors(*, errors: Sequence[Mapping[str, str]]) -> None:
    """
    Raise canonical Backtest validation error when collected runtime-contract items exist.

    Args:
        errors: Collected validation items.
    Returns:
        None.
    Assumptions:
        Items already use deterministic path/code/message literals.
    Raises:
        BacktestValidationError: If at least one validation item was collected.
    Side Effects:
        None.
    """
    if len(errors) == 0:
        return
    raise BacktestValidationError(
        "Backtest request violates runtime defaults contract",
        errors=errors,
    )


__all__ = [
    "validate_signal_overrides_default_only",
    "validate_template_runtime_contract",
]
