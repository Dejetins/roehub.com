"""
Validation of YAML indicator defaults against hard indicator definitions.

Docs: docs/architecture/indicators/indicators-registry-yaml-defaults-v1.md
Related: trading.contexts.indicators.domain.entities.indicator_def,
  trading.contexts.indicators.application.dto.registry_view
"""

from __future__ import annotations

from pathlib import Path

from trading.contexts.indicators.application.dto.registry_view import (
    ExplicitDefaultSpec,
    IndicatorDefaults,
    IndicatorDefaultsDocument,
    RangeDefaultSpec,
)
from trading.contexts.indicators.domain.entities import IndicatorDef, ParamDef, ParamKind

_FLOAT_EPS = 1e-9


class IndicatorDefaultsValidationError(ValueError):
    """
    Raised when defaults YAML violates hard indicator registry constraints.
    """

    def __init__(
        self,
        *,
        config_path: str,
        yaml_path: str,
        indicator_id: str,
        field_name: str,
        expected: str,
        actual: object,
    ) -> None:
        """
        Build a fail-fast validation error with rich diagnostic context.

        Args:
            config_path: Source config file path.
            yaml_path: Dot-path to offending YAML node.
            indicator_id: Indicator identifier tied to failing rule.
            field_name: Parameter or input name bound to failing value.
            expected: Human-readable expectation.
            actual: Actual offending value.
        Returns:
            None.
        Assumptions:
            All fields are already normalized by caller.
        Raises:
            None.
        Side Effects:
            Stores error metadata on the exception instance.
        """
        self.config_path = config_path
        self.yaml_path = yaml_path
        self.indicator_id = indicator_id
        self.field_name = field_name
        self.expected = expected
        self.actual = actual
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """
        Format exception message with stable structured context.

        Args:
            None.
        Returns:
            str: Human-readable fail-fast validation message.
        Assumptions:
            Instance fields are already populated.
        Raises:
            None.
        Side Effects:
            None.
        """
        return (
            f"{self.config_path}: {self.yaml_path} "
            f"(indicator_id={self.indicator_id}, name={self.field_name}) "
            f"expected {self.expected}, got {self.actual!r}"
        )


def validate_indicator_defaults(
    *,
    config_path: str | Path,
    defs: tuple[IndicatorDef, ...],
    defaults: IndicatorDefaultsDocument,
) -> None:
    """
    Validate defaults document against hard indicator definitions.

    Args:
        config_path: Config file path used for error context.
        defs: Hard indicator definitions.
        defaults: Parsed defaults document from YAML.
    Returns:
        None.
    Assumptions:
        `defs` are already internally consistent by domain invariants.
    Raises:
        IndicatorDefaultsValidationError: On unknown ids/fields, bounds/step violations,
            or enum mismatches.
    Side Effects:
        None.
    """
    normalized_config_path = str(config_path)
    if defaults.schema_version != 1:
        raise IndicatorDefaultsValidationError(
            config_path=normalized_config_path,
            yaml_path="schema_version",
            indicator_id="<schema>",
            field_name="schema_version",
            expected="schema_version == 1",
            actual=defaults.schema_version,
        )

    defs_by_id = _build_defs_map(defs=defs, config_path=normalized_config_path)

    for indicator_id in sorted(defaults.defaults.keys()):
        indicator_defaults = defaults.defaults[indicator_id]
        indicator_yaml_path = f"defaults.{indicator_id}"
        if indicator_id not in defs_by_id:
            raise IndicatorDefaultsValidationError(
                config_path=normalized_config_path,
                yaml_path=indicator_yaml_path,
                indicator_id=indicator_id,
                field_name="indicator_id",
                expected=f"one of {sorted(defs_by_id.keys())}",
                actual=indicator_id,
            )

        indicator_def = defs_by_id[indicator_id]
        _validate_inputs(
            config_path=normalized_config_path,
            indicator_id=indicator_id,
            indicator_def=indicator_def,
            indicator_defaults=indicator_defaults,
        )
        _validate_params(
            config_path=normalized_config_path,
            indicator_id=indicator_id,
            indicator_def=indicator_def,
            indicator_defaults=indicator_defaults,
        )


def _build_defs_map(
    *,
    defs: tuple[IndicatorDef, ...],
    config_path: str,
) -> dict[str, IndicatorDef]:
    """
    Build index of hard definitions by indicator_id and check duplicates.

    Args:
        defs: Hard indicator definition tuple.
        config_path: Config path for error context.
    Returns:
        dict[str, IndicatorDef]: Indicator lookup map.
    Assumptions:
        Indicator ids must be unique in registry.
    Raises:
        IndicatorDefaultsValidationError: If duplicate indicator_id is detected.
    Side Effects:
        None.
    """
    defs_by_id: dict[str, IndicatorDef] = {}
    for indicator_def in defs:
        indicator_id = indicator_def.indicator_id.value
        if indicator_id in defs_by_id:
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path="<hard_defs>",
                indicator_id=indicator_id,
                field_name="indicator_id",
                expected="unique indicator_id across hard defs",
                actual=indicator_id,
            )
        defs_by_id[indicator_id] = indicator_def
    return defs_by_id


def _validate_inputs(
    *,
    config_path: str,
    indicator_id: str,
    indicator_def: IndicatorDef,
    indicator_defaults: IndicatorDefaults,
) -> None:
    """
    Validate input defaults for one indicator.

    Args:
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        indicator_def: Hard indicator definition.
        indicator_defaults: YAML defaults for one indicator.
    Returns:
        None.
    Assumptions:
        Only `source` input axis is parameterizable in v1 domain model.
    Raises:
        IndicatorDefaultsValidationError: If input defaults are unknown or invalid.
    Side Effects:
        None.
    """
    allowed_input_values: dict[str, tuple[str, ...]] = {}
    if "source" in indicator_def.axes:
        allowed = tuple(sorted({series.value for series in indicator_def.inputs}))
        allowed_input_values["source"] = allowed

    for input_name in sorted(indicator_defaults.inputs.keys()):
        yaml_path = f"defaults.{indicator_id}.inputs.{input_name}"
        if input_name not in allowed_input_values:
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path=yaml_path,
                indicator_id=indicator_id,
                field_name=input_name,
                expected=f"one of {sorted(allowed_input_values.keys())}",
                actual=input_name,
            )

        spec = indicator_defaults.inputs[input_name]
        if not isinstance(spec, ExplicitDefaultSpec):
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path=f"{yaml_path}.mode",
                indicator_id=indicator_id,
                field_name=input_name,
                expected="mode == 'explicit' for input defaults",
                actual=getattr(spec, "mode", "<unknown>"),
            )

        values = _normalize_unique_string_values(
            values=spec.values,
            config_path=config_path,
            indicator_id=indicator_id,
            yaml_path=f"{yaml_path}.values",
            field_name=input_name,
        )
        allowed_values = allowed_input_values[input_name]
        for index, value in enumerate(values):
            if value not in allowed_values:
                raise IndicatorDefaultsValidationError(
                    config_path=config_path,
                    yaml_path=f"{yaml_path}.values[{index}]",
                    indicator_id=indicator_id,
                    field_name=input_name,
                    expected=f"one of {allowed_values}",
                    actual=value,
                )


def _validate_params(
    *,
    config_path: str,
    indicator_id: str,
    indicator_def: IndicatorDef,
    indicator_defaults: IndicatorDefaults,
) -> None:
    """
    Validate parameter defaults for one indicator.

    Args:
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        indicator_def: Hard indicator definition.
        indicator_defaults: YAML defaults for one indicator.
    Returns:
        None.
    Assumptions:
        Numeric params always provide hard step in hard defs.
    Raises:
        IndicatorDefaultsValidationError: If param defaults are unknown or invalid.
    Side Effects:
        None.
    """
    params_by_name = {param.name: param for param in indicator_def.params}

    for param_name in sorted(indicator_defaults.params.keys()):
        yaml_path = f"defaults.{indicator_id}.params.{param_name}"
        if param_name not in params_by_name:
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path=yaml_path,
                indicator_id=indicator_id,
                field_name=param_name,
                expected=f"one of {sorted(params_by_name.keys())}",
                actual=param_name,
            )

        param_def = params_by_name[param_name]
        spec = indicator_defaults.params[param_name]
        if param_def.kind is ParamKind.ENUM:
            _validate_enum_spec(
                config_path=config_path,
                indicator_id=indicator_id,
                param_name=param_name,
                param_def=param_def,
                spec=spec,
            )
            continue

        if param_def.kind is ParamKind.INT:
            _validate_int_spec(
                config_path=config_path,
                indicator_id=indicator_id,
                param_name=param_name,
                param_def=param_def,
                spec=spec,
            )
            continue

        _validate_float_spec(
            config_path=config_path,
            indicator_id=indicator_id,
            param_name=param_name,
            param_def=param_def,
            spec=spec,
        )


def _validate_enum_spec(
    *,
    config_path: str,
    indicator_id: str,
    param_name: str,
    param_def: ParamDef,
    spec: ExplicitDefaultSpec | RangeDefaultSpec,
) -> None:
    """
    Validate enum parameter defaults.

    Args:
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        param_name: Parameter name.
        param_def: Hard parameter definition.
        spec: YAML defaults spec node.
    Returns:
        None.
    Assumptions:
        Enum params support only explicit mode.
    Raises:
        IndicatorDefaultsValidationError: If mode or values are invalid.
    Side Effects:
        None.
    """
    yaml_path = f"defaults.{indicator_id}.params.{param_name}"
    if not isinstance(spec, ExplicitDefaultSpec):
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=f"{yaml_path}.mode",
            indicator_id=indicator_id,
            field_name=param_name,
            expected="mode == 'explicit' for enum parameter",
            actual=getattr(spec, "mode", "<unknown>"),
        )

    values = _normalize_unique_string_values(
        values=spec.values,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.values",
        field_name=param_name,
    )
    allowed_values = tuple(param_def.enum_values or ())
    for index, value in enumerate(values):
        if value not in allowed_values:
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path=f"{yaml_path}.values[{index}]",
                indicator_id=indicator_id,
                field_name=param_name,
                expected=f"one of {allowed_values}",
                actual=value,
            )


def _validate_int_spec(
    *,
    config_path: str,
    indicator_id: str,
    param_name: str,
    param_def: ParamDef,
    spec: ExplicitDefaultSpec | RangeDefaultSpec,
) -> None:
    """
    Validate integer parameter defaults against hard bounds and step grid.

    Args:
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        param_name: Parameter name.
        param_def: Hard parameter definition.
        spec: YAML defaults spec node.
    Returns:
        None.
    Assumptions:
        `hard_min`, `hard_max`, and `step` are numeric for INT params.
    Raises:
        IndicatorDefaultsValidationError: If mode payload violates int rules.
    Side Effects:
        None.
    """
    hard_min = _require_int_bound(
        param_def.hard_min,
        config_path=config_path,
        indicator_id=indicator_id,
        param_name=param_name,
        expected_field="hard_min",
    )
    hard_max = _require_int_bound(
        param_def.hard_max,
        config_path=config_path,
        indicator_id=indicator_id,
        param_name=param_name,
        expected_field="hard_max",
    )
    hard_step = _require_int_bound(
        param_def.step,
        config_path=config_path,
        indicator_id=indicator_id,
        param_name=param_name,
        expected_field="step",
    )

    yaml_path = f"defaults.{indicator_id}.params.{param_name}"

    if isinstance(spec, ExplicitDefaultSpec):
        if len(set(spec.values)) != len(spec.values):
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path=f"{yaml_path}.values",
                indicator_id=indicator_id,
                field_name=param_name,
                expected="unique values",
                actual=spec.values,
            )
        for index, raw in enumerate(spec.values):
            value = _require_int_scalar(
                raw,
                config_path=config_path,
                indicator_id=indicator_id,
                yaml_path=f"{yaml_path}.values[{index}]",
                field_name=param_name,
            )
            _validate_int_value(
                value=value,
                config_path=config_path,
                indicator_id=indicator_id,
                yaml_path=f"{yaml_path}.values[{index}]",
                field_name=param_name,
                hard_min=hard_min,
                hard_max=hard_max,
                hard_step=hard_step,
            )
        return

    start = _require_int_scalar(
        spec.start,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.start",
        field_name=param_name,
    )
    stop_incl = _require_int_scalar(
        spec.stop_incl,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.stop_incl",
        field_name=param_name,
    )
    step = _require_int_scalar(
        spec.step,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.step",
        field_name=param_name,
    )

    if step <= 0:
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=f"{yaml_path}.step",
            indicator_id=indicator_id,
            field_name=param_name,
            expected="step > 0",
            actual=step,
        )
    if start > stop_incl:
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=f"{yaml_path}.start",
            indicator_id=indicator_id,
            field_name=param_name,
            expected="start <= stop_incl",
            actual=(start, stop_incl),
        )
    if step % hard_step != 0:
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=f"{yaml_path}.step",
            indicator_id=indicator_id,
            field_name=param_name,
            expected=f"step multiple of hard step {hard_step}",
            actual=step,
        )

    _validate_int_value(
        value=start,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.start",
        field_name=param_name,
        hard_min=hard_min,
        hard_max=hard_max,
        hard_step=hard_step,
    )
    _validate_int_value(
        value=stop_incl,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.stop_incl",
        field_name=param_name,
        hard_min=hard_min,
        hard_max=hard_max,
        hard_step=hard_step,
    )


def _validate_float_spec(
    *,
    config_path: str,
    indicator_id: str,
    param_name: str,
    param_def: ParamDef,
    spec: ExplicitDefaultSpec | RangeDefaultSpec,
) -> None:
    """
    Validate float parameter defaults against hard bounds and step grid.

    Args:
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        param_name: Parameter name.
        param_def: Hard parameter definition.
        spec: YAML defaults spec node.
    Returns:
        None.
    Assumptions:
        Float comparisons use epsilon tolerance.
    Raises:
        IndicatorDefaultsValidationError: If mode payload violates float rules.
    Side Effects:
        None.
    """
    hard_min = _require_float_bound(
        param_def.hard_min,
        config_path=config_path,
        indicator_id=indicator_id,
        param_name=param_name,
        expected_field="hard_min",
    )
    hard_max = _require_float_bound(
        param_def.hard_max,
        config_path=config_path,
        indicator_id=indicator_id,
        param_name=param_name,
        expected_field="hard_max",
    )
    hard_step = _require_float_bound(
        param_def.step,
        config_path=config_path,
        indicator_id=indicator_id,
        param_name=param_name,
        expected_field="step",
    )

    yaml_path = f"defaults.{indicator_id}.params.{param_name}"

    if isinstance(spec, ExplicitDefaultSpec):
        if len(set(spec.values)) != len(spec.values):
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path=f"{yaml_path}.values",
                indicator_id=indicator_id,
                field_name=param_name,
                expected="unique values",
                actual=spec.values,
            )
        for index, raw in enumerate(spec.values):
            value = _require_float_scalar(
                raw,
                config_path=config_path,
                indicator_id=indicator_id,
                yaml_path=f"{yaml_path}.values[{index}]",
                field_name=param_name,
            )
            _validate_float_value(
                value=value,
                config_path=config_path,
                indicator_id=indicator_id,
                yaml_path=f"{yaml_path}.values[{index}]",
                field_name=param_name,
                hard_min=hard_min,
                hard_max=hard_max,
                hard_step=hard_step,
            )
        return

    start = _require_float_scalar(
        spec.start,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.start",
        field_name=param_name,
    )
    stop_incl = _require_float_scalar(
        spec.stop_incl,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.stop_incl",
        field_name=param_name,
    )
    step = _require_float_scalar(
        spec.step,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.step",
        field_name=param_name,
    )

    if step <= 0:
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=f"{yaml_path}.step",
            indicator_id=indicator_id,
            field_name=param_name,
            expected="step > 0",
            actual=step,
        )
    if start > stop_incl:
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=f"{yaml_path}.start",
            indicator_id=indicator_id,
            field_name=param_name,
            expected="start <= stop_incl",
            actual=(start, stop_incl),
        )
    if not _is_float_multiple(step / hard_step):
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=f"{yaml_path}.step",
            indicator_id=indicator_id,
            field_name=param_name,
            expected=f"step multiple of hard step {hard_step}",
            actual=step,
        )

    _validate_float_value(
        value=start,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.start",
        field_name=param_name,
        hard_min=hard_min,
        hard_max=hard_max,
        hard_step=hard_step,
    )
    _validate_float_value(
        value=stop_incl,
        config_path=config_path,
        indicator_id=indicator_id,
        yaml_path=f"{yaml_path}.stop_incl",
        field_name=param_name,
        hard_min=hard_min,
        hard_max=hard_max,
        hard_step=hard_step,
    )


def _validate_int_value(
    *,
    value: int,
    config_path: str,
    indicator_id: str,
    yaml_path: str,
    field_name: str,
    hard_min: int,
    hard_max: int,
    hard_step: int,
) -> None:
    """
    Validate one integer value against bounds and hard grid.

    Args:
        value: Value from YAML defaults.
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        yaml_path: YAML path for this value.
        field_name: Parameter name.
        hard_min: Inclusive hard lower bound.
        hard_max: Inclusive hard upper bound.
        hard_step: Hard grid step.
    Returns:
        None.
    Assumptions:
        Hard bounds and step are valid positive integers.
    Raises:
        IndicatorDefaultsValidationError: If value violates bounds or grid alignment.
    Side Effects:
        None.
    """
    if value < hard_min or value > hard_max:
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=yaml_path,
            indicator_id=indicator_id,
            field_name=field_name,
            expected=f"{hard_min} <= value <= {hard_max}",
            actual=value,
        )

    if (value - hard_min) % hard_step != 0:
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=yaml_path,
            indicator_id=indicator_id,
            field_name=field_name,
            expected=f"value on hard grid min={hard_min}, step={hard_step}",
            actual=value,
        )


def _validate_float_value(
    *,
    value: float,
    config_path: str,
    indicator_id: str,
    yaml_path: str,
    field_name: str,
    hard_min: float,
    hard_max: float,
    hard_step: float,
) -> None:
    """
    Validate one float value against bounds and hard grid with epsilon.

    Args:
        value: Value from YAML defaults.
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        yaml_path: YAML path for this value.
        field_name: Parameter name.
        hard_min: Inclusive hard lower bound.
        hard_max: Inclusive hard upper bound.
        hard_step: Hard grid step.
    Returns:
        None.
    Assumptions:
        Hard bounds and step are valid positive floats.
    Raises:
        IndicatorDefaultsValidationError: If value violates bounds or grid alignment.
    Side Effects:
        None.
    """
    if value < hard_min - _FLOAT_EPS or value > hard_max + _FLOAT_EPS:
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=yaml_path,
            indicator_id=indicator_id,
            field_name=field_name,
            expected=f"{hard_min} <= value <= {hard_max}",
            actual=value,
        )

    if not _is_float_multiple((value - hard_min) / hard_step):
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=yaml_path,
            indicator_id=indicator_id,
            field_name=field_name,
            expected=f"value on hard grid min={hard_min}, step={hard_step}",
            actual=value,
        )


def _normalize_unique_string_values(
    *,
    values: tuple[int | float | str, ...],
    config_path: str,
    indicator_id: str,
    yaml_path: str,
    field_name: str,
) -> tuple[str, ...]:
    """
    Normalize explicit string values and ensure uniqueness.

    Args:
        values: Raw explicit values.
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        yaml_path: YAML values path.
        field_name: Input/parameter name.
    Returns:
        tuple[str, ...]: Normalized non-empty strings.
    Assumptions:
        Enum-like defaults must be strings.
    Raises:
        IndicatorDefaultsValidationError: If value is non-string, blank, or duplicated.
    Side Effects:
        None.
    """
    normalized: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path=f"{yaml_path}[{index}]",
                indicator_id=indicator_id,
                field_name=field_name,
                expected="string value",
                actual=type(value).__name__,
            )
        item = value.strip().lower()
        if not item:
            raise IndicatorDefaultsValidationError(
                config_path=config_path,
                yaml_path=f"{yaml_path}[{index}]",
                indicator_id=indicator_id,
                field_name=field_name,
                expected="non-empty string value",
                actual=value,
            )
        normalized.append(item)

    if len(set(normalized)) != len(normalized):
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=yaml_path,
            indicator_id=indicator_id,
            field_name=field_name,
            expected="unique values",
            actual=normalized,
        )

    return tuple(normalized)


def _require_int_scalar(
    value: int | float | str,
    *,
    config_path: str,
    indicator_id: str,
    yaml_path: str,
    field_name: str,
) -> int:
    """
    Require scalar to be strict integer (bool/float/string rejected).

    Args:
        value: Raw scalar from YAML defaults.
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        yaml_path: YAML path for this scalar.
        field_name: Parameter name.
    Returns:
        int: Parsed integer.
    Assumptions:
        INT params do not accept float semantics.
    Raises:
        IndicatorDefaultsValidationError: If value type is not strict int.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=yaml_path,
            indicator_id=indicator_id,
            field_name=field_name,
            expected="integer scalar",
            actual=value,
        )
    return value


def _require_float_scalar(
    value: int | float | str,
    *,
    config_path: str,
    indicator_id: str,
    yaml_path: str,
    field_name: str,
) -> float:
    """
    Require scalar to be numeric for float params (int is accepted).

    Args:
        value: Raw scalar from YAML defaults.
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        yaml_path: YAML path for this scalar.
        field_name: Parameter name.
    Returns:
        float: Parsed float value.
    Assumptions:
        YAML numeric scalars are already decoded to Python numbers.
    Raises:
        IndicatorDefaultsValidationError: If value type is non-numeric.
    Side Effects:
        None.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path=yaml_path,
            indicator_id=indicator_id,
            field_name=field_name,
            expected="numeric scalar",
            actual=value,
        )
    return float(value)


def _require_int_bound(
    value: int | float | None,
    *,
    config_path: str,
    indicator_id: str,
    param_name: str,
    expected_field: str,
) -> int:
    """
    Require hard definition bound for INT params to be strict int.

    Args:
        value: Raw hard bound value.
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        param_name: Parameter name.
        expected_field: Field name in hard definition.
    Returns:
        int: Normalized integer bound.
    Assumptions:
        Hard definitions provide all numeric bounds for int params.
    Raises:
        IndicatorDefaultsValidationError: If hard bound missing or invalid.
    Side Effects:
        None.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, int):
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path="<hard_defs>",
            indicator_id=indicator_id,
            field_name=param_name,
            expected=f"hard {expected_field} must be integer",
            actual=value,
        )
    return value


def _require_float_bound(
    value: int | float | None,
    *,
    config_path: str,
    indicator_id: str,
    param_name: str,
    expected_field: str,
) -> float:
    """
    Require hard definition bound for FLOAT params to be numeric.

    Args:
        value: Raw hard bound value.
        config_path: Config path for error context.
        indicator_id: Indicator identifier.
        param_name: Parameter name.
        expected_field: Field name in hard definition.
    Returns:
        float: Normalized numeric bound.
    Assumptions:
        Hard definitions provide all numeric bounds for float params.
    Raises:
        IndicatorDefaultsValidationError: If hard bound missing or invalid.
    Side Effects:
        None.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        raise IndicatorDefaultsValidationError(
            config_path=config_path,
            yaml_path="<hard_defs>",
            indicator_id=indicator_id,
            field_name=param_name,
            expected=f"hard {expected_field} must be numeric",
            actual=value,
        )
    return float(value)


def _is_float_multiple(value: float) -> bool:
    """
    Check if value is effectively integer under epsilon tolerance.

    Args:
        value: Float ratio candidate.
    Returns:
        bool: True if close to nearest integer.
    Assumptions:
        Ratio can have minor floating-point noise.
    Raises:
        None.
    Side Effects:
        None.
    """
    rounded = round(value)
    return abs(value - rounded) <= _FLOAT_EPS
