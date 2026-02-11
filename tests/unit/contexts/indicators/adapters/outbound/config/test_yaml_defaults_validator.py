from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.indicators.adapters.outbound.config import (
    IndicatorDefaultsValidationError,
    load_indicator_defaults_yaml,
    validate_indicator_defaults,
)
from trading.contexts.indicators.domain.definitions import all_defs


def _write_defaults_yaml(tmp_path: Path, content: str) -> Path:
    """
    Write one temporary defaults YAML file for validation tests.

    Args:
        tmp_path: Pytest temporary directory fixture.
        content: YAML text content.
    Returns:
        Path: Path to written YAML file.
    Assumptions:
        Content is valid UTF-8 text.
    Raises:
        OSError: If write operation fails.
    Side Effects:
        Creates one file in pytest temporary directory.
    """
    file_path = tmp_path / "indicators.yaml"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def _validate_from_text(*, tmp_path: Path, yaml_text: str) -> None:
    """
    Load and validate temporary YAML defaults against hard definitions.

    Args:
        tmp_path: Pytest temporary directory fixture.
        yaml_text: YAML payload text.
    Returns:
        None.
    Assumptions:
        `all_defs()` returns hard registry contracts for validation.
    Raises:
        IndicatorDefaultsValidationError: On contract violation.
    Side Effects:
        Reads temporary file from disk.
    """
    config_path = _write_defaults_yaml(tmp_path=tmp_path, content=yaml_text)
    defaults = load_indicator_defaults_yaml(config_path)
    validate_indicator_defaults(config_path=config_path, defs=all_defs(), defaults=defaults)


def test_yaml_validation_rejects_unknown_indicator_id_with_yaml_path(tmp_path: Path) -> None:
    """
    Verify fail-fast error for unknown indicator id in defaults mapping.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Unknown defaults key must fail before startup completes.
    Raises:
        AssertionError: If expected error path is missing.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  ma.unknown:
    params:
      window:
        mode: explicit
        values: [10]
"""

    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    message = str(exc_info.value)
    assert "defaults.ma.unknown" in message
    assert "indicator_id=ma.unknown" in message


def test_yaml_validation_rejects_unknown_param(tmp_path: Path) -> None:
    """
    Verify fail-fast error for unknown parameter name under known indicator.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Defaults cannot introduce new params absent in hard definition.
    Raises:
        AssertionError: If expected YAML path is missing.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  ma.sma:
    inputs:
      source:
        mode: explicit
        values: ["close"]
    params:
      windows:
        mode: explicit
        values: [10]
"""

    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    assert "defaults.ma.sma.params.windows" in str(exc_info.value)


def test_yaml_validation_rejects_out_of_bounds_range(tmp_path: Path) -> None:
    """
    Verify fail-fast error when range start is outside hard bounds.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Numeric defaults cannot expand hard min/max bounds.
    Raises:
        AssertionError: If expected bounds error path is missing.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  ma.sma:
    inputs:
      source:
        mode: explicit
        values: ["close"]
    params:
      window:
        mode: range
        start: 1
        stop_incl: 10
        step: 1
"""

    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    assert "defaults.ma.sma.params.window.start" in str(exc_info.value)


def test_yaml_validation_rejects_step_mismatch(tmp_path: Path) -> None:
    """
    Verify fail-fast error when YAML float step mismatches hard grid step.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Float step must be multiple of hard step with epsilon tolerance.
    Raises:
        AssertionError: If expected step path is missing.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  volatility.bbands:
    inputs:
      source:
        mode: explicit
        values: ["close"]
    params:
      window:
        mode: explicit
        values: [20]
      mult:
        mode: range
        start: 1.0
        stop_incl: 3.0
        step: 0.025
"""

    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    assert "defaults.volatility.bbands.params.mult.step" in str(exc_info.value)


def test_yaml_validation_rejects_enum_value_not_allowed(tmp_path: Path) -> None:
    """
    Verify fail-fast error when source enum value is outside hard allowed list.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Source defaults must be subset of indicator allowed input values.
    Raises:
        AssertionError: If expected enum value path is missing.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  ma.sma:
    inputs:
      source:
        mode: explicit
        values: ["volume"]
    params:
      window:
        mode: explicit
        values: [10]
"""

    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    assert "defaults.ma.sma.inputs.source.values[0]" in str(exc_info.value)


def test_yaml_validation_accepts_expanded_source_defaults(tmp_path: Path) -> None:
    """
    Verify expanded source defaults list is accepted for source-axis indicators.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        MA indicators expose source axis with the expanded allowed source set.
    Raises:
        AssertionError: If validation unexpectedly fails.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  ma.sma:
    inputs:
      source:
        mode: explicit
        values: ["close", "hlc3", "ohlc4", "low", "high", "open"]
    params:
      window:
        mode: explicit
        values: [20]
"""
    _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)


def test_yaml_validation_rejects_source_for_indicator_without_source_axis(
    tmp_path: Path,
) -> None:
    """
    Verify `inputs.source` is rejected for indicators without source axis.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        OBV uses fixed close/volume inputs and has no configurable input axis.
    Raises:
        AssertionError: If expected validation error is not raised.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  volume.obv:
    inputs:
      source:
        mode: explicit
        values: ["close"]
"""
    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    assert "defaults.volume.obv.inputs.source" in str(exc_info.value)


def test_yaml_validation_rejects_float_range_start_alignment(tmp_path: Path) -> None:
    """
    Verify float range start must align with hard step grid.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        `volatility.bbands.mult` uses hard_min=0.1 and hard_step=0.01.
    Raises:
        AssertionError: If expected alignment error is not raised.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  volatility.bbands:
    inputs:
      source:
        mode: explicit
        values: ["close"]
    params:
      window:
        mode: explicit
        values: [20]
      mult:
        mode: range
        start: 0.105
        stop_incl: 0.205
        step: 0.01
"""
    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    assert "defaults.volatility.bbands.params.mult.start" in str(exc_info.value)


def test_yaml_validation_rejects_float_range_stop_alignment(tmp_path: Path) -> None:
    """
    Verify float range stop_incl must align with hard step grid.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        `volatility.bbands.mult` uses hard_min=0.1 and hard_step=0.01.
    Raises:
        AssertionError: If expected alignment error is not raised.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  volatility.bbands:
    inputs:
      source:
        mode: explicit
        values: ["close"]
    params:
      window:
        mode: explicit
        values: [20]
      mult:
        mode: range
        start: 0.10
        stop_incl: 0.205
        step: 0.01
"""
    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    assert "defaults.volatility.bbands.params.mult.stop_incl" in str(exc_info.value)


def test_yaml_validation_rejects_float_explicit_value_alignment(tmp_path: Path) -> None:
    """
    Verify explicit float values must align to hard step grid.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        `trend.keltner.mult` uses hard_min=0.1 and hard_step=0.01.
    Raises:
        AssertionError: If expected alignment error is not raised.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  trend.keltner:
    params:
      window:
        mode: explicit
        values: [20]
      mult:
        mode: explicit
        values: [1.005]
"""
    with pytest.raises(IndicatorDefaultsValidationError) as exc_info:
        _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)

    assert "defaults.trend.keltner.params.mult.values[0]" in str(exc_info.value)


def test_yaml_validation_accepts_new_indicator_ids_from_expanded_baseline(
    tmp_path: Path,
) -> None:
    """
    Verify YAML can reference newly added baseline indicators.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Expanded hard definitions include the referenced indicators.
    Raises:
        AssertionError: If validation unexpectedly fails.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  momentum.macd:
    inputs:
      source:
        mode: explicit
        values: ["close"]
    params:
      fast_window:
        mode: explicit
        values: [12]
      signal_window:
        mode: explicit
        values: [9]
      slow_window:
        mode: explicit
        values: [26]
  structure.heikin_ashi: {}
  trend.psar:
    params:
      accel_max:
        mode: explicit
        values: [0.2]
      accel_start:
        mode: explicit
        values: [0.02]
      accel_step:
        mode: explicit
        values: [0.02]
  volume.vwap_deviation:
    params:
      mult:
        mode: explicit
        values: [2.0]
      window:
        mode: explicit
        values: [20]
"""
    _validate_from_text(tmp_path=tmp_path, yaml_text=yaml_text)
