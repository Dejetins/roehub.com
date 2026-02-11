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
        step: 0.03
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
