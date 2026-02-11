from __future__ import annotations

from pathlib import Path

from trading.contexts.indicators.adapters.outbound.registry import YamlIndicatorRegistry
from trading.contexts.indicators.domain.definitions import all_defs


def _write_defaults_yaml(tmp_path: Path, content: str) -> Path:
    """
    Write one temporary defaults YAML file for registry tests.

    Args:
        tmp_path: Pytest temporary directory fixture.
        content: YAML text content.
    Returns:
        Path: Path to written file.
    Assumptions:
        Input content is valid UTF-8 text.
    Raises:
        OSError: If file cannot be written.
    Side Effects:
        Creates temporary YAML file.
    """
    file_path = tmp_path / "indicators.yaml"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_merged_registry_output_is_deterministic_and_sorted(tmp_path: Path) -> None:
    """
    Verify deterministic ordering for indicators, params, and inputs in merged view.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Registry sorts by indicator id and nested axis names.
    Raises:
        AssertionError: If ordering is unstable or unsorted.
    Side Effects:
        None.
    """
    yaml_text = """
schema_version: 1
defaults:
  momentum.stoch:
    params:
      smoothing:
        mode: explicit
        values: [3]
      k_window:
        mode: explicit
        values: [5]
      d_window:
        mode: explicit
        values: [3]
  ma.sma:
    params:
      window:
        mode: explicit
        values: [10]
    inputs:
      source:
        mode: explicit
        values: ["close"]
"""
    config_path = _write_defaults_yaml(tmp_path=tmp_path, content=yaml_text)

    registry = YamlIndicatorRegistry.from_yaml(defs=all_defs(), config_path=config_path)

    first_snapshot = registry.list_merged()
    second_snapshot = registry.list_merged()
    assert first_snapshot == second_snapshot

    indicator_ids = [item.indicator_id for item in first_snapshot]
    assert indicator_ids == sorted(indicator_ids)

    for item in first_snapshot:
        param_names = [param.name for param in item.params]
        assert param_names == sorted(param_names)

        input_names = [input_axis.name for input_axis in item.inputs]
        assert input_names == sorted(input_names)


def test_merged_registry_contains_new_baseline_indicators(tmp_path: Path) -> None:
    """
    Verify merged registry includes ids from expanded baseline and their defaults.

    Args:
        tmp_path: Pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Expanded hard definitions are merged with provided YAML defaults.
    Raises:
        AssertionError: If expected merged item or defaults are missing.
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
        values: ["close", "hlc3"]
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
"""
    config_path = _write_defaults_yaml(tmp_path=tmp_path, content=yaml_text)
    registry = YamlIndicatorRegistry.from_yaml(defs=all_defs(), config_path=config_path)

    merged_by_id = {item.indicator_id: item for item in registry.list_merged()}
    assert "momentum.macd" in merged_by_id

    macd = merged_by_id["momentum.macd"]
    assert macd.group == "momentum"
    assert [axis.name for axis in macd.inputs] == ["source"]
    assert [param.name for param in macd.params] == ["fast_window", "signal_window", "slow_window"]
