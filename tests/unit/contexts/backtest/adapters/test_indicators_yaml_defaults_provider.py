from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider


def test_yaml_backtest_grid_defaults_provider_reads_compute_and_signal_defaults(
    tmp_path: Path,
) -> None:
    """
    Verify provider loads compute defaults and `signals.v1.params` defaults from YAML.

    Args:
        tmp_path: pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Defaults payload follows indicators YAML contract.
    Raises:
        AssertionError: If parsed defaults differ from deterministic expectation.
    Side Effects:
        None.
    """
    config_path = tmp_path / "indicators.yaml"
    config_path.write_text(
        (
            "defaults:\n"
            "  ma.sma:\n"
            "    inputs:\n"
            "      source:\n"
            "        mode: explicit\n"
            "        values: [close]\n"
            "    params:\n"
            "      window:\n"
            "        mode: range\n"
            "        start: 10\n"
            "        stop_incl: 30\n"
            "        step: 10\n"
            "    signals:\n"
            "      v1:\n"
            "        params:\n"
            "          cross_up:\n"
            "            mode: explicit\n"
            "            values: [0.4, 0.6]\n"
        ),
        encoding="utf-8",
    )

    provider = YamlBacktestGridDefaultsProvider.from_yaml(config_path=config_path)

    compute_defaults = provider.compute_defaults(indicator_id="MA.SMA")
    assert compute_defaults is not None
    assert compute_defaults.indicator_id.value == "ma.sma"
    assert compute_defaults.source is not None
    assert compute_defaults.source.materialize() == ("close",)
    assert compute_defaults.params["window"].materialize() == (10, 20, 30)

    signal_defaults = provider.signal_param_defaults(indicator_id="ma.sma")
    assert tuple(signal_defaults.keys()) == ("cross_up",)
    assert signal_defaults["cross_up"].materialize() == (0.4, 0.6)


def test_yaml_backtest_grid_defaults_provider_rejects_invalid_axis_mode(tmp_path: Path) -> None:
    """
    Verify provider fails fast on unsupported axis mode literal.

    Args:
        tmp_path: pytest temporary directory fixture.
    Returns:
        None.
    Assumptions:
        Axis mode must be `explicit` or `range`.
    Raises:
        AssertionError: If malformed payload does not raise `ValueError`.
    Side Effects:
        None.
    """
    config_path = tmp_path / "indicators.yaml"
    config_path.write_text(
        (
            "defaults:\n"
            "  ma.sma:\n"
            "    signals:\n"
            "      v1:\n"
            "        params:\n"
            "          cross_up:\n"
            "            mode: broken\n"
            "            values: [0.4]\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mode"):
        YamlBacktestGridDefaultsProvider.from_yaml(config_path=config_path)
