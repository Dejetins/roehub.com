from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from trading.contexts.backtest.adapters.outbound import YamlBacktestGridDefaultsProvider
from trading.contexts.backtest_artifacts.application.services.v2 import (
    supported_indicator_ids_for_signal_rules_v2,
)

_TARGET_SOURCE_CAPABLE_INDICATOR_IDS = (
    "momentum.roc",
    "momentum.rsi",
    "momentum.trix",
    "structure.distance_to_ma_norm",
    "structure.percent_rank",
    "structure.zscore",
    "trend.linreg_slope",
    "volatility.hv",
    "volatility.stddev",
    "volatility.variance",
)
_CANONICAL_SOURCE_VALUES = ("close", "hlc3", "ohlc4", "low", "high", "open")
_ZERO_AXIS_SIGNAL_TARGET_IDS = (
    "structure.candle_stats",
    "volatility.tr",
    "volume.ad_line",
    "volume.obv",
)
_MA_DEFAULTS_SHA256_BY_ENV = {
    "dev": "72f71f253d66b20938b5422dcd0c7f402adae05243cc5fb8ab6c958ecc0bad57",
    "prod": "72f71f253d66b20938b5422dcd0c7f402adae05243cc5fb8ab6c958ecc0bad57",
    "test": "7c22043877347c251adf133cdbaceb893a163383ef81a42da1297713820d0f09",
}


def _indicator_defaults_payload(*, env_name: str) -> dict[str, Any]:
    """
    Load one checked-in indicators YAML defaults payload for deterministic assertions.

    Args:
        env_name: Environment name under `configs/<env>/indicators.yaml`.
    Returns:
        dict[str, object]: Parsed top-level defaults mapping.
    Assumptions:
        Repository-local config file exists and contains a `defaults` mapping.
    Raises:
        KeyError: If the YAML payload misses `defaults`.
        TypeError: If the YAML payload shape is not mapping-like.
    Side Effects:
        Reads one repository-local YAML file from disk.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/indicators.yaml
      - configs/prod/indicators.yaml
      - configs/test/indicators.yaml
    """
    payload = yaml.safe_load(
        Path(f"configs/{env_name}/indicators.yaml").read_text(encoding="utf-8")
    )
    return payload["defaults"]


def _ma_defaults_sha256(*, env_name: str) -> str:
    """
    Hash the parsed `ma.*` defaults subtree to detect accidental scope drift.

    Args:
        env_name: Environment name under `configs/<env>/indicators.yaml`.
    Returns:
        str: Deterministic SHA-256 digest of the parsed `ma.*` subtree.
    Assumptions:
        Parsed JSON serialization with sorted keys is stable for this repository contract.
    Raises:
        KeyError: If the YAML defaults mapping is missing.
    Side Effects:
        Reads one repository-local YAML file from disk.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - configs/dev/indicators.yaml
      - configs/prod/indicators.yaml
      - configs/test/indicators.yaml
    """
    defaults = _indicator_defaults_payload(env_name=env_name)
    ma_defaults = {
        indicator_id: defaults[indicator_id]
        for indicator_id in defaults
        if indicator_id.startswith("ma.")
    }
    return hashlib.sha256(
        json.dumps(ma_defaults, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
            "        values: [hlc3, close]\n"
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
    assert compute_defaults.source.materialize() == ("hlc3", "close")
    assert compute_defaults.params["window"].materialize() == (10, 20, 30)

    signal_defaults = provider.signal_param_defaults(indicator_id="ma.sma")
    assert tuple(signal_defaults.keys()) == ("cross_up",)
    assert signal_defaults["cross_up"].materialize() == (0.4, 0.6)
    assert provider.supported_indicator_ids() == ("ma.sma",)
    assert provider.allowed_source_values(indicator_id="ma.sma") == ("close", "hlc3")
    assert provider.allowed_source_values(indicator_id="volume.obv") == ()


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


def test_yaml_defaults_provider_matches_v2_signal_catalog_for_all_target_envs() -> None:
    """
    Verify env-specific defaults catalogs stay aligned with the explicit R4-01 v2 signal registry.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `dev`, `test`, and `prod` indicator defaults expose the same supported signal catalog.
    Raises:
        AssertionError: If one env drifts from the explicit v2 signal registry.
    Side Effects:
        Reads repository-local config files.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/indicators.yaml
      - configs/test/indicators.yaml
      - configs/prod/indicators.yaml
      - src/trading/contexts/backtest/application/services/v2/signal_rules_engine_v2.py
    """
    expected_indicator_ids = supported_indicator_ids_for_signal_rules_v2()
    for env_name in ("dev", "test", "prod"):
        provider = YamlBacktestGridDefaultsProvider.from_yaml(
            config_path=Path(f"configs/{env_name}/indicators.yaml")
        )
        assert provider.supported_indicator_ids() == expected_indicator_ids


def test_target_source_capable_indicators_keep_canonical_source_catalog_in_yaml() -> None:
    """
    Verify narrowed source-capable indicator defaults keep the full canonical source catalog.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Source-catalog ordering in YAML is part of the deterministic config contract.
    Raises:
        AssertionError: If one targeted indicator loses or reorders canonical sources.
    Side Effects:
        Reads checked-in indicators YAML files.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/indicators.yaml
      - configs/prod/indicators.yaml
      - configs/test/indicators.yaml
    """
    for env_name in ("dev", "prod", "test"):
        defaults = _indicator_defaults_payload(env_name=env_name)
        for indicator_id in _TARGET_SOURCE_CAPABLE_INDICATOR_IDS:
            indicator_payload = defaults[indicator_id]
            assert indicator_payload["inputs"]["source"]["values"] == list(_CANONICAL_SOURCE_VALUES)


def test_zero_axis_signal_targets_keep_missing_yaml_compute_defaults_in_all_envs() -> None:
    """
    Verify the approved zero-axis signal targets intentionally keep `compute_defaults(...) is None`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        These signal targets derive their single-variant grid from hard definitions, not YAML axes.
    Raises:
        AssertionError: If one env starts exposing synthetic compute defaults for these targets.
    Side Effects:
        Reads checked-in indicators YAML files.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - configs/dev/indicators.yaml
      - configs/test/indicators.yaml
      - configs/prod/indicators.yaml
    """
    for env_name in ("dev", "prod", "test"):
        provider = YamlBacktestGridDefaultsProvider.from_yaml(
            config_path=Path(f"configs/{env_name}/indicators.yaml")
        )
        for indicator_id in _ZERO_AXIS_SIGNAL_TARGET_IDS:
            assert provider.compute_defaults(indicator_id=indicator_id) is None


def test_ma_family_defaults_snapshot_remains_unchanged_for_all_target_envs() -> None:
    """
    Verify this prompt scope does not mutate any `ma.*` defaults subtree.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Parsed subtree hashes are sufficient to detect accidental `ma.*` config edits.
    Raises:
        AssertionError: If one environment drifts from the frozen `ma.*` snapshot.
    Side Effects:
        Reads checked-in indicators YAML files.
    Docs:
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
      - docs/architecture/backtest/README.md
    Related:
      - configs/dev/indicators.yaml
      - configs/prod/indicators.yaml
      - configs/test/indicators.yaml
    """
    for env_name, expected_sha256 in _MA_DEFAULTS_SHA256_BY_ENV.items():
        assert _ma_defaults_sha256(env_name=env_name) == expected_sha256
