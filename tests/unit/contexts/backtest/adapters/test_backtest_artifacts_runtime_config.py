from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound.config import (
    build_backtest_artifacts_runtime_config_hash,
    load_backtest_artifacts_runtime_config,
    resolve_backtest_artifacts_config_path,
)
from trading.contexts.backtest.application.services import ArtifactSignalValidationSpecV2
from trading.contexts.backtest.application.services.signals_from_indicators_v1 import (
    supported_indicator_ids_for_signals_v1,
)

_VALID_BACKTEST_ARTIFACTS_CONFIG = """
version: 1
backtest_artifacts:
  artifact_root: artifacts/backtest/v2
  validation_plan:
    price_timeframes: [1h, 1m, 15m]
    mapping_timeframes: [1h, 15m]
    signal_artifacts:
      - timeframe: 1h
        indicator_id: ma.sma
      - timeframe: 15m
        indicator_id: ma.ema
    require_hit_times_manifest: true
  hit_times_grid:
    tp_levels_pct: [2.0, 1.0, 3.0]
    sl_levels_pct: [1.5, 0.5]
  slot_policy:
    slots: [slot_b, slot_a]
  publish_schedule:
    full_rebuild_hour_utc: 2
    full_rebuild_minute_utc: 5
  lookback_policy:
    price_tail_bars_1m: 100
    mapping_tail_bars_1m: 200
    signal_tail_bars_1m: 300
    hit_times_tail_bars_1m: 400
  validation_budgets:
    max_price_bars_per_timeframe: 1000
    max_mapping_rows_per_timeframe: 2000
    max_signal_rows_per_artifact: 3000
    max_hit_times_cells: 4000
    max_hit_times_cells_full_rebuild: 8000
""".strip()


def _write_backtest_artifacts_config(
    tmp_path: Path,
    *,
    body: str,
    filename: str = "backtest_artifacts.yaml",
) -> Path:
    """
    Write temporary Backtest artifact runtime YAML used by config-loader tests.

    Args:
        tmp_path: pytest temporary directory fixture.
        body: Full YAML content.
        filename: Output filename inside `tmp_path`.
    Returns:
        Path: Written config path.
    Assumptions:
        Input text is valid UTF-8.
    Raises:
        OSError: If write operation fails.
    Side Effects:
        Creates one temp YAML file.
    """
    config_path = tmp_path / filename
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_load_backtest_artifacts_runtime_config_reads_yaml_values() -> None:
    """
    Verify loader parses documented artifact pipeline defaults from source-of-truth YAML.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `configs/dev/backtest_artifacts.yaml` is the canonical dev contract.
    Raises:
        AssertionError: If parsed values differ from YAML payload.
    Side Effects:
        None.
    """
    config = load_backtest_artifacts_runtime_config(Path("configs/dev/backtest_artifacts.yaml"))

    assert config.version == 1
    assert config.artifact_root == "artifacts/backtest/v2"
    assert config.validation_plan.price_timeframes == (
        "1m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "1d",
        "2d",
        "3d",
    )
    assert config.validation_plan.mapping_timeframes == (
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "1d",
        "2d",
        "3d",
    )
    expected_signal_targets = tuple(
        (timeframe, indicator_id)
        for timeframe in (
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "1d",
            "2d",
            "3d",
        )
        for indicator_id in supported_indicator_ids_for_signals_v1()
    )
    assert config.validation_plan.signal_artifacts[0].timeframe == "15m"
    assert config.validation_plan.signal_artifacts[0].indicator_id == "ma.dema"
    assert config.validation_plan.signal_artifacts[-1].timeframe == "3d"
    assert config.validation_plan.signal_artifacts[-1].indicator_id == "volume.vwap"
    assert tuple(
        (item.timeframe, item.indicator_id) for item in config.validation_plan.signal_artifacts
    ) == expected_signal_targets
    assert config.validation_plan.require_hit_times_manifest is True
    assert config.hit_times_grid.tp_levels_pct == (0.5, 1.0, 1.5, 2.0, 3.0)
    assert config.hit_times_grid.sl_levels_pct == (0.5, 1.0, 1.5, 2.0, 3.0)
    assert config.slot_policy.slots == ("slot_a", "slot_b")
    assert config.publish_schedule.full_rebuild_hour_utc == 2
    assert config.publish_schedule.full_rebuild_minute_utc == 0
    assert config.lookback_policy.price_tail_bars_1m == 20000
    assert config.lookback_policy.signal_tail_bars_1m == 20000
    assert config.validation_budgets.max_hit_times_cells == 50000000
    assert config.validation_budgets.max_hit_times_cells_full_rebuild == 150000000
    assert config.to_validation_spec().price_timeframes == config.validation_plan.price_timeframes


def test_backtest_artifacts_runtime_config_to_precompute_runtime_settings_includes_r5_hit_times(
) -> None:
    """
    Verify service-layer precompute settings carry explicit R4/R5 runner inputs from YAML.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runner wiring must receive signal tail lookback, hit-times grids, and hit-times budgets
        without hidden defaults.
    Raises:
        AssertionError: If the translated service DTO drops or changes configured R4/R5 inputs.
    Side Effects:
        None.
    """
    config = load_backtest_artifacts_runtime_config(Path("configs/dev/backtest_artifacts.yaml"))

    runtime_settings = config.to_precompute_runtime_settings(
        config_sha256=build_backtest_artifacts_runtime_config_hash(config=config)
    )

    assert runtime_settings.price_tail_bars_1m == 20000
    assert runtime_settings.mapping_tail_bars_1m == 20000
    assert runtime_settings.signal_tail_bars_1m == 20000
    assert runtime_settings.hit_times_tail_bars_1m == 20000
    assert runtime_settings.hit_times_tp_levels_pct == (0.5, 1.0, 1.5, 2.0, 3.0)
    assert runtime_settings.hit_times_sl_levels_pct == (0.5, 1.0, 1.5, 2.0, 3.0)
    assert runtime_settings.max_signal_rows_per_artifact == 5000000
    assert runtime_settings.max_hit_times_cells == 50000000
    assert runtime_settings.max_hit_times_cells_full_rebuild == 150000000


def test_resolve_backtest_artifacts_config_path_precedence() -> None:
    """
    Verify artifact config path resolution uses override env first, then env fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fallback format is `configs/<ROEHUB_ENV>/backtest_artifacts.yaml`.
    Raises:
        AssertionError: If precedence order differs from runtime contract.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "ROEHUB_BACKTEST_ARTIFACTS_CONFIG": "configs/test/custom-backtest_artifacts.yaml",
    }

    assert resolve_backtest_artifacts_config_path(environ=environ) == Path(
        "configs/test/custom-backtest_artifacts.yaml"
    )
    assert resolve_backtest_artifacts_config_path(environ={"ROEHUB_ENV": "test"}) == Path(
        "configs/test/backtest_artifacts.yaml"
    )
    assert resolve_backtest_artifacts_config_path(environ={}) == Path(
        "configs/dev/backtest_artifacts.yaml"
    )


def test_resolve_backtest_artifacts_config_path_rejects_invalid_env_name() -> None:
    """
    Verify unsupported `ROEHUB_ENV` value fails fast for artifact config path resolution.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Allowed environment literals are `dev`, `prod`, and `test`.
    Raises:
        AssertionError: If invalid env value does not raise ValueError.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="ROEHUB_ENV"):
        resolve_backtest_artifacts_config_path(environ={"ROEHUB_ENV": "stage"})


def test_load_backtest_artifacts_runtime_config_rejects_missing_required_keys(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when one strict top-level artifact section is absent.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `validation_budgets` is a strict-required section in R2-04.
    Raises:
        AssertionError: If missing section does not raise ValueError.
    Side Effects:
        Writes one temporary YAML file.
    """
    config_path = _write_backtest_artifacts_config(
        tmp_path,
        body=_VALID_BACKTEST_ARTIFACTS_CONFIG.replace("  validation_budgets:\n", "  x:\n", 1),
    )

    with pytest.raises(ValueError, match="missing required keys"):
        load_backtest_artifacts_runtime_config(config_path)


def test_load_backtest_artifacts_runtime_config_rejects_extra_keys(tmp_path: Path) -> None:
    """
    Verify loader fails fast when unsupported keys appear in a strict nested section.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `publish_schedule` allows only hour/minute UTC fields in R2-04.
    Raises:
        AssertionError: If extra key does not raise ValueError.
    Side Effects:
        Writes one temporary YAML file.
    """
    config_path = _write_backtest_artifacts_config(
        tmp_path,
        body=_VALID_BACKTEST_ARTIFACTS_CONFIG.replace(
            "    full_rebuild_minute_utc: 5",
            "    full_rebuild_minute_utc: 5\n    extra_field: 1",
            1,
        ),
    )

    with pytest.raises(ValueError, match="contains unsupported keys"):
        load_backtest_artifacts_runtime_config(config_path)


def test_load_backtest_artifacts_runtime_config_rejects_invalid_signal_artifact_shape(
    tmp_path: Path,
) -> None:
    """
    Verify loader fails fast when `signal_artifacts` item shape drifts from strict contract.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Each item must contain exactly `timeframe` and `indicator_id`.
    Raises:
        AssertionError: If invalid item shape does not raise ValueError.
    Side Effects:
        Writes one temporary YAML file.
    """
    config_path = _write_backtest_artifacts_config(
        tmp_path,
        body=_VALID_BACKTEST_ARTIFACTS_CONFIG.replace(
            "      - timeframe: 1h\n        indicator_id: ma.sma",
            "      - timeframe: 1h\n        indicator_id: ma.sma\n        extra: 1",
            1,
        ),
    )

    with pytest.raises(ValueError, match="signal_artifacts\\[0\\] contains unsupported keys"):
        load_backtest_artifacts_runtime_config(config_path)


def test_load_backtest_artifacts_runtime_config_rejects_duplicate_targets(tmp_path: Path) -> None:
    """
    Verify loader fails fast when config repeats equivalent validation targets.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Duplicate timeframe literals and duplicate signal identities are rejected.
    Raises:
        AssertionError: If duplicates do not raise ValueError.
    Side Effects:
        Writes one temporary YAML file.
    """
    config_path = _write_backtest_artifacts_config(
        tmp_path,
        body=_VALID_BACKTEST_ARTIFACTS_CONFIG.replace(
            "    mapping_timeframes: [1h, 15m]",
            "    mapping_timeframes: [1h, 15m, 15m]",
            1,
        ),
    )

    with pytest.raises(ValueError, match="mapping_timeframes contains duplicate '15m'"):
        load_backtest_artifacts_runtime_config(config_path)


def test_load_backtest_artifacts_runtime_config_normalizes_equivalent_author_order(
    tmp_path: Path,
) -> None:
    """
    Verify reordered but equivalent author input normalizes into canonical deterministic order.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        R2-04 canonicalizes timeframe, signal, slot, and TP/SL grid ordering.
    Raises:
        AssertionError: If canonical ordering differs from the strict contract.
    Side Effects:
        Writes one temporary YAML file.
    """
    config_path = _write_backtest_artifacts_config(tmp_path, body=_VALID_BACKTEST_ARTIFACTS_CONFIG)

    config = load_backtest_artifacts_runtime_config(config_path)
    validation_spec = config.to_validation_spec()

    assert config.validation_plan.price_timeframes == ("1m", "15m", "1h")
    assert config.validation_plan.mapping_timeframes == ("15m", "1h")
    assert tuple(
        (item.timeframe, item.indicator_id) for item in config.validation_plan.signal_artifacts
    ) == (("15m", "ma.ema"), ("1h", "ma.sma"))
    assert config.hit_times_grid.tp_levels_pct == (1.0, 2.0, 3.0)
    assert config.hit_times_grid.sl_levels_pct == (0.5, 1.5)
    assert config.slot_policy.slots == ("slot_a", "slot_b")
    assert validation_spec.price_timeframes == ("1m", "15m", "1h")
    assert tuple(
        (item.timeframe, item.indicator_id) for item in validation_spec.signal_artifacts
    ) == (
        ("15m", "ma.ema"),
        ("1h", "ma.sma"),
    )


def test_load_backtest_artifacts_runtime_config_derives_prices_mappings_publish_spec(
    tmp_path: Path,
) -> None:
    """
    Verify R3-04 derives an explicit prices+mappings publish spec from the full validation plan.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Source-of-truth config may already contain later-stage `signal_artifacts` and
        `require_hit_times_manifest=true`, while R3-04 publish keeps those families explicitly
        out of scope.
    Raises:
        AssertionError: If the derived stage spec drops price/mapping targets or keeps later-stage
            validation requirements enabled.
    Side Effects:
        Writes one temporary YAML file.
    """
    config_path = _write_backtest_artifacts_config(tmp_path, body=_VALID_BACKTEST_ARTIFACTS_CONFIG)

    config = load_backtest_artifacts_runtime_config(config_path)
    stage_spec = config.to_prices_mappings_publish_validation_spec()

    assert stage_spec.price_timeframes == config.validation_plan.price_timeframes
    assert stage_spec.mapping_timeframes == config.validation_plan.mapping_timeframes
    assert stage_spec.signal_artifacts == ()
    assert stage_spec.require_hit_times_manifest is False


def test_load_backtest_artifacts_runtime_config_derives_precompute_runtime_settings(
    tmp_path: Path,
) -> None:
    """
    Verify config translation exposes explicit R4-02 signal targets to the precompute runner.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Runner runtime settings are a strict subset of the normalized artifact config contract.
    Raises:
        AssertionError: If signal targets, lookback budgets, or guards drift during translation.
    Side Effects:
        Writes one temporary YAML file.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/config/backtest_artifacts_runtime_config.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    config_path = _write_backtest_artifacts_config(tmp_path, body=_VALID_BACKTEST_ARTIFACTS_CONFIG)
    config = load_backtest_artifacts_runtime_config(config_path)
    settings = config.to_precompute_runtime_settings(config_sha256="a" * 64)

    assert settings.price_tail_bars_1m == 100
    assert settings.mapping_tail_bars_1m == 200
    assert settings.signal_artifacts == (
        ArtifactSignalValidationSpecV2(timeframe="15m", indicator_id="ma.ema"),
        ArtifactSignalValidationSpecV2(timeframe="1h", indicator_id="ma.sma"),
    )
    assert settings.hit_times_tail_bars_1m == 400
    assert settings.max_signal_rows_per_artifact == 3000


def test_load_backtest_artifacts_runtime_config_expands_all_supported_signal_artifacts_literal(
    tmp_path: Path,
) -> None:
    """
    Verify machine-readable `all_supported_v1` expands to the full signal registry.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Artifact configs may opt into full signal precompute coverage without enumerating every
        `(timeframe, indicator_id)` pair in YAML.
    Raises:
        AssertionError: If expansion drifts from signal registry ordering.
    Side Effects:
        Writes one temporary YAML file.
    """
    config_path = _write_backtest_artifacts_config(
        tmp_path,
        body=_VALID_BACKTEST_ARTIFACTS_CONFIG.replace(
            "signal_artifacts:\n"
            "      - timeframe: 1h\n"
            "        indicator_id: ma.sma\n"
            "      - timeframe: 15m\n"
            "        indicator_id: ma.ema",
            "signal_artifacts: all_supported_v1",
        ),
    )

    config = load_backtest_artifacts_runtime_config(config_path)

    expected = tuple(
        (timeframe, indicator_id)
        for timeframe in ("15m", "30m", "1h", "2h", "4h", "6h", "8h", "1d", "2d", "3d")
        for indicator_id in supported_indicator_ids_for_signals_v1()
    )
    assert tuple(
        (item.timeframe, item.indicator_id) for item in config.validation_plan.signal_artifacts
    ) == expected


def test_load_backtest_artifacts_runtime_config_rejects_duplicate_yaml_keys(
    tmp_path: Path,
) -> None:
    """
    Verify duplicate YAML keys are rejected before strict schema validation proceeds.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Duplicate YAML keys must not silently override earlier config values.
    Raises:
        AssertionError: If duplicate key does not raise ValueError.
    Side Effects:
        Writes one temporary YAML file.
    """
    config_path = _write_backtest_artifacts_config(
        tmp_path,
        body=_VALID_BACKTEST_ARTIFACTS_CONFIG.replace(
            "  artifact_root: artifacts/backtest/v2",
            "  artifact_root: artifacts/backtest/v2\n  artifact_root: artifacts/backtest/v3",
            1,
        ),
    )

    with pytest.raises(ValueError, match="duplicate YAML key 'artifact_root'"):
        load_backtest_artifacts_runtime_config(config_path)


def test_build_backtest_artifacts_runtime_config_hash_is_deterministic(tmp_path: Path) -> None:
    """
    Verify artifact config hash is deterministic for canonically equivalent payloads.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Canonical ordering must make semantically equivalent author input hash equally.
    Raises:
        AssertionError: If hash differs between equivalent config payloads.
    Side Effects:
        Writes temporary YAML files.
    """
    config_a = load_backtest_artifacts_runtime_config(
        _write_backtest_artifacts_config(tmp_path, body=_VALID_BACKTEST_ARTIFACTS_CONFIG)
    )
    config_b = load_backtest_artifacts_runtime_config(
        _write_backtest_artifacts_config(
            tmp_path,
            body=_VALID_BACKTEST_ARTIFACTS_CONFIG.replace(
                "    price_timeframes: [1h, 1m, 15m]",
                "    price_timeframes: [15m, 1h, 1m]",
                1,
            ),
            filename="backtest_artifacts_b.yaml",
        )
    )

    assert build_backtest_artifacts_runtime_config_hash(config=config_a) == (
        build_backtest_artifacts_runtime_config_hash(config=config_b)
    )
