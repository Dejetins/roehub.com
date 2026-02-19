from __future__ import annotations

from pathlib import Path

import pytest

from trading.contexts.backtest.adapters.outbound.config import (
    load_backtest_runtime_config,
    resolve_backtest_config_path,
)


def _write_backtest_config(tmp_path: Path, *, body: str) -> Path:
    """
    Write temporary Backtest runtime YAML used by config-loader tests.

    Args:
        tmp_path: pytest temporary directory fixture.
        body: Full YAML content.
    Returns:
        Path: Written config path.
    Assumptions:
        Input text is valid UTF-8.
    Raises:
        OSError: If write operation fails.
    Side Effects:
        Creates one temp YAML file.
    """
    config_path = tmp_path / "backtest.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_load_backtest_runtime_config_reads_yaml_values() -> None:
    """
    Verify loader parses documented Backtest defaults from source-of-truth YAML.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Config schema follows BKT-EPIC-01 runtime contract.
    Raises:
        AssertionError: If parsed values differ from YAML payload.
    Side Effects:
        None.
    """
    config = load_backtest_runtime_config(Path("configs/dev/backtest.yaml"))

    assert config.version == 1
    assert config.warmup_bars_default == 200
    assert config.top_k_default == 300
    assert config.preselect_default == 20000


def test_load_backtest_runtime_config_uses_defaults_when_keys_absent(tmp_path: Path) -> None:
    """
    Verify missing optional scalar keys fallback to documented defaults.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Only `version` is required at top level in current runtime schema.
    Raises:
        AssertionError: If fallback defaults are not applied.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest: {}
""".strip(),
    )

    config = load_backtest_runtime_config(config_path)

    assert config.warmup_bars_default == 200
    assert config.top_k_default == 300
    assert config.preselect_default == 20000


def test_resolve_backtest_config_path_precedence() -> None:
    """
    Verify path resolution precedence is override env first, then env fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fallback format is `configs/<ROEHUB_ENV>/backtest.yaml`.
    Raises:
        AssertionError: If precedence order differs from BKT-EPIC-01 contract.
    Side Effects:
        None.
    """
    environ = {
        "ROEHUB_ENV": "prod",
        "ROEHUB_BACKTEST_CONFIG": "configs/test/custom-backtest.yaml",
    }

    assert resolve_backtest_config_path(environ=environ) == Path(
        "configs/test/custom-backtest.yaml"
    )

    assert resolve_backtest_config_path(environ={"ROEHUB_ENV": "test"}) == Path(
        "configs/test/backtest.yaml"
    )

    assert resolve_backtest_config_path(environ={}) == Path("configs/dev/backtest.yaml")


def test_resolve_backtest_config_path_rejects_invalid_env_name() -> None:
    """
    Verify unsupported `ROEHUB_ENV` value fails fast with deterministic message.

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
        resolve_backtest_config_path(environ={"ROEHUB_ENV": "stage"})

