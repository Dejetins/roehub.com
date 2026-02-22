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
    assert config.reporting.top_trades_n_default == 3
    assert config.execution.init_cash_quote_default == 10000.0
    assert config.execution.fixed_quote_default == 100.0
    assert config.execution.safe_profit_percent_default == 30.0
    assert config.execution.slippage_pct_default == 0.01
    assert dict(config.execution.fee_pct_default_by_market_id) == {
        1: 0.075,
        2: 0.1,
        3: 0.075,
        4: 0.1,
    }


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
    assert config.reporting.top_trades_n_default == 3
    assert config.execution.init_cash_quote_default == 10000.0
    assert config.execution.fixed_quote_default == 100.0
    assert config.execution.safe_profit_percent_default == 30.0
    assert config.execution.slippage_pct_default == 0.01
    assert dict(config.execution.fee_pct_default_by_market_id) == {
        1: 0.075,
        2: 0.1,
        3: 0.075,
        4: 0.1,
    }


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


def test_load_backtest_runtime_config_reads_execution_overrides(tmp_path: Path) -> None:
    """
    Verify loader parses explicit execution defaults with fail-fast validation semantics.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `backtest.execution` mapping follows v1 execution-engine runtime schema.
    Raises:
        AssertionError: If parsed execution values mismatch YAML payload.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  warmup_bars_default: 10
  top_k_default: 20
  preselect_default: 30
  reporting:
    top_trades_n_default: 5
  execution:
    init_cash_quote_default: 5000
    fixed_quote_default: 250
    safe_profit_percent_default: 15
    slippage_pct_default: 0.05
    fee_pct_default_by_market_id:
      1: 0.05
      8: 0.2
""".strip(),
    )

    config = load_backtest_runtime_config(config_path)

    assert config.warmup_bars_default == 10
    assert config.top_k_default == 20
    assert config.preselect_default == 30
    assert config.reporting.top_trades_n_default == 5
    assert config.execution.init_cash_quote_default == 5000.0
    assert config.execution.fixed_quote_default == 250.0
    assert config.execution.safe_profit_percent_default == 15.0
    assert config.execution.slippage_pct_default == 0.05
    assert dict(config.execution.fee_pct_default_by_market_id) == {1: 0.05, 8: 0.2}


def test_load_backtest_runtime_config_rejects_invalid_reporting_defaults(tmp_path: Path) -> None:
    """
    Verify loader fails fast when reporting defaults violate deterministic schema bounds.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `backtest.reporting.top_trades_n_default` must be strictly positive.
    Raises:
        AssertionError: If invalid reporting payload does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_backtest_config(
        tmp_path,
        body="""
version: 1
backtest:
  reporting:
    top_trades_n_default: 0
""".strip(),
    )

    with pytest.raises(ValueError, match="backtest.reporting.top_trades_n_default"):
        load_backtest_runtime_config(config_path)
