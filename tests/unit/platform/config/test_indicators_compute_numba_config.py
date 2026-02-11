from __future__ import annotations

from pathlib import Path

import pytest

from trading.platform.config import load_indicators_compute_numba_config


def _write_indicators_config(tmp_path: Path, *, body: str) -> Path:
    """
    Write temporary indicators YAML used by config-loader tests.

    Args:
        tmp_path: pytest temporary path fixture.
        body: Full YAML content.
    Returns:
        Path: Written config path.
    Assumptions:
        Input text is valid UTF-8.
    Raises:
        OSError: If write fails.
    Side Effects:
        Creates one temp file.
    """
    config_path = tmp_path / "indicators.yaml"
    config_path.write_text(body, encoding="utf-8")
    return config_path


def test_load_indicators_compute_numba_config_reads_yaml_compute_section(tmp_path: Path) -> None:
    """
    Verify loader reads `compute.numba` values from indicators YAML.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `ROEHUB_INDICATORS_CONFIG` points to explicit YAML path.
    Raises:
        AssertionError: If parsed config fields mismatch YAML payload.
    Side Effects:
        None.
    """
    config_path = _write_indicators_config(
        tmp_path,
        body="""
schema_version: 1
compute:
  numba:
    numba_num_threads: 3
    numba_cache_dir: ".cache/test-numba"
    max_compute_bytes_total: 10485760
defaults: {}
""".strip(),
    )
    environ = {
        "ROEHUB_INDICATORS_CONFIG": str(config_path),
        "ROEHUB_ENV": "dev",
    }

    config = load_indicators_compute_numba_config(environ=environ)

    assert config.numba_num_threads == 3
    assert str(config.numba_cache_dir) == ".cache/test-numba"
    assert config.max_compute_bytes_total == 10_485_760


def test_load_indicators_compute_numba_config_env_overrides_have_priority(tmp_path: Path) -> None:
    """
    Verify env overrides take priority over YAML and defaults.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `NUMBA_NUM_THREADS` and `NUMBA_CACHE_DIR` are accepted override env keys.
    Raises:
        AssertionError: If loader does not prioritize env values.
    Side Effects:
        None.
    """
    config_path = _write_indicators_config(
        tmp_path,
        body="""
schema_version: 1
compute:
  numba:
    numba_num_threads: 2
    numba_cache_dir: ".cache/yaml-numba"
    max_compute_bytes_total: 5242880
defaults: {}
""".strip(),
    )
    environ = {
        "ROEHUB_INDICATORS_CONFIG": str(config_path),
        "NUMBA_NUM_THREADS": "5",
        "NUMBA_CACHE_DIR": str(tmp_path / "env-numba-cache"),
        "ROEHUB_MAX_COMPUTE_BYTES_TOTAL": "7777777",
    }

    config = load_indicators_compute_numba_config(environ=environ)

    assert config.numba_num_threads == 5
    assert config.numba_cache_dir == tmp_path / "env-numba-cache"
    assert config.max_compute_bytes_total == 7_777_777


def test_load_indicators_compute_numba_config_rejects_invalid_env_name(tmp_path: Path) -> None:
    """
    Verify loader fails fast for unsupported `ROEHUB_ENV` values.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Allowed env names are strictly `dev`, `prod`, and `test`.
    Raises:
        AssertionError: If invalid env value does not raise ValueError.
    Side Effects:
        None.
    """
    _write_indicators_config(
        tmp_path,
        body="""
schema_version: 1
defaults: {}
""".strip(),
    )
    environ = {"ROEHUB_ENV": "staging"}

    with pytest.raises(ValueError):
        load_indicators_compute_numba_config(environ=environ)
