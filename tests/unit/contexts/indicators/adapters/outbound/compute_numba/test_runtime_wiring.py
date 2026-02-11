from __future__ import annotations

import os
from pathlib import Path

import numba
import pytest

from trading.contexts.indicators.adapters.outbound.compute_numba import (
    apply_numba_runtime_config,
    ensure_numba_cache_dir_writable,
)
from trading.platform.config import IndicatorsComputeNumbaConfig


def test_apply_numba_runtime_config_sets_effective_threads(tmp_path: Path) -> None:
    """
    Verify runtime wiring applies configured threads and exports NUMBA env vars.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `numba.set_num_threads` accepts `1` across supported environments.
    Raises:
        AssertionError: If effective threads/env vars mismatch expected values.
    Side Effects:
        Temporarily mutates process env and numba thread setting.
    """
    old_threads = int(numba.get_num_threads())
    old_env_threads = os.environ.get("NUMBA_NUM_THREADS")
    old_env_cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    cache_dir = tmp_path / "numba-cache"
    try:
        config = IndicatorsComputeNumbaConfig(
            numba_num_threads=1,
            numba_cache_dir=cache_dir,
            max_compute_bytes_total=5 * 1024**3,
        )
        effective = apply_numba_runtime_config(config=config)
        assert effective == 1
        assert int(numba.get_num_threads()) == 1
        assert os.environ.get("NUMBA_NUM_THREADS") in (None, "1")
        assert os.environ["NUMBA_CACHE_DIR"] == str(cache_dir)
    finally:
        numba.set_num_threads(old_threads)
        _restore_env("NUMBA_NUM_THREADS", old_env_threads)
        _restore_env("NUMBA_CACHE_DIR", old_env_cache_dir)


def test_ensure_numba_cache_dir_writable_fails_when_path_is_file(tmp_path: Path) -> None:
    """
    Verify fail-fast behavior when NUMBA cache path points to a regular file.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Creating directory over existing file raises `OSError`.
    Raises:
        AssertionError: If expected ValueError is not raised.
    Side Effects:
        Creates one temporary file.
    """
    invalid_path = tmp_path / "numba-cache-file"
    invalid_path.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(ValueError):
        ensure_numba_cache_dir_writable(path=invalid_path)


def _restore_env(key: str, value: str | None) -> None:
    """
    Restore one environment variable to previous state.

    Args:
        key: Environment variable name.
        value: Previous value or None if missing.
    Returns:
        None.
    Assumptions:
        Caller captured original env state before mutation.
    Raises:
        None.
    Side Effects:
        Mutates process environment mapping.
    """
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
