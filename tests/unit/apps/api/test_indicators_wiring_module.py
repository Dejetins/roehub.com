from __future__ import annotations

from pathlib import Path

from apps.api.wiring.modules import indicators as indicators_module
from trading.platform.config import IndicatorsComputeNumbaConfig


def test_build_artifact_precompute_indicators_compute_overrides_total_budget(
    monkeypatch,
) -> None:
    """
    Verify artifact precompute wiring replaces the public compute-budget ceiling.

    Args:
        monkeypatch: pytest fixture used to intercept builder delegation.
    Returns:
        None.
    Assumptions:
        Offline artifact publish must not inherit the public `max_compute_bytes_total` guard.
    Raises:
        AssertionError: If the delegated compute config keeps the public budget.
    Side Effects:
        Monkeypatches the indicators wiring module in memory.
    """
    captured: dict[str, object] = {}
    sentinel = object()
    base_config = IndicatorsComputeNumbaConfig(
        numba_num_threads=7,
        numba_cache_dir=Path(".cache/test-numba"),
        max_compute_bytes_total=123,
        max_variants_per_compute=456,
    )

    def _fake_build_indicators_compute(*, environ, config=None):
        """
        Capture delegated compute-builder args for assertions.

        Args:
            environ: Environment mapping forwarded by the helper.
            config: Optional config override forwarded by the helper.
        Returns:
            object: Sentinel value used to prove delegation return propagation.
        Assumptions:
            The helper under test delegates through `build_indicators_compute`.
        Raises:
            None.
        Side Effects:
            Stores call arguments in the local `captured` mapping.
        """
        captured["environ"] = dict(environ)
        captured["config"] = config
        return sentinel

    monkeypatch.setattr(
        indicators_module,
        "build_indicators_compute",
        _fake_build_indicators_compute,
    )

    result = indicators_module.build_artifact_precompute_indicators_compute(
        environ={"ROEHUB_ENV": "prod"},
        config=base_config,
    )

    delegated_config = captured["config"]
    assert result is sentinel
    assert isinstance(delegated_config, IndicatorsComputeNumbaConfig)
    assert delegated_config.max_compute_bytes_total == (
        indicators_module._ARTIFACT_PRECOMPUTE_MAX_COMPUTE_BYTES_TOTAL
    )
    assert delegated_config.numba_num_threads == base_config.numba_num_threads
    assert delegated_config.numba_cache_dir == base_config.numba_cache_dir
    assert delegated_config.max_variants_per_compute == base_config.max_variants_per_compute
    assert base_config.max_compute_bytes_total == 123


def test_build_artifact_precompute_indicators_compute_loads_base_config_when_missing(
    monkeypatch,
) -> None:
    """
    Verify artifact precompute wiring loads the base config before overriding the budget.

    Args:
        monkeypatch: pytest fixture used to intercept loader and builder calls.
    Returns:
        None.
    Assumptions:
        Production wiring usually does not pass an explicit config instance.
    Raises:
        AssertionError: If the helper skips config loading or fails to override the budget.
    Side Effects:
        Monkeypatches the indicators wiring module in memory.
    """
    captured: dict[str, object] = {}
    loaded_config = IndicatorsComputeNumbaConfig(
        numba_num_threads=5,
        numba_cache_dir=Path(".cache/prod-numba"),
        max_compute_bytes_total=5 * 1024**3,
        max_variants_per_compute=600_000,
    )

    def _fake_loader(*, environ):
        """
        Return a deterministic loaded config for artifact-helper assertions.

        Args:
            environ: Environment mapping forwarded to the loader.
        Returns:
            IndicatorsComputeNumbaConfig: Fixed base runtime config.
        Assumptions:
            The helper under test should load config exactly once.
        Raises:
            None.
        Side Effects:
            Stores the received environment mapping in the local `captured` mapping.
        """
        captured["loader_environ"] = dict(environ)
        return loaded_config

    def _fake_build_indicators_compute(*, environ, config=None):
        """
        Capture delegated build args after the helper applies its override.

        Args:
            environ: Environment mapping forwarded by the helper.
            config: Optional config override forwarded by the helper.
        Returns:
            str: Sentinel marker proving delegation happened.
        Assumptions:
            The helper delegates to `build_indicators_compute` after config loading.
        Raises:
            None.
        Side Effects:
            Stores delegation args in the local `captured` mapping.
        """
        captured["build_environ"] = dict(environ)
        captured["build_config"] = config
        return "ok"

    monkeypatch.setattr(
        indicators_module,
        "load_indicators_compute_numba_config",
        _fake_loader,
    )
    monkeypatch.setattr(
        indicators_module,
        "build_indicators_compute",
        _fake_build_indicators_compute,
    )

    result = indicators_module.build_artifact_precompute_indicators_compute(
        environ={"ROEHUB_ENV": "prod"},
    )

    delegated_config = captured["build_config"]
    assert result == "ok"
    assert captured["loader_environ"] == {"ROEHUB_ENV": "prod"}
    assert captured["build_environ"] == {"ROEHUB_ENV": "prod"}
    assert isinstance(delegated_config, IndicatorsComputeNumbaConfig)
    assert delegated_config.max_compute_bytes_total == (
        indicators_module._ARTIFACT_PRECOMPUTE_MAX_COMPUTE_BYTES_TOTAL
    )
    assert delegated_config.numba_num_threads == loaded_config.numba_num_threads
    assert delegated_config.numba_cache_dir == loaded_config.numba_cache_dir
    assert delegated_config.max_variants_per_compute == loaded_config.max_variants_per_compute
