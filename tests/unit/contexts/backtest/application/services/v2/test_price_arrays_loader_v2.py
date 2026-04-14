from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    SyntheticArtifactStoreV2,
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.application.services import (
    ArtifactPinnedIdentityV2,
    ArtifactSlotPinnedRuntimeContextV2,
    ArtifactSlotResolverV2,
    MmapPriceArraysLoaderV2,
)


@pytest.fixture()
def synthetic_artifact_store_v2(tmp_path: Path) -> SyntheticArtifactStoreV2:
    """
    Build a strict synthetic artifact store used by price-loader tests.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        SyntheticArtifactStoreV2: Deterministic strict artifact store fixture.
    Assumptions:
        Loader tests need valid root, mapping, signal, and hit-times manifests by default.
    Raises:
        OSError: If the synthetic artifact tree cannot be created.
    Side Effects:
        Creates a temporary artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    return build_synthetic_artifact_store_v2(tmp_path=tmp_path)


def test_price_arrays_loader_v2_loads_prices_mappings_and_hit_times_with_mmap(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the mmap loader opens explicit `prices/<tf>`, `mappings/<tf>`, and `hit_times/1m`.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Runtime startup already pinned one immutable slot context before loading arrays.
    Raises:
        AssertionError: If loaded arrays or explicit contract metadata are incorrect.
    Side Effects:
        Memory-maps deterministic `.npy` files from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)

    one_minute_prices = loader.load_price_arrays(context=context, timeframe="1m")
    mapping_arrays = loader.load_mapping_arrays(context=context, timeframe="15m")
    hit_times_arrays = loader.load_hit_times_arrays(context=context)
    repeated_prices = loader.load_price_arrays(context=context, timeframe="1m")
    repeated_mappings = loader.load_mapping_arrays(context=context, timeframe="15m")
    repeated_hit_times = loader.load_hit_times_arrays(context=context)

    assert isinstance(one_minute_prices.open_time, np.memmap)
    assert isinstance(one_minute_prices.ohlcv, np.memmap)
    assert tuple(int(value) for value in one_minute_prices.open_time) == (
        1000,
        2000,
        3000,
        4000,
    )
    assert tuple(int(value) for value in mapping_arrays.bar_open_1m_idx) == (0, 2)
    assert tuple(int(value) for value in mapping_arrays.bar_close_1m_idx) == (1, 3)
    assert hit_times_arrays.manifest.sentinel_index == 4
    assert isinstance(hit_times_arrays.long_tp, np.memmap)
    assert repeated_prices is one_minute_prices
    assert repeated_mappings is mapping_arrays
    assert repeated_hit_times is hit_times_arrays


def test_price_arrays_loader_v2_rejects_price_manifest_path_drift(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify price loading fails fast when root-manifest path metadata drifts.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Runtime price loading must reject manifest path drift before any downstream kernel work.
    Raises:
        AssertionError: If path drift is accepted.
    Side Effects:
        Rewrites the inactive slot root manifest under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    store = synthetic_artifact_store_v2
    root_manifest_path = store.builder.slot_manifest_path(store.coordinates, store.inactive_slot)
    payload = _yaml_payload(root_manifest_path)
    payload["prices"][0]["open_time"]["path"] = "prices/1m/renamed.i64.npy"
    _write_yaml(root_manifest_path, payload)
    context = _inactive_context(store)
    loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)

    with pytest.raises(ValueError, match="manifest path"):
        loader.load_price_arrays(context=context, timeframe="1m")


def test_price_arrays_loader_v2_run_scoped_loader_keeps_cache_ownership_per_run(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify `run-scoped` price loaders isolate large family caches from long-lived prototypes.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        API-owned prototypes may be reused across requests only when every request gets a fresh
        cache-owning clone.
    Raises:
        AssertionError: If `run_scoped()` reuses prototype cache entries for prices, mappings, or
            hit-times.
    Side Effects:
        Memory-maps deterministic `.npy` files from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    prototype_loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)

    prototype_prices = prototype_loader.load_price_arrays(context=context, timeframe="1m")
    prototype_mappings = prototype_loader.load_mapping_arrays(
        context=context,
        timeframe="15m",
    )
    prototype_hit_times = prototype_loader.load_hit_times_arrays(context=context)
    run_scoped_loader = prototype_loader.run_scoped()
    run_scoped_prices = run_scoped_loader.load_price_arrays(context=context, timeframe="1m")
    run_scoped_mappings = run_scoped_loader.load_mapping_arrays(
        context=context,
        timeframe="15m",
    )
    run_scoped_hit_times = run_scoped_loader.load_hit_times_arrays(context=context)

    assert run_scoped_loader is not prototype_loader
    assert prototype_loader.load_price_arrays(context=context, timeframe="1m") is prototype_prices
    assert prototype_loader.load_mapping_arrays(
        context=context,
        timeframe="15m",
    ) is prototype_mappings
    assert prototype_loader.load_hit_times_arrays(context=context) is prototype_hit_times
    assert run_scoped_prices is not prototype_prices
    assert run_scoped_mappings is not prototype_mappings
    assert run_scoped_hit_times is not prototype_hit_times
    assert run_scoped_loader.load_price_arrays(context=context, timeframe="1m") is run_scoped_prices
    assert run_scoped_loader.load_mapping_arrays(
        context=context,
        timeframe="15m",
    ) is run_scoped_mappings
    assert run_scoped_loader.load_hit_times_arrays(context=context) is run_scoped_hit_times


def test_price_arrays_loader_v2_rejects_mapping_bounds_drift(tmp_path: Path) -> None:
    """
    Verify mapping loading fails fast when `bar_close_1m_idx` escapes `prices/1m` bounds.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Runtime mapping loading must reject explicit bounds drift before Stage A uses the arrays.
    Raises:
        AssertionError: If out-of-bounds mapping indexes are accepted.
    Side Effects:
        Builds one synthetic artifact tree with corrupted inactive mapping arrays.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        inactive_mapping_close_idx=np.array([1, 4], dtype=np.uint32),
    )
    context = _inactive_context(store)
    loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)

    with pytest.raises(ValueError, match="within prices/1m bounds"):
        loader.load_mapping_arrays(context=context, timeframe="15m")


def test_price_arrays_loader_v2_rejects_hit_times_sentinel_drift(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify hit-times loading fails fast when `sentinel_index` drifts from timeline count.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Runtime `hit_times/1m` loading must reject sentinel drift before Stage B uses the tables.
    Raises:
        AssertionError: If sentinel drift is accepted.
    Side Effects:
        Rewrites the inactive slot hit-times manifest under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    store = synthetic_artifact_store_v2
    hit_times_manifest_path = store.builder.hit_times_manifest_path(
        store.coordinates,
        store.inactive_slot,
    )
    payload = _yaml_payload(hit_times_manifest_path)
    payload["sentinel_index"] = 3
    _write_yaml(hit_times_manifest_path, payload)
    context = _inactive_context(store)
    loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)

    with pytest.raises(ValueError, match="sentinel_index"):
        loader.load_hit_times_arrays(context=context)


def test_price_arrays_loader_v2_avoids_directory_scanning(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify price loader never uses directory scanning helpers for runtime artifact reads.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        R6-01 runtime loading must stay fully explicit-path and manifest-driven.
    Raises:
        AssertionError: If one loader code path relies on scanning.
    Side Effects:
        Temporarily replaces scanning helpers on `Path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    loader = MmapPriceArraysLoaderV2(artifact_loader=store.loader)
    monkeypatch.setattr(Path, "iterdir", _forbid_directory_scan)
    monkeypatch.setattr(Path, "glob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "rglob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "walk", _forbid_directory_scan)

    price_arrays = loader.load_price_arrays(context=context, timeframe="1m")
    mapping_arrays = loader.load_mapping_arrays(context=context, timeframe="15m")
    hit_times_arrays = loader.load_hit_times_arrays(context=context)

    assert price_arrays.timeframe == "1m"
    assert mapping_arrays.timeframe == "15m"
    assert hit_times_arrays.manifest.timeframe == "1m"


def _inactive_context(
    store: SyntheticArtifactStoreV2,
) -> ArtifactSlotPinnedRuntimeContextV2:
    """
    Resolve one deterministic pinned context for the synthetic inactive slot.

    Args:
        store: Synthetic artifact store fixture.
    Returns:
        ArtifactSlotPinnedRuntimeContextV2: Pinned inactive-slot runtime context.
    Assumptions:
        Tests exercise background-style explicit slot loading against the inactive slot.
    Raises:
        ValueError: If the synthetic slot metadata is inconsistent.
    Side Effects:
        Reads strict slot metadata from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - tests/unit/contexts/backtest/application/services/v2/test_artifact_slot_resolver_v2.py
    """
    resolver = ArtifactSlotResolverV2(artifact_loader=store.loader)
    return resolver.resolve_pinned_context(
        store.coordinates,
        ArtifactPinnedIdentityV2(
            artifact_slot=store.inactive_slot,
            slot_generation=5,
            artifact_asof_date="2026-03-26",
            artifact_manifest_hash="b" * 64,
        ),
    )


def _yaml_payload(path: Path) -> dict[str, Any]:
    """
    Read one YAML file into a mutable mapping for test-time contract drift injection.

    Args:
        path: YAML file path to read.
    Returns:
        dict[str, object]: Mutable payload mapping.
    Assumptions:
        Synthetic manifests are valid YAML mappings before mutation.
    Raises:
        ValueError: If the payload is not a mapping.
    Side Effects:
        Reads one UTF-8 YAML file from disk.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_price_arrays_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """
    Rewrite one YAML mapping with deterministic key order for test-time drift injection.

    Args:
        path: YAML file path to rewrite.
        payload: Mutable YAML mapping to serialize.
    Returns:
        None.
    Assumptions:
        Tests control the payload key order before serialization.
    Raises:
        OSError: If the file cannot be written.
    Side Effects:
        Rewrites one UTF-8 YAML file on disk.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_price_arrays_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _forbid_directory_scan(*_args: object, **_kwargs: object) -> None:
    """
    Fail the test immediately if a loader code path attempts directory scanning.

    Args:
        *_args: Positional arguments ignored by the failure stub.
        **_kwargs: Keyword arguments ignored by the failure stub.
    Returns:
        None.
    Assumptions:
        Explicit-path runtime loaders must never need scanning helpers.
    Raises:
        AssertionError: Always, to signal forbidden scanning usage.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/price_arrays_loader.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_yaml_backtest_artifact_loader_v2.py
    """
    raise AssertionError("directory scanning is forbidden in price arrays loader v2")
