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
    MmapSignalMatrixLoaderV2,
)


@pytest.fixture()
def synthetic_artifact_store_v2(tmp_path: Path) -> SyntheticArtifactStoreV2:
    """
    Build a strict synthetic artifact store used by signal-loader tests.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        SyntheticArtifactStoreV2: Deterministic strict artifact store fixture.
    Assumptions:
        Loader tests need valid root and per-indicator signal manifests by default.
    Raises:
        OSError: If the synthetic artifact tree cannot be created.
    Side Effects:
        Creates a temporary artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    return build_synthetic_artifact_store_v2(tmp_path=tmp_path)


def test_signal_matrix_loader_v2_loads_full_matrix_and_subset_rows(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the mmap loader opens a strict signal matrix and deterministic subset row views.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Runtime start already pinned one immutable slot context before loading signal matrices.
    Raises:
        AssertionError: If loaded matrix or subset row behavior is incorrect.
    Side Effects:
        Memory-maps deterministic `.npy` files from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    loader = MmapSignalMatrixLoaderV2(artifact_loader=store.loader)

    signal_matrix = loader.load_signal_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    slice_rows = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_selection=slice(0, 1),
    )
    first_read = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_selection=(0,),
    )
    second_read = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_selection=(0,),
    )

    assert isinstance(signal_matrix.matrix, np.memmap)
    assert tuple(int(value) for value in signal_matrix.matrix[0]) == (-1, 0)
    assert isinstance(slice_rows, np.memmap)
    assert slice_rows.shape == (1, 2)
    assert np.array_equal(first_read, second_read)


def test_signal_matrix_loader_v2_rejects_catalog_path_drift(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify signal loading fails fast when root-manifest catalog path metadata drifts.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Runtime signal loading must reject catalog path drift before opening the signal manifest.
    Raises:
        AssertionError: If catalog path drift is accepted.
    Side Effects:
        Rewrites the inactive slot root manifest under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    store = synthetic_artifact_store_v2
    root_manifest_path = store.builder.slot_manifest_path(store.coordinates, store.inactive_slot)
    payload = _yaml_payload(root_manifest_path)
    payload["signals"]["manifests"][0]["manifest_path"] = "signals/15m/ma.ema/renamed.yaml"
    _write_yaml(root_manifest_path, payload)
    context = _inactive_context(store)
    loader = MmapSignalMatrixLoaderV2(artifact_loader=store.loader)

    with pytest.raises(ValueError, match="catalog manifest_path"):
        loader.load_signal_matrix(context=context, timeframe="15m", indicator_id="ma.ema")


def test_signal_matrix_loader_v2_rejects_timeline_drift(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify signal loading fails fast when signal-manifest timeline drifts from `prices/<tf>`.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Runtime signal loading must reject timeline drift before Stage A reads the matrix.
    Raises:
        AssertionError: If timeline drift is accepted.
    Side Effects:
        Rewrites the inactive slot signal manifest under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    store = synthetic_artifact_store_v2
    signal_manifest_path = store.builder.signal_paths(
        store.coordinates,
        store.inactive_slot,
        "15m",
        "ma.ema",
    ).manifest
    payload = _yaml_payload(signal_manifest_path)
    payload["timeline"]["bar_count"] = 3
    _write_yaml(signal_manifest_path, payload)
    context = _inactive_context(store)
    loader = MmapSignalMatrixLoaderV2(artifact_loader=store.loader)

    with pytest.raises(ValueError, match="timeline must match prices/15m coverage"):
        loader.load_signal_matrix(context=context, timeframe="15m", indicator_id="ma.ema")


def test_signal_matrix_loader_v2_rejects_unsorted_row_selection(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify subset row loading rejects ambiguous explicit row ordering.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Deterministic subset row reads must reject unsorted or duplicate row indexes.
    Raises:
        AssertionError: If unsorted row selection is accepted.
    Side Effects:
        Reads strict artifact metadata from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    loader = MmapSignalMatrixLoaderV2(artifact_loader=store.loader)

    with pytest.raises(ValueError, match="strictly increasing"):
        loader.load_signal_rows(
            context=context,
            timeframe="15m",
            indicator_id="ma.ema",
            row_selection=(1, 0),
        )


def test_signal_matrix_loader_v2_avoids_directory_scanning(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify signal loader never uses directory scanning helpers for runtime artifact reads.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        R6-01 signal loading must stay fully explicit-path and manifest-driven.
    Raises:
        AssertionError: If one loader code path relies on scanning.
    Side Effects:
        Temporarily replaces scanning helpers on `Path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    loader = MmapSignalMatrixLoaderV2(artifact_loader=store.loader)
    monkeypatch.setattr(Path, "iterdir", _forbid_directory_scan)
    monkeypatch.setattr(Path, "glob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "rglob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "walk", _forbid_directory_scan)

    signal_matrix = loader.load_signal_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    signal_rows = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_selection=slice(0, 1),
    )

    assert signal_matrix.timeframe == "15m"
    assert signal_rows.shape == (1, 2)


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
      - tests/unit/contexts/backtest/application/services/v2/test_signal_matrix_loader_v2.py
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
      - tests/unit/contexts/backtest/application/services/v2/test_signal_matrix_loader_v2.py
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
      - src/trading/contexts/backtest/application/services/v2/signal_matrix_loader.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_yaml_backtest_artifact_loader_v2.py
    """
    raise AssertionError("directory scanning is forbidden in signal matrix loader v2")
