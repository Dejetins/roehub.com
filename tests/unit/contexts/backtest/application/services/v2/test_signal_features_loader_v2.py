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
    MmapSignalFeaturesLoaderV2,
)


@pytest.fixture()
def synthetic_artifact_store_v2(tmp_path: Path) -> SyntheticArtifactStoreV2:
    """
    Build a strict synthetic artifact store used by signal-feature loader tests.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        SyntheticArtifactStoreV2: Deterministic strict artifact store fixture.
    Assumptions:
        Loader tests need valid signal and additive signal-feature manifests by default.
    Raises:
        OSError: If the synthetic artifact tree cannot be created.
    Side Effects:
        Creates a temporary artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    return build_synthetic_artifact_store_v2(tmp_path=tmp_path)


def test_signal_features_loader_v2_loads_full_matrix_and_subset_rows(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the mmap loader opens strict feature matrices and deterministic subset row views.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Runtime start already pinned one immutable slot context before loading feature matrices.
    Raises:
        AssertionError: If loaded matrix or subset row behavior is incorrect.
    Side Effects:
        Memory-maps deterministic `features.f32.npy` files from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    loader = MmapSignalFeaturesLoaderV2(artifact_loader=store.loader)

    signal_features_matrix = loader.load_signal_features_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    repeated_matrix = loader.load_signal_features_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    slice_rows = loader.load_signal_feature_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_selection=slice(0, 1),
    )
    explicit_rows = loader.load_signal_feature_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_selection=(0,),
    )

    assert isinstance(signal_features_matrix.matrix, np.memmap)
    assert repeated_matrix is signal_features_matrix
    assert signal_features_matrix.matrix.dtype == np.float32
    assert signal_features_matrix.matrix.shape == (2, 6)
    assert slice_rows.feature_names == signal_features_matrix.manifest.feature_names
    assert explicit_rows.feature_names == signal_features_matrix.manifest.feature_names
    np.testing.assert_allclose(
        signal_features_matrix.matrix,
        np.array(
            (
                (1.0, 0.0, 1.0, 0.5, -1.0, 1.0),
                (1.0, 1.0, 0.0, 0.5, 1.0, 1.0),
            ),
            dtype=np.float32,
        ),
    )
    assert isinstance(slice_rows.rows, np.memmap)
    assert slice_rows.rows.shape == (1, 6)
    assert explicit_rows.rows.shape == (1, 6)
    np.testing.assert_allclose(
        explicit_rows.rows,
        np.array(((1.0, 0.0, 1.0, 0.5, -1.0, 1.0),), dtype=np.float32),
    )


def test_signal_features_loader_v2_try_load_returns_none_for_legacy_slot(
    tmp_path: Path,
) -> None:
    """
    Verify the optional loader keeps legacy slots readable by returning `None`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Only the absence of the additive `signal_features` reference should produce `None`.
    Raises:
        AssertionError: If legacy slots raise instead of returning `None`.
    Side Effects:
        Builds one synthetic legacy-style inactive slot under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        inactive_include_signal_features=False,
    )
    context = _inactive_context(store)
    loader = MmapSignalFeaturesLoaderV2(artifact_loader=store.loader)

    assert (
        loader.try_load_signal_features_matrix(
            context=context,
            timeframe="15m",
            indicator_id="ma.ema",
        )
        is None
    )
    assert (
        loader.try_load_signal_feature_rows(
            context=context,
            timeframe="15m",
            indicator_id="ma.ema",
            row_selection=(0,),
        )
        is None
    )
    with pytest.raises(ValueError, match="does not declare optional signal_features"):
        loader.load_signal_features_matrix(
            context=context,
            timeframe="15m",
            indicator_id="ma.ema",
        )


def test_signal_features_loader_v2_run_scoped_loader_keeps_cache_ownership_per_run(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify `run-scoped` feature loaders keep same-run reuse without prototype cache retention.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        The long-lived Stage A builder should keep only a prototype loader while each run owns a
        fresh signal-features cache.
    Raises:
        AssertionError: If `run_scoped()` reuses prototype signal-features cache entries.
    Side Effects:
        Memory-maps deterministic `features.f32.npy` files from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    prototype_loader = MmapSignalFeaturesLoaderV2(artifact_loader=store.loader)

    prototype_matrix = prototype_loader.load_signal_features_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    run_scoped_loader = prototype_loader.run_scoped()
    run_scoped_matrix = run_scoped_loader.load_signal_features_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    repeated_run_scoped_matrix = run_scoped_loader.load_signal_features_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )

    assert run_scoped_loader is not prototype_loader
    assert prototype_loader.load_signal_features_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    ) is prototype_matrix
    assert run_scoped_matrix is not prototype_matrix
    assert repeated_run_scoped_matrix is run_scoped_matrix


def test_signal_features_loader_v2_rejects_signal_reference_path_drift(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify feature loading fails fast when signal-manifest feature reference path metadata drifts.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Runtime feature loading must reject manifest-path drift before opening feature metadata.
    Raises:
        AssertionError: If feature-reference path drift is accepted.
    Side Effects:
        Rewrites the inactive slot signal manifest under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
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
    payload["signal_features"]["manifest_path"] = "signal_features/15m/ma.ema/renamed.yaml"
    _write_yaml(signal_manifest_path, payload)
    context = _inactive_context(store)
    loader = MmapSignalFeaturesLoaderV2(artifact_loader=store.loader)

    with pytest.raises(ValueError, match="signal_features manifest_path"):
        loader.load_signal_features_matrix(
            context=context,
            timeframe="15m",
            indicator_id="ma.ema",
        )


def test_signal_features_loader_v2_avoids_directory_scanning(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the feature loader never uses directory scanning helpers for runtime artifact reads.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Additive feature loading must stay fully explicit-path and manifest-driven.
    Raises:
        AssertionError: If one loader code path relies on scanning.
    Side Effects:
        Temporarily replaces scanning helpers on `Path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    store = synthetic_artifact_store_v2
    context = _inactive_context(store)
    loader = MmapSignalFeaturesLoaderV2(artifact_loader=store.loader)
    monkeypatch.setattr(Path, "iterdir", _forbid_directory_scan)
    monkeypatch.setattr(Path, "glob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "rglob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "walk", _forbid_directory_scan)

    signal_features_matrix = loader.load_signal_features_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )
    feature_rows = loader.load_signal_feature_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_selection=slice(0, 1),
    )

    assert signal_features_matrix.timeframe == "15m"
    assert feature_rows.rows.shape == (1, 6)


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
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - tests/unit/contexts/backtest/application/services/v2/test_signal_features_loader_v2.py
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
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_signal_features_loader_v2.py
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
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/test_signal_features_loader_v2.py
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
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/signal_features_loader_v2.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_yaml_backtest_artifact_loader_v2.py
    """
    raise AssertionError("directory scanning is forbidden in signal features loader v2")
