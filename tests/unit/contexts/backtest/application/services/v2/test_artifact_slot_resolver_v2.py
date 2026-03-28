from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    SyntheticArtifactStoreV2,
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.application.services import (
    ArtifactPinnedIdentityV2,
    ArtifactSlotResolverV2,
)


@pytest.fixture()
def synthetic_artifact_store_v2(tmp_path: Path) -> SyntheticArtifactStoreV2:
    """
    Build a strict synthetic artifact store used by slot-resolver tests.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        SyntheticArtifactStoreV2: Deterministic strict artifact store fixture.
    Assumptions:
        Resolver tests need valid two-slot manifests and one published `current.yaml` by default.
    Raises:
        OSError: If the synthetic artifact tree cannot be created.
    Side Effects:
        Creates a temporary artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    return build_synthetic_artifact_store_v2(tmp_path=tmp_path)


def test_artifact_slot_resolver_v2_resolves_active_context_from_current_yaml(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the resolver builds one immutable active slot-pinned context from `current.yaml`.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Sync runtime start should pin the currently published slot identity once.
    Raises:
        AssertionError: If active slot identity or explicit manifest paths are incorrect.
    Side Effects:
        Reads strict artifact metadata from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """
    store = synthetic_artifact_store_v2
    resolver = ArtifactSlotResolverV2(artifact_loader=store.loader)

    context = resolver.resolve_active_context(store.coordinates)

    assert context.coordinates == store.coordinates
    assert context.artifact_slot == store.active_slot
    assert context.slot_generation == 4
    assert context.artifact_asof_date == "2026-03-25"
    assert context.artifact_manifest_hash == store.loader.load_current_pointer(
        store.coordinates
    ).manifest_sha256
    assert context.slot_manifest_path == store.builder.slot_manifest_path(
        store.coordinates,
        store.active_slot,
    )
    assert context.slot_root_path == store.builder.slot_root(store.coordinates, store.active_slot)


def test_artifact_slot_resolver_v2_resolves_pinned_context_without_current_yaml(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify the resolver reopens a pinned slot context from persisted run metadata only.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Background runtime start must ignore `current.yaml` and trust immutable persisted pin
        identity.
    Raises:
        AssertionError: If pinned slot identity or explicit manifest paths are incorrect.
    Side Effects:
        Reads one explicit pinned slot manifest from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
    """
    store = synthetic_artifact_store_v2
    resolver = ArtifactSlotResolverV2(artifact_loader=store.loader)
    pinned_identity = ArtifactPinnedIdentityV2(
        artifact_slot=store.inactive_slot,
        slot_generation=5,
        artifact_asof_date="2026-03-26",
        artifact_manifest_hash="b" * 64,
    )

    context = resolver.resolve_pinned_context(store.coordinates, pinned_identity)

    assert context.coordinates == store.coordinates
    assert context.artifact_slot == store.inactive_slot
    assert context.slot_generation == 5
    assert context.artifact_asof_date == "2026-03-26"
    assert context.artifact_manifest_hash == "b" * 64
    assert context.slot_manifest_path == store.builder.slot_manifest_path(
        store.coordinates,
        store.inactive_slot,
    )


def test_artifact_slot_resolver_v2_rejects_current_yaml_manifest_generation_drift(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify active context bootstrap fails fast when `current.yaml` drifts from slot manifest.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Sync startup must fail before runtime work if `current.yaml` generation is inconsistent.
    Raises:
        AssertionError: If resolver accepts mismatched pointer and manifest metadata.
    Side Effects:
        Rewrites the synthetic `current.yaml` payload under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    store = synthetic_artifact_store_v2
    resolver = ArtifactSlotResolverV2(artifact_loader=store.loader)
    current_pointer_path = store.builder.current_pointer_path(store.coordinates)
    payload = yaml.safe_load(current_pointer_path.read_text(encoding="utf-8"))
    payload["slot_generation"] = 99
    current_pointer_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="slot_generation"):
        resolver.resolve_active_context(store.coordinates)


def test_artifact_slot_resolver_v2_rejects_pinned_generation_drift(
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify pinned background bootstrap fails fast when persisted generation drifts.

    Args:
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        Background startup must reject persisted slot identity drift before runtime work begins.
    Raises:
        AssertionError: If resolver accepts mismatched pinned generation and slot manifest.
    Side Effects:
        Reads strict artifact metadata from the synthetic store.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/domain/entities/backtest_job.py
    """
    store = synthetic_artifact_store_v2
    resolver = ArtifactSlotResolverV2(artifact_loader=store.loader)
    pinned_identity = ArtifactPinnedIdentityV2(
        artifact_slot=store.inactive_slot,
        slot_generation=999,
        artifact_asof_date="2026-03-26",
        artifact_manifest_hash="b" * 64,
    )

    with pytest.raises(ValueError, match="slot_generation"):
        resolver.resolve_pinned_context(store.coordinates, pinned_identity)


def test_artifact_slot_resolver_v2_avoids_directory_scanning(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_artifact_store_v2: SyntheticArtifactStoreV2,
) -> None:
    """
    Verify slot resolver never uses directory scanning helpers during bootstrap.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        synthetic_artifact_store_v2: Fixture with a strict synthetic artifact tree.
    Returns:
        None.
    Assumptions:
        R6-01 bootstrap must stay fully explicit-path and manifest-driven.
    Raises:
        AssertionError: If one resolver code path relies on scanning.
    Side Effects:
        Temporarily replaces scanning helpers on `Path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    store = synthetic_artifact_store_v2
    resolver = ArtifactSlotResolverV2(artifact_loader=store.loader)
    monkeypatch.setattr(Path, "iterdir", _forbid_directory_scan)
    monkeypatch.setattr(Path, "glob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "rglob", _forbid_directory_scan)
    monkeypatch.setattr(Path, "walk", _forbid_directory_scan)

    active_context = resolver.resolve_active_context(store.coordinates)
    pinned_context = resolver.resolve_pinned_context(
        store.coordinates,
        ArtifactPinnedIdentityV2(
            artifact_slot=store.inactive_slot,
            slot_generation=5,
            artifact_asof_date="2026-03-26",
            artifact_manifest_hash="b" * 64,
        ),
    )

    assert active_context.artifact_slot == store.active_slot
    assert pinned_context.artifact_slot == store.inactive_slot


def _forbid_directory_scan(*_args: object, **_kwargs: object) -> None:
    """
    Fail the test immediately if a resolver code path attempts directory scanning.

    Args:
        *_args: Positional arguments ignored by the failure stub.
        **_kwargs: Keyword arguments ignored by the failure stub.
    Returns:
        None.
    Assumptions:
        Explicit-path slot bootstrap must never need scanning helpers.
    Raises:
        AssertionError: Always, to signal forbidden scanning usage.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
      - tests/unit/contexts/backtest/application/services/v2/
        test_yaml_backtest_artifact_loader_v2.py
    """
    raise AssertionError("directory scanning is forbidden in artifact slot resolver v2")
