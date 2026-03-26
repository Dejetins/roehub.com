from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    AtomicArtifactCurrentPointerWriterV2,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest.application.services import (
    ArtifactSlotPublishErrorV2,
    BacktestArtifactSlotPublisherV2,
)


class _FakeJobRepository:
    """
    Deterministic fake repository exposing only the publish-guard pin-count query.
    """

    def __init__(self, *, blocked_total: int = 0) -> None:
        """
        Initialize fake repository with fixed inactive-slot blocking count.

        Args:
            blocked_total: Count returned for `count_active_for_artifact_manifest(...)`.
        Returns:
            None.
        Assumptions:
            Publisher unit tests exercise only the pin-count query.
        Raises:
            None.
        Side Effects:
            Stores last query payload for assertions.
        """
        self.blocked_total = blocked_total
        self.last_call: dict[str, object] | None = None

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return configured blocking count and record the publish-guard query payload.

        Args:
            market_id: Canonical market id for the published symbol root.
            symbol: Instrument symbol.
            artifact_slot: Candidate inactive slot literal.
            artifact_manifest_hash: SHA-256 of inactive slot `manifest.yaml`.
        Returns:
            int: Configured blocking active jobs count.
        Assumptions:
            Other repository methods are not needed in these publisher unit tests.
        Raises:
            None.
        Side Effects:
            Records last query payload in memory.
        """
        self.last_call = {
            "market_id": market_id,
            "symbol": symbol,
            "artifact_slot": artifact_slot,
            "artifact_manifest_hash": artifact_manifest_hash,
        }
        return self.blocked_total


def test_backtest_artifact_slot_publisher_v2_switches_current_yaml_after_strict_validation(
    tmp_path: Path,
) -> None:
    """
    Verify publisher validates strict manifests and atomically switches `current.yaml`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Inactive slot was rebuilt with strict manifests and no active jobs pin it.
    Raises:
        AssertionError: If publish result or written pointer identity is incorrect.
    Side Effects:
        Creates and replaces artifact files under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    repository = _FakeJobRepository(blocked_total=0)
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=store.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=store.builder),
        job_repository=cast(BacktestJobRepository, repository),
        now_provider=lambda: datetime(2026, 3, 26, 3, 4, 5, tzinfo=timezone.utc),
    )

    precheck = publisher.precheck_publish(store.coordinates)
    result = publisher.publish(
        precheck=precheck,
        validation_spec=store.validation_spec,
        asof_date="2026-03-26",
    )

    reloaded_pointer = store.loader.load_current_pointer(store.coordinates)

    assert precheck.ready is True
    assert precheck.inactive_slot == store.inactive_slot
    assert repository.last_call is not None
    assert repository.last_call["market_id"] == 1
    assert repository.last_call["symbol"] == "BTCUSDT"
    assert result.previous_pointer.active_slot == store.active_slot
    assert result.published_pointer.active_slot == store.inactive_slot
    assert result.published_pointer.slot_generation == 5
    assert result.published_pointer.asof_date == "2026-03-26"
    assert result.published_pointer.published_at_utc == "2026-03-26T03:04:05Z"
    assert reloaded_pointer == result.published_pointer
    assert result.validation.slot_manifest is not None
    assert result.validation.slot_manifest.path == store.builder.slot_manifest_path(
        store.coordinates,
        store.inactive_slot,
    )
    assert len(result.validation.signal_manifests) == 1
    assert result.validation.hit_times_manifest is not None
    assert result.validation.diagnostics == ()


def test_backtest_artifact_slot_publisher_v2_blocks_publish_when_inactive_slot_is_pinned(
    tmp_path: Path,
) -> None:
    """
    Verify precheck/publish fail fast when the inactive slot is pinned by active jobs.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Blocking count comes from persisted queued/running jobs with pinned slot metadata.
    Raises:
        AssertionError: If blocked precheck does not yield stable diagnostics.
    Side Effects:
        Creates temporary artifact files under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=store.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=store.builder),
        job_repository=cast(BacktestJobRepository, _FakeJobRepository(blocked_total=2)),
    )

    precheck = publisher.precheck_publish(store.coordinates)

    assert precheck.ready is False
    assert precheck.failure_code == "inactive_slot_pinned"
    with pytest.raises(ArtifactSlotPublishErrorV2, match="slot_b"):
        publisher.publish(
            precheck=precheck,
            validation_spec=store.validation_spec,
            asof_date="2026-03-26",
        )


def test_backtest_artifact_slot_publisher_v2_rejects_missing_strict_artifact_file(
    tmp_path: Path,
) -> None:
    """
    Verify strict publish validation fails when one referenced artifact file is missing.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Publish validation must reject missing files before `current.yaml` switch.
    Raises:
        AssertionError: If missing artifact diagnostics are not stable.
    Side Effects:
        Creates temporary artifact files under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        omit_inactive_files=("prices/1m/ohlcv.f32.npy",),
    )
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=store.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=store.builder),
        job_repository=cast(BacktestJobRepository, _FakeJobRepository(blocked_total=0)),
    )

    precheck = publisher.precheck_publish(store.coordinates)

    with pytest.raises(ArtifactSlotPublishErrorV2) as error_info:
        publisher.publish(
            precheck=precheck,
            validation_spec=store.validation_spec,
            asof_date="2026-03-26",
        )

    error = error_info.value
    assert error.code == "slot_validation_failed"
    assert len(error.diagnostics) > 0
    assert error.diagnostics[0].code == "artifact_file_missing"
    assert error.diagnostics[0].location == "prices[1m].ohlcv"
