from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactContextResolver,
)
from trading.contexts.backtest.application.dto import BacktestCoordinates
from trading.contexts.backtest.application.ports import BacktestArtifactContextUnavailable


def test_filesystem_artifact_context_resolver_reads_current_pointer_and_manifests(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    resolver = FilesystemBacktestArtifactContextResolver(artifact_loader=store.loader)

    metadata = resolver.resolve_context(
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
    )

    assert metadata.artifact_slot == store.active_slot
    assert metadata.artifact_slot_generation == 4
    assert metadata.artifact_asof_date == "2026-03-25"
    assert len(metadata.artifact_manifest_hash) == 64
    assert len(metadata.hit_times_manifest_hash or "") == 64
    assert metadata.published_at_utc == "2026-03-25T02:00:00Z"


def test_filesystem_artifact_context_resolver_reports_artifacts_unavailable(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    resolver = FilesystemBacktestArtifactContextResolver(artifact_loader=store.loader)

    with pytest.raises(BacktestArtifactContextUnavailable):
        resolver.resolve_context(
            coordinates=BacktestCoordinates(
                exchange="binance",
                market_type="spot",
                symbol="ETHUSDT",
            )
        )
