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


def test_resolver_accepts_schema2_without_risk_manifest(tmp_path: Path) -> None:
    import hashlib

    import yaml

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    manifest_path = store.builder.slot_manifest_path(store.coordinates, store.active_slot)
    payload = yaml.safe_load(manifest_path.read_text())
    payload["schema_version"] = 2
    payload.pop("hit_times")
    manifest_path.write_text(yaml.safe_dump(payload))
    pointer_path = store.builder.current_pointer_path(store.coordinates)
    pointer = yaml.safe_load(pointer_path.read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path.write_text(yaml.safe_dump(pointer))
    # The old file may be absent: no-risk source resolution must never open it.
    store.builder.hit_times_paths(store.coordinates, store.active_slot).manifest.unlink()
    metadata = FilesystemBacktestArtifactContextResolver(store.loader).resolve_context(
        coordinates=BacktestCoordinates(exchange="binance", market_type="spot", symbol="BTCUSDT"),
    )
    assert metadata.hit_times_manifest_hash is None
