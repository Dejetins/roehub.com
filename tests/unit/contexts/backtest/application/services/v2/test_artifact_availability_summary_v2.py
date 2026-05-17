from __future__ import annotations

import shutil
from datetime import datetime, timezone

import yaml

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_fs import (
    AtomicArtifactAvailabilitySummaryWriterV2,
)
from trading.contexts.backtest_artifacts.application.services.v2 import (
    BacktestArtifactAvailabilitySummaryGeneratorV2,
)


def test_availability_summary_writes_valid_active_artifact_root(tmp_path) -> None:
    """
    Verify summary generation exposes only active current/manifest artifact availability.

    Args:
        tmp_path: Temporary directory for the synthetic artifact root.
    Returns:
        None.
    Assumptions:
        The synthetic fixture publishes `binance/spot/BTCUSDT` with `15m` coverage.
    Raises:
        AssertionError: If the generated YAML shape or deterministic metadata drifts.
    Side Effects:
        Writes a root-level `availability_summary.yaml` under `tmp_path`.
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    generator = _generator(store=store)

    result = generator.regenerate()
    payload = yaml.safe_load(result.summary_path.read_text(encoding="utf-8"))
    instrument = payload["instruments"]["binance/spot/BTCUSDT"]

    assert result.instrument_count == 1
    assert result.skipped_count == 0
    assert payload["source"] == "artifact_publisher_active_slot_scan"
    assert payload["summary_hash"] == result.summary_hash
    assert instrument["exchange"] == "binance"
    assert instrument["market"] == "spot"
    assert instrument["symbol"] == "BTCUSDT"
    assert instrument["active_slot"] == store.active_slot
    assert instrument["slot_generation"] == 4
    assert instrument["manifest_sha256"] == store.loader.load_current_pointer(
        store.coordinates
    ).manifest_sha256
    assert instrument["backtest_timeframes"] == ["15m"]
    assert instrument["timeframes"]["15m"] == {
        "start_date": "1970-01-01",
        "end_date": "1970-01-01",
        "bars": 2,
        "price_available": True,
        "signals_available": True,
        "mappings_available": True,
        "indicator_ids": ["ma.ema"],
    }
    assert instrument["hit_times"] == {"timeframe": "15m", "available": True}


def test_availability_summary_excludes_symbol_root_with_missing_current(tmp_path) -> None:
    """
    Verify missing `current.yaml` makes the instrument unavailable.

    Args:
        tmp_path: Temporary directory for the synthetic artifact root.
    Returns:
        None.
    Assumptions:
        Slot contents alone are not authoritative without `current.yaml`.
    Raises:
        AssertionError: If missing current pointers leak into the summary.
    Side Effects:
        Deletes one synthetic `current.yaml` and writes summary YAML.
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    store.builder.current_pointer_path(store.coordinates).unlink()

    result = _generator(store=store).regenerate()

    assert result.instrument_count == 0
    assert result.skipped_reasons == {"missing_current": 1}


def test_availability_summary_excludes_missing_active_slot(tmp_path) -> None:
    """
    Verify a `current.yaml` pointing at a missing active slot is excluded.

    Args:
        tmp_path: Temporary directory for the synthetic artifact root.
    Returns:
        None.
    Assumptions:
        The active slot directory must exist before its manifest can be trusted.
    Raises:
        AssertionError: If a missing slot is included.
    Side Effects:
        Removes the active slot directory and writes summary YAML.
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    shutil.rmtree(store.builder.slot_root(store.coordinates, store.active_slot))

    result = _generator(store=store).regenerate()

    assert result.instrument_count == 0
    assert result.skipped_reasons == {"missing_active_slot": 1}


def test_availability_summary_excludes_corrupt_active_manifest(tmp_path) -> None:
    """
    Verify corrupt active `manifest.yaml` excludes the instrument.

    Args:
        tmp_path: Temporary directory for the synthetic artifact root.
    Returns:
        None.
    Assumptions:
        The generator must fail closed per instrument and continue the root scan.
    Raises:
        AssertionError: If invalid active manifests leak into the summary.
    Side Effects:
        Replaces one active manifest with invalid YAML text and writes summary YAML.
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    store.builder.slot_manifest_path(store.coordinates, store.active_slot).write_text(
        "not: [valid",
        encoding="utf-8",
    )

    result = _generator(store=store).regenerate()

    assert result.instrument_count == 0
    assert result.skipped_reasons == {"invalid_active_manifest": 1}


def test_availability_summary_hash_is_stable_for_identical_artifacts(tmp_path) -> None:
    """
    Verify `summary_hash` ignores generation timestamp and is stable for identical artifacts.

    Args:
        tmp_path: Temporary directory for the synthetic artifact root.
    Returns:
        None.
    Assumptions:
        Consumers use `summary_hash` as artifact-state identity, not as a wall-clock marker.
    Raises:
        AssertionError: If identical active artifacts produce different hashes.
    Side Effects:
        Writes summary YAML twice.
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    first = _generator(
        store=store,
        now=datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
    ).regenerate()
    second = _generator(
        store=store,
        now=datetime(2026, 3, 30, 1, 5, tzinfo=timezone.utc),
    ).regenerate()

    assert first.summary_hash == second.summary_hash
    assert first.generated_at_utc != second.generated_at_utc


def _generator(
    *,
    store,
    now: datetime = datetime(2026, 3, 30, 0, 5, tzinfo=timezone.utc),
) -> BacktestArtifactAvailabilitySummaryGeneratorV2:
    return BacktestArtifactAvailabilitySummaryGeneratorV2(
        artifact_root=store.builder.root,
        path_resolver=store.builder,
        artifact_loader=store.loader,
        writer=AtomicArtifactAvailabilitySummaryWriterV2(),
        now_provider=lambda: now,
    )
