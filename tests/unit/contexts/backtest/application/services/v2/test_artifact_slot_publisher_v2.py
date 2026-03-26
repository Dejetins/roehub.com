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
from trading.contexts.backtest.adapters.outbound.config import (
    load_backtest_artifacts_runtime_config,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest.application.services import (
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCoordinatesV2,
    ArtifactSlotPublishErrorV2,
    ArtifactSlotValidationSpecV2,
    BacktestArtifactPrecomputeRunnerV2,
    BacktestArtifactSlotPublisherV2,
)
from trading.shared_kernel.primitives import TimeRange, UtcTimestamp


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


class _NeverCalledPrecomputeRunner:
    """
    Fake precompute runner used to assert publish precheck fails before any rebuild starts.
    """

    def __init__(self) -> None:
        """
        Initialize the fake runner with a deterministic call flag.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Tests only need to know whether `export_canonical_price_1m(...)` was reached.
        Raises:
            None.
        Side Effects:
            Stores `called=False` in memory.
        """
        self.called = False

    def export_canonical_price_1m(
        self,
        request: ArtifactCanonicalPriceExportRequestV2,
    ) -> None:
        """
        Fail immediately because this fake runner must never be invoked in the guarded tests.

        Args:
            request: R3-04 build request that would have been forwarded to the real runner.
        Returns:
            None.
        Assumptions:
            Publisher precheck or stage-spec validation should stop the flow before any build.
        Raises:
            AssertionError: Always, because reaching this method means the guard order regressed.
        Side Effects:
            Sets `called=True` for assertions.
        Docs:
          - docs/architecture/roadmap/base_refactor_plan.md
          - docs/architecture/backtest/backtest-precompute-runner-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        del request
        self.called = True
        raise AssertionError("precompute runner must not be called in this test")


def _write_matching_artifact_runtime_config(tmp_path: Path) -> Path:
    """
    Write strict artifact config YAML whose validation plan matches synthetic store fixtures.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        Path: Written artifact config path.
    Assumptions:
        Synthetic store exposes prices `1m/15m`, mappings `15m`, one `ma.ema` signal, and
        hit-times manifest.
    Raises:
        OSError: If write operation fails.
    Side Effects:
        Creates one temp YAML file.
    """
    config_path = tmp_path / "backtest_artifacts.yaml"
    config_path.write_text(
        """
version: 1
backtest_artifacts:
  artifact_root: artifacts/backtest/v2
  validation_plan:
    price_timeframes: [15m, 1m]
    mapping_timeframes: [15m]
    signal_artifacts:
      - timeframe: 15m
        indicator_id: ma.ema
    require_hit_times_manifest: true
  hit_times_grid:
    tp_levels_pct: [2.0, 1.0]
    sl_levels_pct: [1.0, 2.0]
  slot_policy:
    slots: [slot_b, slot_a]
  publish_schedule:
    full_rebuild_hour_utc: 2
    full_rebuild_minute_utc: 0
  lookback_policy:
    price_tail_bars_1m: 100
    mapping_tail_bars_1m: 100
    signal_tail_bars_1m: 100
    hit_times_tail_bars_1m: 100
  validation_budgets:
    max_price_bars_per_timeframe: 1000
    max_mapping_rows_per_timeframe: 1000
    max_signal_rows_per_artifact: 1000
    max_hit_times_cells: 10000
""".strip(),
        encoding="utf-8",
    )
    return config_path


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
    validation_spec = load_backtest_artifacts_runtime_config(
        _write_matching_artifact_runtime_config(tmp_path)
    ).to_validation_spec()
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
        validation_spec=validation_spec,
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
    validation_spec = load_backtest_artifacts_runtime_config(
        _write_matching_artifact_runtime_config(tmp_path)
    ).to_validation_spec()
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
            validation_spec=validation_spec,
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
    validation_spec = load_backtest_artifacts_runtime_config(
        _write_matching_artifact_runtime_config(tmp_path)
    ).to_validation_spec()
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=store.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=store.builder),
        job_repository=cast(BacktestJobRepository, _FakeJobRepository(blocked_total=0)),
    )

    precheck = publisher.precheck_publish(store.coordinates)

    with pytest.raises(ArtifactSlotPublishErrorV2) as error_info:
        publisher.publish(
            precheck=precheck,
            validation_spec=validation_spec,
            asof_date="2026-03-26",
        )

    error = error_info.value
    assert error.code == "slot_validation_failed"
    assert len(error.diagnostics) > 0
    assert error.diagnostics[0].code == "artifact_file_missing"
    assert error.diagnostics[0].location == "prices[1m].ohlcv"


def test_backtest_artifact_slot_publisher_v2_build_publish_prices_mappings_slot_rejects_full_spec(
    tmp_path: Path,
) -> None:
    """
    Verify the R3-04 stage API rejects validation specs that still require `signals/hit_times`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Prices+mappings publish flow must keep stage scope explicit and must not infer it from
        slot contents.
    Raises:
        AssertionError: If the fake runner is called or the stage API accepts a full spec.
    Side Effects:
        Creates temporary artifact files under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    runner = _NeverCalledPrecomputeRunner()
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=store.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=store.builder),
        job_repository=cast(BacktestJobRepository, _FakeJobRepository(blocked_total=0)),
    )

    with pytest.raises(ValueError, match="signal_artifacts=\\(\\)"):
        publisher.build_publish_prices_mappings_slot(
            request=_prices_mappings_request_v2(store.coordinates),
            precompute_runner=cast(BacktestArtifactPrecomputeRunnerV2, runner),
            validation_spec=store.validation_spec,
        )

    assert runner.called is False


def test_backtest_artifact_slot_publisher_v2_build_publish_prices_mappings_slot_stops_on_pinned_precheck(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify the R3-04 stage flow runs `precheck_publish` before attempting to build inactive slot.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Synthetic store already has an inactive-slot manifest, so the pin guard can detect
        `inactive_slot_pinned`.
    Raises:
        AssertionError: If the fake runner is called or the stable error code changes.
    Side Effects:
        Creates temporary artifact files under `tmp_path`.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-artifact-store-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    runner = _NeverCalledPrecomputeRunner()
    publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=store.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=store.builder),
        job_repository=cast(BacktestJobRepository, _FakeJobRepository(blocked_total=2)),
    )

    with pytest.raises(ArtifactSlotPublishErrorV2) as error_info:
        publisher.build_publish_prices_mappings_slot(
            request=_prices_mappings_request_v2(store.coordinates),
            precompute_runner=cast(BacktestArtifactPrecomputeRunnerV2, runner),
            validation_spec=ArtifactSlotValidationSpecV2(
                price_timeframes=("1m", "15m"),
                mapping_timeframes=("15m",),
                signal_artifacts=(),
                require_hit_times_manifest=False,
            ),
        )

    assert error_info.value.code == "inactive_slot_pinned"
    assert runner.called is False


def _prices_mappings_request_v2(
    coordinates: ArtifactCoordinatesV2,
) -> ArtifactCanonicalPriceExportRequestV2:
    """
    Build one deterministic R3-04 build request used by publisher-only unit tests.

    Args:
        coordinates: Artifact coordinates accepted by `ArtifactCanonicalPriceExportRequestV2`.
    Returns:
        ArtifactCanonicalPriceExportRequestV2: Deterministic request with a stable UTC range.
    Assumptions:
        Publisher-only tests never reach the real runner, so the concrete range contents do not
        matter beyond satisfying strict request validation.
    Raises:
        ValueError: If the supplied coordinates violate strict artifact contracts.
    Side Effects:
        None.
    Docs:
      - docs/architecture/roadmap/base_refactor_plan.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return ArtifactCanonicalPriceExportRequestV2(
        coordinates=coordinates,
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 3, 23, 0, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 3, 26, 0, 0, tzinfo=timezone.utc)),
        ),
        asof_date="2026-03-26",
        generated_at_utc="2026-03-26T03:04:05Z",
    )
