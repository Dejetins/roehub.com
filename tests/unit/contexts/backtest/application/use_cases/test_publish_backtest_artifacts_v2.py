from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence, cast

import pytest

from tests.unit.contexts.backtest.application.services.v2 import (
    test_artifact_precompute_runner_v2 as precompute_runner_testkit_v2,
)
from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    ArtifactPrecomputeFixtureV2,
    build_artifact_precompute_fixture_v2,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    AtomicArtifactCurrentPointerWriterV2,
)
from trading.contexts.backtest.application.ports import BacktestJobRepository
from trading.contexts.backtest_artifacts.application.services.v2.artifact_precompute_runner import (
    BacktestArtifactPrecomputeRunnerV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_slot_publisher import (
    BacktestArtifactSlotPublisherV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.signal_rules_engine_v2 import (
    BacktestSignalRulesEngineV2,
)
from trading.contexts.backtest_artifacts.application.use_cases import (
    PublishBacktestArtifactsV2Request,
    PublishBacktestArtifactsV2UseCase,
)
from trading.contexts.market_data.application.ports.stores import DailyTsOpenCount
from trading.shared_kernel.primitives import InstrumentId, TimeRange, UtcTimestamp

_PUBLISH_NOW_UTC_V2 = datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc)


@dataclass(slots=True)
class _FixedCanonicalCandleIndexReader:
    """
    Deterministic bounds reader exposing one explicit canonical 1m coverage envelope.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_index_reader.py
    """

    first_ts_open: UtcTimestamp | None
    last_ts_open: UtcTimestamp | None
    bounds_calls: list[tuple[InstrumentId, UtcTimestamp]] = field(default_factory=list)

    def bounds(self, instrument_id: InstrumentId) -> tuple[UtcTimestamp, UtcTimestamp] | None:
        """
        Reject unsupported protocol methods in these focused use-case tests.

        Args:
            instrument_id: Instrument identity ignored by this helper.
        Returns:
            tuple[UtcTimestamp, UtcTimestamp] | None: Never returns successfully.
        Assumptions:
            Publish orchestration exercises only `bounds_1m(...)`.
        Raises:
            AssertionError: Always, because tests should not call the broad bounds API here.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - canonical_candle_index_reader.py
        """
        del instrument_id
        raise AssertionError("bounds() is not used by PublishBacktestArtifactsV2UseCase tests")

    def bounds_1m(
        self,
        *,
        instrument_id: InstrumentId,
        before: UtcTimestamp,
    ) -> tuple[UtcTimestamp | None, UtcTimestamp | None]:
        """
        Return fixed canonical 1m bounds while recording the lookup arguments.

        Args:
            instrument_id: Explicit canonical instrument identity requested by the use-case.
            before: Exclusive upper bound passed by the orchestration clock.
        Returns:
            tuple[UtcTimestamp | None, UtcTimestamp | None]: Configured first/last open times.
        Assumptions:
            Tests control the full canonical history explicitly through deterministic row fixtures.
        Raises:
            None.
        Side Effects:
            Records one bounds lookup in memory for later assertions.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
        """
        self.bounds_calls.append((instrument_id, before))
        return self.first_ts_open, self.last_ts_open

    def max_ts_open_lt(
        self,
        *,
        instrument_id: InstrumentId,
        before: UtcTimestamp,
        after: UtcTimestamp | None = None,
    ) -> UtcTimestamp | None:
        """
        Reject unrelated protocol calls in these publish-use-case unit tests.

        Args:
            instrument_id: Instrument identity ignored by this helper.
            before: Exclusive upper bound ignored by this helper.
        Returns:
            UtcTimestamp | None: Never returns successfully.
        Assumptions:
            Publish orchestration does not need incremental catch-up helpers here.
        Raises:
            AssertionError: Always, because this helper should stay unused in this test module.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - canonical_candle_index_reader.py
        """
        del instrument_id, before, after
        raise AssertionError(
            "max_ts_open_lt() is not used by PublishBacktestArtifactsV2UseCase tests"
        )

    def daily_counts(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Sequence[DailyTsOpenCount]:
        """
        Reject unrelated protocol calls in these publish-use-case unit tests.

        Args:
            instrument_id: Instrument identity ignored by this helper.
            time_range: Time range ignored by this helper.
        Returns:
            Sequence[DailyTsOpenCount]: Never returns successfully.
        Assumptions:
            Whole-slot publish orchestration never calls the day-level aggregate API here.
        Raises:
            AssertionError: Always, because this helper should stay unused in this test module.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - canonical_candle_index_reader.py
        """
        del instrument_id, time_range
        raise AssertionError(
            "daily_counts() is not used by PublishBacktestArtifactsV2UseCase tests"
        )

    def distinct_ts_opens(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> Sequence[UtcTimestamp]:
        """
        Reject unrelated protocol calls in these publish-use-case unit tests.

        Args:
            instrument_id: Instrument identity ignored by this helper.
            time_range: Time range ignored by this helper.
        Returns:
            Sequence[UtcTimestamp]: Never returns successfully.
        Assumptions:
            Publish orchestration for one explicit symbol root uses only `bounds_1m(...)`.
        Raises:
            AssertionError: Always, because this helper should stay unused in this test module.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
        Related:
          - canonical_candle_index_reader.py
        """
        del instrument_id, time_range
        raise AssertionError(
            "distinct_ts_opens() is not used by PublishBacktestArtifactsV2UseCase tests"
        )


class _ZeroBlockingJobRepository:
    """
    Fake publish-guard repository that never reports active pins for inactive manifests.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - tests/unit/contexts/backtest/application/use_cases/test_publish_backtest_artifacts_v2.py
    """

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return zero blocking runs for the explicit inactive-slot publish guard query.

        Args:
            market_id: Canonical market id for the artifact target.
            symbol: Instrument symbol under publish.
            artifact_slot: Candidate inactive slot literal.
            artifact_manifest_hash: SHA-256 of the inactive slot root manifest.
        Returns:
            int: Always `0`.
        Assumptions:
            These tests focus on bootstrap and pointer semantics rather than queued/running jobs.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        del market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


@dataclass(frozen=True, slots=True)
class _PublishUseCaseFixtureV2:
    """
    Deterministic bundle of shared publish use-case dependencies for one temp artifact root.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - tests/unit/contexts/backtest/application/use_cases/test_publish_backtest_artifacts_v2.py
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """

    fixture: ArtifactPrecomputeFixtureV2
    index_reader: _FixedCanonicalCandleIndexReader
    use_case: PublishBacktestArtifactsV2UseCase


def test_publish_backtest_artifacts_v2_bootstraps_missing_current_pointer(
    tmp_path: Path,
) -> None:
    """
    Verify bootstrap builds `slot_a`, validates the whole slot, and writes initial `current.yaml`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Symbol root starts without `current.yaml` and without any pre-existing slot manifests.
    Raises:
        AssertionError: If bootstrap diagnostics or written pointer identity are incorrect.
    Side Effects:
        Creates a strict bootstrap artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """
    publish_fixture = _build_publish_use_case_fixture_v2(tmp_path=tmp_path)
    current_pointer_path = publish_fixture.fixture.builder.current_pointer_path(
        publish_fixture.fixture.coordinates
    )
    current_pointer_path.unlink()

    result = publish_fixture.use_case.run(
        PublishBacktestArtifactsV2Request(coordinates=publish_fixture.fixture.coordinates)
    )
    current_pointer = publish_fixture.fixture.loader.load_current_pointer(
        publish_fixture.fixture.coordinates
    )
    signal_manifest = publish_fixture.fixture.loader.load_signal_manifest(
        publish_fixture.fixture.coordinates,
        "slot_a",
        "15m",
        "ma.ema",
    )

    assert result.status == "succeeded"
    assert result.publish_mode == "bootstrap"
    assert result.previous_active_slot is None
    assert result.previous_slot_generation is None
    assert result.published_active_slot == "slot_a"
    assert result.published_slot_generation == 1
    assert result.validation.hit_times_manifest_present is True
    assert result.validation.signal_manifest_count == 1
    assert current_pointer == publish_fixture.fixture.loader.load_current_pointer(
        publish_fixture.fixture.coordinates
    )
    assert current_pointer.active_slot == "slot_a"
    assert current_pointer.slot_generation == 1
    assert current_pointer.manifest_sha256 == result.published_manifest_sha256
    assert signal_manifest.signal_features is not None
    assert current_pointer_path.is_file()
    assert publish_fixture.fixture.builder.slot_manifest_path(
        publish_fixture.fixture.coordinates,
        "slot_a",
    ).is_file()
    assert publish_fixture.fixture.builder.slot_manifest_path(
        publish_fixture.fixture.coordinates,
        "slot_b",
    ).parent.is_dir()
    assert len(publish_fixture.index_reader.bounds_calls) == 1
    assert publish_fixture.index_reader.bounds_calls[0][1] == UtcTimestamp(_PUBLISH_NOW_UTC_V2)


def test_publish_backtest_artifacts_v2_repeated_publish_switches_pointer_and_recreates_pruned_slot(  # noqa: E501
    tmp_path: Path,
) -> None:
    """
    Verify repeated publish removes previous slot and recreates it on the next publish cycle.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        The second publish reuses the same shared orchestration entrypoint and strict slot rules.
    Raises:
        AssertionError: If pointer switching or safe single-slot retention semantics regress.
    Side Effects:
        Creates and publishes two deterministic slots under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """
    publish_fixture = _build_publish_use_case_fixture_v2(tmp_path=tmp_path)
    current_pointer_path = publish_fixture.fixture.builder.current_pointer_path(
        publish_fixture.fixture.coordinates
    )
    current_pointer_path.unlink()
    request = PublishBacktestArtifactsV2Request(coordinates=publish_fixture.fixture.coordinates)

    first_result = publish_fixture.use_case.run(request)
    slot_a_manifest_path = publish_fixture.fixture.builder.slot_manifest_path(
        publish_fixture.fixture.coordinates,
        "slot_a",
    )
    assert slot_a_manifest_path.is_file()

    second_result = publish_fixture.use_case.run(request)
    current_pointer = publish_fixture.fixture.loader.load_current_pointer(
        publish_fixture.fixture.coordinates
    )
    slot_b_signal_manifest = publish_fixture.fixture.loader.load_signal_manifest(
        publish_fixture.fixture.coordinates,
        "slot_b",
        "15m",
        "ma.ema",
    )

    assert first_result.publish_mode == "bootstrap"
    assert second_result.publish_mode == "incremental"
    assert second_result.previous_active_slot == "slot_a"
    assert second_result.previous_slot_generation == 1
    assert second_result.previous_manifest_sha256 == first_result.published_manifest_sha256
    assert second_result.published_active_slot == "slot_b"
    assert second_result.published_slot_generation == 2
    assert second_result.reused_prefix_bars == (
        precompute_runner_testkit_v2._FULL_BUILD_MINUTES_V2
        - publish_fixture.fixture.runtime_settings.price_tail_bars_1m
    )
    assert (
        second_result.rewritten_tail_bars
        == publish_fixture.fixture.runtime_settings.price_tail_bars_1m
    )
    assert second_result.stage_rebuild_stats.signals.reused_prefix_bars > 0
    assert second_result.validation.manifest_sha256 == second_result.published_manifest_sha256
    assert current_pointer.active_slot == "slot_b"
    assert current_pointer.slot_generation == 2
    assert slot_a_manifest_path.exists() is False
    assert slot_a_manifest_path.parent.exists() is False
    assert slot_b_signal_manifest.signal_features is not None
    assert publish_fixture.fixture.builder.slot_manifest_path(
        publish_fixture.fixture.coordinates,
        "slot_b",
    ).is_file()
    third_result = publish_fixture.use_case.run(request)

    assert third_result.publish_mode == "incremental"
    assert third_result.previous_active_slot == "slot_b"
    assert third_result.previous_slot_generation == 2
    assert third_result.published_active_slot == "slot_a"
    assert third_result.published_slot_generation == 3
    assert third_result.reused_prefix_bars == second_result.reused_prefix_bars
    assert third_result.rewritten_tail_bars == second_result.rewritten_tail_bars
    assert slot_a_manifest_path.is_file()
    assert len(publish_fixture.index_reader.bounds_calls) == 3


def test_publish_backtest_artifacts_v2_fails_fast_on_invalid_current_pointer(
    tmp_path: Path,
) -> None:
    """
    Verify invalid `current.yaml` content aborts orchestration before any build starts.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Pointer parsing remains the strict source of truth for active/inactive slot identity.
    Raises:
        AssertionError: If invalid pointer content is accepted or emits the wrong failure class.
    Side Effects:
        Rewrites `current.yaml` under `tmp_path` with invalid slot metadata.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """
    publish_fixture = _build_publish_use_case_fixture_v2(tmp_path=tmp_path)
    current_pointer_path = publish_fixture.fixture.builder.current_pointer_path(
        publish_fixture.fixture.coordinates
    )
    current_pointer_path.write_text(
        "\n".join(
            (
                "schema_version: 1",
                "active_slot: slot_c",
                "slot_generation: 4",
                'asof_date: "2026-03-25"',
                f'manifest_sha256: "{"0" * 64}"',
                'published_at_utc: "2026-03-25T02:00:00Z"',
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact slot"):
        publish_fixture.use_case.run(
            PublishBacktestArtifactsV2Request(coordinates=publish_fixture.fixture.coordinates)
        )

    assert len(publish_fixture.index_reader.bounds_calls) == 1


def _build_publish_use_case_fixture_v2(*, tmp_path: Path) -> _PublishUseCaseFixtureV2:
    """
    Build the shared publish use-case with real runner/publisher services and temp storage.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        _PublishUseCaseFixtureV2: Runtime config fixture plus ready-to-run shared use-case.
    Assumptions:
        Tests use one explicit `binance/spot/BTCUSDT` symbol root with strict signal validation.
    Raises:
        ValueError: If shared runtime wiring becomes inconsistent.
    Side Effects:
        Creates config, pointer file, and later-stage service dependencies under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """
    fixture = build_artifact_precompute_fixture_v2(
        tmp_path=tmp_path,
        validation_signal_artifacts=(("15m", "ma.ema"),),
        precompute_signal_artifacts=(("15m", "ma.ema"),),
        require_hit_times_manifest=True,
    )
    rows = precompute_runner_testkit_v2._build_canonical_rows_v2(
        bar_indexes=tuple(range(precompute_runner_testkit_v2._FULL_BUILD_MINUTES_V2))
    )
    index_reader = _FixedCanonicalCandleIndexReader(
        first_ts_open=UtcTimestamp(rows[0].candle.ts_open.value - timedelta(hours=3)),
        last_ts_open=rows[-1].candle.ts_open,
    )
    grid_builder = precompute_runner_testkit_v2._signal_grid_builder_v2()
    defaults_provider = precompute_runner_testkit_v2._build_signal_test_defaults_provider_v2()
    precompute_runner = BacktestArtifactPrecomputeRunnerV2(
        runtime_settings=fixture.runtime_settings,
        artifact_loader=fixture.loader,
        canonical_candle_reader=precompute_runner_testkit_v2._FakeCanonicalCandleReader(rows=rows),
        defaults_provider=defaults_provider,
        signal_rules_engine=BacktestSignalRulesEngineV2(defaults_provider=defaults_provider),
        indicator_compute=precompute_runner_testkit_v2._DeterministicSignalCompute(
            grid_builder=grid_builder
        ),
        indicator_grid_builder=grid_builder,
    )
    slot_publisher = BacktestArtifactSlotPublisherV2(
        artifact_loader=fixture.loader,
        current_pointer_writer=AtomicArtifactCurrentPointerWriterV2(path_resolver=fixture.builder),
        job_repository=cast(BacktestJobRepository, _ZeroBlockingJobRepository()),
        now_provider=lambda: _PUBLISH_NOW_UTC_V2,
    )
    use_case = PublishBacktestArtifactsV2UseCase(
        canonical_candle_index_reader=index_reader,
        precompute_runner=precompute_runner,
        slot_publisher=slot_publisher,
        validation_spec=fixture.runtime_config.to_validation_spec(),
        now_provider=lambda: _PUBLISH_NOW_UTC_V2,
    )
    return _PublishUseCaseFixtureV2(
        fixture=fixture,
        index_reader=index_reader,
        use_case=use_case,
    )
