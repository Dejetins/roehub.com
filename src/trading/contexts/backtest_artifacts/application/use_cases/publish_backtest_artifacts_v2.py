"""Shared orchestration for bootstrap and steady-state backtest artifact publish v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping

from trading.contexts.backtest_artifacts.application.services.v2 import (
    ArtifactCanonicalPriceExportRequestV2,
    ArtifactCanonicalPriceExportResultV2,
    ArtifactCoordinatesV2,
    ArtifactPublishPrecheckV2,
    ArtifactPublishResultV2,
    ArtifactSlotPublishErrorV2,
    ArtifactSlotValidationSpecV2,
    ArtifactStageRebuildStatsCollectionV2,
    ArtifactTailRebuildBarsV2,
    BacktestArtifactPrecomputeRunnerV2,
    BacktestArtifactSlotPublisherV2,
    artifact_market_id_from_coordinates_v2,
    validate_current_pointer_published_at_utc_v2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    validate_non_negative_manifest_int_v2,
    validate_positive_manifest_int_v2,
)
from trading.contexts.market_data.application.ports.stores import CanonicalCandleIndexReader
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

type PublishBacktestArtifactsModeV2 = Literal["bootstrap", "incremental", "full_rebuild"]
NowProviderV2 = Callable[[], datetime]


def _default_now_provider_v2() -> datetime:
    """
    Return the default wall-clock value for publish use-case orchestration.

    Args:
        None.
    Returns:
        datetime: Timezone-aware UTC datetime.
    Assumptions:
        Manual CLI and future scheduler diagnostics share the same UTC timestamp contract.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
    """
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class PublishBacktestArtifactsV2Request:
    """
    Deterministic request DTO for publishing one explicit backtest artifact symbol root.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
      - apps/cli/commands/backtest_artifact_publish.py
    """

    coordinates: ArtifactCoordinatesV2
    full_rebuild: bool = False

    def __post_init__(self) -> None:
        """
        Validate the minimal explicit manual/scheduled publish request contract.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            One use-case invocation targets exactly one `(exchange, market_type, symbol)`.
        Raises:
            ValueError: If coordinates are missing.
        Side Effects:
            None.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """
        if self.coordinates is None:  # type: ignore[truthy-bool]
            raise ValueError("PublishBacktestArtifactsV2Request.coordinates is required")


@dataclass(frozen=True, slots=True)
class PublishBacktestArtifactsV2ValidationSummary:
    """
    Machine-readable validation summary returned after one successful whole-slot validation.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    slot_manifest_path: Path | None
    manifest_sha256: str
    price_timeframes: tuple[str, ...]
    mapping_timeframes: tuple[str, ...]
    signal_artifacts: tuple[tuple[str, str], ...]
    signal_manifest_count: int
    hit_times_manifest_present: bool
    funding_coverage_status: str | None
    funding_manifest_hash: str | None
    diagnostics_count: int

    def as_dict(self) -> Mapping[str, object]:
        """
        Serialize validation summary into a stable JSON-friendly mapping.

        Args:
            None.
        Returns:
            Mapping[str, object]: Deterministic mapping preserving field order.
        Assumptions:
            CLI output and future scheduler metrics adapters need scalar-friendly payloads.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
          - docs/architecture/backtest/README.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """
        return {
            "slot_manifest_path": (
                None if self.slot_manifest_path is None else str(self.slot_manifest_path)
            ),
            "manifest_sha256": self.manifest_sha256,
            "price_timeframes": list(self.price_timeframes),
            "mapping_timeframes": list(self.mapping_timeframes),
            "signal_artifacts": [
                {"timeframe": timeframe, "indicator_id": indicator_id}
                for timeframe, indicator_id in self.signal_artifacts
            ],
            "signal_manifest_count": self.signal_manifest_count,
            "hit_times_manifest_present": self.hit_times_manifest_present,
            "funding_coverage_status": self.funding_coverage_status,
            "funding_manifest_hash": self.funding_manifest_hash,
            "diagnostics_count": self.diagnostics_count,
        }


@dataclass(frozen=True, slots=True)
class PublishBacktestArtifactsV2Result:
    """
    Deterministic publish diagnostics for one completed shared orchestration run.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/publish_backtest_artifacts_v2.py
      - apps/cli/commands/backtest_artifact_publish.py
    """

    status: str
    publish_mode: PublishBacktestArtifactsModeV2
    coordinates: ArtifactCoordinatesV2
    previous_active_slot: str | None
    previous_slot_generation: int | None
    previous_manifest_sha256: str | None
    published_active_slot: str
    published_slot_generation: int
    published_manifest_sha256: str
    asof_date: str
    published_at_utc: str
    requested_start_utc: str
    requested_end_utc: str
    source_start_utc: str
    source_end_utc: str
    source_candle_count: int
    reused_prefix_bars: int
    rewritten_tail_bars: int
    blocking_active_run_count: int
    validation: PublishBacktestArtifactsV2ValidationSummary
    stage_rebuild_stats: ArtifactStageRebuildStatsCollectionV2 = field(
        default_factory=ArtifactStageRebuildStatsCollectionV2
    )
    tail_rebuild_bars: ArtifactTailRebuildBarsV2 = field(default_factory=ArtifactTailRebuildBarsV2)

    def __post_init__(self) -> None:
        """
        Validate publish diagnostics counters exposed to CLI and scheduler integrations.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Legacy top-level `reused_prefix_bars` / `rewritten_tail_bars` stay aligned with the
            canonical `prices` stage while stage-local stats expose the full pipeline breakdown.
        Raises:
            ValueError: If counts are negative, missing, or inconsistent with stage-local stats.
        Side Effects:
            Normalizes numeric counters through strict integer validation.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
          - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
        """
        object.__setattr__(
            self,
            "source_candle_count",
            validate_positive_manifest_int_v2(self.source_candle_count),
        )
        object.__setattr__(
            self,
            "reused_prefix_bars",
            validate_non_negative_manifest_int_v2(self.reused_prefix_bars),
        )
        object.__setattr__(
            self,
            "rewritten_tail_bars",
            validate_positive_manifest_int_v2(self.rewritten_tail_bars),
        )
        object.__setattr__(
            self,
            "blocking_active_run_count",
            validate_non_negative_manifest_int_v2(self.blocking_active_run_count),
        )
        if self.stage_rebuild_stats is None:  # type: ignore[truthy-bool]
            raise ValueError("PublishBacktestArtifactsV2Result.stage_rebuild_stats is required")
        if self.tail_rebuild_bars is None:  # type: ignore[truthy-bool]
            raise ValueError("PublishBacktestArtifactsV2Result.tail_rebuild_bars is required")
        if self.stage_rebuild_stats.prices.reused_prefix_bars != self.reused_prefix_bars:
            raise ValueError(
                "PublishBacktestArtifactsV2Result.reused_prefix_bars must match "
                "stage_rebuild_stats.prices.reused_prefix_bars"
            )
        if self.stage_rebuild_stats.prices.rewritten_tail_bars != self.rewritten_tail_bars:
            raise ValueError(
                "PublishBacktestArtifactsV2Result.rewritten_tail_bars must match "
                "stage_rebuild_stats.prices.rewritten_tail_bars"
            )
        if self.stage_rebuild_stats.tail_rebuild_bars() != self.tail_rebuild_bars:
            raise ValueError(
                "PublishBacktestArtifactsV2Result.tail_rebuild_bars must match "
                "stage_rebuild_stats rewritten tail totals"
            )

    def as_dict(self) -> Mapping[str, object]:
        """
        Serialize the publish result into a stable JSON-friendly mapping.

        Args:
            None.
        Returns:
            Mapping[str, object]: Deterministic mapping with explicit scalar/list payloads.
        Assumptions:
            CLI output should stay concise while remaining scheduler/metrics friendly.
        Raises:
            None.
        Side Effects:
            None.
        Docs:
          - docs/runbooks/backtest-artifacts-rebuild.md
          - docs/architecture/backtest/README.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
        """
        return {
            "status": self.status,
            "publish_mode": self.publish_mode,
            "coordinates": {
                "exchange": self.coordinates.exchange,
                "market_type": self.coordinates.market_type,
                "symbol": self.coordinates.symbol,
            },
            "previous_active_slot": self.previous_active_slot,
            "previous_slot_generation": self.previous_slot_generation,
            "previous_manifest_sha256": self.previous_manifest_sha256,
            "published_active_slot": self.published_active_slot,
            "published_slot_generation": self.published_slot_generation,
            "published_manifest_sha256": self.published_manifest_sha256,
            "asof_date": self.asof_date,
            "published_at_utc": self.published_at_utc,
            "requested_start_utc": self.requested_start_utc,
            "requested_end_utc": self.requested_end_utc,
            "source_start_utc": self.source_start_utc,
            "source_end_utc": self.source_end_utc,
            "source_candle_count": self.source_candle_count,
            "reused_prefix_bars": self.reused_prefix_bars,
            "rewritten_tail_bars": self.rewritten_tail_bars,
            "stage_rebuild_stats": self.stage_rebuild_stats.as_dict(),
            "tail_rebuild_bars": self.tail_rebuild_bars.as_dict(),
            "blocking_active_run_count": self.blocking_active_run_count,
            "validation": self.validation.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PublishBacktestArtifactsV2UseCase:
    """
    Shared publish orchestration reusable by manual CLI and a later scheduler service.

    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    canonical_candle_index_reader: CanonicalCandleIndexReader
    precompute_runner: BacktestArtifactPrecomputeRunnerV2
    slot_publisher: BacktestArtifactSlotPublisherV2
    validation_spec: ArtifactSlotValidationSpecV2
    now_provider: NowProviderV2 = _default_now_provider_v2

    def run(
        self,
        request: PublishBacktestArtifactsV2Request,
    ) -> PublishBacktestArtifactsV2Result:
        """
        Execute shared bootstrap/incremental/full-rebuild publish orchestration for one target.

        Args:
            request: Explicit publish request for one symbol root.
        Returns:
            PublishBacktestArtifactsV2Result: Deterministic publish diagnostics DTO.
        Assumptions:
            Canonical 1m bounds in market data are the source-of-truth for requested build range,
            while slot validation and pointer switching remain delegated to shared services.
        Raises:
            ArtifactSlotPublishErrorV2: If pin guard or whole-slot validation blocks publish.
            ValueError: If canonical bounds, bootstrap invariants, or strict pointer contracts are
                invalid.
            FileNotFoundError: If a strict artifact path required by shared services is missing.
            OSError: If build writes or atomic pointer switch fail.
        Side Effects:
            Reads ClickHouse bounds, writes inactive-slot artifacts, validates the slot, and
            atomically switches `current.yaml` on success.
        Docs:
          - docs/architecture/backtest/README.md
          - docs/runbooks/backtest-artifacts-rebuild.md
        Related:
          - apps/cli/commands/backtest_artifact_publish.py
          - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
        """
        if self.canonical_candle_index_reader is None:  # type: ignore[truthy-bool]
            raise ValueError(
                "PublishBacktestArtifactsV2UseCase.canonical_candle_index_reader is required"
            )
        if self.precompute_runner is None:  # type: ignore[truthy-bool]
            raise ValueError("PublishBacktestArtifactsV2UseCase.precompute_runner is required")
        if self.slot_publisher is None:  # type: ignore[truthy-bool]
            raise ValueError("PublishBacktestArtifactsV2UseCase.slot_publisher is required")
        if self.validation_spec is None:  # type: ignore[truthy-bool]
            raise ValueError("PublishBacktestArtifactsV2UseCase.validation_spec is required")

        started_at_utc = _validated_now_utc_v2(self.now_provider())
        requested_time_range = _resolve_requested_time_range_v2(
            canonical_candle_index_reader=self.canonical_candle_index_reader,
            coordinates=request.coordinates,
            before=UtcTimestamp(started_at_utc),
        )
        precheck = self.slot_publisher.precheck_publish(request.coordinates)
        _ensure_precheck_ready_v2(precheck=precheck)
        _ensure_bootstrap_slot_roots_v2(
            slot_publisher=self.slot_publisher,
            coordinates=request.coordinates,
            bootstrap=precheck.bootstrap,
        )
        publish_mode = _resolve_publish_mode_v2(
            request=request,
            bootstrap=precheck.bootstrap,
        )
        build_time_range = _resolve_build_time_range_v2(
            requested_time_range=requested_time_range,
            publish_mode=publish_mode,
            precheck=precheck,
            slot_publisher=self.slot_publisher,
        )
        build_request = ArtifactCanonicalPriceExportRequestV2(
            coordinates=request.coordinates,
            time_range=build_time_range,
            asof_date=(
                requested_time_range.end.value.astimezone(timezone.utc) - timedelta(minutes=1)
            ).date().isoformat(),
            generated_at_utc=_utc_timestamp_literal_v2(started_at_utc),
            target_slot=precheck.inactive_slot,
            target_slot_generation=precheck.target_slot_generation,
            reuse_source_slot=(
                None
                if precheck.current_pointer is None or publish_mode != "incremental"
                else precheck.current_pointer.active_slot
            ),
            force_full_rebuild=publish_mode != "incremental",
        )
        build_result = self.precompute_runner.export_canonical_price_1m(build_request)
        publish_result = self.slot_publisher.publish(
            precheck=precheck,
            validation_spec=self.validation_spec,
            asof_date=build_request.asof_date,
        )
        _ensure_build_publish_alignment_v2(
            build_request=build_request,
            build_result=build_result,
            publish_result=publish_result,
        )
        validation_summary = _build_validation_summary_v2(publish_result=publish_result)
        previous_pointer = publish_result.previous_pointer
        return PublishBacktestArtifactsV2Result(
            status="succeeded",
            publish_mode=publish_mode,
            coordinates=request.coordinates,
            previous_active_slot=None if previous_pointer is None else previous_pointer.active_slot,
            previous_slot_generation=(
                None if previous_pointer is None else previous_pointer.slot_generation
            ),
            previous_manifest_sha256=(
                None if previous_pointer is None else previous_pointer.manifest_sha256
            ),
            published_active_slot=publish_result.published_pointer.active_slot,
            published_slot_generation=publish_result.published_pointer.slot_generation,
            published_manifest_sha256=publish_result.published_pointer.manifest_sha256,
            asof_date=publish_result.published_pointer.asof_date,
            published_at_utc=publish_result.published_pointer.published_at_utc,
            requested_start_utc=_utc_timestamp_literal_v2(build_time_range.start.value),
            requested_end_utc=_utc_timestamp_literal_v2(build_time_range.end.value),
            source_start_utc=_utc_timestamp_literal_v2(build_result.source_time_range.start.value),
            source_end_utc=_utc_timestamp_literal_v2(build_result.source_time_range.end.value),
            source_candle_count=build_result.source_candle_count,
            reused_prefix_bars=build_result.reused_prefix_bars,
            rewritten_tail_bars=build_result.rewritten_tail_bars,
            stage_rebuild_stats=build_result.stage_rebuild_stats,
            tail_rebuild_bars=build_result.tail_rebuild_bars,
            blocking_active_run_count=precheck.blocking_active_run_count,
            validation=validation_summary,
        )


def _resolve_requested_time_range_v2(
    *,
    canonical_candle_index_reader: CanonicalCandleIndexReader,
    coordinates: ArtifactCoordinatesV2,
    before: UtcTimestamp,
) -> TimeRange:
    """
    Resolve the canonical full-history source range for one explicit symbol root.

    Args:
        canonical_candle_index_reader: Aggregate reader for canonical 1m bounds.
        coordinates: Artifact coordinates being published.
        before: Exclusive upper bound used to ignore accidental future rows.
    Returns:
        TimeRange: Canonical full source range `[first_ts_open, last_ts_open + 1m)`.
    Assumptions:
        The artifact pipeline still builds bounded tails from a full canonical request envelope,
        letting the precompute runner decide whether prefix reuse is safe.
    Raises:
        ValueError: If the instrument has no canonical 1m data before `before`.
    Side Effects:
        Reads canonical bounds from storage.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/market_data/application/ports/stores/canonical_candle_index_reader.py
    """
    instrument_id = _instrument_id_from_coordinates_v2(coordinates=coordinates)
    first_ts_open, last_ts_open = canonical_candle_index_reader.bounds_1m(
        instrument_id=instrument_id,
        before=before,
    )
    if first_ts_open is None or last_ts_open is None:
        raise ValueError(
            "backtest artifact publish requires canonical 1m data before "
            f"{before.value.isoformat()} for "
            f"{coordinates.exchange}:{coordinates.market_type}:{coordinates.symbol}"
        )
    return TimeRange(
        start=first_ts_open,
        end=UtcTimestamp(last_ts_open.value + timedelta(minutes=1)),
    )


def _resolve_build_time_range_v2(
    *,
    requested_time_range: TimeRange,
    publish_mode: PublishBacktestArtifactsModeV2,
    precheck: ArtifactPublishPrecheckV2,
    slot_publisher: BacktestArtifactSlotPublisherV2,
) -> TimeRange:
    """
    Resolve the source envelope used for the actual inactive-slot build.

    Args:
        requested_time_range: Full canonical ClickHouse bounds for the target symbol.
        publish_mode: Resolved bootstrap/incremental/full-rebuild mode.
        precheck: Publish precheck carrying the active pointer, when present.
        slot_publisher: Publisher dependency exposing the artifact loader.
    Returns:
        TimeRange: Build range that preserves tail semantics for incremental publishes.
    Assumptions:
        Incremental publish is a tail refresh. If ClickHouse later exposes older canonical rows
        than the already-published artifact start, that is a separate head backfill/full-rebuild
        operation and must not silently disable prefix reuse.
    Raises:
        ValueError: If the active manifest lacks a strict `prices/1m` coverage section.
    Side Effects:
        Reads the active slot manifest for incremental publish only.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest_artifacts/application/services/v2/
        artifact_precompute_runner.py
    """
    if publish_mode != "incremental" or precheck.current_pointer is None:
        return requested_time_range
    active_start = _active_one_minute_start_v2(
        precheck=precheck,
        slot_publisher=slot_publisher,
    )
    if active_start.value <= requested_time_range.start.value:
        return requested_time_range
    if active_start.value >= requested_time_range.end.value:
        raise ValueError(
            "incremental artifact publish active prices/1m start must be before requested end; "
            f"got active_start={active_start.value.isoformat()} and "
            f"requested_end={requested_time_range.end.value.isoformat()}"
        )
    return TimeRange(start=active_start, end=requested_time_range.end)


def _active_one_minute_start_v2(
    *,
    precheck: ArtifactPublishPrecheckV2,
    slot_publisher: BacktestArtifactSlotPublisherV2,
) -> UtcTimestamp:
    """
    Return the active slot `prices/1m` start timestamp for incremental tail refreshes.

    Args:
        precheck: Publish precheck carrying the current active pointer.
        slot_publisher: Publisher dependency exposing the artifact loader.
    Returns:
        UtcTimestamp: Active artifact `prices/1m.coverage.open_time_start`.
    Assumptions:
        A valid current pointer always references a strict active slot manifest.
    Raises:
        ValueError: If the current pointer or `prices/1m` section is absent.
    Side Effects:
        Reads one active slot manifest.
    """
    current_pointer = precheck.current_pointer
    if current_pointer is None:
        raise ValueError("incremental artifact publish requires current pointer")
    active_manifest = slot_publisher.artifact_loader.load_slot_manifest(
        precheck.coordinates,
        current_pointer.active_slot,
    )
    for price_section in active_manifest.prices:
        if price_section.timeframe == "1m":
            return UtcTimestamp(
                datetime.fromtimestamp(
                    price_section.coverage.open_time_start / 1000,
                    tz=timezone.utc,
                )
            )
    raise ValueError("incremental artifact publish active manifest missing prices/1m section")


def _ensure_precheck_ready_v2(*, precheck: ArtifactPublishPrecheckV2) -> None:
    """
    Convert shared precheck diagnostics into the stable publish error used by callers.

    Args:
        precheck: Shared publish readiness snapshot from `BacktestArtifactSlotPublisherV2`.
    Returns:
        None.
    Assumptions:
        The use-case must fail before any build work starts when pin guard rejects the target.
    Raises:
        ArtifactSlotPublishErrorV2: If the precheck marked the target as blocked.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if precheck.ready:
        return
    raise ArtifactSlotPublishErrorV2(
        code=precheck.failure_code or "publish_precheck_failed",
        message=precheck.failure_message or "artifact publish precheck failed",
    )


def _ensure_bootstrap_slot_roots_v2(
    *,
    slot_publisher: BacktestArtifactSlotPublisherV2,
    coordinates: ArtifactCoordinatesV2,
    bootstrap: bool,
) -> None:
    """
    Create canonical symbol-root and slot-root directories for bootstrap publish only.

    Args:
        slot_publisher: Shared publisher exposing explicit artifact path resolution.
        coordinates: Artifact coordinates under bootstrap.
        bootstrap: Whether the current orchestration run is a bootstrap publish.
    Returns:
        None.
    Assumptions:
        Directory creation is deterministic because slot paths are fixed and never discovered by
        scanning the filesystem.
    Raises:
        OSError: If one directory cannot be created.
    Side Effects:
        Creates `<symbol-root>/slot_a` and `<symbol-root>/slot_b` when bootstrap is active.
    Docs:
      - docs/runbooks/backtest-artifacts-rebuild.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if not bootstrap:
        return
    artifact_loader = slot_publisher.artifact_loader
    symbol_root = artifact_loader.resolve_current_pointer_path(coordinates).parent
    slot_a_root = artifact_loader.resolve_slot_manifest_path(coordinates, "slot_a").parent
    slot_b_root = artifact_loader.resolve_slot_manifest_path(coordinates, "slot_b").parent
    for directory in (symbol_root, slot_a_root, slot_b_root):
        directory.mkdir(parents=True, exist_ok=True)


def _resolve_publish_mode_v2(
    *,
    request: PublishBacktestArtifactsV2Request,
    bootstrap: bool,
) -> PublishBacktestArtifactsModeV2:
    """
    Resolve the deterministic publish-mode label for result diagnostics.

    Args:
        request: User/scheduler publish request.
        bootstrap: Whether shared precheck resolved bootstrap target identity.
    Returns:
        PublishBacktestArtifactsModeV2: One of `bootstrap`, `incremental`, or `full_rebuild`.
    Assumptions:
        Bootstrap overrides explicit `full_rebuild` reporting because it carries a distinct
        operational meaning for first publish.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/cli/commands/backtest_artifact_publish.py
    """
    if bootstrap:
        return "bootstrap"
    if request.full_rebuild:
        return "full_rebuild"
    return "incremental"


def _ensure_build_publish_alignment_v2(
    *,
    build_request: ArtifactCanonicalPriceExportRequestV2,
    build_result: ArtifactCanonicalPriceExportResultV2,
    publish_result: ArtifactPublishResultV2,
) -> None:
    """
    Guard against orchestration drift between build target identity and published pointer switch.

    Args:
        build_request: Explicit build request forwarded to the precompute runner.
        build_result: Runner output for the materialized inactive slot.
        publish_result: Publisher output after whole-slot validation and pointer switch.
    Returns:
        None.
    Assumptions:
        Shared orchestration must never publish a different slot/generation/manifest than the one
        just built.
    Raises:
        ValueError: If build and publish identities drift.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_precompute_runner.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if build_result.slot != publish_result.published_pointer.active_slot:
        raise ValueError(
            "build/publish slot drift detected: "
            f"{build_result.slot!r} != {publish_result.published_pointer.active_slot!r}"
        )
    if build_result.slot_generation != publish_result.published_pointer.slot_generation:
        raise ValueError(
            "build/publish slot_generation drift detected: "
            f"{build_result.slot_generation!r} != "
            f"{publish_result.published_pointer.slot_generation!r}"
        )
    if build_result.manifest_sha256 != publish_result.published_pointer.manifest_sha256:
        raise ValueError(
            "build/publish manifest hash drift detected: "
            f"{build_result.manifest_sha256!r} != "
            f"{publish_result.published_pointer.manifest_sha256!r}"
        )
    if build_request.asof_date != publish_result.published_pointer.asof_date:
        raise ValueError(
            "build/publish asof_date drift detected: "
            f"{build_request.asof_date!r} != {publish_result.published_pointer.asof_date!r}"
        )


def _build_validation_summary_v2(
    *,
    publish_result: ArtifactPublishResultV2,
) -> PublishBacktestArtifactsV2ValidationSummary:
    """
    Collapse shared validation output into a stable machine-readable summary DTO.

    Args:
        publish_result: Shared publisher result containing whole-slot validation details.
    Returns:
        PublishBacktestArtifactsV2ValidationSummary: Stable validation summary.
    Assumptions:
        Validation already passed, so diagnostics count is usually zero but still explicit.
    Raises:
        ValueError: If validation omitted manifest hash unexpectedly.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    manifest_sha256 = publish_result.validation.manifest_sha256
    if manifest_sha256 is None:
        raise ValueError("publish validation summary requires manifest_sha256")
    slot_manifest = publish_result.validation.slot_manifest
    return PublishBacktestArtifactsV2ValidationSummary(
        slot_manifest_path=None if slot_manifest is None else slot_manifest.path,
        manifest_sha256=manifest_sha256,
        price_timeframes=(
            ()
            if slot_manifest is None
            else tuple(section.timeframe for section in slot_manifest.prices)
        ),
        mapping_timeframes=(
            ()
            if slot_manifest is None
            else tuple(section.timeframe for section in slot_manifest.mappings)
        ),
        signal_artifacts=tuple(
            (manifest.timeframe, manifest.indicator_id)
            for manifest in publish_result.validation.signal_manifests
        ),
        signal_manifest_count=len(publish_result.validation.signal_manifests),
        hit_times_manifest_present=publish_result.validation.hit_times_manifest is not None,
        funding_coverage_status=(
            None
            if slot_manifest is None or slot_manifest.funding is None
            else slot_manifest.funding.coverage_status
        ),
        funding_manifest_hash=(
            None
            if slot_manifest is None or slot_manifest.funding is None
            else slot_manifest.funding.funding_manifest_hash
        ),
        diagnostics_count=len(publish_result.validation.diagnostics),
    )


def _instrument_id_from_coordinates_v2(*, coordinates: ArtifactCoordinatesV2) -> InstrumentId:
    """
    Convert artifact coordinates into the canonical market-data instrument identity.

    Args:
        coordinates: Artifact coordinates for one symbol root.
    Returns:
        InstrumentId: Shared-kernel instrument identity used by canonical candle readers.
    Assumptions:
        Market-id mapping follows the fixed artifact market bridge already used by v2 services.
    Raises:
        ValueError: If coordinates cannot be mapped to a supported `market_id`.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    market_id = artifact_market_id_from_coordinates_v2(coordinates)
    return InstrumentId(MarketId(market_id), Symbol(coordinates.symbol))


def _validated_now_utc_v2(value: datetime) -> datetime:
    """
    Require that the shared use-case clock returns a timezone-aware UTC datetime.

    Args:
        value: Datetime candidate returned by the use-case clock dependency.
    Returns:
        datetime: The same UTC datetime once validated.
    Assumptions:
        Publish diagnostics and canonical-bounds lookup share one clock source.
    Raises:
        ValueError: If the datetime is naive or not UTC.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    utc_offset = value.utcoffset()
    if value.tzinfo is None or utc_offset is None:
        raise ValueError("publish use-case now_provider must return timezone-aware UTC datetime")
    if utc_offset.total_seconds() != 0:
        raise ValueError("publish use-case now_provider must return UTC datetime")
    return value.astimezone(timezone.utc)


def _utc_timestamp_literal_v2(value: datetime) -> str:
    """
    Serialize one UTC datetime into strict `YYYY-MM-DDTHH:MM:SSZ` form.

    Args:
        value: Timezone-aware UTC datetime.
    Returns:
        str: Strict UTC timestamp literal.
    Assumptions:
        CLI and scheduler diagnostics should reuse the same strict timestamp formatting as
        `current.yaml.published_at_utc`.
    Raises:
        ValueError: If the datetime is naive or not UTC.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/contracts.py
    """
    return validate_current_pointer_published_at_utc_v2(
        _validated_now_utc_v2(value).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


__all__ = [
    "PublishBacktestArtifactsModeV2",
    "PublishBacktestArtifactsV2Request",
    "PublishBacktestArtifactsV2Result",
    "PublishBacktestArtifactsV2UseCase",
    "PublishBacktestArtifactsV2ValidationSummary",
]
