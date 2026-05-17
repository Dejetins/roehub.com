"""Publisher-owned availability summary for active backtest artifact roots."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from .contracts import (
    ARTIFACT_MAPPING_TIMEFRAMES_V2,
    ARTIFACT_ROOT_SCHEMA_VERSION_V2,
    ARTIFACT_SIGNAL_TIMEFRAMES_V2,
    AVAILABILITY_SUMMARY_FILENAME_V2,
    AVAILABILITY_SUMMARY_SCHEMA_VERSION_V2,
    AVAILABILITY_SUMMARY_SOURCE_LITERAL_V2,
    CURRENT_ARTIFACT_POINTER_FILENAME_V2,
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    ArtifactManifestDocumentV2,
    ArtifactPriceTimeframeManifestV2,
    BacktestArtifactLoaderV2,
    BacktestArtifactPathResolverV2,
)

NowProviderV2 = Callable[[], datetime]


def _default_now_provider_v2() -> datetime:
    """
    Return a timezone-aware UTC timestamp for availability summary generation.

    Args:
        None.
    Returns:
        datetime: Current UTC wall clock.
    Assumptions:
        Summary generation is a publisher-side batch step, not a request-path operation.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
    Related:
      - src/trading/contexts/backtest_artifacts/application/services/v2/
        artifact_availability_summary.py
    """
    return datetime.now(timezone.utc)


class BacktestArtifactAvailabilitySummaryWriterV2(Protocol):
    """
    Port for atomic root-level `availability_summary.yaml` replacement.

    Docs:
      - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
    Related:
      - src/trading/contexts/backtest_artifacts/adapters/outbound/artifacts_fs/
        availability_summary_writer.py
    """

    def write_availability_summary_atomically(
        self,
        *,
        artifact_root: Path,
        payload: Mapping[str, Any],
    ) -> Path:
        """Atomically replace `<artifact_root>/availability_summary.yaml`."""
        ...


@dataclass(frozen=True, slots=True)
class BacktestArtifactAvailabilitySummaryResultV2:
    """
    Diagnostics returned after one availability summary regeneration.

    Docs:
      - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
    Related:
      - src/trading/contexts/backtest_artifacts/application/services/v2/
        artifact_availability_summary.py
    """

    summary_path: Path
    summary_hash: str
    generated_at_utc: str
    instrument_count: int
    skipped_count: int
    skipped_reasons: Mapping[str, int]

    def __post_init__(self) -> None:
        """
        Freeze skipped-reason diagnostics for stable callers.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Diagnostics are reporting-only and should not be mutated by callers.
        Raises:
            ValueError: If counters are negative.
        Side Effects:
            Replaces `skipped_reasons` with a read-only mapping.
        """
        if self.instrument_count < 0:
            raise ValueError("instrument_count must be non-negative")
        if self.skipped_count < 0:
            raise ValueError("skipped_count must be non-negative")
        if any(count < 0 for count in self.skipped_reasons.values()):
            raise ValueError("skipped reason counts must be non-negative")
        object.__setattr__(
            self,
            "skipped_reasons",
            MappingProxyType(dict(sorted(self.skipped_reasons.items()))),
        )

    def as_dict(self) -> Mapping[str, object]:
        """
        Serialize regeneration diagnostics into a JSON-friendly mapping.

        Args:
            None.
        Returns:
            Mapping[str, object]: Stable scalar diagnostics.
        Assumptions:
            CLI/scheduler logs need compact evidence, not the full YAML payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {
            "summary_path": str(self.summary_path),
            "summary_hash": self.summary_hash,
            "generated_at_utc": self.generated_at_utc,
            "instrument_count": self.instrument_count,
            "skipped_count": self.skipped_count,
            "skipped_reasons": dict(self.skipped_reasons),
        }


@dataclass(frozen=True, slots=True)
class BacktestArtifactAvailabilitySummaryGeneratorV2:
    """
    Scan active artifact pointers and publish the AI availability source-of-truth YAML.

    Docs:
      - docs/architecture/backtest/backtest-ai-configurator-assistant-v1.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
      - apps/cli/commands/backtest_artifact_publish.py
    """

    artifact_root: Path
    path_resolver: BacktestArtifactPathResolverV2
    artifact_loader: BacktestArtifactLoaderV2
    writer: BacktestArtifactAvailabilitySummaryWriterV2
    now_provider: NowProviderV2 = _default_now_provider_v2

    def __post_init__(self) -> None:
        """
        Validate generator dependencies.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            The same artifact root is used by path resolver, loader, and writer wiring.
        Raises:
            ValueError: If a required dependency is absent.
        Side Effects:
            Normalizes `artifact_root` to a `Path`.
        """
        object.__setattr__(self, "artifact_root", Path(self.artifact_root))
        if self.path_resolver is None:  # type: ignore[truthy-bool]
            raise ValueError("path_resolver is required")
        if self.artifact_loader is None:  # type: ignore[truthy-bool]
            raise ValueError("artifact_loader is required")
        if self.writer is None:  # type: ignore[truthy-bool]
            raise ValueError("writer is required")

    def regenerate(self) -> BacktestArtifactAvailabilitySummaryResultV2:
        """
        Build and atomically write the root-level availability summary.

        Args:
            None.
        Returns:
            BacktestArtifactAvailabilitySummaryResultV2: Regeneration diagnostics.
        Assumptions:
            This method is called by trusted publisher/manual paths, never by normal AI requests.
        Raises:
            OSError: If the final atomic write fails.
        Side Effects:
            Scans current artifact YAML state and replaces `availability_summary.yaml`.
        """
        payload, skipped_reasons = self.build_payload()
        summary_path = self.writer.write_availability_summary_atomically(
            artifact_root=self.artifact_root,
            payload=payload,
        )
        instruments = payload.get("instruments", {})
        if not isinstance(instruments, Mapping):
            raise ValueError("availability summary payload instruments must be a mapping")
        skipped_count = sum(skipped_reasons.values())
        return BacktestArtifactAvailabilitySummaryResultV2(
            summary_path=summary_path,
            summary_hash=str(payload["summary_hash"]),
            generated_at_utc=str(payload["generated_at_utc"]),
            instrument_count=len(instruments),
            skipped_count=skipped_count,
            skipped_reasons=skipped_reasons,
        )

    def build_payload(self) -> tuple[dict[str, Any], Mapping[str, int]]:
        """
        Build deterministic summary payload from valid active artifact pointer state.

        Args:
            None.
        Returns:
            tuple[dict[str, Any], Mapping[str, int]]: YAML payload and skip diagnostics.
        Assumptions:
            Only `current.yaml` and the active slot `manifest.yaml` are authoritative here.
        Raises:
            ValueError: If the configured clock is not timezone-aware UTC.
        Side Effects:
            Reads artifact YAML files from `artifact_root`.
        """
        generated_at_utc = _format_utc_timestamp_v2(self.now_provider())
        instruments: dict[str, Any] = {}
        skipped_reasons: Counter[str] = Counter()
        for symbol_root in self._iter_symbol_roots():
            try:
                coordinates = self._coordinates_from_symbol_root(symbol_root)
            except ValueError:
                skipped_reasons["invalid_symbol_root"] += 1
                continue
            instrument_summary = self._load_instrument_summary(
                symbol_root=symbol_root,
                coordinates=coordinates,
                skipped_reasons=skipped_reasons,
            )
            if instrument_summary is None:
                continue
            instruments[_instrument_key_v2(coordinates)] = instrument_summary

        payload: dict[str, Any] = {
            "schema_version": AVAILABILITY_SUMMARY_SCHEMA_VERSION_V2,
            "generated_at_utc": generated_at_utc,
            "artifact_root": str(self.artifact_root),
            "artifact_root_schema_version": ARTIFACT_ROOT_SCHEMA_VERSION_V2,
            "summary_hash": "",
            "source": AVAILABILITY_SUMMARY_SOURCE_LITERAL_V2,
            "instruments": dict(sorted(instruments.items())),
        }
        payload["summary_hash"] = _compute_summary_hash_v2(payload)
        return payload, MappingProxyType(dict(sorted(skipped_reasons.items())))

    def _iter_symbol_roots(self) -> tuple[Path, ...]:
        """
        Return deterministic `<exchange>/<market>/<symbol>` roots under `artifact_root`.

        Args:
            None.
        Returns:
            tuple[Path, ...]: Existing symbol-root directories in stable order.
        Assumptions:
            Artifact root is a three-level namespace before files/slots begin.
        Raises:
            None.
        Side Effects:
            Reads directory entries.
        """
        if not self.artifact_root.is_dir():
            return ()
        roots: list[Path] = []
        for exchange_dir in _sorted_child_dirs_v2(self.artifact_root):
            for market_dir in _sorted_child_dirs_v2(exchange_dir):
                roots.extend(_sorted_child_dirs_v2(market_dir))
        return tuple(roots)

    def _coordinates_from_symbol_root(self, symbol_root: Path) -> ArtifactCoordinatesV2:
        relative_parts = symbol_root.relative_to(self.artifact_root).parts
        if len(relative_parts) != 3:
            raise ValueError(f"invalid artifact symbol root depth: {symbol_root}")
        exchange, market, symbol = relative_parts
        return ArtifactCoordinatesV2(exchange=exchange, market_type=market, symbol=symbol)

    def _load_instrument_summary(
        self,
        *,
        symbol_root: Path,
        coordinates: ArtifactCoordinatesV2,
        skipped_reasons: Counter[str],
    ) -> dict[str, Any] | None:
        current_path = symbol_root / CURRENT_ARTIFACT_POINTER_FILENAME_V2
        if not current_path.is_file():
            skipped_reasons["missing_current"] += 1
            return None
        try:
            pointer = self.artifact_loader.load_current_pointer_from_path(current_path)
        except Exception:  # noqa: BLE001
            skipped_reasons["invalid_current"] += 1
            return None

        slot_root = self.path_resolver.slot_root(coordinates, pointer.active_slot)
        if not slot_root.is_dir():
            skipped_reasons["missing_active_slot"] += 1
            return None
        manifest_path = self.path_resolver.slot_manifest_path(coordinates, pointer.active_slot)
        if not manifest_path.is_file():
            skipped_reasons["missing_active_manifest"] += 1
            return None
        try:
            manifest = self.artifact_loader.load_manifest_from_path(
                manifest_path,
                slot=pointer.active_slot,
            )
        except Exception:  # noqa: BLE001
            skipped_reasons["invalid_active_manifest"] += 1
            return None

        mismatch_reason = _manifest_mismatch_reason_v2(
            coordinates=coordinates,
            pointer=pointer,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        if mismatch_reason is not None:
            skipped_reasons[mismatch_reason] += 1
            return None

        instrument_summary = _instrument_summary_v2(
            coordinates=coordinates,
            pointer=pointer,
            manifest=manifest,
        )
        if instrument_summary is None:
            skipped_reasons["no_backtest_timeframes"] += 1
            return None
        return instrument_summary


def _manifest_mismatch_reason_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    pointer: ArtifactCurrentPointerV2,
    manifest: ArtifactManifestDocumentV2,
    manifest_path: Path,
) -> str | None:
    if manifest.identity != coordinates:
        return "identity_mismatch"
    if manifest.slot_generation != pointer.slot_generation:
        return "current_manifest_identity_mismatch"
    if manifest.asof_date != pointer.asof_date:
        return "current_manifest_identity_mismatch"
    if _file_sha256_hex_v2(manifest_path) != pointer.manifest_sha256:
        return "manifest_hash_mismatch"
    return None


def _instrument_summary_v2(
    *,
    coordinates: ArtifactCoordinatesV2,
    pointer: ArtifactCurrentPointerV2,
    manifest: ArtifactManifestDocumentV2,
) -> dict[str, Any] | None:
    price_by_timeframe = {entry.timeframe: entry for entry in manifest.prices}
    mapping_timeframes = {entry.timeframe for entry in manifest.mappings}
    signal_ids_by_timeframe: dict[str, list[str]] = {}
    for entry in manifest.signals.manifests:
        signal_ids_by_timeframe.setdefault(entry.timeframe, []).append(entry.indicator_id)

    timeframes: dict[str, Any] = {}
    backtest_timeframes: list[str] = []
    for timeframe in ARTIFACT_SIGNAL_TIMEFRAMES_V2:
        price_manifest = price_by_timeframe.get(timeframe)
        indicator_ids = tuple(sorted(set(signal_ids_by_timeframe.get(timeframe, ()))))
        if (
            price_manifest is None
            or timeframe not in mapping_timeframes
            or len(indicator_ids) == 0
        ):
            continue
        coverage = _timeframe_coverage_v2(price_manifest)
        timeframes[timeframe] = {
            "start_date": coverage["start_date"],
            "end_date": coverage["end_date"],
            "bars": price_manifest.coverage.bar_count,
            "price_available": True,
            "signals_available": True,
            "mappings_available": True,
            "indicator_ids": list(indicator_ids),
        }
        backtest_timeframes.append(timeframe)

    if len(backtest_timeframes) == 0:
        return None

    top_level_start_date = max(
        str(timeframes[timeframe]["start_date"]) for timeframe in backtest_timeframes
    )
    top_level_end_date = min(
        str(timeframes[timeframe]["end_date"]) for timeframe in backtest_timeframes
    )
    return {
        "exchange": coordinates.exchange,
        "market": coordinates.market_type,
        "symbol": coordinates.symbol,
        "active_slot": pointer.active_slot,
        "slot_generation": pointer.slot_generation,
        "asof_date": pointer.asof_date,
        "published_at_utc": pointer.published_at_utc,
        "manifest_sha256": pointer.manifest_sha256,
        "start_date": top_level_start_date,
        "end_date": top_level_end_date,
        "backtest_timeframes": backtest_timeframes,
        "timeframes": timeframes,
        "hit_times": {
            "timeframe": manifest.hit_times.timeframe,
            "available": manifest.hit_times.timeframe in ARTIFACT_MAPPING_TIMEFRAMES_V2,
        },
    }


def _timeframe_coverage_v2(
    price_manifest: ArtifactPriceTimeframeManifestV2,
) -> Mapping[str, str]:
    return {
        "start_date": _epoch_millis_or_seconds_to_date_v2(
            price_manifest.coverage.open_time_start
        ),
        "end_date": _epoch_millis_or_seconds_to_date_v2(
            price_manifest.coverage.close_time_end
        ),
    }


def _epoch_millis_or_seconds_to_date_v2(value: int) -> str:
    seconds = value / 1000.0 if abs(value) > 9_999_999_999 else float(value)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).date().isoformat()


def _format_utc_timestamp_v2(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("availability summary now_provider must return timezone-aware UTC")
    normalized = value.astimezone(timezone.utc)
    return normalized.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _compute_summary_hash_v2(payload: Mapping[str, Any]) -> str:
    hash_payload = dict(payload)
    hash_payload.pop("summary_hash", None)
    hash_payload.pop("generated_at_utc", None)
    serialized = json.dumps(
        hash_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256_hex_v2(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sorted_child_dirs_v2(path: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (child for child in path.iterdir() if child.is_dir()),
            key=lambda child: child.name,
        )
    )


def _instrument_key_v2(coordinates: ArtifactCoordinatesV2) -> str:
    return f"{coordinates.exchange}/{coordinates.market_type}/{coordinates.symbol}"


__all__ = [
    "BacktestArtifactAvailabilitySummaryGeneratorV2",
    "BacktestArtifactAvailabilitySummaryResultV2",
    "BacktestArtifactAvailabilitySummaryWriterV2",
    "AVAILABILITY_SUMMARY_FILENAME_V2",
]
