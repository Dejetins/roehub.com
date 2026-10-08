from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import yaml

from trading.contexts.backtest.application.dto import (
    BacktestArtifactMetadata,
    BacktestCoordinates,
    BacktestTpSlHitTimesGridArrays,
    BacktestTpSlHitTimesTableArrays,
)
from trading.contexts.backtest.application.dto.artifact_inputs import (
    BacktestArtifactRuntimeContext,
    BacktestSignalMatrix,
    BacktestSignalMetadata,
)
from trading.contexts.backtest.application.ports.artifact_arrays import BacktestArtifactArrayLoader
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactArrayMetadataV2,
    ArtifactCoordinatesV2,
    ArtifactFundingArraysV2,
    ArtifactInputFileReference,
    ArtifactMappingArraysV2,
    ArtifactPrefixDomain,
    ArtifactPriceArraysV2,
    ArtifactSlotLiteralV2,
    ArtifactSlotPinnedRuntimeContextV2,
    ArtifactTimelineCoverageV2,
    BacktestArtifactLoaderV2,
    BacktestPreparedArtifactSet,
)


@dataclass(frozen=True, slots=True)
class FilesystemBacktestArtifactArrayLoader(BacktestArtifactArrayLoader):
    """One explicit-reference mmap path for native, generated and mixed inputs."""

    artifact_loader: BacktestArtifactLoaderV2

    def resolve_context(
        self,
        *,
        coordinates: BacktestCoordinates,
        artifact_metadata: BacktestArtifactMetadata,
        metadata_only: bool = False,
    ) -> BacktestArtifactRuntimeContext:
        coordinates_v2 = ArtifactCoordinatesV2(
            exchange=coordinates.exchange,
            market_type=coordinates.market_type,
            symbol=coordinates.symbol,
        )
        slot = cast(ArtifactSlotLiteralV2, artifact_metadata.artifact_slot)
        path = self.artifact_loader.resolve_slot_manifest_path(coordinates_v2, slot)
        root = path.parent.resolve(strict=True)
        path = _trusted_path(root, path.name)
        _verify_hash(path, artifact_metadata.artifact_manifest_hash)
        manifest = self.artifact_loader.load_manifest_from_path(path, slot=slot)
        if (
            manifest.slot_generation != artifact_metadata.artifact_slot_generation
            or manifest.asof_date != artifact_metadata.artifact_asof_date
        ):
            raise ValueError("root manifest source identity differs from pinned metadata")
        source = ArtifactSlotPinnedRuntimeContextV2(
            coordinates=coordinates_v2,
            artifact_slot=slot,
            slot_generation=artifact_metadata.artifact_slot_generation,
            artifact_asof_date=artifact_metadata.artifact_asof_date,
            artifact_manifest_hash=artifact_metadata.artifact_manifest_hash,
            slot_root_path=root,
            slot_manifest_path=path,
            slot_manifest=manifest,
        )
        refs: list[ArtifactInputFileReference] = []
        metadata: dict[tuple[str, str], BacktestSignalMetadata] = {}
        root_id = "published-source"
        coverages = {item.timeframe: item.coverage for item in manifest.prices}

        def reference(
            array: ArtifactArrayMetadataV2,
            role: str,
            timeframe: str,
            axes: tuple[str, ...],
            rows: tuple[int, ...] = (),
            levels: tuple[float, ...] = (),
        ) -> None:
            coverage = coverages[timeframe]
            if metadata_only:
                expected_dtype = array.dtype
                expected_shape = array.shape
                expected_axes = axes
                if role.startswith("prices."):
                    expected_dtype = "float32" if role.endswith(".ohlcv") else "int64"
                    expected_shape = ((coverage.bar_count, 5) if role.endswith(".ohlcv")
                                      else (coverage.bar_count,))
                elif role.startswith("mappings."):
                    expected_dtype, expected_shape = "uint32", (coverage.bar_count,)
                elif role.startswith("funding."):
                    expected_dtype = {
                        "funding_time": "int64", "funding_rate": "float64",
                        "mark_price": "float64", "funding_interval_minutes": "uint16",
                        "data_quality": "uint8",
                    }[role.split(".")[1]]
                    assert manifest.funding is not None
                    expected_shape = (manifest.funding.rows_count,)
                elif role.startswith("signals."):
                    expected_dtype, expected_shape = "int8", (len(rows), coverage.bar_count)
                if (array.dtype != expected_dtype or array.shape != expected_shape
                        or array.axis_order != expected_axes):
                    raise ValueError("declared source semantic dtype/shape/axes mismatch")
                file_path = _trusted_path(root, array.path)
                with file_path.open("rb") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version not in ((1, 0), (2, 0)):
                        raise ValueError("unsupported NPY header")
                    # NumPy checks its limit after reading the body. Bound that read first.
                    length_size = 2 if version == (1, 0) else 4
                    raw_length = stream.read(length_size)
                    if len(raw_length) != length_size:
                        raise ValueError("truncated NPY header length")
                    if int.from_bytes(raw_length, "little") > 10_000:
                        raise ValueError("NPY header exceeds metadata budget")
                    stream.seek(-length_size, 1)
                    if version == (1, 0):
                        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                    else:
                        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                    if (shape != array.shape or dtype != np.dtype(array.dtype) or fortran
                            or file_path.stat().st_size != (
                                stream.tell() + math.prod(shape) * dtype.itemsize
                            )):
                        raise ValueError("declared array header/size mismatch")
            refs.append(
                ArtifactInputFileReference(
                    root_id=root_id,
                    relative_path=array.path,
                    sha256=array.sha256,
                    domain=ArtifactPrefixDomain(
                        role=role,
                        timeframe=timeframe,
                        dtype=np.dtype(array.dtype).str,
                        axis_order=axes,
                        origin_utc=_utc(coverage.open_time_start),
                        end_utc=_utc(coverage.close_time_end),
                        row_count=array.shape[0],
                        shape=array.shape,
                    ),
                    row_ids=rows,
                    risk_values=levels,
                )
            )

        for prices in manifest.prices:
            for name in ("open_time", "close_time", "ohlcv"):
                reference(
                    getattr(prices, name),
                    f"prices.{prices.timeframe}.{name}",
                    prices.timeframe,
                    ("time", "field") if name == "ohlcv" else ("time",),
                )
        for mapping in manifest.mappings:
            for name in ("bar_open_1m_idx", "bar_close_1m_idx"):
                reference(
                    getattr(mapping, name),
                    f"mappings.{mapping.timeframe}.{name}",
                    mapping.timeframe,
                    ("time",),
                )
        funding = manifest.funding
        if funding is not None and funding.coverage_status in ("ready", "degraded"):
            for name in (
                "funding_time",
                "funding_rate",
                "mark_price",
                "funding_interval_minutes",
                "data_quality",
            ):
                array = getattr(funding, name)
                if array is None:
                    raise ValueError("funding manifest is missing declared array metadata")
                reference(array, f"funding.{name}", "1m", ("funding_event",))
        for item in manifest.signals.manifests:
            signal_path = _trusted_path(root, item.manifest_path)
            _verify_hash(signal_path, item.manifest_sha256)
            signal = self.artifact_loader.load_signal_manifest_from_path(signal_path, slot=slot)
            _validate_component_source(source, signal, coverages[item.timeframe])
            if signal.timeframe != item.timeframe or signal.indicator_id != item.indicator_id:
                raise ValueError("signal manifest differs from its declared catalog identity")
            # Native signal manifests identify a canonical prefix, verified against the
            # builder's grid/defaults before reuse. Its length is not the full grid size.
            row_ids = tuple(range(signal.rows_count))
            canonical_count = None
            metadata[(item.timeframe, item.indicator_id)] = BacktestSignalMetadata(
                canonical_rows_count=canonical_count,
                manifest=signal,
            )
            reference(
                signal.signals,
                f"signals.{item.indicator_id}",
                item.timeframe,
                ("variant", "time"),
                row_ids,
            )
            if signal.signal_features is not None:
                feature_ref = signal.signal_features
                feature_path = _trusted_path(root, feature_ref.manifest_path)
                _verify_hash(feature_path, feature_ref.manifest_sha256)
                features = self.artifact_loader.load_signal_features_manifest_from_path(
                    feature_path, slot=slot
                )
                _validate_component_source(source, features, None)
                if (
                    features.rows_count != len(row_ids)
                    or features.timeframe != item.timeframe
                    or features.indicator_id != item.indicator_id
                ):
                    raise ValueError("signal features differ from signal component identity")
                reference(
                    features.features,
                    f"signal_features.{item.indicator_id}",
                    item.timeframe,
                    ("variant", "feature"),
                    row_ids,
                )
        hit = None
        hit_hash = None
        if manifest.hit_times is not None:
            hit_path = _trusted_path(root, manifest.hit_times.manifest_path)
            hit_hash = manifest.hit_times.manifest_sha256
            _verify_hash(hit_path, hit_hash)
            hit = self.artifact_loader.load_hit_times_manifest_from_path(hit_path, slot=slot)
            _validate_component_source(source, hit, None)
            coverage = coverages[hit.timeframe]
            if (
                hit.timeline_bar_count != coverage.bar_count
                or hit.sentinel_index != coverage.bar_count
            ):
                raise ValueError("hit-time timeline/sentinel differs from source timeline")
            for side in (() if metadata_only else ("tp", "sl")):
                axis = getattr(hit, f"{side}_values")
                axis_path = _trusted_path(root, axis.path)
                _verify_hash(axis_path, axis.sha256)
                values = np.load(axis_path, mmap_mode="r", allow_pickle=False)
                if values.dtype != np.dtype(np.float32) or values.shape != axis.shape:
                    raise ValueError("native risk axis dtype/shape mismatch")
                levels = tuple(float(Decimal(str(value)) * 100) for value in values)
                reference(
                    axis, f"hit_times.{side}_values", hit.timeframe, ("level",), levels=levels
                )
                for direction in ("long", "short"):
                    name = f"{direction}_{side}"
                    reference(
                        getattr(hit, name).array,
                        f"hit_times.{name}",
                        hit.timeframe,
                        ("level", "time"),
                        levels=levels,
                    )
        return BacktestArtifactRuntimeContext(
            source=source,
            references=tuple(refs),
            trusted_roots={root_id: root},
            signal_metadata=metadata,
            signal_manifests={
                key: info.manifest for key, info in metadata.items() if info.manifest is not None
            },
            hit_times_manifest=hit,
            hit_times_manifest_hash=hit_hash,
        )

    def write_prepared_manifest(
        self,
        *,
        prepared: BacktestPreparedArtifactSet,
        output_directory: Path,
    ) -> Path:
        """Publish the validated job manifest independently of the builder manifest."""
        if output_directory.is_symlink():
            raise ValueError("attempt directory must not be a symlink")
        output_directory.mkdir(parents=True, exist_ok=True)
        root = output_directory.resolve(strict=True)
        target = root / "prepared-inputs.yaml"
        if target.exists() or target.is_symlink():
            raise FileExistsError("prepared manifest already exists")
        if "generated" in prepared.provenance:
            _verify_derived_marker(root, prepared)
        elif any(root.iterdir()):
            raise ValueError("all-native attempt directory is occupied")
        encoded = yaml.safe_dump(prepared.as_mapping(), sort_keys=False)
        if len(encoded.encode("utf-8")) > 8 * 1024**2:
            raise ValueError("prepared manifest metadata disk budget exceeded")
        fd, temporary = tempfile.mkstemp(prefix=".prepared-inputs-", suffix=".yaml", dir=root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            # link is an atomic no-replace publication: a concurrent/foreign target wins.
            os.link(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return target

    def with_prepared_inputs(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
        prepared: BacktestPreparedArtifactSet,
        output_directory: Path,
        acknowledged_prepared_sha256: str | None = None,
        signal_metadata: Mapping[tuple[str, str], BacktestSignalMetadata] | None = None,
    ) -> BacktestArtifactRuntimeContext:
        snapshot = prepared.snapshot
        source = context.source
        if (
            snapshot.coordinates != source.coordinates
            or snapshot.slot != source.artifact_slot
            or snapshot.generation != source.slot_generation
            or snapshot.manifest_sha256 != source.artifact_manifest_hash
        ):
            raise ValueError("prepared inputs differ from the pinned source")
        if acknowledged_prepared_sha256 != prepared.content_sha256:
            raise ValueError("prepared inputs differ from acknowledged content")
        roots = dict(context.trusted_roots)
        if prepared.output_root_id in roots:
            raise ValueError("attempt root must not replace a trusted published root")
        if output_directory.is_symlink():
            raise ValueError("attempt directory must not be a symlink")
        output_root = Path(output_directory).resolve(strict=True)
        if any(
            output_root.is_relative_to(root) or root.is_relative_to(output_root)
            for root in roots.values()
        ):
            raise ValueError("attempt directory overlaps published source")
        saved_path = _trusted_path(output_root, "prepared-inputs.yaml")
        with saved_path.open(encoding="utf-8") as handle:
            saved = yaml.safe_load(handle)
        if saved != prepared.as_mapping():
            raise ValueError("saved prepared manifest differs from acknowledged inputs")
        if "generated" in prepared.provenance:
            _verify_derived_marker(output_root, prepared)
        roots[prepared.output_root_id] = output_root
        source_refs = set(context.references)
        if any(ref not in source_refs for ref in snapshot.source_file_identities):
            raise ValueError("prepared source references differ from native inventory")
        for ref, provenance in zip(prepared.derivatives, prepared.provenance, strict=True):
            if provenance == "reused" and ref not in source_refs:
                raise ValueError("reused derivative differs from native inventory")
        result = replace(
            context,
            references=(*snapshot.source_file_identities, *prepared.derivatives),
            trusted_roots=roots,
            prepared=prepared,
            signal_metadata={**context.signal_metadata, **(signal_metadata or {})},
            hit_times_manifest=None,
            hit_times_manifest_hash=None,
        )
        for ref in result.references:
            _reference_path(result, ref)
        return result

    def load_price_arrays(
        self, *, context: BacktestArtifactRuntimeContext, timeframe: str
    ) -> ArtifactPriceArraysV2:
        manifest = next(
            item for item in context.source.slot_manifest.prices if item.timeframe == timeframe
        )
        return ArtifactPriceArraysV2(
            timeframe=timeframe,
            manifest=manifest,
            open_time=_load_role(context, f"prices.{timeframe}.open_time", timeframe),
            close_time=_load_role(context, f"prices.{timeframe}.close_time", timeframe),
            ohlcv=_load_role(context, f"prices.{timeframe}.ohlcv", timeframe),
        )

    def load_mapping_arrays(
        self, *, context: BacktestArtifactRuntimeContext, timeframe: str
    ) -> ArtifactMappingArraysV2:
        manifest = next(
            item for item in context.source.slot_manifest.mappings if item.timeframe == timeframe
        )
        return ArtifactMappingArraysV2(
            timeframe=timeframe,
            manifest=manifest,
            bar_open_1m_idx=_load_role(context, f"mappings.{timeframe}.bar_open_1m_idx", timeframe),
            bar_close_1m_idx=_load_role(
                context, f"mappings.{timeframe}.bar_close_1m_idx", timeframe
            ),
        )

    def load_funding_arrays(
        self, *, context: BacktestArtifactRuntimeContext
    ) -> ArtifactFundingArraysV2:
        manifest = context.source.slot_manifest.funding
        if manifest is None or manifest.coverage_status not in ("ready", "degraded"):
            raise ValueError("funding arrays require ready/degraded declared coverage")
        return ArtifactFundingArraysV2(
            manifest=manifest,
            funding_manifest_hash=manifest.funding_manifest_hash,
            coverage_status=manifest.coverage_status,
            **{
                name: _load_role(context, f"funding.{name}", "1m")
                for name in (
                    "funding_time",
                    "funding_rate",
                    "mark_price",
                    "funding_interval_minutes",
                    "data_quality",
                )
            },
        )

    def load_signal_matrix(
        self, *, context: BacktestArtifactRuntimeContext, timeframe: str, indicator_id: str
    ) -> BacktestSignalMatrix:
        refs = _references(context, f"signals.{indicator_id}", timeframe)
        matrix, rows = _load_rows(context, refs, risk=False)
        info = context.signal_metadata.get((timeframe, indicator_id), BacktestSignalMetadata(None))
        return BacktestSignalMatrix(
            timeframe=timeframe,
            indicator_id=indicator_id,
            row_ids=tuple(int(row) for row in rows),
            canonical_rows_count=info.canonical_rows_count,
            matrix=matrix,
            manifest=info.manifest if len(refs) == 1 and context.prepared is None else None,
        )

    def load_signal_rows(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
        timeframe: str,
        indicator_id: str,
        row_ids: np.ndarray,
        time_slice: slice,
    ) -> np.ndarray:
        matrix = self.load_signal_matrix(
            context=context, timeframe=timeframe, indicator_id=indicator_id
        )
        positions = {row: index for index, row in enumerate(matrix.row_ids)}
        try:
            physical = np.asarray([positions[int(row)] for row in row_ids], dtype=np.int32)
        except KeyError as error:
            raise ValueError(
                f"requested canonical signal row is missing: {error.args[0]}"
            ) from error
        return copy_signal_rows_i8(matrix.matrix, row_ids=physical, time_slice=time_slice)

    def load_hit_times_grid_arrays(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
    ) -> BacktestTpSlHitTimesGridArrays:
        refs = tuple(ref for ref in context.references if ref.domain.role.startswith("hit_times."))
        if not refs:
            raise FileNotFoundError("prepared inputs do not declare hit-times")
        tables = [ref for ref in refs if ref.domain.axis_order == ("level", "time")]
        if not tables:
            raise ValueError("hit-times require explicit table coverage")
        domain = tables[0].domain
        timeline = (
            domain.timeframe,
            domain.origin_utc,
            domain.end_utc,
            domain.index_origin,
            domain.shape[1],
        )
        if any(
            (
                r.domain.timeframe,
                r.domain.origin_utc,
                r.domain.end_utc,
                r.domain.index_origin,
                r.domain.shape[1],
            )
            != timeline
            for r in tables
        ):
            raise ValueError("hit-time components have incompatible timelines")
        axes = {}
        for side in ("tp", "sl"):
            axis_refs = _references(
                context, f"hit_times.{side}_values", domain.timeframe, required=False
            )
            values = []
            physical_values = []
            for ref in axis_refs:
                array = _load_reference(context, ref)
                expected = np.asarray(
                    [value / 100.0 for value in ref.risk_values], dtype=np.float32
                )
                if array.dtype != np.dtype(np.float32) or not np.allclose(
                    array,
                    expected,
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise ValueError("risk axis values differ from explicit reference")
                values.extend(ref.risk_values)
                physical_values.extend(array)
            if len(set(values)) != len(values):
                raise ValueError("duplicate risk coverage")
            axes[side] = np.asarray(physical_values, dtype=np.float32)[np.argsort(values)]
            for direction in ("long", "short"):
                covered = [
                    value
                    for ref in tables
                    if ref.domain.role == f"hit_times.{direction}_{side}"
                    for value in ref.risk_values
                ]
                if sorted(covered) != sorted(values):
                    raise ValueError("risk tables do not cover the explicit risk axis")
        identity = hashlib.sha256(
            json.dumps(
                [
                    ref.as_mapping()
                    for ref in sorted(
                        refs, key=lambda r: (r.domain.role, r.root_id, r.relative_path)
                    )
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return BacktestTpSlHitTimesGridArrays(
            manifest=context.hit_times_manifest,
            manifest_hash=context.hit_times_manifest_hash,
            input_identity_sha256=identity,
            tp_values=axes["tp"],
            sl_values=axes["sl"],
            timeframe=domain.timeframe,
            sentinel_index=domain.shape[1],
            index_origin=domain.index_origin,
            origin_utc=domain.origin_utc,
            end_utc=domain.end_utc,
        )

    def load_hit_times_table_arrays(
        self,
        *,
        context: BacktestArtifactRuntimeContext,
    ) -> BacktestTpSlHitTimesTableArrays:
        grid = self.load_hit_times_grid_arrays(context=context)
        arrays = {}
        for name in ("long_tp", "long_sl", "short_tp", "short_sl"):
            refs = _references(context, f"hit_times.{name}", grid.timeframe, required=False)
            arrays[name] = (
                _load_rows(context, refs, risk=True)[0]
                if refs
                else np.empty((0, grid.sentinel_index), dtype=np.uint32)
            )
        return BacktestTpSlHitTimesTableArrays(
            manifest=grid.manifest,
            manifest_hash=grid.manifest_hash,
            input_identity_sha256=grid.input_identity_sha256,
            **arrays,
            timeframe=grid.timeframe,
            sentinel_index=grid.sentinel_index,
            index_origin=grid.index_origin,
            origin_utc=grid.origin_utc,
            end_utc=grid.end_utc,
        )


def _verify_derived_marker(root: Path, prepared: BacktestPreparedArtifactSet) -> None:
    with _trusted_path(root, "manifest.yaml").open(encoding="utf-8") as handle:
        marker = yaml.safe_load(handle)
    if not isinstance(marker, dict) or marker.get("manifest_kind") != "derived_artifacts":
        raise ValueError("attempt lacks its derived-artifacts ownership marker")
    result = marker.get("result")
    request = marker.get("request")
    if not isinstance(result, dict) or not isinstance(request, dict):
        raise ValueError("invalid derived-artifacts ownership marker")
    for name in ("output_root_id", "owner_token", "attempt"):
        if result.get(name) != getattr(prepared, name) or request.get(name) != getattr(
            prepared, name
        ):
            raise ValueError("derived-artifacts attempt ownership mismatch")
    if request.get("snapshot") != prepared.snapshot.as_mapping():
        raise ValueError("derived-artifacts source snapshot mismatch")
    generated = [
        ref.as_mapping()
        for ref, provenance in zip(prepared.derivatives, prepared.provenance, strict=True)
        if provenance == "generated"
    ]
    if result.get("files") != generated:
        raise ValueError("generated file inventory differs from builder ownership marker")


def _references(
    context: BacktestArtifactRuntimeContext, role: str, timeframe: str, *, required: bool = True
) -> tuple[ArtifactInputFileReference, ...]:
    refs = tuple(
        ref
        for ref in context.references
        if ref.domain.role == role and ref.domain.timeframe == timeframe
    )
    if required and not refs:
        raise FileNotFoundError(f"input role {role!r} ({timeframe}) is not declared")
    return refs


def _load_role(context: BacktestArtifactRuntimeContext, role: str, timeframe: str) -> np.ndarray:
    refs = _references(context, role, timeframe)
    if len(refs) != 1:
        raise ValueError(f"source role {role!r} requires exactly one explicit reference")
    return _load_reference(context, refs[0])


def _load_rows(
    context: BacktestArtifactRuntimeContext,
    refs: tuple[ArtifactInputFileReference, ...],
    *,
    risk: bool,
) -> tuple[np.ndarray, tuple[int | float, ...]]:
    domain = refs[0].domain
    if any(
        (
            r.domain.timeframe,
            r.domain.origin_utc,
            r.domain.end_utc,
            r.domain.index_origin,
            r.domain.axis_order,
            r.domain.shape[1:],
        )
        != (
            domain.timeframe,
            domain.origin_utc,
            domain.end_utc,
            domain.index_origin,
            domain.axis_order,
            domain.shape[1:],
        )
        for r in refs
    ):
        raise ValueError("matrix components have incompatible logical domains")
    ids = tuple(value for ref in refs for value in (ref.risk_values if risk else ref.row_ids))
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate logical rows in explicit components")
    arrays = [_load_reference(context, ref) for ref in refs]
    expected = np.dtype(np.uint32 if risk else np.int8)
    if any(array.dtype != expected or array.ndim != 2 for array in arrays):
        raise ValueError("matrix component dtype/dimensions mismatch")
    matrix = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    if risk:
        order = np.argsort(ids)
        if not np.array_equal(order, np.arange(len(ids))):
            matrix = np.ascontiguousarray(matrix[order])
            ids = tuple(ids[int(index)] for index in order)
        if np.any(matrix > domain.shape[1]) or domain.index_origin != 0:
            raise ValueError("hit-times violate original sentinel/index domain")
    return matrix, ids


def _load_reference(
    context: BacktestArtifactRuntimeContext, ref: ArtifactInputFileReference
) -> np.ndarray:
    path = _reference_path(context, ref)
    _verify_hash(path, ref.sha256)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if isinstance(array, np.memmap):
        context.mmap_owners.append(array)
    if array.dtype != np.dtype(ref.domain.dtype) or array.shape != ref.domain.shape:
        raise ValueError(f"{path} dtype/shape differs from explicit file reference")
    role = ref.domain.role
    if role.startswith("prices."):
        ohlcv = role.endswith(".ohlcv")
        expected_dtype, ndim = np.dtype("float32" if ohlcv else "int64"), 2 if ohlcv else 1
        if ohlcv and (array.ndim != 2 or array.shape[1] != 5):
            raise ValueError("OHLCV must have five fields")
    elif role.startswith("mappings."):
        expected_dtype, ndim = np.dtype("uint32"), 1
    elif role.startswith("funding."):
        funding_dtypes = {
            "funding_time": "int64",
            "funding_rate": "float64",
            "mark_price": "float64",
            "funding_interval_minutes": "uint16",
            "data_quality": "uint8",
        }
        expected_dtype, ndim = np.dtype(funding_dtypes[role.removeprefix("funding.")]), 1
    elif role.startswith("signals."):
        expected_dtype, ndim = np.dtype("int8"), 2
    elif role.startswith("hit_times."):
        axis = role.endswith("_values")
        expected_dtype, ndim = np.dtype("float32" if axis else "uint32"), 1 if axis else 2
    else:
        raise ValueError(f"unsupported runtime array role: {role}")
    if array.dtype != expected_dtype or array.ndim != ndim:
        raise ValueError(f"{role} requires {expected_dtype.name} with {ndim} dimensions")
    snapshot = context.prepared.snapshot if context.prepared else context.source_snapshot
    if snapshot is not None and ref in snapshot.source_file_identities:
        consumed = next(
            domain
            for domain in snapshot.consumed_domains
            if domain.role == ref.domain.role
        )
        array = array[tuple(slice(0, size) for size in consumed.shape)]
    return array


def _reference_path(
    context: BacktestArtifactRuntimeContext, ref: ArtifactInputFileReference
) -> Path:
    if ref.root_id not in context.trusted_roots:
        raise ValueError("input reference has an untrusted root")
    return _trusted_path(context.trusted_roots[ref.root_id], ref.relative_path)


def _trusted_path(root: Path, relative_path: str) -> Path:
    if (
        Path(relative_path).is_absolute()
        or "\\" in relative_path
        or any(part in ("", ".", "..") for part in relative_path.split("/"))
    ):
        raise ValueError("input path must be normalized and relative")
    path = (root / relative_path).resolve(strict=True)
    if not path.is_relative_to(root.resolve(strict=True)) or not path.is_file():
        raise ValueError("input path escapes trusted root or is not a regular file")
    return path


def _verify_hash(path: Path, expected: str) -> None:
    if _file_sha256_hex(path) != expected:
        raise ValueError(f"{path} SHA-256 does not match declared metadata")


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _validate_component_source(
    source: ArtifactSlotPinnedRuntimeContextV2,
    component: object,
    timeline: ArtifactTimelineCoverageV2 | None,
) -> None:
    if (
        getattr(component, "slot") != source.artifact_slot
        or getattr(component, "slot_generation") != source.slot_generation
        or getattr(component, "asof_date") != source.artifact_asof_date
    ):
        raise ValueError("native component provenance differs from pinned root")
    if timeline is not None and getattr(component, "timeline") != timeline:
        raise ValueError("native component timeline differs from pinned root")


def copy_signal_rows_i8(
    matrix: np.ndarray,
    *,
    row_ids: np.ndarray,
    time_slice: slice,
) -> np.ndarray:
    row_ids_i32 = np.asarray(row_ids, dtype=np.int32)
    if row_ids_i32.ndim != 1 or int(row_ids_i32.size) == 0:
        raise ValueError("row_ids must be a non-empty one-dimensional array")
    if int(row_ids_i32.min()) < 0 or int(row_ids_i32.max()) >= int(matrix.shape[0]):
        raise ValueError(
            "row_ids must be within signal matrix row bounds; "
            f"got min={int(row_ids_i32.min())}, max={int(row_ids_i32.max())}, "
            f"rows={int(matrix.shape[0])}"
        )

    row_selector = _contiguous_row_selector(row_ids_i32)
    selected = matrix[row_selector, time_slice]
    return np.ascontiguousarray(np.asarray(selected, dtype=np.int8))


def _contiguous_row_selector(row_ids: np.ndarray) -> slice | np.ndarray:
    if int(row_ids.size) == 1:
        start = int(row_ids[0])
        return slice(start, start + 1)
    expected = np.arange(int(row_ids[0]), int(row_ids[-1]) + 1, dtype=np.int32)
    if int(expected.size) == int(row_ids.size) and np.array_equal(row_ids, expected):
        return slice(int(row_ids[0]), int(row_ids[-1]) + 1)
    return row_ids


__all__ = ["FilesystemBacktestArtifactArrayLoader", "copy_signal_rows_i8"]
