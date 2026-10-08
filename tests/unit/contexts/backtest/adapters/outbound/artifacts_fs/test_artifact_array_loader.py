from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound.artifacts_fs import (
    FilesystemBacktestArtifactArrayLoader,
    FilesystemBacktestArtifactContextResolver,
)
from trading.contexts.backtest.application.dto import BacktestCoordinates
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
)


def test_filesystem_artifact_array_loader_mmaps_prices_mappings_and_signals(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    price_arrays_15m = loader.load_price_arrays(context=context, timeframe="15m")
    price_arrays_1m = loader.load_price_arrays(context=context, timeframe="1m")
    mapping_arrays = loader.load_mapping_arrays(context=context, timeframe="15m")
    signal_matrix = loader.load_signal_matrix(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
    )

    assert isinstance(price_arrays_15m.open_time, np.memmap)
    assert isinstance(price_arrays_1m.ohlcv, np.memmap)
    assert isinstance(mapping_arrays.bar_open_1m_idx, np.memmap)
    assert isinstance(signal_matrix.matrix, np.memmap)
    assert price_arrays_15m.open_time.dtype == np.int64
    assert price_arrays_15m.ohlcv.dtype == np.float32
    assert mapping_arrays.bar_open_1m_idx.dtype == np.uint32
    assert signal_matrix.matrix.dtype == np.int8


def test_filesystem_artifact_array_loader_mmaps_funding_arrays(tmp_path: Path) -> None:
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="futures",
            symbol="BTCUSDT",
        ),
        include_funding=True,
        funding_coverage_status="degraded",
        funding_reason_codes=("funding_interval_gap",),
    )
    context = _resolve_context(
        store=store,
        coordinates=BacktestCoordinates(
            exchange="binance",
            market_type="futures",
            symbol="BTCUSDT",
        ),
    )
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    funding_arrays = loader.load_funding_arrays(context=context)

    assert funding_arrays.coverage_status == "degraded"
    assert funding_arrays.manifest.coverage_policy == "degraded_with_warning"
    assert isinstance(funding_arrays.funding_time, np.memmap)
    assert funding_arrays.funding_time.dtype == np.int64
    assert funding_arrays.funding_rate.dtype == np.float64
    assert funding_arrays.mark_price.dtype == np.float64
    assert funding_arrays.funding_interval_minutes.dtype == np.uint16
    assert funding_arrays.data_quality.dtype == np.uint8


def test_filesystem_artifact_array_loader_copies_contiguous_and_non_contiguous_rows(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    contiguous = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_ids=np.asarray([0, 1], dtype=np.int32),
        time_slice=slice(0, 2),
    )
    non_contiguous = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_ids=np.asarray([1, 0], dtype=np.int32),
        time_slice=slice(0, 2),
    )

    assert contiguous.flags.c_contiguous
    assert non_contiguous.flags.c_contiguous
    assert contiguous.tolist() == [[-1, 0], [1, 0]]
    assert non_contiguous.tolist() == [[1, 0], [-1, 0]]


def test_filesystem_artifact_array_loader_reports_missing_price_artifact(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    store.builder.price_paths(store.coordinates, store.active_slot, "15m").ohlcv.unlink()
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    with pytest.raises(FileNotFoundError):
        loader.load_price_arrays(context=context, timeframe="15m")


def test_filesystem_artifact_array_loader_reports_missing_mapping_artifact(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    store.builder.mapping_paths(
        store.coordinates,
        store.active_slot,
        "15m",
    ).bar_open_1m_idx.unlink()
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    with pytest.raises(FileNotFoundError):
        loader.load_mapping_arrays(context=context, timeframe="15m")


def test_filesystem_artifact_array_loader_reports_missing_signal_artifact(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    store.builder.signal_paths(
        store.coordinates,
        store.active_slot,
        "15m",
        "ma.ema",
    ).signals.unlink()
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    with pytest.raises(FileNotFoundError):
        loader.load_signal_matrix(
            context=context,
            timeframe="15m",
            indicator_id="ma.ema",
        )


def _resolve_context(
    *,
    store: Any,
    coordinates: BacktestCoordinates | None = None,
):
    effective_coordinates = (
        BacktestCoordinates(
            exchange="binance",
            market_type="spot",
            symbol="BTCUSDT",
        )
        if coordinates is None
        else coordinates
    )
    resolver = FilesystemBacktestArtifactContextResolver(artifact_loader=store.loader)
    metadata = resolver.resolve_context(coordinates=effective_coordinates)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    return loader.resolve_context(
        coordinates=effective_coordinates,
        artifact_metadata=metadata,
    )


def test_explicit_references_ignore_slot_path_resolvers(tmp_path: Path, monkeypatch) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)

    def unexpected(*args, **kwargs):
        raise AssertionError("array loads must not reconstruct slot-derived paths")

    for name in (
        "resolve_price_paths",
        "resolve_mapping_paths",
        "resolve_signal_paths",
        "resolve_funding_paths",
        "resolve_hit_times_paths",
    ):
        monkeypatch.setattr(type(store.loader), name, unexpected)
    assert loader.load_price_arrays(context=context, timeframe="15m").ohlcv.shape[1] == 5
    assert loader.load_mapping_arrays(context=context, timeframe="15m").bar_open_1m_idx.size
    assert loader.load_signal_matrix(
        context=context, timeframe="15m", indicator_id="ma.ema"
    ).row_ids
    assert loader.load_hit_times_table_arrays(context=context).long_tp.size


@pytest.mark.parametrize("row_ids", [(8, 3), (100, 20, 37)])
def test_sparse_signal_ids_are_not_physical_positions(tmp_path: Path, row_ids) -> None:
    from dataclasses import replace

    from trading.contexts.backtest.application.dto.artifact_inputs import BacktestSignalMetadata

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    original = next(ref for ref in context.references if ref.domain.role == "signals.ma.ema")
    root = tmp_path / "attempt"
    root.mkdir()
    matrix = np.asarray([[1, -1, 0, 1], [-1, 1, 1, 0], [0, -1, 1, 0]], dtype=np.int8)
    matrix = matrix[:len(row_ids)]
    np.save(root / "selected.npy", matrix)
    ref = _written_reference(original, root, "selected.npy", row_ids=row_ids)
    context = replace(
        context,
        references=(ref,),
        trusted_roots={"attempt": root},
        signal_metadata={("15m", "ma.ema"): BacktestSignalMetadata(max(row_ids) + 1)},
        signal_manifests={},
    )
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    loaded = loader.load_signal_matrix(context=context, timeframe="15m", indicator_id="ma.ema")
    assert loaded.row_ids == row_ids
    assert loaded.canonical_rows_count == max(row_ids) + 1
    assert loaded.manifest is None
    selected = loader.load_signal_rows(
        context=context,
        timeframe="15m",
        indicator_id="ma.ema",
        row_ids=np.asarray(row_ids[::-1], dtype=np.int32),
        time_slice=slice(1, 3),
    )
    np.testing.assert_array_equal(selected, matrix[::-1, 1:3])
    with pytest.raises(ValueError, match="canonical signal row is missing"):
        loader.load_signal_rows(
            context=context,
            timeframe="15m",
            indicator_id="ma.ema",
            row_ids=np.asarray([0]),
            time_slice=slice(None),
        )


def test_mixed_signal_components_keep_ids_and_reject_overlap(tmp_path: Path) -> None:
    from dataclasses import replace

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    original = next(ref for ref in context.references if ref.domain.role == "signals.ma.ema")
    root = tmp_path / "attempt"
    root.mkdir()
    matrix = np.load(context.trusted_roots[original.root_id] / original.relative_path)
    np.save(root / "extra.npy", -matrix[:1])
    extra = _written_reference(original, root, "extra.npy", row_ids=(7,))
    context = replace(
        context,
        references=(original, extra),
        trusted_roots={**context.trusted_roots, "attempt": root},
    )
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    loaded = loader.load_signal_matrix(context=context, timeframe="15m", indicator_id="ma.ema")
    assert loaded.row_ids == (*original.row_ids, 7)
    np.testing.assert_array_equal(loaded.matrix[-1], -matrix[0])
    overlapping = replace(context, references=(original, replace(extra, row_ids=(0,))))
    with pytest.raises(ValueError, match="duplicate logical rows"):
        loader.load_signal_matrix(context=overlapping, timeframe="15m", indicator_id="ma.ema")


@pytest.mark.parametrize("mode", ["native", "generated", "mixed"])
def test_risk_components_share_one_loader_without_fake_manifest(tmp_path: Path, mode: str) -> None:
    from dataclasses import replace

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    expected = loader.load_hit_times_table_arrays(context=context)
    root = tmp_path / "attempt"
    root.mkdir()
    refs = []
    for ref in context.references:
        if not ref.domain.role.startswith("hit_times.") or mode == "native":
            refs.append(ref)
            continue
        array = np.load(context.trusted_roots[ref.root_id] / ref.relative_path)
        # Separate actual files for each risk level. Mixed uses two trusted roots.
        for index in range(array.shape[0]):
            relative = f"{ref.domain.role}-{index}.npy"
            output = context.trusted_roots[ref.root_id] if mode == "mixed" and index == 0 else root
            np.save(output / relative, array[index : index + 1])
            written = _written_reference(
                ref, output, relative, risk_values=(ref.risk_values[index],)
            )
            if output != root:
                written = replace(written, root_id=ref.root_id)
            refs.append(written)
    context = replace(
        context,
        references=tuple(refs),
        trusted_roots={**context.trusted_roots, "attempt": root},
        hit_times_manifest=None if mode != "native" else context.hit_times_manifest,
        hit_times_manifest_hash=None if mode != "native" else context.hit_times_manifest_hash,
    )
    actual = loader.load_hit_times_table_arrays(context=context)
    grid = loader.load_hit_times_grid_arrays(context=context)
    for name in ("long_tp", "long_sl", "short_tp", "short_sl"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    np.testing.assert_array_equal(grid.tp_values, np.asarray([0.01, 0.02], dtype=np.float32))
    assert (actual.manifest is None) == (mode != "native")
    assert actual.manifest_hash == context.hit_times_manifest_hash
    assert grid.manifest_hash == context.hit_times_manifest_hash
    assert actual.input_identity_sha256 == grid.input_identity_sha256
    assert len(actual.input_identity_sha256) == 64
    assert actual.input_identity_sha256 != actual.manifest_hash
    from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
        BacktestTpSlHitTimesService,
    )

    result = BacktestTpSlHitTimesService(loader).execute(
        normalized_request={
            "risk": {
                "mode": "tp_sl_grid",
                "tp": {"start_pct": 1.0, "stop_pct": 2.0, "step_pct": 1.0},
                "sl": {"start_pct": 1.0, "stop_pct": 2.0, "step_pct": 1.0},
            }
        },
        context=context,
    )
    assert result is not None
    assert result.hit_times_manifest_hash == context.hit_times_manifest_hash
    assert result.hit_times_input_sha256 == grid.input_identity_sha256
    assert result.compact_mapping()["hit_times_input_sha256"] == grid.input_identity_sha256
    assert isinstance(loader.load_price_arrays(context=context, timeframe="15m").ohlcv, np.memmap)


def test_loader_rejects_corruption_and_symlink_escape(tmp_path: Path) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    ref = next(ref for ref in context.references if ref.domain.role == "prices.15m.ohlcv")
    path = context.trusted_roots[ref.root_id] / ref.relative_path
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        loader.load_price_arrays(context=context, timeframe="15m")
    outside = tmp_path / "outside.npy"
    outside.write_bytes(original)
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes trusted root"):
        loader.load_price_arrays(context=context, timeframe="15m")


def test_unknown_root_is_rejected_at_binding(tmp_path: Path) -> None:
    from dataclasses import replace

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    with pytest.raises(ValueError, match="untrusted root"):
        replace(context, references=(replace(context.references[0], root_id="untrusted"),))


def _written_reference(original, root, relative, *, row_ids=(), risk_values=()):
    import hashlib
    from dataclasses import replace

    path = root / relative
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return replace(
        original,
        root_id="attempt",
        relative_path=relative,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        domain=replace(original.domain, shape=array.shape, row_count=array.shape[0]),
        row_ids=row_ids,
        risk_values=risk_values,
    )


def test_prepared_publication_is_atomic_and_requires_exact_acknowledgement(tmp_path: Path) -> None:
    from dataclasses import replace

    import yaml

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    prepared = _native_prepared(store, context)
    output = tmp_path / "attempt"
    path = loader.write_prepared_manifest(prepared=prepared, output_directory=output)
    assert yaml.safe_load(path.read_text()) == prepared.as_mapping()
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        loader.write_prepared_manifest(prepared=prepared, output_directory=output)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="acknowledged"):
        loader.with_prepared_inputs(context=context, prepared=prepared, output_directory=output)
    bound = loader.with_prepared_inputs(
        context=context,
        prepared=prepared,
        output_directory=output,
        acknowledged_prepared_sha256=prepared.content_sha256,
    )
    assert bound.source is context.source
    assert not any(ref.domain.role.startswith("hit_times.") for ref in bound.references)
    assert isinstance(loader.load_price_arrays(context=bound, timeframe="15m").ohlcv, np.memmap)
    with pytest.raises(ValueError, match="acknowledged content"):
        loader.with_prepared_inputs(
            context=context,
            prepared=replace(prepared, owner_token="different-owner"),
            output_directory=output,
            acknowledged_prepared_sha256=prepared.content_sha256,
        )


def test_all_native_publication_preserves_occupied_directory(tmp_path: Path) -> None:
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    prepared = _native_prepared(store, context)
    root = tmp_path / "occupied"
    root.mkdir()
    foreign = root / "foreign.txt"
    foreign.write_text("foreign")
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    with pytest.raises(ValueError, match="occupied"):
        loader.write_prepared_manifest(prepared=prepared, output_directory=root)
    assert foreign.read_text() == "foreign"
    assert not (root / "prepared-inputs.yaml").exists()


def test_forged_attempt_directory_cannot_bind_generated_inputs(tmp_path: Path) -> None:
    from dataclasses import replace

    import yaml

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    prepared = _native_prepared(store, context)
    original = next(ref for ref in context.references if ref.domain.role == "signals.ma.ema")
    root = tmp_path / "forged"
    root.mkdir()
    source = context.trusted_roots[original.root_id] / original.relative_path
    (root / "signals.npy").write_bytes(source.read_bytes())
    generated = _written_reference(original, root, "signals.npy", row_ids=original.row_ids)
    prepared = replace(prepared, derivatives=(generated,), provenance=("generated",))
    (root / "prepared-inputs.yaml").write_text(yaml.safe_dump(prepared.as_mapping()))
    # Even an exact copied ready document is not proof of this builder attempt's ownership.
    (root / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "manifest_kind": "derived_artifacts",
                "result": {"owner_token": "foreign"},
                "request": {},
            }
        )
    )
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    with pytest.raises(ValueError, match="ownership mismatch"):
        loader.with_prepared_inputs(
            context=context,
            prepared=prepared,
            output_directory=root,
            acknowledged_prepared_sha256=prepared.content_sha256,
        )


def _native_prepared(store, context):
    from trading.contexts.backtest_artifacts.application.services.v2 import (
        artifact_manifest_validator,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        ArtifactCandleSnapshot,
        BacktestPreparedArtifactSet,
    )

    refs = tuple(
        ref for ref in context.references if ref.domain.role.startswith(("prices.", "mappings."))
    )
    source = context.source
    snapshot = ArtifactCandleSnapshot(
        coordinates=source.coordinates,
        source_schema=source.slot_manifest.schema_version,
        slot=source.artifact_slot,
        generation=source.slot_generation,
        manifest_sha256=source.artifact_manifest_hash,
        signal_timeframe="15m",
        execution_timeframe="1m",
        source_file_identities=refs,
        consumed_domains=tuple(ref.domain for ref in refs),
    )
    snapshot = artifact_manifest_validator.BacktestArtifactManifestValidatorV2(
        store.loader
    ).attest_snapshot(
        snapshot=snapshot,
        trusted_roots=context.trusted_roots,
    )
    return BacktestPreparedArtifactSet(
        snapshot=snapshot,
        recipe_sha256="a" * 64,
        organization_id="00000000-0000-0000-0000-000000000001",
        job_id="00000000-0000-0000-0000-000000000002",
        owner_token="owned-test",
        attempt=1,
        output_root_id="attempt",
        derivatives=(),
        provenance=(),
    )


@pytest.mark.parametrize(
    "role,dtype",
    [
        ("prices.15m.open_time", "float64"),
        ("prices.15m.ohlcv", "float64"),
        ("mappings.15m.bar_open_1m_idx", "int64"),
        ("funding.funding_rate", "float32"),
    ],
)
def test_loader_rejects_self_consistent_wrong_semantic_dtype(tmp_path: Path, role, dtype) -> None:
    import hashlib
    from dataclasses import replace

    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        coordinates=ArtifactCoordinatesV2(
            exchange="binance", market_type="futures", symbol="BTCUSDT"
        ),
        include_funding=True,
    )
    context = _resolve_context(
        store=store,
        coordinates=BacktestCoordinates(
            exchange="binance", market_type="futures", symbol="BTCUSDT"
        ),
    )
    ref = next(ref for ref in context.references if ref.domain.role == role)
    path = context.trusted_roots[ref.root_id] / ref.relative_path
    array = np.asarray(np.load(path), dtype=dtype)
    np.save(path, array)
    changed = replace(
        ref,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        domain=replace(ref.domain, dtype=array.dtype.str),
    )
    context = replace(
        context, references=tuple(changed if item == ref else item for item in context.references)
    )
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    with pytest.raises(ValueError, match="requires"):
        if role.startswith("prices."):
            loader.load_price_arrays(context=context, timeframe="15m")
        elif role.startswith("mappings."):
            loader.load_mapping_arrays(context=context, timeframe="15m")
        else:
            loader.load_funding_arrays(context=context)


def test_prepared_source_returns_recorded_prefix_mmap_view(tmp_path: Path) -> None:
    from dataclasses import replace
    from datetime import UTC, datetime

    from trading.contexts.backtest_artifacts.application.services.v2 import (
        artifact_manifest_validator,
    )
    from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
        ArtifactPrefixProof,
    )

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    full = loader.load_price_arrays(context=context, timeframe="15m")
    prepared = _native_prepared(store, context)
    end = (
        datetime.fromtimestamp(int(full.close_time[0]) / 1000, tz=UTC)
        .isoformat()
        .replace("+00:00", "Z")
    )
    domains = tuple(
        replace(domain, row_count=1, shape=(1, *domain.shape[1:]), end_utc=end)
        if domain.timeframe == "15m" and domain.axis_order[0] == "time"
        else domain
        for domain in prepared.snapshot.consumed_domains
    )
    snapshot = replace(
        prepared.snapshot, consumed_domains=domains, prefix_proof=ArtifactPrefixProof()
    )
    snapshot = artifact_manifest_validator.BacktestArtifactManifestValidatorV2(
        store.loader
    ).attest_snapshot(
        snapshot=snapshot,
        trusted_roots=context.trusted_roots,
    )
    prepared = replace(prepared, snapshot=snapshot)
    output = tmp_path / "prefix-attempt"
    loader.write_prepared_manifest(prepared=prepared, output_directory=output)
    bound = loader.with_prepared_inputs(
        context=context,
        prepared=prepared,
        output_directory=output,
        acknowledged_prepared_sha256=prepared.content_sha256,
    )
    old = loader.load_price_arrays(context=bound, timeframe="15m")
    assert full.ohlcv.shape[0] > old.ohlcv.shape[0] == 1
    assert isinstance(old.ohlcv, np.memmap)
    assert isinstance(full.ohlcv, np.memmap)
    assert old.ohlcv.filename == full.ohlcv.filename
    np.testing.assert_array_equal(old.ohlcv, full.ohlcv[:1])
    assert loader.load_mapping_arrays(context=bound, timeframe="15m").bar_open_1m_idx.shape == (1,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("organization_id", "00000000-0000-0000-0000-000000000003"),
        ("job_id", "00000000-0000-0000-0000-000000000004"),
        ("owner_token", "other-owner"),
        ("attempt", 2),
    ],
)
def test_acknowledgement_is_bound_to_full_attempt_identity(tmp_path: Path, field, value) -> None:
    from dataclasses import replace

    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    context = _resolve_context(store=store)
    prepared = _native_prepared(store, context)
    changed = replace(prepared, **{field: value})
    assert changed.recipe_sha256 == prepared.recipe_sha256
    loader = FilesystemBacktestArtifactArrayLoader(artifact_loader=store.loader)
    output = tmp_path / "attempt"
    loader.write_prepared_manifest(prepared=changed, output_directory=output)
    with pytest.raises(ValueError, match="acknowledged content"):
        loader.with_prepared_inputs(
            context=context,
            prepared=changed,
            output_directory=output,
            acknowledged_prepared_sha256=prepared.content_sha256,
        )
