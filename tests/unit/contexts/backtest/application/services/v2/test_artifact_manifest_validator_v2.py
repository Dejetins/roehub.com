from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from shutil import copytree

import numpy as np
import pytest
import yaml

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_validator import (  # noqa: E501
    BacktestArtifactManifestValidatorV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    SIGNAL_FEATURE_NAMES_V2,
    ArtifactCandleSnapshot,
    ArtifactCoordinatesV2,
    ArtifactDerivativeBuildRequest,
    ArtifactInputFileReference,
    ArtifactPrefixDomain,
    ArtifactPrefixProof,
    ArtifactRequestedRow,
    BacktestPreparedArtifactSet,
)


def test_backtest_artifact_manifest_validator_v2_accepts_valid_strict_slot(
    tmp_path: Path,
) -> None:
    """
    Verify strict validator accepts a fully valid inactive slot and exposes typed metadata.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Valid slot manifests and arrays should produce no diagnostics before publish switch.
    Raises:
        AssertionError: If a valid strict slot is rejected.
    Side Effects:
        Creates and reads a synthetic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert result.slot_manifest is not None
    assert result.slot_manifest.slot_generation == 5
    assert result.manifest_sha256 is not None
    assert len(result.signal_manifests) == 1
    assert result.signal_manifests[0].indicator_id == "ma.ema"
    assert result.signal_manifests[0].signal_features is not None
    assert result.hit_times_manifest is not None
    assert result.hit_times_manifest.timeline_bar_count == 2
    assert result.diagnostics == ()


def test_backtest_artifact_manifest_validator_v2_accepts_futures_funding(
    tmp_path: Path,
) -> None:
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
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert result.slot_manifest is not None
    assert result.slot_manifest.funding is not None
    assert result.slot_manifest.funding.coverage_status == "degraded"
    assert result.slot_manifest.funding.coverage_policy == "degraded_with_warning"
    assert result.diagnostics == ()


def test_backtest_artifact_manifest_validator_v2_requires_futures_funding(
    tmp_path: Path,
) -> None:
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        coordinates=ArtifactCoordinatesV2(
            exchange="binance",
            market_type="futures",
            symbol="BTCUSDT",
        ),
    )
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert "funding_manifest_missing_for_futures" in tuple(
        diagnostic.code for diagnostic in result.diagnostics
    )


def test_backtest_artifact_manifest_validator_v2_accepts_legacy_slot_without_signal_features(
    tmp_path: Path,
) -> None:
    """
    Verify strict validation keeps accepting old slots that omit additive signal features.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `signal_features` remains optional until a later runtime rollout actually requires it.
    Raises:
        AssertionError: If legacy slots are rejected.
    Side Effects:
        Creates and reads one synthetic legacy-style artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/roadmap/backtest-runtime-acceleration-plan-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        inactive_include_signal_features=False,
    )
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert result.slot_manifest is not None
    assert len(result.signal_manifests) == 1
    assert result.signal_manifests[0].signal_features is None
    assert result.diagnostics == ()


def test_backtest_artifact_manifest_validator_v2_rejects_root_manifest_schema_drift(
    tmp_path: Path,
) -> None:
    """
    Verify strict validator rejects root manifests with unsupported extra keys.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Root manifest schema drift must fail before any deeper array validation.
    Raises:
        AssertionError: If unsupported extra keys are accepted.
    Side Effects:
        Creates and mutates a synthetic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    root_manifest_path = store.builder.slot_manifest_path(store.coordinates, store.inactive_slot)
    payload = yaml.safe_load(root_manifest_path.read_text(encoding="utf-8"))
    payload["unexpected"] = "drift"
    root_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)
    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert result.slot_manifest is None
    assert result.manifest_sha256 is None
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "root_manifest_invalid"
    assert "unexpected keys" in result.diagnostics[0].message


def test_backtest_artifact_manifest_validator_v2_rejects_mapping_price_correspondence_mismatch(
    tmp_path: Path,
) -> None:
    """
    Verify validator reports a stable diagnostic when mapping indexes no longer match price
    timelines.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Bounds and monotonicity may still be valid while exact `prices/<tf>` correspondence fails.
    Raises:
        AssertionError: If correspondence drift is not reported with the documented error code.
    Side Effects:
        Creates and reads a synthetic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        inactive_mapping_open_idx=np.array([1, 2], dtype=np.uint32),
    )
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert tuple(diagnostic.code for diagnostic in result.diagnostics) == (
        "mapping_open_time_correspondence_mismatch",
    )


def test_backtest_artifact_manifest_validator_v2_orders_multiple_diagnostics_deterministically(
    tmp_path: Path,
) -> None:
    """
    Verify validator emits stable diagnostics ordering across multiple simultaneous violations.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Diagnostics ordering must stay deterministic by artifact family and validation stage.
    Raises:
        AssertionError: If diagnostics order is unstable or misses expected violations.
    Side Effects:
        Creates and reads synthetic invalid artifact trees under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        inactive_signal_values=np.array([[-1, 2], [1, 0]], dtype=np.int8),
        inactive_mapping_close_idx=np.array([1, 4], dtype=np.uint32),
        inactive_long_tp=np.array([[1, 2], [0, 2]], dtype=np.uint32),
    )
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    first_result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )
    second_result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    first_codes = tuple(diagnostic.code for diagnostic in first_result.diagnostics)
    second_codes = tuple(diagnostic.code for diagnostic in second_result.diagnostics)

    assert first_codes == second_codes
    assert first_codes == (
        "mapping_close_indexes_out_of_bounds",
        "signal_values_out_of_set",
        "hit_times_table_not_monotone",
    )


def test_backtest_artifact_manifest_validator_v2_rejects_signal_manifest_reference_hash_drift(
    tmp_path: Path,
) -> None:
    """
    Verify validator reports root-catalog drift when one signal manifest hash no longer matches.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Signal manifest file remains valid while only the root reference hash is corrupted.
    Raises:
        AssertionError: If hash drift is not reported with the documented diagnostic code.
    Side Effects:
        Creates and mutates a synthetic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/README.md
      - docs/architecture/backtest/README.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    root_manifest_path = store.builder.slot_manifest_path(store.coordinates, store.inactive_slot)
    payload = yaml.safe_load(root_manifest_path.read_text(encoding="utf-8"))
    payload["signals"]["manifests"][0]["manifest_sha256"] = "0" * 64
    root_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert tuple(diagnostic.code for diagnostic in result.diagnostics) == (
        "signal_manifest_reference_hash_mismatch",
    )


def _prepared_fixture(tmp_path):
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    validator = BacktestArtifactManifestValidatorV2(store.loader)
    manifest = store.loader.load_slot_manifest(store.coordinates, store.inactive_slot)
    root = manifest.path.parent
    # Keep this legacy tiny fixture on canonical whole-second timestamp edges.
    payload = yaml.safe_load(manifest.path.read_text())
    def normalize(value):
        if isinstance(value, dict):
            if value.get("dtype") == "int64" and "path" in value:
                path = root / value["path"]
                np.save(path, np.load(path) * 1000)
                value["sha256"] = sha256(path.read_bytes()).hexdigest()
            for key, child in value.items():
                if key in (
                    "open_time_start", "open_time_end", "close_time_start", "close_time_end",
                ):
                    value[key] = child * 1000
                else:
                    normalize(child)
        elif isinstance(value, list):
            for child in value:
                normalize(child)
    normalize(payload)
    manifest.path.write_text(yaml.safe_dump(payload))
    manifest = store.loader.load_slot_manifest(store.coordinates, store.inactive_slot)
    refs = []
    for price in manifest.prices:
        for name in ("open_time", "close_time", "ohlcv"):
            metadata = getattr(price, name)
            refs.append(_source_ref(root, metadata, f"prices.{price.timeframe}.{name}", price))
    for mapping in manifest.mappings:
        price = next(p for p in manifest.prices if p.timeframe == mapping.timeframe)
        for name in ("bar_open_1m_idx", "bar_close_1m_idx"):
            refs.append(_source_ref(root, getattr(mapping, name),
                                    f"mappings.{mapping.timeframe}.{name}", price))
    snapshot = ArtifactCandleSnapshot(
        store.coordinates, manifest.schema_version, store.inactive_slot,
        manifest.slot_generation, sha256(manifest.path.read_bytes()).hexdigest(),
        "15m", "1m", tuple(refs), tuple(ref.domain for ref in refs),
    )
    roots = {"base": root, "attempt": tmp_path / "attempt"}
    roots["attempt"].mkdir()
    request = ArtifactDerivativeBuildRequest(
        snapshot, (ArtifactRequestedRow("ma.ema", "close", 7, (("window", 20),)),),
        "signals/v1", "a" * 64, "numba/v1", "f32-f64/v1", (), (),
        "attempt", "owner", 1, 100000, 100000,
    )
    snapshot = validator.attest_snapshot(snapshot=snapshot, trusted_roots=roots)
    prepared = BacktestPreparedArtifactSet(
        snapshot, "b" * 64, "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002", "owner", 1, "attempt", (), (),
    )
    return validator, roots, request, prepared


def _source_ref(root, metadata, role, price):
    array = np.load(root / metadata.path, mmap_mode="r", allow_pickle=False)
    return ArtifactInputFileReference(
        "base", metadata.path, metadata.sha256,
        ArtifactPrefixDomain(
            role, price.timeframe, array.dtype.str, metadata.axis_order,
            datetime.fromtimestamp(
                price.coverage.open_time_start / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            datetime.fromtimestamp(
                price.coverage.close_time_end / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            array.shape[0], tuple(array.shape),
        ),
    )


def _derivative(roots, prepared, role, array, *, rows=(), levels=(), root_id="attempt", name=None):
    timeline = next(d for d in prepared.snapshot.consumed_domains
                    if d.role == "prices.15m.open_time")
    name = name or f"{role}-{len(list(roots[root_id].glob('*.npy')))}.npy"
    path = roots[root_id] / name
    np.save(path, array, allow_pickle=False)
    if role.startswith("signals."):
        axes = ("variant", "time")
    elif role.startswith("signal_features."):
        axes = ("variant", "feature")
    else:
        axes = ("level",) if array.ndim == 1 else ("level", "time")
    return ArtifactInputFileReference(
        root_id, name, sha256(path.read_bytes()).hexdigest(),
        replace(timeline, role=role, dtype=array.dtype.str, axis_order=axes,
                shape=array.shape, row_count=array.shape[0]), rows, levels,
    )


def _with_refs(prepared, *refs):
    return replace(prepared, derivatives=tuple(refs), provenance=tuple(
        "generated" if ref.root_id == "attempt" else "reused" for ref in refs))


def test_prepared_accepts_mixed_sparse_rows_levels_and_optional_features(tmp_path):
    validator, roots, request, prepared = _prepared_fixture(tmp_path)
    request = replace(request, tp_levels_pct=(0.5, 2.0))
    refs = [_derivative(roots, prepared, "signals.ma.ema",
                        np.zeros((1, 2), dtype=np.int8), rows=(7,))]
    refs.append(_derivative(roots, prepared, "signals.ma.ema",
                            np.ones((1, 2), dtype=np.int8), rows=(2,), root_id="base"))
    for level, root_id in ((0.5, "base"), (2.0, "attempt")):
        for name in ("tp_values", "long_tp", "short_tp"):
            array = (np.asarray([level / 100], dtype=np.float32) if name.endswith("values")
                     else np.full((1, 2), 2, dtype=np.uint32))
            refs.append(_derivative(roots, prepared, f"hit_times.{name}", array,
                                    levels=(level,), root_id=root_id))
    before = {ref.relative_path: (roots["base"] / ref.relative_path).read_bytes()
              for ref in prepared.snapshot.source_file_identities}
    validator.validate_prepared_inputs(prepared=_with_refs(prepared, *refs), request=request,
                                       trusted_roots=roots)
    assert before == {path: (roots["base"] / path).read_bytes() for path in before}
    loaded = validator.load_validated_reference(reference=refs[0], trusted_roots=roots)
    assert isinstance(loaded, np.memmap)
    assert not loaded.flags.writeable


@pytest.mark.parametrize("problem", ["row", "level", "feature", "signal", "sentinel",
                                     "dtype", "shape", "timeline", "duplicate", "unknown"])
def test_prepared_rejects_incomplete_or_invalid_declared_files(tmp_path, problem):
    validator, roots, request, prepared = _prepared_fixture(tmp_path)
    signal = _derivative(roots, prepared, "signals.ma.ema",
                         np.zeros((1, 2), dtype=np.int8), rows=(7,))
    refs = [signal]
    if problem == "row":
        refs = [replace(signal, row_ids=(8,))]
    elif problem == "level":
        request = replace(request, sl_levels_pct=(1.0,))
    elif problem == "feature":
        refs.append(_derivative(roots, prepared, "signal_features.ma.ema",
                                np.full((1, len(SIGNAL_FEATURE_NAMES_V2)), np.nan,
                                        dtype=np.float32), rows=(7,)))
    elif problem in ("signal", "dtype", "shape"):
        array = np.full((1, 3 if problem == "shape" else 2),
                        2 if problem == "signal" else 0,
                        dtype=np.int16 if problem == "dtype" else np.int8)
        if problem == "dtype":
            # Supported physical dtype but wrong semantic signal dtype.
            array = array.astype(np.float32)
        refs = [_derivative(roots, prepared, "signals.ma.ema", array, rows=(7,))]
    elif problem == "sentinel":
        refs.append(_derivative(roots, prepared, "hit_times.long_tp",
                                np.full((1, 2), 3, dtype=np.uint32), levels=(1.0,)))
    elif problem == "timeline":
        refs = [replace(signal, domain=replace(signal.domain, index_origin=1))]
    elif problem == "duplicate":
        refs.append(_derivative(roots, prepared, "signals.ma.ema",
                                np.ones((1, 2), dtype=np.int8), rows=(7,)))
    else:
        refs = [replace(signal, domain=replace(signal.domain, role="unknown"))]
    with pytest.raises(ValueError):
        validator.validate_prepared_inputs(prepared=_with_refs(prepared, *refs), request=request,
                                           trusted_roots=roots)


@pytest.mark.parametrize("problem", ["sha", "symlink", "root", "missing", "physical_shape"])
def test_declared_file_fail_closed_before_completeness(tmp_path, problem):
    validator, roots, _, prepared = _prepared_fixture(tmp_path)
    ref = _derivative(roots, prepared, "signals.ma.ema", np.zeros((1, 2), dtype=np.int8),
                      rows=(7,))
    path = roots["attempt"] / ref.relative_path
    if problem == "sha":
        path.write_bytes(b"not-npy")
    elif problem == "symlink":
        outside = tmp_path / "outside.npy"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif problem == "root":
        roots = {"base": roots["base"]}
    elif problem == "missing":
        path.unlink()
    else:
        np.save(path, np.zeros((1, 3), dtype=np.int8))
        ref = replace(ref, sha256=sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        validator.validate_declared_files(snapshot=prepared.snapshot, references=(ref,),
                                          trusted_roots=roots)


def test_attestation_rejects_changed_recorded_payload_with_current_file_sha(tmp_path):
    validator, roots, _, prepared = _prepared_fixture(tmp_path)
    snapshot = prepared.snapshot
    target = next(r for r in snapshot.source_file_identities if r.domain.role == "prices.15m.ohlcv")
    path = roots["base"] / target.relative_path
    array = np.load(path)
    array[0, 0] += 1
    np.save(path, array)
    # A current manifest/file SHA cannot substitute for the persisted old prefix proof.
    manifest_path = roots["base"] / "manifest.yaml"
    payload = yaml.safe_load(manifest_path.read_text())
    def update(value):
        if isinstance(value, dict):
            if value.get("path") == target.relative_path:
                value["sha256"] = sha256(path.read_bytes()).hexdigest()
            for child in value.values():
                update(child)
        elif isinstance(value, list):
            for child in value:
                update(child)
    update(payload)
    manifest_path.write_text(yaml.safe_dump(payload))
    snapshot = replace(snapshot, manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
                       source_file_identities=tuple(replace(r, sha256=sha256(path.read_bytes())
                       .hexdigest()) if r == target else r
                       for r in snapshot.source_file_identities))
    with pytest.raises(ValueError, match="prefix proof"):
        validator.attest_snapshot(snapshot=snapshot, trusted_roots=roots)


@pytest.mark.parametrize("problem", ["manifest", "role", "mapping", "proof", "version", "owner"])
def test_prepared_rejects_source_and_request_identity_mismatch(tmp_path, problem):
    validator, roots, request, prepared = _prepared_fixture(tmp_path)
    snapshot = prepared.snapshot
    if problem == "manifest":
        snapshot = replace(snapshot, manifest_sha256="c" * 64)
    elif problem == "role":
        refs = tuple(r for r in snapshot.source_file_identities
                     if not r.domain.role.startswith("mappings."))
        snapshot = replace(snapshot, source_file_identities=refs,
                           consumed_domains=tuple(r.domain for r in refs),
                           prefix_proof=ArtifactPrefixProof())
    elif problem == "mapping":
        ref = next(r for r in snapshot.source_file_identities
                   if r.domain.role.endswith("bar_open_1m_idx"))
        snapshot = replace(snapshot, source_file_identities=tuple(
            replace(r, sha256="c" * 64) if r == ref else r
            for r in snapshot.source_file_identities))
    elif problem == "proof":
        proof = snapshot.prefix_proof
        digests = tuple("c" * 64 for _ in proof.role_digests)
        snapshot = replace(snapshot, prefix_proof=ArtifactPrefixProof(
            "verified", proof.domains, digests,
            ArtifactPrefixProof.combined_digest(proof.domains, digests)))
    elif problem == "version":
        request = replace(request, compute_version="unknown")
    else:
        request = replace(request, owner_token="another-owner")
    with pytest.raises(ValueError):
        if problem in ("version", "owner"):
            validator.validate_prepared_inputs(prepared=prepared, request=request,
                                               trusted_roots=roots)
        else:
            validator.attest_snapshot(snapshot=snapshot, trusted_roots=roots)


def test_attest_preserves_recorded_prefix_when_unconsumed_payload_changes(tmp_path):
    validator, roots, request, _ = _prepared_fixture(tmp_path)
    snapshot = request.snapshot
    close_ref = next(r for r in snapshot.source_file_identities
                     if r.domain.role == "prices.15m.close_time")
    closed = np.load(roots["base"] / close_ref.relative_path)
    end = datetime.fromtimestamp(int(closed[0]) / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    domains = tuple(replace(d, shape=(1, *d.shape[1:]), row_count=1, end_utc=end)
                    if d.timeframe == "15m" else d for d in snapshot.consumed_domains)
    snapshot = replace(snapshot, consumed_domains=domains)
    attested = validator.attest_snapshot(snapshot=snapshot, trusted_roots=roots)
    target = next(r for r in snapshot.source_file_identities if r.domain.role == "prices.15m.ohlcv")
    path = roots["base"] / target.relative_path
    values = np.load(path)
    values[1, 0] += 1
    np.save(path, values)
    digest = sha256(path.read_bytes()).hexdigest()
    manifest_path = roots["base"] / "manifest.yaml"
    # Only the current physical identity changes. Original consumed descriptors stay fixed.
    text = manifest_path.read_text().replace(target.sha256, digest)
    manifest_path.write_text(text)
    replay = replace(attested,
        manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        source_file_identities=tuple(replace(r, sha256=digest) if r == target else r
                                     for r in snapshot.source_file_identities))
    assert validator.attest_snapshot(snapshot=replay, trusted_roots=roots).prefix_proof == (
        attested.prefix_proof)


def test_declared_mixed_risk_monotonicity_is_checked_before_build(tmp_path):
    validator, roots, _, prepared = _prepared_fixture(tmp_path)
    low = _derivative(roots, prepared, "hit_times.long_tp",
                      np.full((1, 2), 2, dtype=np.uint32), levels=(0.5,), root_id="base")
    high = _derivative(roots, prepared, "hit_times.long_tp",
                       np.full((1, 2), 1, dtype=np.uint32), levels=(2.0,))
    with pytest.raises(ValueError, match="monotone"):
        validator.validate_declared_files(snapshot=prepared.snapshot, references=(high, low),
                                          trusted_roots=roots)



def test_attestation_rejects_copied_manifest_outside_pinned_slot(tmp_path):
    validator, roots, request, _ = _prepared_fixture(tmp_path)
    copied = tmp_path / "copied-base"
    copytree(roots["base"], copied)
    with pytest.raises(ValueError, match="pinned published slot"):
        validator.attest_snapshot(snapshot=request.snapshot, trusted_roots={"base": copied})


def test_prepared_validator_accepts_real_s02_builder_domain(tmp_path, tmp_path_factory):
    from tests.unit.contexts.backtest.application.services.v2 import (
        test_derived_artifact_materialization as derivatives,
    )

    built = getattr(derivatives.built, "__wrapped__")(tmp_path_factory)
    fixture, runner, snapshot, _ = built
    request = derivatives.request_for(built, risk=True, selected=(1,))
    manifest = fixture.loader.load_slot_manifest(fixture.coordinates, snapshot.slot)
    root = manifest.path.parent
    refs = list(snapshot.source_file_identities)
    for mapping in manifest.mappings:
        price = next(p for p in manifest.prices if p.timeframe == mapping.timeframe)
        for name in ("bar_open_1m_idx", "bar_close_1m_idx"):
            refs.append(replace(_source_ref(root, getattr(mapping, name),
                                            f"mappings.{mapping.timeframe}.{name}", price),
                                root_id="published-source"))
    snapshot = replace(snapshot, source_file_identities=tuple(refs),
                       consumed_domains=tuple(ref.domain for ref in refs))
    request = replace(request, snapshot=snapshot)
    output = tmp_path / "real-attempt"
    roots = {"published-source": root, request.output_root_id: output}
    validator = BacktestArtifactManifestValidatorV2(fixture.loader)
    snapshot = validator.attest_snapshot(snapshot=snapshot, trusted_roots=roots)
    result = runner.materialize_derived(request, output_directory=output)
    prepared = BacktestPreparedArtifactSet(
        snapshot, "b" * 64, "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000002", request.owner_token, request.attempt,
        request.output_root_id, result.files, tuple("generated" for _ in result.files),
    )
    validator.validate_prepared_inputs(prepared=prepared, request=request, trusted_roots=roots)


@pytest.mark.parametrize("ambiguous", [False, True])
def test_prepared_risk_grid_uses_existing_fraction_tolerance(tmp_path, ambiguous):
    validator, roots, request, prepared = _prepared_fixture(tmp_path)
    request = replace(request, tp_levels_pct=(0.50000001,))
    refs = [_derivative(roots, prepared, "signals.ma.ema",
                        np.zeros((1, 2), dtype=np.int8), rows=(7,))]
    levels = (0.5, 0.500001) if ambiguous else (0.5,)
    for name in ("tp_values", "long_tp", "short_tp"):
        values = (np.asarray(levels, dtype=np.float64) / 100).astype(np.float32)
        array = values if name.endswith("values") else np.full((len(levels), 2), 2, dtype=np.uint32)
        refs.append(_derivative(roots, prepared, f"hit_times.{name}", array, levels=levels))
    if ambiguous:
        with pytest.raises(ValueError, match="ambiguous"):
            validator.validate_prepared_inputs(prepared=_with_refs(prepared, *refs),
                                               request=request, trusted_roots=roots)
    else:
        validator.validate_prepared_inputs(prepared=_with_refs(prepared, *refs),
                                           request=request, trusted_roots=roots)


@pytest.mark.parametrize("missing", ["tp_values", "long_tp", "short_tp"])
def test_declared_risk_requires_coherent_grid_and_both_directions(tmp_path, missing):
    validator, roots, _, prepared = _prepared_fixture(tmp_path)
    refs = []
    for name in ("tp_values", "long_tp", "short_tp"):
        if name == missing:
            continue
        values = (np.asarray([0.005], dtype=np.float32) if name.endswith("values")
                  else np.full((1, 2), 2, dtype=np.uint32))
        refs.append(_derivative(roots, prepared, f"hit_times.{name}", values, levels=(0.5,)))
    with pytest.raises(ValueError, match="incoherent"):
        validator.validate_declared_files(snapshot=prepared.snapshot, references=tuple(refs),
                                          trusted_roots=roots)



def test_attest_replays_original_proof_after_real_npy_shape_append(tmp_path):
    validator, roots, _, prepared = _prepared_fixture(tmp_path)
    original = prepared.snapshot
    root = roots["base"]
    manifest_path = root / "manifest.yaml"
    payload = yaml.safe_load(manifest_path.read_text())
    original_bytes = {ref.relative_path: (root / ref.relative_path).read_bytes()
                      for ref in original.source_file_identities}
    minute_count = next(ref.domain.shape[0] for ref in original.source_file_identities
                        if ref.domain.role == "prices.1m.open_time")
    shift_ms = 10_000_000
    current_refs = []
    for ref in original.source_file_identities:
        path = root / ref.relative_path
        values = np.load(path, allow_pickle=False)
        tail = values.copy()
        if ref.domain.role.endswith((".open_time", ".close_time")):
            tail += shift_ms
        elif ref.domain.role.startswith("mappings."):
            tail += minute_count
        appended = np.concatenate((values, tail), axis=0)
        np.save(path, appended, allow_pickle=False)
        # np.save rewrites a real larger NPY header and payload, not just metadata.
        assert path.read_bytes()[:128] != original_bytes[ref.relative_path][:128]
        np.testing.assert_array_equal(np.load(path)[:values.shape[0]], values)
        end = datetime.fromisoformat(ref.domain.end_utc.replace("Z", "+00:00"))
        current_refs.append(replace(
            ref, sha256=sha256(path.read_bytes()).hexdigest(),
            domain=replace(ref.domain, shape=appended.shape, row_count=appended.shape[0],
                           end_utc=datetime.fromtimestamp(
                               end.timestamp() + shift_ms / 1000, timezone.utc,
                           ).strftime("%Y-%m-%dT%H:%M:%SZ")),
        ))
    by_path = {ref.relative_path: ref for ref in current_refs}

    def update_physical_metadata(value):
        if isinstance(value, dict):
            ref = by_path.get(value.get("path"))
            if ref is not None:
                value.update(sha256=ref.sha256, shape=list(ref.domain.shape))
            for child in value.values():
                update_physical_metadata(child)
        elif isinstance(value, list):
            for child in value:
                update_physical_metadata(child)

    update_physical_metadata(payload)
    for price in payload["prices"]:
        coverage = price["coverage"]
        coverage["bar_count"] *= 2
        coverage["open_time_end"] += shift_ms
        coverage["close_time_end"] += shift_ms
    manifest_path.write_text(yaml.safe_dump(payload))
    replay = replace(original, source_file_identities=tuple(current_refs),
                     manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest())
    assert replay.consumed_domains == original.consumed_domains
    assert replay.prefix_proof == original.prefix_proof
    assert all(new.domain.shape[0] == 2 * old.domain.shape[0]
               for old, new in zip(original.source_file_identities, current_refs, strict=True))
    attested = validator.attest_snapshot(snapshot=replay, trusted_roots=roots)
    assert attested.prefix_proof == original.prefix_proof
    assert attested.prefix_proof.source_domain_sha256 == original.prefix_proof.source_domain_sha256

    # Re-sign the current physical file/manifest after corrupting an old consumed value.
    # Only the retained payload proof can detect this change.
    target = next(ref for ref in current_refs if ref.domain.role == "prices.15m.ohlcv")
    path = root / target.relative_path
    values = np.load(path, allow_pickle=False)
    values[0, 0] += 1
    np.save(path, values, allow_pickle=False)
    changed = replace(target, sha256=sha256(path.read_bytes()).hexdigest())
    by_path[target.relative_path] = changed
    update_physical_metadata(payload)
    manifest_path.write_text(yaml.safe_dump(payload))
    corrupted = replace(
        replay, source_file_identities=tuple(
            changed if ref == target else ref for ref in current_refs),
        manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="source prefix proof mismatch"):
        validator.attest_snapshot(snapshot=corrupted, trusted_roots=roots)
