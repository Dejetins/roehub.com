"""Versioned input wire contracts, independently checked prefix bytes and persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest
import yaml

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.adapters.outbound.persistence.postgres.backtest_job_repository import (  # noqa: E501
    _build_job_insert_parameters,
    _map_job_row,
)
from trading.contexts.backtest.application.dto.input_recipe import BacktestInputRecipe
from trading.contexts.backtest.domain.entities.backtest_job import (
    BacktestJob,
    BacktestJobArtifactPin,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCandleSnapshot,
    ArtifactCoordinatesV2,
    ArtifactDerivativeBuildRequest,
    ArtifactDerivativeBuildResult,
    ArtifactInputFileReference,
    ArtifactPrefixDomain,
    ArtifactPrefixProof,
    ArtifactRequestedRow,
    BacktestPreparedArtifactSet,
)
from trading.shared_kernel.primitives import OrganizationId, UserId

START = "2025-01-01T00:00:00Z"
END = "2025-01-01T00:03:00Z"
ORG = "00000000-0000-0000-0000-000000000001"
JOB = "00000000-0000-0000-0000-000000000002"


def recipe() -> BacktestInputRecipe:
    domain = ArtifactPrefixDomain(
        "prices.1m.open_time", "1m", "<i8", ("time",), START, END, 3, (3,)
    )
    ref = ArtifactInputFileReference(
        "published-source", "prices/1m/open_time.i64.npy", "a" * 64, domain
    )
    snapshot = ArtifactCandleSnapshot(
        ArtifactCoordinatesV2("binance", "spot", "BTCUSDT"),
        1,
        "slot_a",
        4,
        "b" * 64,
        "15m",
        "1m",
        (ref,),
        (domain,),
    )
    return BacktestInputRecipe(
        snapshot,
        START,
        END,
        (ArtifactRequestedRow("ma.ema", "close", 37, (("window", 21),)),),
        "signals/v1",
        "c" * 64,
        "numba/v1",
        "f32-f64/v1",
        "not_applicable",
        None,
    )


def prepared() -> BacktestPreparedArtifactSet:
    value = recipe()
    domain = value.snapshot.source_file_identities[0].domain
    digest = domain.payload_digest(np.array([1, 2, 3], dtype=np.int64))
    proof = ArtifactPrefixProof(
        "verified", (domain,), (digest,), ArtifactPrefixProof.combined_digest((domain,), (digest,))
    )
    snapshot = replace(value.snapshot, prefix_proof=proof)
    signals = ArtifactInputFileReference(
        "attempt-root",
        "signals/ema.npy",
        "d" * 64,
        ArtifactPrefixDomain(
            "signals.ma.ema", "15m", "|i1", ("variant", "time"), START, END, 1, (1, 3)
        ),
        (37,),
    )
    return BacktestPreparedArtifactSet(
        snapshot,
        value.semantic_sha256,
        ORG,
        JOB,
        "owner-1",
        1,
        "attempt-root",
        (signals,),
        ("generated",),
    )


def test_round_trips_have_no_preflight_payload_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("metadata contracts must not scan payloads")

    monkeypatch.setattr(np, "load", forbidden)
    original = recipe()
    encoded = json.loads(json.dumps(original.as_mapping()))
    assert BacktestInputRecipe.from_mapping(encoded) == original
    assert original.snapshot.prefix_proof == ArtifactPrefixProof()
    assert ArtifactCandleSnapshot.from_mapping(original.snapshot.as_mapping()) == original.snapshot
    request = ArtifactDerivativeBuildRequest(
        original.snapshot,
        original.rows,
        "signals/v1",
        "c" * 64,
        "numba/v1",
        "f32-f64/v1",
        (),
        (),
        "attempt-root",
        "owner-1",
        1,
        4096,
        8192,
    )
    assert ArtifactDerivativeBuildRequest.from_mapping(request.as_mapping()) == request
    result = ArtifactDerivativeBuildResult("attempt-root", "owner-1", 1, (), 0)
    assert ArtifactDerivativeBuildResult.from_mapping(result.as_mapping()) == result


def test_prepared_manifest_is_distinct_and_round_trips(tmp_path: Path) -> None:
    value = prepared()
    fixture = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    path = tmp_path / "prepared.yaml"
    path.write_text(yaml.safe_dump(value.as_mapping()))
    assert fixture.loader.load_prepared_manifest_from_path(path) == value
    with pytest.raises(ValueError):
        fixture.loader.load_manifest_from_path(path=path, slot="slot_a")
    payload = value.as_mapping()
    payload["slot"] = "slot_a"
    with pytest.raises(ValueError, match="exact keys"):
        BacktestPreparedArtifactSet.from_mapping(payload)


@pytest.mark.parametrize("version", [None, True, 0, 2, "1", "backtest-input-recipe/v2"])
def test_unknown_recipe_versions_fail(version) -> None:
    payload = recipe().as_mapping()
    payload["schema"] = version
    with pytest.raises(ValueError):
        BacktestInputRecipe.from_mapping(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", []),
        ("rows", {}),
        ("tp_levels_pct", [1, 0.5]),
        ("sl_levels_pct", [True]),
        ("precision_version", ""),
        ("defaults_sha256", "invalid"),
        ("index_domain", "rebased"),
        ("funding_policy", "strict"),
        ("requested_end_utc", START),
    ],
)
def test_recipe_shapes_and_identities_rejected(field, value) -> None:
    payload = recipe().as_mapping()
    payload[field] = value
    with pytest.raises(ValueError):
        BacktestInputRecipe.from_mapping(payload)


@pytest.mark.parametrize("path", ["/tmp/inputs.npy", "../x.npy", "a/../x", "a//x", "a\\x"])
def test_reference_paths_cannot_be_absolute_or_traverse(path) -> None:
    ref = recipe().snapshot.source_file_identities[0]
    with pytest.raises(ValueError):
        replace(ref, relative_path=path)


def test_sparse_rows_are_explicit_and_not_positions() -> None:
    ref = prepared().derivatives[0]
    assert ref.row_ids == (37,)
    assert ArtifactInputFileReference.from_mapping(ref.as_mapping()) == ref
    with pytest.raises(ValueError):
        replace(ref, row_ids=())
    with pytest.raises(ValueError):
        replace(ref, row_ids=(20, 37, 100))
    with pytest.raises(ValueError):
        replace(ref, row_ids=(True,))


def test_prefix_format_matches_independent_bytes_and_append_only_replay(tmp_path: Path) -> None:
    domain = recipe().snapshot.source_file_identities[0].domain
    values = np.array([1, 2, 3], dtype="<i8")
    descriptor = json.dumps(
        domain.as_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    expected = hashlib.sha256(
        len(descriptor).to_bytes(8, "little")
        + descriptor
        + b"\x01\0\0\0\0\0\0\0\x02\0\0\0\0\0\0\0\x03\0\0\0\0\0\0\0"
    ).hexdigest()
    assert domain.payload_digest(values) == expected
    assert domain.payload_digest(values.astype(">i8")) == expected
    old = tmp_path / "old.npy"
    new = tmp_path / "new.npy"
    np.save(old, values)
    np.save(new, np.array([1, 2, 3, 4], dtype="<i8"))
    assert hashlib.sha256(old.read_bytes()).digest() != hashlib.sha256(new.read_bytes()).digest()
    assert domain.payload_digest(np.load(new, mmap_mode="r", allow_pickle=False)[:3]) == expected
    assert domain.payload_digest(np.array([1, 9, 3], dtype="<i8")) != expected
    with pytest.raises(ValueError):
        domain.payload_digest(np.load(new, allow_pickle=False))
    with pytest.raises(ValueError):
        domain.payload_digest(values.astype(np.float32))


def test_empty_funding_proof_differs_from_not_applicable_and_preserves_nan_bits() -> None:
    domain = ArtifactPrefixDomain(
        "funding.rates", "1m", "<f8", ("funding_event",), START, END, 0, (0,)
    )
    assert len(domain.payload_digest(np.array([], dtype=np.float64))) == 64
    value = recipe()
    explicit = replace(value, funding_policy="strict", funding_fingerprint="e" * 64)
    assert explicit.semantic_sha256 != value.semantic_sha256
    domain = replace(domain, row_count=1, shape=(1,))
    a = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    b = np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64)
    assert domain.payload_digest(a) != domain.payload_digest(b)


def test_pending_and_verified_proof_state_guards() -> None:
    good = prepared().snapshot.prefix_proof
    assert ArtifactPrefixProof.from_mapping(good.as_mapping()) == good
    with pytest.raises(ValueError):
        replace(good, state="pending")
    with pytest.raises(ValueError):
        replace(good, source_domain_sha256="e" * 64)
    with pytest.raises(ValueError):
        replace(prepared(), snapshot=recipe().snapshot)
    with pytest.raises(ValueError):
        replace(prepared(), provenance=("reused",))
    with pytest.raises(ValueError):
        replace(prepared(), output_root_id="published-source")
    with pytest.raises(ValueError):
        replace(prepared(), schema_version=True)


def test_semantic_hash_excludes_physical_choices_and_proof_phase() -> None:
    original = recipe()
    physical = replace(
        original.snapshot,
        slot="slot_b",
        generation=10,
        manifest_sha256="e" * 64,
        source_file_identities=(
            replace(
                original.snapshot.source_file_identities[0],
                root_id="another-source",
                relative_path="other.npy",
            ),
        ),
    )
    assert replace(original, snapshot=physical).semantic_sha256 == original.semantic_sha256
    assert (
        replace(original, snapshot=prepared().snapshot).semantic_sha256 == original.semantic_sha256
    )
    for changes in (
        {"rule_version": "signals/v2"},
        {"defaults_sha256": "e" * 64},
        {"compute_version": "numba/v2"},
        {"precision_version": "f64/v1"},
    ):
        assert replace(original, **changes).semantic_sha256 != original.semantic_sha256
    changed = replace(
        original.snapshot,
        source_file_identities=(
            replace(original.snapshot.source_file_identities[0], sha256="f" * 64),
        ),
    )
    assert replace(original, snapshot=changed).semantic_sha256 != original.semantic_sha256
    payload = original.as_mapping()
    payload["retention"] = "on_demand"
    with pytest.raises(ValueError):
        BacktestInputRecipe.from_mapping(payload)


def test_nullable_repository_round_trip_preserves_legacy_hashes() -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    job = BacktestJob.create_queued(
        job_id=UUID(JOB),
        organization_id=OrganizationId.from_string(ORG),
        user_id=UserId.from_string(ORG),
        mode="template",
        created_at=now,
        request_json={"mode": "template"},
        request_hash="1" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="2" * 64,
        backtest_runtime_config_hash="3" * 64,
    )
    legacy = _map_job_row(row=_build_job_insert_parameters(job=job))
    assert legacy == job
    assert legacy.input_recipe_json is None
    candidate = replace(
        job,
        attempt=1,
        artifact_pin=BacktestJobArtifactPin("slot_a", 4, "b" * 64, "2025-01-01"),
        input_recipe_json=recipe().as_mapping(),
        preparation_provenance_json=prepared().as_mapping(),
    )
    restored = _map_job_row(row=_build_job_insert_parameters(job=candidate))
    assert restored == candidate
    assert restored.request_hash == legacy.request_hash
    assert restored.engine_params_hash == legacy.engine_params_hash
    assert restored.backtest_runtime_config_hash == legacy.backtest_runtime_config_hash


def test_verified_prefix_can_be_shorter_than_original_file() -> None:
    snapshot = recipe().snapshot
    full = snapshot.source_file_identities[0].domain
    prefix = replace(full, row_count=2, shape=(2,), end_utc="2025-01-01T00:02:00Z")
    digest = prefix.payload_digest(np.array([1, 2], dtype="<i8"))
    proof = ArtifactPrefixProof(
        "verified", (prefix,), (digest,), ArtifactPrefixProof.combined_digest((prefix,), (digest,))
    )
    candidate = replace(snapshot, consumed_domains=(prefix,), prefix_proof=proof)
    assert candidate.source_file_identities[0].domain == full
    assert ArtifactCandleSnapshot.from_mapping(candidate.as_mapping()) == candidate
    with pytest.raises(ValueError):
        replace(snapshot, consumed_domains=(replace(prefix, index_origin=1),))
    with pytest.raises(ValueError):
        replace(snapshot, consumed_domains=(replace(full, row_count=4, shape=(4,)),))


def test_provenance_is_bound_to_job_organization_attempt_and_recipe() -> None:
    from trading.contexts.backtest.domain.errors import BacktestStorageError

    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    job = BacktestJob.create_queued(
        job_id=UUID(JOB),
        organization_id=OrganizationId.from_string(ORG),
        user_id=UserId.from_string(ORG),
        mode="template",
        created_at=now,
        request_json={"mode": "template"},
        request_hash="1" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="2" * 64,
        backtest_runtime_config_hash="3" * 64,
        artifact_pin=BacktestJobArtifactPin("slot_a", 4, "b" * 64, "2025-01-01"),
    )
    job = replace(
        job,
        attempt=1,
        input_recipe_json=recipe().as_mapping(),
        preparation_provenance_json=prepared().as_mapping(),
    )
    valid_row = _build_job_insert_parameters(job=job)
    for fields in (
        {"organization_id": JOB},
        {"job_id": ORG},
        {"attempt": 77},
        {"recipe_sha256": "e" * 64},
    ):
        wrong = replace(prepared(), **fields).as_mapping()
        with pytest.raises(BacktestStorageError):
            _build_job_insert_parameters(job=replace(job, preparation_provenance_json=wrong))
        with pytest.raises(BacktestStorageError):
            _map_job_row(row={**valid_row, "preparation_provenance_json": wrong})


def test_recipe_consumed_domain_must_cover_requested_interval() -> None:
    original = recipe()
    domain = original.snapshot.consumed_domains[0]
    prefix = replace(domain, row_count=2, shape=(2,), end_utc="2025-01-01T00:02:00Z")
    with pytest.raises(ValueError, match="requested interval"):
        replace(original, snapshot=replace(original.snapshot, consumed_domains=(prefix,)))


def test_verified_recipe_proof_cannot_be_replaced_by_provenance() -> None:
    from trading.contexts.backtest.domain.errors import BacktestStorageError

    value = prepared()
    original = replace(recipe(), snapshot=value.snapshot)
    domain = value.snapshot.consumed_domains[0]
    digest = domain.payload_digest(np.array([1, 99, 3], dtype="<i8"))
    different_proof = ArtifactPrefixProof(
        "verified", (domain,), (digest,), ArtifactPrefixProof.combined_digest((domain,), (digest,))
    )
    altered = replace(value, snapshot=replace(value.snapshot, prefix_proof=different_proof))
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    job = BacktestJob(
        job_id=UUID(JOB),
        organization_id=OrganizationId.from_string(ORG),
        user_id=UserId.from_string(ORG),
        mode="template",
        state="queued",
        created_at=now,
        updated_at=now,
        attempt=1,
        request_json={"mode": "template"},
        request_hash="1" * 64,
        engine_params_hash="2" * 64,
        backtest_runtime_config_hash="3" * 64,
        artifact_pin=BacktestJobArtifactPin("slot_a", 4, "b" * 64, "2025-01-01"),
        input_recipe_json=original.as_mapping(),
        preparation_provenance_json=value.as_mapping(),
    )
    row = _build_job_insert_parameters(job=job)
    with pytest.raises(BacktestStorageError, match="cannot be replaced"):
        _build_job_insert_parameters(
            job=replace(job, preparation_provenance_json=altered.as_mapping())
        )
    with pytest.raises(BacktestStorageError):
        _map_job_row(row={**row, "preparation_provenance_json": altered.as_mapping()})
