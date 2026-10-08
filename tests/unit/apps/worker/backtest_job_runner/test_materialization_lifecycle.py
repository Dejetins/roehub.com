"""Attempt disk ownership and IPC lifecycle, with real files and subprocesses."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import pytest
from numba import get_num_threads

from apps.worker.backtest_job_runner.wiring.modules.child_ipc import (
    acknowledge_prepared_over_socket,
    preflight_from_mapping,
    preflight_to_mapping,
)
from apps.worker.backtest_job_runner.wiring.modules.compute_resources import (
    ATTEMPT_OVERHEAD_BYTES,
    BacktestAttemptDirectory,
    BacktestScratchLimits,
    recover_attempt_directories,
)
from tests.unit.apps.worker.backtest_job_runner.test_child_process_executor import _preflight
from tests.unit.contexts.backtest.application.services.v2 import (
    test_prepare_pools_service as source,
)
from trading.contexts.backtest.application.dto.artifact_inputs import BacktestAttemptInputs
from trading.contexts.backtest.application.services.v2.combo_planning import (
    BacktestComboPlanningConfig,
    BacktestComboPlanningService,
)
from trading.contexts.backtest.application.services.v2.compute_policy import BacktestComputePolicy
from trading.contexts.backtest.application.services.v2.job_orchestration import (
    BacktestRuntimeJobOrchestrationService,
)
from trading.contexts.backtest.application.services.v2.job_scheduling import (
    BacktestNumbaThreadDecision,
)
from trading.contexts.backtest.application.services.v2.no_risk_exact import (
    BacktestNoRiskExactScoringService,
)
from trading.contexts.backtest.application.services.v2.prepare_pools import (
    BacktestPreparePoolsConfig,
    BacktestPreparePoolsService,
)
from trading.contexts.backtest.application.services.v2.tp_sl_exact import (
    BacktestTpSlExactScoringService,
)
from trading.contexts.backtest.application.services.v2.tp_sl_hit_times import (
    BacktestTpSlHitTimesService,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_validator import (  # noqa: E501
    BacktestArtifactManifestValidatorV2,
)

built = source.built
prepared_source = source.prepared_source


def test_reservations_are_bounded_and_200_attempts_release(tmp_path):
    limits = BacktestScratchLimits(
        attempt_bytes=100, worker_bytes=ATTEMPT_OVERHEAD_BYTES + 150, reserve_bytes=1,
    )
    first = BacktestAttemptDirectory.reserve(
        root=tmp_path, owner={"owner_token": str(uuid4())}, estimated_bytes=100, limits=limits
    )
    with pytest.raises(ValueError, match="worker scratch"):
        BacktestAttemptDirectory.reserve(
            root=tmp_path, owner={}, estimated_bytes=100, limits=limits
        )
    assert recover_attempt_directories(root=tmp_path, reconcile_owner=lambda _: True) == 0
    first.cleanup_after_reap()
    baseline_bytes = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    for coordinate in range(200):
        policy = BacktestScratchLimits.from_environ(
            {
                "ROEHUB_BACKTEST_ATTEMPT_DISK_BYTES": str(50 + coordinate % 100),
                "ROEHUB_BACKTEST_WORKER_DISK_BYTES": str(ATTEMPT_OVERHEAD_BYTES + 150),
                "ROEHUB_BACKTEST_DISK_RESERVE_BYTES": "1",
            }
        )
        attempt = BacktestAttemptDirectory.reserve(
            root=tmp_path,
            owner={
                "exchange": "binance",
                "market_type": "spot",
                "symbol": f"COIN{coordinate}USDT",
                "owner_token": str(uuid4()),
            },
            estimated_bytes=10,
            limits=policy,
        )
        assert len(list(tmp_path.glob("attempt-*/ownership.json"))) == 1
        (attempt.path / "inputs").mkdir()
        (attempt.path / "inputs" / "partial.npy").write_bytes(b"partial")
        attempt.cleanup_after_reap()
    assert not list(tmp_path.glob("attempt-*"))
    assert (
        sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()) == baseline_bytes
    )


def test_live_child_holds_lifetime_after_parent_fd_closes(tmp_path):
    attempt = BacktestAttemptDirectory.reserve(
        root=tmp_path,
        owner={"owner_token": str(uuid4())},
        estimated_bytes=0,
        limits=BacktestScratchLimits(reserve_bytes=1),
    )
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        pass_fds=(attempt.lock_fd,),
    )
    os.close(attempt.lock_fd)  # simulates parent descriptor loss, not child death
    try:
        assert recover_attempt_directories(root=tmp_path, reconcile_owner=lambda _: True) == 0
        child.communicate(b"x", timeout=5)
        assert recover_attempt_directories(root=tmp_path, reconcile_owner=lambda _: False) == 0
        assert recover_attempt_directories(root=tmp_path, reconcile_owner=lambda _: True) == 1
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_foreign_marker_and_unreaped_child_are_not_cleaned(tmp_path):
    attempt = BacktestAttemptDirectory.reserve(
        root=tmp_path,
        owner={"owner_token": str(uuid4())},
        estimated_bytes=0,
        limits=BacktestScratchLimits(reserve_bytes=1),
    )
    attempt.child_reaped = False
    with pytest.raises(RuntimeError, match="unreaped"):
        attempt.cleanup_after_reap()
    attempt.child_reaped = True
    marker = attempt.path / "ownership.json"
    original = marker.read_bytes()
    marker.write_text("{}")
    with pytest.raises(ValueError, match="ownership changed"):
        attempt.cleanup_after_reap()
    marker.write_bytes(original)
    attempt.cleanup_after_reap()


def test_ipc_retains_funding_and_recipe(prepared_source):
    fixture, _, snapshot, _ = prepared_source
    _, context = source._input_context(fixture, snapshot.slot)
    recipe = source._input_recipe(prepared_source, context, risk=False)
    preflight = _preflight()
    metadata = replace(
        preflight.artifact_metadata,
        funding_manifest_hash="f" * 64,
        funding_coverage_status="degraded",
        funding_coverage_policy="degraded_with_warning",
        funding_rows_count=3,
        funding_expected_event_count=4,
        funding_missing_event_count=1,
        funding_reason_codes=("missing_event",),
    )
    original = replace(
        preflight,
        artifact_metadata=metadata,
        input_recipe=recipe,
        funding_readiness={"status": "degraded"},
        direction_market_compatibility={"version": "funding_short_policy_v1"},
    )
    assert preflight_from_mapping(payload=preflight_to_mapping(preflight=original)) == original


@pytest.mark.parametrize("mode", ["ack", "parent_loss", "stale", "timeout"])
def test_child_requires_matching_durable_ack(prepared_source, tmp_path, mode):
    fixture, _, snapshot, _ = prepared_source
    _, context = source._input_context(fixture, snapshot.slot)
    recipe = source._input_recipe(prepared_source, context, risk=False)
    _, _, prepared = source._prepare_inputs(prepared_source, context, recipe, tmp_path / "inputs")
    parent, child = socket.socketpair()
    results = []

    def wait_for_ack():
        try:
            results.append(
                acknowledge_prepared_over_socket(
                    fd=child.detach(), prepared=prepared, timeout_seconds=0.3
                )
            )
        except (ValueError, RuntimeError, TimeoutError, OSError) as error:
            results.append(error)

    worker = threading.Thread(target=wait_for_ack)
    worker.start()
    try:
        parent.settimeout(1)
        message = bytearray()
        while not message.endswith(b"\n"):
            message.extend(parent.recv(65536))
        assert b"backtest-prepared-input-event/v1" in message
        if mode == "ack":
            parent.sendall(prepared.content_sha256.encode() + b"\n")
        elif mode == "stale":
            parent.sendall(b"not-the-attested-content\n")
        elif mode == "parent_loss":
            parent.close()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert (
            (results == [prepared.content_sha256])
            if mode == "ack"
            else isinstance(results[0], Exception)
        )
    finally:
        parent.close()
        worker.join(timeout=2)


@pytest.mark.parametrize("risk", [False, True])
def test_real_preparation_occurs_once_before_warmup_and_scoring(prepared_source, tmp_path, risk):
    fixture, runner, snapshot, _ = prepared_source
    source._rewrite_published_inventory(prepared_source, "generated")
    loader, context = source._input_context(fixture, snapshot.slot)
    recipe = source._input_recipe(prepared_source, context, risk=risk)
    request = source._normalized_request()
    request["coordinates"] = asdict(recipe.snapshot.coordinates)
    request["time_range"] = {"start": recipe.requested_start_utc, "end": recipe.requested_end_utc}
    request["indicators"] = [
        {
            "indicator_id": name,
            "sources": ["close", "open"],
            "window": {"start": 21, "stop": 100, "step": 79},
        }
        for name in ("ma.ema", "ma.sma", "ma.wma")
    ]
    request["risk"] = (
        {
            "mode": "tp_sl_grid",
            "tp": {"start_pct": 0.5, "stop_pct": 2.0, "step_pct": 1.5},
            "sl": {"start_pct": 1.0, "stop_pct": 1.0, "step_pct": 1.0},
        }
        if risk
        else {"mode": "none"}
    )
    job_id = uuid4()
    recorded = source._RecordingDerivedBuilder(runner)
    acknowledgements = []

    def acknowledge(prepared):
        acknowledgements.append(prepared)
        return prepared.content_sha256

    metadata = replace(
        _preflight().artifact_metadata,
        artifact_slot=snapshot.slot,
        artifact_asof_date=context.source.slot_manifest.asof_date,
        artifact_slot_generation=recipe.snapshot.generation,
        artifact_manifest_hash=recipe.snapshot.manifest_sha256,
    )
    service = BacktestRuntimeJobOrchestrationService(
        compute_policy=BacktestComputePolicy(
            threads=BacktestNumbaThreadDecision(get_num_threads(), "test_effective_thread_budget")
        ),
        prepare_pools=BacktestPreparePoolsService(
            artifact_array_loader=loader,
            defaults_provider=runner.defaults_provider,
            config=BacktestPreparePoolsConfig(row_prefilter_top_fraction=1.0),
        ),
        combo_planning=BacktestComboPlanningService(
            config=BacktestComboPlanningConfig(combo_top_frac=1.0)
        ),
        no_risk_exact=BacktestNoRiskExactScoringService(),
        tp_sl_hit_times=BacktestTpSlHitTimesService(artifact_array_loader=loader),
        tp_sl_exact=BacktestTpSlExactScoringService(),
        artifact_array_loader=loader,
        derivative_builder=cast(Any, recorded),
        input_validator=BacktestArtifactManifestValidatorV2(artifact_loader=fixture.loader),
        attempt_inputs=BacktestAttemptInputs(
            tmp_path / "attempt-inputs",
            str(uuid4()),
            str(job_id),
            str(uuid4()),
            1,
            10_000_000,
            10_000_000,
            acknowledge,
        ),
    )
    result = service.execute(
        job_id=job_id,
        updated_at=datetime.now(UTC),
        preflight=replace(
            _preflight(),
            normalized_request=request,
            artifact_metadata=metadata,
            input_recipe=recipe,
        ),
    )
    assert len(recorded.requests) == len(acknowledgements) == 1
    assert result.stage_timings["sample_warmup"] >= 0
    assert result.stage_timings["input_materialization_including_attestation"] > 0
    assert result.exact_diagnostics["timing_accounting"]["schema"] == "orchestration_elapsed_v2"


@pytest.mark.parametrize("crash_at", ["mkdir", "pending", "rename"])
def test_restart_recovers_exact_creation_intent_before_ready_marker(tmp_path, crash_at):
    root = tmp_path / "scratch"
    root.mkdir()
    foreign = root / "attempt-foreign"
    foreign.mkdir()
    (foreign / "keep").write_text("foreign")
    script = """
import json, os, sys
from pathlib import Path
from apps.worker.backtest_job_runner.wiring.modules.compute_resources import (
    BacktestAttemptDirectory, BacktestScratchLimits,
)
root, phase = Path(sys.argv[1]), sys.argv[2]
original_open, original_replace, original_dump = os.open, os.replace, json.dump
def open_hook(path, *args, **kwargs):
    if phase == 'mkdir' and str(path).endswith('lifetime.lock'):
        os._exit(91)
    return original_open(path, *args, **kwargs)
def replace_hook(source, target):
    if phase == 'rename' and str(target).endswith('ownership.json'):
        os._exit(91)
    return original_replace(source, target)
def dump_hook(payload, stream, *args, **kwargs):
    if phase == 'pending' and str(stream.name).endswith('ownership.pending'):
        stream.write('{'); stream.flush(); os._exit(91)
    return original_dump(payload, stream, *args, **kwargs)
os.open, os.replace, json.dump = open_hook, replace_hook, dump_hook
BacktestAttemptDirectory.reserve(root=root, owner={'owner_token':'exact-creator'},
                                 estimated_bytes=10,
                                 limits=BacktestScratchLimits(reserve_bytes=1))
"""
    child = subprocess.run([sys.executable, "-c", script, str(root), crash_at], check=False)
    assert child.returncode == 91
    assert len(list(root.glob("attempt-*"))) == 2

    def no_committed_owner(_):
        raise AssertionError("pre-marker creation never committed database admission")

    assert recover_attempt_directories(root=root, reconcile_owner=no_committed_owner) == 1
    assert list(root.glob("attempt-*")) == [foreign]
    assert (foreign / "keep").read_text() == "foreign"
    attempt = BacktestAttemptDirectory.reserve(
        root=root,
        owner={"owner_token": "next"},
        estimated_bytes=10,
        limits=BacktestScratchLimits(reserve_bytes=1),
    )
    attempt.cleanup_after_reap()


def test_native_attempt_reserves_metadata_and_preserves_free_space(tmp_path, monkeypatch):
    import shutil

    limits = BacktestScratchLimits(worker_bytes=ATTEMPT_OVERHEAD_BYTES, reserve_bytes=1)
    attempt = BacktestAttemptDirectory.reserve(
        root=tmp_path, owner={}, estimated_bytes=0, limits=limits,
    )
    assert attempt.generated_bytes == 0
    assert attempt.reserved_bytes == ATTEMPT_OVERHEAD_BYTES
    with pytest.raises(ValueError, match="worker scratch"):
        BacktestAttemptDirectory.reserve(root=tmp_path, owner={}, estimated_bytes=0, limits=limits)
    attempt.cleanup_after_reap()
    monkeypatch.setattr(shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(
        10**9, 0, ATTEMPT_OVERHEAD_BYTES,
    ))
    with pytest.raises(ValueError, match="free-space"):
        BacktestAttemptDirectory.reserve(root=tmp_path, owner={}, estimated_bytes=0, limits=limits)
    assert not list(tmp_path.glob("attempt-*"))


def test_oversized_attempt_ipc_rejected_before_write(tmp_path):
    from apps.worker.backtest_job_runner.wiring.modules.compute_resources import (
        ATTEMPT_METADATA_FILE_BYTES,
        write_attempt_json,
    )

    path = tmp_path / "result.json"
    with pytest.raises(ValueError, match="metadata disk budget"):
        write_attempt_json(path, {"result": "x" * ATTEMPT_METADATA_FILE_BYTES})
    assert not path.exists()


def test_verbose_child_keeps_bounded_tails_without_scratch_files(tmp_path):
    from apps.worker.backtest_job_runner.wiring.modules.process_observation import (
        run_observed_subprocess,
    )

    result = run_observed_subprocess(
        cmd=[sys.executable, "-c",
             "import sys; sys.stdout.write('x'*2000000+'OUT'); "
             "sys.stderr.write('y'*2000000+'ERR')"],
        env=dict(os.environ), timeout_seconds=10, evidence_prefix="bounded-logs",
        metadata={}, output_directory=tmp_path,
    )
    assert result.returncode == 0
    assert len(result.stdout) == len(result.stderr) == 1_048_576
    assert result.stdout.endswith("OUT") and result.stderr.endswith("ERR")
    assert not list(tmp_path.iterdir())


def test_max_supported_top_count_summary_ipc_roundtrip(tmp_path):
    from types import SimpleNamespace

    from apps.worker.backtest_job_runner.wiring.modules.child_ipc import (
        BacktestChildSuccessResult,
        child_result_from_mapping,
        child_success_to_mapping,
    )
    from apps.worker.backtest_job_runner.wiring.modules.compute_resources import write_attempt_json
    from tests.unit.contexts.backtest.application.services.v2.test_top_result_assembly import (
        _assemble_for_job,
    )

    row = _assemble_for_job(uuid4())
    # The test profile permits 100, production profiles permit at most 50.
    # Real assembly stores summaries, not potentially unbounded trade histories.
    rows = tuple(replace(row, rank=rank) for rank in range(1, 101))
    payload = child_success_to_mapping(result=SimpleNamespace(
        top_variants=rows, stage_timings={}, summary_hash="a" * 64,
        cleanup_evidence={}, exact_diagnostics={}, instrumentation_counters={},
    ))
    write_attempt_json(tmp_path / "result.json", payload)
    assert (tmp_path / "result.json").stat().st_size < 1024**2
    restored = child_result_from_mapping(payload=payload)
    assert isinstance(restored, BacktestChildSuccessResult)
    assert len(restored.top_variants) == 100


def test_v1_reservation_gets_overhead_and_exact_dead_owner_recovery(tmp_path):
    import json

    limits = BacktestScratchLimits(worker_bytes=ATTEMPT_OVERHEAD_BYTES, reserve_bytes=1)
    attempt = BacktestAttemptDirectory.reserve(
        root=tmp_path, owner={"owner_token": "legacy"}, estimated_bytes=0, limits=limits,
    )
    marker = attempt.path / "ownership.json"
    payload = json.loads(marker.read_text())
    payload.update(schema="backtest-attempt-directory/v1", reserved_bytes=0)
    marker.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="worker scratch"):
        BacktestAttemptDirectory.reserve(root=tmp_path, owner={}, estimated_bytes=0, limits=limits)
    assert recover_attempt_directories(root=tmp_path, reconcile_owner=lambda _: True) == 0
    os.close(attempt.lock_fd)
    attempt.lock_fd = -1
    assert recover_attempt_directories(root=tmp_path, reconcile_owner=lambda _: False) == 0
    assert recover_attempt_directories(
        root=tmp_path, reconcile_owner=lambda owner: owner == {"owner_token": "legacy"},
    ) == 1
