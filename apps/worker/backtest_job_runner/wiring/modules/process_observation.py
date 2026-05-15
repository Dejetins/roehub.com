from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

_EVIDENCE_DIR_KEY = "ROEHUB_BACKTEST_CHILD_EVIDENCE_DIR"
_SAMPLE_INTERVAL_KEY = "ROEHUB_BACKTEST_CHILD_EVIDENCE_SAMPLE_INTERVAL_SECONDS"
_COLLECT_RSS_KEY = "ROEHUB_BACKTEST_CHILD_EVIDENCE_COLLECT_RSS"
_COLLECT_VMMAP_KEY = "ROEHUB_BACKTEST_CHILD_EVIDENCE_COLLECT_VMMAP"
_DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.2


@dataclass(frozen=True, slots=True)
class ObservedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    evidence: Mapping[str, Any]


def run_observed_subprocess(
    *,
    cmd: Sequence[str],
    env: Mapping[str, str],
    timeout_seconds: float,
    evidence_prefix: str,
    metadata: Mapping[str, Any],
    cancel_event: threading.Event | None = None,
) -> ObservedProcessResult:
    started_at = datetime.now(UTC)
    parent_pid = os.getpid()
    sample_interval = _sample_interval_seconds(env=env)
    evidence_dir = _evidence_dir(env=env)
    collect_rss = _collect_rss(env=env, evidence_dir=evidence_dir)
    collect_vmmap = _collect_vmmap(env=env)
    parent_rss_before = _rss_bytes(parent_pid) if collect_rss else None
    parent_footprint_before = (
        _physical_footprint_bytes(parent_pid) if collect_vmmap else None
    )
    stdout_file = NamedTemporaryFile("w+", encoding="utf-8", delete=False)
    stderr_file = NamedTemporaryFile("w+", encoding="utf-8", delete=False)
    stdout_path = Path(stdout_file.name)
    stderr_path = Path(stderr_file.name)
    process: subprocess.Popen[str] | None = None
    peak_rss_bytes: int | None = None
    peak_physical_footprint_bytes: int | None = None
    timed_out = False
    cancelled = False
    try:
        with stdout_file, stderr_file:
            process = subprocess.Popen(  # noqa: S603
                list(cmd),
                env=dict(env),
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
            deadline = time.monotonic() + timeout_seconds
            while process.poll() is None:
                if collect_rss:
                    rss_bytes = _rss_bytes(process.pid)
                    if rss_bytes is not None:
                        peak_rss_bytes = max(peak_rss_bytes or 0, rss_bytes)
                if collect_vmmap:
                    footprint_bytes = _physical_footprint_bytes(process.pid)
                    if footprint_bytes is not None:
                        peak_physical_footprint_bytes = max(
                            peak_physical_footprint_bytes or 0,
                            footprint_bytes,
                        )
                if time.monotonic() >= deadline:
                    timed_out = True
                    process.kill()
                    process.wait(timeout=10)
                    break
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    _stop_process(process=process)
                    break
                time.sleep(sample_interval)
            returncode = process.returncode
        stdout = _read_text(path=stdout_path)
        stderr = _read_text(path=stderr_path)
        finished_at = datetime.now(UTC)
        parent_rss_after = _rss_bytes(parent_pid) if collect_rss else None
        parent_footprint_after = (
            _physical_footprint_bytes(parent_pid) if collect_vmmap else None
        )
        evidence = {
            "schema": "roehub_child_process_evidence_v1",
            "metadata": dict(metadata),
            "command": list(cmd),
            "pid": None if process is None else process.pid,
            "parent_pid": parent_pid,
            "started_at": started_at.isoformat().replace("+00:00", "Z"),
            "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
            "elapsed_seconds": (finished_at - started_at).total_seconds(),
            "exit_code": returncode,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "peak_rss_bytes": peak_rss_bytes,
            "peak_physical_footprint_bytes": peak_physical_footprint_bytes,
            "parent_rss_before_bytes": parent_rss_before,
            "parent_rss_after_bytes": parent_rss_after,
            "parent_retained_rss_delta_bytes": _delta(parent_rss_before, parent_rss_after),
            "parent_physical_footprint_before_bytes": parent_footprint_before,
            "parent_physical_footprint_after_bytes": parent_footprint_after,
            "parent_retained_physical_footprint_delta_bytes": _delta(
                parent_footprint_before,
                parent_footprint_after,
            ),
            "stdout_tail": _bounded_tail(value=stdout, limit=4000),
            "stderr_tail": _bounded_tail(value=stderr, limit=4000),
        }
        if evidence_dir is not None:
            _write_evidence(
                evidence_dir=evidence_dir,
                prefix=evidence_prefix,
                pid=None if process is None else process.pid,
                evidence=evidence,
            )
        return ObservedProcessResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            evidence=evidence,
        )
    finally:
        stdout_path.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)


def _evidence_dir(*, env: Mapping[str, str]) -> Path | None:
    raw = env.get(_EVIDENCE_DIR_KEY, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser()


def _sample_interval_seconds(*, env: Mapping[str, str]) -> float:
    raw = env.get(_SAMPLE_INTERVAL_KEY, "").strip()
    if not raw:
        return _DEFAULT_SAMPLE_INTERVAL_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SAMPLE_INTERVAL_SECONDS
    return value if value > 0 else _DEFAULT_SAMPLE_INTERVAL_SECONDS


def _collect_rss(*, env: Mapping[str, str], evidence_dir: Path | None) -> bool:
    raw = env.get(_COLLECT_RSS_KEY)
    if raw is not None:
        return _truthy(raw)
    return evidence_dir is not None


def _collect_vmmap(*, env: Mapping[str, str]) -> bool:
    return _truthy(env.get(_COLLECT_VMMAP_KEY, ""))


def _truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _rss_bytes(pid: int) -> int | None:
    try:
        raw = subprocess.check_output(  # noqa: S603
            ["ps", "-o", "rss=", "-p", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).strip()
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    try:
        return int(raw.splitlines()[-1].strip()) * 1024
    except ValueError:
        return None


def _physical_footprint_bytes(pid: int) -> int | None:
    vmmap_path = Path("/usr/bin/vmmap")
    if not vmmap_path.exists():
        return None
    try:
        raw = subprocess.check_output(  # noqa: S603
            [str(vmmap_path), "-summary", str(pid)],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:  # noqa: BLE001
        return None
    for line in raw.splitlines():
        if "Physical footprint" not in line:
            continue
        parsed = _parse_size(line)
        if parsed is not None:
            return parsed
    return None


def _parse_size(value: str) -> int | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([KMG]?)", value)
    if match is None:
        return None
    number = float(match.group(1))
    unit = match.group(2)
    multiplier = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}.get(unit, 1)
    return int(number * multiplier)


def _write_evidence(
    *,
    evidence_dir: Path,
    prefix: str,
    pid: int | None,
    evidence: Mapping[str, Any],
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    pid_token = "unknown" if pid is None else str(pid)
    path = evidence_dir / f"{prefix}-{pid_token}-{suffix}.json"
    path.write_text(
        json.dumps(evidence, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stop_process(*, process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _read_text(*, path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _bounded_tail(*, value: str | None, limit: int) -> str:
    if value is None:
        return ""
    return value[-limit:]


def _delta(before: int | None, after: int | None) -> int | None:
    if before is None or after is None:
        return None
    return after - before
