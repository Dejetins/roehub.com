"""OS capability discovery without consulting the parent numerical runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, TextIO

from trading.contexts.backtest.application.services.v2.job_scheduling import (
    ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS,
    BacktestNumbaThreadDecision,
    backtest_numba_environ,
)

# Eight bounded 8 MiB slots cover input/result IPC, prepared marker, ownership,
# creation journal and file-system block/temporary overhead, even for native jobs.
ATTEMPT_METADATA_FILE_BYTES = 8 * 1024**2
ATTEMPT_OVERHEAD_BYTES = 8 * ATTEMPT_METADATA_FILE_BYTES


def write_attempt_json(path: Path, payload: object) -> None:
    """Bound each attempt IPC file before any bytes reach disk."""
    import json

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if len(encoded) > ATTEMPT_METADATA_FILE_BYTES:
        raise ValueError("attempt metadata disk budget exceeded")
    path.write_text(encoded, encoding="ascii")


OPERATOR_CAP = "ROEHUB_BACKTEST_CPU_CAP"


@dataclass(frozen=True, slots=True)
class BacktestCpuCapacity:
    hardware: int | None
    affinity: int | None
    quota: int | None
    operator_cap: int | None

    @property
    def usable(self) -> int | None:
        known = [
            x
            for x in (self.hardware, self.affinity, self.quota, self.operator_cap)
            if x is not None
        ]
        return min(known) if known else None


def discover_cpu_capacity(environ: Mapping[str, str]) -> BacktestCpuCapacity:
    affinity = None
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            affinity = len(get_affinity(0))
        except OSError:
            pass
    # Inspect the process cgroup and its ancestors, not merely the mount root.
    quota = _linux_cpu_quota()
    cap = None
    raw = environ.get(OPERATOR_CAP, "").strip()
    if raw:
        cap = int(raw)
        if cap < 1:
            raise ValueError(f"{OPERATOR_CAP} must be a positive integer")
    return BacktestCpuCapacity(os.cpu_count(), affinity, quota, cap)


def _linux_cpu_quota() -> int | None:
    membership = Path("/proc/self/cgroup")
    if not membership.exists():
        return None
    quotas = []
    try:
        for line in membership.read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            if controllers == "":
                root = Path("/sys/fs/cgroup")
                current = root / relative.lstrip("/")
                while current.is_relative_to(root):
                    path = current / "cpu.max"
                    if path.exists():
                        amount, period = path.read_text().split()
                        if amount != "max":
                            quotas.append(max(1, int(amount) // int(period)))
                    if current == root:
                        break
                    current = current.parent
            elif "cpu" in controllers.split(","):
                root = Path("/sys/fs/cgroup/cpu")
                current = root / relative.lstrip("/")
                while current.is_relative_to(root):
                    path = current / "cpu.cfs_quota_us"
                    if path.exists():
                        amount = int(path.read_text())
                        if amount > 0:
                            period = int((current / "cpu.cfs_period_us").read_text())
                            quotas.append(max(1, amount // period))
                    if current == root:
                        break
                    current = current.parent
    except (OSError, ValueError, ZeroDivisionError):
        return None
    return min(quotas) if quotas else None


def full_job_resource_environ(
    *,
    environ: Mapping[str, str],
    capacity: BacktestCpuCapacity | None = None,
) -> dict[str, str]:
    result = backtest_numba_environ(environ=environ, scheduling_class="heavy")
    limits = capacity or discover_cpu_capacity(environ)
    decision = BacktestNumbaThreadDecision(
        num_threads=int(result[ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS]),
        source=result["ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"],
    )
    # Preserve default12 on unknown/small hosts. Explicit allocations must fit;
    # an explicit operator cap also opts the default budget into validation.
    if (
        limits.usable is not None
        and decision.num_threads > limits.usable
        and (decision.source != "default_full_job_budget" or limits.operator_cap is not None)
    ):
        raise ValueError("backtest thread allocation exceeds verified CPU capacity")
    return result


@dataclass(frozen=True, slots=True)
class BacktestScratchLimits:
    """Disk reservations are separate from the builder's stricter RAM/cell budgets."""

    attempt_bytes: int = 2 * 1024**3
    worker_bytes: int = 8 * 1024**3
    reserve_bytes: int = 1024**3

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> BacktestScratchLimits:
        limits = cls(
            attempt_bytes=int(environ.get("ROEHUB_BACKTEST_ATTEMPT_DISK_BYTES", str(2 * 1024**3))),
            worker_bytes=int(environ.get("ROEHUB_BACKTEST_WORKER_DISK_BYTES", str(8 * 1024**3))),
            reserve_bytes=int(environ.get("ROEHUB_BACKTEST_DISK_RESERVE_BYTES", str(1024**3))),
        )
        if min(limits.attempt_bytes, limits.worker_bytes, limits.reserve_bytes) <= 0:
            raise ValueError("backtest disk limits must be positive")
        if limits.attempt_bytes > limits.worker_bytes:
            raise ValueError("attempt disk budget exceeds worker budget")
        return limits


@dataclass(slots=True)
class BacktestAttemptDirectory:
    """Parent-owned reservation; the child inherits the flock until it is reaped.

    The on-disk reservation survives parent death. Recovery must obtain the same
    lock AND reconcile the exact database owner before removing files. PID/age
    tests and expired heartbeats alone cannot establish filesystem safety.
    """

    root: Path
    path: Path
    lock_fd: int
    owner: Mapping[str, object]
    reserved_bytes: int
    generated_bytes: int
    child_reaped: bool = True

    @classmethod
    def reserve(
        cls,
        *,
        root: Path,
        owner: Mapping[str, object],
        estimated_bytes: int,
        limits: BacktestScratchLimits,
    ) -> BacktestAttemptDirectory:
        import fcntl
        import json
        import shutil
        from uuid import uuid4

        if type(estimated_bytes) is not int or not 0 <= estimated_bytes <= limits.attempt_bytes:
            raise ValueError("attempt generated disk budget exceeded")
        generated_bytes = estimated_bytes
        estimated_bytes += ATTEMPT_OVERHEAD_BYTES
        if len(json.dumps(dict(owner), ensure_ascii=True)) > ATTEMPT_METADATA_FILE_BYTES // 2:
            raise ValueError("attempt owner metadata disk budget exceeded")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if root.is_symlink():
            raise ValueError("scratch root must not be a symlink")
        root = root.resolve(strict=True)
        with (root / ".admission.lock").open("a+") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            _recover_creation_intent(root=root, journal=admission)
            reserved = 0
            for marker in root.glob("attempt-*/ownership.json"):
                if marker.is_symlink() or marker.parent.is_symlink():
                    raise ValueError("untrusted scratch ownership marker")
                payload = json.loads(marker.read_text())
                if payload.get("schema") not in {
                    "backtest-attempt-directory/v1", "backtest-attempt-directory/v2"
                }:
                    raise ValueError("unknown scratch reservation schema")
                amount = payload["reserved_bytes"]
                if type(amount) is not int or amount < 0:
                    raise ValueError("invalid scratch reservation")
                reserved += amount + (
                    ATTEMPT_OVERHEAD_BYTES if payload["schema"].endswith("/v1") else 0
                )
            if reserved + estimated_bytes > limits.worker_bytes:
                raise ValueError("worker scratch disk budget exceeded")
            if shutil.disk_usage(root).free - reserved - estimated_bytes < limits.reserve_bytes:
                raise ValueError("scratch free-space reserve exceeded")
            path = root / f"attempt-{uuid4().hex}"
            if path.exists():
                raise ValueError("attempt creation name already exists")
            _write_creation_intent(
                admission,
                {
                    "schema": "backtest-attempt-creation/v1",
                    "name": path.name,
                    "owner": dict(owner),
                    "reserved_bytes": estimated_bytes,
                },
            )
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(root_fd)
            finally:
                os.close(root_fd)
            created = False
            lock_fd = -1
            try:
                path.mkdir(mode=0o700)
                created = True
                lock_fd = os.open(path / "lifetime.lock", os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                pending = path / "ownership.pending"
                with pending.open("x") as marker:
                    json.dump(
                        {
                            "schema": "backtest-attempt-directory/v2",
                            "owner": dict(owner),
                            "reserved_bytes": estimated_bytes,
                        },
                        marker,
                    )
                    marker.flush()
                    os.fsync(marker.fileno())
                os.replace(pending, path / "ownership.json")
                for directory in (path, root):
                    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
            except BaseException:
                if lock_fd >= 0:
                    os.close(lock_fd)
                if created:
                    shutil.rmtree(path)
                _write_creation_intent(admission, {})
                raise
            _write_creation_intent(admission, {})
            return cls(root, path, lock_fd, dict(owner), estimated_bytes, generated_bytes)

    def cleanup_after_reap(self) -> None:
        """Only the supervising caller that reaped its child may invoke this method."""
        import fcntl
        import json
        import shutil

        if not self.child_reaped:
            raise RuntimeError("cannot clean an unreaped child attempt")
        if self.lock_fd < 0:
            raise RuntimeError("attempt was already released")
        with (self.root / ".admission.lock").open("a") as admission:
            fcntl.flock(admission, fcntl.LOCK_EX)
            if self.path.is_symlink() or self.path.parent != self.root:
                raise ValueError("attempt ownership path changed")
            marker = self.path / "ownership.json"
            if marker.is_symlink():
                raise ValueError("attempt ownership marker changed")
            payload = json.loads(marker.read_text())
            if payload != {
                "schema": "backtest-attempt-directory/v2",
                "owner": dict(self.owner),
                "reserved_bytes": self.reserved_bytes,
            }:
                raise ValueError("attempt ownership changed")
            shutil.rmtree(self.path)
            os.close(self.lock_fd)
            self.lock_fd = -1


def recover_attempt_directories(
    *,
    root: Path,
    reconcile_owner: Callable[[Mapping[str, object]], bool],
) -> int:
    """Reclaim only unlocked exact owners whose durable lease state was reconciled.

    An inherited lifetime FD keeps a live child protected after parent death.
    Unknown markers, locked attempts and unreconciled database owners are retained.
    """
    import fcntl
    import json
    import shutil

    if not root.exists():
        return 0
    if root.is_symlink():
        raise ValueError("untrusted scratch recovery root")
    root = root.resolve(strict=True)
    with (root / ".admission.lock").open("a+") as admission:
        fcntl.flock(admission, fcntl.LOCK_EX)
        removed = _recover_creation_intent(root=root, journal=admission)
    for path in sorted(root.glob("attempt-*")):
        if not path.is_dir() or path.is_symlink():
            continue
        marker, lifetime = path / "ownership.json", path / "lifetime.lock"
        if marker.is_symlink() or lifetime.is_symlink() or not marker.is_file():
            continue
        try:
            fd = os.open(lifetime, os.O_RDWR | os.O_NOFOLLOW)
        except FileNotFoundError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            try:
                payload = json.loads(marker.read_text())
            except (ValueError, OSError):
                continue
            if (
                set(payload) != {"schema", "owner", "reserved_bytes"}
                or payload["schema"] not in {
                    "backtest-attempt-directory/v1", "backtest-attempt-directory/v2"
                }
                or not isinstance(payload["owner"], dict)
            ):
                continue
            if not reconcile_owner(payload["owner"]):
                continue
            if os.fstat(fd).st_ino != lifetime.stat(follow_symlinks=False).st_ino:
                raise ValueError("attempt lifetime lock changed during recovery")
            with (root / ".admission.lock").open("a") as admission:
                fcntl.flock(admission, fcntl.LOCK_EX)
                shutil.rmtree(path)
                removed += 1
        finally:
            os.close(fd)
    return removed


def _write_creation_intent(journal: TextIO, payload: Mapping[str, object]) -> None:
    import json

    journal.seek(0)
    journal.truncate()
    json.dump(payload, journal)
    journal.flush()
    os.fsync(journal.fileno())


def _recover_creation_intent(*, root: Path, journal: TextIO) -> int:
    """Recover pre-marker creation under the exclusive admission namespace lock.

    Intent is fsynced before mkdir. No child launch or DB admission commit may
    occur until the final marker and directory entries are durable. Unknown
    directories are never discovered by name/age; only this exact recorded name
    may contain unfinished initialization, with no child payloads yet.
    """
    import fcntl
    import json
    import shutil
    from uuid import UUID

    journal.seek(0)
    raw = journal.read()
    if not raw.strip():
        return 0
    try:
        intent = json.loads(raw)
    except ValueError:
        # A torn intent was never fsynced before mkdir, or it was being cleared
        # after a durable final marker. The normal marker scan handles the latter.
        _write_creation_intent(journal, {})
        return 0
    if not intent:
        return 0
    if not isinstance(intent, dict) or set(intent) != {"schema", "name", "owner", "reserved_bytes"}:
        raise ValueError("unrecognized admission creation journal")
    name = intent["name"]
    if (
        intent["schema"] != "backtest-attempt-creation/v1"
        or not isinstance(name, str)
        or not name.startswith("attempt-")
        or UUID(name[8:]).hex != name[8:]
        or not isinstance(intent["owner"], dict)
    ):
        raise ValueError("invalid admission creation identity")
    path = root / name
    if path.is_symlink():
        raise ValueError("creation intent points at a symlink")
    if not path.exists() or (path / "ownership.json").is_file():
        _write_creation_intent(journal, {})
        return 0
    if not path.is_dir() or any(
        child.is_symlink()
        or not child.is_file()
        or child.name not in {"lifetime.lock", "ownership.pending"}
        for child in path.iterdir()
    ):
        raise ValueError("unfinished creation contains foreign payloads")
    fd = -1
    try:
        lifetime = path / "lifetime.lock"
        if lifetime.exists():
            fd = os.open(lifetime, os.O_RDWR | os.O_NOFOLLOW)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        shutil.rmtree(path)
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        _write_creation_intent(journal, {})
        return 1
    finally:
        if fd >= 0:
            os.close(fd)
