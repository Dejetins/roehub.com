"""OS capability discovery without consulting the parent numerical runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trading.contexts.backtest.application.services.v2.job_scheduling import (
    ROEHUB_BACKTEST_EFFECTIVE_NUMBA_NUM_THREADS,
    BacktestNumbaThreadDecision,
    backtest_numba_environ,
)

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
