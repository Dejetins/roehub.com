"""Narrow browser lifecycle policy for explicitly configured background workers."""

from datetime import datetime

from apps.monitoring.operational_health import OperationalStatus

MANAGED_WORKERS = frozenset(
    {
        "market-data-ws",
        "market-data-scheduler",
        "backtest-artifact-publisher",
    }
)


def worker_commands(status: OperationalStatus, now: datetime) -> tuple[str, ...]:
    age = (now - status.observed_at).total_seconds()
    if (
        status.service_id not in MANAGED_WORKERS
        or status.action_ref != "manage_worker"
        or not 0 <= age <= 30
        or status.detail_code == "probe.snapshot_stale"
    ):
        return ()
    # Explicit start/stop are idempotent control requests. Probe reachability alone
    # cannot establish process state or job readiness, so do not infer it here.
    return ("start", "stop")
