"""Metrics from persisted states; never estimate progress from elapsed duration."""

from prometheus_client import CollectorRegistry, Gauge


class ManagedArtifactMetrics:
    def __init__(self, registry: CollectorRegistry):
        self.requests = Gauge(
            "backtest_artifact_managed_requests",
            "Persisted request count",
            ["state"],
            registry=registry,
        )
        self.available = Gauge(
            "backtest_artifact_managed_available",
            "Submission capability available",
            registry=registry,
        )

    def sample(self, repository, installation):
        settings = repository.rows(
            "SELECT * FROM backtest_artifact_installation_settings WHERE installation_id=%(i)s",
            i=installation,
        )
        active = bool(
            settings
            and settings[0]["capability_enabled"]
            and not settings[0]["paused"]
            and settings[0]["legacy_handoff"] == "complete"
        )
        self.available.set(int(active))
        counts = {
            r["state"]: r["n"]
            for r in repository.rows(
                (
                    "SELECT state,count(*) n FROM backtest_artifact_requests WHERE "
                    "installation_id=%(i)s GROUP BY state"
                ),
                i=installation,
            )
        }
        for state in ("waiting_for_source", "queued", "running", "retry_wait", "ready", "error"):
            self.requests.labels(state=state).set(counts.get(state, 0))
