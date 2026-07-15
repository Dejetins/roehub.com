from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_isolated_job_runtime_migration_has_durable_scope_and_limits() -> None:
    sql = (_REPO_ROOT / "migrations/postgres/0019_isolated_job_runtime_v1.sql").read_text()

    for required in (
        "CREATE TABLE job_runtime_jobs",
        "CREATE TABLE job_runtime_attempts",
        "FOREIGN KEY (organization_id, job_id)",
        "UNIQUE (organization_id, semantic_job_key)",
        "artifact_store_manifests",
        "JobEnvelope/v1",
        "image_digest ~ '^sha256:[0-9a-f]{64}$'",
        "envelope ->> 'network' = 'none'",
        "idx_job_runtime_attempts_claim",
        "idx_job_runtime_attempts_recovery",
        "CREATE FUNCTION job_runtime_jobs_guard()",
        "CREATE FUNCTION job_runtime_attempts_guard()",
        "job attempt envelope and identity are immutable",
        "terminal job attempt is immutable",
        "'recovering'",
    ):
        assert required in sql
