from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    PostgresBacktestJobLeaseRepository,
    PostgresBacktestJobRepository,
    PostgresBacktestJobResultsRepository,
)
from trading.contexts.backtest.application.ports import BacktestJobListQuery
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobErrorPayload,
    BacktestJobStageAShortlist,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
from trading.shared_kernel.primitives import UserId


class _FakeGateway:
    """
    Deterministic fake SQL gateway for Backtest job Postgres repository unit tests.

    Docs:
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/gateway.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_repository.py
      - src/trading/contexts/backtest/adapters/outbound/persistence/postgres/
        backtest_job_lease_repository.py
    """

    def __init__(
        self,
        *,
        fetch_one_results: list[Mapping[str, Any] | None | Exception] | None = None,
        fetch_all_results: list[tuple[Mapping[str, Any], ...]] | None = None,
    ) -> None:
        """
        Initialize fake gateway with deterministic queued responses.

        Args:
            fetch_one_results: Sequence of `fetch_one` responses or exceptions.
            fetch_all_results: Sequence of `fetch_all` response tuples.
        Returns:
            None.
        Assumptions:
            Tests control call order and supply enough queued responses.
        Raises:
            None.
        Side Effects:
            Stores mutable response queues and SQL query logs.
        """
        self._fetch_one_results = list(fetch_one_results or [])
        self._fetch_all_results = list(fetch_all_results or [])
        self.fetch_one_queries: list[str] = []
        self.fetch_one_parameters: list[Mapping[str, Any]] = []
        self.fetch_all_queries: list[str] = []
        self.fetch_all_parameters: list[Mapping[str, Any]] = []
        self.execute_queries: list[str] = []

    def fetch_one(self, *, query: str, parameters: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """
        Return next queued `fetch_one` response while recording SQL text.

        Args:
            query: SQL text.
            parameters: SQL parameters mapping.
        Returns:
            Mapping[str, Any] | None: Queued response value.
        Assumptions:
            Queue items are mapping/None/exception.
        Raises:
            Exception: Propagates queued exception values.
        Side Effects:
            Appends SQL query text to in-memory log.
        """
        self.fetch_one_queries.append(query)
        self.fetch_one_parameters.append(parameters)
        if not self._fetch_one_results:
            return None
        result = self._fetch_one_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def fetch_all(
        self,
        *,
        query: str,
        parameters: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        """
        Return next queued `fetch_all` response while recording SQL text.

        Args:
            query: SQL text.
            parameters: SQL parameters mapping.
        Returns:
            tuple[Mapping[str, Any], ...]: Queued response tuple.
        Assumptions:
            Queue items are tuples of mapping rows.
        Raises:
            None.
        Side Effects:
            Appends SQL query text to in-memory log.
        """
        self.fetch_all_queries.append(query)
        self.fetch_all_parameters.append(parameters)
        if not self._fetch_all_results:
            return tuple()
        return self._fetch_all_results.pop(0)

    def execute(self, *, query: str, parameters: Mapping[str, Any]) -> None:
        """
        Record side-effect SQL statement call.

        Args:
            query: SQL text.
            parameters: SQL parameters mapping.
        Returns:
            None.
        Assumptions:
            Tests assert side effects via recorded SQL log.
        Raises:
            None.
        Side Effects:
            Appends SQL query text to execute log.
        """
        self.execute_queries.append(query)
        _ = parameters


def test_lease_repository_claim_uses_skip_locked_and_fifo_order() -> None:
    """
    Verify claim SQL uses `FOR UPDATE SKIP LOCKED` and queued FIFO deterministic order.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Claim order contract is `created_at ASC, job_id ASC` for queued jobs.
    Raises:
        AssertionError: If SQL shape misses required claim clauses.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_one_results=[_build_job_row(state="running")])
    repository = PostgresBacktestJobLeaseRepository(gateway=gateway)

    claimed = repository.claim_next(
        now=datetime(2026, 2, 22, 19, 0, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        lease_seconds=60,
    )

    assert claimed is not None
    assert claimed.state == "running"
    assert "FOR UPDATE SKIP LOCKED" in gateway.fetch_one_queries[0]
    assert "ORDER BY created_at ASC, job_id ASC" in gateway.fetch_one_queries[0]
    assert "lease_expires_at <= %(now)s" in gateway.fetch_one_queries[0]


def test_job_repository_list_for_user_uses_keyset_desc_ordering() -> None:
    """
    Verify list SQL uses deterministic keyset ordering and cursor predicate contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        List order contract is `created_at DESC, job_id DESC` with tuple cursor predicate.
    Raises:
        AssertionError: If ordering/predicate clauses are missing.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_all_results=[(_build_job_row(state="queued"),)])
    repository = PostgresBacktestJobRepository(gateway=gateway)
    user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    cursor = BacktestJobListCursor(
        created_at=datetime(2026, 2, 22, 19, 5, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000811"),
    )

    page = repository.list_for_user(
        query=BacktestJobListQuery(user_id=user_id, limit=20, state=None, cursor=cursor)
    )

    assert len(page.items) == 1
    assert "ORDER BY created_at DESC, job_id DESC" in gateway.fetch_all_queries[0]
    assert "(created_at, job_id) <" in gateway.fetch_all_queries[0]
    assert "%(state)s::text IS NULL" in gateway.fetch_all_queries[0]
    assert "state = %(state)s::text" in gateway.fetch_all_queries[0]
    assert "%(cursor_created_at)s::timestamptz IS NULL" in gateway.fetch_all_queries[0]
    assert "%(cursor_created_at)s::timestamptz" in gateway.fetch_all_queries[0]


def test_job_repository_create_serializes_mappingproxy_request_payload() -> None:
    """Ensure create flow can persist MappingProxyType-backed request_json.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Domain job aggregate stores JSON payloads as immutable mappings.
    Raises:
        AssertionError: If request_json/spec payload serialization is not a string.
    Side Effects:
        None.
    """

    gateway = _FakeGateway(fetch_one_results=[_build_job_row(state="queued")])
    repository = PostgresBacktestJobRepository(gateway=gateway)
    created_at = datetime(2026, 2, 22, 18, 0, tzinfo=timezone.utc)

    job = BacktestJob(
        job_id=UUID("00000000-0000-0000-0000-000000000810"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        state="queued",
        created_at=created_at,
        updated_at=created_at,
        request_json={"mode": "template", "template": {"timeframe": "1m"}},
        request_hash="a" * 64,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=9,
            artifact_manifest_hash="d" * 64,
            artifact_asof_date="2026-03-24",
        ),
        execution_mode="background_manual_legacy",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
        stage="stage_a",
        processed_units=0,
        total_units=0,
    )

    created = repository.create(job=job)

    assert created.job_id == job.job_id
    assert created.artifact_pin is not None
    assert len(gateway.fetch_one_parameters) == 1
    assert isinstance(gateway.fetch_one_parameters[0]["request_json"], str)
    assert gateway.fetch_one_parameters[0]["artifact_slot"] == "slot_b"
    assert gateway.fetch_one_parameters[0]["artifact_slot_generation"] == 9
    assert gateway.fetch_one_parameters[0]["artifact_manifest_hash"] == "d" * 64
    assert gateway.fetch_one_parameters[0]["execution_mode"] == "background_manual_legacy"
    assert gateway.fetch_one_parameters[0]["market_id"] == 1
    assert gateway.fetch_one_parameters[0]["symbol"] == "BTCUSDT"
    assert gateway.fetch_one_parameters[0]["timeframe"] == "1h"
    assert gateway.fetch_one_parameters[0]["requested_top_n"] == 25
    assert gateway.fetch_one_parameters[0]["ranking_primary_metric"] == "profit_factor"
    assert gateway.fetch_one_parameters[0]["ranking_secondary_metric"] == "win_rate_pct"


def test_job_repository_create_with_top_variants_persists_terminal_sync_inline_rows_atomically(
) -> None:
    """
    Verify atomic sync-inline create path inserts job row and summary-only top rows together.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R7-02 persists final sync-inline run metadata and top rows via the existing jobs family.
    Raises:
        AssertionError: If SQL or serialized parameters miss required summary-only clauses.
    Side Effects:
        None.
    """
    job_id = UUID("00000000-0000-0000-0000-000000000910")
    persisted_row = dict(_build_job_row(state="succeeded"))
    persisted_row["job_id"] = str(job_id)
    persisted_row["execution_mode"] = "sync_inline"
    persisted_row["artifact_slot_generation"] = 11
    persisted_row["artifact_asof_date"] = "2026-03-28"
    gateway = _FakeGateway(fetch_one_results=[persisted_row])
    repository = PostgresBacktestJobRepository(gateway=gateway)
    created_at = datetime(2026, 3, 28, 18, 0, tzinfo=timezone.utc)
    finished_at = datetime(2026, 3, 28, 18, 0, 5, tzinfo=timezone.utc)

    job = BacktestJob(
        job_id=job_id,
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        mode="template",
        state="succeeded",
        created_at=created_at,
        updated_at=finished_at,
        started_at=created_at,
        finished_at=finished_at,
        request_json={"mode": "template", "warmup_bars": 200},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=11,
            artifact_manifest_hash="d" * 64,
            artifact_asof_date="2026-03-28",
        ),
        execution_mode="sync_inline",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1h",
        requested_top_n=25,
        ranking_primary_metric="profit_factor",
        ranking_secondary_metric="win_rate_pct",
        stage="finalizing",
        processed_units=100,
        total_units=100,
        progress_updated_at=finished_at,
        locked_by=None,
        locked_at=None,
        lease_expires_at=None,
        heartbeat_at=None,
        attempt=1,
    )
    top_rows = (
        BacktestJobTopVariant(
            job_id=job_id,
            rank=1,
            variant_key="a" * 64,
            indicator_variant_key="b" * 64,
            variant_index=0,
            total_return_pct=12.34,
            payload_json={"schema_version": 1},
            summary_metrics_json={"total_return_pct": 12.34, "profit_factor": 1.23},
            best_tp_pct=4.0,
            best_sl_pct=2.0,
            report_table_md=None,
            trades_json=None,
            updated_at=finished_at,
        ),
    )

    persisted = repository.create_with_top_variants(job=job, top_variants=top_rows)

    assert persisted.state == "succeeded"
    assert gateway.fetch_one_parameters[0]["execution_mode"] == "sync_inline"
    assert isinstance(gateway.fetch_one_parameters[0]["rows_json"], str)
    assert "INSERT INTO backtest_jobs" in gateway.fetch_one_queries[0]
    assert "INSERT INTO backtest_job_top_variants" in gateway.fetch_one_queries[0]
    assert "item -> 'summary_metrics_json' AS summary_metrics_json" in gateway.fetch_one_queries[0]
    assert "NULL::TEXT AS report_table_md" in gateway.fetch_one_queries[0]
    assert "NULL::JSONB AS trades_json" in gateway.fetch_one_queries[0]
    assert "ORDER BY" in gateway.fetch_one_queries[0]
    assert "(item ->> 'rank')::INTEGER ASC" in gateway.fetch_one_queries[0]
    assert "(item ->> 'variant_key') ASC" in gateway.fetch_one_queries[0]


def test_job_repository_count_active_for_artifact_manifest_uses_pin_and_instrument_filters(
) -> None:
    """
    Verify publish-guard count SQL filters active jobs by pinned slot/hash and instrument.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Template-mode jobs read instrument identity from `request_json`, saved-mode jobs from
        `spec_payload_json`.
    Raises:
        AssertionError: If SQL shape misses pin or instrument predicates.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_one_results=[{"active_total": 2}])
    repository = PostgresBacktestJobRepository(gateway=gateway)

    blocked_total = repository.count_active_for_artifact_manifest(
        market_id=1,
        symbol="BTCUSDT",
        artifact_slot="slot_b",
        artifact_manifest_hash="d" * 64,
    )

    assert blocked_total == 2
    assert "state IN ('queued', 'running')" in gateway.fetch_one_queries[0]
    assert (
        "execution_mode IN ('background_auto', 'background_manual_legacy')"
        in gateway.fetch_one_queries[0]
    )
    assert "artifact_slot = %(artifact_slot)s" in gateway.fetch_one_queries[0]
    assert "artifact_manifest_hash = %(artifact_manifest_hash)s" in gateway.fetch_one_queries[0]
    assert "market_id = %(market_id)s" in gateway.fetch_one_queries[0]
    assert "symbol = %(symbol)s" in gateway.fetch_one_queries[0]
    assert "request_json -> 'template' -> 'instrument_id'" in gateway.fetch_one_queries[0]
    assert "spec_payload_json -> 'instrument_id'" in gateway.fetch_one_queries[0]


def test_job_repository_cancel_preserves_existing_running_cancel_marker() -> None:
    """
    Verify running cancel SQL keeps the first `cancel_requested_at` marker unchanged.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        R8-03 keeps running cancel idempotent while queued cancel still transitions directly to
        `cancelled`.
    Raises:
        AssertionError: If SQL shape would overwrite an existing running cancel marker.
    Side Effects:
        None.
    """
    first_cancel_at = datetime(2026, 3, 29, 12, 5, tzinfo=timezone.utc)
    persisted_row = dict(_build_job_row(state="running"))
    persisted_row["cancel_requested_at"] = first_cancel_at
    persisted_row["updated_at"] = first_cancel_at
    gateway = _FakeGateway(
        fetch_one_results=[persisted_row]
    )
    repository = PostgresBacktestJobRepository(gateway=gateway)

    updated = repository.cancel(
        job_id=UUID("00000000-0000-0000-0000-000000000810"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111"),
        cancel_requested_at=datetime(2026, 3, 29, 12, 6, tzinfo=timezone.utc),
    )

    assert updated is not None
    assert updated.state == "running"
    assert updated.cancel_requested_at == first_cancel_at
    assert "cancel_requested_at IS NOT NULL" in gateway.fetch_one_queries[0]
    assert "THEN cancel_requested_at" in gateway.fetch_one_queries[0]
    assert "THEN updated_at" in gateway.fetch_one_queries[0]


def test_lease_repository_heartbeat_uses_active_lease_owner_predicate() -> None:
    """
    Verify heartbeat SQL enforces `(job_id, locked_by, lease_expires_at > now)` predicate.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Heartbeat writes are allowed only for active lease owner.
    Raises:
        AssertionError: If predicate clauses are missing.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_one_results=[_build_job_row(state="running")])
    repository = PostgresBacktestJobLeaseRepository(gateway=gateway)

    row = repository.heartbeat(
        job_id=UUID("00000000-0000-0000-0000-000000000810"),
        now=datetime(2026, 2, 22, 19, 0, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        lease_seconds=60,
    )

    assert row is not None
    assert "locked_by = %(locked_by)s" in gateway.fetch_one_queries[0]
    assert "lease_expires_at > %(now)s" in gateway.fetch_one_queries[0]
    assert "state = 'running'" in gateway.fetch_one_queries[0]


def test_lease_repository_finish_uses_active_lease_owner_predicate() -> None:
    """
    Verify finish SQL enforces active lease owner predicate and writes failed error payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Terminal writes are conditional on lease owner and active lease.
    Raises:
        AssertionError: If predicate or payload clauses are missing.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_one_results=[_build_job_row(state="failed")])
    repository = PostgresBacktestJobLeaseRepository(gateway=gateway)

    finished = repository.finish(
        job_id=UUID("00000000-0000-0000-0000-000000000810"),
        now=datetime(2026, 2, 22, 19, 1, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        next_state="failed",
        last_error="Execution failed",
        last_error_json=BacktestJobErrorPayload(
            code="validation_error",
            message="Execution failed",
            details={"stage": "stage_b"},
        ),
    )

    assert finished is not None
    assert finished.state == "failed"
    assert "state = 'running'" in gateway.fetch_one_queries[0]
    assert "lease_expires_at > %(now)s" in gateway.fetch_one_queries[0]
    assert "last_error_json" in gateway.fetch_one_queries[0]


def test_results_repository_replace_snapshot_uses_delete_insert_single_statement() -> None:
    """
    Verify top-k snapshot replace SQL includes delete+insert in one CTE statement.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Snapshot write policy replaces whole top-k snapshot transactionally.
    Raises:
        AssertionError: If SQL shape misses delete/insert or lease predicate clauses.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_one_results=[{"applied": True}])
    repository = PostgresBacktestJobResultsRepository(gateway=gateway)
    job_id = UUID("00000000-0000-0000-0000-000000000810")

    changed = repository.replace_top_variants_snapshot(
        job_id=job_id,
        now=datetime(2026, 2, 22, 19, 2, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        rows=(
            BacktestJobTopVariant(
                job_id=job_id,
                rank=1,
                variant_key="a" * 64,
                indicator_variant_key="b" * 64,
                variant_index=0,
                total_return_pct=12.34,
                payload_json={"schema_version": 1},
                summary_metrics_json={
                    "total_return_pct": 12.34,
                    "profit_factor": 1.23,
                },
                best_tp_pct=4.0,
                best_sl_pct=2.0,
                report_table_md=None,
                trades_json=None,
                updated_at=datetime(2026, 2, 22, 19, 2, tzinfo=timezone.utc),
            ),
        ),
    )

    assert changed is True
    assert "DELETE FROM backtest_job_top_variants" in gateway.fetch_one_queries[0]
    assert "INSERT INTO backtest_job_top_variants" in gateway.fetch_one_queries[0]
    assert "state = 'running'" in gateway.fetch_one_queries[0]
    assert "lease_expires_at > %(now)s" in gateway.fetch_one_queries[0]
    assert "item -> 'summary_metrics_json' AS summary_metrics_json" in gateway.fetch_one_queries[0]
    assert "NULL::TEXT AS report_table_md" in gateway.fetch_one_queries[0]
    assert "NULL::JSONB AS trades_json" in gateway.fetch_one_queries[0]


def test_job_repository_get_accepts_legacy_rows_with_null_persisted_run_metadata() -> None:
    """
    Verify row mapper keeps legacy jobs readable when additive persisted-run columns stay null.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Migration rollout must not require destructive backfill for existing rows.
    Raises:
        AssertionError: If null additive metadata breaks job mapping.
    Side Effects:
        None.
    """
    legacy_row = dict(_build_job_row(state="queued"))
    legacy_row["execution_mode"] = None
    legacy_row["market_id"] = None
    legacy_row["symbol"] = None
    legacy_row["timeframe"] = None
    legacy_row["requested_top_n"] = None
    legacy_row["ranking_primary_metric"] = None
    legacy_row["ranking_secondary_metric"] = None
    gateway = _FakeGateway(fetch_one_results=[legacy_row])
    repository = PostgresBacktestJobRepository(gateway=gateway)

    row = repository.get(job_id=UUID("00000000-0000-0000-0000-000000000810"))

    assert row is not None
    assert row.execution_mode is None
    assert row.market_id is None
    assert row.symbol is None
    assert row.timeframe is None
    assert row.requested_top_n is None
    assert row.ranking_primary_metric is None
    assert row.ranking_secondary_metric is None


def test_results_repository_list_top_variants_maps_legacy_rows_to_summary_only_shape() -> None:
    """
    Verify legacy top rows with null additive columns are still readable as summary-only rows.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Legacy rows may miss `summary_metrics_json/best_tp_pct/best_sl_pct` during transition.
    Raises:
        AssertionError: If mapper fails to synthesize deterministic summary-only fields.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(
        fetch_all_results=[
            (
                {
                    "job_id": "00000000-0000-0000-0000-000000000810",
                    "rank": 1,
                    "variant_key": "a" * 64,
                    "indicator_variant_key": "b" * 64,
                    "variant_index": 0,
                    "total_return_pct": 12.34,
                    "payload_json": {
                        "schema_version": 1,
                        "risk_params": {
                            "tp_enabled": True,
                            "tp_pct": 4.0,
                            "sl_enabled": True,
                            "sl_pct": 2.0,
                        },
                    },
                    "summary_metrics_json": None,
                    "best_tp_pct": None,
                    "best_sl_pct": None,
                    "updated_at": datetime(2026, 2, 22, 19, 2, tzinfo=timezone.utc),
                },
            )
        ]
    )
    repository = PostgresBacktestJobResultsRepository(gateway=gateway)

    rows = repository.list_top_variants(
        job_id=UUID("00000000-0000-0000-0000-000000000810"),
        limit=10,
    )

    assert len(rows) == 1
    assert rows[0].summary_metrics_json["total_return_pct"] == 12.34
    assert rows[0].best_tp_pct == 4.0
    assert rows[0].best_sl_pct == 2.0
    assert rows[0].report_table_md is None
    assert rows[0].trades_json is None


def test_results_repository_save_stage_a_shortlist_uses_lease_guarded_upsert() -> None:
    """
    Verify Stage-A shortlist SQL uses lease guard and deterministic ON CONFLICT upsert.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Stage-A shortlist is persisted only by active lease owner.
    Raises:
        AssertionError: If SQL shape misses lease predicate or upsert clause.
    Side Effects:
        None.
    """
    gateway = _FakeGateway(fetch_one_results=[{"job_id": "00000000-0000-0000-0000-000000000810"}])
    repository = PostgresBacktestJobResultsRepository(gateway=gateway)
    job_id = UUID("00000000-0000-0000-0000-000000000810")

    applied = repository.save_stage_a_shortlist(
        job_id=job_id,
        now=datetime(2026, 2, 22, 19, 3, tzinfo=timezone.utc),
        locked_by="worker-a-1",
        shortlist=BacktestJobStageAShortlist(
            job_id=job_id,
            stage_a_indexes=(1, 3, 8),
            stage_a_variants_total=100,
            risk_total=4,
            preselect_used=20,
            updated_at=datetime(2026, 2, 22, 19, 3, tzinfo=timezone.utc),
        ),
    )

    assert applied is True
    assert "ON CONFLICT (job_id)" in gateway.fetch_one_queries[0]
    assert "lease_expires_at > %(now)s" in gateway.fetch_one_queries[0]
    assert "state = 'running'" in gateway.fetch_one_queries[0]


def _build_job_row(*, state: str) -> Mapping[str, Any]:
    """
    Build deterministic Backtest jobs SQL row fixture.

    Args:
        state: Job state literal to embed.
    Returns:
        Mapping[str, Any]: Row payload matching `backtest_jobs` schema.
    Assumptions:
        Running rows include lease fields; terminal rows clear lease fields.
    Raises:
        None.
    Side Effects:
        None.
    """
    running_like = state == "running"
    terminal_like = state in {"succeeded", "failed", "cancelled"}
    created_at = datetime(2026, 2, 22, 18, 0, tzinfo=timezone.utc)
    updated_at = (
        datetime(2026, 2, 22, 18, 6, tzinfo=timezone.utc)
        if terminal_like
        else datetime(2026, 2, 22, 18, 1, tzinfo=timezone.utc)
    )

    return {
        "job_id": "00000000-0000-0000-0000-000000000810",
        "user_id": "00000000-0000-0000-0000-000000000111",
        "mode": "template",
        "state": state,
        "created_at": created_at,
        "updated_at": updated_at,
        "started_at": datetime(2026, 2, 22, 18, 0, 10, tzinfo=timezone.utc)
        if state != "queued"
        else None,
        "finished_at": datetime(2026, 2, 22, 18, 5, tzinfo=timezone.utc) if terminal_like else None,
        "cancel_requested_at": None,
        "request_json": {"mode": "template"},
        "request_hash": "a" * 64,
        "spec_hash": None,
        "spec_payload_json": None,
        "engine_params_hash": "b" * 64,
        "backtest_runtime_config_hash": "c" * 64,
        "artifact_slot": "slot_b",
        "artifact_slot_generation": 9,
        "artifact_manifest_hash": "d" * 64,
        "artifact_asof_date": "2026-03-24",
        "execution_mode": "background_manual_legacy",
        "market_id": 1,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "requested_top_n": 25,
        "ranking_primary_metric": "profit_factor",
        "ranking_secondary_metric": "win_rate_pct",
        "stage": (
            "stage_b" if state == "running" else "finalizing" if state == "succeeded" else "stage_a"
        ),
        "processed_units": 10,
        "total_units": 100,
        "progress_updated_at": updated_at,
        "locked_by": "worker-a-1" if running_like else None,
        "locked_at": datetime(2026, 2, 22, 18, 0, 10, tzinfo=timezone.utc)
        if running_like
        else None,
        "lease_expires_at": datetime(2026, 2, 22, 18, 2, tzinfo=timezone.utc)
        if running_like
        else None,
        "heartbeat_at": datetime(2026, 2, 22, 18, 1, tzinfo=timezone.utc) if running_like else None,
        "attempt": 1,
        "last_error": "Execution failed" if state == "failed" else None,
        "last_error_json": {
            "code": "validation_error",
            "message": "Execution failed",
            "details": {"stage": "stage_b"},
        }
        if state == "failed"
        else None,
    }
