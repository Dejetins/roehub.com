from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

import pytest

from trading.contexts.backtest.application.dto import (
    BacktestVariantPayloadV1,
    BacktestVariantPreview,
    RunBacktestRequest,
    RunBacktestResponse,
    RunBacktestTemplate,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    CurrentUser,
)
from trading.contexts.backtest.application.use_cases.backtest_jobs_api_v1 import (
    _build_request_hash_from_request_json,
    _build_sha256_from_payload,
)
from trading.contexts.backtest.application.use_cases.backtest_runs_api_v1 import (
    CreateAndRunBacktestSyncInlineUseCase,
    LaunchBacktestRunWithAutoFallbackUseCase,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobArtifactPin,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    Symbol,
    Timeframe,
    TimeRange,
    UserId,
    UtcTimestamp,
)


@dataclass
class _FakeRunUseCase:
    """
    Minimal deterministic sync run use-case fake for persisted sync-inline orchestration tests.
    """

    response: RunBacktestResponse | None = None
    error: Exception | None = None
    last_request_payload: Mapping[str, Any] | None = None

    def execute(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        request_payload: Mapping[str, Any] | None = None,
        run_control=None,
    ) -> RunBacktestResponse:
        """
        Return preconfigured sync response or raise configured canonical error.

        Args:
            request: Parsed run request DTO.
            current_user: Authenticated owner identity.
            request_payload: Optional strict API payload snapshot.
            run_control: Optional cooperative cancellation handle.
        Returns:
            RunBacktestResponse: Preconfigured sync response fixture.
        Assumptions:
            Persistence orchestration tests do not execute real staged compute.
        Raises:
            Exception: Propagates configured fake error.
        Side Effects:
            Stores the last captured `request_payload` for orchestration assertions.
        """
        _ = request, current_user, run_control
        self.last_request_payload = request_payload
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response

    def build_variant_report(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        variant_payload: Any,
        include_trades: bool = False,
        run_control=None,
    ):
        """
        Reject unexpected lazy report calls in these orchestration-focused tests.

        Args:
            request: Parsed run request DTO.
            current_user: Authenticated owner identity.
            variant_payload: Explicit variant payload.
            include_trades: Include-trades flag.
            run_control: Optional cooperative cancellation handle.
        Returns:
            None.
        Assumptions:
            These tests exercise only `execute(...)`.
        Raises:
            AssertionError: Always, because report generation is out of scope here.
        Side Effects:
            None.
        """
        _ = request, current_user, variant_payload, include_trades, run_control
        raise AssertionError("build_variant_report is not expected in this test")


@dataclass
class _FakeJobRepository:
    """
    Minimal in-memory repository fake capturing persisted sync-inline terminal snapshots.
    """

    created_job: BacktestJob | None = None
    created_rows: tuple[BacktestJobTopVariant, ...] = tuple()

    def create(self, *, job: BacktestJob) -> BacktestJob:
        """
        Reject unexpected queued/background create calls in sync-inline orchestration tests.

        Args:
            job: Queued job snapshot.
        Returns:
            BacktestJob: Echoed job snapshot when used unexpectedly.
        Assumptions:
            This test module exercises only `create_with_top_variants(...)`.
        Raises:
            AssertionError: Always, because plain `create(...)` is out of scope here.
        Side Effects:
            None.
        """
        _ = job
        raise AssertionError("create is not expected in this test")

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
    ) -> BacktestJob:
        """
        Record atomic persisted sync-inline snapshot payload and echo the terminal job row.

        Args:
            job: Terminal job snapshot.
            top_variants: Summary-only persisted top rows.
        Returns:
            BacktestJob: Echoed terminal job snapshot.
        Assumptions:
            Tests assert captured side effects directly from in-memory state.
        Raises:
            None.
        Side Effects:
            Stores last persisted job row and top rows tuple.
        """
        self.created_job = job
        self.created_rows = top_variants
        return job

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        """
        Return last persisted job when ids match; otherwise return `None`.

        Args:
            job_id: Requested job identifier.
            user_id: Optional owner filter.
        Returns:
            BacktestJob | None: Captured persisted job or `None`.
        Assumptions:
            Tests use only one persisted job snapshot at a time.
        Raises:
            None.
        Side Effects:
            None.
        """
        if self.created_job is None or self.created_job.job_id != job_id:
            return None
        if user_id is not None and self.created_job.user_id != user_id:
            return None
        return self.created_job

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        """
        Return empty deterministic page because list semantics are out of scope here.

        Args:
            query: User list query payload.
        Returns:
            BacktestJobListPage: Empty page fixture.
        Assumptions:
            Sync-inline orchestration tests do not exercise history/list reads.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = query
        return BacktestJobListPage(items=tuple(), next_cursor=None)

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        """
        Reject unexpected cancel calls in sync-inline orchestration tests.

        Args:
            job_id: Requested job identifier.
            user_id: Owner identifier.
            cancel_requested_at: Cancel timestamp.
        Returns:
            BacktestJob | None: Never returns because this path is not expected.
        Assumptions:
            Sync-inline success/failure tests do not exercise cancel semantics.
        Raises:
            AssertionError: Always, because cancel is out of scope here.
        Side Effects:
            None.
        """
        _ = job_id, user_id, cancel_requested_at
        raise AssertionError("cancel is not expected in this test")

    def count_active_for_user(self, *, user_id: UserId) -> int:
        """
        Return zero active jobs because quota semantics are out of scope here.

        Args:
            user_id: Owner identifier.
        Returns:
            int: Always `0`.
        Assumptions:
            Sync-inline orchestration tests do not exercise active-jobs quotas.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = user_id
        return 0

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return zero because publish-guard counting is out of scope for these tests.

        Args:
            market_id: Canonical market id.
            symbol: Canonical symbol.
            artifact_slot: Candidate slot literal.
            artifact_manifest_hash: Candidate manifest hash.
        Returns:
            int: Always `0`.
        Assumptions:
            Sync-inline orchestration tests do not exercise inactive-slot publish guards.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


@dataclass
class _FakePreflightUseCase:
    """
    Minimal deterministic full-budget preflight fake for auto-fallback orchestration tests.
    """

    error: Exception | None = None
    calls: int = 0

    def preflight(
        self,
        *,
        request: RunBacktestRequest,
        current_user: CurrentUser,
        run_control=None,
    ) -> None:
        """
        Record call and raise configured error when requested.

        Args:
            request: Parsed run request DTO.
            current_user: Authenticated owner identity.
            run_control: Optional cooperative cancellation handle.
        Returns:
            None.
        Assumptions:
            Auto-fallback tests assert only call count and propagated errors.
        Raises:
            Exception: Configured fake error.
        Side Effects:
            Increments in-memory call counter.
        """
        _ = request, current_user, run_control
        self.calls += 1
        if self.error is not None:
            raise self.error


@dataclass
class _FakeBackgroundCreateUseCase:
    """
    Minimal queued background-run creator fake for auto-fallback orchestration tests.
    """

    created_job: BacktestJob | None = None
    calls: int = 0
    last_command: Any | None = None
    last_current_user: CurrentUser | None = None

    def execute(self, *, command, current_user: CurrentUser) -> BacktestJob:
        """
        Return configured queued run snapshot and record command for assertions.

        Args:
            command: Canonical queued background-run create command.
            current_user: Authenticated owner identity.
        Returns:
            BacktestJob: Preconfigured queued background run snapshot.
        Assumptions:
            Auto-fallback orchestration tests inspect the captured command directly.
        Raises:
            AssertionError: If the fake was not configured with a created job.
        Side Effects:
            Stores last command and increments in-memory call counter.
        """
        self.calls += 1
        self.last_command = command
        self.last_current_user = current_user
        if self.created_job is None:
            raise AssertionError("created_job is not configured")
        return self.created_job


def test_create_and_run_backtest_sync_inline_persists_run_and_summary_rows() -> None:
    """
    Verify orchestrator persists terminal sync-inline run row and summary-only top rows.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Inner sync use-case already performed preflight and returned slot-pinned artifact metadata.
    Raises:
        AssertionError: If persisted run metadata or summary-only top rows drift from contract.
    Side Effects:
        None.
    """
    repo = _FakeJobRepository()
    response = _template_run_response()
    now_values = iter(
        (
            datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 28, 12, 0, 3, tzinfo=timezone.utc),
        )
    )
    run_use_case = _FakeRunUseCase(response=response)
    use_case = CreateAndRunBacktestSyncInlineUseCase(
        run_use_case=run_use_case,
        job_repository=repo,
        backtest_runtime_config_hash="f" * 64,
        engine_version="signal_tf + 1m_risk",
        now_provider=lambda: next(now_values),
        run_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000910"),
    )

    persisted = use_case.execute(
        request=_template_request(),
        current_user=CurrentUser(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
        ),
        request_payload=_template_request_payload(),
    )

    assert persisted.run_id == UUID("00000000-0000-0000-0000-000000000910")
    assert persisted.state == "succeeded"
    assert persisted.execution_mode == "sync_inline"
    assert persisted.execution_profile_mode == "exact_no_risk_parity"
    assert persisted.engine_version == "signal_tf + 1m_risk"
    assert persisted.artifact_slot == "slot_b"
    assert persisted.artifact_slot_generation == 11
    assert persisted.artifact_asof_date == "2026-03-28"
    assert persisted.artifact_manifest_hash == "c" * 64
    assert run_use_case.last_request_payload is not None
    assert run_use_case.last_request_payload["execution_profile_mode"] == "exact_no_risk_parity"
    assert repo.created_job is not None
    assert repo.created_job.execution_mode == "sync_inline"
    assert repo.created_job.state == "succeeded"
    assert repo.created_job.market_id == 1
    assert repo.created_job.symbol == "BTCUSDT"
    assert repo.created_job.timeframe == "1m"
    assert repo.created_job.requested_top_n == 2
    assert repo.created_job.ranking_primary_metric == "total_return_pct"
    assert repo.created_job.ranking_secondary_metric is None
    assert repo.created_job.artifact_pin is not None
    assert repo.created_job.artifact_pin.artifact_slot == "slot_b"
    assert repo.created_job.artifact_pin.artifact_slot_generation == 11
    assert repo.created_job.artifact_pin.artifact_manifest_hash == "c" * 64
    assert repo.created_job.artifact_pin.artifact_asof_date == "2026-03-28"
    assert repo.created_job.request_json["template"]["execution"]["fee_pct"] == 0.075
    assert repo.created_job.request_json["template"]["direction_mode"] == "long-short"
    assert "execution_profile_mode" not in repo.created_job.request_json
    assert repo.created_job.execution_profile_mode_hint == "exact_no_risk_parity"
    assert repo.created_job.effective_execution_profile_mode == "exact_no_risk_parity"
    assert repo.created_job.request_hash == _build_sha256_from_payload(
        payload=repo.created_job.request_json
    )
    assert len(repo.created_rows) == 1
    assert repo.created_rows[0].rank == 1
    assert repo.created_rows[0].variant_key == "a" * 64
    assert repo.created_rows[0].report_table_md is None
    assert repo.created_rows[0].trades_json is None
    assert repo.created_rows[0].summary_metrics_json["win_rate_pct"] == 60.0
    assert repo.created_rows[0].best_tp_pct == 4.0
    assert repo.created_rows[0].best_sl_pct == 2.0


def test_create_and_run_backtest_sync_inline_forces_redesigned_internal_profile() -> None:
    """
    Verify persisted `POST /backtests` sync launch forces the parity-first no-risk exact profile.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Public launch payload stays unchanged, so the D0 corrective split must happen via additive
        internal metadata only.
    Raises:
        AssertionError: If sync-inline launch forwards or persists hybrid_conservative instead of
            the new parity-first `exact_no_risk_parity` mode.
    Side Effects:
        None.
    """
    repo = _FakeJobRepository()
    run_use_case = _FakeRunUseCase(response=_template_run_response())
    use_case = CreateAndRunBacktestSyncInlineUseCase(
        run_use_case=run_use_case,
        job_repository=repo,
        backtest_runtime_config_hash="f" * 64,
        engine_version="signal_tf + 1m_risk",
        now_provider=lambda: datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
        run_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000910"),
    )

    use_case.execute(
        request=_template_request(),
        current_user=CurrentUser(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
        ),
        request_payload={
            **_template_request_payload(),
            "execution_profile_mode": "exact_parallel",
        },
    )

    assert run_use_case.last_request_payload is not None
    assert run_use_case.last_request_payload["execution_profile_mode"] == "exact_no_risk_parity"
    assert repo.created_job is not None
    assert "execution_profile_mode" not in repo.created_job.request_json
    assert repo.created_job.execution_profile_mode_hint == "exact_no_risk_parity"
    assert repo.created_job.effective_execution_profile_mode == "exact_no_risk_parity"


def test_request_hash_ignores_internal_execution_profile_mode_across_exact_and_hybrid_modes(
) -> None:
    """
    Verify internal execution-profile routing metadata does not affect canonical request hashes.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
      - docs/architecture/backtest/backtest-hybrid-shortlist-runtime-v1.md
    Related:
      - tests/unit/contexts/backtest/application/use_cases/test_backtest_runs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Internal `execution_profile_mode` may switch between exact and hybrid runtime paths but
        must remain excluded from request-hash semantics.
    Raises:
        AssertionError: If canonical request hashes diverge only because internal profile
            metadata changes.
    Side Effects:
        None.
    """
    base_payload = _template_request_payload()
    exact_hash = _build_request_hash_from_request_json(
        payload={**base_payload, "execution_profile_mode": "exact_small"}
    )
    hybrid_hash = _build_request_hash_from_request_json(
        payload={**base_payload, "execution_profile_mode": "hybrid_conservative"}
    )
    canonical_hash = _build_sha256_from_payload(payload=base_payload)

    assert exact_hash == canonical_hash
    assert hybrid_hash == canonical_hash


def test_create_and_run_backtest_sync_inline_keeps_preflight_validation_error_without_persistence(
) -> None:
    """
    Verify canonical preflight validation failure short-circuits before any persistence writes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Inner sync use-case remains the source of deterministic budget/preflight failures.
    Raises:
        AssertionError: If repository write is attempted after validation failure.
    Side Effects:
        None.
    """
    repo = _FakeJobRepository()
    use_case = CreateAndRunBacktestSyncInlineUseCase(
        run_use_case=_FakeRunUseCase(
            error=BacktestValidationError("Backtest request exceeds compute budget")
        ),
        job_repository=repo,
        backtest_runtime_config_hash="f" * 64,
        engine_version="signal_tf + 1m_risk",
        now_provider=lambda: datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
        run_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000910"),
    )

    with pytest.raises(BacktestValidationError, match="exceeds compute budget"):
        use_case.execute(
            request=_template_request(),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
            ),
            request_payload=_template_request_payload(),
        )

    assert repo.created_job is None
    assert repo.created_rows == tuple()


def test_launch_backtest_with_auto_fallback_returns_sync_inline_when_sync_budgets_pass() -> None:
    """
    Verify orchestration returns sync-inline result unchanged when half-budgets pass.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Sync-inline orchestrator already persisted run metadata before auto-fallback wrapper.
    Raises:
        AssertionError: If fallback preflight/create paths are touched on sync success.
    Side Effects:
        None.
    """
    sync_response = replace(
        _template_run_response(),
        run_id=UUID("00000000-0000-0000-0000-000000000910"),
        state="succeeded",
        execution_mode="sync_inline",
        engine_version="signal_tf + 1m_risk",
        engine_params_hash="d" * 64,
    )
    preflight_use_case = _FakePreflightUseCase()
    create_use_case = _FakeBackgroundCreateUseCase()
    use_case = LaunchBacktestRunWithAutoFallbackUseCase(
        sync_inline_use_case=_FakeRunUseCase(response=sync_response),
        background_preflight_use_case=preflight_use_case,
        background_create_use_case=create_use_case,
        engine_version="signal_tf + 1m_risk",
    )

    persisted = use_case.execute(
        request=_template_request(),
        current_user=CurrentUser(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
        ),
        request_payload=_template_request_payload(),
    )

    assert persisted.execution_mode == "sync_inline"
    assert persisted.run_id == UUID("00000000-0000-0000-0000-000000000910")
    assert preflight_use_case.calls == 0
    assert create_use_case.calls == 0


def test_launch_backtest_with_auto_fallback_keeps_canonical_nr2_sync_inline_response() -> None:
    """
    Verify canonical `NR2` no-risk sync launch stays on `sync_inline` and keeps the internal
    `exact_no_risk_parity` execution profile instead of drifting into `background_auto`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        The sync wrapper should preserve the corrected planner decision end to end once the
        canonical no-risk request is admitted by sync launch budgeting.
    Raises:
        AssertionError: If the launch wrapper mutates the sync response or touches fallback paths.
    Side Effects:
        None.
    """
    sync_response = replace(
        _template_run_response(),
        run_id=UUID("00000000-0000-0000-0000-000000000912"),
        state="succeeded",
        execution_mode="sync_inline",
        execution_profile_mode="exact_no_risk_parity",
        engine_version="signal_tf + 1m_risk",
        engine_params_hash="f" * 64,
    )
    preflight_use_case = _FakePreflightUseCase()
    create_use_case = _FakeBackgroundCreateUseCase()
    use_case = LaunchBacktestRunWithAutoFallbackUseCase(
        sync_inline_use_case=_FakeRunUseCase(response=sync_response),
        background_preflight_use_case=preflight_use_case,
        background_create_use_case=create_use_case,
        engine_version="signal_tf + 1m_risk",
    )

    launched = use_case.execute(
        request=_template_request(),
        current_user=CurrentUser(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
        ),
        request_payload=_template_request_payload(),
    )

    assert launched.execution_mode == "sync_inline"
    assert launched.execution_profile_mode == "exact_no_risk_parity"
    assert launched.run_id == UUID("00000000-0000-0000-0000-000000000912")
    assert preflight_use_case.calls == 0
    assert create_use_case.calls == 0


def test_launch_backtest_with_auto_fallback_routes_heavy_valid_request_to_background_auto(
) -> None:
    """
    Verify sync launch-budget classification can queue heavy-but-valid exact requests earlier.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Sync planner may now signal `background_auto_required` before Stage A/Stage B execution,
        while full-budget preflight remains the hard-reject source of truth.
    Raises:
        AssertionError: If background routing or propagated execution profile drifts.
    Side Effects:
        None.
    """
    preflight_use_case = _FakePreflightUseCase()
    create_use_case = _FakeBackgroundCreateUseCase(created_job=_background_auto_job())
    use_case = LaunchBacktestRunWithAutoFallbackUseCase(
        sync_inline_use_case=_FakeRunUseCase(
            error=RoehubError(
                code="validation_error",
                message=(
                    "Backtest request exceeds sync launch budget and should run in background"
                ),
                details={
                    "error": "background_auto_required",
                    "execution_mode": "background_auto",
                    "execution_profile_mode": "exact_parallel",
                    "stage_a_variants_total": 32000,
                    "stage_b_variants_total": 240000,
                    "estimated_memory_bytes": 2147483648,
                },
            )
        ),
        background_preflight_use_case=preflight_use_case,
        background_create_use_case=create_use_case,
        engine_version="signal_tf + 1m_risk",
    )

    launched = use_case.execute(
        request=_template_request(),
        current_user=CurrentUser(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
        ),
        request_payload=_template_request_payload(),
    )

    assert preflight_use_case.calls == 1
    assert create_use_case.calls == 1
    assert create_use_case.last_command is not None
    assert create_use_case.last_command.execution_mode == "background_auto"
    assert create_use_case.last_command.execution_profile_mode == "exact_parallel"
    assert launched.execution_mode == "background_auto"
    assert launched.execution_profile_mode == "exact_parallel"
    assert launched.run_id == UUID("00000000-0000-0000-0000-000000000911")


def test_launch_backtest_with_auto_fallback_creates_background_auto_after_guard_overflow() -> None:
    """
    Verify guard overflow falls back to explicit queued `background_auto` launch.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Full-budget preflight succeeds and queued run creation reuses existing jobs create flow.
    Raises:
        AssertionError: If fallback metadata or execution mode drift from R8-02 contract.
    Side Effects:
        None.
    """
    preflight_use_case = _FakePreflightUseCase()
    create_use_case = _FakeBackgroundCreateUseCase(
        created_job=_background_auto_job(),
    )
    use_case = LaunchBacktestRunWithAutoFallbackUseCase(
        sync_inline_use_case=_FakeRunUseCase(
            error=RoehubError(
                code="validation_error",
                message="Backtest variants guard exceeded",
                details={
                    "error": "max_variants_per_compute_exceeded",
                    "stage": "stage_a",
                    "total_variants": 999,
                    "max_variants_per_compute": 100,
                    "execution_profile_mode": "exact_parallel",
                },
            )
        ),
        background_preflight_use_case=preflight_use_case,
        background_create_use_case=create_use_case,
        engine_version="signal_tf + 1m_risk",
    )

    launched = use_case.execute(
        request=_template_request(),
        current_user=CurrentUser(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
        ),
        request_payload=_template_request_payload(),
    )

    assert preflight_use_case.calls == 1
    assert create_use_case.calls == 1
    assert create_use_case.last_command is not None
    assert create_use_case.last_command.execution_mode == "background_auto"
    assert create_use_case.last_command.execution_profile_mode == "exact_parallel"
    assert launched.run_id == UUID("00000000-0000-0000-0000-000000000911")
    assert launched.state == "queued"
    assert launched.execution_mode == "background_auto"
    assert launched.engine_version == "signal_tf + 1m_risk"
    assert launched.artifact_slot == "slot_b"
    assert launched.artifact_slot_generation == 11
    assert launched.artifact_asof_date == "2026-03-28"
    assert launched.artifact_manifest_hash == "c" * 64
    assert launched.top_k == 2
    assert launched.preselect == 100
    assert launched.variants == tuple()
    assert launched.engine_params_hash == "e" * 64


def test_launch_backtest_with_auto_fallback_keeps_legacy_request_json_profile_fallback() -> None:
    """
    Verify queued launch response keeps explicit legacy `request_json` fallback for old rows.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Historical persisted rows may predate additive execution-profile metadata and still need
        compatibility fallback during launch-response mapping.
    Raises:
        AssertionError: If the background launch response no longer accepts legacy row snapshots.
    Side Effects:
        None.
    """
    legacy_background_run = replace(
        _background_auto_job(),
        request_json={
            **_template_request_payload(),
            "execution_profile_mode": "exact_parallel",
        },
        execution_profile_mode_hint=None,
        effective_execution_profile_mode=None,
    )
    preflight_use_case = _FakePreflightUseCase()
    create_use_case = _FakeBackgroundCreateUseCase(created_job=legacy_background_run)
    use_case = LaunchBacktestRunWithAutoFallbackUseCase(
        sync_inline_use_case=_FakeRunUseCase(
            error=RoehubError(
                code="validation_error",
                message="Backtest variants guard exceeded",
                details={
                    "error": "max_variants_per_compute_exceeded",
                    "stage": "stage_a",
                    "total_variants": 999,
                    "max_variants_per_compute": 100,
                    "execution_profile_mode": "exact_parallel",
                },
            )
        ),
        background_preflight_use_case=preflight_use_case,
        background_create_use_case=create_use_case,
        engine_version="signal_tf + 1m_risk",
    )

    launched = use_case.execute(
        request=_template_request(),
        current_user=CurrentUser(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
        ),
        request_payload=_template_request_payload(),
    )

    assert preflight_use_case.calls == 1
    assert create_use_case.calls == 1
    assert launched.execution_mode == "background_auto"
    assert launched.execution_profile_mode == "exact_parallel"


def test_launch_backtest_with_auto_fallback_rethrows_full_budget_overflow_without_create() -> None:
    """
    Verify full-budget guard failure returns deterministic `422` without queued run creation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Auto-fallback must stop before persistence when full budgets still overflow.
    Raises:
        AssertionError: If queued background create is attempted after failed full-budget preflight.
    Side Effects:
        None.
    """
    preflight_use_case = _FakePreflightUseCase(
        error=RoehubError(
            code="validation_error",
            message="Backtest memory guard exceeded",
            details={
                "error": "max_compute_bytes_total_exceeded",
                "stage": "stage_a",
                "estimated_memory_bytes": 4096,
                "max_compute_bytes_total": 2048,
            },
        )
    )
    create_use_case = _FakeBackgroundCreateUseCase()
    use_case = LaunchBacktestRunWithAutoFallbackUseCase(
        sync_inline_use_case=_FakeRunUseCase(
            error=RoehubError(
                code="validation_error",
                message="Backtest variants guard exceeded",
                details={
                    "error": "max_variants_per_compute_exceeded",
                    "stage": "stage_b",
                    "total_variants": 1200,
                    "max_variants_per_compute": 500,
                },
            )
        ),
        background_preflight_use_case=preflight_use_case,
        background_create_use_case=create_use_case,
        engine_version="signal_tf + 1m_risk",
    )

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            request=_template_request(),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
            ),
            request_payload=_template_request_payload(),
        )

    assert error_info.value.message == "Backtest memory guard exceeded"
    assert preflight_use_case.calls == 1
    assert create_use_case.calls == 0


def test_launch_backtest_with_auto_fallback_skips_non_guard_validation_errors() -> None:
    """
    Verify non-guard validation errors propagate unchanged without fallback attempts.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Only structured guard overflow literals are eligible for `background_auto`.
    Raises:
        AssertionError: If wrapper masks unrelated validation errors behind fallback behavior.
    Side Effects:
        None.
    """
    preflight_use_case = _FakePreflightUseCase()
    create_use_case = _FakeBackgroundCreateUseCase()
    use_case = LaunchBacktestRunWithAutoFallbackUseCase(
        sync_inline_use_case=_FakeRunUseCase(
            error=RoehubError(
                code="validation_error",
                message="Backtest request top_k must be <= 300",
                details={
                    "errors": [
                        {
                            "path": "body.top_k",
                            "code": "max_value",
                            "message": "top_k must be <= 300",
                        }
                    ]
                },
            )
        ),
        background_preflight_use_case=preflight_use_case,
        background_create_use_case=create_use_case,
        engine_version="signal_tf + 1m_risk",
    )

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            request=_template_request(),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000777")
            ),
            request_payload=_template_request_payload(),
        )

    assert error_info.value.message == "Backtest request top_k must be <= 300"
    assert preflight_use_case.calls == 0
    assert create_use_case.calls == 0


def _template_request() -> RunBacktestRequest:
    """
    Build deterministic template-mode run request used by persisted sync-inline tests.

    Args:
        None.
    Returns:
        RunBacktestRequest: Minimal template-mode request fixture.
    Assumptions:
        One indicator grid is sufficient for sync-inline orchestration unit coverage.
    Raises:
        ValueError: If fixture violates application DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestRequest(
        time_range=TimeRange(
            start=UtcTimestamp(datetime(2026, 3, 28, 0, 0, tzinfo=timezone.utc)),
            end=UtcTimestamp(datetime(2026, 3, 28, 1, 0, tzinfo=timezone.utc)),
        ),
        template=RunBacktestTemplate(
            instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
            timeframe=Timeframe("1m"),
            indicator_grids=(
                GridSpec(
                    indicator_id=IndicatorId("ma.sma"),
                    params={"window": ExplicitValuesSpec(name="window", values=(20,))},
                ),
            ),
            signal_grids={
                "ma.sma": {
                    "cross_up": ExplicitValuesSpec(name="cross_up", values=(0.5,))
                }
            },
            risk_params={
                "sl_enabled": True,
                "sl_pct": 2.0,
                "tp_enabled": True,
                "tp_pct": 4.0,
            },
            execution_params={
                "fee_pct": 0.075,
                "fixed_quote": 100.0,
                "init_cash_quote": 10000.0,
                "safe_profit_percent": 30.0,
                "slippage_pct": 0.01,
            },
        ),
        warmup_bars=200,
        top_k=2,
        preselect=100,
    )


def _template_request_payload() -> Mapping[str, Any]:
    """
    Build strict API-like request snapshot used for persisted `request_json` assertions.

    Args:
        None.
    Returns:
        Mapping[str, Any]: Deterministic JSON-compatible request payload fixture.
    Assumptions:
        Payload shape mirrors strict `POST /backtests` template-mode request contract.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "time_range": {
            "start": "2026-03-28T00:00:00Z",
            "end": "2026-03-28T01:00:00Z",
        },
        "template": {
            "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
            "timeframe": "1m",
            "indicator_grids": [
                {
                    "indicator_id": "ma.sma",
                    "params": {"window": {"mode": "explicit", "values": [20]}},
                }
            ],
            "signal_grids": {
                "ma.sma": {"cross_up": {"mode": "explicit", "values": [0.5]}}
            },
            "risk_grid": {
                "sl_enabled": True,
                "tp_enabled": True,
                "sl": {"mode": "explicit", "values": [2.0]},
                "tp": {"mode": "explicit", "values": [4.0]},
            },
            "execution": {
                "fee_pct": 0.075,
                "fixed_quote": 100.0,
                "init_cash_quote": 10000.0,
                "safe_profit_percent": 30.0,
                "slippage_pct": 0.01,
            },
        },
        "top_k": 2,
        "preselect": 100,
    }


def _template_run_response() -> RunBacktestResponse:
    """
    Build deterministic inner sync response fixture for persisted sync-inline orchestration tests.

    Args:
        None.
    Returns:
        RunBacktestResponse: Completed sync response with slot-pinned artifact metadata.
    Assumptions:
        Ranked variants are already sorted by deterministic top-k contract.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestResponse(
        mode="template",
        strategy_id=None,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        top_k=2,
        preselect=100,
        direction_mode="long-short",
        sizing_mode="all_in",
        execution_params={
            "fee_pct": 0.075,
            "fixed_quote": 100.0,
            "init_cash_quote": 10000.0,
            "safe_profit_percent": 30.0,
            "slippage_pct": 0.01,
        },
        variants=(
            BacktestVariantPreview(
                variant_index=0,
                variant_key="a" * 64,
                indicator_variant_key="b" * 64,
                total_return_pct=12.34,
                payload=BacktestVariantPayloadV1(
                    indicator_selections=(
                        IndicatorVariantSelection(
                            indicator_id="ma.sma",
                            inputs={"source": "close"},
                            params={"window": 20},
                        ),
                    ),
                    signal_params={"ma.sma": {"cross_up": 0.5}},
                    risk_params={
                        "sl_enabled": True,
                        "sl_pct": 2.0,
                        "tp_enabled": True,
                        "tp_pct": 4.0,
                    },
                    execution_params={
                        "fee_pct": 0.075,
                        "fixed_quote": 100.0,
                        "init_cash_quote": 10000.0,
                        "safe_profit_percent": 30.0,
                        "slippage_pct": 0.01,
                    },
                    direction_mode="long-short",
                    sizing_mode="all_in",
                ),
                summary_metrics_json={
                    "profit_factor": 1.23,
                    "win_rate_pct": 60.0,
                },
            ),
        ),
        total_indicator_compute_calls=1,
        artifact_slot="slot_b",
        artifact_slot_generation=11,
        artifact_asof_date="2026-03-28",
        artifact_manifest_hash="c" * 64,
        execution_profile_mode="exact_no_risk_parity",
    )


def _background_auto_job() -> BacktestJob:
    """
    Build deterministic queued `background_auto` run snapshot for orchestration tests.

    Args:
        None.
    Returns:
        BacktestJob: Queued persisted run fixture with explicit `background_auto` mode.
    Assumptions:
        `request_json` already contains resolved defaults as created by jobs create flow.
    Raises:
        ValueError: If fixture violates persisted-run invariants.
    Side Effects:
        None.
    """
    return BacktestJob.create_queued(
        job_id=UUID("00000000-0000-0000-0000-000000000911"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000777"),
        mode="template",
        created_at=datetime(2026, 3, 28, 12, 0, 5, tzinfo=timezone.utc),
        request_json=_template_request_payload(),
        request_hash="f" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="e" * 64,
        backtest_runtime_config_hash="d" * 64,
        artifact_pin=BacktestJobArtifactPin(
            artifact_slot="slot_b",
            artifact_slot_generation=11,
            artifact_manifest_hash="c" * 64,
            artifact_asof_date="2026-03-28",
        ),
        execution_mode="background_auto",
        execution_profile_mode_hint="exact_parallel",
        effective_execution_profile_mode="exact_parallel",
        market_id=1,
        symbol="BTCUSDT",
        timeframe="1m",
        requested_top_n=2,
        ranking_primary_metric="total_return_pct",
        ranking_secondary_metric=None,
    )
