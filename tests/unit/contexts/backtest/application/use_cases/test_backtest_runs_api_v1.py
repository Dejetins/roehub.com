from __future__ import annotations

from dataclasses import dataclass
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
from trading.contexts.backtest.application.use_cases.backtest_runs_api_v1 import (
    CreateAndRunBacktestSyncInlineUseCase,
)
from trading.contexts.backtest.domain.entities import BacktestJob, BacktestJobTopVariant
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
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
            None.
        """
        _ = request, current_user, request_payload, run_control
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
    use_case = CreateAndRunBacktestSyncInlineUseCase(
        run_use_case=_FakeRunUseCase(response=response),
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
    assert persisted.engine_version == "signal_tf + 1m_risk"
    assert persisted.artifact_slot == "slot_b"
    assert persisted.artifact_slot_generation == 11
    assert persisted.artifact_asof_date == "2026-03-28"
    assert persisted.artifact_manifest_hash == "c" * 64
    assert repo.created_job is not None
    assert repo.created_job.execution_mode == "sync_inline"
    assert repo.created_job.state == "succeeded"
    assert repo.created_job.market_id == 1
    assert repo.created_job.symbol == "BTCUSDT"
    assert repo.created_job.requested_top_n == 2
    assert repo.created_job.request_json["template"]["execution"]["fee_pct"] == 0.075
    assert repo.created_job.request_json["template"]["direction_mode"] == "long-short"
    assert len(repo.created_rows) == 1
    assert repo.created_rows[0].rank == 1
    assert repo.created_rows[0].variant_key == "a" * 64
    assert repo.created_rows[0].report_table_md is None
    assert repo.created_rows[0].trades_json is None
    assert repo.created_rows[0].summary_metrics_json["win_rate_pct"] == 60.0
    assert repo.created_rows[0].best_tp_pct == 4.0
    assert repo.created_rows[0].best_sl_pct == 2.0


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
        top_trades_n=1,
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
        "warmup_bars": 200,
        "top_k": 2,
        "preselect": 100,
        "top_trades_n": 1,
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
        warmup_bars=200,
        top_k=2,
        preselect=100,
        top_trades_n=1,
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
    )
