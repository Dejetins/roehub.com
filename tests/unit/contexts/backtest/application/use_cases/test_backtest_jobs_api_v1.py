from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import UUID

import pytest

from trading.contexts.backtest.application.dto import RunBacktestRequest, RunBacktestTemplate
from trading.contexts.backtest.application.ports import (
    BacktestJobListPage,
    BacktestJobListQuery,
    BacktestStrategySnapshot,
    CurrentUser,
)
from trading.contexts.backtest.application.services import (
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
    ArtifactPinnedIdentityV2,
    ArtifactSlotLiteralV2,
    BacktestArtifactLoaderV2,
)
from trading.contexts.backtest.application.use_cases import (
    CancelBacktestJobUseCase,
    CreateBacktestJobCommand,
    CreateBacktestJobUseCase,
    GetBacktestJobStatusUseCase,
    GetBacktestJobTopUseCase,
    ListBacktestJobsUseCase,
)
from trading.contexts.backtest.application.use_cases.backtest_jobs_api_v1 import (
    _build_sha256_from_payload,
)
from trading.contexts.backtest.domain.entities import (
    BacktestJob,
    BacktestJobStageAShortlist,
    BacktestJobTopVariant,
)
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.backtest.domain.value_objects import BacktestJobListCursor
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


class _FakeJobRepository:
    """
    Deterministic in-memory fake for Backtest job repository use-case tests.
    """

    def __init__(
        self,
        *,
        active_total: int = 0,
        jobs_by_id: Mapping[UUID, BacktestJob] | None = None,
        list_page: BacktestJobListPage | None = None,
    ) -> None:
        """
        Initialize fake repository with deterministic state and optional fixtures.

        Args:
            active_total: Initial active jobs count returned by quota read.
            jobs_by_id: Optional seeded jobs mapping.
            list_page: Optional deterministic list page fixture.
        Returns:
            None.
        Assumptions:
            Tests mutate fake state directly through repository methods.
        Raises:
            None.
        Side Effects:
            Stores mutable in-memory maps/counters for test assertions.
        """
        self.active_total = active_total
        self.jobs_by_id = dict(jobs_by_id or {})
        self.list_page = list_page or BacktestJobListPage(items=tuple(), next_cursor=None)
        self.last_create_job: BacktestJob | None = None
        self.last_cancel_call: tuple[UUID, UserId] | None = None
        self.last_list_query: BacktestJobListQuery | None = None

    def create(self, *, job: BacktestJob) -> BacktestJob:
        """
        Persist job snapshot into in-memory store.

        Args:
            job: Job snapshot to persist.
        Returns:
            BacktestJob: Persisted job snapshot.
        Assumptions:
            Job ids are unique in test setup.
        Raises:
            None.
        Side Effects:
            Mutates in-memory jobs map and records last create call.
        """
        self.last_create_job = job
        self.jobs_by_id[job.job_id] = job
        return job

    def create_with_top_variants(
        self,
        *,
        job: BacktestJob,
        top_variants: tuple[BacktestJobTopVariant, ...],
    ) -> BacktestJob:
        """
        Reject unexpected atomic sync-inline persistence calls in jobs API unit tests.

        Args:
            job: Terminal job snapshot.
            top_variants: Persisted summary-only top rows.
        Returns:
            BacktestJob: Echoed job snapshot when used unexpectedly.
        Assumptions:
            `CreateBacktestJobUseCase` covers queued background job creation only.
        Raises:
            AssertionError: Always, because sync-inline persistence is out of scope here.
        Side Effects:
            None.
        """
        _ = job, top_variants
        raise AssertionError("create_with_top_variants is not expected in these tests")

    def get(self, *, job_id: UUID, user_id: UserId | None = None) -> BacktestJob | None:
        """
        Read one job from in-memory store with optional owner filter.

        Args:
            job_id: Requested job identifier.
            user_id: Optional owner filter.
        Returns:
            BacktestJob | None: Matching snapshot or `None`.
        Assumptions:
            Owner filter semantics match repository contract.
        Raises:
            None.
        Side Effects:
            None.
        """
        job = self.jobs_by_id.get(job_id)
        if job is None:
            return None
        if user_id is not None and job.user_id != user_id:
            return None
        return job

    def list_for_user(self, *, query: BacktestJobListQuery) -> BacktestJobListPage:
        """
        Return preconfigured list page and record query for assertions.

        Args:
            query: List query payload.
        Returns:
            BacktestJobListPage: Preconfigured deterministic page fixture.
        Assumptions:
            Tests validate query fields separately.
        Raises:
            None.
        Side Effects:
            Records last list query payload.
        """
        self.last_list_query = query
        return self.list_page

    def cancel(
        self,
        *,
        job_id: UUID,
        user_id: UserId,
        cancel_requested_at: datetime,
    ) -> BacktestJob | None:
        """
        Simulate deterministic cancel semantics for in-memory owner jobs.

        Args:
            job_id: Requested job identifier.
            user_id: Owner identifier.
            cancel_requested_at: Cancel timestamp.
        Returns:
            BacktestJob | None: Updated snapshot or `None`.
        Assumptions:
            Fake uses domain helper `request_cancel` for lifecycle behavior.
        Raises:
            None.
        Side Effects:
            Mutates in-memory job state and records cancel call args.
        """
        _ = cancel_requested_at
        self.last_cancel_call = (job_id, user_id)
        job = self.jobs_by_id.get(job_id)
        if job is None or job.user_id != user_id:
            return None
        updated = job.request_cancel(changed_at=cancel_requested_at)
        self.jobs_by_id[job_id] = updated
        return updated

    def count_active_for_user(self, *, user_id: UserId) -> int:
        """
        Return deterministic active jobs counter fixture.

        Args:
            user_id: Owner identifier.
        Returns:
            int: Active jobs count fixture.
        Assumptions:
            Counter fixture is configured by test setup.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = user_id
        return self.active_total

    def count_active_for_artifact_manifest(
        self,
        *,
        market_id: int,
        symbol: str,
        artifact_slot: str,
        artifact_manifest_hash: str,
    ) -> int:
        """
        Return deterministic zero blocking pins for create-use-case tests.

        Args:
            market_id: Requested market id.
            symbol: Requested symbol.
            artifact_slot: Candidate slot literal.
            artifact_manifest_hash: Candidate manifest hash.
        Returns:
            int: Always `0` for these unit tests.
        Assumptions:
            Create-use-case tests do not exercise publish guard repository queries.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = market_id, symbol, artifact_slot, artifact_manifest_hash
        return 0


class _FakeArtifactLoader:
    """
    Deterministic fake loader returning a fixed strict `current.yaml` payload.
    """

    def __init__(
        self,
        *,
        pointer: ArtifactCurrentPointerV2 | None = None,
        error: Exception | None = None,
    ) -> None:
        """
        Initialize fake loader with one optional pointer or raised error.

        Args:
            pointer: Optional strict pointer payload returned on reads.
            error: Optional exception raised instead of returning a pointer.
        Returns:
            None.
        Assumptions:
            Create-use-case tests need only `load_current_pointer(...)`.
        Raises:
            None.
        Side Effects:
            Stores last requested coordinates for assertions.
        """
        self._pointer = pointer or _artifact_pointer(slot="slot_a", generation=7)
        self._error = error
        self.last_coordinates: ArtifactCoordinatesV2 | None = None

    def load_current_pointer(self, coordinates: ArtifactCoordinatesV2) -> ArtifactCurrentPointerV2:
        """
        Return a fixed strict pointer payload for the requested coordinates.

        Args:
            coordinates: Requested artifact coordinates.
        Returns:
            ArtifactCurrentPointerV2: Fixed strict pointer payload.
        Assumptions:
            Other loader methods are not needed in create-use-case unit tests.
        Raises:
            Exception: Propagates configured loader failure.
        Side Effects:
            Records last requested coordinates.
        """
        self.last_coordinates = coordinates
        if self._error is not None:
            raise self._error
        return self._pointer


class _FakeResultsRepository:
    """
    Deterministic in-memory fake for Backtest job results repository tests.
    """

    def __init__(self, *, rows: tuple[BacktestJobTopVariant, ...]) -> None:
        """
        Initialize fake results repository with fixed top rows tuple.

        Args:
            rows: Deterministic top rows fixture.
        Returns:
            None.
        Assumptions:
            Rows are already sorted by repository ordering contract.
        Raises:
            None.
        Side Effects:
            Stores last requested limit for assertions.
        """
        self.rows = rows
        self.last_limit: int | None = None

    def list_top_variants(self, *, job_id: UUID, limit: int) -> tuple[BacktestJobTopVariant, ...]:
        """
        Return deterministic slice of preconfigured top rows fixture.

        Args:
            job_id: Requested job identifier.
            limit: Top limit value.
        Returns:
            tuple[BacktestJobTopVariant, ...]: Deterministic rows subset.
        Assumptions:
            Fake ignores job_id because tests control fixture scope.
        Raises:
            None.
        Side Effects:
            Records requested limit value.
        """
        _ = job_id
        self.last_limit = limit
        return self.rows[:limit]

    def replace_top_variants_snapshot(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        rows: tuple[BacktestJobTopVariant, ...],
    ) -> bool:
        """
        Satisfy repository protocol for worker-only snapshot writes in use-case tests.

        Args:
            job_id: Job identifier.
            now: Snapshot timestamp.
            locked_by: Lease owner marker.
            rows: Replacement top rows.
        Returns:
            bool: Always `True` for this in-memory fake.
        Assumptions:
            EPIC-11 API use-case tests never call this method.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (job_id, now, locked_by, rows)
        return True

    def save_stage_a_shortlist(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        shortlist: BacktestJobStageAShortlist,
    ) -> bool:
        """
        Satisfy repository protocol for worker shortlist writes in use-case tests.

        Args:
            job_id: Job identifier.
            now: Upsert timestamp.
            locked_by: Lease owner marker.
            shortlist: Stage-A shortlist payload.
        Returns:
            bool: Always `True` for this in-memory fake.
        Assumptions:
            EPIC-11 API use-case tests never call this method.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = (job_id, now, locked_by, shortlist)
        return True

    def get_stage_a_shortlist(self, *, job_id: UUID) -> BacktestJobStageAShortlist | None:
        """
        Satisfy repository protocol for worker shortlist reads in use-case tests.

        Args:
            job_id: Job identifier.
        Returns:
            BacktestJobStageAShortlist | None: Always `None` in this fake.
        Assumptions:
            EPIC-11 API use-case tests do not depend on shortlist payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = job_id
        return None


@dataclass(frozen=True, slots=True)
class _FakeStrategyReader:
    """
    Deterministic strategy reader fake returning one optional snapshot fixture.
    """

    snapshot: BacktestStrategySnapshot | None

    def load_any(self, *, strategy_id: UUID) -> BacktestStrategySnapshot | None:
        """
        Return preconfigured snapshot independent from requested strategy id.

        Args:
            strategy_id: Requested strategy identifier.
        Returns:
            BacktestStrategySnapshot | None: Configured snapshot fixture.
        Assumptions:
            Use-case tests focus on create-flow policy and hashes.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = strategy_id
        return self.snapshot


class _RuntimeContractDefaultsProvider:
    """
    Minimal defaults-provider fake for jobs create-use-case runtime-contract tests.
    """

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Return compute defaults for supported indicators used in create-flow tests.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            GridSpec | None: Defaults grid for supported indicators or `None`.
        Assumptions:
            Create flow needs only support catalog and signal defaults for validation.
        Raises:
            None.
        Side Effects:
            None.
        """
        normalized_id = indicator_id.strip().lower()
        if normalized_id != "ma.sma":
            return None
        return GridSpec(
            indicator_id=IndicatorId("ma.sma"),
            source=ExplicitValuesSpec(name="source", values=("close",)),
            params={"window": ExplicitValuesSpec(name="window", values=(20,))},
        )

    def signal_param_defaults(self, *, indicator_id: str) -> Mapping[str, ExplicitValuesSpec]:
        """
        Return deterministic signal defaults mapping for supported indicators.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            Mapping[str, ExplicitValuesSpec]: Signal defaults mapping or empty mapping.
        Assumptions:
            `ma.sma.cross_up=0.5` is the default-only signal contract for tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        if indicator_id.strip().lower() != "ma.sma":
            return {}
        return {"cross_up": ExplicitValuesSpec(name="cross_up", values=(0.5,))}

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return deterministic supported indicator ids for create-flow validation tests.

        Args:
            None.
        Returns:
            tuple[str, ...]: Supported indicator id tuple.
        Assumptions:
            Removed ids must be absent from this catalog.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ("ma.sma",)

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Return deterministic source catalog for one supported indicator.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            tuple[str, ...]: Allowed source literals or empty tuple.
        Assumptions:
            Source catalog is not directly inspected in create-flow tests.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ("close",) if indicator_id.strip().lower() == "ma.sma" else ()



def test_create_backtest_job_use_case_persists_effective_snapshot_and_hashes() -> None:
    """
    Verify template-mode create flow stores effective scalar defaults/execution and hashes.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Request payload follows strict API envelope shape expected by worker decoder.
    Raises:
        AssertionError: If create snapshot payload or hashes are inconsistent.
    Side Effects:
        None.
    """
    repository = _FakeJobRepository(active_total=0)
    artifact_loader = _FakeArtifactLoader()
    use_case = CreateBacktestJobUseCase(
        job_repository=repository,
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=300,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="c" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, artifact_loader),
        now_provider=lambda: datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        job_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000901"),
    )

    run_request = RunBacktestRequest(
        time_range=_time_range(),
        template=_template(),
        top_k=5,
        top_trades_n=2,
    )
    command = CreateBacktestJobCommand(
        run_request=run_request,
        request_payload=_template_request_payload(),
    )

    created = use_case.execute(
        command=command,
        current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
    )

    assert created.job_id == UUID("00000000-0000-0000-0000-000000000901")
    assert created.mode == "template"
    assert created.state == "queued"
    assert created.request_json["warmup_bars"] == 200
    assert created.request_json["top_k"] == 5
    assert created.request_json["preselect"] == 20000
    assert created.request_json["top_trades_n"] == 2
    assert created.request_json["template"]["execution"] == {
        "fee_pct": 0.075,
        "fixed_quote": 100.0,
        "init_cash_quote": 10000.0,
        "safe_profit_percent": 30.0,
        "slippage_pct": 0.01,
    }
    assert created.spec_hash is None
    assert len(created.request_hash) == 64
    assert len(created.engine_params_hash) == 64
    assert created.backtest_runtime_config_hash == "c" * 64
    assert created.artifact_pin is not None
    assert created.artifact_pin.artifact_slot == "slot_a"
    assert created.artifact_pin.artifact_slot_generation == 7
    assert artifact_loader.last_coordinates == ArtifactCoordinatesV2(
        exchange="binance",
        market_type="spot",
        symbol="BTCUSDT",
    )


def test_create_backtest_job_use_case_artifact_pin_converts_to_pinned_identity_v2() -> None:
    """
    Verify persisted create-time artifact pin fields map losslessly into the R6-01 pin DTO.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Background runtime bootstrap reuses the exact pin fields persisted at job creation time.
    Raises:
        AssertionError: If create-time pin fields drift from the new pinned-identity contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-runtime-kernels-v2.md
    Related:
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_resolver.py
    """
    repository = _FakeJobRepository(active_total=0)
    artifact_loader = _FakeArtifactLoader(pointer=_artifact_pointer(slot="slot_b", generation=9))
    use_case = CreateBacktestJobUseCase(
        job_repository=repository,
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=300,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="c" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, artifact_loader),
        now_provider=lambda: datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        job_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000912"),
    )

    created = use_case.execute(
        command=CreateBacktestJobCommand(
            run_request=RunBacktestRequest(
                time_range=_time_range(),
                template=_template(),
            ),
            request_payload=_template_request_payload(),
        ),
        current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
    )

    assert created.artifact_pin is not None
    pinned_identity = ArtifactPinnedIdentityV2(
        artifact_slot=created.artifact_pin.artifact_slot,
        slot_generation=created.artifact_pin.artifact_slot_generation,
        artifact_asof_date=created.artifact_pin.artifact_asof_date,
        artifact_manifest_hash=created.artifact_pin.artifact_manifest_hash,
    )

    assert pinned_identity.artifact_slot == "slot_b"
    assert pinned_identity.slot_generation == 9
    assert pinned_identity.artifact_asof_date == "2026-03-24"
    assert pinned_identity.artifact_manifest_hash == "a" * 64
    assert created.execution_mode == "background_auto"
    assert created.market_id == 1
    assert created.symbol == "BTCUSDT"
    assert created.timeframe == "1m"
    assert created.requested_top_n == 300
    assert created.ranking_primary_metric == "total_return_pct"
    assert created.ranking_secondary_metric is None



def test_create_backtest_job_use_case_saved_mode_persists_spec_hash_and_snapshot() -> None:
    """
    Verify saved-mode create flow persists `spec_hash/spec_payload_json` and effective overrides.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Saved strategy snapshot fixture belongs to current user.
    Raises:
        AssertionError: If saved-mode reproducibility fields are missing.
    Side Effects:
        None.
    """
    strategy_snapshot = _strategy_snapshot(
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
    )
    repository = _FakeJobRepository(active_total=0)
    artifact_loader = _FakeArtifactLoader(pointer=_artifact_pointer(slot="slot_b", generation=9))
    use_case = CreateBacktestJobUseCase(
        job_repository=repository,
        strategy_reader=_FakeStrategyReader(snapshot=strategy_snapshot),
        top_k_persisted_default=300,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="d" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, artifact_loader),
        now_provider=lambda: datetime(2026, 2, 23, 12, 1, tzinfo=timezone.utc),
        job_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000902"),
    )

    strategy_id = UUID("00000000-0000-0000-0000-000000000501")
    run_request = RunBacktestRequest(
        time_range=_time_range(),
        strategy_id=strategy_id,
    )
    command = CreateBacktestJobCommand(
        run_request=run_request,
        request_payload={
            "time_range": {
                "start": "2026-02-21T00:00:00+00:00",
                "end": "2026-02-21T01:00:00+00:00",
            },
            "strategy_id": str(strategy_id),
        },
    )

    created = use_case.execute(
        command=command,
        current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
    )

    assert created.mode == "saved"
    assert created.spec_hash is not None
    assert created.spec_payload_json == strategy_snapshot.spec_payload
    assert created.request_json["strategy_id"] == str(strategy_id)
    assert created.request_json["overrides"]["execution"]["fee_pct"] == 0.075
    assert created.artifact_pin is not None
    assert created.artifact_pin.artifact_slot == "slot_b"
    assert created.artifact_pin.artifact_manifest_hash == "a" * 64
    assert created.execution_mode == "background_auto"
    assert created.market_id == 1
    assert created.symbol == "BTCUSDT"
    assert created.timeframe == "1m"
    assert created.requested_top_n == 300
    assert created.ranking_primary_metric == "total_return_pct"
    assert created.ranking_secondary_metric is None


def test_create_backtest_job_use_case_accepts_background_auto_execution_mode() -> None:
    """
    Verify create flow can persist explicit `background_auto` mode for `/backtests` fallback.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `POST /backtests` and compatibility jobs creation now share `background_auto` as the
        canonical active background literal.
    Raises:
        AssertionError: If create flow ignores command-level execution mode override.
    Side Effects:
        None.
    """
    repository = _FakeJobRepository(active_total=0)
    artifact_loader = _FakeArtifactLoader(pointer=_artifact_pointer(slot="slot_b", generation=9))
    use_case = CreateBacktestJobUseCase(
        job_repository=repository,
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=300,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="e" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, artifact_loader),
        now_provider=lambda: datetime(2026, 2, 23, 12, 2, tzinfo=timezone.utc),
        job_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000913"),
    )

    created = use_case.execute(
        command=CreateBacktestJobCommand(
            run_request=RunBacktestRequest(
                time_range=_time_range(),
                template=_template(),
            ),
            request_payload=_template_request_payload(),
            execution_mode="background_auto",
        ),
        current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
    )

    assert created.execution_mode == "background_auto"
    assert repository.last_create_job is not None
    assert repository.last_create_job.execution_mode == "background_auto"


def test_create_backtest_job_use_case_excludes_execution_profile_mode_from_request_hash() -> None:
    """
    Verify persisted-only `execution_profile_mode` metadata does not affect request identity.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Exact profile selection changes launch routing/read-model semantics, but not exact result
        semantics for the same canonical request payload.
    Raises:
        AssertionError: If `request_hash` starts depending on persisted-only profile metadata.
    Side Effects:
        None.
    """
    repository = _FakeJobRepository(active_total=0)
    artifact_loader = _FakeArtifactLoader(pointer=_artifact_pointer(slot="slot_b", generation=9))
    use_case = CreateBacktestJobUseCase(
        job_repository=repository,
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=300,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="e" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, artifact_loader),
        now_provider=lambda: datetime(2026, 2, 23, 12, 3, tzinfo=timezone.utc),
        job_id_factory=lambda: UUID("00000000-0000-0000-0000-000000000914"),
    )

    created = use_case.execute(
        command=CreateBacktestJobCommand(
            run_request=RunBacktestRequest(
                time_range=_time_range(),
                template=_template(),
            ),
            request_payload=_template_request_payload(),
            execution_mode="background_auto",
            execution_profile_mode="exact_parallel",
        ),
        current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
    )

    assert created.request_json["execution_profile_mode"] == "exact_parallel"
    assert created.request_hash == _build_sha256_from_payload(
        payload={
            key: value
            for key, value in created.request_json.items()
            if key != "execution_profile_mode"
        }
    )


def test_create_backtest_job_use_case_rejects_missing_current_yaml_for_pinning() -> None:
    """
    Verify create flow fails fast when requested instrument has no published `current.yaml`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Background job reproducibility requires strict artifact pin metadata at create time.
    Raises:
        AssertionError: If missing pointer is not mapped to deterministic validation error.
    Side Effects:
        None.
    """
    use_case = CreateBacktestJobUseCase(
        job_repository=_FakeJobRepository(active_total=0),
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=300,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=300,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="d" * 64,
        artifact_loader=cast(
            BacktestArtifactLoaderV2,
            _FakeArtifactLoader(error=FileNotFoundError("missing current.yaml")),
        ),
    )

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=RunBacktestRequest(
                    time_range=_time_range(),
                    template=_template(),
                ),
                request_payload=_template_request_payload(),
            ),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )

    assert error_info.value.code == "validation_error"
    assert error_info.value.details == {
        "errors": [
            {
                "path": "body.template.instrument_id",
                "code": "artifact_unavailable",
                "message": "missing current.yaml for binance:spot:BTCUSDT",
            }
        ]
    }


def test_create_backtest_job_use_case_rejects_top_k_above_persisted_cap() -> None:
    """
    Verify create flow returns deterministic `validation_error` for `top_k` over persisted cap.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        EPIC-11 `top_k <= top_k_persisted_default` invariant is mandatory.
    Raises:
        AssertionError: If error code/details are not deterministic.
    Side Effects:
        None.
    """
    use_case = CreateBacktestJobUseCase(
        job_repository=_FakeJobRepository(active_total=0),
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=10,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=10,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="e" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, _FakeArtifactLoader()),
    )

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=RunBacktestRequest(
                    time_range=_time_range(),
                    template=_template(),
                    top_k=11,
                ),
                request_payload=_template_request_payload(),
            ),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )

    assert error_info.value.code == "validation_error"
    assert error_info.value.details == {
        "errors": [
            {
                "path": "body.top_k",
                "code": "max_value",
                "message": "top_k must be <= 10",
            }
        ]
    }



def test_create_backtest_job_use_case_rejects_active_quota_exceeded() -> None:
    """
    Verify create flow returns deterministic `validation_error` when active quota is exceeded.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Active jobs are counted as `queued + running`.
    Raises:
        AssertionError: If quota violation does not map to deterministic error.
    Side Effects:
        None.
    """
    use_case = CreateBacktestJobUseCase(
        job_repository=_FakeJobRepository(active_total=3),
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=10,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=10,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="f" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, _FakeArtifactLoader()),
    )

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=RunBacktestRequest(time_range=_time_range(), template=_template()),
                request_payload=_template_request_payload(),
            ),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )

    assert error_info.value.code == "validation_error"
    assert error_info.value.details == {
        "errors": [
            {
                "path": "body",
                "code": "quota_exceeded",
                "message": "active jobs limit reached (3/3)",
            }
        ]
    }


def test_create_backtest_job_use_case_rejects_removed_indicator_id() -> None:
    """
    Verify create flow rejects removed indicator ids via shared R1 runtime contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runtime defaults catalog is the authoritative list of supported indicator ids.
    Raises:
        AssertionError: If removed ids are not rejected with deterministic validation details.
    Side Effects:
        None.
    """
    use_case = CreateBacktestJobUseCase(
        job_repository=_FakeJobRepository(active_total=0),
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=10,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=10,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="f" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, _FakeArtifactLoader()),
        defaults_provider=_RuntimeContractDefaultsProvider(),
        allowed_request_timeframes=("1m",),
    )

    with pytest.raises(BacktestValidationError) as error_info:
        use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=RunBacktestRequest(
                    time_range=_time_range(),
                    template=RunBacktestTemplate(
                        instrument_id=InstrumentId(
                            market_id=MarketId(1),
                            symbol=Symbol("BTCUSDT"),
                        ),
                        timeframe=Timeframe("1m"),
                        indicator_grids=(
                            GridSpec(
                                indicator_id=IndicatorId("momentum.macd"),
                                params={
                                    "fast_window": ExplicitValuesSpec(
                                        name="fast_window",
                                        values=(12,),
                                    ),
                                },
                            ),
                        ),
                    ),
                ),
                request_payload=_template_request_payload(),
            ),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )

    assert error_info.value.errors == (
        {
            "path": "body.template.indicator_grids[0].indicator_id",
            "code": "unsupported_value",
            "message": "indicator_id 'momentum.macd' is not supported",
        },
    )


def test_create_backtest_job_use_case_rejects_default_only_signal_override() -> None:
    """
    Verify create flow rejects request-level signal params that differ from server defaults.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `signals.v1.params` is enforced as `default-only` in template mode.
    Raises:
        AssertionError: If non-default signal overrides are not rejected deterministically.
    Side Effects:
        None.
    """
    use_case = CreateBacktestJobUseCase(
        job_repository=_FakeJobRepository(active_total=0),
        strategy_reader=_FakeStrategyReader(snapshot=None),
        top_k_persisted_default=10,
        max_active_jobs_per_user=3,
        warmup_bars_default=200,
        top_k_default=10,
        preselect_default=20000,
        top_trades_n_default=3,
        init_cash_quote_default=10000.0,
        fixed_quote_default=100.0,
        safe_profit_percent_default=30.0,
        slippage_pct_default=0.01,
        fee_pct_default_by_market_id={1: 0.075},
        backtest_runtime_config_hash="f" * 64,
        artifact_loader=cast(BacktestArtifactLoaderV2, _FakeArtifactLoader()),
        defaults_provider=_RuntimeContractDefaultsProvider(),
        allowed_request_timeframes=("1m",),
    )

    with pytest.raises(BacktestValidationError) as error_info:
        use_case.execute(
            command=CreateBacktestJobCommand(
                run_request=RunBacktestRequest(
                    time_range=_time_range(),
                    template=RunBacktestTemplate(
                        instrument_id=InstrumentId(
                            market_id=MarketId(1),
                            symbol=Symbol("BTCUSDT"),
                        ),
                        timeframe=Timeframe("1m"),
                        indicator_grids=(
                            GridSpec(
                                indicator_id=IndicatorId("ma.sma"),
                                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
                            ),
                        ),
                        signal_grids={
                            "ma.sma": {
                                "cross_up": ExplicitValuesSpec(name="cross_up", values=(0.6,))
                            }
                        },
                    ),
                ),
                request_payload=_template_request_payload(),
            ),
            current_user=CurrentUser(
                user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")
            ),
        )

    assert error_info.value.errors == (
        {
            "path": "body.template.signal_grids.ma.sma.cross_up",
            "code": "forbidden_override",
            "message": "signals.v1.params is default-only",
        },
    )



def test_get_status_use_case_returns_403_for_foreign_and_404_for_missing() -> None:
    """
    Verify owner policy returns `403` for foreign existing job and `404` for missing job.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Use-case reads job without owner SQL filter first.
    Raises:
        AssertionError: If error mapping violates EPIC-11 owner contract.
    Side Effects:
        None.
    """
    owner_job = _queued_job(
        job_id=UUID("00000000-0000-0000-0000-000000000810"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000999"),
    )
    repository = _FakeJobRepository(jobs_by_id={owner_job.job_id: owner_job})
    use_case = GetBacktestJobStatusUseCase(job_repository=repository)

    with pytest.raises(RoehubError) as forbidden_error:
        use_case.execute(
            job_id=owner_job.job_id,
            current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
        )
    assert forbidden_error.value.code == "forbidden"

    with pytest.raises(RoehubError) as not_found_error:
        use_case.execute(
            job_id=UUID("00000000-0000-0000-0000-000000000811"),
            current_user=CurrentUser(user_id=UserId.from_string("00000000-0000-0000-0000-000000000111")),
        )
    assert not_found_error.value.code == "not_found"



def test_get_top_use_case_validates_limit_and_reads_rows() -> None:
    """
    Verify top use-case validates limit against persisted cap and returns deterministic rows.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repository returns rows ordered by `rank ASC, variant_key ASC`.
    Raises:
        AssertionError: If limit validation or rows retrieval contract breaks.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    owner_job = _queued_job(
        job_id=UUID("00000000-0000-0000-0000-000000000820"),
        user_id=owner_user_id,
    )
    row = BacktestJobTopVariant(
        job_id=owner_job.job_id,
        rank=1,
        variant_key="a" * 64,
        indicator_variant_key="b" * 64,
        variant_index=0,
        total_return_pct=10.0,
        payload_json={"schema_version": 1},
        summary_metrics_json={"total_return_pct": 10.0, "profit_factor": 1.2},
        best_tp_pct=4.0,
        best_sl_pct=2.0,
        report_table_md=None,
        trades_json=None,
        updated_at=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
    )

    repository = _FakeJobRepository(jobs_by_id={owner_job.job_id: owner_job})
    results_repository = _FakeResultsRepository(rows=(row,))
    use_case = GetBacktestJobTopUseCase(
        job_repository=repository,
        results_repository=results_repository,
        top_k_persisted_default=5,
    )

    result = use_case.execute(
        job_id=owner_job.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
        limit=1,
    )
    assert result.rows == (row,)
    assert results_repository.last_limit == 1

    with pytest.raises(RoehubError) as error_info:
        use_case.execute(
            job_id=owner_job.job_id,
            current_user=CurrentUser(user_id=owner_user_id),
            limit=6,
        )
    assert error_info.value.code == "validation_error"



def test_cancel_use_case_returns_updated_owner_snapshot() -> None:
    """
    Verify cancel use-case returns idempotent status payload after owner validation.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Fake repository uses domain `request_cancel` lifecycle method.
    Raises:
        AssertionError: If cancel operation does not update state snapshot.
    Side Effects:
        None.
    """
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    owner_job = _queued_job(
        job_id=UUID("00000000-0000-0000-0000-000000000830"),
        user_id=owner_user_id,
    )
    repository = _FakeJobRepository(jobs_by_id={owner_job.job_id: owner_job})
    use_case = CancelBacktestJobUseCase(
        job_repository=repository,
        now_provider=lambda: datetime(2026, 2, 23, 12, 5, tzinfo=timezone.utc),
    )

    cancelled = use_case.execute(
        job_id=owner_job.job_id,
        current_user=CurrentUser(user_id=owner_user_id),
    )

    assert cancelled.state == "cancelled"
    assert repository.last_cancel_call == (owner_job.job_id, owner_user_id)



def test_list_use_case_passes_keyset_query_to_repository() -> None:
    """
    Verify list use-case forwards state/limit/cursor into repository keyset query object.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Repository query object validates deterministic limit bounds.
    Raises:
        AssertionError: If query forwarding contract is broken.
    Side Effects:
        None.
    """
    cursor = BacktestJobListCursor(
        created_at=datetime(2026, 2, 23, 12, 0, tzinfo=timezone.utc),
        job_id=UUID("00000000-0000-0000-0000-000000000840"),
    )
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000111")
    page = BacktestJobListPage(
        items=(_queued_job(job_id=cursor.job_id, user_id=owner_user_id),),
        next_cursor=None,
    )
    repository = _FakeJobRepository(list_page=page)
    use_case = ListBacktestJobsUseCase(job_repository=repository)

    result = use_case.execute(
        current_user=CurrentUser(user_id=owner_user_id),
        state="queued",
        limit=25,
        cursor=cursor,
    )

    assert result == page
    assert repository.last_list_query is not None
    assert repository.last_list_query.state == "queued"
    assert repository.last_list_query.limit == 25
    assert repository.last_list_query.cursor == cursor



def _template_request_payload() -> Mapping[str, Any]:
    """
    Build minimal valid API template payload used by create use-case command fixture.

    Args:
        None.
    Returns:
        Mapping[str, Any]: API transport payload fixture.
    Assumptions:
        Shape is compatible with `BacktestsPostRequest` strict DTO.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "time_range": {
            "start": "2026-02-21T00:00:00+00:00",
            "end": "2026-02-21T01:00:00+00:00",
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
        },
    }


def _artifact_pointer(
    *,
    slot: ArtifactSlotLiteralV2,
    generation: int,
) -> ArtifactCurrentPointerV2:
    """
    Build deterministic strict `current.yaml` payload fixture for artifact pinning tests.

    Args:
        slot: Active slot literal.
        generation: Positive slot generation.
    Returns:
        ArtifactCurrentPointerV2: Strict pointer payload fixture.
    Assumptions:
        Fixture hash/date/timestamp literals follow the R2-02 strict pointer contract.
    Raises:
        ValueError: If one fixture field violates strict pointer invariants.
    Side Effects:
        None.
    """
    payload = {
        "schema_version": 1,
        "active_slot": slot,
        "slot_generation": generation,
        "asof_date": "2026-03-24",
        "manifest_sha256": "a" * 64,
        "published_at_utc": "2026-03-24T02:00:00Z",
    }
    return ArtifactCurrentPointerV2(
        path=Path("/tmp/artifacts/backtest/v2/binance/spot/BTCUSDT/current.yaml"),
        active_slot=slot,
        raw_payload=payload,
        schema_version=1,
        slot_generation=generation,
        asof_date="2026-03-24",
        manifest_sha256="a" * 64,
        published_at_utc="2026-03-24T02:00:00Z",
    )


def _queued_job(*, job_id: UUID, user_id: UserId) -> BacktestJob:
    """
    Build deterministic queued job fixture for EPIC-11 use-case unit tests.

    Args:
        job_id: Deterministic job identifier.
        user_id: Job owner identifier.
    Returns:
        BacktestJob: Queued domain job snapshot fixture.
    Assumptions:
        Hash literals are valid lowercase SHA-256 placeholders.
    Raises:
        ValueError: If one fixture field violates domain invariants.
    Side Effects:
        None.
    """
    return BacktestJob.create_queued(
        job_id=job_id,
        user_id=user_id,
        mode="template",
        created_at=datetime(2026, 2, 23, 11, 55, tzinfo=timezone.utc),
        request_json={"mode": "template", "top_k": 5},
        request_hash="a" * 64,
        spec_hash=None,
        spec_payload_json=None,
        engine_params_hash="b" * 64,
        backtest_runtime_config_hash="c" * 64,
    )



def _template() -> RunBacktestTemplate:
    """
    Build deterministic ad-hoc template fixture for create-use-case tests.

    Args:
        None.
    Returns:
        RunBacktestTemplate: Minimal valid template fixture.
    Assumptions:
        One indicator axis is sufficient for create-flow tests.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestTemplate(
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            ),
        ),
    )



def _strategy_snapshot(*, user_id: UserId) -> BacktestStrategySnapshot:
    """
    Build deterministic saved strategy snapshot fixture for create-use-case tests.

    Args:
        user_id: Snapshot owner identifier.
    Returns:
        BacktestStrategySnapshot: Valid saved snapshot fixture.
    Assumptions:
        Snapshot spec payload contains non-empty JSON object.
    Raises:
        ValueError: If fixture violates snapshot invariants.
    Side Effects:
        None.
    """
    return BacktestStrategySnapshot(
        strategy_id=UUID("00000000-0000-0000-0000-000000000501"),
        user_id=user_id,
        is_deleted=False,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "close"},
                params={"window": 20},
            ),
        ),
        spec_payload={
            "schema_version": 1,
            "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
            "timeframe": "1m",
            "indicators": [
                {
                    "id": "ma.sma",
                    "inputs": {"source": "close"},
                    "params": {"window": 20},
                }
            ],
        },
    )



def _time_range() -> TimeRange:
    """
    Build deterministic UTC half-open time range fixture for request DTO tests.

    Args:
        None.
    Returns:
        TimeRange: Shared time range fixture.
    Assumptions:
        Range start is strictly before range end.
    Raises:
        ValueError: If fixture violates `TimeRange` invariants.
    Side Effects:
        None.
    """
    return TimeRange(
        start=UtcTimestamp(datetime(2026, 2, 21, 0, 0, tzinfo=timezone.utc)),
        end=UtcTimestamp(datetime(2026, 2, 21, 1, 0, tzinfo=timezone.utc)),
    )
