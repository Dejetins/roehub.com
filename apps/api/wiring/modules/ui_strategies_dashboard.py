from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from math import ceil
from typing import Literal, Mapping, Protocol
from uuid import UUID

from fastapi import APIRouter

from apps.api.dto.ui_strategies_dashboard import (
    RefreshStatus,
    SourceStatus,
    StrategyBreakdownPanelResponse,
    StrategyBreakdownRowResponse,
    StrategyChartResponse,
    StrategyDashboardActionsResponse,
    StrategyDashboardCompatibilityReadinessResponse,
    StrategyDashboardExchangeAccountReadinessResponse,
    StrategyDashboardFooterStatusResponse,
    StrategyDashboardLiveProfileResponse,
    StrategyDashboardPaperAccountingResponse,
    StrategyDashboardRefreshControlResponse,
    StrategyDashboardResponse,
    StrategyDashboardSelectedStrategyResponse,
    StrategyDashboardSelectorFiltersResponse,
    StrategyDashboardSelectorResponse,
    StrategyDashboardSelectorRowResponse,
    StrategyDashboardSelectorTotalsResponse,
    StrategyDashboardSourceResponse,
    StrategyExecutionOutcomeLinkResponse,
    StrategyExecutionOutcomeLinksResponse,
    StrategyHourlyResultResponse,
    StrategyHourlyResultsResponse,
    StrategyMetricGridResponse,
    StrategyMetricResponse,
    StrategyMonthlyStatsResponse,
    StrategySeriesPanelResponse,
    StrategySignalJournalResponse,
    StrategySignalJournalRowResponse,
    StrategyTradesResponse,
)
from apps.api.monitoring import (
    record_exchange_account_state_sync,
    record_exchange_config_guard,
)
from apps.api.routes.ui_strategies_dashboard import (
    build_ui_strategies_dashboard_router as build_ui_strategies_dashboard_api_router,
)
from apps.api.wiring.modules.strategy import (
    _build_compatibility_readiness_service,
    _build_live_profile_repository,
    _build_repositories,
    _build_signal_repository,
    _resolve_strategy_runtime_settings,
    is_strategy_api_enabled,
)
from trading.contexts.identity.adapters.inbound.api.deps import RequireCurrentUserDependency
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExchangeAccountProjectionRepository,
    InMemoryExecutionIntentRepository,
    InMemoryPaperAccountingRepository,
    PostgresExchangeAccountProjectionRepository,
    PostgresExecutionIntentRepository,
    PostgresPaperAccountingRepository,
    SystemLiveExecutionClock,
)
from trading.contexts.live_execution.application import ExchangeAccountProjectionService
from trading.contexts.live_execution.domain import (
    AccountProjectionReadiness,
    ExecutionProducerOutcomeLink,
    ExpectedInstrumentConfig,
    StrategyPaperAccountingSnapshot,
)
from trading.contexts.strategy.adapters.outbound import SystemStrategyClock
from trading.contexts.strategy.application import (
    CurrentUser,
    StrategyCompatibilityReadinessService,
)
from trading.contexts.strategy.application.ports.repositories import (
    LiveStrategyProfileRepository,
    StrategyRepository,
    StrategyRunRepository,
    StrategySignalRepository,
)
from trading.contexts.strategy.domain.entities import (
    LiveStrategyProfile,
    Strategy,
    StrategyRun,
    StrategySignal,
)
from trading.shared_kernel.primitives import UserId

_DEFAULT_REFRESH_INTERVAL_SECONDS = 15
_MINIMUM_MANUAL_REFRESH_SECONDS = 10
_STRATEGY_SELECTOR_LIMIT = 25
_CHART_MAX_POINTS = 600
_SERIES_MAX_POINTS = 600
_TRADES_LIMIT = 50

_STAT_SOURCE = "strategy_stat_projections"
_CANDLE_SOURCE = "market_candles"
_TRADES_SOURCE = "execution_fills"
_EVENTS_SOURCE = "strategy_events"
_RUNTIME_METADATA_SOURCE = "strategy_run_metadata"
_LIVE_PROFILE_SOURCE = "strategy_live_profiles"
_COMPATIBILITY_SOURCE = "strategy_compatibility_readiness"
_EXCHANGE_ACCOUNT_SOURCE = "exchange_account_projection"
_SIGNAL_JOURNAL_SOURCE = "strategy_signals"
_PAPER_ACCOUNTING_SOURCE = "strategy_paper_accounting"
_EXECUTION_OUTCOMES_SOURCE = "execution_producer_outcomes"
_SIGNAL_JOURNAL_LIMIT = 20
_EXECUTION_OUTCOME_LIMIT = 20


@dataclass(frozen=True, slots=True)
class _RefreshDecision:
    status: RefreshStatus
    next_allowed_refresh_at: datetime | None
    retry_after_seconds: int | None


class AccountProjectionReadinessService(Protocol):
    def get_readiness(
        self,
        *,
        owner_user_id: UserId,
        exchange_connection_id: UUID | None,
        requirement: ExpectedInstrumentConfig | None,
    ) -> AccountProjectionReadiness: ...


class PaperAccountingReadService(Protocol):
    def get_latest_accounting_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID
    ) -> StrategyPaperAccountingSnapshot | None: ...


class ExecutionOutcomeReadService(Protocol):
    def list_producer_outcome_links_for_strategy(
        self, *, owner_user_id: UserId, strategy_id: UUID, limit: int
    ) -> tuple[ExecutionProducerOutcomeLink, ...]: ...


class StrategyDashboardManualRefreshLimiter:
    def __init__(self, *, interval_seconds: int = _MINIMUM_MANUAL_REFRESH_SECONDS) -> None:
        if interval_seconds < 1:
            raise ValueError(
                "StrategyDashboardManualRefreshLimiter requires positive interval_seconds"
            )
        self._interval = timedelta(seconds=interval_seconds)
        self._next_allowed_by_user: dict[str, datetime] = {}

    def resolve(
        self,
        *,
        user_id: str,
        requested_at: datetime,
        refresh: Literal["initial", "auto", "manual"],
    ) -> _RefreshDecision:
        if refresh != "manual":
            return _RefreshDecision(
                status="fresh",
                next_allowed_refresh_at=self._next_allowed_by_user.get(user_id),
                retry_after_seconds=None,
            )

        next_allowed = self._next_allowed_by_user.get(user_id)
        if next_allowed is not None and requested_at < next_allowed:
            return _RefreshDecision(
                status="rate_limited",
                next_allowed_refresh_at=next_allowed,
                retry_after_seconds=max(1, ceil((next_allowed - requested_at).total_seconds())),
            )

        next_allowed = requested_at + self._interval
        self._next_allowed_by_user[user_id] = next_allowed
        return _RefreshDecision(
            status="fresh",
            next_allowed_refresh_at=next_allowed,
            retry_after_seconds=None,
        )


class StrategyDashboardQueryService:
    def __init__(
        self,
        *,
        strategy_repository: StrategyRepository | None,
        run_repository: StrategyRunRepository | None,
        profile_repository: LiveStrategyProfileRepository | None = None,
        signal_repository: StrategySignalRepository | None = None,
        compatibility_readiness_service: StrategyCompatibilityReadinessService | None = None,
        account_projection_service: AccountProjectionReadinessService | None = None,
        paper_accounting_service: PaperAccountingReadService | None = None,
        execution_outcome_service: ExecutionOutcomeReadService | None = None,
        refresh_limiter: StrategyDashboardManualRefreshLimiter | None = None,
    ) -> None:
        self._strategy_repository = strategy_repository
        self._run_repository = run_repository
        self._profile_repository = profile_repository
        self._signal_repository = signal_repository
        self._compatibility_readiness_service = compatibility_readiness_service
        self._account_projection_service = account_projection_service
        self._paper_accounting_service = paper_accounting_service
        self._execution_outcome_service = execution_outcome_service
        self._refresh_limiter = refresh_limiter or StrategyDashboardManualRefreshLimiter()

    def get_dashboard(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy_id: str | None,
        state: Literal["active", "all"],
        cursor: str | None,
        refresh: Literal["initial", "auto", "manual"],
    ) -> StrategyDashboardResponse:
        generated_at = datetime.now(UTC)
        refresh_decision = self._refresh_limiter.resolve(
            user_id=str(principal.user_id),
            requested_at=generated_at,
            refresh=refresh,
        )
        strategies, runs_by_strategy_id, dynamic_sources = self._load_strategy_state(
            user_id=principal.user_id,
            generated_at=generated_at,
        )
        selected_strategy = _select_strategy(strategies=strategies, strategy_id=strategy_id)
        selected_run = (
            runs_by_strategy_id.get(selected_strategy.strategy_id)
            if selected_strategy is not None
            else None
        )
        live_profile, live_profile_source = self._load_live_profile(
            principal=principal,
            strategy=selected_strategy,
            generated_at=generated_at,
        )
        signal_journal, signal_journal_source = self._load_signal_journal(
            principal=principal,
            strategy=selected_strategy,
            generated_at=generated_at,
        )
        paper_accounting, paper_accounting_source = self._load_paper_accounting(
            principal=principal,
            strategy=selected_strategy,
            generated_at=generated_at,
        )
        execution_outcomes, execution_outcomes_source = self._load_execution_outcomes(
            principal=principal,
            strategy=selected_strategy,
            generated_at=generated_at,
        )
        compatibility_readiness, compatibility_source = self._load_compatibility_readiness(
            principal=principal,
            strategy=selected_strategy,
            generated_at=generated_at,
        )
        account_readiness, account_source = self._load_exchange_account_readiness(
            principal=principal,
            strategy=selected_strategy,
            profile=live_profile,
            generated_at=generated_at,
        )
        sources = [
            *dynamic_sources,
            live_profile_source,
            compatibility_source,
            account_source,
            signal_journal_source,
            paper_accounting_source,
            execution_outcomes_source,
            _runtime_metadata_source(run=selected_run, generated_at=generated_at),
            _source(
                name=_CANDLE_SOURCE,
                status="unavailable",
                generated_at=generated_at,
                detail="bounded strategy candle/trade chart projection is not migrated yet",
            ),
            _source(
                name=_STAT_SOURCE,
                status="unavailable",
                generated_at=generated_at,
                detail=(
                    "strategy statistics, drawdown, equity and hourly projections are "
                    "not migrated yet"
                ),
            ),
            _source(
                name=_TRADES_SOURCE,
                status="unavailable",
                generated_at=generated_at,
                detail="bounded strategy fill/trade detail read-model is not migrated yet",
            ),
            _source(
                name=_EVENTS_SOURCE,
                status="unavailable",
                generated_at=generated_at,
                detail="strategy events are not exposed as a dashboard panel read-model yet",
            ),
        ]
        effective_refresh_status = _resolve_refresh_status(
            refresh_decision=refresh_decision,
            sources=sources,
        )
        selected_symbol = _selected_symbol(strategy=selected_strategy)

        return StrategyDashboardResponse(
            generated_at=generated_at,
            refresh_status=effective_refresh_status,
            next_allowed_refresh_at=refresh_decision.next_allowed_refresh_at,
            retry_after_seconds=refresh_decision.retry_after_seconds,
            sources=sources,
            selected_strategy=_build_selected_strategy(
                strategy=selected_strategy,
                run=selected_run,
                generated_at=generated_at,
                requested_strategy_id=strategy_id,
            ),
            live_profile=_build_live_profile(profile=live_profile, strategy=selected_strategy),
            compatibility_readiness=compatibility_readiness,
            exchange_account_readiness=account_readiness,
            strategy_selector=_build_strategy_selector(
                strategies=strategies,
                runs_by_strategy_id=runs_by_strategy_id,
                selected_strategy=selected_strategy,
                state=state,
                cursor=cursor,
            ),
            chart=_build_unavailable_chart(symbol=selected_symbol),
            metric_grid=_build_metric_grid(),
            monthly_stats=_build_monthly_stats(),
            long_short=_build_long_short(),
            risk_execution=_build_risk_execution(run=selected_run),
            drawdown=_build_unavailable_series(title="drawdown"),
            equity_curve=_build_unavailable_series(title="equity_curve"),
            hourly_results=_build_hourly_results(),
            trades=_build_trades(),
            signal_journal=signal_journal,
            paper_accounting=paper_accounting,
            execution_outcomes=execution_outcomes,
            footer_status=StrategyDashboardFooterStatusResponse(
                connection_status="degraded" if _has_degraded_sources(sources) else "ok",
                data_status="degraded" if _has_degraded_sources(sources) else "actual",
                api_label="Roehub API",
                latency_ms=None,
                capital_usdt=None,
                server_time=generated_at,
            ),
            refresh_control=StrategyDashboardRefreshControlResponse(
                manual_refresh_available=True,
                autorefresh_enabled=True,
                interval_seconds=_DEFAULT_REFRESH_INTERVAL_SECONDS,
                preset_key="15s",
                generated_at=generated_at,
                next_allowed_refresh_at=refresh_decision.next_allowed_refresh_at,
                retry_after_seconds=refresh_decision.retry_after_seconds,
                last_refresh_reason=refresh,
                refresh_status=effective_refresh_status,
            ),
        )

    def _load_execution_outcomes(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy: Strategy | None,
        generated_at: datetime,
    ) -> tuple[StrategyExecutionOutcomeLinksResponse, StrategyDashboardSourceResponse]:
        if strategy is None:
            return (
                _build_empty_execution_outcomes(reason="selected_strategy_not_found"),
                _source(
                    name=_EXECUTION_OUTCOMES_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="no selected strategy execution outcome links are available",
                ),
            )
        if self._execution_outcome_service is None:
            return (
                _build_empty_execution_outcomes(reason="execution_outcomes_not_configured"),
                _source(
                    name=_EXECUTION_OUTCOMES_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="execution outcome repository is not configured",
                ),
            )
        try:
            links = self._execution_outcome_service.list_producer_outcome_links_for_strategy(
                owner_user_id=principal.user_id,
                strategy_id=strategy.strategy_id,
                limit=_EXECUTION_OUTCOME_LIMIT,
            )
        except Exception as error:  # noqa: BLE001
            return (
                _build_empty_execution_outcomes(reason=str(error)),
                _source(
                    name=_EXECUTION_OUTCOMES_SOURCE,
                    status="degraded",
                    generated_at=generated_at,
                    detail=str(error),
                ),
            )
        if not links:
            return (
                _build_empty_execution_outcomes(reason="execution_outcomes_empty"),
                _source(
                    name=_EXECUTION_OUTCOMES_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="no source-event to execution outcome links exist yet",
                ),
            )
        return (
            StrategyExecutionOutcomeLinksResponse(
                source=_EXECUTION_OUTCOMES_SOURCE,
                state="ready",
                limit=_EXECUTION_OUTCOME_LIMIT,
                items=[_build_execution_outcome_link(link=link) for link in links],
                degradation_reason=None,
            ),
            _source(
                name=_EXECUTION_OUTCOMES_SOURCE,
                status="available",
                generated_at=generated_at,
                age_seconds=_age_seconds(
                    generated_at=generated_at,
                    observed_at=max(link.updated_at for link in links),
                ),
                detail=f"{len(links)} execution outcome links loaded",
            ),
        )

    def _load_paper_accounting(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy: Strategy | None,
        generated_at: datetime,
    ) -> tuple[StrategyDashboardPaperAccountingResponse, StrategyDashboardSourceResponse]:
        if strategy is None:
            return (
                _build_empty_paper_accounting(reason="selected_strategy_not_found"),
                _source(
                    name=_PAPER_ACCOUNTING_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="no selected strategy paper accounting is available",
                ),
            )
        if self._paper_accounting_service is None:
            return (
                _build_empty_paper_accounting(reason="paper_accounting_not_configured"),
                _source(
                    name=_PAPER_ACCOUNTING_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="paper accounting repository is not configured",
                ),
            )
        try:
            accounting = self._paper_accounting_service.get_latest_accounting_for_strategy(
                owner_user_id=principal.user_id,
                strategy_id=strategy.strategy_id,
            )
        except Exception as error:  # noqa: BLE001
            return (
                _build_empty_paper_accounting(reason=str(error)),
                _source(
                    name=_PAPER_ACCOUNTING_SOURCE,
                    status="degraded",
                    generated_at=generated_at,
                    detail=str(error),
                ),
            )
        if accounting is None:
            return (
                _build_empty_paper_accounting(reason="paper_accounting_empty"),
                _source(
                    name=_PAPER_ACCOUNTING_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="no paper accounting rows exist for selected strategy",
                ),
            )
        return (
            StrategyDashboardPaperAccountingResponse(
                source=_PAPER_ACCOUNTING_SOURCE,
                state="ready",
                reserved_budget=accounting.reserved_budget,
                position_quantity=accounting.position_quantity,
                average_entry_price=accounting.average_entry_price,
                equity=accounting.equity,
                realized_pnl=accounting.realized_pnl,
                unrealized_pnl=accounting.unrealized_pnl,
                fee_total=accounting.fee_total,
                funding_total=accounting.funding_total,
                fee_model=accounting.fee_model,
                funding_model=accounting.funding_model,
                pnl_complete=accounting.pnl_complete,
                completeness_reason=accounting.completeness_reason,
                updated_at=accounting.created_at,
                degradation_reason=(
                    None if accounting.pnl_complete else accounting.completeness_reason
                ),
            ),
            _source(
                name=_PAPER_ACCOUNTING_SOURCE,
                status="available" if accounting.pnl_complete else "degraded",
                generated_at=generated_at,
                age_seconds=_age_seconds(
                    generated_at=generated_at,
                    observed_at=accounting.created_at,
                ),
                detail=f"paper accounting {accounting.completeness_reason}",
            ),
        )

    def _load_exchange_account_readiness(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy: Strategy | None,
        profile: LiveStrategyProfile | None,
        generated_at: datetime,
    ) -> tuple[
        StrategyDashboardExchangeAccountReadinessResponse,
        StrategyDashboardSourceResponse,
    ]:
        if strategy is None:
            return (
                _build_empty_exchange_account_readiness(reason="selected_strategy_not_found"),
                _source(
                    name=_EXCHANGE_ACCOUNT_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="no selected strategy account projection is available",
                ),
            )
        if self._account_projection_service is None:
            return (
                _build_empty_exchange_account_readiness(
                    reason="account_projection_not_configured"
                ),
                _source(
                    name=_EXCHANGE_ACCOUNT_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="account projection repository is not configured",
                ),
            )
        requirement = _account_requirement_for_strategy(strategy=strategy)
        exchange_connection_id = (
            profile.exchange_connection_id
            if profile is not None and profile.exchange_connection_id is not None
            else None
        )
        try:
            readiness = self._account_projection_service.get_readiness(
                owner_user_id=principal.user_id,
                exchange_connection_id=exchange_connection_id,
                requirement=requirement,
            )
        except Exception as error:  # noqa: BLE001
            reason = str(error)
            return (
                _build_empty_exchange_account_readiness(reason=reason),
                _source(
                    name=_EXCHANGE_ACCOUNT_SOURCE,
                    status="degraded",
                    generated_at=generated_at,
                    detail=reason,
                ),
            )
        primary_reason = readiness.reason_codes[0] if readiness.reason_codes else "unknown"
        record_exchange_account_state_sync(
            status=readiness.status,
            reason=primary_reason,
            age_seconds=readiness.age_seconds,
        )
        record_exchange_config_guard(
            status=(
                "mismatch"
                if readiness.status == "config_mismatch"
                else "verified"
                if readiness.status == "fresh"
                else "degraded"
            ),
            reason=primary_reason,
        )
        panel_state = "ready" if readiness.status == "fresh" else "degraded"
        return (
            StrategyDashboardExchangeAccountReadinessResponse(
                source=_EXCHANGE_ACCOUNT_SOURCE,
                state=panel_state,
                status=readiness.status,
                reason_codes=list(readiness.reason_codes),
                exchange_connection_id=(
                    str(readiness.exchange_connection_id)
                    if readiness.exchange_connection_id is not None
                    else None
                ),
                instrument_key=readiness.instrument_key,
                market_type=readiness.market_type,
                account_snapshot_id=(
                    str(readiness.account_snapshot_id)
                    if readiness.account_snapshot_id is not None
                    else None
                ),
                config_guard_result_id=(
                    str(readiness.config_guard_result_id)
                    if readiness.config_guard_result_id is not None
                    else None
                ),
                age_seconds=readiness.age_seconds,
                checked_at=readiness.checked_at,
                ready_for_risk=readiness.ready_for_risk,
                degradation_reason=None if readiness.ready_for_risk else primary_reason,
            ),
            _source(
                name=_EXCHANGE_ACCOUNT_SOURCE,
                status="available" if readiness.status == "fresh" else "degraded",
                generated_at=generated_at,
                age_seconds=readiness.age_seconds,
                detail=f"{readiness.status}: {primary_reason}",
            ),
        )

    def _load_compatibility_readiness(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy: Strategy | None,
        generated_at: datetime,
    ) -> tuple[
        StrategyDashboardCompatibilityReadinessResponse,
        StrategyDashboardSourceResponse,
    ]:
        if strategy is None:
            return (
                _build_empty_compatibility(reason="selected_strategy_not_found"),
                _source(
                    name=_COMPATIBILITY_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="no selected strategy compatibility readiness is available",
                ),
            )
        if self._compatibility_readiness_service is None:
            return (
                _build_empty_compatibility(reason="compatibility_readiness_not_configured"),
                _source(
                    name=_COMPATIBILITY_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    detail="compatibility readiness service is not configured",
                ),
            )
        try:
            report = self._compatibility_readiness_service.check_strategy(
                strategy_id=strategy.strategy_id,
                current_user=CurrentUser(user_id=principal.user_id),
            )
        except Exception as error:  # noqa: BLE001
            return (
                _build_empty_compatibility(reason=str(error)),
                _source(
                    name=_COMPATIBILITY_SOURCE,
                    status="degraded",
                    generated_at=generated_at,
                    detail=str(error),
                ),
            )
        state = "degraded" if report.launch_blocked else "ready"
        return (
            StrategyDashboardCompatibilityReadinessResponse(
                source=_COMPATIBILITY_SOURCE,
                state=state,
                compatibility_state=report.compatibility_state,
                compatibility_reason_codes=list(report.compatibility_reason_codes),
                market_data_state=report.market_data_state,
                market_data_reason_codes=list(report.market_data_reason_codes),
                market_data_stream_name=report.market_data_stream_name,
                market_data_age_seconds=report.market_data_age_seconds,
                launch_blocked=report.launch_blocked,
                launch_blocked_reason=report.launch_blocked_reason,
                checked_at=report.checked_at,
                degradation_reason=report.launch_blocked_reason if report.launch_blocked else None,
            ),
            _source(
                name=_COMPATIBILITY_SOURCE,
                status="degraded" if report.launch_blocked else "available",
                generated_at=generated_at,
                age_seconds=_age_seconds(
                    generated_at=generated_at,
                    observed_at=report.checked_at,
                ),
                detail=(
                    f"{report.compatibility_state}; market data {report.market_data_state}"
                ),
            ),
        )

    def _load_signal_journal(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy: Strategy | None,
        generated_at: datetime,
    ) -> tuple[StrategySignalJournalResponse, StrategyDashboardSourceResponse]:
        if strategy is None:
            return (
                _build_empty_signal_journal(reason="selected_strategy_not_found"),
                _source(
                    name=_SIGNAL_JOURNAL_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    age_seconds=None,
                    detail="no selected strategy signal journal is available",
                ),
            )
        if self._signal_repository is None:
            return (
                _build_empty_signal_journal(reason="signal_repository_not_configured"),
                _source(
                    name=_SIGNAL_JOURNAL_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    age_seconds=None,
                    detail="strategy signal repository is not configured",
                ),
            )
        try:
            signals = self._signal_repository.list_latest_for_strategy(
                owner_user_id=principal.user_id,
                strategy_id=strategy.strategy_id,
                limit=_SIGNAL_JOURNAL_LIMIT,
            )
        except Exception as error:  # noqa: BLE001
            return (
                _build_empty_signal_journal(reason=str(error)),
                _source(
                    name=_SIGNAL_JOURNAL_SOURCE,
                    status="degraded",
                    generated_at=generated_at,
                    age_seconds=None,
                    detail=str(error),
                ),
            )
        if not signals:
            return (
                _build_empty_signal_journal(reason="signal_journal_empty"),
                _source(
                    name=_SIGNAL_JOURNAL_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    age_seconds=None,
                    detail="no StrategySignal journal rows exist for selected strategy",
                ),
            )
        latest = max(
            (signal.created_at or signal.bar_ts_close for signal in signals),
            default=generated_at,
        )
        return (
            StrategySignalJournalResponse(
                source=_SIGNAL_JOURNAL_SOURCE,
                state="ready",
                limit=_SIGNAL_JOURNAL_LIMIT,
                items=[_build_signal_journal_row(signal=signal) for signal in signals],
                degradation_reason=None,
            ),
            _source(
                name=_SIGNAL_JOURNAL_SOURCE,
                status="available",
                generated_at=generated_at,
                age_seconds=_age_seconds(generated_at=generated_at, observed_at=latest),
                detail=f"{len(signals)} latest StrategySignal rows loaded",
            ),
        )

    def _load_live_profile(
        self,
        *,
        principal: CurrentUserPrincipal,
        strategy: Strategy | None,
        generated_at: datetime,
    ) -> tuple[LiveStrategyProfile | None, StrategyDashboardSourceResponse]:
        if strategy is None:
            return (
                None,
                _source(
                    name=_LIVE_PROFILE_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    age_seconds=None,
                    detail="no selected strategy profile is available",
                ),
            )
        if self._profile_repository is None:
            return (
                None,
                _source(
                    name=_LIVE_PROFILE_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    age_seconds=None,
                    detail="live profile repository is not configured",
                ),
            )
        try:
            profile = self._profile_repository.get_for_strategy(
                owner_user_id=principal.user_id,
                strategy_id=strategy.strategy_id,
            )
        except Exception as error:  # noqa: BLE001
            return (
                None,
                _source(
                    name=_LIVE_PROFILE_SOURCE,
                    status="degraded",
                    generated_at=generated_at,
                    detail=str(error),
                ),
            )
        if profile is None:
            return (
                None,
                _source(
                    name=_LIVE_PROFILE_SOURCE,
                    status="unavailable",
                    generated_at=generated_at,
                    age_seconds=None,
                    detail="live profile has not been created yet; safe default is monitor_only",
                ),
            )
        return (
            profile,
            _source(
                name=_LIVE_PROFILE_SOURCE,
                status="available",
                generated_at=generated_at,
                age_seconds=_age_seconds(
                    generated_at=generated_at,
                    observed_at=profile.updated_at,
                ),
                detail=f"{profile.mode} profile readiness {profile.readiness_status}",
            ),
        )

    def _load_strategy_state(
        self,
        *,
        user_id: UserId,
        generated_at: datetime,
    ) -> tuple[
        tuple[Strategy, ...],
        dict[UUID, StrategyRun],
        list[StrategyDashboardSourceResponse],
    ]:
        if self._strategy_repository is None:
            return (
                (),
                {},
                [
                    _source(
                        name="strategy_strategies",
                        status="unavailable",
                        generated_at=generated_at,
                        detail="strategy API is disabled for this runtime",
                    ),
                    _source(
                        name="strategy_runs",
                        status="unavailable",
                        generated_at=generated_at,
                        detail="strategy run repository is not configured",
                    ),
                ],
            )

        try:
            strategies = self._strategy_repository.list_for_user(
                user_id=user_id,
                include_deleted=False,
            )
            strategy_source = _source(
                name="strategy_strategies",
                status="available",
                generated_at=generated_at,
                age_seconds=_latest_strategy_age_seconds(
                    strategies=tuple(strategies),
                    generated_at=generated_at,
                ),
                detail=f"{len(strategies)} owner strategies loaded",
            )
        except Exception as error:  # noqa: BLE001
            return (
                (),
                {},
                [
                    _source(
                        name="strategy_strategies",
                        status="degraded",
                        generated_at=generated_at,
                        detail=str(error),
                    ),
                    _source(
                        name="strategy_runs",
                        status="unavailable",
                        generated_at=generated_at,
                        detail="strategy list failed before run lookup",
                    ),
                ],
            )

        if self._run_repository is None:
            return (
                tuple(strategies),
                {},
                [
                    strategy_source,
                    _source(
                        name="strategy_runs",
                        status="unavailable",
                        generated_at=generated_at,
                        detail="strategy run repository is not configured",
                    ),
                ],
            )

        runs_by_strategy_id: dict[UUID, StrategyRun] = {}
        run_error: Exception | None = None
        for strategy in tuple(strategies)[:_STRATEGY_SELECTOR_LIMIT]:
            try:
                active_run = self._run_repository.find_active_for_strategy(
                    user_id=user_id,
                    strategy_id=strategy.strategy_id,
                )
            except Exception as error:  # noqa: BLE001
                run_error = error
                break
            if active_run is not None:
                runs_by_strategy_id[strategy.strategy_id] = active_run

        return (
            tuple(strategies),
            runs_by_strategy_id,
            [
                strategy_source,
                _source(
                    name="strategy_runs",
                    status="degraded" if run_error is not None else "available",
                    generated_at=generated_at,
                    age_seconds=_latest_run_age_seconds(
                        runs=tuple(runs_by_strategy_id.values()),
                        generated_at=generated_at,
                    ),
                    detail=(
                        str(run_error)
                        if run_error is not None
                        else f"{len(runs_by_strategy_id)} active owner runs loaded"
                    ),
                ),
            ],
        )


def build_ui_strategies_dashboard_router(
    *,
    environ: Mapping[str, str],
    current_user_dependency: RequireCurrentUserDependency,
) -> APIRouter:
    service = build_strategy_dashboard_service(environ=environ)
    return build_ui_strategies_dashboard_api_router(
        dashboard_service=service,
        current_user_dependency=current_user_dependency,
    )


def build_strategy_dashboard_service(
    *,
    environ: Mapping[str, str],
) -> StrategyDashboardQueryService:
    if not is_strategy_api_enabled(environ=environ):
        return StrategyDashboardQueryService(
            strategy_repository=None,
            run_repository=None,
            profile_repository=None,
            signal_repository=None,
            compatibility_readiness_service=None,
            account_projection_service=None,
            paper_accounting_service=None,
            execution_outcome_service=None,
        )
    settings = _resolve_strategy_runtime_settings(environ=environ)
    strategy_repository, run_repository, _event_repository = _build_repositories(settings=settings)
    profile_repository = _build_live_profile_repository(settings=settings)
    signal_repository = _build_signal_repository(settings=settings)
    compatibility_readiness_service = _build_compatibility_readiness_service(
        environ=environ,
        settings=settings,
        strategy_repository=strategy_repository,
        event_repository=None,
        clock=SystemStrategyClock(),
    )
    return StrategyDashboardQueryService(
        strategy_repository=strategy_repository,
        run_repository=run_repository,
        profile_repository=profile_repository,
        signal_repository=signal_repository,
        compatibility_readiness_service=compatibility_readiness_service,
        account_projection_service=_build_account_projection_service(settings=settings),
        paper_accounting_service=_build_paper_accounting_service(settings=settings),
        execution_outcome_service=_build_execution_outcome_service(settings=settings),
    )


def _build_account_projection_service(
    *,
    settings,
) -> ExchangeAccountProjectionService:
    repository = (
        PostgresExchangeAccountProjectionRepository(
            gateway=_build_live_execution_gateway(settings=settings)
        )
        if settings.postgres_dsn
        else InMemoryExchangeAccountProjectionRepository()
    )
    return ExchangeAccountProjectionService(
        repository=repository,
        clock=SystemLiveExecutionClock(),
    )


def _build_paper_accounting_service(*, settings):
    if settings.postgres_dsn:
        return PostgresPaperAccountingRepository(
            gateway=_build_live_execution_gateway(settings=settings)
        )
    return InMemoryPaperAccountingRepository()


def _build_execution_outcome_service(*, settings):
    if settings.postgres_dsn:
        return PostgresExecutionIntentRepository(
            gateway=_build_live_execution_gateway(settings=settings)
        )
    return InMemoryExecutionIntentRepository()


def _build_live_execution_gateway(*, settings):
    from trading.contexts.strategy.adapters.outbound import PsycopgStrategyPostgresGateway

    return PsycopgStrategyPostgresGateway(dsn=settings.postgres_dsn)


def _source(
    *,
    name: str,
    status: SourceStatus,
    generated_at: datetime,
    detail: str,
    age_seconds: int | None = 0,
) -> StrategyDashboardSourceResponse:
    return StrategyDashboardSourceResponse(
        name=name,
        status=status,
        generated_at=generated_at,
        age_seconds=age_seconds,
        detail=detail,
    )


def _runtime_metadata_source(
    *,
    run: StrategyRun | None,
    generated_at: datetime,
) -> StrategyDashboardSourceResponse:
    if run is None:
        return _source(
            name=_RUNTIME_METADATA_SOURCE,
            status="unavailable",
            generated_at=generated_at,
            age_seconds=None,
            detail="no active selected strategy run metadata is available",
        )
    if not run.metadata_json and run.checkpoint_ts_open is None:
        return _source(
            name=_RUNTIME_METADATA_SOURCE,
            status="degraded",
            generated_at=generated_at,
            age_seconds=_age_seconds(generated_at=generated_at, observed_at=run.updated_at),
            detail="active strategy run has not published warmup or rollup metadata yet",
        )
    return _source(
        name=_RUNTIME_METADATA_SOURCE,
        status="available",
        generated_at=generated_at,
        age_seconds=_age_seconds(generated_at=generated_at, observed_at=run.updated_at),
        detail="active selected strategy run warmup/rollup metadata loaded",
    )


def _resolve_refresh_status(
    *,
    refresh_decision: _RefreshDecision,
    sources: list[StrategyDashboardSourceResponse],
) -> RefreshStatus:
    if refresh_decision.status == "rate_limited":
        return "rate_limited"
    if _has_degraded_sources(sources):
        return "degraded"
    return "fresh"


def _has_degraded_sources(sources: list[StrategyDashboardSourceResponse]) -> bool:
    return any(source.status in {"degraded", "unavailable"} for source in sources)


def _latest_strategy_age_seconds(
    *,
    strategies: tuple[Strategy, ...],
    generated_at: datetime,
) -> int | None:
    if not strategies:
        return None
    return _age_seconds(
        generated_at=generated_at,
        observed_at=max(strategy.created_at for strategy in strategies),
    )


def _latest_run_age_seconds(
    *,
    runs: tuple[StrategyRun, ...],
    generated_at: datetime,
) -> int | None:
    if not runs:
        return None
    return _age_seconds(
        generated_at=generated_at,
        observed_at=max(run.updated_at for run in runs),
    )


def _age_seconds(*, generated_at: datetime, observed_at: datetime) -> int:
    return max(0, int((generated_at - observed_at).total_seconds()))


def _select_strategy(
    *,
    strategies: tuple[Strategy, ...],
    strategy_id: str | None,
) -> Strategy | None:
    if strategy_id:
        return next(
            (strategy for strategy in strategies if str(strategy.strategy_id) == strategy_id),
            None,
        )
    return strategies[0] if strategies else None


def _build_selected_strategy(
    *,
    strategy: Strategy | None,
    run: StrategyRun | None,
    generated_at: datetime,
    requested_strategy_id: str | None,
) -> StrategyDashboardSelectedStrategyResponse:
    if strategy is None:
        return StrategyDashboardSelectedStrategyResponse(
            source="strategy_strategies",
            state="empty",
            strategy_id=requested_strategy_id,
            name=None,
            version=None,
            exchange=None,
            market_type=None,
            symbols=[],
            timeframe=None,
            direction=None,
            capital_usdt=None,
            commission_percent=None,
            slippage_percent=None,
            created_at=None,
            updated_at=None,
            status="unknown",
            run_state=None,
            latest_update=None,
            actions=StrategyDashboardActionsResponse(
                can_create=True,
                can_clone=False,
                can_delete=False,
                can_run=False,
                can_stop=False,
            ),
            degradation_reason=(
                "selected_strategy_not_found"
                if requested_strategy_id
                else "strategy_read_model_empty"
            ),
        )

    exchange, market_type, symbol = _parse_instrument_key(strategy.spec.instrument_key)
    is_live = run is not None and run.is_active()
    latest_update = run.updated_at if run is not None else strategy.created_at
    return StrategyDashboardSelectedStrategyResponse(
        source="strategy_strategies",
        state="ready",
        strategy_id=str(strategy.strategy_id),
        name=strategy.name,
        version=f"v{strategy.spec.schema_version}",
        exchange=exchange,
        market_type=market_type or strategy.spec.market_type,
        symbols=[symbol],
        timeframe=strategy.spec.timeframe.code,
        direction="long / short",
        capital_usdt=None,
        commission_percent=None,
        slippage_percent=None,
        created_at=strategy.created_at,
        updated_at=latest_update,
        status="live" if is_live else "stopped",
        run_state=run.state if run is not None else None,
        latest_update=latest_update,
        actions=StrategyDashboardActionsResponse(
            can_create=True,
            can_clone=True,
            can_delete=True,
            can_run=not is_live,
            can_stop=is_live,
        ),
        degradation_reason="stat_projection_unavailable",
    )


def _build_live_profile(
    *,
    profile: LiveStrategyProfile | None,
    strategy: Strategy | None,
) -> StrategyDashboardLiveProfileResponse:
    if strategy is None:
        return StrategyDashboardLiveProfileResponse(
            source=_LIVE_PROFILE_SOURCE,
            state="empty",
            mode="monitor_only",
            exchange_connection_id=None,
            sizing_method="fixed_quote",
            sizing_value=Decimal("0"),
            max_position_notional=None,
            max_orders_per_run=0,
            max_notional_per_run=Decimal("0"),
            readiness_status="blocked",
            readiness_reason="strategy_read_model_empty",
            updated_at=None,
            degradation_reason="strategy_read_model_empty",
        )
    if profile is None:
        return StrategyDashboardLiveProfileResponse(
            source=_LIVE_PROFILE_SOURCE,
            state="degraded",
            mode="monitor_only",
            exchange_connection_id=None,
            sizing_method="fixed_quote",
            sizing_value=Decimal("0"),
            max_position_notional=None,
            max_orders_per_run=0,
            max_notional_per_run=Decimal("0"),
            readiness_status="ready",
            readiness_reason="monitor_only_no_exchange_submit",
            updated_at=None,
            degradation_reason="live_profile_not_created_default_monitor_only",
        )
    return StrategyDashboardLiveProfileResponse(
        source=_LIVE_PROFILE_SOURCE,
        state="ready" if profile.readiness_status == "ready" else "degraded",
        mode=profile.mode,
        exchange_connection_id=(
            str(profile.exchange_connection_id)
            if profile.exchange_connection_id is not None
            else None
        ),
        sizing_method=profile.sizing_method,
        sizing_value=profile.sizing_value,
        max_position_notional=profile.max_position_notional,
        max_orders_per_run=profile.max_orders_per_run,
        max_notional_per_run=profile.max_notional_per_run,
        readiness_status=profile.readiness_status,
        readiness_reason=profile.readiness_reason,
        updated_at=profile.updated_at,
        degradation_reason=(
            None
            if profile.readiness_status == "ready"
            else profile.readiness_reason
        ),
    )


def _build_empty_compatibility(
    *, reason: str
) -> StrategyDashboardCompatibilityReadinessResponse:
    return StrategyDashboardCompatibilityReadinessResponse(
        source=_COMPATIBILITY_SOURCE,
        state="empty",
        compatibility_state="not_launchable",
        compatibility_reason_codes=[reason],
        market_data_state="pending",
        market_data_reason_codes=[reason],
        market_data_stream_name=None,
        market_data_age_seconds=None,
        launch_blocked=True,
        launch_blocked_reason=reason,
        checked_at=None,
        degradation_reason=reason,
    )


def _build_empty_exchange_account_readiness(
    *, reason: str
) -> StrategyDashboardExchangeAccountReadinessResponse:
    return StrategyDashboardExchangeAccountReadinessResponse(
        source=_EXCHANGE_ACCOUNT_SOURCE,
        state="empty",
        status="degraded",
        reason_codes=[reason],
        exchange_connection_id=None,
        instrument_key=None,
        market_type=None,
        account_snapshot_id=None,
        config_guard_result_id=None,
        age_seconds=None,
        checked_at=None,
        ready_for_risk=False,
        degradation_reason=reason,
    )


def _account_requirement_for_strategy(*, strategy: Strategy) -> ExpectedInstrumentConfig:
    return ExpectedInstrumentConfig(
        instrument_key=strategy.spec.instrument_key,
        market_type=strategy.spec.market_type,
        expected_margin_mode=None if strategy.spec.market_type == "spot" else "isolated",
        expected_position_mode="net" if strategy.spec.market_type == "spot" else "one_way",
        min_notional=Decimal("0"),
    )


def _parse_instrument_key(instrument_key: str) -> tuple[str | None, str | None, str]:
    parts = instrument_key.split(":")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2].upper()
    return None, None, instrument_key.upper()


def _selected_symbol(*, strategy: Strategy | None) -> str | None:
    if strategy is None:
        return None
    _exchange, _market_type, symbol = _parse_instrument_key(strategy.spec.instrument_key)
    return symbol


def _build_strategy_selector(
    *,
    strategies: tuple[Strategy, ...],
    runs_by_strategy_id: dict[UUID, StrategyRun],
    selected_strategy: Strategy | None,
    state: Literal["active", "all"],
    cursor: str | None,
) -> StrategyDashboardSelectorResponse:
    rows = [
        _build_selector_row(strategy=strategy, run=runs_by_strategy_id.get(strategy.strategy_id))
        for strategy in strategies[:_STRATEGY_SELECTOR_LIMIT]
    ]
    active_rows = [row for row in rows if row.status == "live"]
    visible_rows = active_rows if state == "active" else rows
    symbols = {symbol for row in rows for symbol in row.symbols}
    return StrategyDashboardSelectorResponse(
        source="strategy_strategies",
        state="ready" if visible_rows else "empty",
        filters=StrategyDashboardSelectorFiltersResponse(
            state=state,
            cursor=cursor,
            limit=_STRATEGY_SELECTOR_LIMIT,
            query="",
            sort="updated",
        ),
        totals=StrategyDashboardSelectorTotalsResponse(
            strategies=len(strategies),
            active=len(active_rows),
            stopped=sum(1 for row in rows if row.status == "stopped"),
            degraded=sum(1 for row in rows if row.status == "degraded"),
            symbols=len(symbols),
        ),
        items=visible_rows,
        selected_strategy_id=str(selected_strategy.strategy_id) if selected_strategy else None,
        next_cursor=None if len(strategies) <= _STRATEGY_SELECTOR_LIMIT else "next",
        degradation_reason=None if visible_rows else "strategy_read_model_empty",
    )


def _build_selector_row(
    *,
    strategy: Strategy,
    run: StrategyRun | None,
) -> StrategyDashboardSelectorRowResponse:
    exchange, market_type, symbol = _parse_instrument_key(strategy.spec.instrument_key)
    is_live = run is not None and run.is_active()
    return StrategyDashboardSelectorRowResponse(
        strategy_id=str(strategy.strategy_id),
        name=strategy.name,
        version=f"v{strategy.spec.schema_version}",
        exchange=exchange,
        market_type=market_type or strategy.spec.market_type,
        symbols=[symbol],
        timeframe=strategy.spec.timeframe.code,
        status="live" if is_live else "stopped",
        run_state=run.state if run is not None else None,
        latest_activity=run.updated_at if run is not None else strategy.created_at,
    )


def _build_unavailable_chart(*, symbol: str | None) -> StrategyChartResponse:
    return StrategyChartResponse(
        source=_CANDLE_SOURCE,
        state="unavailable",
        symbol=symbol,
        range="1y",
        max_points=_CHART_MAX_POINTS,
        candles=[],
        markers=[],
        degradation_reason="strategy_candles_projection_unavailable",
    )


def _build_metric_grid() -> StrategyMetricGridResponse:
    metrics = [
        "total_return",
        "best_sharpe",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "trades",
        "avg_hold",
        "exposure",
        "avg_trade",
    ]
    return StrategyMetricGridResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        items=[_unavailable_metric(key=metric, source=_STAT_SOURCE) for metric in metrics],
        degradation_reason="strategy_statistics_projection_unavailable",
    )


def _unavailable_metric(*, key: str, source: str) -> StrategyMetricResponse:
    return StrategyMetricResponse(
        key=key,
        label=key,
        value=None,
        formatted="Unavailable",
        direction="neutral",
        status="unavailable",
        source=source,
    )


def _build_monthly_stats() -> StrategyMonthlyStatsResponse:
    return StrategyMonthlyStatsResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        columns=[
            "year",
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "jun",
            "jul",
            "aug",
            "sep",
            "oct",
            "nov",
            "dec",
            "total",
        ],
        rows=[],
        degradation_reason="monthly_statistics_projection_unavailable",
    )


def _build_long_short() -> StrategyBreakdownPanelResponse:
    return StrategyBreakdownPanelResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        rows=[
            _unavailable_breakdown("win_rate"),
            _unavailable_breakdown("trades"),
            _unavailable_breakdown("return"),
            _unavailable_breakdown("profit_factor"),
            _unavailable_breakdown("avg_trade"),
        ],
        degradation_reason="long_short_statistics_projection_unavailable",
    )


def _build_risk_execution(*, run: StrategyRun | None) -> StrategyBreakdownPanelResponse:
    if run is not None and (run.metadata_json or run.checkpoint_ts_open is not None):
        return StrategyBreakdownPanelResponse(
            source=_RUNTIME_METADATA_SOURCE,
            state="ready",
            rows=[
                _runtime_breakdown("run_state", run.state),
                _runtime_breakdown("warmup_progress", _format_warmup_progress(run.metadata_json)),
                _runtime_breakdown("warmup_satisfied", _format_warmup_satisfied(run.metadata_json)),
                _runtime_breakdown(
                    "rollup_bucket_count_1m",
                    _format_rollup_count(run.metadata_json),
                ),
                _runtime_breakdown(
                    "checkpoint_ts_open",
                    _format_optional_datetime(run.checkpoint_ts_open),
                ),
                _runtime_breakdown("last_error", run.last_error),
            ],
            degradation_reason=None,
        )
    return StrategyBreakdownPanelResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        rows=[
            _unavailable_breakdown("avg_holding_time"),
            _unavailable_breakdown("avg_bars_in_trade"),
            _unavailable_breakdown("commissions"),
            _unavailable_breakdown("execution_paid"),
            _unavailable_breakdown("max_consecutive_losses"),
            _unavailable_breakdown("worst_trade"),
        ],
        degradation_reason="risk_execution_projection_unavailable",
    )


def _unavailable_breakdown(key: str) -> StrategyBreakdownRowResponse:
    return StrategyBreakdownRowResponse(
        key=key,
        label=key,
        long_value=None,
        short_value=None,
        total_value="Unavailable",
        direction="neutral",
    )


def _runtime_breakdown(key: str, value: str | None) -> StrategyBreakdownRowResponse:
    return StrategyBreakdownRowResponse(
        key=key,
        label=key,
        long_value=None,
        short_value=None,
        total_value=value or "Unavailable",
        direction="neutral",
    )


def _format_warmup_progress(metadata_json: Mapping[str, object]) -> str | None:
    warmup = metadata_json.get("warmup")
    if not isinstance(warmup, Mapping):
        return None
    bars = _non_negative_int_or_none(warmup.get("bars"))
    processed = _non_negative_int_or_none(warmup.get("processed_bars"))
    if bars is None or processed is None:
        return None
    return f"{processed}/{bars}"


def _format_warmup_satisfied(metadata_json: Mapping[str, object]) -> str | None:
    warmup = metadata_json.get("warmup")
    if not isinstance(warmup, Mapping):
        return None
    satisfied = warmup.get("satisfied")
    if isinstance(satisfied, bool):
        return "yes" if satisfied else "no"
    return None


def _format_rollup_count(metadata_json: Mapping[str, object]) -> str | None:
    rollup = metadata_json.get("rollup")
    if not isinstance(rollup, Mapping):
        return None
    bucket_count = _non_negative_int_or_none(rollup.get("bucket_count_1m"))
    return None if bucket_count is None else str(bucket_count)


def _non_negative_int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _format_optional_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _build_unavailable_series(*, title: str) -> StrategySeriesPanelResponse:
    return StrategySeriesPanelResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        title=title,
        max_points=_SERIES_MAX_POINTS,
        points=[],
        degradation_reason=f"{title}_projection_unavailable",
    )


def _build_hourly_results() -> StrategyHourlyResultsResponse:
    empty_total = StrategyHourlyResultResponse(
        hour_bucket="total",
        win_rate_percent=None,
        pnl_percent=None,
        direction="neutral",
    )
    return StrategyHourlyResultsResponse(
        source=_STAT_SOURCE,
        state="unavailable",
        items=[],
        total=empty_total,
        degradation_reason="hourly_statistics_projection_unavailable",
    )


def _build_trades() -> StrategyTradesResponse:
    return StrategyTradesResponse(
        source=_TRADES_SOURCE,
        state="unavailable",
        limit=_TRADES_LIMIT,
        items=[],
        next_cursor=None,
        degradation_reason="strategy_trades_projection_unavailable",
    )


def _build_empty_signal_journal(*, reason: str) -> StrategySignalJournalResponse:
    return StrategySignalJournalResponse(
        source=_SIGNAL_JOURNAL_SOURCE,
        state="empty",
        limit=_SIGNAL_JOURNAL_LIMIT,
        items=[],
        degradation_reason=reason,
    )


def _build_empty_paper_accounting(*, reason: str) -> StrategyDashboardPaperAccountingResponse:
    return StrategyDashboardPaperAccountingResponse(
        source=_PAPER_ACCOUNTING_SOURCE,
        state="empty",
        reserved_budget=None,
        position_quantity=None,
        average_entry_price=None,
        equity=None,
        realized_pnl=None,
        unrealized_pnl=None,
        fee_total=None,
        funding_total=None,
        fee_model=None,
        funding_model=None,
        pnl_complete=False,
        completeness_reason=reason,
        updated_at=None,
        degradation_reason=reason,
    )


def _build_empty_execution_outcomes(*, reason: str) -> StrategyExecutionOutcomeLinksResponse:
    return StrategyExecutionOutcomeLinksResponse(
        source=_EXECUTION_OUTCOMES_SOURCE,
        state="empty",
        limit=_EXECUTION_OUTCOME_LIMIT,
        items=[],
        degradation_reason=reason,
    )


def _build_execution_outcome_link(
    *, link: ExecutionProducerOutcomeLink
) -> StrategyExecutionOutcomeLinkResponse:
    return StrategyExecutionOutcomeLinkResponse(
        source_event_id=str(link.source_event_id),
        source_type=link.source_type,
        source_event_ref=link.source_event_ref,
        strategy_signal_id=(
            str(link.strategy_signal_id) if link.strategy_signal_id is not None else None
        ),
        outcome=link.outcome,
        outcome_reason=link.outcome_reason,
        intent_id=str(link.intent_id) if link.intent_id is not None else None,
        intent_status=link.intent_status,
        intent_status_reason=link.intent_status_reason,
        risk_status=link.risk_status,
        risk_reason=link.risk_reason,
        order_status=link.order_status,
        order_status_reason=link.order_status_reason,
        notification_event_type=link.notification_event_type,
        notification_reason=link.notification_reason,
        updated_at=link.updated_at,
    )


def _build_signal_journal_row(*, signal: StrategySignal) -> StrategySignalJournalRowResponse:
    return StrategySignalJournalRowResponse(
        signal_id=str(signal.signal_id),
        strategy_run_id=str(signal.strategy_run_id),
        live_profile_id=str(signal.live_profile_id) if signal.live_profile_id is not None else None,
        mode=signal.mode,
        outcome=signal.outcome,
        signal_action=signal.signal_action,
        side=signal.side,
        reason_code=signal.reason_code,
        reference_price=signal.reference_price,
        instrument_key=signal.instrument_key,
        market_type=signal.market_type,
        timeframe=signal.timeframe,
        bar_ts_open=signal.bar_ts_open,
        bar_ts_close=signal.bar_ts_close,
        source_message_id=signal.source_message_id,
        created_at=signal.created_at,
    )
