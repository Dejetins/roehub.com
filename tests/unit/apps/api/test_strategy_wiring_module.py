from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest

from apps.api.exchange_control_client import (
    ExchangeConnectionCommandResult,
    ExchangeControlAccountStateSnapshot,
    ExchangeControlBalanceSnapshot,
    ExchangeControlInstrumentFilterSnapshot,
    ExchangeControlOpenOrderSnapshot,
    ExchangeControlPositionSnapshot,
    InMemoryExchangeControlClient,
)
from apps.api.wiring.modules import strategy as strategy_wiring
from apps.api.wiring.modules.strategy import (
    ExchangeControlReadinessChecker,
    is_strategy_api_enabled,
)
from trading.contexts.backtest.application.ports import (
    BacktestJobRepository,
    ResearchOrganizationScope,
    ResearchOrganizationScopeResolver,
)
from trading.contexts.live_execution.adapters.outbound import (
    InMemoryExchangeAccountProjectionRepository,
)
from trading.contexts.live_execution.application import ExchangeAccountProjectionService
from trading.contexts.strategy.application import ExchangeConnectionReadinessContext
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import OrganizationId, UserId


class _StaticResearchScopeResolver:
    def __init__(self, *, organization_id: OrganizationId) -> None:
        self._organization_id = organization_id

    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        return ResearchOrganizationScope(
            organization_id=self._organization_id,
            user_id=user_id,
        )


class _AmbiguousResearchScopeResolver:
    def resolve(self, *, user_id: UserId) -> ResearchOrganizationScope:
        _ = user_id
        raise RoehubError(
            code="research.organization_scope_ambiguous",
            message="Research organization scope is ambiguous",
            details={"reason": "multiple_active_memberships"},
        )


class _OwnershipCheckingBacktestRepository:
    def __init__(
        self,
        *,
        owner_user_id: UserId,
        owner_organization_id: OrganizationId,
    ) -> None:
        self._owner_user_id = owner_user_id
        self._owner_organization_id = owner_organization_id
        self.get_calls: list[tuple[UUID, OrganizationId, UserId | None]] = []
        self.variant_calls = 0

    def get(
        self,
        *,
        job_id: UUID,
        organization_id: OrganizationId,
        user_id: UserId | None = None,
    ) -> Any | None:
        self.get_calls.append((job_id, organization_id, user_id))
        if (
            organization_id != self._owner_organization_id
            or user_id != self._owner_user_id
        ):
            return None
        return cast(Any, object())

    def get_top_variant_by_public_key(self, **kwargs: Any) -> Any | None:
        _ = kwargs
        self.variant_calls += 1
        raise AssertionError("variant lookup must not run after ownership rejection")


@pytest.mark.parametrize(
    ("request_user_id", "resolved_organization_id"),
    [
        (
            UserId.from_string("00000000-0000-0000-0000-000000000811"),
            OrganizationId.from_string("00000000-0000-0000-0000-000000000902"),
        ),
        (
            UserId.from_string("00000000-0000-0000-0000-000000000812"),
            OrganizationId.from_string("00000000-0000-0000-0000-000000000901"),
        ),
    ],
)
def test_strategy_variant_reader_rejects_cross_owner_job_before_variant_lookup(
    request_user_id: UserId,
    resolved_organization_id: OrganizationId,
) -> None:
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000811")
    owner_organization_id = OrganizationId.from_string(
        "00000000-0000-0000-0000-000000000901"
    )
    repository = _OwnershipCheckingBacktestRepository(
        owner_user_id=owner_user_id,
        owner_organization_id=owner_organization_id,
    )
    reader = strategy_wiring._BacktestJobRepositoryVariantLaunchReader(
        repository=cast(BacktestJobRepository, repository),
        organization_scope_resolver=cast(
            ResearchOrganizationScopeResolver,
            _StaticResearchScopeResolver(
                organization_id=resolved_organization_id,
            ),
        ),
    )
    job_id = UUID("00000000-0000-0000-0000-000000000999")

    with pytest.raises(RoehubError) as error_info:
        reader.get(
            user_id=request_user_id,
            job_id=job_id,
            variant_key="variant-1",
        )

    assert error_info.value.code == "strategy_variant_launch.not_found"
    assert repository.get_calls == [
        (job_id, resolved_organization_id, request_user_id)
    ]
    assert repository.variant_calls == 0


def test_strategy_variant_reader_fails_closed_on_ambiguous_organization_scope() -> None:
    repository = _OwnershipCheckingBacktestRepository(
        owner_user_id=UserId.from_string("00000000-0000-0000-0000-000000000811"),
        owner_organization_id=OrganizationId.from_string(
            "00000000-0000-0000-0000-000000000901"
        ),
    )
    reader = strategy_wiring._BacktestJobRepositoryVariantLaunchReader(
        repository=cast(BacktestJobRepository, repository),
        organization_scope_resolver=cast(
            ResearchOrganizationScopeResolver,
            _AmbiguousResearchScopeResolver(),
        ),
    )

    with pytest.raises(RoehubError) as error_info:
        reader.get(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000811"),
            job_id=UUID("00000000-0000-0000-0000-000000000999"),
            variant_key="variant-1",
        )

    assert error_info.value.code == "research.organization_scope_ambiguous"
    assert repository.get_calls == []
    assert repository.variant_calls == 0


class _StaticClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value


class _FuturesExchangeControlClient(InMemoryExchangeControlClient):
    def __init__(
        self,
        *,
        connection: ExchangeConnectionCommandResult,
        snapshot: ExchangeControlAccountStateSnapshot,
    ) -> None:
        self._connection = connection
        self._snapshot = snapshot

    def list_connections(
        self, *, owner_user_id: str, request_id: str | None = None
    ) -> tuple[ExchangeConnectionCommandResult, ...]:
        _ = owner_user_id, request_id
        return (self._connection,)

    def read_account_state(
        self,
        *,
        owner_user_id: str,
        connection_id: str,
        instrument_keys: tuple[str, ...] = (),
        request_id: str | None = None,
    ) -> ExchangeControlAccountStateSnapshot:
        _ = owner_user_id, connection_id, instrument_keys, request_id
        return self._snapshot


def _write_strategy_config(tmp_path: Path, *, api_enabled: bool) -> Path:
    """
    Write minimal valid Strategy runtime config for API toggle tests.

    Args:
        tmp_path: pytest temporary path fixture.
        api_enabled: Desired `strategy.api.enabled` toggle value.
    Returns:
        Path: Written config path.
    Assumptions:
        Minimal payload still contains mandatory live-worker sections.
    Raises:
        OSError: If write fails.
    Side Effects:
        Creates one temporary YAML file.
    """
    path = tmp_path / "strategy.yaml"
    path.write_text(
        (
            "version: 1\n"
            "strategy:\n"
            f"  api:\n    enabled: {'true' if api_enabled else 'false'}\n"
            "  live_worker:\n"
            "    enabled: true\n"
            "    poll_interval_seconds: 5\n"
            "    redis_streams:\n"
            "      enabled: true\n"
            "      host: redis\n"
            "      port: 6379\n"
            "      db: 0\n"
            "      socket_timeout_s: 2.0\n"
            "      connect_timeout_s: 2.0\n"
            "      stream_prefix: md.candles.1m\n"
            "      consumer_group: strategy.live_runner.v1\n"
            "      read_count: 100\n"
            "      block_ms: 100\n"
            "  realtime_output:\n"
            "    redis_streams:\n"
            "      enabled: false\n"
            "      host: redis\n"
            "      port: 6379\n"
            "      db: 0\n"
            "      socket_timeout_s: 2.0\n"
            "      connect_timeout_s: 2.0\n"
            "      metrics_stream_prefix: strategy.metrics.v1.user\n"
            "      events_stream_prefix: strategy.events.v1.user\n"
            "  telegram:\n"
            "    enabled: false\n"
            "    mode: log_only\n"
            "    bot_token_env: TELEGRAM_BOT_TOKEN\n"
            "    api_base_url: https://api.telegram.org\n"
            "    send_timeout_s: 2.0\n"
            "    debounce_failed_seconds: 600\n"
            "  metrics:\n"
            "    port: 9207\n"
            "  producer:\n"
            "    enabled: false\n"
            "    allow_all: false\n"
            "    allowed_modes:\n"
            "      - paper\n"
            "      - testnet\n"
            "    allowed_user_ids: []\n"
            "    allowed_strategy_ids: []\n"
        ),
        encoding="utf-8",
    )
    return path


def test_is_strategy_api_enabled_reads_yaml_toggle(tmp_path: Path) -> None:
    """
    Verify Strategy API toggle is read from source-of-truth `strategy.yaml`.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        `ROEHUB_STRATEGY_CONFIG` points to temporary runtime config file.
    Raises:
        AssertionError: If toggle value does not match YAML.
    Side Effects:
        None.
    """
    disabled_path = _write_strategy_config(tmp_path, api_enabled=False)

    assert (
        is_strategy_api_enabled(
            environ={"ROEHUB_STRATEGY_CONFIG": str(disabled_path)},
        )
        is False
    )

    enabled_path = _write_strategy_config(tmp_path, api_enabled=True)
    assert (
        is_strategy_api_enabled(
            environ={"ROEHUB_STRATEGY_CONFIG": str(enabled_path)},
        )
        is True
    )


def test_is_strategy_api_enabled_env_override_has_priority(tmp_path: Path) -> None:
    """
    Verify `ROEHUB_STRATEGY_API_ENABLED` override has higher priority than YAML toggle.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Scalar env overrides follow STR-EPIC-06 precedence contract.
    Raises:
        AssertionError: If env override does not affect result.
    Side Effects:
        None.
    """
    config_path = _write_strategy_config(tmp_path, api_enabled=True)

    assert (
        is_strategy_api_enabled(
            environ={
                "ROEHUB_STRATEGY_CONFIG": str(config_path),
                "ROEHUB_STRATEGY_API_ENABLED": "0",
            },
        )
        is False
    )


def test_is_strategy_api_enabled_rejects_invalid_override_literal(tmp_path: Path) -> None:
    """
    Verify invalid boolean env override literal fails fast with deterministic error.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Boolean env parser accepts strict literals only.
    Raises:
        AssertionError: If invalid literal does not raise ValueError.
    Side Effects:
        None.
    """
    config_path = _write_strategy_config(tmp_path, api_enabled=True)

    with pytest.raises(ValueError, match="ROEHUB_STRATEGY_API_ENABLED"):
        is_strategy_api_enabled(
            environ={
                "ROEHUB_STRATEGY_CONFIG": str(config_path),
                "ROEHUB_STRATEGY_API_ENABLED": "enabled",
            },
        )


def test_exchange_control_readiness_checker_accepts_safe_testnet_futures_short() -> None:
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000205")
    connection_id = UUID("00000000-0000-0000-0000-00000000e102")
    repository = InMemoryExchangeAccountProjectionRepository()
    checker = ExchangeControlReadinessChecker(
        client=_FuturesExchangeControlClient(
            connection=_connection(connection_id=connection_id),
            snapshot=_safe_futures_snapshot(),
        ),
        account_projection_service=ExchangeAccountProjectionService(
            repository=repository,
            clock=_StaticClock(datetime(2026, 6, 17, 12, 1, tzinfo=UTC)),
        ),
    )

    readiness = checker.check_trading_ready(
        organization_id=OrganizationId(
            UUID("00000000-0000-4000-8000-000000000010")
        ),
        owner_user_id=owner_user_id,
        exchange_connection_id=connection_id,
        context=ExchangeConnectionReadinessContext(
            mode="testnet",
            market_type="futures",
            symbol="BTCUSDT",
            direction="short",
            notional=Decimal("50"),
        ),
    )

    assert readiness.eligible is True
    assert readiness.reason == "safe_testnet_futures_short_1x_isolated_verified"
    assert repository.projections
    assert repository.config_results[-1].reason_codes == ("verify_only_config_ok",)


def test_exchange_control_readiness_checker_blocks_without_projection_store() -> None:
    owner_user_id = UserId.from_string("00000000-0000-0000-0000-000000000205")
    connection_id = UUID("00000000-0000-0000-0000-00000000e102")
    checker = ExchangeControlReadinessChecker(
        client=_FuturesExchangeControlClient(
            connection=_connection(connection_id=connection_id),
            snapshot=_safe_futures_snapshot(),
        ),
        account_projection_service=None,
    )

    readiness = checker.check_trading_ready(
        organization_id=OrganizationId(
            UUID("00000000-0000-4000-8000-000000000010")
        ),
        owner_user_id=owner_user_id,
        exchange_connection_id=connection_id,
        context=ExchangeConnectionReadinessContext(
            mode="testnet",
            market_type="futures",
            symbol="BTCUSDT",
            direction="short",
            notional=Decimal("50"),
        ),
    )

    assert readiness.eligible is False
    assert readiness.reason == "account_projection_repository_unavailable"


def _connection(*, connection_id: UUID) -> ExchangeConnectionCommandResult:
    now = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    return ExchangeConnectionCommandResult(
        connection_id=str(connection_id),
        credential_version_id="00000000-0000-0000-0000-00000000f102",
        exchange_name="binance",
        market_type="futures",
        environment="testnet",
        label="Binance futures testnet",
        permissions="trade",
        requested_permissions="trade",
        exchange_permissions="trade",
        effective_permissions="trade",
        permission_warnings=(),
        api_key="****1234",
        status="active",
        status_reason=None,
        validation_status="valid_trade_enabled",
        validation_reason="trade_permission_detected",
        ip_restriction_status="not_restricted_testnet",
        last_validated_at=now,
        created_at=now,
        updated_at=now,
        disabled_at=None,
        archived_at=None,
        requested_capability="trading",
        effective_capability="trading",
        connection_readiness="ready_for_trading",
        connection_readiness_reason="trading_policy_ok",
        permissions_deprecated=True,
    )


def _safe_futures_snapshot() -> ExchangeControlAccountStateSnapshot:
    observed_at = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    return ExchangeControlAccountStateSnapshot(
        exchange_name="binance",
        market_type="futures",
        environment="testnet",
        account_mode="futures",
        sync_status="fresh",
        sync_reason="account_state_read_ok",
        source_hash="c" * 64,
        observed_at=observed_at,
        balances=(
            ExchangeControlBalanceSnapshot(
                asset="USDT",
                free=Decimal("100"),
                locked=Decimal("0"),
                total=Decimal("100"),
            ),
        ),
        positions=(
            ExchangeControlPositionSnapshot(
                instrument_key="binance:futures:BTCUSDT",
                side="net",
                quantity=Decimal("0"),
                entry_price=Decimal("0"),
                leverage=Decimal("1"),
                margin_mode="isolated",
                position_mode="one_way",
            ),
        ),
        open_orders=(
            ExchangeControlOpenOrderSnapshot(
                instrument_key="binance:futures:BTCUSDT",
                exchange_order_ref="order-1",
                side="buy",
                order_type="limit",
                quantity=Decimal("0.001"),
                price=Decimal("50000"),
                status="new",
            ),
        ),
        instrument_filters=(
            ExchangeControlInstrumentFilterSnapshot(
                instrument_key="binance:futures:BTCUSDT",
                tick_size=Decimal("0.1"),
                step_size=Decimal("0.001"),
                min_qty=Decimal("0.001"),
                min_notional=Decimal("50"),
                max_leverage=Decimal("125"),
            ),
        ),
    )
