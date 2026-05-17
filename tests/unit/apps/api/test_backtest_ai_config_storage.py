from __future__ import annotations

from fastapi import Request

from apps.api.routes import build_backtests_router
from apps.api.wiring.modules.backtest import build_backtest_ai_configurator_use_cases
from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiConfigJobsUseCase,
    BacktestAiConversationUseCase,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import PaidLevel, UserId


def test_backtest_ai_configurator_wiring_builds_storage_boundary_without_enabling_ui() -> None:
    use_cases = build_backtest_ai_configurator_use_cases(
        environ={
            "ROEHUB_ENV": "test",
            "STRATEGY_PG_DSN": "postgresql://roehub:roehub@localhost:5432/roehub",
        }
    )

    assert use_cases is not None
    assert isinstance(use_cases.jobs, BacktestAiConfigJobsUseCase)
    assert isinstance(use_cases.conversations, BacktestAiConversationUseCase)
    assert use_cases.runtime_config.conversation.retention_days == 30
    assert use_cases.runtime_config.conversation.max_conversations_per_user == 50
    assert use_cases.runtime_config.conversation.max_messages_per_conversation == 100
    assert use_cases.runtime_config.enabled is False
    assert use_cases.runtime_config.queue.lease_seconds == 120
    assert use_cases.runtime_config.to_quota_config().quota_for(
        paid_level=PaidLevel.free()
    ).requests_per_5h == 3


def test_backtest_ai_configurator_wiring_without_dsn_keeps_storage_unavailable() -> None:
    use_cases = build_backtest_ai_configurator_use_cases(
        environ={
            "ROEHUB_ENV": "test",
        }
    )

    assert use_cases is None


def test_stage_one_does_not_register_browser_visible_ai_config_routes() -> None:
    router = build_backtests_router(
        runtime_defaults_service=object(),  # type: ignore[arg-type]
        preflight_service=object(),  # type: ignore[arg-type]
        current_user_dependency=_CurrentUserDependency(),
        jobs_use_case=None,
    )

    paths = {getattr(route, "path", "") for route in router.routes}

    assert not any("/backtests/ai-config" in path for path in paths)


class _CurrentUserDependency:
    def __call__(self, request: Request) -> CurrentUserPrincipal:
        _ = request
        return CurrentUserPrincipal(
            user_id=UserId.from_string("00000000-0000-0000-0000-000000000411"),
            paid_level=PaidLevel.free(),
        )
