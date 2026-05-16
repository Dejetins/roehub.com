from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from trading.contexts.backtest.application.ai_configurator.dto import BacktestAiConfigJob
from trading.contexts.backtest.application.ai_configurator.services.catalog import (
    BacktestAiAllowedCatalog,
)


@dataclass(frozen=True, slots=True)
class BacktestConfigAgentRequest:
    job: BacktestAiConfigJob
    catalog: BacktestAiAllowedCatalog


@dataclass(frozen=True, slots=True)
class BacktestConfigAgentResponse:
    raw_output: str | None
    model_id: str | None = None
    model_path_hash: str | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None
    audit_json: Mapping[str, Any] | None = None


class BacktestConfigAgentGateway(Protocol):
    def run_config_session(
        self,
        request: BacktestConfigAgentRequest,
    ) -> BacktestConfigAgentResponse:
        """
        Run one backend-controlled configurator agent session.

        Implementations may use model tools, but only through backend-owned,
        allowlisted tool executors. The agent must not receive a full catalog
        prompt blob or arbitrary filesystem access.
        """
        ...


__all__ = [
    "BacktestConfigAgentGateway",
    "BacktestConfigAgentRequest",
    "BacktestConfigAgentResponse",
]
