from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from trading.shared_kernel.primitives import UserId

PaperScenarioCoverageState = Literal["covered", "blocked"]


@dataclass(frozen=True, slots=True)
class PaperScenarioCoverageResult:
    coverage_result_id: UUID
    owner_user_id: UserId
    scenario_matrix_row_id: UUID
    scenario_key: str
    source_job_id: UUID
    source_variant_key: str
    mode: str
    market_type: str
    symbol: str
    entry_sizing: str
    risk_mode: str
    direction: str
    coverage_state: PaperScenarioCoverageState
    coverage_reason: str
    strategy_id: UUID | None
    live_profile_id: UUID | None
    strategy_run_id: UUID | None
    strategy_signal_id: UUID | None
    source_event_id: UUID | None
    intent_id: UUID | None
    paper_order_id: UUID | None
    paper_fill_id: UUID | None
    accounting_id: UUID | None
    fee_model: str | None
    funding_model: str | None
    pnl_complete: bool
    no_exchange_dispatch: bool
    checked_at: datetime

    def __post_init__(self) -> None:
        if self.coverage_state == "covered":
            required_ids = (
                self.strategy_id,
                self.strategy_run_id,
                self.strategy_signal_id,
                self.source_event_id,
                self.intent_id,
                self.paper_order_id,
                self.paper_fill_id,
                self.accounting_id,
            )
            if any(value is None for value in required_ids):
                raise ValueError("covered paper scenario requires full execution ids")
        if not self.scenario_key.strip():
            raise ValueError("PaperScenarioCoverageResult scenario_key must be non-empty")
        if not self.coverage_reason.strip():
            raise ValueError("PaperScenarioCoverageResult coverage_reason must be non-empty")
        if self.mode != "paper":
            raise ValueError("PaperScenarioCoverageResult mode must be paper")
        if self.market_type not in {"spot", "futures"}:
            raise ValueError("PaperScenarioCoverageResult market_type is unsupported")
        if self.entry_sizing not in {"fixed_quote", "fixed_equity_pct"}:
            raise ValueError("PaperScenarioCoverageResult entry_sizing is unsupported")
        if self.risk_mode != "single_position_cap":
            raise ValueError("PaperScenarioCoverageResult risk_mode is unsupported")
        if self.direction not in {"long", "short"}:
            raise ValueError("PaperScenarioCoverageResult direction is unsupported")
        if not self.no_exchange_dispatch:
            raise ValueError("PaperScenarioCoverageResult requires no_exchange_dispatch=true")
