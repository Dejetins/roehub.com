from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from trading.contexts.live_execution.adapters.outbound.persistence.in_memory import (
    InMemoryPaperScenarioCoverageRepository,
)
from trading.contexts.live_execution.application import PaperScenarioCoverageService
from trading.contexts.live_execution.domain import PaperScenarioCoverageResult
from trading.shared_kernel.primitives import UserId

_USER_ID = UserId.from_string("00000000-0000-0000-0000-000000070001")
_SCENARIO_KEY = "a" * 64
_NOW = datetime(2026, 6, 17, 12, 0, tzinfo=UTC)


def test_paper_coverage_records_latest_result_by_scenario_key() -> None:
    repository = InMemoryPaperScenarioCoverageRepository()
    service = PaperScenarioCoverageService(repository=repository)
    first = _coverage_result(
        coverage_result_id=UUID("00000000-0000-0000-0000-000000070101"),
        checked_at=_NOW,
        coverage_reason="paper_no_exchange_submit",
    )
    second = _coverage_result(
        coverage_result_id=UUID("00000000-0000-0000-0000-000000070102"),
        checked_at=_NOW + timedelta(minutes=1),
        coverage_reason="paper_spot_short_borrow_not_modeled",
        pnl_complete=False,
    )

    service.record(result=first)
    recorded = service.record(result=second)
    loaded = service.get_latest_by_scenario_key(
        owner_user_id=_USER_ID,
        scenario_key=_SCENARIO_KEY,
    )

    assert recorded.coverage_result_id == second.coverage_result_id
    assert loaded is not None
    assert loaded.coverage_result_id == second.coverage_result_id
    assert loaded.coverage_reason == "paper_spot_short_borrow_not_modeled"
    assert len(repository.results) == 1


def _coverage_result(
    *,
    coverage_result_id: UUID,
    checked_at: datetime,
    coverage_reason: str,
    pnl_complete: bool = True,
) -> PaperScenarioCoverageResult:
    return PaperScenarioCoverageResult(
        coverage_result_id=coverage_result_id,
        owner_user_id=_USER_ID,
        scenario_matrix_row_id=UUID("00000000-0000-0000-0000-000000070201"),
        scenario_key=_SCENARIO_KEY,
        source_job_id=UUID("00000000-0000-0000-0000-000000070301"),
        source_variant_key="variant-1",
        mode="paper",
        market_type="spot",
        symbol="BTCUSDT",
        entry_sizing="fixed_quote",
        risk_mode="single_position_cap",
        direction="long",
        coverage_state="covered",
        coverage_reason=coverage_reason,
        strategy_id=uuid4(),
        live_profile_id=uuid4(),
        strategy_run_id=uuid4(),
        strategy_signal_id=uuid4(),
        source_event_id=uuid4(),
        intent_id=uuid4(),
        paper_order_id=uuid4(),
        paper_fill_id=uuid4(),
        accounting_id=uuid4(),
        fee_model="paper_fixed_bps_10",
        funding_model="spot_not_applicable",
        pnl_complete=pnl_complete,
        no_exchange_dispatch=True,
        checked_at=checked_at,
    )
