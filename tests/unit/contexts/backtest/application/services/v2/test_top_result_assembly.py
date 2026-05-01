from datetime import UTC, datetime
from uuid import UUID, uuid4

from trading.contexts.backtest.application.dto import (
    BacktestNoRiskTopResult,
    BacktestTpSlTopResult,
)
from trading.contexts.backtest.application.services.v2 import (
    BacktestTopResultAssemblyService,
)


def test_top_result_assembly_keeps_storage_hash_separate_from_public_key() -> None:
    job_id = uuid4()
    result = BacktestTopResultAssemblyService().assemble(
        job_id=job_id,
        normalized_request=_request(risk={"mode": "none"}),
        top_results=(
            BacktestNoRiskTopResult(
                rank=1,
                score=10.0,
                indicator_rows={"ma.dema": 17},
                metrics={"total_return_pct": 10.0, "trade_count": 3.0},
                metadata={
                    "ma.dema.source": "close",
                    "ma.dema.window": 192,
                    "confirm_count": 1,
                    "proxy_score": 0.1,
                },
            ),
        ),
        updated_at=datetime(2026, 5, 1, tzinfo=UTC),
    )

    row = result.top_variants[0]
    public_key = row.payload_json["public_variant_key"]
    assert row.variant_key == row.payload_json["variant_hash"]
    assert len(row.variant_key) == 64
    assert public_key.startswith(
        f"job_{job_id.hex[:8]}{job_id.hex[-4:]}__dema_close_w192__risk_none"
    )
    assert public_key != row.variant_key
    assert row.trades_json is None
    assert row.report_table_md is None
    assert "top_result_assembly" in result.stage_timings


def test_variant_hash_is_stable_across_jobs_but_public_key_is_job_scoped() -> None:
    first = _assemble_for_job(UUID("00000000-0000-0000-0000-000000000701"))
    second = _assemble_for_job(UUID("00000000-0000-0000-0000-000000000702"))

    assert first.variant_key == second.variant_key
    assert first.payload_json["public_variant_key"] != second.payload_json["public_variant_key"]


def test_tp_sl_top_result_assembly_persists_best_cell_summary_only() -> None:
    result = BacktestTopResultAssemblyService().assemble(
        job_id=uuid4(),
        normalized_request=_request(
            risk={
                "mode": "tp_sl_grid",
                "tp": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5},
                "sl": {"start_pct": 2.0, "stop_pct": 25.0, "step_pct": 0.5},
            }
        ),
        top_results=(
            BacktestTpSlTopResult(
                rank=1,
                score=33.0,
                indicator_rows={"ma.dema": 17},
                best_tp_idx=2,
                best_sl_idx=3,
                metrics={
                    "total_return_pct": 33.0,
                    "trade_count": 4.0,
                    "best_tp_pct": 3.0,
                    "best_sl_pct": 3.5,
                },
                metadata={"ma.dema.source": "close", "ma.dema.window": 192},
            ),
        ),
        updated_at=datetime(2026, 5, 1, tzinfo=UTC),
    )

    row = result.top_variants[0]
    assert row.best_tp_pct == 3.0
    assert row.best_sl_pct == 3.5
    assert row.payload_json["readable_params"]["best_tp_pct"] == 3.0
    assert row.trades_json is None


def _assemble_for_job(job_id: UUID):
    return BacktestTopResultAssemblyService().assemble(
        job_id=job_id,
        normalized_request=_request(risk={"mode": "none"}),
        top_results=(
            BacktestNoRiskTopResult(
                rank=1,
                score=10.0,
                indicator_rows={"ma.dema": 17},
                metrics={"total_return_pct": 10.0, "trade_count": 3.0},
                metadata={"ma.dema.source": "close", "ma.dema.window": 192},
            ),
        ),
        updated_at=datetime(2026, 5, 1, tzinfo=UTC),
    ).top_variants[0]


def _request(*, risk: dict):
    return {
        "risk": risk,
        "execution": {"direction_mode": "long_short_reversal", "sizing": {"mode": "all_in"}},
        "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
    }
