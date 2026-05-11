from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from trading.contexts.backtest.application.ports.backtest_ai_configurator import (
    BacktestAiConfigJobRepository,
    BacktestAiConfigLeaseRepository,
)

from .dto import BacktestAiConfigEvent, BacktestAiConfigEventName, BacktestAiConfigJob

_FAKE_WORKER_ID = "backtest-ai-config-fake-worker"
_FAKE_MODEL_ID = "deterministic-fake-worker-v1"
_FAKE_STAGE_EVENTS: tuple[tuple[str, str, int], ...] = (
    ("preparing_catalog", "Preparing the current /backtests catalog.", 20),
    ("assembling_prompt", "Assembling a deterministic placeholder request.", 35),
    ("generating", "Generating a placeholder configuration.", 55),
    ("validating_json", "Checking placeholder JSON shape.", 70),
    ("validating_business", "Checking /backtests business rules.", 85),
)


@dataclass(frozen=True, slots=True)
class BacktestAiConfigFakeWorkerUseCase:
    """
    Deterministic Iteration 02 worker path that proves queue and event semantics.
    """

    job_repository: BacktestAiConfigJobRepository
    lease_repository: BacktestAiConfigLeaseRepository
    lease_seconds: int = 60
    max_attempts: int = 1
    locked_by: str = _FAKE_WORKER_ID

    def process_next(self, *, now: datetime | None = None) -> BacktestAiConfigJob | None:
        effective_now = datetime.now(UTC) if now is None else now
        claimed = self.lease_repository.claim_next(
            now=effective_now,
            locked_by=self.locked_by,
            lease_seconds=self.lease_seconds,
            max_attempts=self.max_attempts,
        )
        if claimed is None:
            return None

        for event_name, message, progress in _FAKE_STAGE_EVENTS:
            self.job_repository.append_event(
                event=_event(
                    job=claimed,
                    event_name=event_name,
                    message=message,
                    progress=progress,
                    created_at=effective_now,
                )
            )

        finished = self.lease_repository.finish(
            job_id=claimed.job_id,
            now=effective_now,
            locked_by=self.locked_by,
            next_state="ready",
            assistant_message=(
                "Я собрал тестовую конфигурацию для BTCUSDT на 15m. "
                "Это deterministic fake response без MLX runtime."
            ),
            validated_config_json=_placeholder_config(),
            suggestions_json=(
                {"message": "Добавить stop loss / take profit grid"},
            ),
            validation_errors_json=(),
            model_id=_FAKE_MODEL_ID,
            model_path_hash=None,
        )
        if finished is None:
            return None

        self.job_repository.append_event(
            event=_event(
                job=finished,
                event_name="ready",
                message="Configuration is ready to load.",
                progress=100,
                created_at=effective_now,
            )
        )
        return finished


def _event(
    *,
    job: BacktestAiConfigJob,
    event_name: str,
    message: str,
    progress: int,
    created_at: datetime,
) -> BacktestAiConfigEvent:
    return BacktestAiConfigEvent(
        event_id=uuid4(),
        job_id=job.job_id,
        owner_user_id=job.owner_user_id,
        event_name=cast(BacktestAiConfigEventName, event_name),
        message=message,
        payload_json={
            "job_id": str(job.job_id),
            "status": event_name,
            "message": message,
            "progress": progress,
        },
        created_at=created_at,
    )


def _placeholder_config() -> dict[str, Any]:
    return {
        "coordinates": {
            "exchange": "binance",
            "market_type": "spot",
            "symbol": "BTCUSDT",
        },
        "timeframe": "15m",
        "time_range": {
            "start": "2023-01-01T00:00:00Z",
            "end": "2024-01-01T00:00:00Z",
        },
        "indicators": [
            {
                "indicator_id": "momentum.rsi",
                "sources": ["close"],
                "window": {"start": 7, "stop": 28, "step": 7},
            }
        ],
        "risk": {"mode": "none"},
        "execution": {
            "direction_mode": "long_short_reversal",
            "fee_rate": 0.00075,
            "slippage_rate": 0.0001,
            "initial_cash_quote": 10000,
            "sizing": {"mode": "fixed_equity_pct", "equity_pct": 10},
            "profit_lock": {"enabled": False},
            "close_on_end": True,
        },
        "ranking": {"primary_metric": "total_return_pct", "direction": "desc"},
        "top_n": 100,
    }


__all__ = ["BacktestAiConfigFakeWorkerUseCase"]
