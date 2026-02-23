from __future__ import annotations

from typing import Any, Mapping, Protocol

from trading.contexts.backtest.application.dto import RunBacktestRequest


class BacktestJobRequestDecoder(Protocol):
    """
    Decode persisted Backtest job `request_json` payload into application request DTO.

    Docs:
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
      - docs/architecture/backtest/backtest-jobs-storage-pg-state-machine-v1.md
    Related:
      - src/trading/contexts/backtest/application/dto/run_backtest.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest_job_runner_v1.py
      - apps/worker/backtest_job_runner/wiring/modules/backtest_job_runner.py
    """

    def decode(self, *, payload: Mapping[str, Any]) -> RunBacktestRequest:
        """
        Decode persisted JSON payload into deterministic `RunBacktestRequest`.

        Args:
            payload: Persisted `backtest_jobs.request_json` mapping payload.
        Returns:
            RunBacktestRequest: Decoded application-layer request DTO.
        Assumptions:
            Payload shape follows canonical `POST /backtests` request semantics.
        Raises:
            ValueError: If payload cannot be converted to deterministic request DTO.
        Side Effects:
            None.
        """
        ...


__all__ = ["BacktestJobRequestDecoder"]
