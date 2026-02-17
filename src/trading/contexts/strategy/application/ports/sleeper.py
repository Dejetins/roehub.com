from __future__ import annotations

from typing import Protocol


class StrategyRunnerSleeper(Protocol):
    """
    StrategyRunnerSleeper — deterministic sleep abstraction for repair(read) retry backoff.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/services/live_runner.py
      - src/trading/contexts/strategy/adapters/outbound/time/system_runner_sleeper.py
    """

    def sleep(self, *, seconds: float) -> None:
        """
        Sleep current thread for configured backoff duration.

        Args:
            seconds: Non-negative sleep duration in seconds.
        Returns:
            None.
        Assumptions:
            Strategy live-runner uses bounded retries and deterministic backoff policy.
        Raises:
            ValueError: If implementation rejects invalid duration values.
        Side Effects:
            Blocks current execution context for requested duration.
        """
        ...
