from __future__ import annotations

import time

from trading.contexts.strategy.application.ports import StrategyRunnerSleeper


class SystemRunnerSleeper(StrategyRunnerSleeper):
    """
    SystemRunnerSleeper — wall-clock sleeper for live-runner repair backoff policy.

    Docs:
      - docs/architecture/strategy/strategy-live-runner-redis-streams-v1.md
    Related:
      - src/trading/contexts/strategy/application/ports/sleeper.py
      - src/trading/contexts/strategy/application/services/live_runner.py
      - apps/worker/strategy_live_runner/wiring/modules/strategy_live_runner.py
    """

    def sleep(self, *, seconds: float) -> None:
        """
        Sleep process thread for provided non-negative duration.

        Args:
            seconds: Sleep duration in seconds.
        Returns:
            None.
        Assumptions:
            Retry backoff durations are bounded by live-runner runtime config.
        Raises:
            ValueError: If duration is negative.
        Side Effects:
            Blocks current thread.
        """
        if seconds < 0:
            raise ValueError("SystemRunnerSleeper.seconds must be non-negative")
        if seconds == 0:
            return
        time.sleep(seconds)
