from __future__ import annotations

from threading import Event, Lock
from time import monotonic
from typing import Callable


class BacktestRunCancelledV1(ValueError):
    """
    Cooperative cancellation error for sync/job staged loops.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-job-runner-worker-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - apps/api/routes/backtests.py
    """

    def __init__(self, *, reason: str, stage: str) -> None:
        """
        Initialize deterministic cancellation error payload.

        Args:
            reason: Cancellation reason literal.
            stage: Current staged-run stage literal.
        Returns:
            None.
        Assumptions:
            Reason and stage are non-empty stable diagnostic strings.
        Raises:
            ValueError: If reason or stage is empty.
        Side Effects:
            None.
        """
        normalized_reason = reason.strip().lower()
        normalized_stage = stage.strip().lower()
        if not normalized_reason:
            raise ValueError("BacktestRunCancelledV1 reason must be non-empty")
        if not normalized_stage:
            raise ValueError("BacktestRunCancelledV1 stage must be non-empty")
        self.reason = normalized_reason
        self.stage = normalized_stage
        super().__init__(
            "Backtest run cancelled cooperatively "
            f"(reason={self.reason}, stage={self.stage})"
        )


class BacktestRunControlV1:
    """
    Thread-safe cooperative run-control primitive for staged backtest execution.

    Docs:
      - docs/architecture/backtest/backtest-refactor-perf-plan-v1.md
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - src/trading/contexts/backtest/application/services/staged_core_runner_v1.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
      - apps/api/routes/backtests.py
    """

    def __init__(
        self,
        *,
        deadline_seconds: float | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        """
        Initialize cooperative run-control with optional hard wall-time deadline.

        Args:
            deadline_seconds: Optional wall-time deadline in seconds from construction.
            monotonic_clock: Optional monotonic clock provider for deterministic tests.
        Returns:
            None.
        Assumptions:
            Deadline is absolute for the full run and must be positive when configured.
        Raises:
            ValueError: If configured deadline is non-positive.
        Side Effects:
            Captures start monotonic timestamp.
        """
        if deadline_seconds is not None and deadline_seconds <= 0.0:
            raise ValueError("BacktestRunControlV1.deadline_seconds must be > 0")
        self._clock = monotonic if monotonic_clock is None else monotonic_clock
        self._started_at = float(self._clock())
        self._deadline_seconds = float(deadline_seconds) if deadline_seconds is not None else None
        self._cancelled = Event()
        self._reason_lock = Lock()
        self._cancel_reason: str | None = None

    def cancel(self, *, reason: str) -> None:
        """
        Mark run as cancelled from external controller (disconnect, explicit signal, etc).

        Args:
            reason: Cancellation reason literal.
        Returns:
            None.
        Assumptions:
            First cancellation reason wins and stays immutable for diagnostics.
        Raises:
            ValueError: If reason is empty.
        Side Effects:
            Sets cancellation event for all threads checking this control object.
        """
        normalized_reason = reason.strip().lower()
        if not normalized_reason:
            raise ValueError("BacktestRunControlV1.cancel reason must be non-empty")
        with self._reason_lock:
            if self._cancel_reason is None:
                self._cancel_reason = normalized_reason
        self._cancelled.set()

    def raise_if_cancelled(self, *, stage: str) -> None:
        """
        Raise cooperative cancellation error when token is cancelled or deadline exceeded.

        Args:
            stage: Current stage literal where check is executed.
        Returns:
            None.
        Assumptions:
            Method is called repeatedly inside hot loops (Stage A / Stage B / streaming jobs).
        Raises:
            BacktestRunCancelledV1: If token is cancelled or hard deadline is exceeded.
            ValueError: If stage is empty.
        Side Effects:
            Auto-cancels token with `deadline_exceeded` reason when deadline elapsed.
        """
        normalized_stage = stage.strip().lower()
        if not normalized_stage:
            raise ValueError("BacktestRunControlV1.raise_if_cancelled stage must be non-empty")
        if self._deadline_seconds is not None:
            elapsed = float(self._clock()) - self._started_at
            if elapsed >= self._deadline_seconds:
                self.cancel(reason="deadline_exceeded")
        if not self._cancelled.is_set():
            return
        reason = self.cancel_reason or "cancelled"
        raise BacktestRunCancelledV1(reason=reason, stage=normalized_stage)

    @property
    def cancel_reason(self) -> str | None:
        """
        Return first cancellation reason when token is cancelled.

        Args:
            None.
        Returns:
            str | None: First cancellation reason or `None` when still active.
        Assumptions:
            Cancellation reason is immutable after first transition.
        Raises:
            None.
        Side Effects:
            None.
        """
        with self._reason_lock:
            return self._cancel_reason


__all__ = [
    "BacktestRunCancelledV1",
    "BacktestRunControlV1",
]
