from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId

if TYPE_CHECKING:
    from trading.contexts.backtest.application.ai_configurator.dto import (
        BacktestAiConfigEvent,
        BacktestAiConfigJob,
        BacktestAiConfigLlmAttempt,
        BacktestAiConfigTerminalState,
        BacktestAiQuotaEvent,
    )


class BacktestAiConfigJobRepository(Protocol):
    def create_with_quota_event(
        self,
        *,
        job: BacktestAiConfigJob,
        event: BacktestAiConfigEvent,
        quota_event: BacktestAiQuotaEvent,
    ) -> BacktestAiConfigJob:
        """
        Atomically persist one queued AI config job, initial event and quota charge.
        """
        ...

    def get(
        self,
        *,
        job_id: UUID,
        owner_user_id: UserId | None = None,
    ) -> BacktestAiConfigJob | None:
        """
        Load one AI config job with optional owner filter.
        """
        ...

    def find_by_idempotency_key(
        self,
        *,
        owner_user_id: UserId,
        idempotency_key: str,
    ) -> BacktestAiConfigJob | None:
        """
        Resolve one owner-scoped idempotent create request.
        """
        ...

    def record_quota_event(self, *, event: BacktestAiQuotaEvent) -> None:
        """
        Persist quota/capacity rejections that do not have a job row.
        """
        ...

    def append_event(self, *, event: BacktestAiConfigEvent) -> None:
        """
        Append an observable, non-reasoning event for SSE/poll replay.
        """
        ...

    def record_llm_attempt(self, *, attempt: BacktestAiConfigLlmAttempt) -> None:
        """
        Persist raw generate/repair attempt audit data for later scrubbed export.
        """
        ...

    def count_quota_events(
        self,
        *,
        owner_user_id: UserId,
        occurred_after: datetime,
    ) -> int:
        """
        Count charged logical AI configurator requests in a quota window.
        """
        ...

    def count_queued_for_user(self, *, owner_user_id: UserId) -> int:
        """
        Count owner queued AI config jobs for admission control.
        """
        ...

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        """
        Count owner active AI config jobs for admission control.
        """
        ...

    def count_active_global(self) -> int:
        """
        Count service-wide active AI config jobs for capacity control.
        """
        ...


class BacktestAiConfigLeaseRepository(Protocol):
    def claim_next(
        self,
        *,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
        max_attempts: int,
    ) -> BacktestAiConfigJob | None:
        """
        Claim one queued or expired-lease job using durable lease semantics.
        """
        ...

    def heartbeat(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        lease_seconds: int,
    ) -> BacktestAiConfigJob | None:
        """
        Extend a running/repairing job lease if the worker still owns it.
        """
        ...

    def finish(
        self,
        *,
        job_id: UUID,
        now: datetime,
        locked_by: str,
        next_state: BacktestAiConfigTerminalState,
        assistant_message: str | None = None,
        validated_config_json: dict[str, object] | None = None,
        suggestions_json: tuple[dict[str, object], ...] = (),
        validation_errors_json: tuple[dict[str, object], ...] = (),
        model_id: str | None = None,
        model_path_hash: str | None = None,
        last_error: str | None = None,
        last_error_json: dict[str, object] | None = None,
    ) -> BacktestAiConfigJob | None:
        """
        Finish a leased job in one deterministic terminal state.
        """
        ...


__all__ = [
    "BacktestAiConfigJobRepository",
    "BacktestAiConfigLeaseRepository",
]
