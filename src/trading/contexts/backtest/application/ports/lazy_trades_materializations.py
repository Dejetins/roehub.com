from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping, Protocol
from uuid import UUID

from trading.shared_kernel.primitives import UserId

BacktestLazyTradesMaterializationStatus = Literal[
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
]


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesMaterializationRequest:
    owner_user_id: UserId
    job_id: UUID
    public_variant_key: str
    variant_hash: str
    request_hash: str
    engine_params_hash: str
    artifact_manifest_hash: str
    cache_key: str
    cache_status: str
    ttl_seconds: int
    requested_at: datetime
    priority_class: str = "interactive"


@dataclass(frozen=True, slots=True)
class BacktestLazyTradesMaterializationTask:
    task_id: UUID
    owner_user_id: UserId
    job_id: UUID
    public_variant_key: str
    variant_hash: str
    request_hash: str
    engine_params_hash: str
    artifact_manifest_hash: str
    cache_key: str
    status: BacktestLazyTradesMaterializationStatus
    priority_class: str
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    locked_by: str | None
    locked_at: datetime | None
    lease_expires_at: datetime | None
    heartbeat_at: datetime | None
    attempt: int
    last_error: str | None
    last_error_json: Mapping[str, Any] | None
    cache_status: str
    cache_path: str | None
    ttl_seconds: int


class BacktestLazyTradesMaterializationRepository(Protocol):
    def find_by_identity(
        self,
        *,
        owner_user_id: UserId,
        job_id: UUID,
        public_variant_key: str,
        cache_key: str,
    ) -> BacktestLazyTradesMaterializationTask | None:
        """
        Find an existing owner-scoped task before quota evaluation.

        Replays for the same owner/job/public variant/cache identity must not consume
        a new lazy-detail quota slot.
        """
        ...

    def request_materialization(
        self,
        *,
        request: BacktestLazyTradesMaterializationRequest,
    ) -> BacktestLazyTradesMaterializationTask:
        """
        Create or replay one owner-scoped lazy trades materialization task.

        The operation is idempotent for the same owner/job/public variant/cache identity.
        """
        ...

    def count_active_for_user(self, *, owner_user_id: UserId) -> int:
        """
        Count owner active lazy detail tasks (`queued + running`) for admission.
        """
        ...

    def count_created_for_user_since(
        self,
        *,
        owner_user_id: UserId,
        created_after: datetime,
    ) -> int:
        """
        Count owner lazy detail materialization creates inside one admission-rate window.
        """
        ...

    def count_active_global(self) -> int:
        """
        Count all active lazy detail tasks (`queued + running`) for global saturation.
        """
        ...


__all__ = [
    "BacktestLazyTradesMaterializationRepository",
    "BacktestLazyTradesMaterializationRequest",
    "BacktestLazyTradesMaterializationStatus",
    "BacktestLazyTradesMaterializationTask",
]
