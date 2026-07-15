from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from uuid import UUID, uuid4

from trading.contexts.rl_trading.domain.risk_sizing_policy import (
    RlRiskSizingPolicyConfig,
    RlRiskSizingPolicyKey,
    RlRiskSizingPolicyRecord,
    validate_rl_risk_sizing_policy,
)


class InMemoryRlRiskSizingPolicyRepository:
    def __init__(self) -> None:
        self._records: dict[
            tuple[str, str, str, str, str, str], RlRiskSizingPolicyRecord
        ] = {}
        self.audit_events: list[dict[str, object]] = []

    def get_policy(self, *, key: RlRiskSizingPolicyKey) -> RlRiskSizingPolicyRecord | None:
        return self._records.get(key.persistence_key)

    def upsert_policy(
        self,
        *,
        key: RlRiskSizingPolicyKey,
        config: RlRiskSizingPolicyConfig,
        observed_at: datetime,
    ) -> RlRiskSizingPolicyRecord:
        previous = self._records.get(key.persistence_key)
        policy_id = previous.policy_id if previous is not None else uuid4()
        created_at = previous.created_at if previous is not None else observed_at
        record = RlRiskSizingPolicyRecord(
            policy_id=policy_id,
            key=key,
            config=config,
            validation=validate_rl_risk_sizing_policy(config=config),
            created_at=created_at,
            updated_at=observed_at,
        )
        self._records[key.persistence_key] = record
        self.audit_events.append(
            {
                "event_id": uuid4(),
                "event_type": "upsert",
                "policy_id": policy_id,
                "key": key.persistence_key,
                "validation_status": record.validation.status,
                "validation_reasons": record.validation.reasons,
                "created_at": observed_at,
            }
        )
        return replace(record)

    def audit_event_count(self, *, policy_id: UUID) -> int:
        return sum(1 for event in self.audit_events if event["policy_id"] == policy_id)
