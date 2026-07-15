from __future__ import annotations

from trading.contexts.backtest.application.services import (
    RESEARCH_REQUEST_NAMESPACE,
    build_research_content_hash,
    build_research_idempotency_key_hash,
)
from trading.shared_kernel.primitives import OrganizationId


def test_idempotency_hash_is_organization_scoped_and_versioned() -> None:
    organization_a = OrganizationId.from_string(
        "00000000-0000-0000-0000-000000000011"
    )
    organization_b = OrganizationId.from_string(
        "00000000-0000-0000-0000-000000000012"
    )

    digest_a = build_research_idempotency_key_hash(
        organization_id=organization_a,
        idempotency_key="same-key",
    )
    digest_a_replay = build_research_idempotency_key_hash(
        organization_id=organization_a,
        idempotency_key=" same-key ",
    )
    digest_b = build_research_idempotency_key_hash(
        organization_id=organization_b,
        idempotency_key="same-key",
    )

    assert RESEARCH_REQUEST_NAMESPACE == "research-request/v1"
    assert digest_a == digest_a_replay
    assert digest_a != digest_b
    assert len(digest_a) == 64


def test_content_hash_is_canonical_and_organization_neutral() -> None:
    payload_a = {
        "coordinates": {"market_id": 1, "symbol": "BTCUSDT"},
        "indicators": [{"id": "ma.sma", "params": {"period": 20}}],
    }
    payload_b = {
        "indicators": [{"params": {"period": 20}, "id": "ma.sma"}],
        "coordinates": {"symbol": "BTCUSDT", "market_id": 1},
    }

    digest_a = build_research_content_hash(payload=payload_a)
    digest_b = build_research_content_hash(payload=payload_b)

    assert digest_a == digest_b
    assert len(digest_a) == 64
