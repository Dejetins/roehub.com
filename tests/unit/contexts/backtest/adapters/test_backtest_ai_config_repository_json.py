from __future__ import annotations

import json
from types import MappingProxyType

from trading.contexts.backtest.adapters.outbound.persistence.postgres import (
    backtest_ai_config_repository,
)


def test_json_dumps_serializes_immutable_mapping_payloads() -> None:
    payload = MappingProxyType(
        {
            "status": "queued",
            "details": MappingProxyType({"attempts": (1, 2)}),
        }
    )

    encoded = backtest_ai_config_repository._json_dumps(payload)

    assert json.loads(encoded) == {
        "details": {"attempts": [1, 2]},
        "status": "queued",
    }
