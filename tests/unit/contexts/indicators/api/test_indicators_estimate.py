from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.routes import build_indicators_router
from trading.contexts.indicators.application.ports.registry import IndicatorRegistry
from trading.contexts.indicators.domain.definitions import all_defs
from trading.contexts.indicators.domain.entities import IndicatorDef, IndicatorId
from trading.contexts.indicators.domain.errors import UnknownIndicatorError


class _RegistryStub(IndicatorRegistry):
    """
    Deterministic indicator registry stub for API endpoint unit tests.
    """

    def __init__(self, *, defs: tuple[IndicatorDef, ...]) -> None:
        """
        Build immutable lookup map from provided hard definitions.

        Args:
            defs: Hard indicator definitions.
        Returns:
            None.
        Assumptions:
            Indicator ids are unique across provided definitions.
        Raises:
            ValueError: If duplicate indicator id is detected.
        Side Effects:
            None.
        """
        defs_by_id: dict[str, IndicatorDef] = {}
        for definition in defs:
            key = definition.indicator_id.value
            if key in defs_by_id:
                raise ValueError(f"duplicate indicator_id: {key}")
            defs_by_id[key] = definition
        self._defs = defs
        self._defs_by_id = defs_by_id

    def list_defs(self) -> tuple[IndicatorDef, ...]:
        """
        Return hard definitions in deterministic order.

        Args:
            None.
        Returns:
            tuple[IndicatorDef, ...]: Hard definitions snapshot.
        Assumptions:
            Snapshot is immutable for test runtime.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self._defs

    def get_def(self, indicator_id: IndicatorId) -> IndicatorDef:
        """
        Resolve one definition by id.

        Args:
            indicator_id: Target indicator identifier.
        Returns:
            IndicatorDef: Matching hard definition.
        Assumptions:
            Indicator id normalization is enforced by value object.
        Raises:
            UnknownIndicatorError: If id is absent.
        Side Effects:
            None.
        """
        definition = self._defs_by_id.get(indicator_id.value)
        if definition is None:
            raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")
        return definition

    def list_merged(self) -> tuple[object, ...]:
        """
        Return empty merged view for protocol completeness in these tests.

        Args:
            None.
        Returns:
            tuple[object, ...]: Empty placeholder tuple.
        Assumptions:
            Estimate endpoint tests do not require merged registry payload.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ()

    def get_merged(self, indicator_id: IndicatorId) -> object:
        """
        Always raise because merged item resolution is not used in these tests.

        Args:
            indicator_id: Target indicator identifier.
        Returns:
            object: Never returns.
        Assumptions:
            Merged-view contract is irrelevant for estimate endpoint tests.
        Raises:
            UnknownIndicatorError: Always.
        Side Effects:
            None.
        """
        raise UnknownIndicatorError(f"unknown indicator_id: {indicator_id.value}")


def _client(
    *,
    max_variants_per_compute: int,
    max_compute_bytes_total: int,
) -> TestClient:
    """
    Build TestClient with configurable estimate guards.

    Args:
        max_variants_per_compute: Variants guard limit for router.
        max_compute_bytes_total: Memory guard limit for router.
    Returns:
        TestClient: Ready HTTP client over in-memory FastAPI app.
    Assumptions:
        Registry hard definitions are deterministic for test process.
    Raises:
        ValueError: If router guard configuration is invalid.
    Side Effects:
        None.
    """
    app = FastAPI()
    registry = _RegistryStub(defs=all_defs())
    app.include_router(
        build_indicators_router(
            registry=registry,
            max_variants_per_compute=max_variants_per_compute,
            max_compute_bytes_total=max_compute_bytes_total,
        )
    )
    return TestClient(app)


def _valid_estimate_payload() -> dict[str, Any]:
    """
    Build deterministic valid payload for `POST /indicators/estimate`.

    Args:
        None.
    Returns:
        dict[str, Any]: Request payload with two indicators and SL/TP axes.
    Assumptions:
        Hard indicator ids and params exist in baseline definitions.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "timeframe": "1m",
        "time_range": {
            "start": "2026-02-11T10:00:00Z",
            "end": "2026-02-11T11:00:00Z",
        },
        "indicators": [
            {
                "indicator_id": "ma.sma",
                "source": {"mode": "explicit", "values": ["open", "close"]},
                "params": {
                    "window": {"mode": "explicit", "values": [10, 20, 30]},
                },
            },
            {
                "indicator_id": "trend.adx",
                "params": {
                    "window": {"mode": "range", "start": 10, "stop_incl": 11, "step": 1},
                    "smoothing": {"mode": "explicit", "values": [14]},
                },
            },
        ],
        "risk": {
            "sl": {"mode": "explicit", "values": [0.01, 0.02]},
            "tp": {"mode": "explicit", "values": [0.03, 0.04, 0.05]},
        },
    }


def test_post_indicators_estimate_returns_totals_only_without_axis_preview() -> None:
    """
    Verify response contract includes only schema version and totals fields.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Valid payload should pass both variants and memory guards.
    Raises:
        AssertionError: If response contains unexpected fields or wrong totals.
    Side Effects:
        None.
    """
    client = _client(
        max_variants_per_compute=600_000,
        max_compute_bytes_total=5 * 1024**3,
    )

    response = client.post("/indicators/estimate", json=_valid_estimate_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "schema_version": 1,
        "total_variants": 72,
        "estimated_memory_bytes": 67_112_464,
    }
    assert "axes" not in payload
    assert "first" not in payload
    assert "last" not in payload


def test_post_indicators_estimate_returns_422_when_variants_guard_exceeded() -> None:
    """
    Verify variants guard returns deterministic payload with actual and limit.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Payload variants total is deterministic and larger than configured limit.
    Raises:
        AssertionError: If status code or payload shape differs from contract.
    Side Effects:
        None.
    """
    client = _client(
        max_variants_per_compute=10,
        max_compute_bytes_total=5 * 1024**3,
    )

    response = client.post("/indicators/estimate", json=_valid_estimate_payload())

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "error": "max_variants_per_compute_exceeded",
            "total_variants": 72,
            "max_variants_per_compute": 10,
        }
    }


def test_post_indicators_estimate_returns_422_when_memory_guard_exceeded() -> None:
    """
    Verify memory guard returns deterministic payload with actual and limit.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Payload memory estimate is deterministic and exceeds low configured limit.
    Raises:
        AssertionError: If status code or payload shape differs from contract.
    Side Effects:
        None.
    """
    client = _client(
        max_variants_per_compute=600_000,
        max_compute_bytes_total=1_000_000,
    )

    response = client.post("/indicators/estimate", json=_valid_estimate_payload())

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "error": "max_compute_bytes_total_exceeded",
            "estimated_memory_bytes": 67_112_464,
            "max_compute_bytes_total": 1_000_000,
        }
    }


def test_post_indicators_estimate_returns_422_for_missing_required_source_axis() -> None:
    """
    Verify source-parametrized indicators require `source` axis in request.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        `ma.sma` requires source axis by hard indicator definition.
    Raises:
        AssertionError: If request is not rejected with deterministic validation payload.
    Side Effects:
        None.
    """
    client = _client(
        max_variants_per_compute=600_000,
        max_compute_bytes_total=5 * 1024**3,
    )
    payload = _valid_estimate_payload()
    payload["indicators"][0].pop("source")

    response = client.post("/indicators/estimate", json=payload)

    assert response.status_code == 422
    assert response.json() == {
        "detail": {
            "error": "grid_validation_error",
            "message": "source axis is required by indicator definition",
        }
    }
