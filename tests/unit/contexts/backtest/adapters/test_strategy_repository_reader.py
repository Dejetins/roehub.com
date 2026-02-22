from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import UUID

import pytest

from trading.contexts.backtest.adapters.outbound import StrategyRepositoryBacktestStrategyReader
from trading.contexts.backtest.domain.errors import BacktestStorageError
from trading.contexts.strategy.adapters.outbound import InMemoryStrategyRepository
from trading.contexts.strategy.domain.entities import Strategy, StrategySpecV1
from trading.shared_kernel.primitives import UserId


@dataclass(frozen=True, slots=True)
class _SpecWrapper:
    """
    Strategy spec wrapper overriding `to_json` payload for ACL adapter tests.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - tests/unit/contexts/backtest/adapters/test_strategy_repository_reader.py
      - src/trading/contexts/backtest/adapters/outbound/acl/strategy_repository_reader.py
      - src/trading/contexts/strategy/domain/entities/strategy_spec_v1.py
    """

    base_spec: StrategySpecV1
    payload: Mapping[str, Any]

    @property
    def instrument_id(self):
        """
        Proxy `instrument_id` property from wrapped base spec.

        Args:
            None.
        Returns:
            object: Shared-kernel instrument id value object.
        Assumptions:
            Wrapped base spec remains immutable during test execution.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.base_spec.instrument_id

    @property
    def timeframe(self):
        """
        Proxy `timeframe` property from wrapped base spec.

        Args:
            None.
        Returns:
            object: Shared-kernel timeframe value object.
        Assumptions:
            Wrapped base spec remains immutable during test execution.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.base_spec.timeframe

    @property
    def indicators(self):
        """
        Proxy `indicators` property from wrapped base spec.

        Args:
            None.
        Returns:
            tuple[Mapping[str, Any], ...]: Wrapped indicators tuple.
        Assumptions:
            Wrapped base spec remains immutable during test execution.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.base_spec.indicators

    def to_json(self) -> dict[str, Any]:
        """
        Return custom payload used by ACL parser tests.

        Args:
            None.
        Returns:
            dict[str, Any]: Deterministic custom spec JSON payload.
        Assumptions:
            Payload contains only JSON-compatible values.
        Raises:
            None.
        Side Effects:
            None.
        """
        return dict(self.payload)


def test_strategy_repository_reader_maps_saved_snapshot_payload() -> None:
    """
    Verify ACL reader maps saved strategy snapshot with signal/risk/execution payload blocks.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Saved strategy payload may include optional backtest-specific fields.
    Raises:
        AssertionError: If mapped snapshot fields differ from deterministic contract.
    Side Effects:
        None.
    """
    strategy = _build_strategy(strategy_id=UUID("00000000-0000-0000-0000-000000000111"))
    spec_payload = strategy.spec.to_json()
    spec_payload.update(
        {
            "direction_mode": "long-only",
            "sizing_mode": "fixed_quote",
            "signal_grids": {
                "ma.sma": {
                    "cross_up": {"mode": "explicit", "values": [0.4, 0.6]},
                }
            },
            "risk_grid": {
                "sl_enabled": True,
                "tp_enabled": True,
                "sl": {"mode": "explicit", "values": [2.0]},
                "tp": {"mode": "range", "start": 4.0, "stop_incl": 6.0, "step": 2.0},
            },
            "risk": {
                "sl_enabled": True,
                "tp_enabled": True,
                "sl_pct": 2.0,
                "tp_pct": 6.0,
            },
            "execution": {
                "init_cash_quote": 5000.0,
                "fee_pct": 0.05,
            },
        }
    )
    object.__setattr__(
        strategy,
        "spec",
        _SpecWrapper(base_spec=strategy.spec, payload=spec_payload),
    )

    repository = InMemoryStrategyRepository()
    repository.create(strategy=strategy)

    reader = StrategyRepositoryBacktestStrategyReader(repository=repository)
    snapshot = reader.load_any(strategy_id=strategy.strategy_id)

    assert snapshot is not None
    assert snapshot.direction_mode == "long-only"
    assert snapshot.sizing_mode == "fixed_quote"
    assert snapshot.signal_grids is not None
    assert snapshot.risk_grid is not None
    assert snapshot.risk_grid.sl is not None
    assert snapshot.risk_grid.tp is not None
    assert snapshot.signal_grids["ma.sma"]["cross_up"].materialize() == (0.4, 0.6)
    assert snapshot.risk_grid.sl_enabled is True
    assert snapshot.risk_grid.tp_enabled is True
    assert snapshot.risk_grid.sl.materialize() == (2.0,)
    assert snapshot.risk_grid.tp.materialize() == (4.0, 6.0)
    assert snapshot.risk_params == {
        "sl_enabled": True,
        "sl_pct": 2.0,
        "tp_enabled": True,
        "tp_pct": 6.0,
    }
    assert snapshot.execution_params == {"fee_pct": 0.05, "init_cash_quote": 5000.0}
    assert snapshot.spec_payload is not None
    assert snapshot.spec_payload["direction_mode"] == "long-only"


def test_strategy_repository_reader_returns_none_when_strategy_missing() -> None:
    """
    Verify ACL reader returns `None` for missing strategy id.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Missing saved strategy lookup should not raise adapter errors.
    Raises:
        AssertionError: If missing strategy does not return `None`.
    Side Effects:
        None.
    """
    reader = StrategyRepositoryBacktestStrategyReader(repository=InMemoryStrategyRepository())

    snapshot = reader.load_any(strategy_id=UUID("00000000-0000-0000-0000-000000000999"))

    assert snapshot is None


def test_strategy_repository_reader_raises_storage_error_on_invalid_signal_grid_mode() -> None:
    """
    Verify ACL reader raises deterministic storage error on malformed signal-grid payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Unknown `signal_grids.*.mode` literals are invalid and must fail fast.
    Raises:
        AssertionError: If malformed payload does not raise `BacktestStorageError`.
    Side Effects:
        None.
    """
    strategy = _build_strategy(strategy_id=UUID("00000000-0000-0000-0000-000000000222"))
    spec_payload = strategy.spec.to_json()
    spec_payload["signal_grids"] = {
        "ma.sma": {
            "cross_up": {"mode": "broken", "values": [0.4]},
        }
    }
    object.__setattr__(
        strategy,
        "spec",
        _SpecWrapper(base_spec=strategy.spec, payload=spec_payload),
    )

    repository = InMemoryStrategyRepository()
    repository.create(strategy=strategy)
    reader = StrategyRepositoryBacktestStrategyReader(repository=repository)

    with pytest.raises(BacktestStorageError, match="signal_grids\\.ma\\.sma\\.cross_up\\.mode"):
        reader.load_any(strategy_id=strategy.strategy_id)


def _build_strategy(*, strategy_id: UUID) -> Strategy:
    """
    Build deterministic Strategy aggregate fixture for ACL adapter unit tests.

    Args:
        strategy_id: Strategy identifier.
    Returns:
        Strategy: Persistable strategy aggregate fixture.
    Assumptions:
        StrategySpec payload follows v1 immutable storage contract.
    Raises:
        ValueError: If payload violates domain invariants.
    Side Effects:
        None.
    """
    spec = StrategySpecV1.from_json(
        payload={
            "instrument_id": {
                "market_id": 1,
                "symbol": "BTCUSDT",
            },
            "instrument_key": "binance:spot:BTCUSDT",
            "market_type": "spot",
            "timeframe": "1m",
            "signal_template": "MA(20)",
            "indicators": [
                {
                    "id": "ma.sma",
                    "inputs": {"source": "close"},
                    "params": {"window": 20},
                }
            ],
        }
    )

    return Strategy.create(
        strategy_id=strategy_id,
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000777"),
        spec=spec,
        created_at=datetime(2026, 2, 16, 0, 0, tzinfo=timezone.utc),
    )
