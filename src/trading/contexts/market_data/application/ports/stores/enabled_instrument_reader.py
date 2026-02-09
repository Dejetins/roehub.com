from __future__ import annotations

from typing import Protocol, Sequence

from trading.shared_kernel.primitives import InstrumentId


class EnabledInstrumentReader(Protocol):
    """
    Read enabled tradable instruments from reference storage.

    Contract:
    - list_enabled_tradable() -> Sequence[InstrumentId]

    Semantics:
    - returns instruments satisfying `status='ENABLED'` and `is_tradable=1`.
    - order is implementation-defined.
    """

    def list_enabled_tradable(self) -> Sequence[InstrumentId]:
        ...

