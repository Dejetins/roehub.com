from __future__ import annotations

from dataclasses import dataclass

EXCHANGE_CONTROL_SERVICE_IDENTITY = "exchange-control"


@dataclass(frozen=True)
class ExchangeControlServiceIdentity:
    name: str = EXCHANGE_CONTROL_SERVICE_IDENTITY

    def __post_init__(self) -> None:
        if self.name != EXCHANGE_CONTROL_SERVICE_IDENTITY:
            raise ValueError("exchange-control service identity must be 'exchange-control'")


def build_exchange_control_service_identity(*, name: str) -> ExchangeControlServiceIdentity:
    return ExchangeControlServiceIdentity(name=name)


__all__ = [
    "EXCHANGE_CONTROL_SERVICE_IDENTITY",
    "ExchangeControlServiceIdentity",
    "build_exchange_control_service_identity",
]

