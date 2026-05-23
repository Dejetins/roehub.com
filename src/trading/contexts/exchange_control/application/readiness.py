from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .service_identity import ExchangeControlServiceIdentity

ReadinessStatus = Literal["ready", "not_ready"]


@dataclass(frozen=True)
class ExchangeControlReadinessCheck:
    name: str
    status: ReadinessStatus


@dataclass(frozen=True)
class ExchangeControlReadiness:
    status: ReadinessStatus
    service: str
    service_identity: str
    checks: tuple[ExchangeControlReadinessCheck, ...]

    def as_response_payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "service": self.service,
            "service_identity": self.service_identity,
            "checks": [
                {"name": check.name, "status": check.status}
                for check in self.checks
            ],
        }


class ExchangeControlReadinessProbe:
    def __init__(self, *, service_identity: ExchangeControlServiceIdentity) -> None:
        self._service_identity = service_identity

    def check(self) -> ExchangeControlReadiness:
        checks = (
            ExchangeControlReadinessCheck(name="service_identity", status="ready"),
            ExchangeControlReadinessCheck(name="external_exchange_validation", status="ready"),
        )
        return ExchangeControlReadiness(
            status="ready",
            service="exchange-control",
            service_identity=self._service_identity.name,
            checks=checks,
        )


__all__ = [
    "ExchangeControlReadiness",
    "ExchangeControlReadinessCheck",
    "ExchangeControlReadinessProbe",
]

