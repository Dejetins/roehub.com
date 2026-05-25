from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .secret_cipher import (
    ExchangeCredentialSecret,
    ExchangeSecretCipher,
    ExchangeSecretCipherError,
)
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
    def __init__(
        self,
        *,
        service_identity: ExchangeControlServiceIdentity,
        secret_cipher: ExchangeSecretCipher,
        transit_required: bool = False,
    ) -> None:
        self._service_identity = service_identity
        self._secret_cipher = secret_cipher
        self._transit_required = transit_required

    def check(self) -> ExchangeControlReadiness:
        checks = [
            ExchangeControlReadinessCheck(name="service_identity", status="ready"),
            ExchangeControlReadinessCheck(name="external_exchange_validation", status="ready"),
        ]
        if self._transit_required:
            checks.append(
                ExchangeControlReadinessCheck(
                    name="secret_cipher_transit",
                    status=self._check_transit(),
                )
            )
        status: ReadinessStatus = (
            "ready" if all(check.status == "ready" for check in checks) else "not_ready"
        )
        return ExchangeControlReadiness(
            status=status,
            service="exchange-control",
            service_identity=self._service_identity.name,
            checks=tuple(checks),
        )

    def _check_transit(self) -> ReadinessStatus:
        try:
            self._secret_cipher.fingerprint(
                ExchangeCredentialSecret(value="roehub-readiness")
            )
        except ExchangeSecretCipherError:
            return "not_ready"
        return "ready"


__all__ = [
    "ExchangeControlReadiness",
    "ExchangeControlReadinessCheck",
    "ExchangeControlReadinessProbe",
]
