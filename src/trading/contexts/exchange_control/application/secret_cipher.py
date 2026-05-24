from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

TRANSIT_KEY_NAME = "roehub-exchange-credentials"


class ExchangeSecretCipherError(RuntimeError):
    """Sanitized secret-cipher error safe for logs and API responses."""


@dataclass(frozen=True, repr=False)
class ExchangeCredentialSecret:
    value: str

    def __post_init__(self) -> None:
        if self.value == "":
            raise ValueError("exchange credential secret must not be empty")

    def as_base64(self) -> str:
        return base64.b64encode(self.value.encode("utf-8")).decode("ascii")

    @classmethod
    def from_base64(cls, value: str) -> "ExchangeCredentialSecret":
        try:
            decoded = base64.b64decode(value.encode("ascii"), validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise ExchangeSecretCipherError("transit plaintext response is invalid") from exc
        return cls(value=decoded)

    def __repr__(self) -> str:
        return "ExchangeCredentialSecret(<redacted>)"


@dataclass(frozen=True, repr=False)
class ExchangeCredentialCiphertext:
    value: str

    def __post_init__(self) -> None:
        if not self.value.startswith("vault:v"):
            raise ValueError("exchange credential ciphertext must be a Transit ciphertext")

    def __repr__(self) -> str:
        return "ExchangeCredentialCiphertext(<redacted>)"


@dataclass(frozen=True, repr=False)
class ExchangeCredentialFingerprint:
    value: str

    def __post_init__(self) -> None:
        if self.value == "":
            raise ValueError("exchange credential fingerprint must not be empty")

    def __repr__(self) -> str:
        return "ExchangeCredentialFingerprint(<redacted>)"


@runtime_checkable
class ExchangeSecretCipher(Protocol):
    def encrypt(self, secret: ExchangeCredentialSecret) -> ExchangeCredentialCiphertext: ...

    def decrypt(self, ciphertext: ExchangeCredentialCiphertext) -> ExchangeCredentialSecret: ...

    def fingerprint(self, secret: ExchangeCredentialSecret) -> ExchangeCredentialFingerprint: ...


class DeterministicInMemoryExchangeSecretCipher:
    """Test/dev-only cipher with deterministic output and no external runtime."""

    _prefix = "vault:v1:deterministic:"

    def __init__(self, *, key_name: str = TRANSIT_KEY_NAME) -> None:
        if key_name != TRANSIT_KEY_NAME:
            raise ValueError("exchange-control Transit key must be roehub-exchange-credentials")
        self._key_name = key_name

    def encrypt(self, secret: ExchangeCredentialSecret) -> ExchangeCredentialCiphertext:
        digest = hashlib.sha256(f"{self._key_name}:{secret.value}".encode("utf-8")).hexdigest()
        return ExchangeCredentialCiphertext(value=f"{self._prefix}{digest}")

    def decrypt(self, ciphertext: ExchangeCredentialCiphertext) -> ExchangeCredentialSecret:
        raise ExchangeSecretCipherError("exchange secret decrypt is unavailable for test cipher")

    def fingerprint(self, secret: ExchangeCredentialSecret) -> ExchangeCredentialFingerprint:
        digest = hmac.new(
            self._key_name.encode("utf-8"),
            secret.value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return ExchangeCredentialFingerprint(value=f"hmac-sha256:{digest}")


__all__ = [
    "TRANSIT_KEY_NAME",
    "DeterministicInMemoryExchangeSecretCipher",
    "ExchangeCredentialCiphertext",
    "ExchangeCredentialFingerprint",
    "ExchangeCredentialSecret",
    "ExchangeSecretCipher",
    "ExchangeSecretCipherError",
]
