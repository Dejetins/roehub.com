from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from trading.contexts.exchange_control.application.secret_cipher import (
    TRANSIT_KEY_NAME,
    ExchangeCredentialCiphertext,
    ExchangeCredentialFingerprint,
    ExchangeCredentialSecret,
    ExchangeSecretCipherError,
)


@dataclass(frozen=True)
class OpenBaoTransitExchangeSecretCipher:
    address: str
    token: str
    key_name: str = TRANSIT_KEY_NAME
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not self.address:
            raise ValueError("OPENBAO_ADDR is required")
        if not self.token:
            raise ValueError("ROEHUB_EXCHANGE_CONTROL_TRANSIT_TOKEN is required")
        if self.key_name != TRANSIT_KEY_NAME:
            raise ValueError("exchange-control Transit key must be roehub-exchange-credentials")

    def encrypt(self, secret: ExchangeCredentialSecret) -> ExchangeCredentialCiphertext:
        payload = self._post_json(
            path=f"/v1/transit/encrypt/{self.key_name}",
            payload={"plaintext": secret.as_base64()},
        )
        ciphertext = _read_string(payload, "data", "ciphertext")
        return ExchangeCredentialCiphertext(value=ciphertext)

    def decrypt(self, ciphertext: ExchangeCredentialCiphertext) -> ExchangeCredentialSecret:
        payload = self._post_json(
            path=f"/v1/transit/decrypt/{self.key_name}",
            payload={"ciphertext": ciphertext.value},
        )
        plaintext = _read_string(payload, "data", "plaintext")
        return ExchangeCredentialSecret.from_base64(plaintext)

    def fingerprint(self, secret: ExchangeCredentialSecret) -> ExchangeCredentialFingerprint:
        payload = self._post_json(
            path=f"/v1/transit/hmac/{self.key_name}/sha2-256",
            payload={"input": secret.as_base64()},
        )
        digest = _read_string(payload, "data", "hmac")
        return ExchangeCredentialFingerprint(value=digest)

    def _post_json(self, *, path: str, payload: dict[str, str]) -> dict[str, Any]:
        request = urllib.request.Request(
            url=f"{self.address.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Vault-Token": self.token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ExchangeSecretCipherError(
                f"transit request failed with status {exc.code}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ExchangeSecretCipherError("transit request failed") from exc


def _read_string(payload: dict[str, Any], *path: str) -> str:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise ExchangeSecretCipherError("transit response missing expected field")
        current = current[key]
    if not isinstance(current, str) or current == "":
        raise ExchangeSecretCipherError("transit response field is invalid")
    return current

__all__ = ["OpenBaoTransitExchangeSecretCipher"]
