from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trading.contexts.exchange_control.application.secret_cipher import (
    TRANSIT_KEY_NAME,
    ExchangeCredentialCiphertext,
    ExchangeCredentialFingerprint,
    ExchangeCredentialSecret,
    ExchangeSecretCipherError,
)
from trading.platform.secrets import OpenBaoUnavailableError, SecureTokenFile
from trading.platform.secrets.transport import (
    normalize_openbao_address,
    open_without_redirect,
)


@dataclass(frozen=True)
class OpenBaoTransitExchangeSecretCipher:
    address: str
    credential_source: SecureTokenFile | None = None
    token: str | None = field(default=None, repr=False)
    key_name: str = TRANSIT_KEY_NAME
    timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if not self.address:
            raise ValueError("OPENBAO_ADDR is required")
        object.__setattr__(self, "address", normalize_openbao_address(self.address))
        if (self.credential_source is None) == (self.token is None):
            raise ValueError("exactly one OpenBao service credential source is required")
        if self.token is not None:
            SecureTokenFile(Path(self.token))
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
                "X-Vault-Token": self._read_credential(),
            },
            method="POST",
        )
        try:
            with open_without_redirect(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ExchangeSecretCipherError(
                f"transit request failed with status {exc.code}"
            ) from exc
        except (OSError, json.JSONDecodeError, OpenBaoUnavailableError) as exc:
            raise ExchangeSecretCipherError("transit request failed") from exc

    def _read_credential(self) -> str:
        if self.credential_source is not None:
            return self.credential_source.read()
        if self.token is None:
            raise OpenBaoUnavailableError("OpenBao service credential is unavailable")
        return SecureTokenFile(Path(self.token)).read()


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
