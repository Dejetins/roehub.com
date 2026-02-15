from __future__ import annotations

from typing import Protocol


class ExchangeKeysSecretCipher(Protocol):
    """
    ExchangeKeysSecretCipher — envelope encryption port for exchange keys secrets.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v2.md
    Related:
      - src/trading/contexts/identity/application/use_cases/create_exchange_key.py
      - src/trading/contexts/identity/adapters/outbound/security/exchange_keys/
        aes_gcm_envelope_secret_cipher.py
      - migrations/postgres/0004_identity_exchange_keys_v2.sql
    """

    def encrypt_secret(self, *, secret: str, aad: str) -> bytes:
        """
        Encrypt plaintext secret into versioned opaque blob for persistence.

        Args:
            secret: Plaintext secret (`api_secret` or `passphrase`).
            aad: Deterministic additional authenticated data binding payload.
        Returns:
            bytes: Encrypted opaque blob.
        Assumptions:
            Plaintext is kept in-memory only and never logged. `aad` is non-secret.
        Raises:
            ValueError: If inputs are invalid or encryption fails.
        Side Effects:
            None.
        """
        ...

    def decrypt_secret(self, *, secret_enc: bytes, aad: str) -> str:
        """
        Decrypt persisted blob back into plaintext secret.

        Args:
            secret_enc: Encrypted opaque blob from persistence.
            aad: Deterministic additional authenticated data binding payload.
        Returns:
            str: Decrypted plaintext secret.
        Assumptions:
            Decrypted value is used transiently and never exposed in API responses.
        Raises:
            ValueError: If blob format is invalid, input is invalid, or authentication fails.
        Side Effects:
            None.
        """
        ...
