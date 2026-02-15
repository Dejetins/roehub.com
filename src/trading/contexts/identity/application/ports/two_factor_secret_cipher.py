from __future__ import annotations

from typing import Protocol


class TwoFactorSecretCipher(Protocol):
    """
    TwoFactorSecretCipher — порт envelope encryption для TOTP секрета.

    Docs:
      - docs/architecture/identity/identity-2fa-totp-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/use_cases/setup_two_factor_totp.py
      - src/trading/contexts/identity/application/use_cases/verify_two_factor_totp.py
      - src/trading/contexts/identity/adapters/outbound/security/two_factor/
        aes_gcm_envelope_secret_cipher.py
    """

    def encrypt_secret(self, *, secret: str) -> bytes:
        """
        Encrypt plaintext TOTP secret into opaque blob suitable for persistence.

        Args:
            secret: Base32 TOTP secret in plaintext form.
        Returns:
            bytes: Encrypted opaque secret blob.
        Assumptions:
            Plaintext secret is non-empty and must never be persisted as plaintext.
        Raises:
            ValueError: If encryption inputs are invalid or encryption fails.
        Side Effects:
            None.
        """
        ...

    def decrypt_secret(self, *, secret_enc: bytes) -> str:
        """
        Decrypt persisted secret blob for runtime TOTP verification only.

        Args:
            secret_enc: Opaque encrypted blob from persistence.
        Returns:
            str: Plaintext base32 TOTP secret.
        Assumptions:
            Decrypted value is kept in-memory only and never logged.
        Raises:
            ValueError: If blob format is invalid or decryption fails.
        Side Effects:
            None.
        """
        ...
