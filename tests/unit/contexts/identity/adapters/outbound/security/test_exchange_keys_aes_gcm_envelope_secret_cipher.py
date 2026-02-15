from __future__ import annotations

import pytest

from trading.contexts.identity.adapters.outbound.security.exchange_keys import (
    AesGcmEnvelopeExchangeKeysSecretCipher,
)


def test_exchange_keys_cipher_encrypt_decrypt_roundtrip() -> None:
    """
    Verify exchange keys envelope cipher preserves plaintext on encrypt/decrypt roundtrip.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Cipher blob format is versioned and compatible for immediate decryption.
    Raises:
        AssertionError: If decrypted value differs from original plaintext.
    Side Effects:
        None.
    """
    cipher = AesGcmEnvelopeExchangeKeysSecretCipher(
        kek_b64="cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    )

    encrypted = cipher.encrypt_secret(secret="super-secret-value")
    decrypted = cipher.decrypt_secret(secret_enc=encrypted)

    assert decrypted == "super-secret-value"


def test_exchange_keys_cipher_rejects_tampered_blob_with_deterministic_error() -> None:
    """
    Verify tampered encrypted blob fails authentication with deterministic error message.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        AES-GCM authentication tag mismatch raises deterministic ValueError.
    Raises:
        AssertionError: If tampered blob is unexpectedly accepted.
    Side Effects:
        None.
    """
    cipher = AesGcmEnvelopeExchangeKeysSecretCipher(
        kek_b64="cm9laHViLWRldi1leGNoYW5nZS1rZXkta2VrLTAwMDE=",
    )
    encrypted = bytearray(cipher.encrypt_secret(secret="tamper-me"))
    encrypted[-1] ^= 0xFF

    with pytest.raises(ValueError, match="Encrypted exchange key blob authentication failed"):
        cipher.decrypt_secret(secret_enc=bytes(encrypted))


def test_exchange_keys_cipher_requires_valid_kek_base64() -> None:
    """
    Verify cipher constructor rejects malformed base64 KEK values.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Runtime KEK must be valid base64 in all environments.
    Raises:
        AssertionError: If malformed KEK input is accepted.
    Side Effects:
        None.
    """
    with pytest.raises(ValueError, match="IDENTITY_EXCHANGE_KEYS_KEK_B64 must be valid base64"):
        AesGcmEnvelopeExchangeKeysSecretCipher(kek_b64="not-base64")
