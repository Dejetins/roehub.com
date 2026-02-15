from __future__ import annotations

import base64
import binascii
import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from trading.contexts.identity.application.ports.exchange_keys_secret_cipher import (
    ExchangeKeysSecretCipher,
)

_BLOB_VERSION_V1 = 1
_NONCE_LENGTH = 12
_HEADER_STRUCT = struct.Struct(">BBBH")
_ENVELOPE_AAD = b"roehub.identity.exchange_keys.v1"
_SUPPORTED_KEK_LENGTHS = {16, 24, 32}


class AesGcmEnvelopeExchangeKeysSecretCipher(ExchangeKeysSecretCipher):
    """
    AesGcmEnvelopeExchangeKeysSecretCipher — AES-GCM envelope cipher for exchange secrets.

    Docs:
      - docs/architecture/identity/identity-exchange-keys-storage-2fa-gate-policy-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/exchange_keys_secret_cipher.py
      - apps/api/wiring/modules/identity.py
      - migrations/postgres/0003_identity_exchange_keys_v1.sql
    """

    def __init__(self, *, kek_b64: str) -> None:
        """
        Initialize envelope cipher from base64 KEK (`IDENTITY_EXCHANGE_KEYS_KEK_B64`).

        Args:
            kek_b64: Base64-encoded KEK bytes.
        Returns:
            None.
        Assumptions:
            KEK length must be valid AES key size (16/24/32 bytes).
        Raises:
            ValueError: If KEK is blank, malformed, or unsupported length.
        Side Effects:
            None.
        """
        normalized_kek_b64 = kek_b64.strip()
        if not normalized_kek_b64:
            raise ValueError("AesGcmEnvelopeExchangeKeysSecretCipher requires non-empty kek_b64")
        try:
            kek_bytes = base64.b64decode(normalized_kek_b64, validate=True)
        except binascii.Error as error:
            raise ValueError("IDENTITY_EXCHANGE_KEYS_KEK_B64 must be valid base64") from error
        if len(kek_bytes) not in _SUPPORTED_KEK_LENGTHS:
            raise ValueError(
                "IDENTITY_EXCHANGE_KEYS_KEK_B64 must decode to 16, 24, or 32 bytes for AES-GCM"
            )
        self._kek = kek_bytes

    def encrypt_secret(self, *, secret: str) -> bytes:
        """
        Encrypt plaintext secret using DEK+KEK envelope AES-GCM format.

        Args:
            secret: Plaintext secret value.
        Returns:
            bytes: Versioned encrypted opaque blob.
        Assumptions:
            Secret is non-empty and never logged.
        Raises:
            ValueError: If secret is empty.
        Side Effects:
            Uses OS CSPRNG for DEK and nonces.
        """
        normalized_secret = secret.strip()
        if not normalized_secret:
            raise ValueError("AesGcmEnvelopeExchangeKeysSecretCipher secret must be non-empty")

        plaintext = normalized_secret.encode("utf-8")
        dek = os.urandom(32)

        dek_nonce = os.urandom(_NONCE_LENGTH)
        secret_nonce = os.urandom(_NONCE_LENGTH)
        encrypted_dek = AESGCM(self._kek).encrypt(dek_nonce, dek, _ENVELOPE_AAD)
        encrypted_secret = AESGCM(dek).encrypt(secret_nonce, plaintext, _ENVELOPE_AAD)

        header = _HEADER_STRUCT.pack(
            _BLOB_VERSION_V1,
            len(dek_nonce),
            len(secret_nonce),
            len(encrypted_dek),
        )
        return b"".join((header, dek_nonce, encrypted_dek, secret_nonce, encrypted_secret))

    def decrypt_secret(self, *, secret_enc: bytes) -> str:
        """
        Decrypt versioned envelope blob and return plaintext secret.

        Args:
            secret_enc: Encrypted secret blob from storage.
        Returns:
            str: Plaintext secret value.
        Assumptions:
            Decryption is used only transiently and never exposed to API.
        Raises:
            ValueError: If blob format is invalid or authentication fails.
        Side Effects:
            None.
        """
        blob = bytes(secret_enc)
        if not blob:
            raise ValueError("AesGcmEnvelopeExchangeKeysSecretCipher secret_enc must be non-empty")

        version, dek_nonce_len, secret_nonce_len, encrypted_dek_len, payload = _parse_header(
            blob=blob
        )
        if version != _BLOB_VERSION_V1:
            raise ValueError("Unsupported encrypted exchange key blob version")
        if dek_nonce_len != _NONCE_LENGTH:
            raise ValueError("Encrypted exchange key blob contains invalid DEK nonce length")
        if secret_nonce_len != _NONCE_LENGTH:
            raise ValueError("Encrypted exchange key blob contains invalid secret nonce length")
        if encrypted_dek_len <= 16:
            raise ValueError("Encrypted exchange key blob contains invalid encrypted DEK length")

        dek_nonce = payload[:dek_nonce_len]
        encrypted_dek_start = dek_nonce_len
        encrypted_dek_end = encrypted_dek_start + encrypted_dek_len
        encrypted_dek = payload[encrypted_dek_start:encrypted_dek_end]
        secret_nonce_start = encrypted_dek_end
        secret_nonce_end = secret_nonce_start + secret_nonce_len
        secret_nonce = payload[secret_nonce_start:secret_nonce_end]
        encrypted_secret = payload[secret_nonce_end:]

        if len(encrypted_secret) <= 16:
            raise ValueError("Encrypted exchange key blob contains invalid encrypted payload")

        try:
            dek = AESGCM(self._kek).decrypt(dek_nonce, encrypted_dek, _ENVELOPE_AAD)
            plaintext = AESGCM(dek).decrypt(secret_nonce, encrypted_secret, _ENVELOPE_AAD)
        except InvalidTag as error:
            raise ValueError("Encrypted exchange key blob authentication failed") from error

        try:
            secret = plaintext.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("Encrypted exchange key plaintext is not valid UTF-8") from error
        if not secret:
            raise ValueError("Encrypted exchange key plaintext is empty")
        return secret



def _parse_header(*, blob: bytes) -> tuple[int, int, int, int, bytes]:
    """
    Parse envelope blob header and return metadata plus payload bytes.

    Args:
        blob: Complete encrypted blob.
    Returns:
        tuple[int, int, int, int, bytes]: `(version, dek_nonce_len, secret_nonce_len,
            encrypted_dek_len, payload)` tuple.
    Assumptions:
        Header follows deterministic `_HEADER_STRUCT` binary layout.
    Raises:
        ValueError: If blob is too short or payload is truncated.
    Side Effects:
        None.
    """
    if len(blob) < _HEADER_STRUCT.size:
        raise ValueError("Encrypted exchange key blob is too short")
    version, dek_nonce_len, secret_nonce_len, encrypted_dek_len = _HEADER_STRUCT.unpack_from(blob)
    payload = blob[_HEADER_STRUCT.size :]
    minimum_payload = dek_nonce_len + encrypted_dek_len + secret_nonce_len + 17
    if len(payload) < minimum_payload:
        raise ValueError("Encrypted exchange key blob payload is truncated")
    return version, dek_nonce_len, secret_nonce_len, encrypted_dek_len, payload
