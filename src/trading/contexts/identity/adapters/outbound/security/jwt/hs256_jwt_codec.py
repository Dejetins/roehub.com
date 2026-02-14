from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any

from trading.contexts.identity.application.ports.clock import IdentityClock
from trading.contexts.identity.application.ports.jwt_codec import (
    IdentityJwtClaims,
    JwtCodec,
    JwtDecodeError,
)
from trading.shared_kernel.primitives import PaidLevel, UserId

_HEADER: dict[str, str] = {
    "alg": "HS256",
    "typ": "JWT",
}


class Hs256JwtCodec(JwtCodec):
    """
    Hs256JwtCodec — deterministic HS256 JWT codec for identity cookie tokens.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
    Related:
      - src/trading/contexts/identity/application/ports/jwt_codec.py
      - src/trading/contexts/identity/application/use_cases/telegram_login.py
      - src/trading/contexts/identity/adapters/outbound/security/current_user/
        jwt_cookie_current_user.py
    """

    def __init__(
        self,
        *,
        secret_key: str,
        clock: IdentityClock,
        leeway_seconds: int = 0,
    ) -> None:
        """
        Initialize HS256 codec with signing key and runtime clock.

        Args:
            secret_key: JWT signing key.
            clock: Runtime clock for expiration checks.
            leeway_seconds: Optional expiration leeway.
        Returns:
            None.
        Assumptions:
            Secret key is stable per deployment environment.
        Raises:
            ValueError: If secret key is empty, clock missing, or leeway negative.
        Side Effects:
            None.
        """
        normalized_secret = secret_key.strip()
        if not normalized_secret:
            raise ValueError("Hs256JwtCodec requires non-empty secret_key")
        if clock is None:  # type: ignore[truthy-bool]
            raise ValueError("Hs256JwtCodec requires clock")
        if leeway_seconds < 0:
            raise ValueError("Hs256JwtCodec requires leeway_seconds >= 0")

        self._secret_key = normalized_secret.encode("utf-8")
        self._clock = clock
        self._leeway_seconds = leeway_seconds

    def encode(self, *, claims: IdentityJwtClaims) -> str:
        """
        Encode typed claims into deterministic compact HS256 JWT string.

        Args:
            claims: Identity JWT claims.
        Returns:
            str: Signed compact token.
        Assumptions:
            Claims already validated by `IdentityJwtClaims` invariants.
        Raises:
            ValueError: If claim serialization fails.
        Side Effects:
            None.
        """
        header_segment = _to_b64url_json(payload=_HEADER)
        payload_segment = _to_b64url_json(
            payload={
                "exp": int(claims.expires_at.timestamp()),
                "iat": int(claims.issued_at.timestamp()),
                "paid_level": str(claims.paid_level),
                "sub": str(claims.user_id),
            }
        )

        signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
        signature = hmac.new(self._secret_key, signing_input, hashlib.sha256).digest()
        signature_segment = _to_b64url_bytes(raw=signature)
        return f"{header_segment}.{payload_segment}.{signature_segment}"

    def decode(self, *, token: str) -> IdentityJwtClaims:
        """
        Verify JWT signature and temporal claims and return typed claims.

        Args:
            token: Compact JWT token.
        Returns:
            IdentityJwtClaims: Verified claims.
        Assumptions:
            Token uses HS256 and includes `sub`, `paid_level`, `iat`, and `exp`.
        Raises:
            JwtDecodeError: If token format/signature/claims are invalid.
        Side Effects:
            None.
        """
        token_value = token.strip()
        if not token_value:
            raise JwtDecodeError(code="missing_token", message="JWT token is empty")

        segments = token_value.split(".")
        if len(segments) != 3:
            raise JwtDecodeError(
                code="invalid_token_format",
                message="JWT token must contain 3 dot-separated segments",
            )

        header_segment, payload_segment, signature_segment = segments
        header = _from_b64url_json(segment=header_segment)
        if header.get("alg") != "HS256" or header.get("typ") != "JWT":
            raise JwtDecodeError(
                code="invalid_header",
                message="JWT header must contain alg=HS256 and typ=JWT",
            )

        signing_input = f"{header_segment}.{payload_segment}".encode("utf-8")
        expected_signature = hmac.new(self._secret_key, signing_input, hashlib.sha256).digest()
        provided_signature = _from_b64url_bytes(segment=signature_segment)
        if not hmac.compare_digest(expected_signature, provided_signature):
            raise JwtDecodeError(
                code="invalid_signature",
                message="JWT signature verification failed",
            )

        payload = _from_b64url_json(segment=payload_segment)
        subject = str(payload.get("sub", "")).strip()
        level = str(payload.get("paid_level", "")).strip()
        iat_raw = payload.get("iat")
        exp_raw = payload.get("exp")
        if not subject or not level or iat_raw is None or exp_raw is None:
            raise JwtDecodeError(
                code="invalid_claims",
                message="JWT payload must contain sub, paid_level, iat, and exp",
            )

        try:
            issued_at = datetime.fromtimestamp(int(iat_raw), tz=timezone.utc)
            expires_at = datetime.fromtimestamp(int(exp_raw), tz=timezone.utc)
            user_id = UserId.from_string(subject)
            paid_level = PaidLevel(level)
        except (OSError, OverflowError, TypeError, ValueError) as error:
            raise JwtDecodeError(
                code="invalid_claims",
                message="JWT payload claims are malformed",
            ) from error

        now = self._clock.now()
        now_offset = now.utcoffset()
        if now.tzinfo is None or now_offset is None:
            raise ValueError("Hs256JwtCodec clock must return timezone-aware UTC datetime")
        if now_offset.total_seconds() != 0:
            raise ValueError("Hs256JwtCodec clock must return timezone-aware UTC datetime")
        now_ts = int(now.timestamp())
        if int(exp_raw) <= now_ts - self._leeway_seconds:
            raise JwtDecodeError(code="expired_token", message="JWT token is expired")

        return IdentityJwtClaims(
            user_id=user_id,
            paid_level=paid_level,
            issued_at=issued_at,
            expires_at=expires_at,
        )



def _to_b64url_json(*, payload: dict[str, Any]) -> str:
    """
    Serialize mapping into deterministic base64url-encoded JSON segment.

    Args:
        payload: JSON-serializable mapping.
    Returns:
        str: Base64url string without padding.
    Assumptions:
        Deterministic serialization requires sorted keys and compact separators.
    Raises:
        ValueError: If payload cannot be JSON-serialized.
    Side Effects:
        None.
    """
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return _to_b64url_bytes(raw=raw)



def _from_b64url_json(*, segment: str) -> dict[str, Any]:
    """
    Decode base64url JSON segment into mapping.

    Args:
        segment: Base64url token segment.
    Returns:
        dict[str, Any]: Decoded JSON object.
    Assumptions:
        Segment contains JSON object representation.
    Raises:
        JwtDecodeError: If segment cannot be decoded into JSON object.
    Side Effects:
        None.
    """
    raw = _from_b64url_bytes(segment=segment)
    try:
        loaded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise JwtDecodeError(
            code="invalid_token_format",
            message="JWT segment is not valid JSON",
        ) from error
    if not isinstance(loaded, dict):
        raise JwtDecodeError(
            code="invalid_token_format",
            message="JWT JSON segment must be an object",
        )
    return loaded



def _to_b64url_bytes(*, raw: bytes) -> str:
    """
    Encode raw bytes as base64url string without padding.

    Args:
        raw: Binary payload.
    Returns:
        str: Base64url string without trailing `=` characters.
    Assumptions:
        Input bytes are finite and memory-safe for in-memory encoding.
    Raises:
        None.
    Side Effects:
        None.
    """
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")



def _from_b64url_bytes(*, segment: str) -> bytes:
    """
    Decode base64url segment with optional missing padding.

    Args:
        segment: Base64url-encoded string.
    Returns:
        bytes: Decoded binary payload.
    Assumptions:
        Segment uses URL-safe alphabet.
    Raises:
        JwtDecodeError: If segment is not valid base64url.
    Side Effects:
        None.
    """
    padding = "=" * ((4 - len(segment) % 4) % 4)
    candidate = f"{segment}{padding}".encode("ascii")
    try:
        return base64.urlsafe_b64decode(candidate)
    except (ValueError, UnicodeEncodeError) as error:
        raise JwtDecodeError(
            code="invalid_token_format",
            message="JWT segment is not valid base64url",
        ) from error
