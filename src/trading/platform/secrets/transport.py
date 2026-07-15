"""Credential-safe HTTP primitives for the embedded OpenBao boundary."""

from __future__ import annotations

import ipaddress
import urllib.request
from http.client import HTTPResponse
from typing import IO, cast
from urllib.parse import urlsplit


class _RejectRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        _ = req, fp, code, msg, headers, newurl
        return None


def normalize_openbao_address(address: str) -> str:
    """Validate an OpenBao origin and return its canonical base URL."""

    normalized = address.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OpenBao address must be an HTTP(S) origin")
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("OpenBao address must not contain credentials or selectors")
    if parsed.scheme == "http" and not _is_trusted_plaintext_host(parsed.hostname):
        raise ValueError("plaintext OpenBao is restricted to loopback or embedded DNS")
    return normalized


def open_without_redirect(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> HTTPResponse:
    """Open exactly one HTTP origin; credential-bearing redirects are forbidden."""

    opener = urllib.request.build_opener(_RejectRedirectHandler())
    return cast(HTTPResponse, opener.open(request, timeout=timeout))


def _is_trusted_plaintext_host(hostname: str) -> bool:
    lowered = hostname.lower()
    if lowered in {"localhost", "openbao"}:
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


__all__ = ["normalize_openbao_address", "open_without_redirect"]
