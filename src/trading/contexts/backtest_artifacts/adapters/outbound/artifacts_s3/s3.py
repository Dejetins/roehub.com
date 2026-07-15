from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

from trading.contexts.backtest_artifacts.domain import ArtifactStoreError
from trading.integration import (
    MAX_ARTIFACT_BLOB_BYTES,
    ArtifactBlobDescriptor,
    ArtifactStoreDescriptor,
    sha256_digest,
)
from trading.platform.secrets import (
    OpenBaoSecretResolver,
    SecretKind,
    SecretReference,
    SecretReferenceError,
)

_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")


@dataclass(frozen=True, slots=True, repr=False)
class S3ResolvedCredentials:
    access_key_id: str
    secret_access_key: str

    def __post_init__(self) -> None:
        if not 3 <= len(self.access_key_id) <= 128 or not 8 <= len(self.secret_access_key) <= 256:
            raise ValueError("resolved S3 credentials are invalid")

    def __repr__(self) -> str:
        return "S3ResolvedCredentials(<redacted>)"


@dataclass(frozen=True, slots=True)
class S3ConnectionConfig:
    endpoint: str
    bucket: str
    region: str
    credentials_ref: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.path not in {"", "/"}
        ):
            raise ValueError("S3 endpoint must be an HTTP(S) origin")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("S3 endpoint must not contain credentials or selectors")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "minio"}:
            raise ValueError("plaintext S3 endpoint is restricted to controlled local fixtures")
        if _BUCKET_RE.fullmatch(self.bucket) is None:
            raise ValueError("S3 bucket is invalid")
        if not re.fullmatch(r"^[a-z0-9-]{1,32}$", self.region):
            raise ValueError("S3 region is invalid")
        try:
            SecretReference.parse(self.credentials_ref, expected_kind=SecretKind.STORAGE)
        except SecretReferenceError as error:
            raise ValueError("S3 credentials must use an OpenBao storage reference") from error


class S3CompatibleBlobStore:
    """Small path-style S3 adapter with AWS Signature Version 4."""

    def __init__(
        self,
        *,
        config: S3ConnectionConfig,
        credentials: S3ResolvedCredentials,
        materialization_root: Path,
        timeout_seconds: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not 0 < timeout_seconds <= 30:
            raise ValueError("S3 timeout must be in (0, 30] seconds")
        self._config = config
        self._credentials = credentials
        self._endpoint = config.endpoint.rstrip("/")
        self._materialization_root = materialization_root.expanduser().resolve()
        self._materialization_root.mkdir(parents=True, exist_ok=True, mode=0o750)
        self._client = httpx.Client(
            transport=transport,
            timeout=timeout_seconds,
            follow_redirects=False,
        )

    @property
    def descriptor(self) -> ArtifactStoreDescriptor:
        return ArtifactStoreDescriptor(schema="ArtifactStore/v1", backend="s3_compatible")

    def ensure_bucket(self) -> None:
        response = self._request("HEAD", key=None)
        if response.status_code == 200:
            return
        if response.status_code != 404:
            raise ArtifactStoreError(code="artifact.s3_unavailable")
        response = self._request("PUT", key=None, payload=b"")
        if response.status_code not in {200, 204, 409}:
            raise ArtifactStoreError(code="artifact.s3_unavailable")

    def close(self) -> None:
        self._client.close()

    def put_bytes(self, payload: bytes, *, media_type: str) -> ArtifactBlobDescriptor:
        if len(payload) > MAX_ARTIFACT_BLOB_BYTES:
            raise ArtifactStoreError(code="artifact.blob_too_large")
        digest = sha256_digest(payload)
        response = self._request(
            "PUT",
            key=self._key(digest),
            payload=payload,
            headers={"content-type": media_type, "x-amz-meta-roehub-digest": digest},
        )
        if response.status_code not in {200, 201, 204}:
            raise ArtifactStoreError(code="artifact.s3_write_failed")
        if self.read_bytes(digest=digest) != payload:
            raise ArtifactStoreError(code="artifact.s3_write_corrupted")
        return ArtifactBlobDescriptor(
            digest=digest,
            size_bytes=len(payload),
            media_type=media_type,
        )

    def read_bytes(self, *, digest: str) -> bytes:
        response = self._request(
            "GET",
            key=self._key(digest),
            max_response_bytes=MAX_ARTIFACT_BLOB_BYTES,
        )
        if response.status_code == 404:
            raise ArtifactStoreError(code="artifact.not_found")
        if response.status_code != 200:
            raise ArtifactStoreError(code="artifact.s3_unavailable")
        payload = response.content
        if sha256_digest(payload) != digest:
            raise ArtifactStoreError(code="artifact.digest_mismatch")
        return payload

    def exists(self, *, digest: str) -> bool:
        response = self._request("HEAD", key=self._key(digest))
        if response.status_code == 404:
            return False
        if response.status_code != 200:
            raise ArtifactStoreError(code="artifact.s3_unavailable")
        return True

    def materialize(self, *, digest: str, cache_key: str) -> Path:
        if (
            not cache_key
            or len(cache_key) > 512
            or any(character.isspace() for character in cache_key)
        ):
            raise ArtifactStoreError(code="artifact.materialization_key_invalid")
        payload = self.read_bytes(digest=digest)
        namespace = hashlib.sha256(cache_key.encode()).hexdigest()
        root = self._materialization_root / namespace
        root.mkdir(parents=True, exist_ok=True, mode=0o750)
        destination = root / digest.removeprefix("sha256:")
        if destination.exists():
            if sha256_digest(destination.read_bytes()) != digest:
                raise ArtifactStoreError(code="artifact.digest_mismatch")
            return destination
        temporary = root / f".{uuid4().hex}.tmp"
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fchmod(stream.fileno(), 0o440)
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
        if sha256_digest(destination.read_bytes()) != digest:
            raise ArtifactStoreError(code="artifact.digest_mismatch")
        return destination

    def delete(self, *, digest: str) -> None:
        response = self._request("DELETE", key=self._key(digest))
        if response.status_code not in {200, 202, 204, 404}:
            raise ArtifactStoreError(code="artifact.s3_unavailable")
        filename = digest.removeprefix("sha256:")
        for materialized in self._materialization_root.glob(f"*/{filename}"):
            materialized.unlink(missing_ok=True)
            try:
                materialized.parent.rmdir()
            except OSError:
                pass

    def _key(self, digest: str) -> str:
        match = _DIGEST_RE.fullmatch(digest)
        if match is None:
            raise ArtifactStoreError(code="artifact.digest_invalid")
        value = match.group(1)
        return f"blobs/sha256/{value[:2]}/{value}"

    def _request(
        self,
        method: str,
        *,
        key: str | None,
        payload: bytes = b"",
        headers: dict[str, str] | None = None,
        max_response_bytes: int | None = None,
    ) -> httpx.Response:
        now = datetime.now(UTC)
        path = "/" + quote(self._config.bucket, safe="")
        if key is not None:
            path += "/" + quote(key, safe="/")
        url = self._endpoint + path
        parsed = urlsplit(url)
        host = parsed.netloc
        payload_hash = hashlib.sha256(payload).hexdigest()
        request_headers = {
            "host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": now.strftime("%Y%m%dT%H%M%SZ"),
        }
        request_headers.update(
            {key.lower(): value.strip() for key, value in (headers or {}).items()}
        )
        canonical_headers = "".join(
            f"{name}:{' '.join(value.split())}\n" for name, value in sorted(request_headers.items())
        )
        signed_headers = ";".join(sorted(request_headers))
        canonical_request = "\n".join(
            (method, path, "", canonical_headers, signed_headers, payload_hash)
        )
        date = now.strftime("%Y%m%d")
        scope = f"{date}/{self._config.region}/s3/aws4_request"
        string_to_sign = "\n".join(
            (
                "AWS4-HMAC-SHA256",
                request_headers["x-amz-date"],
                scope,
                hashlib.sha256(canonical_request.encode()).hexdigest(),
            )
        )
        signing_key = self._signing_key(date)
        signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
        request_headers["authorization"] = (
            "AWS4-HMAC-SHA256 "
            f"Credential={self._credentials.access_key_id}/{scope},"
            f"SignedHeaders={signed_headers},Signature={signature}"
        )
        try:
            if max_response_bytes is None:
                return self._client.request(
                    method,
                    url,
                    content=payload if method in {"PUT", "POST"} else None,
                    headers=request_headers,
                )
            with self._client.stream(method, url, headers=request_headers) as response:
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > max_response_bytes:
                        raise ArtifactStoreError(code="artifact.blob_too_large")
                    chunks.append(chunk)
                return httpx.Response(
                    response.status_code,
                    headers=response.headers,
                    content=b"".join(chunks),
                    request=response.request,
                )
        except httpx.HTTPError as error:
            raise ArtifactStoreError(code="artifact.s3_unavailable") from error

    def _signing_key(self, date: str) -> bytes:
        date_key = hmac.new(
            ("AWS4" + self._credentials.secret_access_key).encode(),
            date.encode(),
            hashlib.sha256,
        ).digest()
        region_key = hmac.new(date_key, self._config.region.encode(), hashlib.sha256).digest()
        service_key = hmac.new(region_key, b"s3", hashlib.sha256).digest()
        return hmac.new(service_key, b"aws4_request", hashlib.sha256).digest()


def resolve_s3_credentials(
    *,
    config: S3ConnectionConfig,
    resolver: OpenBaoSecretResolver,
) -> S3ResolvedCredentials:
    value = resolver.resolve(
        config.credentials_ref,
        expected_kind=SecretKind.STORAGE,
    )
    try:
        payload = json.loads(value.reveal_text())
    except json.JSONDecodeError as error:
        raise ArtifactStoreError(code="artifact.s3_credentials_invalid") from error
    if not isinstance(payload, dict):
        raise ArtifactStoreError(code="artifact.s3_credentials_invalid")
    resolved = (payload.get("access_key_id"), payload.get("secret_access_key"))
    if not all(isinstance(item, str) for item in resolved):
        raise ArtifactStoreError(code="artifact.s3_credentials_invalid")
    try:
        return S3ResolvedCredentials(*cast(tuple[str, str], resolved))
    except (TypeError, ValueError) as error:
        raise ArtifactStoreError(code="artifact.s3_credentials_invalid") from error


__all__ = [
    "S3CompatibleBlobStore",
    "S3ConnectionConfig",
    "S3ResolvedCredentials",
    "resolve_s3_credentials",
]
