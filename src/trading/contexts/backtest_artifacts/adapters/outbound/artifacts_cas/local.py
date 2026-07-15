from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
from pathlib import Path
from threading import RLock
from uuid import uuid4

from trading.contexts.backtest_artifacts.domain import ArtifactStoreError
from trading.integration import (
    MAX_ARTIFACT_BLOB_BYTES,
    ArtifactBlobDescriptor,
    ArtifactStoreDescriptor,
    sha256_digest,
)

_DIGEST_RE = re.compile(r"^sha256:([0-9a-f]{64})$")


class LocalCasBlobStore:
    """Immutable local CAS backed by a host-mounted durable directory."""

    def __init__(self, *, root: Path, materialization_root: Path | None = None) -> None:
        self._root = root.expanduser().resolve()
        self._blob_root = self._root / "blobs" / "sha256"
        self._incoming_root = self._root / ".incoming"
        self._materialization_root = (
            materialization_root.expanduser().resolve()
            if materialization_root is not None
            else self._root / "materialized"
        )
        for directory in (self._blob_root, self._incoming_root, self._materialization_root):
            directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        self._verified_fingerprints: dict[str, set[tuple[int, int, int, int, int]]] = {}
        self._verification_lock = RLock()

    @property
    def descriptor(self) -> ArtifactStoreDescriptor:
        return ArtifactStoreDescriptor(schema="ArtifactStore/v1", backend="local_cas")

    def put_bytes(self, payload: bytes, *, media_type: str) -> ArtifactBlobDescriptor:
        if len(payload) > MAX_ARTIFACT_BLOB_BYTES:
            raise ArtifactStoreError(code="artifact.blob_too_large")
        digest = sha256_digest(payload)
        target = self._blob_path(digest)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
        temporary = self._incoming_root / f"{uuid4().hex}.blob"
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o640,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(payload)
                stream.flush()
                os.fchmod(stream.fileno(), 0o440)
                os.fsync(stream.fileno())
            if sha256_digest(temporary.read_bytes()) != digest:
                raise ArtifactStoreError(code="artifact.write_corrupted")
            try:
                os.link(temporary, target, follow_symlinks=False)
                self._fsync_directory(target.parent)
            except FileExistsError:
                self._verify_path(target, digest=digest)
        finally:
            temporary.unlink(missing_ok=True)
        self._verify_path(target, digest=digest)
        return ArtifactBlobDescriptor(
            digest=digest,
            size_bytes=len(payload),
            media_type=media_type,
        )

    def read_bytes(self, *, digest: str) -> bytes:
        path = self._blob_path(digest)
        return self._verify_path(path, digest=digest)

    def exists(self, *, digest: str) -> bool:
        path = self._blob_path(digest)
        try:
            self._verify_integrity(path, digest=digest)
        except ArtifactStoreError as error:
            if error.code == "artifact.not_found":
                return False
            raise
        return True

    def materialize(self, *, digest: str, cache_key: str) -> Path:
        if (
            not cache_key
            or len(cache_key) > 512
            or any(character.isspace() for character in cache_key)
        ):
            raise ArtifactStoreError(code="artifact.materialization_key_invalid")
        source = self._blob_path(digest)
        self._verify_integrity(source, digest=digest)
        namespace = hashlib.sha256(cache_key.encode()).hexdigest()
        destination_root = self._materialization_root / namespace
        destination_root.mkdir(parents=True, exist_ok=True, mode=0o750)
        destination = destination_root / digest.removeprefix("sha256:")
        if destination.exists():
            self._verify_integrity(destination, digest=digest)
            return destination
        temporary = destination_root / f".{uuid4().hex}.tmp"
        hardlinked = False
        try:
            try:
                os.link(source, temporary, follow_symlinks=False)
                hardlinked = True
            except OSError:
                with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                    output_stream.flush()
                    os.fchmod(output_stream.fileno(), 0o440)
                    os.fsync(output_stream.fileno())
            try:
                os.link(temporary, destination, follow_symlinks=False)
                self._fsync_directory(destination_root)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        if hardlinked:
            self._remember_verified(destination, digest=digest)
        else:
            self._verify_integrity(destination, digest=digest)
        return destination

    def delete(self, *, digest: str) -> None:
        path = self._blob_path(digest)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        else:
            self._fsync_directory(path.parent)
        filename = digest.removeprefix("sha256:")
        for materialized in self._materialization_root.glob(f"*/{filename}"):
            materialized.unlink(missing_ok=True)
            try:
                materialized.parent.rmdir()
            except OSError:
                pass
        with self._verification_lock:
            self._verified_fingerprints.pop(digest, None)

    def _blob_path(self, digest: str) -> Path:
        match = _DIGEST_RE.fullmatch(digest)
        if match is None:
            raise ArtifactStoreError(code="artifact.digest_invalid")
        value = match.group(1)
        return self._blob_root / value[:2] / value

    def _verify_path(self, path: Path, *, digest: str) -> bytes:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError as error:
            raise ArtifactStoreError(code="artifact.not_found") from error
        except OSError as error:
            raise ArtifactStoreError(code="artifact.unsafe_path") from error
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            file_stat = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_nlink < 1
                or file_stat.st_size > MAX_ARTIFACT_BLOB_BYTES
            ):
                raise ArtifactStoreError(code="artifact.unsafe_path")
            hasher = hashlib.sha256()
            chunks: list[bytes] = []
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
                chunks.append(chunk)
        if "sha256:" + hasher.hexdigest() != digest:
            raise ArtifactStoreError(code="artifact.digest_mismatch")
        self._remember_fingerprint(digest=digest, file_stat=file_stat)
        return b"".join(chunks)

    def _verify_integrity(self, path: Path, *, digest: str) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError as error:
            raise ArtifactStoreError(code="artifact.not_found") from error
        except OSError as error:
            raise ArtifactStoreError(code="artifact.unsafe_path") from error
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            file_stat = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(file_stat.st_mode)
                or file_stat.st_nlink < 1
                or file_stat.st_size > MAX_ARTIFACT_BLOB_BYTES
            ):
                raise ArtifactStoreError(code="artifact.unsafe_path")
            fingerprint = self._fingerprint(file_stat)
            with self._verification_lock:
                if fingerprint in self._verified_fingerprints.get(digest, set()):
                    return
            hasher = hashlib.sha256()
            while chunk := stream.read(1024 * 1024):
                hasher.update(chunk)
        if "sha256:" + hasher.hexdigest() != digest:
            raise ArtifactStoreError(code="artifact.digest_mismatch")
        self._remember_fingerprint(digest=digest, file_stat=file_stat)

    def _remember_verified(self, path: Path, *, digest: str) -> None:
        try:
            file_stat = path.stat(follow_symlinks=False)
        except OSError as error:
            raise ArtifactStoreError(code="artifact.unsafe_path") from error
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > MAX_ARTIFACT_BLOB_BYTES:
            raise ArtifactStoreError(code="artifact.unsafe_path")
        self._remember_fingerprint(digest=digest, file_stat=file_stat)

    def _remember_fingerprint(self, *, digest: str, file_stat: os.stat_result) -> None:
        with self._verification_lock:
            self._verified_fingerprints.setdefault(digest, set()).add(self._fingerprint(file_stat))

    @staticmethod
    def _fingerprint(file_stat: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["LocalCasBlobStore"]
