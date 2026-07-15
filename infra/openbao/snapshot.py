"""Encrypted OpenBao Raft snapshot backup and restore operations."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from trading.platform.secrets import OpenBaoUnavailableError, SecureTokenFile
from trading.platform.secrets.transport import (
    normalize_openbao_address,
    open_without_redirect,
)

SCHEMA = "io.roehub.openbao-snapshot-result/v1"
MAX_SNAPSHOT_BYTES = 1_073_741_824
MAX_ENCRYPTED_BYTES = MAX_SNAPSHOT_BYTES + 1_048_576
MAX_AUXILIARY_FILE_BYTES = 65_536


class SnapshotOperationError(RuntimeError):
    """Sanitized failure for backup or restore operations."""


def backup_snapshot(
    *,
    address: str,
    credential_path: Path,
    recipient_path: Path,
    destination: Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Fetch a Raft snapshot and atomically persist only its age-encrypted form."""

    _require_age()
    address = normalize_openbao_address(address)
    recipient = _read_regular_file(
        recipient_path,
        restricted=False,
        max_bytes=MAX_AUXILIARY_FILE_BYTES,
    )
    destination = _absolute_path(destination, label="backup destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if os.path.lexists(destination) or os.path.lexists(metadata_path):
        raise SnapshotOperationError("backup destination already exists")

    request = urllib.request.Request(
        url=f"{address.rstrip('/')}/v1/sys/storage/raft/snapshot",
        headers={
            "X-Vault-Token": SecureTokenFile(Path(credential_path)).read(),
            "Accept": "application/octet-stream",
        },
        method="GET",
    )
    try:
        with open_without_redirect(request, timeout=timeout_seconds) as response:
            snapshot = response.read(MAX_SNAPSHOT_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise SnapshotOperationError(
            f"OpenBao snapshot request failed with status {error.code}"
        ) from error
    except (OSError, TimeoutError) as error:
        raise SnapshotOperationError("OpenBao snapshot request is unavailable") from error
    if not snapshot or len(snapshot) > MAX_SNAPSHOT_BYTES:
        raise SnapshotOperationError("OpenBao snapshot size is invalid")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        with _temporary_restricted_file(recipient) as safe_recipient:
            with temporary.open("wb") as output:
                completed = subprocess.run(
                    ["age", "--encrypt", "--recipients-file", str(safe_recipient)],
                    input=snapshot,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=timeout_seconds,
                )
        if completed.returncode != 0 or temporary.stat().st_size <= 0:
            raise SnapshotOperationError("age encryption failed")
        os.replace(temporary, destination)
        os.chmod(destination, stat.S_IRUSR | stat.S_IWUSR)
    except (OSError, subprocess.SubprocessError) as error:
        raise SnapshotOperationError("encrypted backup write failed") from error
    finally:
        snapshot = b""
        temporary.unlink(missing_ok=True)

    ciphertext = _read_regular_file(
        destination,
        restricted=True,
        max_bytes=MAX_ENCRYPTED_BYTES,
    )
    result = {
        "schema": SCHEMA,
        "operation": "backup",
        "status": "passed",
        "encrypted": True,
        "ciphertext_bytes": len(ciphertext),
        "ciphertext_sha256": hashlib.sha256(ciphertext).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_metadata(metadata_path, result)
    return result


def restore_snapshot(
    *,
    address: str,
    credential_path: Path,
    recovery_path: Path,
    source: Path,
    force_new_storage: bool = False,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Decrypt one approved backup and restore; force requires an explicit fresh-volume flag."""

    _require_age()
    address = normalize_openbao_address(address)
    source = _absolute_path(source, label="encrypted backup")
    encrypted_snapshot = _read_regular_file(
        source,
        restricted=False,
        max_bytes=MAX_ENCRYPTED_BYTES,
    )
    _verify_backup_metadata(source=source, encrypted_snapshot=encrypted_snapshot)
    recovery_identity = _read_regular_file(
        recovery_path,
        restricted=True,
        max_bytes=MAX_AUXILIARY_FILE_BYTES,
    )
    try:
        with _temporary_restricted_file(recovery_identity) as safe_identity:
            completed = subprocess.run(
                ["age", "--decrypt", "--identity", str(safe_identity)],
                input=encrypted_snapshot,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=timeout_seconds,
            )
    except subprocess.SubprocessError as error:
        raise SnapshotOperationError("age decryption failed") from error
    snapshot = completed.stdout
    if completed.returncode != 0 or not snapshot or len(snapshot) > MAX_SNAPSHOT_BYTES:
        raise SnapshotOperationError("age decryption failed")

    credential = SecureTokenFile(Path(credential_path)).read()
    if force_new_storage:
        _verify_fresh_storage(
            address=address,
            credential=credential,
            timeout_seconds=timeout_seconds,
        )
    endpoint = "snapshot-force" if force_new_storage else "snapshot"
    request = urllib.request.Request(
        url=f"{address.rstrip('/')}/v1/sys/storage/raft/{endpoint}",
        data=snapshot,
        headers={
            "Content-Type": "application/octet-stream",
            "X-Vault-Token": credential,
        },
        method="POST",
    )
    try:
        with open_without_redirect(request, timeout=timeout_seconds) as response:
            status_code = int(response.status)
    except urllib.error.HTTPError as error:
        raise SnapshotOperationError(
            f"OpenBao snapshot restore failed with status {error.code}"
        ) from error
    except (OSError, TimeoutError) as error:
        raise SnapshotOperationError("OpenBao snapshot restore is unavailable") from error
    finally:
        snapshot = b""
    return {
        "schema": SCHEMA,
        "operation": "restore",
        "status": "passed",
        "encrypted_source": True,
        "source_digest_verified": True,
        "force_new_storage": force_new_storage,
        "fresh_storage_guard": "passed" if force_new_storage else "not_required",
        "http_status": status_code,
        "restored_at": datetime.now(timezone.utc).isoformat(),
    }


def _absolute_path(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise SnapshotOperationError(f"{label} path must be absolute")
    return candidate


def _read_regular_file(path: Path, *, restricted: bool, max_bytes: int) -> bytes:
    candidate = _absolute_path(path, label="required file")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise SnapshotOperationError("secure no-follow file access is unavailable")
    try:
        descriptor = os.open(candidate, os.O_RDONLY | os.O_CLOEXEC | no_follow)
    except OSError as error:
        raise SnapshotOperationError("required backup file is unavailable") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > max_bytes:
            raise SnapshotOperationError("required backup file is invalid")
        if restricted and info.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise SnapshotOperationError("restricted file permissions are unsafe")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise SnapshotOperationError("required backup file is invalid")
        return data
    except OSError as error:
        raise SnapshotOperationError("required backup file is unreadable") from error
    finally:
        os.close(descriptor)


@contextmanager
def _temporary_restricted_file(data: bytes) -> Iterator[Path]:
    fd, name = tempfile.mkstemp(prefix="roehub-openbao-material-")
    path = Path(name)
    try:
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        yield path
    finally:
        if fd >= 0:
            os.close(fd)
        path.unlink(missing_ok=True)


def _verify_backup_metadata(*, source: Path, encrypted_snapshot: bytes) -> None:
    metadata_path = source.with_suffix(source.suffix + ".metadata.json")
    raw_metadata = _read_regular_file(
        metadata_path,
        restricted=True,
        max_bytes=MAX_AUXILIARY_FILE_BYTES,
    )
    try:
        metadata = json.loads(raw_metadata)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SnapshotOperationError("backup metadata is invalid") from error
    expected_digest = hashlib.sha256(encrypted_snapshot).hexdigest()
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") != SCHEMA
        or metadata.get("operation") != "backup"
        or metadata.get("status") != "passed"
        or metadata.get("encrypted") is not True
        or metadata.get("ciphertext_bytes") != len(encrypted_snapshot)
        or not isinstance(metadata.get("ciphertext_sha256"), str)
        or not hmac.compare_digest(metadata["ciphertext_sha256"], expected_digest)
    ):
        raise SnapshotOperationError("backup metadata does not match ciphertext")


def _verify_fresh_storage(
    *,
    address: str,
    credential: str,
    timeout_seconds: float,
) -> None:
    mounts = _request_json(
        address=address,
        credential=credential,
        method="GET",
        path="/v1/sys/mounts",
        timeout_seconds=timeout_seconds,
    )
    auth = _request_json(
        address=address,
        credential=credential,
        method="GET",
        path="/v1/sys/auth",
        timeout_seconds=timeout_seconds,
    )
    policies = _request_json(
        address=address,
        credential=credential,
        method="LIST",
        path="/v1/sys/policies/acl",
        timeout_seconds=timeout_seconds,
    )
    mount_keys = set(_mapping_data(mounts))
    auth_keys = set(_mapping_data(auth))
    policy_keys = _mapping_data(policies).get("keys")
    if (
        not mount_keys.issubset({"cubbyhole/", "identity/", "sys/"})
        or auth_keys != {"token/"}
        or not isinstance(policy_keys, list)
        or not set(policy_keys).issubset({"default", "response-wrapping", "root"})
    ):
        raise SnapshotOperationError("force restore requires a fresh OpenBao storage")


def _request_json(
    *,
    address: str,
    credential: str,
    method: str,
    path: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=f"{address}{path}",
        headers={"Accept": "application/json", "X-Vault-Token": credential},
        method=method,
    )
    try:
        with open_without_redirect(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_AUXILIARY_FILE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise SnapshotOperationError(
            f"OpenBao freshness check failed with status {error.code}"
        ) from error
    except (OSError, TimeoutError) as error:
        raise SnapshotOperationError("OpenBao freshness check is unavailable") from error
    if len(raw) > MAX_AUXILIARY_FILE_BYTES:
        raise SnapshotOperationError("OpenBao freshness response is invalid")
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SnapshotOperationError("OpenBao freshness response is invalid") from error
    if not isinstance(payload, dict):
        raise SnapshotOperationError("OpenBao freshness response is invalid")
    return payload


def _mapping_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        raise SnapshotOperationError("OpenBao freshness response is invalid")
    return data


def _require_age() -> None:
    if shutil.which("age") is None:
        raise SnapshotOperationError("age executable is required")


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    if os.path.lexists(path):
        raise SnapshotOperationError("backup metadata destination already exists")
    data = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, stat.S_IRUSR | stat.S_IWUSR)
        temporary.write_bytes(data)
        os.replace(temporary, path)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    backup = subparsers.add_parser("backup")
    backup.add_argument("--address", required=True)
    backup.add_argument("--credential-path", type=Path, required=True)
    backup.add_argument("--recipient-path", type=Path, required=True)
    backup.add_argument("--destination", type=Path, required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--address", required=True)
    restore.add_argument("--credential-path", type=Path, required=True)
    restore.add_argument("--recovery-path", type=Path, required=True)
    restore.add_argument("--source", type=Path, required=True)
    restore.add_argument("--force-new-storage", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "backup":
            result = backup_snapshot(
                address=args.address,
                credential_path=args.credential_path,
                recipient_path=args.recipient_path,
                destination=args.destination,
            )
        else:
            result = restore_snapshot(
                address=args.address,
                credential_path=args.credential_path,
                recovery_path=args.recovery_path,
                source=args.source,
                force_new_storage=args.force_new_storage,
            )
    except (SnapshotOperationError, OpenBaoUnavailableError) as error:
        print(
            json.dumps(
                {"schema": SCHEMA, "status": "failed", "reason": str(error)},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
