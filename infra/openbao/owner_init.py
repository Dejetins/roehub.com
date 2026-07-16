"""Owner-operated OpenBao bootstrap without custody-material disclosure.

The module deliberately does not generate or decrypt PGP material.  It accepts
three owner-provided *public* PGP recipients, asks OpenBao to encrypt the
initial shares and administrator credential, and writes only the ciphertext to
an owner-controlled directory.  A second command can provision least-privilege
AppRoles after the owner has unsealed OpenBao locally.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from trading.platform.secrets import SecureTokenFile
from trading.platform.secrets.transport import normalize_openbao_address, open_without_redirect

ROOT = Path(__file__).resolve().parents[2]
POLICIES = ROOT / "infra" / "openbao" / "policies"
SCHEMA = "io.roehub.openbao-owner-init/v1alpha1"
_MAX_RECIPIENT_BYTES = 1024 * 1024
_INITIAL_FILES = frozenset(
    {
        "initial-admin.pgp",
        "owner-init.json",
        "unseal-share-1.pgp",
        "unseal-share-2.pgp",
        "unseal-share-3.pgp",
    }
)


class OwnerInitError(RuntimeError):
    """A secret-safe owner bootstrap failure."""


@dataclass(frozen=True, slots=True)
class _Client:
    address: str
    credential: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "address", normalize_openbao_address(self.address))

    def request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, Any] | None = None,
        *,
        extra_headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.credential is not None:
            headers["X-Vault-Token"] = self.credential
        if extra_headers is not None:
            headers.update(extra_headers)
        data = None
        if payload is not None:
            data = json.dumps(payload, sort_keys=True).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.address}{path}", data=data, headers=headers, method=method
        )
        try:
            with open_without_redirect(request, timeout=10) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            raise OwnerInitError(
                f"OpenBao request returned HTTP {int(error.code)} for {method} {path}"
            ) from error
        except (OSError, TimeoutError) as error:
            raise OwnerInitError("OpenBao request is unavailable") from error
        if not body:
            return {}
        try:
            decoded = json.loads(body)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise OwnerInitError("OpenBao returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise OwnerInitError("OpenBao returned an invalid payload")
        return decoded


@dataclass(frozen=True, slots=True)
class OwnerInitResult:
    status: str
    delivery_dir: Path
    recipient_count: int
    service_count: int = 0

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": SCHEMA,
            "status": self.status,
            "delivery_dir": str(self.delivery_dir),
            "recipient_count": self.recipient_count,
        }
        if self.status in {"initialized", "already_initialized"}:
            payload.update(
                {
                    "unseal_shares": 3,
                    "unseal_threshold": 2,
                    "initial_admin_encrypted": True,
                }
            )
        if self.service_count:
            payload["service_count"] = self.service_count
            payload["response_wrapped"] = True
            payload["initial_admin_revoked"] = True
        return payload


@dataclass(frozen=True, slots=True)
class _ServiceRole:
    delivery_name: str
    role_name: str
    policy_name: str
    policy_file: str


_SERVICE_ROLES = (
    _ServiceRole("api", "roehub-api", "roehub-api", "roehub-api.hcl"),
    _ServiceRole(
        "backup-recovery",
        "roehub-backup-recovery",
        "roehub-backup-recovery",
        "roehub-backup-recovery.hcl",
    ),
    _ServiceRole(
        "exchange-execution",
        "roehub-exchange-execution",
        "roehub-exchange-execution",
        "roehub-exchange-execution.hcl",
    ),
    _ServiceRole("identity", "roehub-identity", "roehub-identity", "roehub-identity.hcl"),
    _ServiceRole(
        "notification-dispatcher",
        "roehub-notification-dispatcher",
        "roehub-notification-dispatcher",
        "roehub-notification-dispatcher.hcl",
    ),
    _ServiceRole(
        "secret-operator",
        "roehub-secret-operator",
        "roehub-secret-operator",
        "roehub-secret-operator.hcl",
    ),
    _ServiceRole(
        "telegram-bot-worker",
        "roehub-telegram-bot-worker",
        "roehub-telegram-bot-worker",
        "roehub-telegram-bot-worker.hcl",
    ),
)


def initialize_owner_custody(
    *,
    address: str,
    recipient_paths: Sequence[Path],
    delivery_dir: Path,
) -> OwnerInitResult:
    """Initialize a fresh OpenBao instance with 3 encrypted, owner-held shares."""

    recipients = _load_public_recipients(recipient_paths)
    destination = _private_delivery_target(delivery_dir)
    client = _Client(address)
    health = _health_status(client.address)

    if destination.exists():
        _validate_initial_delivery(destination)
        if health in {200, 429, 472, 473, 503}:
            return OwnerInitResult(
                status="already_initialized",
                delivery_dir=destination,
                recipient_count=len(recipients),
            )
        raise OwnerInitError("existing custody delivery does not match OpenBao state")
    if health != 501:
        raise OwnerInitError("OpenBao must be fresh and uninitialized for owner init")

    encoded_recipients = [base64.b64encode(value).decode("ascii") for value in recipients]
    response = client.request_json(
        "PUT",
        "/v1/sys/init",
        {
            "secret_shares": 3,
            "secret_threshold": 2,
            "pgp_keys": encoded_recipients,
            "root_token_pgp_key": encoded_recipients[0],
        },
    )
    encrypted_shares = response.get("keys_base64")
    encrypted_admin = response.get("root_token")
    if (
        not isinstance(encrypted_shares, list)
        or len(encrypted_shares) != 3
        or not all(isinstance(value, str) and value for value in encrypted_shares)
        or not isinstance(encrypted_admin, str)
        or not encrypted_admin
    ):
        raise OwnerInitError("OpenBao initialization returned invalid encrypted custody material")

    _write_delivery(
        destination,
        {
            "unseal-share-1.pgp": _decode_ciphertext(encrypted_shares[0]),
            "unseal-share-2.pgp": _decode_ciphertext(encrypted_shares[1]),
            "unseal-share-3.pgp": _decode_ciphertext(encrypted_shares[2]),
            "initial-admin.pgp": _decode_ciphertext(encrypted_admin),
            "owner-init.json": _json_bytes(
                {
                    "schema": SCHEMA,
                    "status": "prepared",
                    "initial_admin_encrypted": True,
                    "recipient_count": 3,
                    "unseal_shares": 3,
                    "unseal_threshold": 2,
                }
            ),
        },
    )
    if _health_status(client.address) != 503:
        raise OwnerInitError("OpenBao did not enter the sealed state after owner init")
    return OwnerInitResult(
        status="initialized", delivery_dir=destination, recipient_count=len(recipients)
    )


def provision_service_credentials(
    *,
    address: str,
    administrator_token_file: Path,
    delivery_dir: Path,
) -> OwnerInitResult:
    """Create narrowly scoped AppRole bootstrap files after owner unseal.

    The delivery directory contains one subdirectory per service, each with a
    public RoleID and a response-wrapped one-time SecretID.  It is intentionally
    not a shared directory: the installation owner mounts each subdirectory only
    into that service's credential bootstrap boundary.
    """

    destination = _private_delivery_target(delivery_dir)
    client = _Client(address, SecureTokenFile(administrator_token_file).read())
    health = _health_status(client.address)
    if destination.exists():
        _validate_service_delivery(destination)
        if health == 200:
            return OwnerInitResult(
                status="already_provisioned",
                delivery_dir=destination,
                recipient_count=0,
                service_count=len(_SERVICE_ROLES),
            )
        raise OwnerInitError("existing service delivery does not match OpenBao state")
    if health != 200:
        raise OwnerInitError("OpenBao must be unsealed before service credential provisioning")

    _ensure_openbao_layout(client)
    files: dict[str, bytes] = {}
    for specification in _SERVICE_ROLES:
        role_id, wrapping_token = _issue_wrapped_role(client, specification)
        files[f"{specification.delivery_name}/role-id"] = role_id.encode("utf-8") + b"\n"
        files[f"{specification.delivery_name}/wrapped-secret-id"] = (
            wrapping_token.encode("utf-8") + b"\n"
        )
    files["service-delivery.json"] = _json_bytes(
        {
            "schema": SCHEMA,
            "status": "prepared",
            "response_wrapped": True,
            "service_count": len(_SERVICE_ROLES),
        }
    )
    _write_delivery(destination, files)
    client.request_json("POST", "/v1/auth/token/revoke-self", {})
    return OwnerInitResult(
        status="provisioned",
        delivery_dir=destination,
        recipient_count=0,
        service_count=len(_SERVICE_ROLES),
    )


def _load_public_recipients(recipient_paths: Sequence[Path]) -> tuple[bytes, bytes, bytes]:
    if len(recipient_paths) != 3:
        raise OwnerInitError("exactly three public PGP recipients are required")
    values = tuple(
        _read_regular_file(path, maximum_bytes=_MAX_RECIPIENT_BYTES) for path in recipient_paths
    )
    if len({hashlib.sha256(value).digest() for value in values}) != 3:
        raise OwnerInitError("public PGP recipients must be distinct")
    _verify_public_pgp_inputs(recipient_paths)
    return values  # type: ignore[return-value]


def _verify_public_pgp_inputs(paths: Sequence[Path]) -> None:
    gpg = shutil.which("gpg")
    if gpg is None:
        raise OwnerInitError("GnuPG is required to validate public PGP recipients")
    with tempfile.TemporaryDirectory(prefix="roehub-openbao-pgp-check-") as temporary_name:
        temporary_root = Path(temporary_name)
        temporary_root.chmod(0o700)
        for path in paths:
            completed = subprocess.run(
                [
                    gpg,
                    "--batch",
                    "--quiet",
                    "--homedir",
                    str(temporary_root),
                    "--import-options",
                    "import-show",
                    "--dry-run",
                    "--import",
                    str(path.expanduser().resolve()),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=15,
            )
            if completed.returncode != 0:
                raise OwnerInitError("a supplied PGP recipient is invalid")


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    resolved = path.expanduser().resolve()
    try:
        descriptor = os.open(resolved, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise OwnerInitError("required owner material is unavailable") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0 or info.st_size > maximum_bytes:
            raise OwnerInitError("required owner material is unsafe")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            value = stream.read(maximum_bytes + 1)
    finally:
        os.close(descriptor)
    if not value or len(value) > maximum_bytes:
        raise OwnerInitError("required owner material is invalid")
    return value


def _private_delivery_target(path: Path) -> Path:
    destination = path.expanduser().resolve()
    parent = destination.parent
    try:
        info = parent.stat()
    except OSError as error:
        raise OwnerInitError("owner delivery parent is unavailable") from error
    if not parent.is_dir() or stat.S_IMODE(info.st_mode) & 0o077:
        raise OwnerInitError("owner delivery parent must be a private directory")
    if destination.is_symlink():
        raise OwnerInitError("owner delivery path must not be a symlink")
    return destination


def _decode_ciphertext(value: str) -> bytes:
    try:
        ciphertext = base64.b64decode(value, validate=True)
    except (ValueError, UnicodeError) as error:
        raise OwnerInitError("OpenBao returned invalid encrypted custody material") from error
    if not ciphertext or len(ciphertext) > _MAX_RECIPIENT_BYTES:
        raise OwnerInitError("OpenBao returned invalid encrypted custody material")
    return ciphertext


def _write_delivery(destination: Path, files: Mapping[str, bytes]) -> None:
    if destination.exists():
        raise OwnerInitError("owner delivery destination already exists")
    stage = Path(tempfile.mkdtemp(prefix=".roehub-openbao-", dir=destination.parent))
    stage.chmod(0o700)
    try:
        for relative, value in files.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.parent.chmod(0o700)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(value)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(descriptor)
        os.rename(stage, destination)
        _fsync_directory(destination.parent)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def _validate_initial_delivery(destination: Path) -> None:
    _validate_private_directory(destination)
    files = {entry.name for entry in destination.iterdir() if entry.is_file()}
    if files != _INITIAL_FILES or any(entry.is_dir() for entry in destination.iterdir()):
        raise OwnerInitError("existing owner custody delivery is unsafe")
    for name in files:
        _validate_private_file(destination / name)
    try:
        metadata = json.loads((destination / "owner-init.json").read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise OwnerInitError("existing owner custody delivery is invalid") from error
    if metadata != {
        "initial_admin_encrypted": True,
        "recipient_count": 3,
        "schema": SCHEMA,
        "status": "prepared",
        "unseal_shares": 3,
        "unseal_threshold": 2,
    }:
        raise OwnerInitError("existing owner custody delivery is invalid")


def _validate_service_delivery(destination: Path) -> None:
    _validate_private_directory(destination)
    metadata_path = destination / "service-delivery.json"
    _validate_private_file(metadata_path)
    try:
        metadata = json.loads(metadata_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise OwnerInitError("existing service credential delivery is invalid") from error
    if metadata != {
        "response_wrapped": True,
        "schema": SCHEMA,
        "service_count": len(_SERVICE_ROLES),
        "status": "prepared",
    }:
        raise OwnerInitError("existing service credential delivery is invalid")
    expected = {"service-delivery.json", *(role.delivery_name for role in _SERVICE_ROLES)}
    if {entry.name for entry in destination.iterdir()} != expected:
        raise OwnerInitError("existing service credential delivery is unsafe")
    for role in _SERVICE_ROLES:
        service_root = destination / role.delivery_name
        _validate_private_directory(service_root)
        if {entry.name for entry in service_root.iterdir()} != {"role-id", "wrapped-secret-id"}:
            raise OwnerInitError("existing service credential delivery is unsafe")
        _validate_private_file(service_root / "role-id")
        _validate_private_file(service_root / "wrapped-secret-id")


def _validate_private_directory(path: Path) -> None:
    try:
        info = path.stat()
    except OSError as error:
        raise OwnerInitError("owner delivery is unavailable") from error
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise OwnerInitError("owner delivery is unsafe")


def _validate_private_file(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise OwnerInitError("owner delivery is unavailable") from error
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_size <= 0
        or info.st_size > _MAX_RECIPIENT_BYTES
        or stat.S_IMODE(info.st_mode) != 0o600
    ):
        raise OwnerInitError("owner delivery is unsafe")


def _health_status(address: str) -> int | None:
    request = urllib.request.Request(
        f"{address}/v1/sys/health?standbyok=true&perfstandbyok=true", method="GET"
    )
    try:
        with open_without_redirect(request, timeout=5) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)
    except (OSError, TimeoutError):
        return None


def _ensure_openbao_layout(client: _Client) -> None:
    mounts = client.request_json("GET", "/v1/sys/mounts")
    _ensure_mount(client, mounts, "kv/", "kv", {"type": "kv", "options": {"version": "2"}})
    _ensure_mount(client, mounts, "transit/", "transit", {"type": "transit"})
    client.request_json("POST", "/v1/kv/config", {"max_versions": 5})
    client.request_json(
        "POST",
        "/v1/transit/keys/roehub-exchange-credentials",
        {
            "allow_plaintext_backup": False,
            "exportable": False,
            "type": "aes256-gcm96",
        },
    )
    transit_key = client.request_json("GET", "/v1/transit/keys/roehub-exchange-credentials")
    if not transit_key:
        raise OwnerInitError("OpenBao transit key verification returned an invalid payload")
    auth_methods = client.request_json("GET", "/v1/sys/auth")
    _ensure_mount(client, auth_methods, "approle/", "approle", {"type": "approle"})
    for role in _SERVICE_ROLES:
        policy_path = POLICIES / role.policy_file
        try:
            policy = policy_path.read_text(encoding="utf-8")
        except OSError as error:
            raise OwnerInitError("OpenBao policy asset is unavailable") from error
        client.request_json("PUT", f"/v1/sys/policies/acl/{role.policy_name}", {"policy": policy})


def _ensure_mount(
    client: _Client,
    existing: Mapping[str, Any],
    mount_name: str,
    expected_type: str,
    create_payload: Mapping[str, Any],
) -> None:
    mount = existing.get(mount_name)
    if mount is None:
        endpoint = "auth" if expected_type == "approle" else "mounts"
        client.request_json(
            "POST", f"/v1/sys/{endpoint}/{mount_name.rstrip('/')}", create_payload
        )
        return
    if not isinstance(mount, Mapping) or mount.get("type") != expected_type:
        raise OwnerInitError("existing OpenBao mount has an incompatible type")


def _issue_wrapped_role(client: _Client, specification: _ServiceRole) -> tuple[str, str]:
    client.request_json(
        "POST",
        f"/v1/auth/approle/role/{specification.role_name}",
        {
            "bind_secret_id": True,
            "secret_id_num_uses": 1,
            "secret_id_ttl": "30m",
            "token_max_ttl": "30m",
            "token_no_default_policy": True,
            "token_policies": [specification.policy_name],
            "token_ttl": "15m",
        },
    )
    role_response = client.request_json(
        "GET", f"/v1/auth/approle/role/{specification.role_name}/role-id"
    )
    role_id = role_response.get("data", {}).get("role_id")
    wrapped = client.request_json(
        "POST",
        f"/v1/auth/approle/role/{specification.role_name}/secret-id",
        {},
        extra_headers={"X-Vault-Wrap-TTL": "5m"},
    )
    wrap_info = wrapped.get("wrap_info")
    expected_path = f"auth/approle/role/{specification.role_name}/secret-id"
    if (
        not isinstance(role_id, str)
        or not role_id
        or not isinstance(wrap_info, Mapping)
        or not isinstance(wrap_info.get("token"), str)
        or not wrap_info.get("token")
        or wrap_info.get("creation_path") != expected_path
        or not isinstance(wrap_info.get("ttl"), int)
        or not 0 < int(wrap_info["ttl"]) <= 300
    ):
        raise OwnerInitError("OpenBao AppRole delivery is invalid")
    return role_id, str(wrap_info["token"])


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


__all__ = [
    "OwnerInitError",
    "OwnerInitResult",
    "initialize_owner_custody",
    "provision_service_credentials",
]
