"""Disposable OpenBao runtime proof for Stage 08.

The verifier captures all initialization and service credentials in memory or mode-0600
temporary files, emits only boolean/count evidence, and always removes Docker volumes.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from infra.openbao.snapshot import backup_snapshot, restore_snapshot
from trading.platform.secrets import (
    OpenBaoPermissionError,
    OpenBaoSecretNotFoundError,
    OpenBaoSecretResolver,
    SecretKind,
    SecretValue,
    SecureTokenFile,
)
from trading.platform.secrets.transport import (
    normalize_openbao_address,
    open_without_redirect,
)

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "infra" / "docker" / "openbao-embedded.compose.yml"
POLICIES = ROOT / "infra" / "openbao" / "policies"
RESULT_SCHEMA = "io.roehub.openbao-runtime-proof/v1"


class RuntimeProofError(RuntimeError):
    """A sanitized runtime-proof failure."""


class HttpStatusError(RuntimeProofError):
    def __init__(
        self,
        status: int,
        method: str,
        path: str,
        category: str = "http_error",
    ) -> None:
        super().__init__(
            f"OpenBao request failed with status {status}: {method} {path} ({category})"
        )
        self.status = status
        self.category = category


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
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 10.0,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        headers.update(extra_headers or {})
        if self.credential is not None:
            headers["X-Vault-Token"] = self.credential
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url=f"{self.address}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with open_without_redirect(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            error_body = error.read(16_384)
            category = "http_error"
            if b"valid hex or base64" in error_body:
                category = "invalid_encoding"
            elif b"key is shorter than minimum" in error_body.lower():
                category = "invalid_key_too_short"
            elif b"key is longer than maximum" in error_body.lower():
                category = "invalid_key_too_long"
            elif b"failed to compute combined key" in error_body.lower():
                category = "invalid_key_combination"
            elif b"failed to verify recovery key" in error_body.lower():
                category = "invalid_recovery_key"
            elif b"failed to setup unseal key" in error_body.lower():
                category = "invalid_unseal_setup"
            elif b"failed to decrypt keys from storage" in error_body.lower():
                category = "invalid_stored_key"
            elif b"unseal failed, invalid key" in error_body.lower():
                category = "barrier_key_mismatch"
            elif b"invalid key" in error_body.lower():
                category = "invalid_key"
            raise HttpStatusError(int(error.code), method, path, category) from error
        except (OSError, TimeoutError) as error:
            raise RuntimeProofError("OpenBao request is unavailable") from error
        if not body:
            return {}
        try:
            result = json.loads(body)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise RuntimeProofError("OpenBao returned invalid JSON") from error
        if not isinstance(result, dict):
            raise RuntimeProofError("OpenBao returned an invalid payload")
        return result


def _compose(
    project: str,
    port: int,
    *arguments: str,
    compose_override: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "ROEHUB_OPENBAO_PORT": str(port)}
    command = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        str(COMPOSE),
    ]
    if compose_override is not None:
        command.extend(("--file", str(compose_override)))
    completed = subprocess.run(
        [*command, *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    if check and completed.returncode != 0:
        raise RuntimeProofError("Docker Compose operation failed")
    return completed


def _cleanup_project(
    project: str,
    port: int,
    *,
    compose_override: Path | None = None,
) -> None:
    teardown = _compose(
        project,
        port,
        "down",
        "--volumes",
        "--remove-orphans",
        compose_override=compose_override,
        check=False,
    )
    leftovers: list[str] = []
    for kind, arguments in (
        (
            "containers",
            [
                "docker",
                "ps",
                "--all",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.ID}}",
            ],
        ),
        (
            "volumes",
            [
                "docker",
                "volume",
                "ls",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.Name}}",
            ],
        ),
        (
            "networks",
            [
                "docker",
                "network",
                "ls",
                "--filter",
                f"label=com.docker.compose.project={project}",
                "--format",
                "{{.ID}}",
            ],
        ),
    ):
        inspected = subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if inspected.returncode != 0 or inspected.stdout.strip():
            leftovers.append(kind)
    if teardown.returncode != 0 or leftovers:
        raise RuntimeProofError(
            "OpenBao runtime cleanup failed" + (f" for {','.join(leftovers)}" if leftovers else "")
        )


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _health_status(address: str) -> int | None:
    request = urllib.request.Request(
        f"{address}/v1/sys/health?standbyok=true&perfstandbyok=true",
        method="GET",
    )
    try:
        with open_without_redirect(request, timeout=2) as response:
            return int(response.status)
    except urllib.error.HTTPError as error:
        return int(error.code)
    except (OSError, TimeoutError):
        return None


def _wait_for_status(
    address: str,
    expected: set[int],
    *,
    label: str,
    timeout: float = 45.0,
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = _health_status(address)
        if status in expected:
            return int(status)
        time.sleep(0.25)
    raise RuntimeProofError(f"OpenBao did not reach the expected health state: {label}")


def _initialize_disposable(client: _Client) -> tuple[tuple[str, ...], str]:
    result = client.request_json(
        "PUT",
        "/v1/sys/init",
        {"secret_shares": 1, "secret_threshold": 1},
    )
    key_items = result.get("keys_base64")
    admin_credential = result.get("root_token")
    if (
        not isinstance(key_items, list)
        or len(key_items) != 1
        or not isinstance(key_items[0], str)
        or not isinstance(admin_credential, str)
    ):
        raise RuntimeProofError("OpenBao initialization returned invalid material")
    return (key_items[0],), admin_credential


def _initialize_with_owner_pgp(
    client: _Client,
    temporary_root: Path,
) -> tuple[tuple[str, str], str, list[str]]:
    gpg_binary = shutil.which("gpg")
    if gpg_binary is None:
        raise RuntimeProofError("GnuPG is required for owner-custody proof")
    public_keys: list[str] = []
    owner_environments: list[dict[str, str]] = []
    for index in range(3):
        # Keep the path short enough for the local gpg-agent Unix socket.
        gpg_home = temporary_root / f"g{index + 1}"
        gpg_home.mkdir(mode=stat.S_IRWXU)
        environment = {**os.environ, "GNUPGHOME": str(gpg_home)}
        batch_path = temporary_root / f"gpg-owner-{index + 1}.batch"
        batch_path.write_text(
            "\n".join(
                (
                    "Key-Type: RSA",
                    "Key-Length: 2048",
                    "Key-Usage: sign",
                    "Subkey-Type: RSA",
                    "Subkey-Length: 2048",
                    "Subkey-Usage: encrypt",
                    f"Name-Real: Roehub Stage08 Custodian {index + 1}",
                    f"Name-Email: custodian-{index + 1}@invalid",
                    "Expire-Date: 1d",
                    "%no-protection",
                    "%commit",
                    "",
                )
            ),
            encoding="utf-8",
        )
        batch_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        generated = subprocess.run(
            [gpg_binary, "--batch", "--generate-key", str(batch_path)],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
        if generated.returncode != 0:
            raise RuntimeProofError(f"owner PGP key generation failed for custodian {index + 1}")
        listed = subprocess.run(
            [
                gpg_binary,
                "--batch",
                "--with-colons",
                "--fingerprint",
                "--list-secret-keys",
            ],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        fingerprint = next(
            (line.split(":")[9] for line in listed.stdout.splitlines() if line.startswith("fpr:")),
            None,
        )
        if listed.returncode != 0 or not fingerprint:
            raise RuntimeProofError("owner PGP fingerprint lookup failed")
        exported = subprocess.run(
            [gpg_binary, "--batch", "--export", fingerprint],
            env=environment,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if exported.returncode != 0 or not exported.stdout:
            raise RuntimeProofError("owner PGP public key export failed")
        public_keys.append(base64.b64encode(exported.stdout).decode("ascii"))
        owner_environments.append(environment)

    result = client.request_json(
        "PUT",
        "/v1/sys/init",
        {
            "secret_shares": 3,
            "secret_threshold": 2,
            "pgp_keys": public_keys[:3],
            "root_token_pgp_key": public_keys[0],
        },
    )
    encrypted_shares = result.get("keys_base64")
    encrypted_admin = result.get("root_token")
    if (
        not isinstance(encrypted_shares, list)
        or len(encrypted_shares) != 3
        or not all(isinstance(item, str) and item for item in encrypted_shares)
        or not isinstance(encrypted_admin, str)
        or not encrypted_admin
    ):
        raise RuntimeProofError("owner-PGP initialization returned invalid material")
    shares = tuple(
        _api_unseal_share(
            _gpg_decrypt_text(
                encrypted_shares[index],
                binary=gpg_binary,
                environment=owner_environments[index],
            )
        )
        for index in range(2)
    )
    admin_credential = _gpg_decrypt_text(
        encrypted_admin,
        binary=gpg_binary,
        environment=owner_environments[0],
    )
    if len(set(shares)) != 2 or not admin_credential:
        raise RuntimeProofError("owner-PGP custody decryption failed")
    seal_status = client.request_json("GET", "/v1/sys/seal-status")
    if (
        seal_status.get("n") != 3
        or seal_status.get("t") != 2
        or seal_status.get("progress") != 0
        or seal_status.get("sealed") is not True
    ):
        raise RuntimeProofError("owner-PGP seal parameters were not persisted")
    return (
        (shares[0], shares[1]),
        admin_credential,
        [
            *encrypted_shares,
            encrypted_admin,
        ],
    )


def _gpg_decrypt_text(
    encoded: str,
    *,
    binary: str,
    environment: dict[str, str],
) -> str:
    try:
        ciphertext = base64.b64decode(encoded, validate=True)
    except (ValueError, UnicodeError) as error:
        raise RuntimeProofError("owner-PGP material encoding is invalid") from error
    decrypted = subprocess.run(
        [binary, "--batch", "--quiet", "--decrypt"],
        env=environment,
        input=ciphertext,
        capture_output=True,
        check=False,
        timeout=10,
    )
    if decrypted.returncode != 0 or not decrypted.stdout:
        raise RuntimeProofError("owner-PGP material decryption failed")
    try:
        value = decrypted.stdout.decode("utf-8").strip()
    except UnicodeError as error:
        raise RuntimeProofError("owner-PGP material is invalid") from error
    if not value or any(character.isspace() for character in value):
        raise RuntimeProofError("owner-PGP material is invalid")
    return value


def _api_unseal_share(value: str) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{66}", value):
        return value.lower()
    try:
        decoded = base64.b64decode(value, validate=True)
    except (ValueError, UnicodeError) as error:
        raise RuntimeProofError(
            f"owner-PGP unseal share encoding is invalid (length={len(value)})"
        ) from error
    if len(decoded) != 33:
        raise RuntimeProofError(
            "owner-PGP unseal share encoding is invalid "
            f"(length={len(value)}, decoded={len(decoded)})"
        )
    return value


def _unseal(client: _Client, materials: Sequence[str], *, phase: str) -> None:
    result: dict[str, Any] = {}
    for index, material in enumerate(materials, start=1):
        try:
            result = client.request_json("PUT", "/v1/sys/unseal", {"key": material})
        except HttpStatusError as error:
            status = client.request_json("GET", "/v1/sys/seal-status")
            progress = status.get("progress")
            threshold = status.get("t")
            migration = status.get("migration")
            raise RuntimeProofError(
                f"OpenBao rejected {phase} unseal share "
                f"{index} with status {error.status} ({error.category}; "
                f"progress={progress}; threshold={threshold}; migration={migration})"
            ) from error
    if result.get("sealed") is not False:
        raise RuntimeProofError("OpenBao unseal did not complete")


def _write_credential(path: Path, value: str) -> SecureTokenFile:
    path.write_text(value, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return SecureTokenFile(path)


def _policy_payload(path: Path, substitutions: dict[str, str] | None = None) -> str:
    value = path.read_text(encoding="utf-8")
    for name, replacement in (substitutions or {}).items():
        if not replacement.replace("-", "").isalnum():
            raise RuntimeProofError("policy identity is invalid")
        value = value.replace(f"${{{name}}}", replacement)
    if "${" in value:
        raise RuntimeProofError("policy template is incomplete")
    return value


def _issue_approle(
    administrator: _Client,
    role: str,
    policy: str,
    forbidden: list[str],
    temporary_root: Path,
) -> tuple[str, dict[str, bool]]:
    administrator.request_json(
        "POST",
        f"/v1/auth/approle/role/{role}",
        {
            "bind_secret_id": True,
            "secret_id_num_uses": 1,
            "secret_id_ttl": "30m",
            "token_max_ttl": "30m",
            "token_no_default_policy": True,
            "token_policies": [policy],
            "token_ttl": "15m",
        },
    )
    public_part = (
        administrator.request_json("GET", f"/v1/auth/approle/role/{role}/role-id")
        .get("data", {})
        .get("role_id")
    )
    wrapped = administrator.request_json(
        "POST",
        f"/v1/auth/approle/role/{role}/secret-id",
        {},
        extra_headers={"X-Vault-Wrap-TTL": "5m"},
    )
    wrap_info = wrapped.get("wrap_info")
    expected_creation_path = f"auth/approle/role/{role}/secret-id"
    if (
        not isinstance(public_part, str)
        or not isinstance(wrap_info, dict)
        or not isinstance(wrap_info.get("token"), str)
        or wrap_info.get("creation_path") != expected_creation_path
        or not isinstance(wrap_info.get("ttl"), int)
        or not 0 < wrap_info["ttl"] <= 300
    ):
        raise RuntimeProofError("AppRole credential issuance failed")
    wrapping_token = wrap_info["token"]
    wrapped_source = _write_credential(
        temporary_root / f"{role}.wrapped",
        wrapping_token,
    )
    wrapped_mode_is_restricted = (
        stat.S_IMODE(wrapped_source.path.stat().st_mode) == stat.S_IRUSR | stat.S_IWUSR
    )
    wrapped_from_file = wrapped_source.read()
    unwrapped = _Client(administrator.address, wrapped_from_file).request_json(
        "POST",
        "/v1/sys/wrapping/unwrap",
        {},
    )
    wrapped_source.path.unlink()
    private_part = unwrapped.get("data", {}).get("secret_id")
    if not isinstance(private_part, str):
        raise RuntimeProofError("AppRole wrapped credential is invalid")
    try:
        _Client(administrator.address, wrapping_token).request_json(
            "POST",
            "/v1/sys/wrapping/unwrap",
            {},
        )
    except HttpStatusError as error:
        if error.status not in {400, 401, 403}:
            raise
    else:
        raise RuntimeProofError("AppRole wrapped credential was reusable")
    login = _Client(administrator.address).request_json(
        "POST",
        "/v1/auth/approle/login",
        {"role_id": public_part, "secret_id": private_part},
    )
    credential = login.get("auth", {}).get("client_token")
    policies = login.get("auth", {}).get("policies")
    lease_duration = login.get("auth", {}).get("lease_duration")
    renewable = login.get("auth", {}).get("renewable")
    if (
        not isinstance(credential, str)
        or policies != [policy]
        or not isinstance(lease_duration, int)
        or not 0 < lease_duration <= 900
        or renewable is not True
    ):
        raise RuntimeProofError("AppRole login returned a broad or invalid identity")
    renewed = _Client(administrator.address, credential).request_json(
        "POST",
        "/v1/auth/token/renew-self",
        {"increment": "15m"},
    )
    renewed_auth = renewed.get("auth", {})
    renewed_credential = renewed_auth.get("client_token")
    renewed_duration = renewed_auth.get("lease_duration")
    if (
        not isinstance(renewed_credential, str)
        or renewed_auth.get("policies") != [policy]
        or not isinstance(renewed_duration, int)
        or not 0 < renewed_duration <= 1800
    ):
        raise RuntimeProofError("AppRole token renewal exceeded its identity boundary")
    forbidden.extend(
        (
            public_part,
            wrapping_token,
            private_part,
            credential,
            renewed_credential,
        )
    )
    return renewed_credential, {
        "response_wrapped": True,
        "one_time_unwrap": True,
        "mode_0600_delivery": (wrapped_mode_is_restricted and not wrapped_source.path.exists()),
        "token_renewal": True,
    }


def _expect_denied(
    client: _Client, method: str, path: str, payload: dict[str, Any] | None = None
) -> None:
    try:
        client.request_json(method, path, payload)
    except HttpStatusError as error:
        if error.status in {401, 403}:
            return
        raise
    raise RuntimeProofError("least-privilege request was unexpectedly allowed")


def _setup(
    administrator: _Client,
    forbidden: list[str],
    temporary_root: Path,
) -> tuple[dict[str, str], dict[str, bool]]:
    administrator.request_json(
        "POST", "/v1/sys/mounts/kv", {"type": "kv", "options": {"version": "2"}}
    )
    administrator.request_json("POST", "/v1/kv/config", {"max_versions": 5})
    administrator.request_json("POST", "/v1/sys/mounts/transit", {"type": "transit"})
    administrator.request_json(
        "POST",
        "/v1/transit/keys/roehub-exchange-credentials",
        {"allow_plaintext_backup": False, "exportable": False, "type": "aes256-gcm96"},
    )
    administrator.request_json("POST", "/v1/sys/auth/approle", {"type": "approle"})

    policy_files = {
        "roehub-api": POLICIES / "roehub-api.hcl",
        "roehub-backup-recovery": POLICIES / "roehub-backup-recovery.hcl",
        "roehub-exchange-execution": POLICIES / "roehub-exchange-execution.hcl",
        "roehub-identity": POLICIES / "roehub-identity.hcl",
        "roehub-notification-dispatcher": POLICIES / "roehub-notification-dispatcher.hcl",
        "roehub-telegram-bot-worker": POLICIES / "roehub-telegram-bot-worker.hcl",
        "roehub-secret-operator": POLICIES / "roehub-secret-operator.hcl",
    }
    for name, path in policy_files.items():
        administrator.request_json(
            "PUT", f"/v1/sys/policies/acl/{name}", {"policy": _policy_payload(path)}
        )
    plugin_policy = "roehub-plugin-runtime-org-a-instance-a"
    administrator.request_json(
        "PUT",
        f"/v1/sys/policies/acl/{plugin_policy}",
        {
            "policy": _policy_payload(
                POLICIES / "roehub-plugin-runtime.template.hcl",
                {"organization_id": "org-a", "instance_id": "instance-a"},
            )
        },
    )

    specifications = {
        "api": ("api", "roehub-api"),
        "backup": ("backup", "roehub-backup-recovery"),
        "exchange": ("exchange", "roehub-exchange-execution"),
        "identity": ("identity", "roehub-identity"),
        "notifications": ("notifications", "roehub-notification-dispatcher"),
        "telegram-worker": ("telegram-worker", "roehub-telegram-bot-worker"),
        "operator": ("operator", "roehub-secret-operator"),
        "plugin": ("plugin", plugin_policy),
    }
    identities: dict[str, str] = {}
    proofs: list[dict[str, bool]] = []
    for name, (role, policy) in specifications.items():
        identity, proof = _issue_approle(
            administrator,
            role,
            policy,
            forbidden,
            temporary_root,
        )
        identities[name] = identity
        proofs.append(proof)
    if len(set(identities.values())) != len(identities):
        raise RuntimeProofError("service identities are shared")
    return identities, {
        key: all(proof[key] for proof in proofs)
        for key in (
            "response_wrapped",
            "one_time_unwrap",
            "mode_0600_delivery",
            "token_renewal",
        )
    }


def _write_value(client: _Client, path: str, field: str, value: str) -> int:
    response = client.request_json("POST", f"/v1/kv/data/{path}", {"data": {field: value}})
    version = response.get("data", {}).get("version")
    if not isinstance(version, int):
        raise RuntimeProofError("KV write did not return a version")
    return version


def _exercise_contracts(
    address: str,
    temporary_root: Path,
    identities: dict[str, str],
    forbidden: list[str],
) -> dict[str, Any]:
    canaries = {
        name: secrets.token_urlsafe(24)
        for name in (
            "exchange",
            "oidc-v1",
            "oidc-v2",
            "plugin",
            "plugin-other",
            "telegram",
            "telegram-recipient",
        )
    }
    forbidden.extend(canaries.values())
    operator = _Client(address, identities["operator"])
    initial_writes = (
        _write_value(
            operator,
            "roehub/exchange/org-a/connection-a",
            "credential",
            canaries["exchange"],
        ),
        _write_value(
            operator,
            "roehub/oidc/provider-a",
            "client_secret",
            canaries["oidc-v1"],
        ),
        _write_value(
            operator,
            "roehub/plugins/org-a/instance-a",
            "credential",
            canaries["plugin"],
        ),
        _write_value(
            operator,
            "roehub/plugins/org-a/instance-b",
            "credential",
            canaries["plugin-other"],
        ),
        _write_value(
            operator,
            "roehub/telegram/providers/org-a/instance-a",
            "bot_token",
            canaries["telegram"],
        ),
    )
    if initial_writes != (1, 1, 1, 1, 1):
        raise RuntimeProofError("initial KV versions are invalid")

    sources = {
        name: _write_credential(temporary_root / f"{name}.credential", value)
        for name, value in identities.items()
    }
    identity_resolver = OpenBaoSecretResolver(address, sources["identity"])
    notification_resolver = OpenBaoSecretResolver(address, sources["notifications"])
    telegram_worker_resolver = OpenBaoSecretResolver(
        address, sources["telegram-worker"]
    )
    exchange_resolver = OpenBaoSecretResolver(address, sources["exchange"])
    plugin_resolver = OpenBaoSecretResolver(address, sources["plugin"])
    api_client = _Client(address, identities["api"])

    resolved_values = (
        identity_resolver.resolve(
            "openbao://kv/roehub/oidc/provider-a?version=1#client_secret",
            expected_kind=SecretKind.OIDC,
        ).reveal_text(),
        notification_resolver.resolve(
            "openbao://kv/roehub/telegram/providers/org-a/instance-a#bot_token",
            expected_kind=SecretKind.TELEGRAM,
        ).reveal_text(),
        exchange_resolver.resolve(
            "openbao://kv/roehub/exchange/org-a/connection-a#credential",
            expected_kind=SecretKind.EXCHANGE,
        ).reveal_text(),
        plugin_resolver.resolve(
            "openbao://kv/roehub/plugins/org-a/instance-a#credential",
            expected_kind=SecretKind.PLUGIN,
        ).reveal_text(),
    )
    if resolved_values != (
        canaries["oidc-v1"],
        canaries["telegram"],
        canaries["exchange"],
        canaries["plugin"],
    ):
        raise RuntimeProofError("typed secret resolution returned an invalid value")

    recipient_reference = (
        "openbao://kv/roehub/telegram/recipients/"
        "org-a/instance-a/user-a/binding-a#chat_id"
    )
    telegram_worker_resolver.store(
        recipient_reference,
        value=SecretValue.from_text(canaries["telegram-recipient"]),
        expected_kind=SecretKind.TELEGRAM,
    )
    if (
        notification_resolver.resolve(
            recipient_reference,
            expected_kind=SecretKind.TELEGRAM,
        ).reveal_text()
        != canaries["telegram-recipient"]
    ):
        raise RuntimeProofError("Telegram recipient secret boundary is invalid")

    try:
        identity_resolver.resolve(
            "openbao://kv/roehub/telegram/providers/org-a/instance-a#bot_token"
        )
    except OpenBaoPermissionError:
        pass
    else:
        raise RuntimeProofError("identity service crossed the Telegram policy boundary")
    try:
        plugin_resolver.resolve("openbao://kv/roehub/plugins/org-a/instance-b#credential")
    except OpenBaoPermissionError:
        pass
    else:
        raise RuntimeProofError("plugin service crossed its instance policy boundary")
    try:
        notification_resolver.store(
            recipient_reference,
            value=SecretValue.from_text("dispatcher-write-must-be-denied"),
            expected_kind=SecretKind.TELEGRAM,
        )
    except OpenBaoPermissionError:
        pass
    else:
        raise RuntimeProofError("notification dispatcher wrote a recipient secret")

    api_client.request_json("GET", "/v1/kv/metadata/roehub/oidc/provider-a")
    _expect_denied(api_client, "GET", "/v1/kv/data/roehub/oidc/provider-a")

    exchange_client = _Client(address, identities["exchange"])
    transit = (
        exchange_client.request_json(
            "POST",
            "/v1/transit/encrypt/roehub-exchange-credentials",
            {"plaintext": base64.b64encode(canaries["exchange"].encode()).decode()},
        )
        .get("data", {})
        .get("ciphertext")
    )
    if not isinstance(transit, str):
        raise RuntimeProofError("Transit encryption returned invalid data")
    forbidden.append(transit)
    plaintext = (
        exchange_client.request_json(
            "POST",
            "/v1/transit/decrypt/roehub-exchange-credentials",
            {"ciphertext": transit},
        )
        .get("data", {})
        .get("plaintext")
    )
    if (
        not isinstance(plaintext, str)
        or base64.b64decode(plaintext).decode() != canaries["exchange"]
    ):
        raise RuntimeProofError("Transit round trip failed")
    _expect_denied(
        api_client,
        "POST",
        "/v1/transit/decrypt/roehub-exchange-credentials",
        {"ciphertext": transit},
    )

    if (
        _write_value(
            operator,
            "roehub/oidc/provider-a",
            "client_secret",
            canaries["oidc-v2"],
        )
        != 2
    ):
        raise RuntimeProofError("OIDC secret rotation did not create version 2")
    if (
        identity_resolver.resolve("openbao://kv/roehub/oidc/provider-a#client_secret").reveal_text()
        != canaries["oidc-v2"]
    ):
        raise RuntimeProofError("live resolver did not observe OIDC version 2")
    operator.request_json("POST", "/v1/kv/delete/roehub/oidc/provider-a", {"versions": [2]})
    try:
        identity_resolver.resolve("openbao://kv/roehub/oidc/provider-a#client_secret")
    except OpenBaoSecretNotFoundError:
        pass
    else:
        raise RuntimeProofError("soft-deleted latest secret remained readable")
    if (
        identity_resolver.resolve(
            "openbao://kv/roehub/oidc/provider-a?version=1#client_secret"
        ).reveal_text()
        != canaries["oidc-v1"]
    ):
        raise RuntimeProofError("explicit OIDC version selection failed")
    operator.request_json("POST", "/v1/kv/undelete/roehub/oidc/provider-a", {"versions": [2]})
    version_one = (
        operator.request_json("GET", "/v1/kv/data/roehub/oidc/provider-a?version=1")
        .get("data", {})
        .get("data")
    )
    if not isinstance(version_one, dict):
        raise RuntimeProofError("rollback source version is unavailable")
    rollback = (
        operator.request_json("POST", "/v1/kv/data/roehub/oidc/provider-a", {"data": version_one})
        .get("data", {})
        .get("version")
    )
    if rollback != 3:
        raise RuntimeProofError("rollback did not create a new auditable version")
    if (
        identity_resolver.resolve("openbao://kv/roehub/oidc/provider-a#client_secret").reveal_text()
        != canaries["oidc-v1"]
    ):
        raise RuntimeProofError("OIDC rollback version is invalid")

    ops_status = identity_resolver.readiness().as_ops_status()
    return {
        "canaries": canaries,
        "sources": sources,
        "ops_status": ops_status,
        "rotation_versions": [1, 2, 3],
    }


def _forbidden_scan(payloads: Sequence[bytes | str], forbidden: Sequence[str]) -> None:
    for payload in payloads:
        data = payload if isinstance(payload, bytes) else payload.encode("utf-8")
        for value in forbidden:
            if value.encode("utf-8") in data:
                raise RuntimeProofError("forbidden material was found in runtime evidence")


def verify(
    *,
    export_encrypted_backup: Path | None = None,
    recovery_identity_path: Path | None = None,
    recovery_recipient_path: Path | None = None,
    compose_override: Path | None = None,
) -> dict[str, Any]:
    if not COMPOSE.exists():
        raise RuntimeProofError("OpenBao Compose file is unavailable")
    if compose_override is not None:
        compose_override = compose_override.expanduser().resolve()
        if not compose_override.is_file():
            raise RuntimeProofError("OpenBao Compose override is unavailable")
    project = f"roehub-stage08-{os.getpid()}-{secrets.token_hex(4)}"
    port = _available_port()
    address = f"http://127.0.0.1:{port}"
    forbidden: list[str] = []
    safe_result: dict[str, Any] | None = None
    with tempfile.TemporaryDirectory(prefix="r8-") as temporary_name:
        temporary_root = Path(temporary_name)
        try:
            _compose(
                project,
                port,
                "config",
                "--quiet",
                compose_override=compose_override,
            )
            _compose(
                project,
                port,
                "up",
                "--detach",
                compose_override=compose_override,
            )
            if _wait_for_status(address, {501}, label="fresh-uninitialized") != 501:
                raise RuntimeProofError("fresh OpenBao was not uninitialized")
            unseal_materials, admin_credential, encrypted_custody = _initialize_with_owner_pgp(
                _Client(address), temporary_root
            )
            forbidden.extend((*unseal_materials, admin_credential, *encrypted_custody))
            if _wait_for_status(address, {503}, label="initialized-sealed") != 503:
                raise RuntimeProofError("initialized OpenBao was not sealed")
            persisted_seal_status = _Client(address).request_json("GET", "/v1/sys/seal-status")
            if (
                persisted_seal_status.get("n") != 3
                or persisted_seal_status.get("t") != 2
                or persisted_seal_status.get("progress") != 0
            ):
                raise RuntimeProofError("owner-PGP seal parameters changed before owner unseal")
            _unseal(_Client(address), unseal_materials, phase="initial owner")
            _wait_for_status(address, {200}, label="initial-unsealed")

            administrator = _Client(address, admin_credential)
            identities, approle_proof = _setup(
                administrator,
                forbidden,
                temporary_root,
            )
            exercised = _exercise_contracts(address, temporary_root, identities, forbidden)
            administrator.request_json("POST", "/v1/auth/token/revoke-self", {})
            _expect_denied(administrator, "GET", "/v1/sys/mounts")

            _compose(
                project,
                port,
                "restart",
                "openbao",
                compose_override=compose_override,
            )
            _wait_for_status(address, {503}, label="restart-sealed")
            try:
                exercised["sources"]["identity"].read()
            except Exception as error:  # pragma: no cover - defensive redaction boundary
                raise RuntimeProofError(
                    "service credential file did not survive restart"
                ) from error
            _unseal(_Client(address), unseal_materials, phase="restart owner")
            _wait_for_status(address, {200}, label="restart-unsealed")
            identity_resolver = OpenBaoSecretResolver(address, exercised["sources"]["identity"])
            if (
                identity_resolver.resolve(
                    "openbao://kv/roehub/oidc/provider-a#client_secret"
                ).reveal_text()
                != exercised["canaries"]["oidc-v1"]
            ):
                raise RuntimeProofError("Raft state did not survive container restart")

            if (recovery_identity_path is None) != (recovery_recipient_path is None):
                raise RuntimeProofError("OpenBao recovery key pair is incomplete")
            if recovery_identity_path is None:
                recovery_path = temporary_root / "recovery.agekey"
                recipient_path = temporary_root / "recipient.txt"
                created = subprocess.run(
                    ["age-keygen", "--output", str(recovery_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=10,
                )
                if created.returncode != 0:
                    raise RuntimeProofError("age recovery identity generation failed")
                recovery_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                recipient = subprocess.run(
                    ["age-keygen", "--y", str(recovery_path)],
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
                if recipient.returncode != 0 or not recipient.stdout:
                    raise RuntimeProofError("age recipient derivation failed")
                recipient_path.write_bytes(recipient.stdout)
                recipient_path.chmod(
                    stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
                )
            else:
                if recovery_recipient_path is None:  # pragma: no cover - narrowed above
                    raise RuntimeProofError("OpenBao recovery key pair is incomplete")
                recovery_path = recovery_identity_path.expanduser().resolve()
                recipient_path = recovery_recipient_path.expanduser().resolve()
                if (
                    not recovery_path.is_file()
                    or recovery_path.is_symlink()
                    or recovery_path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO)
                    or not recipient_path.is_file()
                    or recipient_path.is_symlink()
                ):
                    raise RuntimeProofError("OpenBao recovery key pair is unsafe")

            backup_path = temporary_root / "openbao.snap.age"
            backup_result = backup_snapshot(
                address=address,
                credential_path=exercised["sources"]["backup"].path,
                recipient_path=recipient_path,
                destination=backup_path,
            )
            _forbidden_scan(
                [
                    backup_path.read_bytes(),
                    json.dumps(backup_result),
                    (backup_path.with_suffix(".age.metadata.json")).read_bytes(),
                ],
                forbidden,
            )

            logs_before_recovery = _compose(
                project,
                port,
                "logs",
                "--no-color",
                compose_override=compose_override,
                check=False,
            )
            audit_before_recovery = _compose(
                project,
                port,
                "exec",
                "--no-TTY",
                "openbao",
                "/bin/sh",
                "-c",
                "test ! -f /openbao/logs/audit.log || cat /openbao/logs/audit.log",
                compose_override=compose_override,
                check=False,
            )
            _forbidden_scan(
                [
                    logs_before_recovery.stdout,
                    logs_before_recovery.stderr,
                    audit_before_recovery.stdout,
                    audit_before_recovery.stderr,
                ],
                forbidden,
            )

            _compose(
                project,
                port,
                "down",
                "--volumes",
                "--remove-orphans",
                compose_override=compose_override,
            )
            _compose(
                project,
                port,
                "up",
                "--detach",
                compose_override=compose_override,
            )
            _wait_for_status(address, {501}, label="recovery-uninitialized")
            fresh_unseal, fresh_admin = _initialize_disposable(_Client(address))
            forbidden.extend((*fresh_unseal, fresh_admin))
            _unseal(_Client(address), fresh_unseal, phase="fresh recovery")
            _wait_for_status(address, {200}, label="recovery-bootstrap-unsealed")
            fresh_path = _write_credential(temporary_root / "fresh.credential", fresh_admin)
            restore_result = restore_snapshot(
                address=address,
                credential_path=fresh_path.path,
                recovery_path=recovery_path,
                source=backup_path,
                force_new_storage=True,
            )
            _wait_for_status(address, {503}, label="restored-sealed")
            _compose(
                project,
                port,
                "restart",
                "openbao",
                compose_override=compose_override,
            )
            _wait_for_status(address, {503}, label="restored-config-reloaded")
            _unseal(_Client(address), unseal_materials, phase="restored owner")
            _wait_for_status(address, {200}, label="restored-unsealed")
            restored_resolver = OpenBaoSecretResolver(address, exercised["sources"]["identity"])
            if (
                restored_resolver.resolve(
                    "openbao://kv/roehub/oidc/provider-a#client_secret"
                ).reveal_text()
                != exercised["canaries"]["oidc-v1"]
            ):
                raise RuntimeProofError("restored secret version is incorrect")
            _expect_denied(
                _Client(address, identities["api"]),
                "GET",
                "/v1/kv/data/roehub/oidc/provider-a",
            )
            if export_encrypted_backup is not None:
                export_path = export_encrypted_backup.expanduser()
                if not export_path.is_absolute() or os.path.lexists(export_path):
                    raise RuntimeProofError("encrypted backup export path is unsafe")
                export_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(backup_path, export_path)
                export_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                export_metadata = export_path.with_suffix(
                    export_path.suffix + ".metadata.json"
                )
                shutil.copyfile(
                    backup_path.with_suffix(backup_path.suffix + ".metadata.json"),
                    export_metadata,
                )
                export_metadata.chmod(stat.S_IRUSR | stat.S_IWUSR)

            final_logs = _compose(
                project,
                port,
                "logs",
                "--no-color",
                compose_override=compose_override,
                check=False,
            )
            safe_result = {
                "schema": RESULT_SCHEMA,
                "status": "passed",
                "image_digest_pinned": True,
                "compose_config": "passed",
                "initialization_states": ["uninitialized", "sealed", "unsealed"],
                "production_pgp_bootstrap": "passed",
                "production_unseal_shares": 3,
                "production_unseal_threshold": 2,
                "initial_admin_pgp_encrypted": True,
                "raft_restart_persistence": "passed",
                "service_identities": len(identities),
                "approle_response_wrapping": (
                    "passed" if approle_proof["response_wrapped"] else "failed"
                ),
                "approle_one_time_unwrap": (
                    "passed" if approle_proof["one_time_unwrap"] else "failed"
                ),
                "wrapped_file_delivery": (
                    "passed" if approle_proof["mode_0600_delivery"] else "failed"
                ),
                "service_token_renewal": ("passed" if approle_proof["token_renewal"] else "failed"),
                "bootstrap_credential_revoked": "passed",
                "shared_broad_tokens": False,
                "api_value_access": "denied",
                "api_transit_decrypt": "denied",
                "typed_kinds": [item.value for item in SecretKind],
                "version_rotation": exercised["rotation_versions"],
                "live_reference_rotation": "passed",
                "soft_delete_undelete": "passed",
                "rollback_new_version": "passed",
                "encrypted_backup": backup_result["encrypted"],
                "backup_source_digest_verified": restore_result["source_digest_verified"],
                "fresh_volume_force_restore": restore_result["status"],
                "fresh_storage_guard": restore_result["fresh_storage_guard"],
                "restored_config_reload": "passed",
                "ops_status": exercised["ops_status"],
                "forbidden_output_scan": "passed",
                "cleanup": "pending",
            }
            _forbidden_scan(
                [final_logs.stdout, final_logs.stderr, json.dumps(safe_result)], forbidden
            )
        except RuntimeProofError as error:
            diagnostic_logs = _compose(
                project,
                port,
                "logs",
                "--no-color",
                compose_override=compose_override,
                check=False,
            )
            normalized_logs = (diagnostic_logs.stdout + diagnostic_logs.stderr).lower()
            diagnostic_signals = [
                label
                for label, marker in (
                    ("raft_join", "join raft"),
                    ("raft_unseal_wait", "waiting for unseal keys"),
                    ("seal_migration", "entering seal migration mode"),
                    ("barrier_initialized", "security barrier initialized"),
                    ("stored_keys_forced", "forcing shares/threshold to 1"),
                    ("threshold_2_init", "shares=3 threshold=2"),
                    ("threshold_1_init", "shares=1 threshold=1"),
                )
                if marker in normalized_logs
            ]
            initialized_match = re.search(
                r"security barrier initialized[^\n]*" r"(stored=\d+ shares=\d+ threshold=\d+)",
                normalized_logs,
            )
            if initialized_match is not None:
                diagnostic_signals.append(initialized_match.group(1).replace(" ", "_"))
            context = ",".join(diagnostic_signals) or "none"
            raise RuntimeProofError(f"{error}; server_context={context}") from error
        finally:
            _cleanup_project(project, port, compose_override=compose_override)
    if safe_result is None:
        raise RuntimeProofError("OpenBao runtime proof did not produce a result")
    safe_result["cleanup"] = "passed"
    return safe_result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-encrypted-backup", type=Path)
    parser.add_argument("--compose-override", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(
            export_encrypted_backup=args.export_encrypted_backup,
            compose_override=args.compose_override,
        )
    except RuntimeProofError as error:
        print(
            json.dumps(
                {"schema": RESULT_SCHEMA, "status": "failed", "reason": str(error)},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
