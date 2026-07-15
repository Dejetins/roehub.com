"""The sole product adapter permitted to invoke Docker Engine and Compose."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from trading.contexts.operations import (
    ControlOperationError,
    OperationAction,
    OperationRequest,
    OperationResult,
    OperationState,
)

_PROJECT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_MUTATING = {
    OperationAction.START,
    OperationAction.STOP,
    OperationAction.RESTART,
    OperationAction.RECOVER,
    OperationAction.INSTALL,
    OperationAction.UPDATE,
    OperationAction.ROLLBACK,
}
_RELEASE_ACTIONS = {
    OperationAction.INSTALL,
    OperationAction.UPDATE,
    OperationAction.ROLLBACK,
}


@dataclass(frozen=True, slots=True)
class _PolicySnapshot:
    policy: dict[str, Any]
    compose: dict[str, Any]
    compose_bytes: bytes
    compose_sha256: str
    policy_sha256: str
    release_manifest_sha256: str


class _EffectReceiptStore:
    """Persist proof written only after a Docker effect completed successfully."""

    def __init__(self, root: Path) -> None:
        self._root = _prepare_private_directory(root, code="control_agent.receipts_unsafe")

    def record(
        self,
        *,
        request: OperationRequest,
        snapshot: _PolicySnapshot,
        fingerprints: Mapping[str, Mapping[str, object]],
        release_before: str | None,
        release_after: str | None,
    ) -> None:
        payload = {
            "schema": "io.roehub.control-effect-receipt/v1alpha1",
            "operation_id": str(request.operation_id),
            "request_digest": request.request_digest,
            "action": request.action.value,
            "profile": request.profile,
            "compose_sha256": snapshot.compose_sha256,
            "policy_sha256": snapshot.policy_sha256,
            "release_manifest_sha256": snapshot.release_manifest_sha256,
            "release_before": release_before,
            "release_after": release_after,
            "fingerprints": {
                key: dict(value) for key, value in sorted(fingerprints.items())
            },
        }
        encoded = _json_bytes(payload)
        target = self._root / f"{request.operation_id}.json"
        try:
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            if _read_secure_bytes(target, code="control_agent.receipt_corrupt") != encoded:
                raise ControlOperationError(code="control_agent.receipt_conflict")
            return
        except OSError as error:
            raise ControlOperationError(code="control_agent.receipt_unavailable") from error
        try:
            _write_all(descriptor, encoded, code="control_agent.receipt_unavailable")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self._root)

    def read(self, request: OperationRequest) -> dict[str, Any] | None:
        target = self._root / f"{request.operation_id}.json"
        if not target.exists():
            return None
        try:
            payload = json.loads(
                _read_secure_bytes(target, code="control_agent.receipt_corrupt")
            )
        except json.JSONDecodeError as error:
            raise ControlOperationError(code="control_agent.receipt_corrupt") from error
        expected = (
            isinstance(payload, dict)
            and payload.get("schema") == "io.roehub.control-effect-receipt/v1alpha1"
            and payload.get("operation_id") == str(request.operation_id)
            and payload.get("request_digest") == request.request_digest
            and payload.get("action") == request.action.value
            and payload.get("profile") == request.profile
            and isinstance(payload.get("fingerprints"), dict)
        )
        if not expected:
            raise ControlOperationError(code="control_agent.receipt_corrupt")
        return payload


class _ReleaseStateStore:
    """Track the installed release version with an atomic owner-local file."""

    def __init__(self, path: Path) -> None:
        self._path = path.expanduser().resolve()
        _prepare_private_directory(
            self._path.parent,
            code="control_agent.release_state_unsafe",
        )

    def current(self) -> str | None:
        if not self._path.exists():
            return None
        try:
            payload = json.loads(
                _read_secure_bytes(self._path, code="control_agent.release_state_corrupt")
            )
        except json.JSONDecodeError as error:
            raise ControlOperationError(code="control_agent.release_state_corrupt") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "io.roehub.installed-release/v1alpha1"
            or not isinstance(payload.get("version"), str)
            or _SEMVER.fullmatch(payload["version"]) is None
        ):
            raise ControlOperationError(code="control_agent.release_state_corrupt")
        return str(payload["version"])

    def validate_transition(self, *, action: OperationAction, target: str) -> str | None:
        current = self.current()
        if _SEMVER.fullmatch(target) is None:
            raise ControlOperationError(code="control_agent.release_rejected")
        if action == OperationAction.INSTALL:
            valid = current is None
        elif action == OperationAction.UPDATE:
            valid = current is not None and _semver_key(target) > _semver_key(current)
        elif action == OperationAction.ROLLBACK:
            valid = current is not None and _semver_key(target) < _semver_key(current)
        else:
            raise ControlOperationError(code="control_agent.release_transition_invalid")
        if not valid:
            raise ControlOperationError(code="control_agent.release_transition_invalid")
        return current

    def write(self, version: str) -> None:
        payload = _json_bytes(
            {
                "schema": "io.roehub.installed-release/v1alpha1",
                "version": version,
            }
        )
        temporary = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                _write_all(
                    descriptor,
                    payload,
                    code="control_agent.release_state_unavailable",
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, self._path)
            _fsync_directory(self._path.parent)
        except OSError as error:
            raise ControlOperationError(code="control_agent.release_state_unavailable") from error
        finally:
            temporary.unlink(missing_ok=True)


class DockerComposeControlBackend:
    """Map typed requests to fixed Compose argv using an owner-protected bundle snapshot."""

    def __init__(
        self,
        *,
        profile_root: Path,
        project: str,
        trusted_release_manifest: Path,
        effect_receipt_dir: Path | None = None,
        release_state_path: Path | None = None,
        docker_binary: str = "docker",
        timeout_seconds: float = 240.0,
    ) -> None:
        if _PROJECT.fullmatch(project) is None or timeout_seconds <= 0:
            raise ControlOperationError(code="control_agent.configuration_invalid")
        root = profile_root.expanduser()
        if root.is_symlink():
            raise ControlOperationError(code="control_agent.policy_unsafe")
        self._root = root.resolve()
        self._compose = self._root / "compose.yaml"
        self._policy_path = self._root / "control-policy.json"
        self._generation_path = self._root / "generation-manifest.json"
        self._trusted_release_manifest = trusted_release_manifest.expanduser().resolve()
        self._project = project
        self._docker_binary = docker_binary
        self._timeout = timeout_seconds
        self._receipts = (
            _EffectReceiptStore(effect_receipt_dir)
            if effect_receipt_dir is not None
            else None
        )
        self._release_state = (
            _ReleaseStateStore(release_state_path)
            if release_state_path is not None
            else None
        )
        self._load_and_validate_policy()

    def current_release(self) -> str | None:
        """Return the owner-local installed release state without exposing its path."""

        return self._release_state.current() if self._release_state is not None else None

    def execute(self, request: OperationRequest) -> OperationResult:
        snapshot = self._load_and_validate_policy()
        self._validate_request(request, snapshot.policy)
        release_before: str | None = None
        release_after: str | None = None
        if request.action in _RELEASE_ACTIONS:
            if self._release_state is None or request.release_version is None:
                raise ControlOperationError(code="control_agent.release_handler_unavailable")
            release_before = self._release_state.validate_transition(
                action=request.action,
                target=request.release_version,
            )
            release_after = request.release_version
        execution_compose = snapshot.compose_bytes
        if request.action in _MUTATING:
            if self._receipts is None:
                raise ControlOperationError(code="control_agent.receipt_store_unavailable")
            image_ids = self._validate_installed_images(snapshot.policy)
            execution_compose = self._execution_compose_bytes(snapshot, image_ids)
        if request.action in {OperationAction.INSPECT, OperationAction.DIAGNOSTICS}:
            active = self._active_services(snapshot.compose_bytes)
            expected = set(snapshot.policy["default_services"])
            detail = "topology.ready" if expected.issubset(active) else "topology.degraded"
            return self._result(request, detail_code=detail, active=active)

        if request.action == OperationAction.STOP:
            self._run_compose(execution_compose, "stop", *request.services, effect=True)
            detail = "topology.stopped"
        elif request.action == OperationAction.RESTART:
            self._run_compose(execution_compose, "restart", *request.services, effect=True)
            self._run_compose(
                execution_compose,
                "up",
                "-d",
                "--no-build",
                "--pull",
                "never",
                "--no-deps",
                "--wait",
                "--wait-timeout",
                "180",
                *request.services,
                effect=True,
            )
            detail = "topology.restarted"
        elif request.action in {
            OperationAction.START,
            OperationAction.RECOVER,
            OperationAction.INSTALL,
            OperationAction.UPDATE,
            OperationAction.ROLLBACK,
        }:
            args = ["up", "-d", "--no-build", "--pull", "never"]
            if not request.services:
                args.extend(["--wait", "--wait-timeout", "180"])
            args.extend(request.services)
            self._run_compose(execution_compose, *args, effect=True)
            detail = {
                OperationAction.START: "topology.started",
                OperationAction.RECOVER: "topology.recovered",
                OperationAction.INSTALL: "release.installed",
                OperationAction.UPDATE: "release.updated",
                OperationAction.ROLLBACK: "release.rolled_back",
            }[request.action]
        else:
            raise ControlOperationError(code="operation.handler_unavailable")

        if release_after is not None:
            assert self._release_state is not None
            self._release_state.write(release_after)
        active = self._active_services(snapshot.compose_bytes)
        fingerprints = self._service_fingerprints(snapshot.compose_bytes)
        assert self._receipts is not None
        self._receipts.record(
            request=request,
            snapshot=snapshot,
            fingerprints=fingerprints,
            release_before=release_before,
            release_after=release_after,
        )
        return self._result(request, detail_code=detail, active=active)

    def reconcile(self, request: OperationRequest) -> OperationResult:
        snapshot = self._load_and_validate_policy()
        self._validate_request(request, snapshot.policy)
        active = self._active_services(snapshot.compose_bytes)
        targets = set(request.services or snapshot.policy["default_services"])
        receipt = self._receipts.read(request) if self._receipts is not None else None
        complete = receipt is not None
        if complete and receipt is not None:
            complete = (
                receipt.get("compose_sha256") == snapshot.compose_sha256
                and receipt.get("policy_sha256") == snapshot.policy_sha256
                and receipt.get("release_manifest_sha256")
                == snapshot.release_manifest_sha256
            )
        if complete and request.action == OperationAction.STOP:
            complete = not (targets & active)
        elif complete and request.action in _MUTATING:
            fingerprints = self._service_fingerprints(snapshot.compose_bytes)
            expected_fingerprints = receipt.get("fingerprints", {}) if receipt else {}
            complete = targets.issubset(active) and all(
                fingerprints.get(service) == expected_fingerprints.get(service)
                and fingerprints.get(service, {}).get("health") == "healthy"
                for service in targets
            )
        if complete and request.action in _RELEASE_ACTIONS:
            complete = (
                self._release_state is not None
                and self._release_state.current() == request.release_version
                and receipt is not None
                and receipt.get("release_after") == request.release_version
            )
        if request.action not in _MUTATING:
            complete = True
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.SUCCEEDED if complete else OperationState.UNKNOWN,
            detail_code="operation.reconciled" if complete else "operation.effect_unknown",
            active_services=tuple(sorted(active)),
        )

    def _validate_request(
        self,
        request: OperationRequest,
        policy: Mapping[str, Any],
    ) -> None:
        if request.profile != policy["profile"]:
            raise ControlOperationError(code="control_agent.profile_rejected")
        allowed = set(policy["allowed_services"])
        if not set(request.services).issubset(allowed):
            raise ControlOperationError(code="control_agent.service_rejected")
        if request.action == OperationAction.RESTART:
            restartable = {
                name
                for name, spec in policy["services"].items()
                if isinstance(spec, dict) and spec.get("restart_allowed") is True
            }
            if not request.services or not set(request.services).issubset(restartable):
                raise ControlOperationError(code="control_agent.restart_rejected")
        if request.release_version is not None and (
            request.release_version != policy["release_version"]
        ):
            raise ControlOperationError(code="control_agent.release_rejected")

    def _load_and_validate_policy(self) -> _PolicySnapshot:
        try:
            compose_bytes = _read_secure_bytes(
                self._compose,
                code="control_agent.policy_unsafe",
            )
            policy_bytes = _read_secure_bytes(
                self._policy_path,
                code="control_agent.policy_unsafe",
            )
            generation_bytes = _read_secure_bytes(
                self._generation_path,
                code="control_agent.policy_unsafe",
            )
            release_bytes = _read_secure_bytes(
                self._trusted_release_manifest,
                code="control_agent.release_manifest_unsafe",
            )
            policy = json.loads(policy_bytes)
            generation = json.loads(generation_bytes)
            release_manifest = json.loads(release_bytes)
            compose = yaml.safe_load(compose_bytes)
        except (json.JSONDecodeError, yaml.YAMLError) as error:
            raise ControlOperationError(code="control_agent.policy_invalid") from error
        if not all(
            isinstance(item, dict)
            for item in (policy, generation, release_manifest, compose)
        ):
            raise ControlOperationError(code="control_agent.policy_invalid")
        if policy.get("schema") != "io.roehub.control-policy/v1alpha1":
            raise ControlOperationError(code="control_agent.policy_schema_rejected")
        if release_manifest.get("schema") != "io.roehub.release/v1alpha1":
            raise ControlOperationError(code="control_agent.release_manifest_rejected")
        release_sha = hashlib.sha256(release_bytes).hexdigest()
        inputs = generation.get("inputs")
        outputs = generation.get("outputs")
        if (
            not isinstance(inputs, dict)
            or inputs.get("release_manifest_sha256") != release_sha
            or not isinstance(outputs, dict)
        ):
            raise ControlOperationError(code="control_agent.release_manifest_unbound")
        if policy.get("release_version") != release_manifest.get("version"):
            raise ControlOperationError(code="control_agent.release_version_mismatch")
        compose_sha = hashlib.sha256(compose_bytes).hexdigest()
        policy_sha = hashlib.sha256(policy_bytes).hexdigest()
        for name, actual_hash in {
            "compose.yaml": compose_sha,
            "control-policy.json": policy_sha,
        }.items():
            row = outputs.get(name)
            expected_hash = row.get("sha256") if isinstance(row, dict) else None
            if expected_hash != actual_hash:
                raise ControlOperationError(code="control_agent.policy_hash_mismatch")
        services = compose.get("services")
        policy_services = policy.get("services")
        images = release_manifest.get("images")
        release_references = {
            row.get("reference")
            for row in images.values()
            if isinstance(images, dict) and isinstance(row, dict)
        } if isinstance(images, dict) else set()
        if not isinstance(services, dict) or not isinstance(policy_services, dict):
            raise ControlOperationError(code="control_agent.policy_invalid")
        if set(services) != set(policy_services):
            raise ControlOperationError(code="control_agent.service_set_mismatch")
        if not set(policy.get("default_services", ())).issubset(policy_services):
            raise ControlOperationError(code="control_agent.policy_invalid")
        for name, expected in policy_services.items():
            actual = services.get(name)
            if not isinstance(expected, dict) or not isinstance(actual, dict):
                raise ControlOperationError(code="control_agent.service_policy_invalid")
            mounts = actual.get("volumes", [])
            if any("docker.sock" in str(item) for item in mounts):
                raise ControlOperationError(code="control_agent.docker_socket_mount_rejected")
            bind_mounts = [str(item) for item in mounts if str(item).startswith(("./", "../"))]
            if any(not item.endswith(":ro") for item in bind_mounts):
                raise ControlOperationError(code="control_agent.mount_rejected")
            actual_spec = {
                "image": actual.get("image"),
                "mounts": mounts,
                "environment_names": sorted((actual.get("environment") or {}).keys()),
                "resources": actual.get("deploy", {}).get("resources", {}).get("limits", {}),
                "command_sha256": hashlib.sha256(
                    json.dumps(actual.get("command", []), separators=(",", ":")).encode()
                ).hexdigest(),
                "release_reference": expected.get("release_reference"),
                "restart_allowed": expected.get("restart_allowed"),
            }
            if actual_spec != expected:
                raise ControlOperationError(code="control_agent.service_policy_mismatch")
            if not isinstance(expected.get("restart_allowed"), bool):
                raise ControlOperationError(code="control_agent.service_policy_invalid")
            image = expected.get("image")
            release_reference = expected.get("release_reference")
            if not isinstance(image, str) or not isinstance(release_reference, str):
                raise ControlOperationError(code="control_agent.image_policy_invalid")
            if release_reference not in release_references:
                raise ControlOperationError(code="control_agent.image_not_in_release")
            if "@sha256:" in image and image != release_reference:
                raise ControlOperationError(code="control_agent.image_rejected")
        return _PolicySnapshot(
            policy=policy,
            compose=compose,
            compose_bytes=compose_bytes,
            compose_sha256=compose_sha,
            policy_sha256=policy_sha,
            release_manifest_sha256=release_sha,
        )

    def _validate_installed_images(
        self,
        policy: Mapping[str, Any],
    ) -> dict[str, str]:
        validated: dict[str, str] = {}
        for spec in policy["services"].values():
            image = str(spec["image"])
            release_reference = str(spec["release_reference"])
            if image in validated:
                continue
            if "@sha256:" in image:
                completed = self._run(
                    [
                        self._docker_binary,
                        "image",
                        "inspect",
                        image,
                        "--format",
                        "{{json .RepoDigests}}",
                    ]
                )
                try:
                    repo_digests = json.loads(completed.stdout)
                except json.JSONDecodeError as error:
                    raise ControlOperationError(
                        code="control_agent.image_digest_mismatch"
                    ) from error
                expected_digest = release_reference.rsplit("@", 1)[1]
                if not isinstance(repo_digests, list) or not any(
                    isinstance(item, str)
                    and item.rsplit("@", 1)[-1] == expected_digest
                    for item in repo_digests
                ):
                    raise ControlOperationError(code="control_agent.image_digest_mismatch")
                validated[image] = release_reference
            else:
                completed = self._run(
                    [self._docker_binary, "image", "inspect", image, "--format", "{{.Id}}"]
                )
                image_id = completed.stdout.strip()
                expected_digest = release_reference.rsplit("@", 1)[1]
                if image_id != expected_digest:
                    raise ControlOperationError(code="control_agent.image_digest_mismatch")
                validated[image] = image_id
        return validated

    @staticmethod
    def _execution_compose_bytes(
        snapshot: _PolicySnapshot,
        image_ids: Mapping[str, str],
    ) -> bytes:
        compose = json.loads(json.dumps(snapshot.compose))
        for service in compose["services"].values():
            image = str(service["image"])
            service["image"] = image_ids[image]
            service.pop("build", None)
        return yaml.safe_dump(compose, sort_keys=True).encode("utf-8")

    def _active_services(self, compose_bytes: bytes) -> set[str]:
        completed = self._run_compose(
            compose_bytes,
            "ps",
            "--services",
            "--status",
            "running",
        )
        return {line.strip() for line in completed.stdout.splitlines() if line.strip()}

    def _service_fingerprints(
        self,
        compose_bytes: bytes,
    ) -> dict[str, dict[str, object]]:
        completed = self._run_compose(compose_bytes, "ps", "-q")
        identifiers = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
        if not identifiers:
            return {}
        inspected = self._run(
            [self._docker_binary, "container", "inspect", *identifiers]
        )
        try:
            payload = json.loads(inspected.stdout)
        except json.JSONDecodeError as error:
            raise ControlOperationError(code="control_agent.docker_response_invalid") from error
        if not isinstance(payload, list):
            raise ControlOperationError(code="control_agent.docker_response_invalid")
        fingerprints: dict[str, dict[str, object]] = {}
        for row in payload:
            config = row.get("Config") if isinstance(row, dict) else None
            state = row.get("State") if isinstance(row, dict) else None
            labels = config.get("Labels") if isinstance(config, dict) else None
            service = labels.get("com.docker.compose.service") if isinstance(labels, dict) else None
            if not isinstance(service, str) or not isinstance(state, dict):
                raise ControlOperationError(code="control_agent.docker_response_invalid")
            fingerprints[service] = {
                "container_id": row.get("Id"),
                "image_id": row.get("Image"),
                "started_at": state.get("StartedAt"),
                "running": state.get("Running") is True,
                "health": (
                    state.get("Health", {}).get("Status")
                    if isinstance(state.get("Health"), dict)
                    else None
                ),
            }
        return fingerprints

    def _run_compose(
        self,
        compose_bytes: bytes,
        *args: str,
        effect: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return self._run(
            [
                self._docker_binary,
                "compose",
                "-p",
                self._project,
                "--project-directory",
                str(self._root),
                "-f",
                "-",
                *args,
            ],
            effect=effect,
            input_bytes=compose_bytes,
        )

    def _run(
        self,
        command: Sequence[str],
        *,
        effect: bool = False,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                list(command),
                cwd=self._root,
                input=input_bytes.decode("utf-8") if input_bytes is not None else None,
                text=True,
                capture_output=True,
                check=False,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise ControlOperationError(
                code="operation.effect_unknown" if effect else "control_agent.docker_timeout"
            ) from error
        except OSError as error:
            raise ControlOperationError(code="control_agent.docker_unavailable") from error
        if completed.returncode != 0:
            raise ControlOperationError(
                code=(
                    "operation.effect_unknown"
                    if effect
                    else "control_agent.docker_operation_failed"
                )
            )
        return completed

    @staticmethod
    def _result(
        request: OperationRequest,
        *,
        detail_code: str,
        active: Iterable[str],
    ) -> OperationResult:
        return OperationResult(
            operation_id=request.operation_id,
            action=request.action,
            profile=request.profile,
            state=OperationState.SUCCEEDED,
            detail_code=detail_code,
            active_services=tuple(sorted(active)),
        )


def _semver_key(value: str) -> tuple[int, int, int]:
    match = _SEMVER.fullmatch(value)
    if match is None:
        raise ControlOperationError(code="control_agent.release_rejected")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _prepare_private_directory(path: Path, *, code: str) -> Path:
    candidate = path.expanduser()
    if candidate.exists() and candidate.is_symlink():
        raise ControlOperationError(code=code)
    resolved = candidate.resolve()
    resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved.chmod(0o700)
    info = resolved.stat()
    if info.st_uid not in {0, os.geteuid()} or stat.S_IMODE(info.st_mode) & 0o077:
        raise ControlOperationError(code=code)
    return resolved


def _read_secure_bytes(path: Path, *, code: str) -> bytes:
    candidate = path.expanduser()
    _validate_path_chain(candidate, code=code)
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ControlOperationError(code=code) from error
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid not in {0, os.geteuid()}
            or stat.S_IMODE(info.st_mode) & 0o022
        ):
            raise ControlOperationError(code=code)
        return os.read(descriptor, info.st_size + 1)
    finally:
        os.close(descriptor)


def _validate_path_chain(path: Path, *, code: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as error:
            raise ControlOperationError(code=code) from error
        if stat.S_ISLNK(info.st_mode):
            raise ControlOperationError(code=code)
        if not _path_component_is_trusted(info):
            raise ControlOperationError(code=code)


def _path_component_is_trusted(info: os.stat_result) -> bool:
    if info.st_uid not in {0, os.geteuid()}:
        return False
    if not stat.S_IMODE(info.st_mode) & 0o022:
        return True
    return (
        info.st_uid == 0
        and stat.S_ISDIR(info.st_mode)
        and bool(info.st_mode & stat.S_ISVTX)
    )


def _write_all(descriptor: int, payload: bytes, *, code: str) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise ControlOperationError(code=code)
        offset += written


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["DockerComposeControlBackend"]
