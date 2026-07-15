"""Emergency-first Roehub host CLI."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from apps.cli.main.main import main as domain_cli_main
from apps.control_agent.auth import read_private_credential
from apps.control_agent.docker_backend import DockerComposeControlBackend
from trading.contexts.operations import (
    ControlOperationError,
    OperationAction,
    OperationRequest,
)
from trading.contexts.operations.adapters import ControlAgentUnixClient

_DEFAULT_SOCKET = "/var/run/roehub/control-agent.sock"
_DEFAULT_IDENTITY_FILE = "/etc/roehub/control-agent-owner.credential"
_DOMAIN_COMMANDS = {
    "artifacts",
    "backfill-1m",
    "backtest-artifact-publish",
    "funding-rate-catchup",
    "local-auth-bootstrap",
    "plugins",
    "providers",
    "rest-catchup",
    "sync-instruments",
    "telegram",
}


def _parser(environ: Mapping[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roehubctl")
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path(environ.get("ROEHUB_CONTROL_AGENT_SOCKET", _DEFAULT_SOCKET)),
    )
    parser.add_argument(
        "--identity-file",
        type=Path,
        default=Path(
            environ.get("ROEHUB_CONTROL_AGENT_OWNER_IDENTITY_FILE", _DEFAULT_IDENTITY_FILE)
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-config")
    validate.add_argument("--profile-root", type=Path, required=True)
    validate.add_argument("--trusted-release-manifest", type=Path, required=True)
    validate.add_argument("--project", default="roehub-validation")
    effective = commands.add_parser("effective")
    effective.add_argument("--path", type=Path, required=True)
    for name in ("doctor", "inspect"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=("base", "trading", "ml"), default="base")
    for name in ("start", "stop", "restart", "recover"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=("base", "trading", "ml"), default="base")
        command.add_argument("--service", action="append", default=[])
        command.add_argument("--operation-id", type=UUID)
    for name in ("install", "update", "rollback"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=("base", "trading", "ml"), default="base")
        command.add_argument("--release-version", required=True)
        command.add_argument("--operation-id", type=UUID)
    for name in ("backup", "restore", "backup-cancel", "restore-cancel"):
        command = commands.add_parser(name)
        command.add_argument("--profile", choices=("base", "trading", "ml"), default="base")
        command.add_argument("--subject-id", required=True)
        command.add_argument("--operation-id", type=UUID)
    owner = commands.add_parser("owner")
    owner.add_argument("action", choices=("init",))
    owner.add_argument("args", nargs=argparse.REMAINDER)
    for name in sorted(_DOMAIN_COMMANDS):
        command = commands.add_parser(name)
        command.add_argument("args", nargs=argparse.REMAINDER)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    values = os.environ if environ is None else environ
    parser = _parser(values)
    args = parser.parse_args(argv)
    if args.command == "validate-config":
        try:
            DockerComposeControlBackend(
                profile_root=args.profile_root,
                project=args.project,
                trusted_release_manifest=args.trusted_release_manifest,
            )
        except ControlOperationError as error:
            return _print_error(error.code)
        return _print_payload(
            {
                "schema": "io.roehub.config-validation/v1alpha1",
                "status": "passed",
                "profile_root": str(args.profile_root.expanduser().resolve()),
            }
        )
    if args.command == "effective":
        candidate = args.path.expanduser()
        if (
            candidate.name != "effective-config.redacted.json"
            or candidate.is_symlink()
        ):
            return _print_error("roehubctl.effective_config_path_rejected")
        try:
            payload = json.loads(candidate.resolve().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return _print_error("roehubctl.effective_config_invalid")
        redacted = _redact(payload)
        try:
            _assert_no_residual_secret(redacted)
        except ValueError:
            return _print_error("roehubctl.effective_config_redaction_failed")
        return _print_payload(redacted)
    if args.command == "owner":
        return domain_cli_main(["local-auth-bootstrap", *args.args])
    if args.command in _DOMAIN_COMMANDS:
        return domain_cli_main([args.command, *args.args])
    try:
        client = ControlAgentUnixClient(
            socket_path=args.socket,
            identity="installation_owner",
            identity_key=read_private_credential(args.identity_file),
            timeout_seconds=300.0,
        )
        action = {
            "doctor": OperationAction.DIAGNOSTICS,
            "inspect": OperationAction.INSPECT,
            "start": OperationAction.START,
            "stop": OperationAction.STOP,
            "restart": OperationAction.RESTART,
            "recover": OperationAction.RECOVER,
            "install": OperationAction.INSTALL,
            "update": OperationAction.UPDATE,
            "rollback": OperationAction.ROLLBACK,
            "backup": OperationAction.BACKUP,
            "restore": OperationAction.RESTORE,
            "backup-cancel": OperationAction.BACKUP_CANCEL,
            "restore-cancel": OperationAction.RESTORE_CANCEL,
        }[args.command]
        request = OperationRequest(
            operation_id=getattr(args, "operation_id", None) or uuid4(),
            action=action,
            profile=args.profile,
            services=tuple(getattr(args, "service", [])),
            release_version=getattr(args, "release_version", None),
            subject_id=getattr(args, "subject_id", None),
        )
        result = client.submit(request)
    except ControlOperationError as error:
        return _print_error(error.code)
    return _print_payload(result.model_dump(mode="json", by_alias=True, exclude_none=True))


def _redact(value: Any, *, key: str = "") -> Any:
    lowered = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    markers = (
        "api_key",
        "authorization",
        "ciphertext",
        "cookie",
        "credential",
        "dsn",
        "hmac",
        "password",
        "private_key",
        "secret",
        "session",
        "signature",
        "token",
    )
    if any(marker in lowered for marker in markers):
        return "[redacted]"
    if isinstance(value, dict):
        return {str(item_key): _redact(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, key=key) for item in value]
    return value


def _assert_no_residual_secret(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_residual_secret(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_no_residual_secret(item)
        return
    if not isinstance(value, str) or value == "[redacted]":
        return
    unsafe = (
        re.search(r"-----BEGIN [A-Z ]+PRIVATE KEY-----", value),
        re.search(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}", value),
        re.search(r"(?i)\b(?:postgres(?:ql)?|redis)://[^\s:/]+:[^\s@]+@", value),
    )
    if any(unsafe):
        raise ValueError("residual secret-shaped value")


def _print_error(code: str) -> int:
    _print_payload(
        {
            "schema": "io.roehub.roehubctl-error/v1alpha1",
            "status": "failed",
            "code": code,
        }
    )
    return 2


def _print_payload(payload: object) -> int:
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
