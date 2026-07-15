"""Host-service entrypoint for control-agent."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from apps.control_agent.auth import ServiceIdentityAuthorizer
from apps.control_agent.backup_backend import (
    InstallationBackupControlBackend,
    RecoveryControlBackend,
)
from apps.control_agent.docker_backend import DockerComposeControlBackend
from apps.control_agent.job_rpc import (
    start_job_control_server,
    stop_job_control_server,
)
from apps.control_agent.job_runtime_backend import ControlAgentJobDockerRunner
from apps.control_agent.server import serve
from trading.contexts.operations import ControlOperationService
from trading.contexts.operations.adapters import AppendOnlyOperationJournal


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roehub-control-agent")
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--trusted-release-manifest", type=Path, required=True)
    parser.add_argument("--profile", choices=("base", "trading", "ml"), required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--job-socket", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--api-token-file", type=Path, required=True)
    parser.add_argument("--owner-token-file", type=Path, required=True)
    parser.add_argument("--job-token-file", type=Path, required=True)
    parser.add_argument("--backup-policy", type=Path)
    args = parser.parse_args(argv)
    runtime_backend = DockerComposeControlBackend(
        profile_root=args.profile_root,
        project=args.project,
        trusted_release_manifest=args.trusted_release_manifest,
        effect_receipt_dir=args.journal.with_suffix(args.journal.suffix + ".effects"),
        release_state_path=args.journal.with_suffix(args.journal.suffix + ".release-state"),
    )
    backend = runtime_backend
    before_lock = None
    if args.backup_policy is not None:
        backup_backend = InstallationBackupControlBackend(
            policy_path=args.backup_policy,
            receipt_root=args.journal.with_suffix(args.journal.suffix + ".backup-receipts"),
        )
        backend = RecoveryControlBackend(
            runtime_backend=runtime_backend,
            backup_backend=backup_backend,
            current_release=runtime_backend.current_release,
        )
        before_lock = backend.request_cancellation
    journal = AppendOnlyOperationJournal(path=args.journal)
    service = ControlOperationService(
        backend=backend,
        journal=journal,
        before_lock=before_lock,
    )
    authorizer = ServiceIdentityAuthorizer(
        api_token_file=args.api_token_file,
        owner_token_file=args.owner_token_file,
        job_token_file=args.job_token_file,
        replay_state_dir=args.journal.with_suffix(args.journal.suffix + ".auth-replay"),
    )
    job_server, job_thread = start_job_control_server(
        socket_path=args.job_socket,
        runner=ControlAgentJobDockerRunner(),
        authorizer=authorizer,
    )
    try:
        serve(
            socket_path=args.socket,
            service=service,
            journal=journal,
            authorizer=authorizer,
        )
    finally:
        stop_job_control_server(
            server=job_server,
            thread=job_thread,
            socket_path=args.job_socket,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
