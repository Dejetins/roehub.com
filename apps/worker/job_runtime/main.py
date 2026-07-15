from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from apps.control_agent.auth import read_private_credential
from apps.worker.job_runtime.control_agent_client import ControlAgentJobUnixClient
from apps.worker.job_runtime.executor import JobAttemptExecutor
from apps.worker.job_runtime.oci_runner import OciJobRunner


def _parser(environ: Mapping[str, str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="job-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--artifact-root", type=Path, required=True)
    doctor.add_argument(
        "--control-agent-job-socket",
        type=Path,
        default=Path(
            environ.get(
                "ROEHUB_CONTROL_AGENT_JOB_SOCKET",
                "/run/roehub-control-agent/job-control.sock",
            )
        ),
    )
    doctor.add_argument(
        "--control-agent-job-identity-file",
        type=Path,
        default=Path(
            environ.get(
                "ROEHUB_CONTROL_AGENT_JOB_IDENTITY_FILE",
                "/run/roehub-control-agent/job-runtime.identity",
            )
        ),
    )
    return parser


def _doctor(
    artifact_root: Path,
    *,
    control_socket: Path,
    identity_file: Path,
) -> int:
    root = artifact_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    sentinel = root / ".job-runtime-doctor"
    sentinel.write_text("packaged\n", encoding="utf-8")
    if sentinel.read_text(encoding="utf-8") != "packaged\n":
        raise RuntimeError("job runtime artifact volume is not writable")
    sentinel.unlink()
    if not callable(OciJobRunner) or not callable(JobAttemptExecutor.execute):
        raise RuntimeError("job runtime executor is unavailable")
    if control_socket.is_socket() and identity_file.is_file():
        client = ControlAgentJobUnixClient(
            socket_path=control_socket,
            identity_key=read_private_credential(identity_file),
        )
        client.ping()
        docker_control = "authenticated-typed-unix-socket"
    else:
        docker_control = "typed-unix-socket-not-mounted"
    print(
        json.dumps(
            {
                "artifact_volume": "writable",
                "executor": "JobAttemptExecutor",
                "oci_runner": "OciJobRunner",
                "docker_control": docker_control,
                "status": "ready",
            },
            sort_keys=True,
        )
    )
    return 0


def main(
    argv: list[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> int:
    values = os.environ if environ is None else environ
    args = _parser(values).parse_args(argv)
    if args.command == "doctor":
        return _doctor(
            args.artifact_root,
            control_socket=args.control_agent_job_socket,
            identity_file=args.control_agent_job_identity_file,
        )
    raise RuntimeError("unsupported job runtime command")


if __name__ == "__main__":
    raise SystemExit(main())
