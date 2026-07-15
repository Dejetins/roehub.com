from __future__ import annotations

import argparse
import json
import stat
from datetime import datetime
from pathlib import Path

from apps.control_agent.job_runtime_backend import ControlAgentJobDockerRunner
from apps.worker.job_runtime.recovery import JobRuntimeRecovery
from trading.integration.job_runtime_postgres import PostgresJobRuntimeCatalog


def _private_text(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if not resolved.is_file() or resolved.is_symlink() or mode & 0o077:
        raise RuntimeError("restart probe DSN file must be a private regular file")
    return resolved.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn-file", type=Path, required=True)
    parser.add_argument("--now", required=True)
    parser.add_argument("--worker-heartbeat-before", required=True)
    parser.add_argument("--recovery-claimed-before", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()

    catalog = PostgresJobRuntimeCatalog(dsn=_private_text(args.dsn_file))
    recovery = JobRuntimeRecovery(
        catalog=catalog,
        runtime_root=args.runtime_root,
        command_runner=ControlAgentJobDockerRunner(),
    )
    recovered = recovery.recover(
        now=datetime.fromisoformat(args.now),
        worker_heartbeat_before=datetime.fromisoformat(args.worker_heartbeat_before),
        recovery_claimed_before=datetime.fromisoformat(args.recovery_claimed_before),
    )
    print(
        json.dumps(
            {
                "schema": "io.roehub.job-runtime-restart-proof/v1",
                "recovered_attempts": len(recovered),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
