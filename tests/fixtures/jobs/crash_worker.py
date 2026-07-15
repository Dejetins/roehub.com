from __future__ import annotations

import argparse
import stat
from datetime import UTC, datetime
from pathlib import Path

from apps.control_agent.job_runtime_backend import ControlAgentJobDockerRunner
from apps.worker.job_runtime.oci_runner import OciJobRunner
from trading.integration.job_runtime_postgres import PostgresJobRuntimeCatalog


def _private_text(path: Path) -> str:
    resolved = path.expanduser().resolve(strict=True)
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if not resolved.is_file() or resolved.is_symlink() or mode & 0o077:
        raise RuntimeError("crash worker DSN file must be a private regular file")
    return resolved.read_text(encoding="utf-8").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn-file", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--utility-image-digest", required=True)
    args = parser.parse_args()

    catalog = PostgresJobRuntimeCatalog(dsn=_private_text(args.dsn_file))
    claimed = catalog.claim_next(worker_id="stage15.lost.001", now=datetime.now(UTC))
    if claimed is None:
        raise RuntimeError("crash worker found no queued attempt")
    attempt_root = args.runtime_root.expanduser().resolve() / claimed.envelope.attempt_id.hex
    OciJobRunner(
        utility_image_digest=args.utility_image_digest,
        command_runner=ControlAgentJobDockerRunner(),
    ).run(
        envelope=claimed.envelope,
        input_root=attempt_root / "input",
        output_root=attempt_root / "output",
        heartbeat=lambda: catalog.heartbeat(
            organization_id=claimed.envelope.organization_id,
            attempt_id=claimed.envelope.attempt_id,
            worker_id="stage15.lost.001",
            now=datetime.now(UTC),
        ),
    )


if __name__ == "__main__":
    main()
