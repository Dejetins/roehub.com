#!/usr/bin/env python3
"""Run the required real Docker boundary for every installation profile."""

from __future__ import annotations

import filecmp
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.release.generate_installation_config import run as run_generator

ROOT = Path(__file__).resolve().parents[2]


class InstallationRuntimeError(RuntimeError):
    """Raised when Docker or one generated profile fails the runtime boundary."""


def _run(command: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _compare_directories(first: Path, second: Path) -> None:
    comparison = filecmp.dircmp(first, second)
    if comparison.left_only or comparison.right_only or comparison.funny_files:
        raise InstallationRuntimeError(
            "deterministic render file set differs: "
            f"left_only={comparison.left_only}, right_only={comparison.right_only}, "
            f"funny={comparison.funny_files}"
        )
    for name in comparison.common_files:
        if (first / name).read_bytes() != (second / name).read_bytes():
            raise InstallationRuntimeError(f"deterministic render differs: {first / name}")
    for name in comparison.common_dirs:
        _compare_directories(first / name, second / name)


def verify_runtime() -> None:
    client_server = _run(
        ["docker", "version", "--format", "{{.Client.Version}}|{{.Server.Version}}"]
    ).stdout.strip()
    compose_version = _run(["docker", "compose", "version", "--short"]).stdout.strip()
    cache_root = Path.home() / ".cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(prefix="roehub-stage03-a-", dir=cache_root) as first_raw,
        tempfile.TemporaryDirectory(prefix="roehub-stage03-b-", dir=cache_root) as second_raw,
    ):
        first = Path(first_raw)
        second = Path(second_raw)
        if run_generator(["--output", str(first), "--write"]) != 0:
            raise InstallationRuntimeError("first installation render failed")
        if run_generator(["--output", str(first), "--check"]) != 0:
            raise InstallationRuntimeError("installation render check failed")
        if run_generator(["--output", str(second), "--write"]) != 0:
            raise InstallationRuntimeError("second installation render failed")
        _compare_directories(first, second)

        for profile in ("base", "trading", "ml"):
            compose = first / profile / "compose.yaml"
            _run(["docker", "compose", "-f", str(compose), "config", "--quiet"])
            rendered = _run(["docker", "compose", "-f", str(compose), "config"]).stdout
            lowered = rendered.lower()
            if "latest" in lowered or "mainnet" in lowered:
                raise InstallationRuntimeError(
                    f"unsafe value in generated Compose profile: {profile}"
                )
            consumer = _run(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(compose),
                    "run",
                    "--rm",
                    "--no-deps",
                    "config-consumer",
                ]
            )
            if consumer.stdout.strip() != "config-consumer-ok":
                raise InstallationRuntimeError(
                    f"config consumer returned unexpected output: {profile}"
                )
    print(
        "installation runtime verification passed: "
        f"docker={client_server}, compose={compose_version}, profiles=base,trading,ml"
    )


def main() -> int:
    try:
        verify_runtime()
    except (InstallationRuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"installation runtime verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
