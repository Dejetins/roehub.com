"""Host driver retains handles and samples; exit codes are not acceptance verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / ".codex/agents/generated/postgresql-only-storage-v1/reports/S09"


def phase(name, mode, duration):
    assert not (OUT / (name + "-summary.json")).exists()
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in (
            "tools/qa/postgres_production_*.py",
            ".codex/agents/generated/postgresql-only-storage-v1/reports/S09/measurement-contract.md",
            "src/trading/contexts/market_data/adapters/outbound/persistence/postgres/*.py",
            "src/trading/platform/postgres_budget*.py",
        )
        for p in ROOT.glob(pattern)
    }
    (OUT / (name + "-source-hashes.json")).write_text(json.dumps(sources, indent=2))
    with (
        (OUT / (name + "-monitor.log")).open("x") as monlog,
        (OUT / (name + "-run.log")).open("x") as runlog,
        (OUT / (name + "-server-resources.jsonl")).open("x") as resources,
    ):
        monitor = subprocess.Popen(
            [
                "docker",
                "exec",
                "roehub-s09-client",
                "python",
                "tools/qa/postgres_production_monitor.py",
                name,
            ],
            stdout=monlog,
            stderr=subprocess.STDOUT,
        )
        process = subprocess.Popen(
            [
                "docker",
                "exec",
                "roehub-s09-client",
                "python",
                "tools/qa/postgres_production_workload.py",
                name,
                "--mode",
                mode,
                "--duration",
                str(duration),
            ],
            stdout=runlog,
            stderr=subprocess.STDOUT,
        )
        print(
            json.dumps(
                {
                    "phase": name,
                    "driver_pid": process.pid,
                    "monitor_pid": monitor.pid,
                    "duration": duration,
                }
            ),
            flush=True,
        )
        started = time.monotonic()
        last = 0
        while process.poll() is None:
            row: dict[str, Any] = {"utc": time.time(), "elapsed_s": time.monotonic() - started}
            for key, command in [
                ("cpu", ["cat", "/sys/fs/cgroup/cpu.stat"]),
                (
                    "memory",
                    [
                        "cat",
                        "/sys/fs/cgroup/memory.current",
                        "/sys/fs/cgroup/memory.peak",
                        "/sys/fs/cgroup/memory.stat",
                        "/sys/fs/cgroup/memory.events",
                    ],
                ),
                ("rss", ["sh", "-c", "awk '/^Name:|^VmRSS:/{print}' /proc/[0-9]*/status"]),
            ]:
                r = subprocess.run(
                    ["docker", "exec", "roehub-s09-pg", *command],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                row[key] = {"exit": r.returncode, "data": r.stdout}
            resources.write(json.dumps(row) + "\n")
            resources.flush()
            if time.monotonic() - last >= 60:
                progress = OUT / (name + "-progress.json")
                print(progress.read_text() if progress.exists() else name + " starting", flush=True)
                last = time.monotonic()
            time.sleep(5)
        (OUT / (name + "-monitor-stop")).touch()
        monitor.wait(timeout=20)
        result = {
            "process_exit": process.returncode,
            "monitor_exit": monitor.returncode,
            "wall_s": time.monotonic() - started,
        }
        (OUT / (name + "-driver.json")).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        assert process.returncode == 0 and monitor.returncode == 0, result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--mode", default="normal")
    p.add_argument("--duration", type=int, default=30)
    a = p.parse_args()
    phase(a.name, a.mode, a.duration)
