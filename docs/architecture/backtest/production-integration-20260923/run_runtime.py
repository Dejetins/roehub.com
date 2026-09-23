"""Own one disposable existing-image Postgres; execute normal child and DB readback."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    output = Path(__file__).parent.resolve()
    root = output.parents[3]
    scratch = Path(tempfile.mkdtemp(prefix="roehub-production-runtime-"))
    name = "roehub-production-proof-2705"
    began = time.monotonic()
    result = {"status": "running", "cap_seconds": 300, "container": name}
    env = {**os.environ, "S6_DSN": "postgresql://roehub:local-proof@127.0.0.1:55439/s6"}
    created = False
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "--memory",
                "512m",
                "-e",
                "POSTGRES_USER=roehub",
                "-e",
                "POSTGRES_PASSWORD=local-proof",
                "-e",
                "POSTGRES_DB=s6",
                "-p",
                "127.0.0.1:55439:5432",
                "5d004e058f52",
            ],
            check=True,
            capture_output=True,
        )
        created = True
        for _ in range(50):
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", "roehub"], capture_output=True
            )
            if ready.returncode == 0:
                break
            time.sleep(0.2)
        setup = subprocess.run(
            [sys.executable, "-I", "-B", str(output / "bootstrap_database.py")],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert setup.returncode == 0, setup.stderr[-8000:]
        result["bootstrap"] = json.loads(setup.stdout)
        cases = json.loads(
            (output / (sys.argv[1] if len(sys.argv) > 1 else "parity-cases.json")).read_text()
        )
        task = dict(
            source=str(root),
            dataset="/Users/daniildegtyarev/Projects/roehub.com/"
            ".local_artifacts/ethusdt-binance-usdm-testset-20260920T001656Z",
            policy="production",
            scratch=str(scratch),
            cache=str(scratch / "cache"),
            trace_index=1,
            cases=cases[:2],
        )
        taskfile = scratch / "task.json"
        taskfile.write_text(json.dumps(task))
        proof = subprocess.run(
            [sys.executable, "-I", "-B", str(output / "runtime_proof.py"), "--task", str(taskfile)],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=300 - (time.monotonic() - began),
        )
        assert proof.returncode == 0, proof.stderr[-12000:]
        result["proof"] = json.loads(proof.stdout)
        assert result["proof"]["cleanup_files"] == []
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        if created:
            cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            result["container_removed"] = cleanup.returncode == 0
        shutil.rmtree(scratch)
        result["scratch_removed"] = not scratch.exists()
        result["elapsed_seconds"] = time.monotonic() - began
        (output / (sys.argv[2] if len(sys.argv) > 2 else "runtime-result.json")).write_text(
            json.dumps(result, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
