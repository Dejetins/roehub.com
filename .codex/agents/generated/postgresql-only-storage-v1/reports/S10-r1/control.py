"""Exact installed-controller actions with durable operation identity."""

import json
import subprocess
import sys
import uuid
from pathlib import Path

root = Path(".local_artifacts/platform-local").resolve()
out = Path(__file__).resolve().parent
action, *services = sys.argv[1:]
operation = str(uuid.uuid4())
args = [
    "docker",
    "compose",
    "-p",
    "roehub-control",
    "-f",
    str(root / "control-compose.yaml"),
    "exec",
    "-T",
    "control-agent",
    "python",
    "-m",
    "apps.roehubctl.main.main",
    "--socket",
    "/var/run/roehub/control.sock",
    "--identity-file",
    str(root / "private/owner"),
    action,
    "--profile",
    "trading",
    "--operation-id",
    operation,
]
for service in services:
    args += ["--service", service]
record = {"action": action, "services": services, "operation_id": operation}
path = out / ("operation-" + operation + ".json")
path.write_text(json.dumps(record, indent=2) + "\n")
r = subprocess.run(args, capture_output=True, text=True)
record.update(returncode=r.returncode, stdout=r.stdout, stderr=r.stderr)
path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
sys.exit(r.returncode or (0 if json.loads(r.stdout).get("state") == "succeeded" else 1))
