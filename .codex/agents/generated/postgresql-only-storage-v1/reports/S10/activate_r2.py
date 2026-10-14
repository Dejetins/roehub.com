"""Activate reviewed PG-only images without rotating any user/private credentials."""

import json
import subprocess
from pathlib import Path

from tools.installation.local_platform import render

root = Path(".local_artifacts/platform-local").resolve()
config = json.loads((root / "installation.json").read_text())
release = json.loads((root / "release.json").read_text())
config["runtime_image"] = "roehub/local-runtime:s10-pg-only-r2-20261014"
config["control_image"] = subprocess.check_output(
    [
        "docker",
        "image",
        "inspect",
        "roehub/local-control:s10-pg-only-r2-20261014",
        "--format",
        "{{.Id}}",
    ],
    text=True,
).strip()
image = subprocess.check_output(
    ["docker", "image", "inspect", config["runtime_image"], "--format", "{{.Id}}"], text=True
).strip()
release["images"]["runtime"]["reference"] = "roehub/local-runtime@" + image
source = (json.dumps(config, indent=2) + "\n").encode()
release_source = (json.dumps(release, indent=2) + "\n").encode()
outputs = render(config, release, source=source, release_source=release_source)
(root / "installation.json").write_bytes(source)
(root / "release.json").write_bytes(release_source)
for name, value in outputs.items():
    (root / name).write_bytes(value)
print(
    json.dumps({"runtime": image, "controller": config["control_image"], "outputs": list(outputs)})
)
