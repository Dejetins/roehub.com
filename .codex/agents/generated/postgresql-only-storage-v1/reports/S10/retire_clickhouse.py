"""Remove only the stopped discovered CH container; archive obsolete config, retain volume."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(".local_artifacts/platform-local").resolve()
out = Path(__file__).resolve().parent
entry = json.loads((out / "entry-installation.json").read_text())
expected = next(x for x in entry if x["Name"] == "/roehub-local-clickhouse-1")
current = json.loads(subprocess.check_output(["docker", "inspect", "roehub-local-clickhouse-1"]))[0]
assert current["Id"] == expected["Id"] and not current["State"]["Running"]
assert any(m.get("Name") == "roehub-navigator-ch" for m in current["Mounts"])
subprocess.run(["docker", "rm", current["Id"]], check=True, capture_output=True)
paths = [
    root / "private/clickhouse.env",
    root / "private/clickhouse-workloads",
    *sorted((root / "config").glob("clickhouse-*")),
    root / "render_runtime.py",
]
archived = {}
for path in paths:
    relative = path.relative_to(root)
    target = root / "retired-s10/obsolete-config" / relative
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    before = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ([path] if path.is_file() else path.rglob("*"))
        if p.is_file()
    }
    path.rename(target)
    archived.update(before)
(
    root / "render_runtime.py"
).write_text('''"""Render only the explicit current v2 installation and image-bound release."""
import json
from pathlib import Path
from tools.installation.local_platform import render
root = Path(__file__).resolve().parent
source = (root / "installation.json").read_bytes()
release_source = (root / "release.json").read_bytes()
outputs = render(json.loads(source), json.loads(release_source),
                 source=source, release_source=release_source)
for name, value in outputs.items():
    (root / name).write_bytes(value)
print("Rendered current PostgreSQL-only installation")
''')
volume = json.loads(
    subprocess.check_output(["docker", "volume", "inspect", "roehub-navigator-ch"])
)[0]
record = {
    "removed_container": current["Id"],
    "retained_inert_volume": volume["Name"],
    "volume_deleted": False,
    "archived_obsolete_files": archived,
    "archive_root": str(root / "retired-s10/obsolete-config"),
    "renderer_replaced": str(root / "render_runtime.py"),
}
(out / "clickhouse-retired.json").write_text(json.dumps(record, indent=2) + "\n")
print(
    json.dumps(
        {
            "removed_exact_container": True,
            "archived_files": len(archived),
            "volume_retained": volume["Name"],
        }
    )
)
