"""Complete the market-generation reset: retain old slots outside the active root."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

root = Path(".local_artifacts/platform-local").resolve()
artifacts = Path(json.loads((root / "installation.json").read_text())["artifact_root"])
retired = root / "retired-s10"
proc = subprocess.Popen(
    [
        "age",
        "-d",
        "-i",
        ".local_artifacts/s10-recovery-keys/identity.agekey",
        str(root / "backups/s10-before-20261014/artifacts-before.tar.gz.age"),
    ],
    stdout=subprocess.PIPE,
)
count = 0
with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
    for member in tar:
        if not member.isfile():
            continue
        relative = Path(member.name).relative_to("artifacts")
        path = artifacts / relative
        if not path.is_file():
            path = retired / "current-pointers" / relative
        assert (
            path.is_file()
            and hashlib.sha256(path.read_bytes()).digest()
            == hashlib.sha256(tar.extractfile(member).read()).digest()
        ), str(relative)
        count += 1
assert proc.wait() == 0
# Every coordinate with old manifests is retained as a unit (including old journals).
coordinates = sorted({p.parent.parent for p in artifacts.glob("*/*/*/slot_*/manifest.yaml")})
manifest = []
for path in coordinates:
    assert not (path / "current.yaml").exists()
    target = retired / "artifact-coordinates" / path.relative_to(artifacts)
    assert not target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    path.rename(target)
    manifest.append(str(path.relative_to(artifacts)))
record = {
    "all_prior_artifact_files_exact": count,
    "coordinates_archived": manifest,
    "destination": str(retired / "artifact-coordinates"),
    "deleted_files": 0,
    "reason": "fresh_generation_requires_empty_active_coordinate",
}
Path(__file__).with_name("slot-archive.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"prior_files_exact": count, "coordinates": len(manifest), "deleted": 0}))
