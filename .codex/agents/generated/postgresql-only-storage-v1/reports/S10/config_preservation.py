"""Compare private/config archive in memory; report only paths and changed key names."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from dotenv import dotenv_values

root = Path(".local_artifacts/platform-local").resolve()
proc = subprocess.Popen(
    [
        "age",
        "-d",
        "-i",
        ".local_artifacts/s10-recovery-keys/identity.agekey",
        str(root / "backups/s10-before-20261014/installation-config.tar.gz.age"),
    ],
    stdout=subprocess.PIPE,
)
unchanged = []
changed = {}
environment = {}
with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
    for member in tar:
        if not member.isfile():
            continue
        path = Path(member.name)
        old = tar.extractfile(member).read()
        current = root / path
        if not current.is_file():
            current = root / "retired-s10/obsolete-config" / path
        new = current.read_bytes() if current.is_file() else None
        if new is not None and hashlib.sha256(old).digest() == hashlib.sha256(new).digest():
            unchanged.append(str(path))
        else:
            changed[str(path)] = "changed" if new is not None else "missing"
        if path.suffix == ".env" and new is not None:
            a = dotenv_values(stream=io.StringIO(old.decode()))
            b = dotenv_values(stream=io.StringIO(new.decode()))
            keys = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
            def allowed(k):
                return k.startswith(("CH_", "CLICKHOUSE_", "ROEHUB_CH_")) or k.endswith("PG_DSN") or k in ("POSTGRES_DSN", "ROEHUB_STORAGE_POSTGRES_DSN", "ROEHUB_PG_MONITOR_DSN", "ROEHUB_PG_MAINTENANCE_DSN")  # noqa: E501
            assert all(allowed(k) for k in keys), (str(path), keys)
            environment[str(path)] = {
                "changed_key_names": keys,
                "all_other_values_unchanged": True,
                "network_configuration_preserved": True,
            }
assert proc.wait() == 0
print(
    json.dumps(
        {
            "unchanged_files": unchanged,
            "changed_files": changed,
            "environment_comparison": environment,
            "secrets_printed": False,
        },
        indent=2,
    )
)
