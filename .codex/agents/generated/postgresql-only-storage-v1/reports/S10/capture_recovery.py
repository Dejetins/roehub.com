"""Quiesced legacy local capture; not the seven-owner release backup protocol."""

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

root = Path(".local_artifacts/platform-local").resolve()
out = Path(__file__).resolve().parent
backup = root / "backups/s10-before-20261014"
keys = Path(".local_artifacts/s10-recovery-keys").resolve()
backup.mkdir(mode=0o700)
keys.mkdir(mode=0o700)
key = keys / "identity.agekey"
subprocess.run(["age-keygen", "-o", str(key)], check=True, capture_output=True)
key.chmod(0o600)
recipient = subprocess.check_output(["age-keygen", "-y", str(key)], text=True).strip()
record = {
    "kind": "legacy-local-scoped-recovery",
    "backup_root": str(backup),
    "key_directory": str(keys),
    "files": {},
}


def encrypt(name, produce):
    target = backup / (name + ".age")
    with target.open("xb") as dest:
        os.chmod(target, 0o600)
        proc = subprocess.Popen(["age", "-r", recipient], stdin=subprocess.PIPE, stdout=dest)
        produce(proc.stdin)
        proc.stdin.close()
        assert proc.wait() == 0
    digest = hashlib.sha256()
    proc = subprocess.Popen(["age", "-d", "-i", str(key), str(target)], stdout=subprocess.PIPE)
    while chunk := proc.stdout.read(1024 * 1024):
        digest.update(chunk)
    assert proc.wait() == 0
    record["files"][target.name] = {
        "bytes": target.stat().st_size,
        "ciphertext_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "decrypted_sha256": digest.hexdigest(),
        "decrypt_verified": True,
    }


def config(stream):
    with tarfile.open(fileobj=stream, mode="w|gz") as tar:
        for name in [
            "installation.json",
            "release.json",
            "compose.yaml",
            "control-compose.yaml",
            "control-policy.json",
            "generation-manifest.json",
            "config",
            "private",
            "control-state",
        ]:
            tar.add(root / name, arcname=name)


def pg(stream):
    subprocess.run(
        [
            "docker",
            "exec",
            "roehub-local-postgresql-1",
            "sh",
            "-c",
            'pg_dump -Fc -U "$POSTGRES_USER" -d "$POSTGRES_DB"',
        ],
        stdout=stream,
        check=True,
    )


def artifacts(stream):
    path = Path(json.loads((root / "installation.json").read_text())["artifact_root"])
    with tarfile.open(fileobj=stream, mode="w|gz") as tar:
        tar.add(path, arcname="artifacts")


encrypt("installation-config.tar.gz", config)
encrypt("postgres-before.dump", pg)
encrypt("artifacts-before.tar.gz", artifacts)
(out / "recovery-capture.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
