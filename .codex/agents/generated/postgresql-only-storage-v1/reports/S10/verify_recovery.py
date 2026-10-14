"""Restore encrypted pre-cutover snapshot into an exact isolated empty target."""

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

root = Path(".local_artifacts/platform-local").resolve()
out = Path(__file__).resolve().parent
backup = root / "backups/s10-before-20261014"
key = Path(".local_artifacts/s10-recovery-keys/identity.agekey").resolve()


def decrypt(name):
    return subprocess.Popen(
        ["age", "-d", "-i", str(key), str(backup / name)], stdout=subprocess.PIPE
    )


proc = decrypt("postgres-before.dump.age")
r = subprocess.run(
    [
        "docker",
        "exec",
        "-i",
        "roehub-s10-recovery-verify",
        "pg_restore",
        "-U",
        "postgres",
        "-d",
        "restore",
        "--exit-on-error",
        "--no-owner",
        "--no-privileges",
    ],
    stdin=proc.stdout,
    capture_output=True,
)
assert proc.wait() == 0
assert r.returncode == 0, r.stderr.decode()[:500]


def query(container, sql):
    command = [
        "docker",
        "exec",
        "-i",
        container,
        "sh",
        "-c",
        'psql -X -q -At -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"',
    ]
    if container.endswith("verify"):
        command = [
            "docker",
            "exec",
            "-i",
            container,
            "psql",
            "-X",
            "-q",
            "-At",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            "postgres",
            "-d",
            "restore",
        ]
    return subprocess.check_output(command, input=sql.encode()).decode().strip()


source = "roehub-local-postgresql-1"
tables = query(
    source, "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1"
).splitlines()
rows = {}
for table in tables:
    assert table.replace("_", "").isalnum()
    sql = f"""SELECT json_build_object('count',count(*),'md5',md5(coalesce(string_agg(
              row_to_json(t)::text,E'\\n' ORDER BY row_to_json(t)::text COLLATE "C"),''))) FROM "{table}" t"""  # noqa: E501
    original = json.loads(query(source, sql))
    restored = json.loads(query("roehub-s10-recovery-verify", sql))
    assert original == restored, (table, original, restored)
    rows[table] = original
(out / "preserved-public-baseline.json").write_text(json.dumps(rows, indent=2) + "\n")
verified = {}
for filename, base in [
    ("installation-config.tar.gz.age", root),
    (
        "artifacts-before.tar.gz.age",
        Path(json.loads((root / "installation.json").read_text())["artifact_root"]),
    ),
]:
    proc = decrypt(filename)
    count = 0
    with tarfile.open(fileobj=proc.stdout, mode="r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            relative = Path(member.name)
            if filename.startswith("artifacts"):
                relative = relative.relative_to("artifacts")
            data = tar.extractfile(member).read()
            assert (
                hashlib.sha256(data).digest()
                == hashlib.sha256((base / relative).read_bytes()).digest()
            ), str(relative)
            count += 1
    assert proc.wait() == 0
    verified[filename] = count
record = {
    "restore_target": "roehub-s10-recovery-verify",
    "isolated_empty_restore": True,
    "public_tables_exact": len(rows),
    "archive_files_exact": verified,
    "user_config_secret_state": "recoverable",
    "source_database_mutated": False,
}
(out / "recovery-verified.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record))
