"""Prepare exact local v2 config and separated roles after verified recovery."""

import json
import secrets
import subprocess
from pathlib import Path

from dotenv import dotenv_values
from psycopg import sql
from psycopg.conninfo import make_conninfo

from tools.installation.local_platform import render

root = Path(".local_artifacts/platform-local").resolve()
out = Path(__file__).resolve().parent
assert (
    json.loads((out / "recovery-verified.json").read_text())["user_config_secret_state"]
    == "recoverable"
)
pg = dotenv_values(root / "private/postgresql.env")
roles = {
    name: secrets.token_urlsafe(36)
    for name in ["roehub_app", "roehub_monitor", "roehub_maintenance"]
}
statements = []
for role, password in roles.items():
    statements.append(
        sql.SQL(
            "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION"
        )
        .format(sql.Identifier(role), sql.Literal(password))
        .as_string()
    )
statements.extend(
    [
        "GRANT pg_monitor TO roehub_monitor",
        "GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO roehub_app,roehub_monitor,roehub_maintenance",  # noqa: E501
    ]
)
command = [
    "docker",
    "exec",
    "-i",
    "roehub-local-postgresql-1",
    "sh",
    "-c",
    'psql -X -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"',
]
r = subprocess.run(
    command,
    input="BEGIN;\n" + ";\n".join(statements) + ";\nCOMMIT;",
    text=True,
    capture_output=True,
)
assert r.returncode == 0, "role provisioning failed; private diagnostic required"


def dsn(user, password):
    return make_conninfo(
        host="postgresql",
        port=5432,
        dbname=pg["POSTGRES_DB"],
        user=user,
        password=password,
        sslmode="disable",
    )


def write_env(path, values):
    assert all("\n" not in str(v) and "'" not in str(v) for v in values.values())
    path.write_text("".join(f"{k}='{v}'\n" for k, v in values.items()))
    path.chmod(0o600)


app = dsn("roehub_app", roles["roehub_app"])
for name in ["api", "worker", "controller"]:
    path = root / "private" / (name + ".env")
    values = {
        k: v
        for k, v in dotenv_values(path).items()
        if not k.startswith(("CH_", "CLICKHOUSE_", "ROEHUB_CH_"))
    }
    for k in list(values):
        if k.endswith("PG_DSN") or k in ["POSTGRES_DSN", "ROEHUB_STORAGE_POSTGRES_DSN"]:
            values[k] = app
    values["STRATEGY_PG_DSN"] = app
    write_env(path, values)
write_env(
    root / "private/observer.env",
    {
        "ROEHUB_PG_MONITOR_DSN": dsn("roehub_monitor", roles["roehub_monitor"]),
        "ROEHUB_PG_MAINTENANCE_DSN": dsn("roehub_maintenance", roles["roehub_maintenance"]),
    },
)
write_env(
    root / "private/s10-operator.env",
    {
        "ROEHUB_STORAGE_POSTGRES_DSN": dsn(pg["POSTGRES_USER"], pg["POSTGRES_PASSWORD"]),
        "STRATEGY_PG_DSN": app,
    },
)
config = json.loads((root / "installation.json").read_text())
config = {k: v for k, v in config.items() if "clickhouse" not in k}
config.update(
    schema="io.roehub.local-installation/v2alpha1",
    runtime_image="roehub/local-runtime:s10-pg-only-20261014",
    postgres_role_contract="separated-v1",
    postgres_budget_volume="roehub-local-pg-budget",
)
config["control_image"] = subprocess.check_output(
    [
        "docker",
        "image",
        "inspect",
        "roehub/local-control:s10-pg-only-20261014",
        "--format",
        "{{.Id}}",
    ],
    text=True,
).strip()
release = json.loads((root / "release.json").read_text())
release["images"].pop("clickhouse")
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
    json.dumps(
        {
            "roles_created": list(roles),
            "runtime_image": image,
            "config_schema": config["schema"],
            "outputs": list(outputs),
        }
    )
)
