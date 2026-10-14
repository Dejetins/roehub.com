"""Disposable real Docker recreation, refused boundaries and interrupted-stop retry."""

import json
import subprocess
import sys
from pathlib import Path

PREFIX = "roehub-s10-replacement3"
ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


if len(sys.argv) == 1:
    docker("network", "create", PREFIX)
    docker("volume", "create", PREFIX + "-data")
    docker("volume", "create", PREFIX + "-registry")
    docker(
        "run",
        "-d",
        "--name",
        PREFIX + "-pg",
        "--network",
        PREFIX,
        "-e",
        "POSTGRES_HOST_AUTH_METHOD=trust",
        "-v",
        PREFIX + "-data:/var/lib/postgresql/data",
        "postgres:16",
    )
    docker(
        "run",
        "-d",
        "--name",
        PREFIX + "-supervisor",
        "--entrypoint",
        "sleep",
        "roehub/s08-proof:local",
        "infinity",
    )
    docker("stop", PREFIX + "-supervisor")
    docker(
        "run",
        "-d",
        "--name",
        PREFIX + "-client",
        "--network",
        PREFIX,
        "--user",
        "0:0",
        "--entrypoint",
        "sleep",
        "-e",
        "PYTHONPATH=/work/src:/work",
        "-e",
        "ROEHUB_PG_BUDGET_ROOT=/run/roehub-pg-budget",
        "-e",
        "ROEHUB_PG_BUDGET_REQUIRED=1",
        "-v",
        str(ROOT) + ":/work:ro",
        "-v",
        PREFIX + "-registry:/run/roehub-pg-budget",
        "-v",
        "/var/run/docker.sock:/var/run/docker.sock",
        "-w",
        "/work",
        "roehub/s08-proof:local",
        "infinity",
    )
    subprocess.run(
        [
            "docker",
            "exec",
            PREFIX + "-client",
            "python",
            str(Path(__file__).relative_to(ROOT)),
            "client",
        ],
        check=True,
    )
else:
    import time

    from tools.installation.postgres_budget_recovery import (
        Engine,
        EngineHTTPError,
        enroll,
        read_pin,
        replace_recover,
        resume,
    )
    from trading.platform.postgres_budget import connect

    root = Path("/run/roehub-pg-budget")
    root.chmod(0o755)
    for name in ["authority.json"] + [
        f"{k}-{i}.json"
        for k, n in [("market", 16), ("control", 24), ("admin", 8)]
        for i in range(n)
    ]:
        (root / name).touch(mode=0o600)
    engine = Engine("/var/run/docker.sock")
    name = PREFIX + "-pg"
    supervisor = PREFIX + "-supervisor"
    for _ in range(40):
        try:
            engine.execute(
                name,
                ["sh", "-ec", 'test "$(sed -n 8p "$PGDATA/postmaster.pid" | tr -d " ")" = ready'],
            )
            break
        except RuntimeError:
            time.sleep(0.25)
    dsn = f"host={name} port=5432 dbname=postgres user=postgres sslmode=disable"
    enroll(root, engine, name, dsn)
    failures = []
    try:
        replace_recover(root, engine, name, supervisor)
    except RuntimeError as error:
        failures.append(str(error))
    else:
        raise AssertionError("existing old must fail")
    # A real abruptly exiting client leaves its durable reservation quarantined.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import os;from trading.platform.postgres_budget import connect;"
            f'c=connect({dsn!r},budget_class="market");os._exit(0)',
        ],
        check=True,
    )
    pin = read_pin(root)
    old = engine.inspect(name)
    engine.call("DELETE", f'/containers/{old["Id"]}?force=true')
    body = {
        "Image": "postgres:16",
        "Env": ["POSTGRES_HOST_AUTH_METHOD=trust"],
        "HostConfig": {"Binds": [PREFIX + "-data:/var/lib/postgresql/data"], "NetworkMode": PREFIX},
        "NetworkingConfig": {
            "EndpointsConfig": {PREFIX: {"IPAMConfig": {"IPv4Address": pin["endpoint"]["address"]}}}
        },
    }
    new = engine.call("POST", f"/containers/create?name={name}", body)["Id"]
    engine.call("POST", f"/containers/{new}/start")
    for _ in range(40):
        try:
            engine.execute(
                name,
                ["sh", "-ec", 'test "$(sed -n 8p "$PGDATA/postmaster.pid" | tr -d " ")" = ready'],
            )
            break
        except RuntimeError:
            time.sleep(0.25)

    # Deliberately report a foreign mount: no authority or slot mutation permitted.
    class WrongMount(Engine):
        def inspect(self, container):
            result = super().inspect(container)
            if result["Id"] == new:
                result["Mounts"][0]["Source"] = "foreign"
            return result

    before = (root / "authority.json").read_bytes()
    try:
        replace_recover(root, WrongMount("/var/run/docker.sock"), name, supervisor)
    except RuntimeError as error:
        failures.append(str(error))
    else:
        raise AssertionError("foreign mount must fail")
    assert (root / "authority.json").read_bytes() == before

    # A transport/500 failure is not evidence that the enrolled target was removed.
    class UnknownOld(Engine):
        def inspect(self, container):
            if container == old["Id"]:
                raise EngineHTTPError(500)
            return super().inspect(container)

    try:
        replace_recover(root, UnknownOld("/var/run/docker.sock"), name, supervisor)
    except RuntimeError as error:
        failures.append(str(error))
    else:
        raise AssertionError("unknown old status must fail")
    assert (root / "authority.json").read_bytes() == before
    owner = engine.call(
        "POST",
        "/containers/create?name=" + PREFIX + "-owner",
        {
            "Image": "roehub/s08-proof:local",
            "Entrypoint": ["sleep", "infinity"],
            "HostConfig": {"Binds": [PREFIX + "-data:/data:ro"]},
        },
    )["Id"]
    engine.call("POST", f"/containers/{owner}/start")
    try:
        replace_recover(root, engine, name, supervisor)
    except RuntimeError as error:
        failures.append(str(error))
    else:
        raise AssertionError("active volume owner must fail")
    assert (root / "authority.json").read_bytes() == before
    engine.call("POST", f"/containers/{owner}/stop?t=30")

    class LostStop(Engine):
        def call(self, method, path, body=None, *, raw=False):
            value = super().call(method, path, body, raw=raw)
            if path.endswith("/stop?t=30"):
                raise TimeoutError("injected lost stop response")
            return value

    try:
        replace_recover(root, LostStop("/var/run/docker.sock"), name, supervisor)
    except TimeoutError:
        pass
    else:
        raise AssertionError("stop interruption missing")
    assert json.loads((root / "authority.json").read_text())["recovery_pending"]
    assert (root / "market-0.json").stat().st_size > 0
    recovered = replace_recover(root, engine, name, supervisor)
    repeated = replace_recover(root, engine, name, supervisor)
    assert repeated["cleared_reservations"] == 0
    resumed = resume(root, engine, name, dsn)
    with connect(dsn, budget_class="admin") as c:
        assert c.execute("SELECT 42").fetchone()[0] == 42
    print(
        json.dumps(
            {
                "fail_closed": failures,
                "lost_stop_kept_quarantine": True,
                "recovery": recovered,
                "resume": resumed,
                "counted_query": 42,
                "completed_stop_repeat": repeated,
            }
        )
    )
