"""Resolver deadlines and violent caller shutdown cannot leave active DNS work."""

import os
import subprocess
import sys
import time

import pytest
from psycopg.conninfo import conninfo_to_dict

from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    ingestion_gateway as pg,  # noqa: E501
)
from trading.contexts.market_data.application.services.ingestion_retry import (
    TemporaryIngestionError,  # noqa: E501
)


def test_resolution_preserves_tls_host_and_repeats_each_lifecycle():
    first = conninfo_to_dict(
        pg._resolve_until("host=localhost sslmode=verify-full", time.monotonic() + 2)
    )
    second = conninfo_to_dict(
        pg._resolve_until("host=localhost sslmode=verify-full", time.monotonic() + 2)
    )
    assert first == second
    assert isinstance(first["host"], str) and isinstance(first["hostaddr"], str)
    assert first["sslmode"] == "verify-full"
    assert set(first["host"].split(",")) == {"localhost"}
    assert len(first["hostaddr"].split(",")) == len(first["host"].split(","))


def _stalled_code(pidfile):
    return (
        "import os,time;open("
        + repr(str(pidfile))
        + ",'w').write(str(os.getpid()));time.sleep(60);"
    )


def _alive(pid):
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return bool(result.stdout.strip()) and not result.stdout.strip().startswith("Z")


def test_dns_deadline_kills_and_reaps_resolver(monkeypatch, tmp_path):
    run = subprocess.run
    pidfile = tmp_path / "resolver"

    def stalled(args, **kwargs):
        args = list(args)
        args[2] = args[2].replace("print(json.dumps", _stalled_code(pidfile) + "print(json.dumps")
        return run(args, **kwargs)

    monkeypatch.setattr(pg.subprocess, "run", stalled)
    with pytest.raises(TemporaryIngestionError, match="postgres_dns_(timeout|unavailable)"):
        pg._resolve_until("host=localhost", time.monotonic() + 0.3)
    monkeypatch.undo()
    assert not _alive(int(pidfile.read_text()))


@pytest.mark.parametrize("kill", [False, True])
def test_dns_resolver_exits_when_writer_is_terminated(tmp_path, kill):
    pidfile = tmp_path / "resolver"
    code = f"""
import subprocess,time
from trading.contexts.market_data.adapters.outbound.persistence.postgres import (
    ingestion_gateway as pg,
)
run=subprocess.run
def stalled(args,**kwargs):
    args=list(args)
    args[2]=args[2].replace("print(json.dumps",{_stalled_code(pidfile)!r}+"print(json.dumps")
    return run(args,**kwargs)
pg.subprocess.run=stalled
pg._resolve_until("host=localhost",time.monotonic()+30)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": "src:."}
    )
    try:
        stop = time.monotonic() + 3
        while not pidfile.exists() and time.monotonic() < stop:
            time.sleep(0.01)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        parent.kill() if kill else parent.terminate()
        parent.wait(timeout=2)
        stop = time.monotonic() + 2
        while _alive(pid) and time.monotonic() < stop:
            time.sleep(0.02)
        assert not _alive(pid)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait()


def test_reconnect_refreshes_changed_addresses_under_original_tls_name(monkeypatch):
    answers = iter([b'["127.0.0.1"]', b'["::1","127.0.0.2"]'])
    monkeypatch.setattr(
        pg.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=next(answers)),
    )
    first = conninfo_to_dict(
        pg._resolve_until("host=database sslmode=verify-full", time.monotonic() + 1)
    )
    second = conninfo_to_dict(
        pg._resolve_until("host=database sslmode=verify-full", time.monotonic() + 1)
    )
    assert first["hostaddr"] == "127.0.0.1"
    assert second["hostaddr"] == "::1" and second["host"] == "database"
    assert second["sslmode"] == "verify-full"
