"""Bounded single-address resolution retaining the TLS verification hostname."""

import ipaddress
import json
import os
import subprocess
import sys
import time

from psycopg.conninfo import conninfo_to_dict, make_conninfo

from trading.platform.postgres_budget import BudgetExhausted


def _endpoint_params(dsn: str) -> dict[str, str]:
    params = {k: str(v) for k, v in conninfo_to_dict(dsn).items() if v is not None}
    host = params.get("host", "")
    if not host or params.get("service") or "," in host or "," in params.get("port", ""):
        raise ValueError("gateway requires one explicit PostgreSQL endpoint")
    return params


def _resolved_dsn(dsn: str) -> str:
    """Validate prepared conninfo; preserve original host for TLS verification."""
    params = _endpoint_params(dsn)
    host, address = params["host"], params.get("hostaddr", "")
    if host.startswith("/"):
        return dsn
    if address:
        ipaddress.ip_address(address)
    else:
        ipaddress.ip_address(host)
        params["hostaddr"] = host
    return make_conninfo("", **params)


def _resolve_until(dsn: str, stop: float) -> str:
    """Kill and reap bounded DNS work on timeout; re-resolve every new connection.

    The child sees hostname/port only, never credentials. Original host/TLS settings
    remain intact. libpq tries resolved IPv4/IPv6 addresses inside the same deadline.
    """
    params = _endpoint_params(dsn)
    host = params["host"]
    if host.startswith("/"):
        return dsn
    if params.get("hostaddr"):
        return _resolved_dsn(dsn)
    try:
        ipaddress.ip_address(host)
    except ValueError:
        remaining = stop - time.monotonic()
        if remaining <= 0:
            raise BudgetExhausted("postgres_dns_timeout")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import socket,json,sys,os,time,threading\n"
                    "parent=int(sys.argv[3]);stop=float(sys.argv[4])\n"
                    "def watch():\n"
                    " while os.getppid()==parent and time.monotonic()<stop: time.sleep(.05)\n"
                    " os._exit(124)\n"
                    "threading.Thread(target=watch,daemon=True).start()\n"
                    "print(json.dumps(list(dict.fromkeys(r[4][0] for r in "
                    "socket.getaddrinfo(sys.argv[1],sys.argv[2],type=socket.SOCK_STREAM)))))",
                    host,
                    params.get("port", "5432"),
                    str(os.getpid()),
                    str(stop),
                ],
                capture_output=True,
                timeout=remaining,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise BudgetExhausted("postgres_dns_timeout") from None
        if result.returncode:
            raise BudgetExhausted("postgres_dns_unavailable")
        addresses = json.loads(result.stdout)
        if not addresses:
            raise BudgetExhausted("postgres_dns_unavailable")
        for address in addresses:
            ipaddress.ip_address(address)
        params["hostaddr"] = addresses[0]
        params["host"] = host
    else:
        params["hostaddr"] = host
    return make_conninfo("", **params)
