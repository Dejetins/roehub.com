"""Installation-owned ClickHouse workload identities, grants and private provisioning.

Provisioning is explicit and additive. Existing passwords are reused byte-for-byte;
no existing user is altered or deleted. Server users are mounted from a private file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import secrets
from pathlib import Path
from xml.etree import ElementTree as ET

WORKLOADS = ("stream", "interactive", "backfill", "artifacts")
READ_TABLES = (
    "ref_market",
    "ref_instruments",
    "canonical_candles_1m",
    "canonical_candles_1m_stats",
    "source_absent_candles_1m",
)
WRITE_TABLES = (
    "raw_binance_klines_1m",
    "raw_bybit_klines_1m",
    "canonical_candles_1m",
    "canonical_candles_1m_stats",
    "source_absent_candles_1m",
)


def grants(workload: str, database: str = "market_data") -> tuple[str, ...]:
    """Exact application table grants, no DDL, access management or wildcard database."""
    if workload not in WORKLOADS or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", database):
        raise ValueError("invalid workload/database")
    result = [f"GRANT SELECT ON {database}.{table}" for table in READ_TABLES]
    if workload in {"stream", "backfill"}:
        result += [
            f"GRANT SELECT ON {database}.{table}"
            for table in ("raw_binance_klines_1m", "raw_bybit_klines_1m")
        ]
        result += [f"GRANT INSERT ON {database}.{table}" for table in WRITE_TABLES]
    if workload in {"backfill", "artifacts"}:
        result += [
            f"GRANT SELECT ON {database}.{table}"
            for table in ("canonical_funding_rates", "funding_instrument_universe")
        ]
    if workload == "backfill":
        result += [
            f"GRANT INSERT ON {database}.{table}"
            for table in (
                "ref_market",
                "ref_instruments",
                "canonical_funding_rates",
                "raw_binance_funding_rates",
                "raw_bybit_funding_rates",
                "funding_instrument_universe",
            )
        ]
    return tuple(result)


def private_users(
    passwords: dict[str, str], database: str = "market_data", *,
    default_user_disabled: bool = False,
) -> bytes:
    """Render only into the private installation directory, never public outputs."""
    root = ET.Element("clickhouse")
    users = ET.SubElement(root, "users")
    # With CLICKHOUSE_USER != default, the official entrypoint removes default.
    # A networks-only fragment would recreate it without authentication and stop
    # server startup. Use this mode only after verifying that entrypoint contract.
    if not default_user_disabled:
        default = ET.SubElement(users, "default")
        networks = ET.SubElement(default, "networks", {"replace": "replace"})
        for ip in ("127.0.0.1", "::1"):
            ET.SubElement(networks, "ip").text = ip
    for workload in WORKLOADS:
        password = passwords[workload]
        if len(password) < 32 or password.strip() != password:
            raise ValueError("workload credential must contain at least 32 characters")
        user = ET.SubElement(users, "roehub_" + workload)
        ET.SubElement(user, "password_sha256_hex").text = hashlib.sha256(
            password.encode()
        ).hexdigest()
        ET.SubElement(ET.SubElement(user, "networks"), "ip").text = "::/0"
        ET.SubElement(user, "profile").text = "roehub_" + workload
        ET.SubElement(user, "quota").text = "default"
        grant_nodes = ET.SubElement(user, "grants")
        for grant in grants(workload, database):
            ET.SubElement(grant_nodes, "query").text = grant
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8") + b"\n"


def provision(
    directory: Path, *, database: str = "market_data", default_user_disabled: bool = False,
) -> None:
    """Create only absent class credentials; refuse unsafe existing files/directories."""
    if directory.is_symlink():
        raise ValueError("private directory symlink is forbidden")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.stat().st_mode & 0o077:
        raise ValueError("private directory must be mode 0700")
    passwords = {}
    for workload in WORKLOADS:
        path = directory / (workload + ".password")
        if path.is_symlink():
            raise ValueError("credential symlink is forbidden")
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as output:
                output.write(secrets.token_hex(32))
                output.flush()
                os.fsync(output.fileno())
        if path.stat().st_mode & 0o077:
            raise ValueError("credential must be mode 0600")
        passwords[workload] = path.read_text()
    payload = private_users(passwords, database, default_user_disabled=default_user_disabled)
    path = directory / "users.xml"
    if path.is_symlink():
        raise ValueError("users file symlink is forbidden")
    if path.exists():
        if path.read_bytes() != payload or path.stat().st_mode & 0o077:
            raise ValueError("existing users configuration differs; explicit migration required")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-directory", required=True, type=Path)
    parser.add_argument("--database", default="market_data")
    parser.add_argument(
        "--default-user-disabled", action="store_true",
        help="Verified official entrypoint removes default via a non-default CLICKHOUSE_USER",
    )
    args = parser.parse_args()
    provision(
        args.private_directory, database=args.database,
        default_user_disabled=args.default_user_disabled,
    )
    print("Additive workload credentials prepared; no server or client activated")


if __name__ == "__main__":
    main()
