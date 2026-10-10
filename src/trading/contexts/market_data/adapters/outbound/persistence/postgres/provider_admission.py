"""Atomic multi-window reservations and durable concurrency, using database time.

No SQL transaction spans HTTP. Uncertain permits remain charged across connection
loss/restart. Recovery requires confirmed owner I/O termination, never expiry.
"""

import os
import socket
from dataclasses import dataclass, field
from typing import Mapping
from uuid import UUID, uuid4

import psycopg

from trading.contexts.market_data.application.ports.provider_admission import ProviderRequest
from trading.contexts.market_data.application.services.source_retry import TemporarySourceError


def _connect(dsn: str):
    connection = psycopg.connect(
        dsn,
        autocommit=True,
        connect_timeout=2,
        tcp_user_timeout=1000,
        keepalives=1,
        keepalives_idle=1,
        keepalives_interval=1,
        keepalives_count=2,
    )
    connection.execute("SET statement_timeout=1000")
    connection.execute("SET lock_timeout=1000")
    return connection


@dataclass
class PostgresProviderAdmission:
    dsn: str = field(repr=False)

    def reserve(self, request: ProviderRequest):
        connection = None
        try:
            connection = _connect(self.dsn)
            token = uuid4()
            with connection.transaction():
                connection.execute(
                    "INSERT INTO market_data_provider_scopes(provider,scope) VALUES (%s,%s) "
                    "ON CONFLICT DO NOTHING",
                    (request.provider, request.scope),
                )
                row = connection.execute(
                    "SELECT EXTRACT(EPOCH FROM clock_timestamp()), "
                    "CASE WHEN cooldown_until > clock_timestamp() THEN "
                    "EXTRACT(EPOCH FROM cooldown_until-clock_timestamp()) ELSE 0 END "
                    "FROM market_data_provider_scopes WHERE provider=%s AND scope=%s FOR UPDATE",
                    (request.provider, request.scope),
                ).fetchone()
                assert row is not None
                now, cooldown = float(row[0]), float(row[1])
                if cooldown > 0:
                    raise TemporarySourceError(rate_limited=True, retry_after_s=cooldown)
                count = connection.execute(
                    "SELECT count(*) FROM market_data_provider_permits "
                    "WHERE provider=%s AND scope=%s",
                    (request.provider, request.scope),
                ).fetchone()
                assert count is not None
                if count[0] >= request.concurrency:
                    raise TemporarySourceError(rate_limited=True, retry_after_s=0.25)
                for window in request.windows:
                    bucket = int(now // window.seconds)
                    connection.execute(
                        "INSERT INTO market_data_provider_windows "
                        "(provider,scope,dimension,seconds,bucket,spent) VALUES (%s,%s,%s,%s,%s,0) "
                        "ON CONFLICT (provider,scope,dimension,seconds) DO UPDATE SET "
                        "spent=CASE WHEN market_data_provider_windows.bucket>=EXCLUDED.bucket "
                        'THEN market_data_provider_windows.spent ELSE 0 '
                        'END,bucket=GREATEST(market_data_provider_windows.bucket,EXCLUDED.'
                        'bucket)',
                        (request.provider, request.scope, window.dimension, window.seconds, bucket),
                    )
                    spent = connection.execute(
                        "SELECT spent,bucket FROM market_data_provider_windows WHERE provider=%s "
                        "AND scope=%s AND dimension=%s AND seconds=%s",
                        (request.provider, request.scope, window.dimension, window.seconds),
                    ).fetchone()
                    assert spent is not None
                    if spent[1] > bucket:
                        raise TemporarySourceError(
                            rate_limited=True, retry_after_s=(spent[1] + 1) * window.seconds - now
                        )
                    if spent[0] + window.cost > window.limit:
                        raise TemporarySourceError(
                            rate_limited=True, retry_after_s=(bucket + 1) * window.seconds - now
                        )
                    connection.execute(
                        "UPDATE market_data_provider_windows SET spent=spent+%s "
                        "WHERE provider=%s AND scope=%s AND dimension=%s AND seconds=%s",
                        (
                            window.cost,
                            request.provider,
                            request.scope,
                            window.dimension,
                            window.seconds,
                        ),
                    )
                connection.execute(
                    "INSERT INTO market_data_provider_permits "
                    "(token,provider,scope,owner_pid,owner_host) VALUES (%s,%s,%s,%s,%s)",
                    (token, request.provider, request.scope, os.getpid(), socket.gethostname()),
                )
            return PostgresProviderPermit(self.dsn, connection, request, token)
        except Exception as error:
            if connection is not None:
                connection.close()
            if isinstance(error, TemporarySourceError):
                raise
            raise TemporarySourceError(retry_after_s=2) from error


@dataclass
class PostgresProviderPermit:
    dsn: str = field(repr=False)
    connection: psycopg.Connection
    request: ProviderRequest
    token: UUID

    def check(self) -> None:
        try:
            row = self.connection.execute(
                "SELECT 1 FROM market_data_provider_permits WHERE token=%s", (self.token,)
            ).fetchone()
            if row is None:
                raise RuntimeError("provider_permit_lost")
        except Exception as error:
            raise TemporarySourceError(retry_after_s=2) from error

    def complete(self, *, headers: Mapping[str, str], cooldown_s: float = 0) -> None:
        # The caller has already stopped I/O, so a fresh session may safely release.
        self.connection.close()
        try:
            with _connect(self.dsn) as connection, connection.transaction():
                connection.execute(
                    "SELECT 1 FROM market_data_provider_scopes WHERE provider=%s AND scope=%s "
                    "FOR UPDATE",
                    (self.request.provider, self.request.scope),
                )
                connection.execute(
                    "UPDATE market_data_provider_scopes SET cooldown_until=GREATEST("
                    "cooldown_until,clock_timestamp()+(%s * interval '1 second')) "
                    "WHERE provider=%s AND scope=%s",
                    (cooldown_s, self.request.provider, self.request.scope),
                )
                for key, value in headers.items():
                    # Observed usage can only raise the local charge, never reduce it.
                    if key.lower() == "x-mbx-used-weight-1m" and str(value).isdigit():
                        connection.execute(
                            "UPDATE market_data_provider_windows SET spent=GREATEST(spent,%s) "
                            "WHERE provider=%s AND scope=%s AND dimension=ANY(%s) AND seconds=60 "
                            "AND bucket=floor(EXTRACT(EPOCH FROM clock_timestamp())/60)",
                            (
                                int(value),
                                self.request.provider,
                                self.request.scope,
                                [
                                    w.dimension
                                    for w in self.request.windows
                                    if w.dimension.endswith("_weight")
                                ],
                            ),
                        )
                connection.execute(
                    "DELETE FROM market_data_provider_permits WHERE token=%s", (self.token,)
                )
        except Exception as error:
            # Failed/unknown commit leaves capacity charged; no local fallback.
            raise TemporarySourceError(retry_after_s=2) from error
