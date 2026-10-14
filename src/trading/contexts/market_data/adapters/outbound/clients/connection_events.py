"""Safe, bounded connection lifecycle logs; never include provider payloads or URLs."""

import logging
import socket
import time
from typing import Callable

from websockets.exceptions import ConnectionClosed


class ConnectionEvents:
    def __init__(self, logger: logging.Logger, market_id: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.logger, self.market_id, self.clock = logger, market_id, clock
        self.started: float | None = None
        self.last_log = float("-inf")
        self.attempts = 0

    def failed(self, error: Exception) -> None:
        now = self.clock()
        first = self.started is None
        if first:
            self.started = now
        assert self.started is not None
        self.attempts += 1
        if first or now - self.last_log >= 60:
            code = ("dns_unavailable" if isinstance(error, socket.gaierror) else
                    "connection_timeout" if isinstance(error, TimeoutError) else
                    "connection_closed" if isinstance(error, ConnectionClosed) else
                    "network_unavailable" if isinstance(error, OSError) else
                    "stream_processing_failed")
            self.logger.warning(
                "stream_connection_waiting market_id=%s code=%s attempts=%s elapsed_s=%.1f",
                self.market_id, code, self.attempts, now - self.started,
            )
            self.last_log = now

    def connected(self) -> None:
        if self.started is not None:
            self.logger.info(
                "stream_connection_restored market_id=%s attempts=%s outage_s=%.1f",
                self.market_id, self.attempts, self.clock() - self.started,
            )
        else:
            self.logger.info("stream_connection_opened market_id=%s", self.market_id)
        self.started = None
        self.attempts = 0
