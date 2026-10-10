"""Exclusive, reusable HTTP I/O in a lean child that never imports the app entrypoint."""

import subprocess
import sys
import threading
import time
from typing import Callable

from trading.contexts.market_data.adapters.outbound.http_transport_worker import (
    COMMAND_LIMIT,
    RESPONSE_LIMIT,
    read_frame,
    write_frame,
)
from trading.contexts.market_data.application.services.source_retry import TemporarySourceError


class ManagedHttpTransport:
    def __init__(self):
        self.process = subprocess.Popen(
            [sys.executable, '-m',
             'trading.contexts.market_data.adapters.outbound.http_transport_worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0, close_fds=True,
        )
        self.closed = False
        self._stop_lock = threading.Lock()

    def stop(self):
        with self._stop_lock:
            if self.closed:
                return
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()  # Capacity stays charged until OS confirms death.
            if self.process.stdin is not None:
                self.process.stdin.close()
            if self.process.stdout is not None:
                self.process.stdout.close()
            self.closed = True

    def get(self, *, url, params, deadline: float, check: Callable[[], None]):
        result = []
        errors = []
        done = threading.Event()

        def exchange():
            try:
                write_frame(self.process.stdin, (url, dict(params), deadline), COMMAND_LIMIT)
                # Only our child writes this private pipe, using JSON-decoded HTTP data.
                result.append(read_frame(self.process.stdout, RESPONSE_LIMIT))
            except Exception as error:
                errors.append(error)
            finally:
                done.set()

        reader = None
        try:
            check()
            if deadline <= time.monotonic():
                raise TemporarySourceError()
            reader = threading.Thread(target=exchange, daemon=True)
            reader.start()
            while not done.wait(min(0.1, max(0, deadline - time.monotonic()))):
                if time.monotonic() >= deadline:
                    raise TemporarySourceError()
                check()
            check()
            if errors:
                raise TemporarySourceError() from errors[0]
            if time.monotonic() >= deadline or not result:
                raise TemporarySourceError()
            return result[0]
        except BaseException:
            self.stop()
            raise
        finally:
            if reader is not None:
                reader.join()
