"""Private pipe worker: bounded HTTP body, absolute deadline and owner-death stop.

Empty package ancestors keep this worker independent of application imports.
Pickle is confined to inherited local pipes; provider bytes are decoded as JSON.
"""

import json
import logging
import os
import pickle
import queue
import struct
import sys
import threading
import time

import requests

COMMAND_LIMIT = 64 * 1024
RESPONSE_LIMIT = 32 * 1024 * 1024


def read_frame(stream, limit):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            part = stream.read(min(65536, size-len(result)))
            if not part:
                raise EOFError()
            result.extend(part)
        return result
    size = struct.unpack('!I', exact(4))[0]
    if size > limit:
        raise ValueError('private_http_frame_too_large')
    return pickle.loads(exact(size))


def write_frame(stream, value, limit):
    data = pickle.dumps(value)
    if len(data) > limit:
        raise ValueError('private_http_frame_too_large')
    view = memoryview(struct.pack('!I', len(data)) + data)
    while view:
        written = stream.write(view)
        if not written:
            raise EOFError()
        view = view[written:]
    stream.flush()


def main():
    logging.disable(logging.CRITICAL)
    commands = queue.Queue(maxsize=1)

    def read_commands():
        while True:
            try:
                command = read_frame(sys.stdin.buffer, COMMAND_LIMIT)
            except (EOFError, OSError, ValueError, pickle.UnpicklingError):
                # This reader remains active during HTTP. Parent death closes stdin.
                os._exit(1)
            commands.put(command)

    threading.Thread(target=read_commands, daemon=True).start()

    def respond(value):
        write_frame(sys.stdout.buffer, value, RESPONSE_LIMIT)

    with requests.Session() as session:
        session.trust_env = False
        while True:
            url, params, deadline = commands.get()
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                respond(('temporary',))
                continue
            watchdog = threading.Timer(timeout, lambda: os._exit(2))
            watchdog.daemon = True
            watchdog.start()
            try:
                with session.get(
                    url, params=params, timeout=timeout, stream=True, allow_redirects=False
                ) as response:
                    content = bytearray()
                    for chunk in response.iter_content(chunk_size=65536):
                        content.extend(chunk)
                        if len(content) > 16 * 1024 * 1024:
                            raise ValueError('source_body_too_large')
                    try:
                        body = json.loads(content)
                    except ValueError:
                        body = None
                    respond(('ok', response.status_code, dict(response.headers), body))
            except requests.RequestException:
                respond(('temporary',))
            except Exception:
                respond(('invalid',))
            finally:
                watchdog.cancel()


if __name__ == '__main__':
    try:
        main()
    finally:
        # The command reader can be blocked in stdin; never await it at interpreter exit.
        os._exit(0)
