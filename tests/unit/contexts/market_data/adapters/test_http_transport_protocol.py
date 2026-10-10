import io
import struct
import subprocess
import sys

import pytest

from trading.contexts.market_data.adapters.outbound.http_transport_worker import (
    read_frame,
    write_frame,
)


class PartialPipe(io.BytesIO):
    def read(self, count: int | None = -1):
        return super().read(3 if count is None or count < 0 else min(3, count))

    def write(self, value):
        return super().write(value[:3])


def test_private_protocol_handles_partial_pipe_reads_and_writes():
    pipe = PartialPipe()
    value = ('ok', 200, {}, [[1, '1.5', '2.5']] * 1000)
    write_frame(pipe, value, 32768)
    pipe.seek(0)
    assert read_frame(pipe, 32768) == value


def test_private_protocol_rejects_oversized_frame_before_reading_body():
    with pytest.raises(ValueError, match='frame_too_large'):
        read_frame(io.BytesIO(struct.pack('!I', 100000)), 1024)
    with pytest.raises(ValueError, match='frame_too_large'):
        write_frame(io.BytesIO(), 'x'*1000, 10)


@pytest.mark.parametrize('data', [b'\x00\x00', struct.pack('!I', 100)+b'x'])
def test_private_protocol_rejects_partial_eof(data):
    with pytest.raises(EOFError):
        read_frame(io.BytesIO(data), 1024)


def test_worker_import_does_not_load_application_or_numpy():
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; import trading.contexts.market_data.adapters.outbound.http_transport_worker; '
         'assert "numpy" not in sys.modules; '
         'assert not any(k.startswith("trading.contexts.market_data.application") '
         'for k in sys.modules)'],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
