import logging
import socket

from trading.contexts.market_data.adapters.outbound.clients.connection_events import (
    ConnectionEvents,
)


def test_expected_outage_is_coalesced_and_restored_without_payload(caplog):
    clock = [0.0]
    events = ConnectionEvents(logging.getLogger("test.connection"), 1, lambda: clock[0])
    with caplog.at_level(logging.INFO):
        for _ in range(100):
            events.failed(socket.gaierror("secret-provider-url"))
        clock[0] = 61
        events.failed(TimeoutError("secret-provider-url"))
        clock[0] = 62
        events.connected()
    assert len(caplog.records) == 3
    assert "attempts=101" in caplog.text
    assert "outage_s=62.0" in caplog.text
    assert "stream_connection_restored" in caplog.text
    assert "secret-provider-url" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)
