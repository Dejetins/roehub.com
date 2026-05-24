from __future__ import annotations

from trading.contexts.exchange_control.adapters.outbound import postgres_connections


def test_fingerprint_text_accepts_transit_text_values() -> None:
    assert postgres_connections._fingerprint_text("hmac-sha256:abc123") == (
        "hmac-sha256:abc123"
    )
    assert postgres_connections._fingerprint_text(memoryview(b"hmac-sha256:abc123")) == (
        "hmac-sha256:abc123"
    )


def test_fingerprint_text_maps_legacy_binary_hashes_to_stable_hex() -> None:
    legacy_hash = bytes([0, 160, 255]) + (b"x" * 29)

    assert postgres_connections._fingerprint_text(legacy_hash) == (
        "legacy-bytea-sha256:00a0ff" + ("78" * 29)
    )
