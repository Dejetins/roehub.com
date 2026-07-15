from __future__ import annotations

import hashlib
import json
import secrets
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from infra.openbao import snapshot
from trading.contexts.exchange_control.adapters.outbound.openbao_transit import (
    OpenBaoTransitExchangeSecretCipher,
)
from trading.contexts.exchange_control.application.secret_cipher import (
    ExchangeCredentialSecret,
    ExchangeSecretCipherError,
)
from trading.platform.secrets import (
    OpenBaoSecretResolver,
    OpenBaoUnavailableError,
    SecureTokenFile,
)


def _restricted_file(path: Path, value: str) -> Path:
    path.write_text(value, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path


@contextmanager
def _redirect_boundary() -> Iterator[tuple[str, list[str]]]:
    received_credentials: list[str] = []

    class TargetHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            received_credentials.append(self.headers.get("X-Vault-Token", ""))
            self.send_response(204)
            self.end_headers()

        do_POST = do_GET

        def log_message(self, format: str, *args: object) -> None:
            _ = format, args
            return None

    target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)
    target_address = f"http://127.0.0.1:{target.server_port}/capture"

    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(307)
            self.send_header("Location", target_address)
            self.end_headers()

        do_POST = do_GET

        def log_message(self, format: str, *args: object) -> None:
            _ = format, args
            return None

    redirect = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    threads = [
        Thread(target=target.serve_forever, daemon=True),
        Thread(target=redirect.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        yield f"http://127.0.0.1:{redirect.server_port}", received_credentials
    finally:
        redirect.shutdown()
        target.shutdown()
        redirect.server_close()
        target.server_close()
        for thread in threads:
            thread.join(timeout=2)


def test_credential_bearing_clients_reject_cross_origin_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_marker = secrets.token_urlsafe(32)
    credential_path = _restricted_file(tmp_path / "service.credential", credential_marker)
    recipient_path = tmp_path / "recipient.txt"
    recipient_path.write_text("age1test-only-recipient", encoding="utf-8")
    recipient_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    monkeypatch.setattr(snapshot, "_require_age", lambda: None)

    with _redirect_boundary() as (address, received_credentials):
        resolver = OpenBaoSecretResolver(
            address=address,
            token_source=SecureTokenFile(credential_path),
        )
        with pytest.raises(OpenBaoUnavailableError, match="status 307"):
            resolver.resolve("openbao://kv/roehub/oidc/provider-a#client_secret")

        cipher = OpenBaoTransitExchangeSecretCipher(
            address=address,
            credential_source=SecureTokenFile(credential_path),
        )
        with pytest.raises(ExchangeSecretCipherError, match="status 307"):
            cipher.encrypt(ExchangeCredentialSecret(value=secrets.token_urlsafe(24)))

        with pytest.raises(snapshot.SnapshotOperationError, match="status 307"):
            snapshot.backup_snapshot(
                address=address,
                credential_path=credential_path,
                recipient_path=recipient_path,
                destination=tmp_path / "backup.snap.age",
            )

    assert received_credentials == []


def test_snapshot_rejects_symlinked_credential_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_credential = _restricted_file(
        tmp_path / "real.credential",
        secrets.token_urlsafe(32),
    )
    linked_credential = tmp_path / "linked.credential"
    linked_credential.symlink_to(real_credential)
    recipient_path = _restricted_file(
        tmp_path / "recipient.txt",
        "age1test-only-recipient",
    )
    monkeypatch.setattr(snapshot, "_require_age", lambda: None)
    monkeypatch.setattr(
        snapshot,
        "open_without_redirect",
        lambda *_args, **_kwargs: pytest.fail("HTTP must not be attempted"),
    )

    with pytest.raises(OpenBaoUnavailableError, match="unavailable"):
        snapshot.backup_snapshot(
            address="http://127.0.0.1:8200",
            credential_path=linked_credential,
            recipient_path=recipient_path,
            destination=tmp_path / "backup.snap.age",
        )


def test_restore_rejects_ciphertext_metadata_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _restricted_file(tmp_path / "backup.snap.age", "encrypted-placeholder")
    metadata = {
        "schema": snapshot.SCHEMA,
        "operation": "backup",
        "status": "passed",
        "encrypted": True,
        "ciphertext_bytes": source.stat().st_size,
        "ciphertext_sha256": hashlib.sha256(b"different").hexdigest(),
    }
    metadata_path = source.with_suffix(source.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    metadata_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    recovery_path = _restricted_file(
        tmp_path / "recovery.agekey",
        "AGE-SECRET-KEY-TEST-ONLY",
    )
    credential_path = _restricted_file(
        tmp_path / "restore.credential",
        secrets.token_urlsafe(32),
    )
    monkeypatch.setattr(snapshot, "_require_age", lambda: None)

    with pytest.raises(snapshot.SnapshotOperationError, match="does not match"):
        snapshot.restore_snapshot(
            address="http://127.0.0.1:8200",
            credential_path=credential_path,
            recovery_path=recovery_path,
            source=source,
        )


def test_force_restore_rejects_non_fresh_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        "/v1/sys/mounts": {
            "data": {
                "cubbyhole/": {},
                "identity/": {},
                "kv/": {},
                "sys/": {},
            }
        },
        "/v1/sys/auth": {"data": {"token/": {}}},
        "/v1/sys/policies/acl": {
            "data": {"keys": ["default", "response-wrapping", "root"]}
        },
    }
    monkeypatch.setattr(
        snapshot,
        "_request_json",
        lambda *, path, **_kwargs: responses[path],
    )

    with pytest.raises(snapshot.SnapshotOperationError, match="fresh OpenBao storage"):
        snapshot._verify_fresh_storage(
            address="http://127.0.0.1:8200",
            credential=secrets.token_urlsafe(32),
            timeout_seconds=1.0,
        )
