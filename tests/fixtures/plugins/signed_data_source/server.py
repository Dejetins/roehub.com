from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_PROTOCOL = "roehub.plugin.rpc/v1alpha1"
_SIGNING_CONTEXT = b"roehub-plugin-service-identity-v1alpha1\0"
_CONFIG = json.loads(Path("/plugin/fixture-config.json").read_text(encoding="utf-8"))
_PUBLIC_KEY = Ed25519PublicKey.from_public_bytes(base64.b64decode(_CONFIG["public_key_b64"]))
_COUNTERS = {"health": 0, "metrics": 0, "query": 0, "rejected": 0}
_USED_NONCES: set[str] = set()
_NONCE_LOCK = Lock()


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.b64decode(value + padding, altchars=b"-_", validate=True)


def _claims(header_value: str, *, capability: str) -> dict[str, Any]:
    scheme, identity = header_value.split(" ", 1)
    if scheme != "RoehubPluginIdentity":
        raise ValueError("identity scheme is invalid")
    header_segment, payload_segment, signature_segment = identity.split(".")
    header = json.loads(_b64url_decode(header_segment))
    payload = json.loads(_b64url_decode(payload_segment))
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    if header != {"alg": "Ed25519", "kid": "stage12-gateway", "typ": "RoehubPlugin"}:
        raise ValueError("identity header is invalid")
    _PUBLIC_KEY.verify(
        _b64url_decode(signature_segment),
        _SIGNING_CONTEXT + signing_input,
    )
    now = int(time.time())
    if (
        payload.get("contract") != "PluginServiceIdentity/v1alpha1"
        or payload.get("organization_id") != _CONFIG["organization_id"]
        or payload.get("instance_id") != _CONFIG["instance_id"]
        or payload.get("package_digest") != _CONFIG["package_digest"]
        or payload.get("package_version") != _CONFIG["package_version"]
        or payload.get("capability") != capability
        or capability not in _CONFIG["allowed_capabilities"]
        or not payload.get("issued_at") <= now < payload.get("expires_at")
        or payload.get("expires_at") - payload.get("issued_at") > 60
    ):
        raise ValueError("identity scope is invalid")
    nonce_id = payload.get("nonce_id")
    UUID(nonce_id)
    with _NONCE_LOCK:
        if nonce_id in _USED_NONCES:
            raise ValueError("identity nonce was already used")
        _USED_NONCES.add(nonce_id)
    return payload


def _filesystem_probe() -> str:
    try:
        Path("/write-probe").write_text("forbidden", encoding="utf-8")
    except OSError:
        return "denied"
    return "unexpectedly_writable"


def _network_probe(host: str, port: int) -> str:
    try:
        with socket.create_connection((host, port), timeout=0.4):
            return "unexpectedly_reachable"
    except OSError:
        return "denied"


class Handler(BaseHTTPRequestHandler):
    server_version = "RoehubFixturePlugin/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1alpha1/health":
            self._serve(capability="data.read", kind="health")
            return
        if self.path == "/v1alpha1/metrics":
            self._serve(capability="data.read", kind="metrics")
            return
        self._json(404, {"contract": "PluginResponse/v1alpha1", "status": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        capability_by_path = {
            "/v1alpha1/data-source/query": ("data.read", "query"),
            "/v1alpha1/panel/describe": ("panel.describe", "panel"),
            "/v1alpha1/app/action": ("app.action", "action"),
            "/v1alpha1/notification-provider/send": (
                "notification.send",
                "notification",
            ),
        }
        route = capability_by_path.get(self.path)
        if route is None:
            self._json(404, {"contract": "PluginResponse/v1alpha1", "status": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 65536:
            self._json(413, {"contract": "PluginResponse/v1alpha1", "status": "too_large"})
            return
        if length:
            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeError, json.JSONDecodeError):
                self._json(400, {"contract": "PluginResponse/v1alpha1", "status": "invalid"})
                return
            if not isinstance(payload, dict):
                self._json(400, {"contract": "PluginResponse/v1alpha1", "status": "invalid"})
                return
        self._serve(capability=route[0], kind=route[1])

    def _serve(self, *, capability: str, kind: str) -> None:
        if self.headers.get("X-Roehub-Plugin-Protocol") != _PROTOCOL:
            self._json(426, {"contract": "PluginResponse/v1alpha1", "status": "protocol"})
            return
        try:
            claims = _claims(self.headers.get("Authorization", ""), capability=capability)
        except (InvalidSignature, KeyError, TypeError, ValueError):
            _COUNTERS["rejected"] += 1
            self._json(403, {"contract": "PluginResponse/v1alpha1", "status": "forbidden"})
            return
        _COUNTERS[kind] = _COUNTERS.get(kind, 0) + 1
        if kind == "health":
            data: dict[str, Any] = {
                "contract": "PluginResponse/v1alpha1",
                "status": "ready",
                "protocol": _PROTOCOL,
                "uid": os.geteuid(),
                "filesystem_write": _filesystem_probe(),
                "platform_database": _network_probe("postgresql", 5432),
                "external_egress": _network_probe("1.1.1.1", 443),
            }
        elif kind == "metrics":
            data = {
                "contract": "PluginResponse/v1alpha1",
                "status": "ready",
                "counters": dict(_COUNTERS),
            }
        else:
            data = {
                "contract": "PluginResponse/v1alpha1",
                "status": "succeeded",
                "capability": capability,
                "organization_id": claims["organization_id"],
                "instance_id": claims["instance_id"],
                "package_digest": claims["package_digest"],
                "rows": [{"timestamp": 1, "value": 42.0}] if kind == "query" else [],
            }
        self._json(200, data)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Roehub-Plugin-Protocol", _PROTOCOL)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        _ = format, args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-digest", required=True)
    parser.add_argument("--package-version", required=True)
    args = parser.parse_args()
    _CONFIG["package_digest"] = args.package_digest
    _CONFIG["package_version"] = args.package_version
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


if __name__ == "__main__":
    main()
