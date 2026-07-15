from __future__ import annotations

import argparse
import base64
import json
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import UUID

import psycopg
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

_PROTOCOL = "roehub.plugin.rpc/v1alpha1"
_SIGNING_CONTEXT = b"roehub-plugin-service-identity-v1alpha1\0"
_CONFIG = json.loads(Path("/plugin/fixture-config.json").read_text(encoding="utf-8"))
_PUBLIC_KEY = Ed25519PublicKey.from_public_bytes(base64.b64decode(_CONFIG["public_key_b64"]))
_USED_NONCES: set[str] = set()
_NONCE_LOCK = Lock()
_PACKAGE_DIGEST = ""
_PACKAGE_VERSION = ""


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
    if header != {"alg": "Ed25519", "kid": "stage13-gateway", "typ": "RoehubPlugin"}:
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
        or payload.get("package_digest") != _PACKAGE_DIGEST
        or payload.get("package_version") != _PACKAGE_VERSION
        or payload.get("capability") != capability
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


def _query_frame(payload: dict[str, Any], claims: dict[str, Any]) -> dict[str, Any]:
    limits = payload.get("limits")
    if (
        payload.get("contract") != "DataSourceQuery/v1"
        or payload.get("read_only") is not True
        or payload.get("dimensions") != ["timestamp"]
        or payload.get("measures") != ["pnl", "drawdown"]
        or not isinstance(limits, dict)
    ):
        raise ValueError("bounded data-source query is invalid")
    row_limit = min(max(int(limits.get("rows", 0)), 1), 1000)
    timeout_ms = min(max(int(limits.get("timeout_ms", 0)), 50), 5000)
    dataset = payload.get("dataset")
    if dataset not in {
        "portfolio.pnl",
        "portfolio.slow",
        "portfolio.oversized",
        "portfolio.ignore-timeout",
    }:
        raise ValueError("dataset is unsupported")
    if dataset == "portfolio.ignore-timeout":
        time.sleep(2)
    with psycopg.connect(_CONFIG["database_dsn"]) as connection:
        with connection.transaction(), connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, true)",
                (f"{timeout_ms}ms",),
            )
            if dataset == "portfolio.slow":
                cursor.execute("SELECT pg_sleep(2)")
            cursor.execute(
                """SELECT observed_at, pnl, drawdown
                   FROM stage13_portfolio_points
                   WHERE organization_id = %s
                   ORDER BY observed_at
                   LIMIT %s""",
                (claims["organization_id"], row_limit),
            )
            records = cursor.fetchall()
    if dataset == "portfolio.oversized":
        records = [records[index % len(records)] for index in range(1_000)]
    now = datetime.now(UTC)
    return {
        "contract": "RoehubDataFrame/v1",
        "frame_id": "stage13.external.portfolio",
        "title": "Controlled external portfolio",
        "columns": [
            {
                "key": "timestamp",
                "label": "Observed at",
                "data_type": "timestamp",
                "role": "dimension",
                "unit": {"kind": "timestamp", "symbol": "UTC", "scale": 1.0},
                "nullable": False,
            },
            {
                "key": "pnl",
                "label": "Portfolio PnL",
                "data_type": "number",
                "role": "measure",
                "unit": {"kind": "currency", "symbol": "USD", "scale": 1.0},
                "nullable": False,
            },
            {
                "key": "drawdown",
                "label": "Drawdown",
                "data_type": "number",
                "role": "measure",
                "unit": {"kind": "percent", "symbol": "%", "scale": 1.0},
                "nullable": False,
            },
        ],
        "rows": [
            {
                "timestamp": record[0].astimezone(UTC).isoformat(),
                "pnl": float(record[1]),
                "drawdown": float(record[2]),
            }
            for record in records
        ],
        "metadata": {
            "source_label": "Controlled external PostgreSQL fixture",
            "query_label": "Organization-scoped portfolio series",
            "generated_at": now.isoformat(),
            "attributes": {"read_only": True},
        },
        "freshness": {
            "status": "fresh",
            "observed_at": now.isoformat(),
            "age_seconds": 0,
            "max_age_seconds": 60,
        },
        "notices": [],
        "partial": False,
        "errors": [],
    }


def _contributions() -> dict[str, Any]:
    query = {
        "contract": "DataSourceQuery/v1",
        "dataset": "portfolio.pnl",
        "dimensions": ["timestamp"],
        "measures": ["pnl", "drawdown"],
        "filters": [],
        "row_limit": 200,
        "byte_limit": 262144,
        "point_limit": 1000,
        "timeout_ms": 3000,
        "read_only": True,
    }
    panel = {
        "contract": "RoehubPanelContribution/v1",
        "contribution_id": "stage13.portfolio.pnl",
        "title": "External portfolio series",
        "description": "Bounded organization-scoped portfolio data.",
        "renderer": "trading-time-series",
        "source": {"instance_id": _CONFIG["instance_id"], "query": query},
        "presentation": {
            "x_column": "timestamp",
            "y_columns": ["pnl", "drawdown"],
            "table_columns": ["timestamp", "pnl", "drawdown"],
            "default_view": "visual",
        },
    }
    app = {
        "contract": "RoehubAppContribution/v1",
        "contribution_id": "stage13.portfolio.research",
        "title": "Portfolio research",
        "description": "Composition of host-rendered declarative panels.",
        "sections": [
            {
                "section_id": "overview",
                "title": "Overview",
                "panel_contribution_ids": [panel["contribution_id"]],
            }
        ],
    }
    return {"panel": panel, "app": app}


class Handler(BaseHTTPRequestHandler):
    server_version = "RoehubStage13Fixture/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/v1alpha1/health":
            self._json(404, {"contract": "PluginResponse/v1alpha1", "status": "not_found"})
            return
        if self.headers.get("X-Roehub-Plugin-Protocol") != _PROTOCOL:
            self._json(426, {"contract": "PluginResponse/v1alpha1", "status": "protocol"})
            return
        try:
            _claims(self.headers.get("Authorization", ""), capability="data.read")
        except (InvalidSignature, KeyError, TypeError, ValueError):
            self._json(403, {"contract": "PluginResponse/v1alpha1", "status": "forbidden"})
            return
        self._json(200, {"contract": "PluginResponse/v1alpha1", "status": "ready"})

    def do_POST(self) -> None:  # noqa: N802
        route = {
            "/v1alpha1/data-source/query": "data.read",
            "/v1alpha1/panel/describe": "panel.describe",
        }.get(self.path)
        if route is None:
            self._json(404, {"contract": "PluginResponse/v1alpha1", "status": "not_found"})
            return
        if self.headers.get("X-Roehub-Plugin-Protocol") != _PROTOCOL:
            self._json(426, {"contract": "PluginResponse/v1alpha1", "status": "protocol"})
            return
        try:
            claims = _claims(self.headers.get("Authorization", ""), capability=route)
        except (InvalidSignature, KeyError, TypeError, ValueError):
            self._json(403, {"contract": "PluginResponse/v1alpha1", "status": "forbidden"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= 65536:
            self._json(413, {"contract": "PluginResponse/v1alpha1", "status": "too_large"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("request must be an object")
            if route == "data.read":
                frame = _query_frame(payload, claims)
                self._json(
                    200,
                    {
                        "contract": "PluginResponse/v1alpha1",
                        "status": "succeeded",
                        "frame": frame,
                    },
                )
            else:
                self._json(
                    200,
                    {
                        "contract": "PluginResponse/v1alpha1",
                        "status": "succeeded",
                        **_contributions(),
                    },
                )
        except psycopg.errors.QueryCanceled:
            self._json(
                504,
                {"contract": "PluginResponse/v1alpha1", "status": "cancelled"},
            )
        except psycopg.Error:
            self._json(
                503,
                {"contract": "PluginResponse/v1alpha1", "status": "unavailable"},
            )
        except (KeyError, TypeError, ValueError):
            self._json(422, {"contract": "PluginResponse/v1alpha1", "status": "invalid"})

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
    global _PACKAGE_DIGEST, _PACKAGE_VERSION
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-digest", required=True)
    parser.add_argument("--package-version", required=True)
    args = parser.parse_args()
    _PACKAGE_DIGEST = args.package_digest
    _PACKAGE_VERSION = args.package_version
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


if __name__ == "__main__":
    main()
