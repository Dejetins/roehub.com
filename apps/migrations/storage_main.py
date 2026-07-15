"""CLI for the greenfield Roehub storage lifecycle."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from apps.migrations.storage import (
    StorageLifecycleError,
    bootstrap_storage,
    build_storage_status,
    load_storage_config,
    load_storage_endpoints,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roehub-storage")
    parser.add_argument(
        "command",
        choices=("bootstrap", "capabilities", "readiness", "status"),
        help="Storage lifecycle operation. Readiness and status require applied schemas.",
    )
    parser.add_argument(
        "--service-config",
        required=True,
        help="Generated service-config.json; never pass raw credentials here.",
    )
    parser.add_argument("--postgres-manifest", default="")
    parser.add_argument("--clickhouse-manifest", default="")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path for the secret-free status document.",
    )
    return parser


def _resolve_path(raw: str, *, default: Path, label: str) -> Path:
    path = Path(raw).expanduser().resolve() if raw.strip() else default.resolve()
    if not path.is_file():
        raise StorageLifecycleError(f"{label} does not exist")
    return path


def _write_status(payload: dict[str, object], path: str) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.strip():
        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main(argv: list[str] | None = None) -> int:
    """Execute one storage lifecycle operation without exposing endpoint values."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    try:
        service_config = _resolve_path(
            args.service_config,
            default=repo_root / "service-config.json",
            label="generated service configuration",
        )
        postgres_manifest = _resolve_path(
            args.postgres_manifest,
            default=repo_root / "migrations" / "postgres" / "manifest.json",
            label="PostgreSQL migration manifest",
        )
        clickhouse_manifest = _resolve_path(
            args.clickhouse_manifest,
            default=repo_root / "migrations" / "clickhouse" / "manifest.json",
            label="ClickHouse migration manifest",
        )
        config = load_storage_config(service_config)
        endpoints = load_storage_endpoints(config=config, environ=os.environ)
        if args.command == "bootstrap":
            with contextlib.redirect_stdout(sys.stderr):
                payload = bootstrap_storage(
                    config=config,
                    endpoints=endpoints,
                    repo_root=repo_root,
                    postgres_manifest=postgres_manifest,
                    clickhouse_manifest=clickhouse_manifest,
                )
        else:
            require_schema = args.command in {"readiness", "status"}
            payload = build_storage_status(
                config=config,
                endpoints=endpoints,
                repo_root=repo_root,
                postgres_manifest=postgres_manifest,
                clickhouse_manifest=clickhouse_manifest,
                require_schema=require_schema,
            )
        _write_status(payload, args.output_json)
    except StorageLifecycleError as error:
        print(f"Storage lifecycle failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:  # noqa: BLE001
        print(
            f"Storage lifecycle failed safely: {type(error).__name__}; details redacted",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
