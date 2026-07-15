from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import Mapping

from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_cas import LocalCasBlobStore
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres import (
    PostgresArtifactCatalogRepository,
)
from trading.contexts.backtest_artifacts.application import ArtifactStoreService
from trading.contexts.backtest_artifacts.domain import ArtifactStoreError
from trading.shared_kernel.primitives import OrganizationId


class ArtifactsCli:
    def run(self, argv: list[str]) -> int:
        parser = argparse.ArgumentParser(prog="roehubctl artifacts")
        subparsers = parser.add_subparsers(dest="action", required=True)
        install = subparsers.add_parser("install")
        install.add_argument("bundle", type=Path)
        install.add_argument("--organization-id", required=True)
        install.add_argument("--catalog-dsn-file", type=Path, required=True)
        install.add_argument("--cas-root", type=Path, required=True)
        install.add_argument("--publisher-keys", type=Path, required=True)
        install.add_argument("--quota-bytes", type=int)
        args = parser.parse_args(argv)

        try:
            organization_id = OrganizationId.from_string(args.organization_id)
        except ValueError as error:
            parser.error(f"--organization-id is invalid: {error}")
        dsn = _read_private_text(args.catalog_dsn_file, parser=parser)
        public_keys = _read_public_keys(args.publisher_keys, parser=parser)
        service = ArtifactStoreService(
            blobs=LocalCasBlobStore(root=args.cas_root),
            catalog=PostgresArtifactCatalogRepository(dsn=dsn),
            trusted_public_keys=public_keys,
        )
        try:
            if args.quota_bytes is not None:
                service.set_quota(
                    organization_id=organization_id,
                    max_bytes=args.quota_bytes,
                )
            manifest = service.install_bundle(
                organization_id=organization_id,
                bundle_root=args.bundle,
            )
        except ArtifactStoreError as error:
            print(
                json.dumps(
                    {
                        "schema": "io.roehub.artifact-install/v1",
                        "status": "failed",
                        "code": error.code,
                    },
                    sort_keys=True,
                )
            )
            return 2
        print(
            json.dumps(
                {
                    "schema": "io.roehub.artifact-install/v1",
                    "status": "installed",
                    "bundle_id": manifest.bundle_id,
                    "version": manifest.version,
                    "manifest_digest": manifest.manifest_digest,
                },
                sort_keys=True,
            )
        )
        return 0


def _read_private_text(path: Path, *, parser: argparse.ArgumentParser) -> str:
    expanded = path.expanduser()
    try:
        descriptor = os.open(expanded, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        parser.error(f"--catalog-dsn-file cannot be read: {error}")
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or stat.S_IMODE(file_stat.st_mode) & 0o077:
            parser.error("--catalog-dsn-file must be a regular file with mode 0600")
        with os.fdopen(descriptor, encoding="utf-8", closefd=False) as stream:
            value = stream.read(64 * 1024 + 1).strip()
        if len(value) > 64 * 1024:
            parser.error("--catalog-dsn-file is too large")
    finally:
        os.close(descriptor)
    if not value:
        parser.error("--catalog-dsn-file is empty")
    return value


def _read_public_keys(path: Path, *, parser: argparse.ArgumentParser) -> Mapping[str, str]:
    try:
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"--publisher-keys is invalid: {error}")
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in payload.items()
    ):
        parser.error("--publisher-keys must be a string-to-string JSON object")
    return payload


__all__ = ["ArtifactsCli"]
