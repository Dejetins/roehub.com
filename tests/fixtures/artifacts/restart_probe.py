from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_cas import LocalCasBlobStore
from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres import (
    PostgresArtifactCatalogRepository,
)
from trading.contexts.backtest_artifacts.application import ArtifactStoreService
from trading.shared_kernel.primitives import OrganizationId


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn-file", type=Path, required=True)
    parser.add_argument("--cas-root", type=Path, required=True)
    parser.add_argument("--publisher-keys", type=Path, required=True)
    parser.add_argument("--organization-id", required=True)
    parser.add_argument("--manifest-digest", required=True)
    args = parser.parse_args()

    service = ArtifactStoreService(
        blobs=LocalCasBlobStore(root=args.cas_root),
        catalog=PostgresArtifactCatalogRepository(dsn=args.dsn_file.read_text().strip()),
        trusted_public_keys=json.loads(args.publisher_keys.read_text()),
    )
    payload = service.read_entry(
        organization_id=OrganizationId.from_string(args.organization_id),
        manifest_digest=args.manifest_digest,
        path="demo/hello.json",
    )
    if not payload.startswith(b"{"):
        return 1
    print(json.dumps({"schema": "io.roehub.artifact-restart-proof/v1", "status": "passed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
