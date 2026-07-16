"""CLI boundary for secure owner-operated OpenBao bootstrap."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from infra.openbao.owner_init import (
    OwnerInitError,
    initialize_owner_custody,
    provision_service_credentials,
)


class OpenBaoOwnerInitCli:
    """Issue encrypted custody material without printing any credential value."""

    def run(self, argv: Sequence[str]) -> int:
        parser = _build_parser()
        args = parser.parse_args(list(argv))
        try:
            if args.action == "initialize":
                result = initialize_owner_custody(
                    address=args.address,
                    recipient_paths=tuple(args.pgp_recipient),
                    delivery_dir=args.delivery_dir,
                )
            else:
                result = provision_service_credentials(
                    address=args.address,
                    administrator_token_file=args.administrator_token_file,
                    delivery_dir=args.delivery_dir,
                )
        except OwnerInitError as error:
            parser.error(str(error))
        print(json.dumps(result.as_dict(), sort_keys=True))
        return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="roehubctl openbao-owner-init")
    actions = parser.add_subparsers(dest="action", required=True)

    initialize = actions.add_parser("initialize")
    initialize.add_argument("--address", required=True)
    initialize.add_argument("--pgp-recipient", action="append", type=Path, required=True)
    initialize.add_argument("--delivery-dir", type=Path, required=True)

    provision = actions.add_parser("provision-services")
    provision.add_argument("--address", required=True)
    provision.add_argument("--administrator-token-file", type=Path, required=True)
    provision.add_argument("--delivery-dir", type=Path, required=True)
    return parser


__all__ = ["OpenBaoOwnerInitCli"]
