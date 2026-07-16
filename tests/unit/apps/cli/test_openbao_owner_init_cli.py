from __future__ import annotations

from pathlib import Path

import pytest

from apps.cli.commands.openbao_owner_init import OpenBaoOwnerInitCli


def test_openbao_owner_init_requires_exactly_three_recipients(tmp_path: Path) -> None:
    delivery_parent = tmp_path / "owner"
    delivery_parent.mkdir(mode=0o700)
    recipient = delivery_parent / "recipient.asc"
    recipient.write_text("public", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        OpenBaoOwnerInitCli().run(
            [
                "initialize",
                "--address",
                "http://127.0.0.1:8200",
                "--pgp-recipient",
                str(recipient),
                "--delivery-dir",
                str(delivery_parent / "delivery"),
            ]
        )
