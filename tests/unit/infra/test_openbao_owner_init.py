from __future__ import annotations

import base64
import json
import stat
from pathlib import Path
from typing import Any

import pytest

from infra.openbao import owner_init


def _private_directory(path: Path) -> Path:
    path.mkdir()
    path.chmod(0o700)
    return path


def _recipients(root: Path) -> tuple[Path, Path, Path]:
    paths = tuple(root / f"recipient-{index}.asc" for index in range(1, 4))
    for index, path in enumerate(paths, start=1):
        path.write_text(f"public-recipient-{index}", encoding="utf-8")
        path.chmod(0o600)
    return paths  # type: ignore[return-value]


def test_owner_init_writes_only_encrypted_delivery_and_requires_sealed_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner_root = _private_directory(tmp_path / "owner")
    recipients = _recipients(owner_root)
    requests: list[tuple[str, str, dict[str, Any] | None]] = []
    health = iter((501, 503))

    monkeypatch.setattr(owner_init, "_verify_public_pgp_inputs", lambda _paths: None)
    monkeypatch.setattr(owner_init, "_health_status", lambda _address: next(health))

    def request(
        self: owner_init._Client,  # noqa: SLF001
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        requests.append((method, path, payload))
        assert path == "/v1/sys/init"
        return {
            "keys_base64": [
                base64.b64encode(b"encrypted-share-one").decode("ascii"),
                base64.b64encode(b"encrypted-share-two").decode("ascii"),
                base64.b64encode(b"encrypted-share-three").decode("ascii"),
            ],
            "root_token": base64.b64encode(b"encrypted-admin").decode("ascii"),
        }

    monkeypatch.setattr(owner_init._Client, "request_json", request)  # noqa: SLF001

    result = owner_init.initialize_owner_custody(
        address="http://127.0.0.1:8200",
        recipient_paths=recipients,
        delivery_dir=owner_root / "delivery",
    )

    assert result.status == "initialized"
    delivery = owner_root / "delivery"
    assert stat.S_IMODE(delivery.stat().st_mode) == 0o700
    assert (delivery / "unseal-share-1.pgp").read_bytes() == b"encrypted-share-one"
    assert (delivery / "unseal-share-2.pgp").read_bytes() == b"encrypted-share-two"
    assert (delivery / "unseal-share-3.pgp").read_bytes() == b"encrypted-share-three"
    assert (delivery / "initial-admin.pgp").read_bytes() == b"encrypted-admin"
    assert stat.S_IMODE((delivery / "initial-admin.pgp").stat().st_mode) == 0o600
    metadata_bytes = (delivery / "owner-init.json").read_bytes()
    assert b"encrypted-share" not in metadata_bytes
    assert b"encrypted-admin" not in metadata_bytes
    assert requests == [
        (
            "PUT",
            "/v1/sys/init",
            {
                "secret_shares": 3,
                "secret_threshold": 2,
                "pgp_keys": [
                    base64.b64encode(path.read_bytes()).decode("ascii") for path in recipients
                ],
                "root_token_pgp_key": base64.b64encode(recipients[0].read_bytes()).decode(
                    "ascii"
                ),
            },
        )
    ]


def test_owner_init_is_idempotent_only_for_verified_existing_delivery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner_root = _private_directory(tmp_path / "owner")
    recipients = _recipients(owner_root)
    destination = owner_root / "delivery"
    destination.mkdir()
    destination.chmod(0o700)
    for filename in (
        "unseal-share-1.pgp",
        "unseal-share-2.pgp",
        "unseal-share-3.pgp",
        "initial-admin.pgp",
    ):
        path = destination / filename
        path.write_bytes(b"ciphertext")
        path.chmod(0o600)
    metadata = {
        "initial_admin_encrypted": True,
        "recipient_count": 3,
        "schema": owner_init.SCHEMA,
        "status": "prepared",
        "unseal_shares": 3,
        "unseal_threshold": 2,
    }
    metadata_path = destination / "owner-init.json"
    metadata_path.write_bytes(json.dumps(metadata, sort_keys=True).encode("utf-8"))
    metadata_path.chmod(0o600)

    monkeypatch.setattr(owner_init, "_verify_public_pgp_inputs", lambda _paths: None)
    monkeypatch.setattr(owner_init, "_health_status", lambda _address: 503)
    monkeypatch.setattr(
        owner_init._Client,  # noqa: SLF001
        "request_json",
        lambda *_args, **_kwargs: pytest.fail("existing delivery must not call init"),
    )

    result = owner_init.initialize_owner_custody(
        address="http://127.0.0.1:8200",
        recipient_paths=recipients,
        delivery_dir=destination,
    )

    assert result.status == "already_initialized"


def test_owner_init_rejects_duplicate_public_recipients_without_calling_openbao(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner_root = _private_directory(tmp_path / "owner")
    recipients = _recipients(owner_root)
    recipients[2].write_bytes(recipients[1].read_bytes())

    monkeypatch.setattr(owner_init, "_verify_public_pgp_inputs", lambda _paths: None)
    monkeypatch.setattr(
        owner_init._Client,  # noqa: SLF001
        "request_json",
        lambda *_args, **_kwargs: pytest.fail("duplicate recipients must not call init"),
    )

    with pytest.raises(owner_init.OwnerInitError, match="distinct"):
        owner_init.initialize_owner_custody(
            address="http://127.0.0.1:8200",
            recipient_paths=recipients,
            delivery_dir=owner_root / "delivery",
        )


def test_service_provisioning_creates_separate_wrapped_delivery_and_revokes_admin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    owner_root = _private_directory(tmp_path / "owner")
    token_file = owner_root / "administrator-token"
    token_file.write_text("admin-token", encoding="utf-8")
    token_file.chmod(0o600)
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    monkeypatch.setattr(owner_init, "_health_status", lambda _address: 200)

    def request(
        self: owner_init._Client,  # noqa: SLF001
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        **_kwargs: object,
    ) -> dict[str, Any]:
        calls.append((method, path, payload))
        if path == "/v1/sys/mounts":
            return {}
        if path == "/v1/sys/auth":
            return {}
        if path == "/v1/transit/keys/roehub-exchange-credentials" and method == "GET":
            return {"data": {"exportable": False}}
        if path.endswith("/role-id"):
            return {"data": {"role_id": f"role-{len(calls)}"}}
        if path.endswith("/secret-id"):
            role = path.split("/")[-2]
            return {
                "wrap_info": {
                    "creation_path": f"auth/approle/role/{role}/secret-id",
                    "token": f"wrapped-{role}",
                    "ttl": 300,
                }
            }
        return {}

    monkeypatch.setattr(owner_init._Client, "request_json", request)  # noqa: SLF001

    result = owner_init.provision_service_credentials(
        address="http://127.0.0.1:8200",
        administrator_token_file=token_file,
        delivery_dir=owner_root / "services",
    )

    assert result.status == "provisioned"
    delivery = owner_root / "services"
    assert (delivery / "service-delivery.json").is_file()
    for role in owner_init._SERVICE_ROLES:  # noqa: SLF001
        service_root = delivery / role.delivery_name
        assert {entry.name for entry in service_root.iterdir()} == {
            "role-id",
            "wrapped-secret-id",
        }
        assert stat.S_IMODE((service_root / "wrapped-secret-id").stat().st_mode) == 0o600
    assert calls[-1] == ("POST", "/v1/auth/token/revoke-self", {})
