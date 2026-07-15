from __future__ import annotations

import pytest

from trading.platform.secrets import SecretKind, SecretReference, SecretReferenceError


def test_typed_reference_parses_kind_field_and_version() -> None:
    reference = SecretReference.parse(
        "openbao://kv/roehub/oidc/provider-a?version=7#client_secret",
        expected_kind=SecretKind.OIDC,
    )

    assert reference.kind is SecretKind.OIDC
    assert reference.resource == ("provider-a",)
    assert reference.field == "client_secret"
    assert reference.version == 7
    assert reference.kv_v2_path == "kv/data/roehub/oidc/provider-a"
    assert "provider-a" not in repr(reference)


@pytest.mark.parametrize(
    "raw",
    [
        "openbao://kv/roehub/exchange/connection-a",
        "openbao://kv/roehub/unknown/resource#value",
        "openbao://kv/another-root/oidc/provider-a#client_secret",
        "openbao://kv/roehub/oidc/provider-a?version=0#client_secret",
        "openbao://kv/roehub/oidc/provider-a?version=1&extra=1#client_secret",
        "openbao://kv/roehub/oidc/%2e%2e/provider-a#client_secret",
    ],
)
def test_invalid_or_ambiguous_reference_fails_closed(raw: str) -> None:
    with pytest.raises(SecretReferenceError):
        SecretReference.parse(raw)


def test_reference_kind_must_match_consumer() -> None:
    with pytest.raises(SecretReferenceError, match="does not match"):
        SecretReference.parse(
            "openbao://kv/roehub/telegram/org-a#bot_token",
            expected_kind=SecretKind.OIDC,
        )
