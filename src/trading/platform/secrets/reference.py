"""Canonical opaque references for OpenBao KV v2 secrets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from urllib.parse import parse_qs, urlsplit

_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
_FIELD_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


class SecretReferenceError(ValueError):
    """Raised when an OpenBao reference is ambiguous or outside its root."""


class SecretKind(str, Enum):
    """Secret categories with separate policy boundaries."""

    EXCHANGE = "exchange"
    OIDC = "oidc"
    PLUGIN = "plugins"
    STORAGE = "storage"
    TELEGRAM = "telegram"


@dataclass(frozen=True, slots=True, repr=False)
class SecretReference:
    """Parsed secret reference; the referenced value is never part of this object."""

    mount: str
    root: tuple[str, ...]
    kind: SecretKind
    resource: tuple[str, ...]
    field: str
    version: int | None = None

    @classmethod
    def parse(
        cls,
        raw: str,
        *,
        expected_root: str = "kv/roehub",
        expected_kind: SecretKind | None = None,
    ) -> "SecretReference":
        if not isinstance(raw, str) or not raw or len(raw) > 512:
            raise SecretReferenceError("secret reference must be a non-empty bounded string")
        if "%" in raw or any(character.isspace() for character in raw):
            raise SecretReferenceError("secret reference must not contain escaping or whitespace")

        parsed = urlsplit(raw)
        if parsed.scheme != "openbao" or not parsed.netloc:
            raise SecretReferenceError("secret reference must use openbao://")
        if parsed.username or parsed.password or parsed.port:
            raise SecretReferenceError("secret reference authority must contain only a mount")
        if not _SEGMENT_RE.fullmatch(parsed.netloc):
            raise SecretReferenceError("secret reference mount is invalid")

        expected_parts = tuple(expected_root.split("/"))
        if len(expected_parts) < 2 or any(
            not _SEGMENT_RE.fullmatch(item) for item in expected_parts
        ):
            raise SecretReferenceError("configured secret root is invalid")
        if parsed.netloc != expected_parts[0]:
            raise SecretReferenceError("secret reference is outside configured OpenBao root")

        path_parts = tuple(item for item in parsed.path.split("/") if item)
        if any(not _SEGMENT_RE.fullmatch(item) for item in path_parts):
            raise SecretReferenceError("secret reference path is invalid")
        root = expected_parts[1:]
        if path_parts[: len(root)] != root:
            raise SecretReferenceError("secret reference is outside configured OpenBao root")
        remaining = path_parts[len(root) :]
        if len(remaining) < 2:
            raise SecretReferenceError("secret reference requires kind and resource identity")
        try:
            kind = SecretKind(remaining[0])
        except ValueError as error:
            raise SecretReferenceError("secret reference kind is unsupported") from error
        if expected_kind is not None and kind is not expected_kind:
            raise SecretReferenceError("secret reference kind does not match the consumer")

        if not parsed.fragment or not _FIELD_RE.fullmatch(parsed.fragment):
            raise SecretReferenceError("secret reference requires a valid field fragment")
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
        if set(query) - {"version"} or any(len(values) != 1 for values in query.values()):
            raise SecretReferenceError("secret reference accepts only one version selector")
        version: int | None = None
        if "version" in query:
            raw_version = query["version"][0]
            if not raw_version.isdigit() or raw_version.startswith("0"):
                raise SecretReferenceError("secret reference version must be a positive integer")
            version = int(raw_version)
            if version > 2_147_483_647:
                raise SecretReferenceError("secret reference version is too large")

        return cls(
            mount=parsed.netloc,
            root=root,
            kind=kind,
            resource=remaining[1:],
            field=parsed.fragment,
            version=version,
        )

    @property
    def kv_v2_path(self) -> str:
        return "/".join((self.mount, "data", *self.root, self.kind.value, *self.resource))

    @property
    def metadata_path(self) -> str:
        return "/".join((self.mount, "metadata", *self.root, self.kind.value, *self.resource))

    def __repr__(self) -> str:
        version = "latest" if self.version is None else str(self.version)
        return f"SecretReference(kind={self.kind.value!r}, version={version!r}, value=<opaque>)"
