from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class InstallationId:
    """Stable identifier of one self-hosted Roehub installation."""

    value: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.value, UUID):
            raise ValueError(f"InstallationId requires UUID value, got {self.value!r}")

    @classmethod
    def from_string(cls, raw_value: str) -> InstallationId:
        stripped = raw_value.strip()
        if not stripped:
            raise ValueError("InstallationId.from_string requires non-empty value")
        return cls(UUID(stripped))

    def __str__(self) -> str:
        return str(self.value)
