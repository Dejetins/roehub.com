from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class UserId:
    """
    UserId — сквозной идентификатор пользователя в формате UUID.

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
      - docs/architecture/shared-kernel-primitives.md
    Related:
      - src/trading/shared_kernel/primitives/paid_level.py
      - src/trading/contexts/identity/domain/entities/user.py
      - src/trading/contexts/identity/application/ports/current_user.py
    """

    value: UUID

    def __post_init__(self) -> None:
        """
        Validate UUID value type for user identifier.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `UserId` must wrap concrete `uuid.UUID` value.
        Raises:
            ValueError: If `value` is not a UUID instance.
        Side Effects:
            None.
        """
        if not isinstance(self.value, UUID):
            raise ValueError(f"UserId requires UUID value, got {self.value!r}")

    @classmethod
    def from_string(cls, raw_value: str) -> UserId:
        """
        Parse user identifier from canonical UUID string representation.

        Args:
            raw_value: Raw UUID string.
        Returns:
            UserId: Parsed user id value object.
        Assumptions:
            Input string is expected to be non-empty and UUID-compatible.
        Raises:
            ValueError: If UUID parsing fails.
        Side Effects:
            None.
        """
        stripped = raw_value.strip()
        if not stripped:
            raise ValueError("UserId.from_string requires non-empty value")
        return cls(UUID(stripped))

    def __str__(self) -> str:
        """
        Return canonical string representation of wrapped UUID.

        Args:
            None.
        Returns:
            str: UUID string.
        Assumptions:
            Wrapped UUID is valid.
        Raises:
            None.
        Side Effects:
            None.
        """
        return str(self.value)
