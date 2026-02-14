from __future__ import annotations

from dataclasses import dataclass

_ALLOWED_LEVELS = ("base", "free", "pro", "ultra")


@dataclass(frozen=True, slots=True)
class PaidLevel:
    """
    PaidLevel — тарифный уровень пользователя (`free|base|pro|ultra`).

    Docs:
      - docs/architecture/identity/identity-telegram-login-user-model-v1.md
      - docs/architecture/shared-kernel-primitives.md
    Related:
      - src/trading/shared_kernel/primitives/user_id.py
      - src/trading/contexts/identity/domain/entities/user.py
      - src/trading/contexts/identity/adapters/outbound/security/jwt/hs256_jwt_codec.py
    """

    value: str

    def __post_init__(self) -> None:
        """
        Normalize and validate paid-level value.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Level name is case-insensitive on input and stored in lowercase.
        Raises:
            ValueError: If level is blank or not from allowed enum.
        Side Effects:
            Mutates stored value to normalized lowercase representation.
        """
        normalized = self.value.strip().lower()
        object.__setattr__(self, "value", normalized)

        if normalized not in _ALLOWED_LEVELS:
            raise ValueError(
                f"PaidLevel must be one of {sorted(_ALLOWED_LEVELS)}, got {self.value!r}"
            )

    @classmethod
    def free(cls) -> PaidLevel:
        """
        Build default free paid-level value.

        Args:
            None.
        Returns:
            PaidLevel: Default free level.
        Assumptions:
            `free` is always present in allowed levels list.
        Raises:
            ValueError: If enum constants become inconsistent.
        Side Effects:
            None.
        """
        return cls("free")

    def __str__(self) -> str:
        """
        Return normalized paid-level value.

        Args:
            None.
        Returns:
            str: Level code in lowercase.
        Assumptions:
            Value already passed constructor validation.
        Raises:
            None.
        Side Effects:
            None.
        """
        return self.value
