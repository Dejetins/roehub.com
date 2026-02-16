from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class RoehubError(Exception):
    """
    RoehubError — canonical platform-level error contract for API/domain boundaries.

    Docs:
      - docs/architecture/api/api-errors-and-422-payload-v1.md
      - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
    Related:
      - apps/api/common/errors.py
      - src/trading/contexts/strategy/application/use_cases
      - apps/api/routes/strategies.py
    """

    code: str
    message: str
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """
        Validate canonical error fields and freeze details into deterministic plain payloads.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `code` is a stable machine-readable token used for HTTP mapping.
        Raises:
            ValueError: If `code` or `message` are blank.
            TypeError: If `details` is not mapping-compatible when provided.
        Side Effects:
            Mutates internal frozen dataclass slot `details` with normalized payload copy.
        """
        normalized_code = self.code.strip()
        normalized_message = self.message.strip()
        if not normalized_code:
            raise ValueError("RoehubError.code must be non-empty")
        if not normalized_message:
            raise ValueError("RoehubError.message must be non-empty")

        object.__setattr__(self, "code", normalized_code)
        object.__setattr__(self, "message", normalized_message)

        if self.details is None:
            return
        if not isinstance(self.details, Mapping):
            raise TypeError("RoehubError.details must be a mapping when provided")
        normalized_details = _normalize_payload_value(value=dict(self.details))
        if not isinstance(normalized_details, Mapping):
            raise TypeError("RoehubError.details normalization must produce mapping")
        object.__setattr__(self, "details", normalized_details)

    def to_payload(self) -> dict[str, Any]:
        """
        Build deterministic API payload representation.

        Args:
            None.
        Returns:
            dict[str, Any]: `{"error": {"code", "message", "details"}}` payload.
        Assumptions:
            `details` payload is already normalized during object initialization.
        Raises:
            None.
        Side Effects:
            None.
        """
        details_payload: Mapping[str, Any] = self.details if self.details is not None else {}
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": dict(details_payload),
            }
        }



def _normalize_payload_value(*, value: Any) -> Any:
    """
    Normalize nested payload values into deterministic plain-Python structures.

    Args:
        value: Any JSON-compatible value.
    Returns:
        Any: Normalized scalar/list/dict representation.
    Assumptions:
        Non-JSON values are stringified for safe deterministic error payloads.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        sorted_items = sorted(value.items(), key=lambda item: str(item[0]))
        for raw_key, raw_value in sorted_items:
            normalized_mapping[str(raw_key)] = _normalize_payload_value(value=raw_value)
        return normalized_mapping

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_payload_value(value=item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)
