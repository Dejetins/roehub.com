"""
Shared API error handlers for RoehubError contract and deterministic 422 payloads.

Docs:
  - docs/architecture/api/api-errors-and-422-payload-v1.md
  - docs/architecture/strategy/strategy-api-immutable-crud-clone-run-control-v1.md
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse

from trading.platform.errors import RoehubError

_ROEHUB_STATUS_BY_CODE: Mapping[str, int] = {
    "validation_error": 422,
    "not_found": 404,
    "forbidden": 403,
    "conflict": 409,
    "unauthorized": 401,
    "unexpected_error": 500,
}



def register_api_error_handlers(*, app: FastAPI) -> None:
    """
    Register global API handlers for RoehubError and deterministic FastAPI validation errors.

    Args:
        app: FastAPI application instance.
    Returns:
        None.
    Assumptions:
        Handlers are installed once during application startup.
    Raises:
        ValueError: If `app` dependency is missing.
    Side Effects:
        Mutates FastAPI exception-handler registry.
    """
    if app is None:  # type: ignore[truthy-bool]
        raise ValueError("register_api_error_handlers requires app")

    app.add_exception_handler(RoehubError, roehub_error_handler)
    app.add_exception_handler(RequestValidationError, request_validation_error_handler)



def roehub_error_handler(_request: Request, error: Exception) -> JSONResponse:
    """
    Convert RoehubError into deterministic JSON response payload.

    Args:
        _request: Starlette request object (unused).
        error: Raised RoehubError instance.
    Returns:
        JSONResponse: Response with contract payload `{"error": ...}`.
    Assumptions:
        Error status code is derived from RoehubError.code via stable mapping table.
    Raises:
        None.
    Side Effects:
        None.
    """
    roehub_error = cast(RoehubError, error)
    status_code = _status_code_for_error_code(code=roehub_error.code)
    details = _normalize_error_details(code=roehub_error.code, details=roehub_error.details)
    payload = {
        "error": {
            "code": roehub_error.code,
            "message": roehub_error.message,
            "details": details,
        }
    }
    return JSONResponse(status_code=status_code, content=payload)



def request_validation_error_handler(_request: Request, error: Exception) -> JSONResponse:
    """
    Convert FastAPI RequestValidationError to canonical Roehub `validation_error` payload.

    Args:
        _request: Starlette request object (unused).
        error: Raised validation exception from FastAPI/Pydantic.
    Returns:
        JSONResponse: HTTP 422 payload with deterministically sorted `details.errors` list.
    Assumptions:
        Validation errors include `loc`, `type`, and `msg` attributes.
    Raises:
        None.
    Side Effects:
        None.
    """
    validation_error = cast(RequestValidationError, error)
    raw_errors = validation_error.errors()
    normalized_errors = _sorted_validation_errors(raw_errors=raw_errors)
    roehub_error = RoehubError(
        code="validation_error",
        message="Validation failed",
        details={
            "errors": normalized_errors,
        },
    )
    return roehub_error_handler(_request, roehub_error)



def _status_code_for_error_code(*, code: str) -> int:
    """
    Resolve HTTP status code for canonical Roehub error code.

    Args:
        code: Machine-readable RoehubError code.
    Returns:
        int: HTTP status code.
    Assumptions:
        Unknown codes are treated as unexpected internal errors.
    Raises:
        None.
    Side Effects:
        None.
    """
    return _ROEHUB_STATUS_BY_CODE.get(code, 500)



def _normalize_error_details(*, code: str, details: Mapping[str, Any] | None) -> dict[str, Any]:
    """
    Normalize RoehubError details payload and enforce deterministic validation error ordering.

    Args:
        code: RoehubError code.
        details: Optional details mapping attached to RoehubError.
    Returns:
        dict[str, Any]: Deterministic plain details mapping.
    Assumptions:
        Validation details may contain `errors` list with path/code/message entries.
    Raises:
        None.
    Side Effects:
        None.
    """
    if details is None:
        return {}

    normalized: dict[str, Any] = {}
    for key in sorted(details):
        value = details[key]
        if code == "validation_error" and key == "errors":
            normalized_errors = _sorted_validation_errors(raw_errors=value)
            normalized[key] = normalized_errors
            continue
        normalized[key] = _normalize_json_compatible_value(value=value)
    return normalized



def _sorted_validation_errors(*, raw_errors: Any) -> list[dict[str, str]]:
    """
    Convert raw validation errors into deterministic list sorted by path, code, and message.

    Args:
        raw_errors: Raw iterable from FastAPI validation subsystem.
    Returns:
        list[dict[str, str]]: Sorted normalized validation items.
    Assumptions:
        Unknown raw shapes are stringified for deterministic payload stability.
    Raises:
        None.
    Side Effects:
        None.
    """
    if not isinstance(raw_errors, Sequence) or isinstance(raw_errors, (str, bytes, bytearray)):
        return []

    normalized_items: list[dict[str, str]] = []
    for raw_error in raw_errors:
        if not isinstance(raw_error, Mapping):
            normalized_items.append(
                {
                    "path": "unknown",
                    "code": "validation_error",
                    "message": str(raw_error),
                }
            )
            continue

        if "path" in raw_error and "code" in raw_error and "message" in raw_error:
            normalized_items.append(
                {
                    "path": str(raw_error["path"]),
                    "code": str(raw_error["code"]),
                    "message": str(raw_error["message"]),
                }
            )
            continue

        path = _normalize_error_path(loc=raw_error.get("loc"))
        code = _normalize_error_code(raw_type=raw_error.get("type"))
        message = str(raw_error.get("msg", "Validation error"))
        normalized_items.append(
            {
                "path": path,
                "code": code,
                "message": message,
            }
        )

    return sorted(
        normalized_items,
        key=lambda item: (item["path"], item["code"], item["message"]),
    )



def _normalize_error_path(*, loc: Any) -> str:
    """
    Convert FastAPI/Pydantic `loc` tuple into dot-delimited deterministic path string.

    Args:
        loc: Raw location object from validation error.
    Returns:
        str: Dot-delimited path, for example `body.overrides.timeframe`.
    Assumptions:
        Location may be tuple/list of path segments and integer indices.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(loc, Sequence) and not isinstance(loc, (str, bytes, bytearray)):
        path_parts = [str(part) for part in loc]
        if path_parts:
            return ".".join(path_parts)
    if loc is None:
        return "unknown"
    return str(loc)



def _normalize_error_code(*, raw_type: Any) -> str:
    """
    Normalize raw validation error type into stable machine-readable code.

    Args:
        raw_type: Raw `type` value from validation error mapping.
    Returns:
        str: Stable error code.
    Assumptions:
        Missing required fields are represented with Pydantic `missing` type.
    Raises:
        None.
    Side Effects:
        None.
    """
    if raw_type is None:
        return "validation_error"

    normalized = str(raw_type).strip().lower()
    if not normalized:
        return "validation_error"

    if normalized == "missing" or normalized.endswith(".missing"):
        return "required"

    return normalized



def _normalize_json_compatible_value(*, value: Any) -> Any:
    """
    Normalize details values into deterministic JSON-compatible structures.

    Args:
        value: Raw details value.
    Returns:
        Any: JSON-compatible scalar/list/mapping representation.
    Assumptions:
        Unknown objects are stringified to preserve deterministic payload rendering.
    Raises:
        None.
    Side Effects:
        None.
    """
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, Any] = {}
        sorted_items = sorted(value.items(), key=lambda item: str(item[0]))
        for raw_key, raw_value in sorted_items:
            normalized_mapping[str(raw_key)] = _normalize_json_compatible_value(value=raw_value)
        return normalized_mapping

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_compatible_value(value=item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)
