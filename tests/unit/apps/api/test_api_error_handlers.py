from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from apps.api.common import register_api_error_handlers
from trading.platform.errors import RoehubError


class _ValidationPayload(BaseModel):
    """
    Validation payload model with deliberately non-lexicographic field order for sorting test.
    """

    model_config = ConfigDict(extra="forbid")

    b: int
    a: int



def test_roehub_error_handler_maps_error_to_http_status_and_payload() -> None:
    """
    Verify RoehubError is converted into deterministic API payload and status mapping.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Conflict code must be mapped to HTTP 409 by shared API error handler.
    Raises:
        AssertionError: If payload shape or HTTP status mapping is broken.
    Side Effects:
        None.
    """
    app = FastAPI()
    register_api_error_handlers(app=app)

    @app.get("/boom")
    def boom() -> None:
        """
        Raise deterministic RoehubError for handler contract test.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Shared handler captures RoehubError and converts to JSONResponse.
        Raises:
            RoehubError: Always for test contract.
        Side Effects:
            None.
        """
        raise RoehubError(code="conflict", message="Conflict happened", details={"run_id": "abc"})

    client = TestClient(app)
    response = client.get("/boom")

    assert response.status_code == 409
    assert response.json() == {
        "error": {
            "code": "conflict",
            "message": "Conflict happened",
            "details": {"run_id": "abc"},
        }
    }



def test_request_validation_error_handler_returns_sorted_validation_errors() -> None:
    """
    Verify validation handler returns deterministic `validation_error` payload with sorted errors.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Errors are sorted lexicographically by path, then code, then message.
    Raises:
        AssertionError: If response payload differs from deterministic contract.
    Side Effects:
        None.
    """
    app = FastAPI()
    register_api_error_handlers(app=app)

    @app.post("/validate")
    def validate(payload: _ValidationPayload) -> dict[str, int]:
        """
        Echo validated payload for deterministic validation error handler test.

        Args:
            payload: Strict validation payload.
        Returns:
            dict[str, int]: Echo payload mapping.
        Assumptions:
            Endpoint is used only for negative validation-path contract checks.
        Raises:
            None.
        Side Effects:
            None.
        """
        return {"a": payload.a, "b": payload.b}

    client = TestClient(app)
    response = client.post("/validate", json={"z": 1})

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Validation failed",
            "details": {
                "errors": [
                    {
                        "path": "body.a",
                        "code": "required",
                        "message": "Field required",
                    },
                    {
                        "path": "body.b",
                        "code": "required",
                        "message": "Field required",
                    },
                    {
                        "path": "body.z",
                        "code": "extra_forbidden",
                        "message": "Extra inputs are not permitted",
                    },
                ]
            },
        }
    }
