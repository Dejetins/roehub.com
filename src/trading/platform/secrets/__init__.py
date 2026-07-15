"""Typed secret references and fail-closed OpenBao clients."""

from .openbao import (
    OpenBaoPermissionError,
    OpenBaoReadiness,
    OpenBaoSecretNotFoundError,
    OpenBaoSecretResolver,
    OpenBaoUnavailableError,
    SecretResolutionError,
    SecretValue,
    SecureCredentialFile,
    SecureTokenFile,
)
from .reference import SecretKind, SecretReference, SecretReferenceError

__all__ = [
    "OpenBaoPermissionError",
    "OpenBaoReadiness",
    "OpenBaoSecretNotFoundError",
    "OpenBaoSecretResolver",
    "OpenBaoUnavailableError",
    "SecretKind",
    "SecretReference",
    "SecretReferenceError",
    "SecretResolutionError",
    "SecretValue",
    "SecureCredentialFile",
    "SecureTokenFile",
]
