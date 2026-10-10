"""Shared REST admission. A sent reservation is never refunded; release requires stopped I/O."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol


@dataclass(frozen=True)
class QuotaWindow:
    dimension: str
    seconds: int
    limit: int
    cost: int

    def __post_init__(self):
        if not self.dimension or not 0 < self.cost <= self.limit or self.seconds < 1:
            raise ValueError("invalid_provider_quota")


@dataclass(frozen=True)
class ProviderRequest:
    provider: str
    scope: str
    concurrency: int
    windows: tuple[QuotaWindow, ...]

    def __post_init__(self):
        if not self.provider or not 1 <= len(self.scope) <= 128 or self.concurrency < 1:
            raise ValueError("invalid_provider_scope")
        if not self.windows or len({(w.dimension, w.seconds) for w in self.windows}) != len(
            self.windows
        ):
            raise ValueError("duplicate_or_empty_provider_windows")


class ProviderPermit(Protocol):
    def check(self) -> None: ...
    def complete(self, *, headers: Mapping[str, str], cooldown_s: float = 0) -> None:
        """Called only after response consumption or confirmed transport termination."""
        ...


class ProviderAdmission(Protocol):
    def reserve(self, request: ProviderRequest) -> ProviderPermit: ...


# Execution ownership is propagated only inside the worker's synchronous scope.
# Adapters use the callback without importing the scheduler or PostgreSQL types.

_io_guard: ContextVar[Callable[[], None]] = ContextVar("market_data_io_guard", default=lambda: None)


@contextmanager
def provider_io_guard(check: Callable[[], None]):
    token = _io_guard.set(check)
    try:
        yield
    finally:
        _io_guard.reset(token)


def check_provider_io_owner() -> None:
    _io_guard.get()()
