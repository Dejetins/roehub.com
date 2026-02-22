from __future__ import annotations

from typing import Any, Callable, Protocol, TypeVar, overload

_F = TypeVar("_F", bound=Callable[..., Any])


class _Decorator(Protocol):
    def __call__(self, __func: _F, /) -> _F: ...


class _Config:
    CACHE_DIR: str


config: _Config


@overload
def njit(__func: _F, /, *args: Any, **kwargs: Any) -> _F: ...
@overload
def njit(*args: Any, **kwargs: Any) -> _Decorator: ...


def prange(*args: int) -> range: ...
def get_num_threads() -> int: ...
def set_num_threads(n: int) -> None: ...
