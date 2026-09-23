"""Explicit per-call ownership; never reused by another job or warmup."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass(slots=True)
class BacktestJobScratch:
    job_id: UUID
    _references: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    closed: bool = field(default=False, init=False)

    def retain(self, name: str, value: Any) -> None:
        if self.closed:
            raise RuntimeError("job scratch is closed")
        if name in self._references:
            raise ValueError("scratch identity already retained")
        self._references[name] = value

    def get(self, name: str) -> Any:
        if self.closed:
            raise RuntimeError("job scratch is closed")
        return self._references.get(name)

    @property
    def retained_count(self) -> int:
        return len(self._references)

    def clear(self) -> None:
        self._references.clear()
        self.closed = True
