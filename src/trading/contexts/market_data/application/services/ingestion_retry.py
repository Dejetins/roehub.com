"""Retry policy independent of provider/storage drivers; wiring classifies driver errors."""

from collections.abc import Callable

from .source_retry import TemporarySourceError


class TemporaryIngestionError(RuntimeError):
    def __init__(self, code: str, retry_after_s: float = 0) -> None:
        self.code = code
        self.retry_after_s = retry_after_s
        super().__init__(code)


def transient_ingestion_error(error: Exception) -> str | None:
    if isinstance(error, (TemporarySourceError, TemporaryIngestionError)):
        return error.code
    if isinstance(error, (ConnectionError, TimeoutError)):
        return "storage_unavailable"
    return None


def retry_delay(error: Exception, attempt: int, *, base: float = 15, maximum: float = 300) -> float:
    return max(
        min(maximum, base * 2 ** min(attempt - 1, 16)), float(getattr(error, "retry_after_s", 0))
    )


ErrorClassifier = Callable[[Exception], str | None]
