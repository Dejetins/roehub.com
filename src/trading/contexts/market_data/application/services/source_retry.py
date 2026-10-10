"""Sanitized transient source failure shared by REST adapters and durable jobs."""


class TemporarySourceError(RuntimeError):
    def __init__(self, *, rate_limited: bool = False, retry_after_s: float = 0) -> None:
        self.code = "source_rate_limited" if rate_limited else "source_unavailable"
        self.retry_after_s = max(0, retry_after_s)
        super().__init__(self.code)


class IncompleteHistoryError(TemporarySourceError):
    """A completed fill could not confirm the requested range in durable storage."""

    def __init__(self) -> None:
        super().__init__(retry_after_s=30)
        self.code = "source_history_incomplete"
        self.args = (self.code,)
