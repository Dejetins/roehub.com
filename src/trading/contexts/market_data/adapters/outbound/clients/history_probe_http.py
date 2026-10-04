"""A bounded, redacted transport for interactive history-bound discovery."""

from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Mapping

from trading.contexts.market_data.adapters.outbound.clients.common_http.http_client import (
    HttpClient,
    HttpResponse,
)


@dataclass
class HistoryProbeHttpClient:
    delegate: HttpClient
    budget_seconds: float = 25
    started: float = field(default_factory=monotonic)

    def get_json(
        self,
        *,
        url: str,
        params: Mapping[str, Any],
        timeout_s: float,
        retries: int,
        backoff_base_s: float,
        backoff_max_s: float,
        backoff_jitter_s: float,
    ) -> HttpResponse:
        # Metadata GETs are replay-safe; candle commands are never replayed here.
        # Recompute the remaining budget before the single transient-read retry.
        for attempt in range(2):
            remaining = self.budget_seconds - (monotonic() - self.started)
            if remaining <= 0:
                raise RuntimeError("History metadata lookup timed out")
            try:
                return self.delegate.get_json(
                    url=url,
                    params=params,
                    timeout_s=min(timeout_s, 5, remaining),
                    retries=0,
                    backoff_base_s=0,
                    backoff_max_s=0,
                    backoff_jitter_s=0,
                )
            except Exception:
                if attempt == 1:
                    raise RuntimeError("History metadata is unavailable") from None
        raise RuntimeError("History metadata is unavailable")
