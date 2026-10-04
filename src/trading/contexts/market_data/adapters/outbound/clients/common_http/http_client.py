from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Mapping, Protocol

import requests

from trading.contexts.market_data.application.services.source_retry import TemporarySourceError


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status_code: int
    headers: Mapping[str, str]
    body: Any


class HttpClient(Protocol):
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
        ...


class RequestsHttpClient(HttpClient):
    """
    Минимальный HTTP клиент для REST ingestion.

    - requests.get(...)
    - retries + экспоненциальный backoff с jitter
    - возвращает JSON body как python-объект
    """

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
        for attempt in range(retries + 1):
            try:
                r = requests.get(url, params=dict(params), timeout=timeout_s)
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError) as error:
                failure = TemporarySourceError()
                if attempt == retries:
                    raise failure from error
            else:
                headers = {str(k): str(v) for k, v in r.headers.items()}
                lower = {k.lower(): v for k, v in headers.items()}
                after = retry_after_seconds(lower.get("retry-after"))
                limited = r.status_code in (429, 418)
                if limited or r.status_code == 408 or 500 <= r.status_code <= 599:
                    failure = TemporarySourceError(rate_limited=limited, retry_after_s=after)
                elif r.status_code != 200:
                    # Permanent request/auth/region errors must not enter an endless retry loop.
                    raise RuntimeError(f"source_http_{r.status_code}")
                else:
                    try:
                        body = r.json()
                    except ValueError as error:
                        if attempt == retries:
                            raise RuntimeError("source_invalid_json") from error
                        _sleep_backoff(attempt=attempt, base_s=backoff_base_s,
                                       max_s=backoff_max_s, jitter_s=backoff_jitter_s)
                        continue
                    code = str(body.get("retCode")) if isinstance(body, dict) else "0"
                    if code not in {"10000", "10006", "10016", "429"}:
                        return HttpResponse(status_code=200, headers=headers, body=body)
                    reset = lower.get("x-bapi-limit-reset-timestamp")
                    try:
                        reset_after = float(reset or 0) / 1000 - time.time()
                    except ValueError:
                        reset_after = 0
                    failure = TemporarySourceError(
                        rate_limited=code in {"10006", "429"},
                        retry_after_s=max(after, min(604800, reset_after)),
                    )
                # A source-directed cooldown is persisted by the job runner; don't
                # block the worker/hold its control checkpoint while sleeping for it.
                if attempt == retries or failure.retry_after_s > 0:
                    raise failure
            _sleep_backoff(
                attempt=attempt, base_s=backoff_base_s,
                max_s=backoff_max_s, jitter_s=backoff_jitter_s,
            )
        raise TemporarySourceError()


def retry_after_seconds(value: str | None) -> float:
    if not value:
        return 0
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return 0
    return max(0, min(604800, seconds)) if math.isfinite(seconds) else 0


def _sleep_backoff(*, attempt: int, base_s: float, max_s: float, jitter_s: float) -> None:
    exp = min(max_s, base_s * (2**attempt))
    jitter = random.random() * jitter_s
    time.sleep(exp + jitter)
