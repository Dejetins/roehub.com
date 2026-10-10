from __future__ import annotations

import atexit
import math
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import Condition
from typing import Any, Mapping, Protocol

from trading.contexts.market_data.application.ports.provider_admission import (
    check_provider_io_owner,
)
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
    """Bounded exclusive session pool, absolute call deadline and shared admission."""

    def __init__(self, *, admission=None, policy=None, transport_capacity=4):
        if not isinstance(transport_capacity, int) or transport_capacity < 1:
            raise ValueError('positive_transport_capacity_required')
        self.admission = admission
        self.policy = policy
        self._capacity = transport_capacity
        self._transports = []
        self._available = []
        self._lock = Condition()
        self._closed = False
        atexit.register(self.close)

    @classmethod
    def from_config(cls, config, environ):
        from trading.contexts.market_data.adapters.outbound.persistence.postgres.provider_admission import (  # noqa: E501
            PostgresProviderAdmission,
        )

        from .provider_policy import (
            ProviderPolicy,
        )
        return cls(admission=PostgresProviderAdmission(environ["STRATEGY_PG_DSN"]),
                   policy=ProviderPolicy(config),
                   transport_capacity=config.ingestion.rest_concurrency_instruments)

    def close(self):
        with self._lock:
            self._closed = True
            for transport in self._transports:
                transport.stop()
            self._transports.clear()
            self._available.clear()
            self._lock.notify_all()

    def _transport(self, deadline, check):
        from .managed_transport import ManagedHttpTransport
        while True:
            check()
            with self._lock:
                if self._closed:
                    raise RuntimeError("provider_http_closed")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TemporarySourceError()
                if self._available:
                    return self._available.pop()
                if len(self._transports) < self._capacity:
                    transport = ManagedHttpTransport()
                    self._transports.append(transport)
                    return transport
                self._lock.wait(timeout=min(0.1, remaining))

    def _return_transport(self, transport):
        with self._lock:
            if transport.closed:
                if transport in self._transports:
                    self._transports.remove(transport)
            elif not self._closed:
                self._available.append(transport)
            self._lock.notify_all()

    def get_json(
        self, *, url: str, params: Mapping[str, Any], timeout_s: float, retries: int,
        backoff_base_s: float, backoff_max_s: float, backoff_jitter_s: float,
    ) -> HttpResponse:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive_finite_http_deadline_required')
        if self.admission is None or self.policy is None:
            raise RuntimeError('shared_provider_admission_required')
        deadline = time.monotonic() + timeout_s
        for attempt in range(retries + 1):
            if time.monotonic() >= deadline:
                raise TemporarySourceError()
            check_provider_io_owner()
            permit = self.admission.reserve(self.policy.request(url, params))
            headers = {}
            cooldown = 0
            transport = None
            try:
                def check():
                    check_provider_io_owner()
                    permit.check()
                transport = self._transport(deadline, check)
                result = transport.get(url=url, params=params, deadline=deadline, check=check)
                if result[0] == 'temporary':
                    raise TemporarySourceError()
                if result[0] == 'invalid':
                    raise RuntimeError('source_invalid_json')
                _, status, headers, body = result
                lower = {k.lower(): v for k, v in headers.items()}
                after = retry_after_seconds(lower.get('retry-after'))
                cooldown = after
                limited = status in (429, 418)
                code = str(body.get('retCode')) if isinstance(body, dict) else '0'
                if limited or code in {'10006', '429'}:
                    reset = lower.get('x-bapi-limit-reset-timestamp')
                    try:
                        reset_after = max(0, float(reset or 0) / 1000 - time.time())
                    except ValueError:
                        reset_after = 0
                    cooldown = max(after, min(604800, reset_after), 1)
                    raise TemporarySourceError(rate_limited=True, retry_after_s=cooldown)
                if status == 403 and self.policy.request(url, params).provider == 'bybit':
                    cooldown = max(600, after)
                    raise TemporarySourceError(rate_limited=True, retry_after_s=cooldown)
                if status == 408 or 500 <= status <= 599 or code in {'10000', '10016'}:
                    raise TemporarySourceError(retry_after_s=after)
                if status != 200:
                    raise RuntimeError(f'source_http_{status}')
                if body is None:
                    if attempt == retries:
                        raise RuntimeError('source_invalid_json')
                    raise TemporarySourceError()
                return HttpResponse(status_code=status, headers=headers, body=body)
            except TemporarySourceError as error:
                if attempt == retries or error.retry_after_s > 0:
                    raise
            finally:
                # get() consumes the response or kills AND joins its owning process.
                if transport is not None:
                    self._return_transport(transport)
                permit.complete(headers=headers, cooldown_s=cooldown)
            delay = min(backoff_max_s, backoff_base_s * (2**attempt))
            delay += random.random() * backoff_jitter_s
            remaining = deadline - time.monotonic()
            if delay >= remaining:
                raise TemporarySourceError()
            time.sleep(delay)
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
