from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import requests


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
        attempt = 0
        last_exc: Exception | None = None

        while attempt <= retries:
            try:
                r = requests.get(url, params=dict(params), timeout=timeout_s)
                headers = {str(k): str(v) for k, v in r.headers.items()}

                # 429/5xx — retry
                if r.status_code in (429, 418) or (500 <= r.status_code <= 599):
                    _sleep_backoff(
                        attempt=attempt,
                        base_s=backoff_base_s,
                        max_s=backoff_max_s,
                        jitter_s=backoff_jitter_s,
                    )
                    attempt += 1
                    continue

                # Остальные не-200 считаем ошибкой
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code} for {url} params={params} body={r.text[:500]}")  # noqa: E501

                try:
                    body = r.json()
                except Exception as e:  # noqa: BLE001
                    raise RuntimeError(f"Invalid JSON from {url} params={params}: {r.text[:500]}") from e  # noqa: E501

                return HttpResponse(status_code=200, headers=headers, body=body)

            except Exception as e:  # noqa: BLE001
                last_exc = e
                _sleep_backoff(
                    attempt=attempt,
                    base_s=backoff_base_s,
                    max_s=backoff_max_s,
                    jitter_s=backoff_jitter_s,
                )
                attempt += 1

        raise RuntimeError(f"HTTP request failed after retries url={url} params={params}") from last_exc  # noqa: E501


def _sleep_backoff(*, attempt: int, base_s: float, max_s: float, jitter_s: float) -> None:
    exp = min(max_s, base_s * (2**attempt))
    jitter = random.random() * jitter_s
    time.sleep(exp + jitter)
