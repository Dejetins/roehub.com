from __future__ import annotations

from urllib.parse import urlparse

from starlette.requests import Request

_FORWARDED_HOST_KEY = "host"
_FORWARDED_PROTO_KEY = "proto"


def same_origin_rejection_reason(
    *,
    request: Request,
    fail_closed_without_origin: bool,
) -> str | None:
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    if origin is None and referer is None:
        if fail_closed_without_origin:
            return "csrf_required"
        return None

    expected_origins = _expected_origin_sources(request=request)
    for candidate in (origin, referer):
        if candidate is None:
            continue
        parsed = urlparse(candidate)
        if parsed.netloc and not _matches_expected_origin(
            candidate_scheme=parsed.scheme,
            candidate_netloc=parsed.netloc,
            expected_origins=expected_origins,
        ):
            return "csrf_origin_mismatch"
    return None


def _expected_origin_sources(*, request: Request) -> tuple[tuple[str, str | None], ...]:
    sources: list[tuple[str, str | None]] = []
    host = _normalize_header_value(request.headers.get("host"))
    if host:
        sources.append((host, None))

    forwarded_hosts = _split_forwarded_values(request.headers.get("x-forwarded-host"))
    forwarded_protos = _split_forwarded_values(request.headers.get("x-forwarded-proto"))
    for index, forwarded_host in enumerate(forwarded_hosts):
        forwarded_proto = _select_forwarded_proto(
            forwarded_protos=forwarded_protos,
            index=index,
        )
        sources.append((forwarded_host, forwarded_proto))
        sources.append((forwarded_host, None))

    sources.extend(_forwarded_header_sources(value=request.headers.get("forwarded")))
    return tuple(dict.fromkeys(sources))


def _matches_expected_origin(
    *,
    candidate_scheme: str,
    candidate_netloc: str,
    expected_origins: tuple[tuple[str, str | None], ...],
) -> bool:
    normalized_candidate_netloc = _normalize_header_value(candidate_netloc)
    normalized_candidate_scheme = _normalize_header_value(candidate_scheme)
    if not normalized_candidate_netloc:
        return True
    for expected_netloc, expected_scheme in expected_origins:
        if normalized_candidate_netloc != expected_netloc:
            continue
        if expected_scheme is None or not normalized_candidate_scheme:
            return True
        if normalized_candidate_scheme == expected_scheme:
            return True
    return False


def _split_forwarded_values(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(
        normalized
        for part in value.split(",")
        if (normalized := _normalize_header_value(part))
    )


def _select_forwarded_proto(
    *,
    forwarded_protos: tuple[str, ...],
    index: int,
) -> str | None:
    if index < len(forwarded_protos):
        return forwarded_protos[index]
    if len(forwarded_protos) == 1:
        return forwarded_protos[0]
    return None


def _forwarded_header_sources(*, value: str | None) -> tuple[tuple[str, str | None], ...]:
    if value is None:
        return ()
    sources: list[tuple[str, str | None]] = []
    for entry in value.split(","):
        params: dict[str, str] = {}
        for raw_param in entry.split(";"):
            key, separator, raw_value = raw_param.partition("=")
            if not separator:
                continue
            normalized_key = _normalize_header_value(key)
            if normalized_key not in {_FORWARDED_HOST_KEY, _FORWARDED_PROTO_KEY}:
                continue
            params[normalized_key] = _normalize_header_value(raw_value.strip('"'))
        forwarded_host = params.get(_FORWARDED_HOST_KEY)
        if forwarded_host:
            sources.append((forwarded_host, params.get(_FORWARDED_PROTO_KEY)))
            sources.append((forwarded_host, None))
    return tuple(sources)


def _normalize_header_value(value: str | None) -> str:
    if value is None:
        return ""
    return value.strip().lower()
