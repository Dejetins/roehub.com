"""SSR i18n helpers for Roehub Web templates."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Request

SUPPORTED_LOCALES: tuple[str, ...] = ("en", "ru")
DEFAULT_LOCALE = "en"
LOCALE_COOKIE_NAME = "roehub_locale"

_LOCALES_PATH = Path(__file__).resolve().parents[1] / "locales"


def normalize_locale(raw_locale: str | None) -> str:
    """Return a supported locale code or the default fallback."""
    candidate = "" if raw_locale is None else raw_locale.strip().lower()
    if candidate in SUPPORTED_LOCALES:
        return candidate
    return DEFAULT_LOCALE


def resolve_locale(*, request: Request) -> str:
    """Resolve SSR locale from explicit cookie, then the English default."""
    raw_cookie_locale = request.cookies.get(LOCALE_COOKIE_NAME)
    if raw_cookie_locale is not None:
        return normalize_locale(raw_cookie_locale)
    return DEFAULT_LOCALE


def translate(*, locale: str, key: str) -> str:
    """Translate a catalog key with English fallback and key fallback."""
    normalized_locale = normalize_locale(locale)
    catalog = load_catalog(locale=normalized_locale)
    value = catalog.get(key)
    if isinstance(value, str):
        return value
    fallback_value = load_catalog(locale=DEFAULT_LOCALE).get(key)
    if isinstance(fallback_value, str):
        return fallback_value
    return key


@lru_cache(maxsize=len(SUPPORTED_LOCALES))
def load_catalog(*, locale: str) -> dict[str, Any]:
    """Load one locale catalog from disk."""
    normalized_locale = normalize_locale(locale)
    catalog_path = _LOCALES_PATH / f"{normalized_locale}.json"
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def catalog_key_sets() -> dict[str, set[str]]:
    """Return locale catalog key sets for parity tests."""
    return {
        locale: set(load_catalog(locale=locale).keys())
        for locale in SUPPORTED_LOCALES
    }
