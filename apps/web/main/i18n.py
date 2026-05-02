"""Small SSR i18n helper for the Roehub Web shell."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path

DEFAULT_LOCALE = "en"
SUPPORTED_LOCALES = ("en", "ru")
LOCALE_COOKIE_NAME = "roehub_locale"

_LOCALE_ROOT = Path(__file__).resolve().parents[1] / "locales"


def normalize_locale(raw_locale: str | None) -> str | None:
    """Return a supported short locale code or None for unsupported input."""
    if raw_locale is None:
        return None

    normalized = raw_locale.strip().lower().replace("_", "-")
    if not normalized:
        return None

    language = normalized.split("-", maxsplit=1)[0]
    if language in SUPPORTED_LOCALES:
        return language
    return None


def resolve_locale(
    *,
    query_locale: str | None,
    cookie_locale: str | None,
    accept_language: str | None,
) -> str:
    """Resolve SSR locale from query, cookie, Accept-Language, then default."""
    for candidate in (
        normalize_locale(query_locale),
        normalize_locale(cookie_locale),
        _resolve_accept_language(accept_language=accept_language),
    ):
        if candidate is not None:
            return candidate
    return DEFAULT_LOCALE


def build_translator(*, locale: str) -> Callable[[str], str]:
    resolved_locale = normalize_locale(locale) or DEFAULT_LOCALE

    def translate_key(key: str) -> str:
        return translate(locale=resolved_locale, key=key)

    return translate_key


def translate(*, locale: str, key: str) -> str:
    catalogs = load_catalogs()
    resolved_locale = normalize_locale(locale) or DEFAULT_LOCALE
    return catalogs[resolved_locale].get(key, catalogs[DEFAULT_LOCALE].get(key, key))


def assert_catalog_keys_match() -> None:
    """Load catalogs and raise if supported locale key sets diverge."""
    load_catalogs()


@lru_cache(maxsize=1)
def load_catalogs() -> Mapping[str, Mapping[str, str]]:
    catalogs: dict[str, dict[str, str]] = {}
    for locale in SUPPORTED_LOCALES:
        catalog_path = _LOCALE_ROOT / f"{locale}.json"
        raw_catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(raw_catalog, dict):
            raise ValueError(f"Locale catalog {catalog_path} must be a JSON object")
        catalogs[locale] = {str(key): str(value) for key, value in raw_catalog.items()}

    reference_keys = set(catalogs[DEFAULT_LOCALE])
    for locale, catalog in catalogs.items():
        locale_keys = set(catalog)
        if locale_keys != reference_keys:
            missing = sorted(reference_keys - locale_keys)
            extra = sorted(locale_keys - reference_keys)
            raise ValueError(
                f"Locale catalog {locale} keys mismatch: missing={missing}, extra={extra}"
            )
    return catalogs


def _resolve_accept_language(*, accept_language: str | None) -> str | None:
    if accept_language is None:
        return None

    for language_range in accept_language.split(","):
        locale_part = language_range.split(";", maxsplit=1)[0]
        normalized = normalize_locale(locale_part)
        if normalized is not None:
            return normalized
    return None
