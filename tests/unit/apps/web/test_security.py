from __future__ import annotations

from pathlib import Path

from apps.web.main.security import sanitize_next_path

REPO_ROOT = Path(__file__).resolve().parents[4]
WEB_TEMPLATES = REPO_ROOT / "apps" / "web" / "templates"
WEB_DIST = REPO_ROOT / "apps" / "web" / "dist"


def test_sanitize_next_path_allows_relative_route() -> None:
    assert sanitize_next_path(raw_next="/strategies") == "/strategies"


def test_sanitize_next_path_allows_local_query_path() -> None:
    assert sanitize_next_path(raw_next="/login?next=/settings") == "/login?next=/settings"


def test_sanitize_next_path_rejects_absolute_external_target() -> None:
    assert sanitize_next_path(raw_next="https://evil.example/path") == "/"


def test_sanitize_next_path_rejects_protocol_relative_target() -> None:
    assert sanitize_next_path(raw_next="//evil.example/path") == "/"


def test_base_shell_has_no_external_cdn_script() -> None:
    base_template = (WEB_TEMPLATES / "base.html").read_text(encoding="utf-8")

    assert "https://unpkg.com" not in base_template
    assert "https://cdn" not in base_template
    assert "/assets/vendor/htmx.min.js" in base_template
    assert (WEB_DIST / "vendor" / "htmx.min.js").exists()


def test_auth_templates_do_not_embed_inline_scripts() -> None:
    auth_templates = [
        WEB_TEMPLATES / "base.html",
        WEB_TEMPLATES / "pages" / "login.html",
        WEB_TEMPLATES / "pages" / "logout.html",
        WEB_TEMPLATES / "pages" / "register.html",
        WEB_TEMPLATES / "fragments" / "auth" / "login_modal.html",
    ]

    for template_path in auth_templates:
        template = template_path.read_text(encoding="utf-8")
        assert "<script>" not in template
        assert "window.location.assign" not in template
        assert "fetch('/api/auth/logout'" not in template


def test_shell_no_longer_references_public_user_badge_partial() -> None:
    shell_templates = [
        WEB_TEMPLATES / "base.html",
        WEB_TEMPLATES / "pages" / "placeholder.html",
        WEB_TEMPLATES / "fragments" / "shell" / "user_badge.html",
        WEB_TEMPLATES / "components" / "user_badge.html",
    ]

    for template_path in shell_templates:
        assert "/_partial/user_badge" not in template_path.read_text(encoding="utf-8")
