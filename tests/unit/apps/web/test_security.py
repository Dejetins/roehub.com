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
    assert (WEB_DIST / "vendor" / "lightweight-charts.standalone.production.js").exists()
    assert (WEB_DIST / "vendor" / "lightweight-charts.NOTICE.txt").exists()


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


def test_settings_notifications_expose_scoped_telegram_hooks_without_chat_id_copy() -> None:
    settings_template = (WEB_TEMPLATES / "pages" / "settings.html").read_text(
        encoding="utf-8"
    )
    notifications_fragment = (
        WEB_TEMPLATES / "fragments" / "account" / "notifications.html"
    ).read_text(encoding="utf-8")
    settings_asset = (WEB_DIST / "js" / "pages" / "settings.js").read_text(
        encoding="utf-8"
    )

    assert 'data-notification-scoped-endpoint="/api/ui/account/notifications/scoped"' in (
        settings_template
    )
    assert "data-notification-scoped" in notifications_fragment
    assert "data-telegram-binding-status" in notifications_fragment
    assert "data-report-schedule-toggle=\"weekly\"" in notifications_fragment
    assert "data-report-schedule-toggle=\"monthly\"" in notifications_fragment
    assert "chat_id" not in notifications_fragment
    assert "notificationScopedEndpoint" in settings_asset
    assert "data-scoped-mode-option" in settings_asset
    assert "settings.notifications.telegram_bound" in settings_asset
