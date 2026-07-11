import json
from pathlib import Path


def test_notification_settings_render_owner_scoped_telegram_delivery_counter() -> None:
    template = Path(
        "apps/web/templates/fragments/account/notifications.html"
    ).read_text(encoding="utf-8")
    script = Path("apps/web/dist/js/pages/settings.js").read_text(encoding="utf-8")

    assert "data-notification-sent-count" in template
    assert "data-notification-sent-last" in template
    assert "telegram_sent_total" in script
    assert "last_telegram_sent_at" in script
    assert "recipient_address_ref" not in template


def test_notification_delivery_counter_copy_exists_in_both_locales() -> None:
    for locale in ("en", "ru"):
        payload = json.loads(
            Path(f"apps/web/locales/{locale}.json").read_text(encoding="utf-8")
        )
        assert payload["settings.notifications.sent_messages"]
        assert payload["settings.notifications.last_sent"]
        assert payload["settings.notifications.no_sent_messages"]
