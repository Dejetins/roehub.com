from __future__ import annotations

import re
from pathlib import Path

from apps.web.main.security import sanitize_next_path

_TEMPLATE_ROOT = Path("apps/web/templates")
_CSS_ROOT = Path("apps/web/dist/css")


def test_sanitize_next_path_allows_relative_route() -> None:
    assert sanitize_next_path(raw_next="/strategies") == "/strategies"


def test_sanitize_next_path_rejects_absolute_external_target() -> None:
    assert sanitize_next_path(raw_next="https://evil.example/path") == "/"


def test_sanitize_next_path_rejects_protocol_relative_target() -> None:
    assert sanitize_next_path(raw_next="//evil.example/path") == "/"


def test_sanitize_next_path_rejects_non_path_values() -> None:
    assert sanitize_next_path(raw_next="javascript:alert(1)") == "/"
    assert sanitize_next_path(raw_next="settings") == "/"


def test_base_shell_does_not_reference_external_cdn_scripts() -> None:
    base_template = (_TEMPLATE_ROOT / "base.html").read_text(encoding="utf-8")

    assert "https://unpkg.com" not in base_template
    assert "https://cdn" not in base_template
    assert 'src="/assets/vendor/htmx.min.js"' in base_template


def test_base_shell_avoids_decorative_monogram_and_guest_badge() -> None:
    base_template = (_TEMPLATE_ROOT / "base.html").read_text(encoding="utf-8")
    user_badge = (_TEMPLATE_ROOT / "components/user_badge.html").read_text(encoding="utf-8")

    assert "rh-brand__mark" not in base_template
    assert ">RH<" not in base_template
    assert "GUEST" not in base_template
    assert "GUEST" not in user_badge


def test_global_page_background_does_not_use_orange_grid() -> None:
    base_css = (_CSS_ROOT / "base.css").read_text(encoding="utf-8")

    body_block_match = re.search(r"body\s*\{(?P<body>.*?)\}", base_css, re.DOTALL)
    assert body_block_match is not None
    assert "linear-gradient" not in body_block_match.group("body")
    assert "--rh-grid-line" not in base_css


def test_auth_templates_do_not_embed_inline_scripts() -> None:
    for template_name in ("pages/login.html", "pages/logout.html", "pages/register.html"):
        template = (_TEMPLATE_ROOT / template_name).read_text(encoding="utf-8")
        inline_scripts = [
            tag for tag in re.findall(r"<script\b[^>]*>", template) if " src=" not in tag
        ]

        assert inline_scripts == []


def test_shell_fragment_does_not_depend_on_public_user_badge_partial_route() -> None:
    base_template = (_TEMPLATE_ROOT / "base.html").read_text(encoding="utf-8")
    shell_fragment = (_TEMPLATE_ROOT / "fragments/shell/user_badge.html").read_text(
        encoding="utf-8"
    )

    assert "/_partial/user_badge" not in base_template
    assert "/_partial/user_badge" not in shell_fragment
    assert 'include "components/user_badge.html"' in shell_fragment
