from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
WEB_ROOT = REPO_ROOT / "apps" / "web"


def test_backtests_template_keeps_ai_chat_shell_without_retired_job_routes() -> None:
    template = (WEB_ROOT / "templates/pages/backtests.html").read_text(encoding="utf-8")
    retired_endpoint = "/backtests" + "/ai-config" + "/jobs"

    assert f"/api{retired_endpoint}" not in template
    assert "data-ai-log" in template
    assert "data-ai-prompt" in template
    assert "data-ai-submit" in template
    assert "data-ai-timeline" in template
    assert "data-ai-mode" not in template


def test_backtests_ai_js_does_not_call_retired_one_shot_jobs() -> None:
    source = (WEB_ROOT / "dist/js/pages/backtests.js").read_text(encoding="utf-8")
    retired_endpoint = "/backtests" + "/ai-config" + "/jobs"

    assert f"/api{retired_endpoint}" not in source
    assert "EventSource" not in source
    assert "activeAiStatusRequest" not in source
    assert "currentAiPayload" not in source
    assert "current_config: buildRequestPayload(root)" not in source
    assert "recordAiFeedback" not in source
    assert "data-ai-mode" not in source


def test_backtests_ai_load_configuration_does_not_auto_create_backtest_job() -> None:
    source = (WEB_ROOT / "dist/js/pages/backtests.js").read_text(encoding="utf-8")

    assert "function applyAiConfiguration" not in source
    assert "recordAiFeedback" not in source


def test_backtests_ai_locale_notice_and_load_action_copy_are_bilingual() -> None:
    en = json.loads((WEB_ROOT / "locales/en.json").read_text(encoding="utf-8"))
    ru = json.loads((WEB_ROOT / "locales/ru.json").read_text(encoding="utf-8"))

    assert "Prompts and AI outputs may be saved" in en["backtests.ai.notice"]
    assert "Запросы и ответы AI могут сохраняться" in ru["backtests.ai.notice"]
    assert "exchange keys" in en["backtests.ai.notice"]
    assert "exchange keys" in ru["backtests.ai.notice"]
    mode_prefix = "backtests.ai." + "mode"
    assert not any(key.startswith(mode_prefix) for key in en)
    assert not any(key.startswith(mode_prefix) for key in ru)


def test_backtests_cancel_confirmation_is_site_modal_and_bilingual() -> None:
    template = (WEB_ROOT / "templates/pages/backtests.html").read_text(encoding="utf-8")
    source = (WEB_ROOT / "dist/js/pages/backtests.js").read_text(encoding="utf-8")
    css = (WEB_ROOT / "dist/css/pages/backtests.css").read_text(encoding="utf-8")
    en = json.loads((WEB_ROOT / "locales/en.json").read_text(encoding="utf-8"))
    ru = json.loads((WEB_ROOT / "locales/ru.json").read_text(encoding="utf-8"))

    assert "data-job-cancel-dialog" in template
    assert "data-job-cancel-confirm" in template
    assert "openCancelDialog(root" in source
    assert "window.confirm" not in _function_body(source, "confirmCancelDialog")
    assert "backtests-cancel-dialog__panel" in css
    assert en["backtests.cancel_confirm.confirm"] == "Cancel job"
    assert ru["backtests.cancel_confirm.confirm"] == "Отменить job"


def _function_body(source: str, name: str) -> str:
    marker = f"function {name}("
    start = source.index(marker)
    brace_start = source.index("{", start)
    depth = 0
    for index in range(brace_start, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace_start:index]
    raise AssertionError(f"function not closed: {name}")
