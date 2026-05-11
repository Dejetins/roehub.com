from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
WEB_ROOT = REPO_ROOT / "apps" / "web"


def test_backtests_template_exposes_ai_configurator_controls_and_routes() -> None:
    template = (WEB_ROOT / "templates/pages/backtests.html").read_text(encoding="utf-8")

    assert 'data-ai-config-jobs-endpoint="/api/backtests/ai-config/jobs"' in template
    assert (
        'data-ai-config-events-endpoint-template="/api/backtests/ai-config/jobs/{job_id}/events"'
        in template
    )
    assert (
        'data-ai-config-feedback-endpoint-template="/api/backtests/ai-config/jobs/{job_id}/feedback"'
        in template
    )
    assert "data-ai-log" in template
    assert "data-ai-prompt" in template
    assert "data-ai-submit" in template
    assert "data-ai-timeline" in template
    assert 'data-ai-mode="{{ value }}"' in template
    for mode in [
        "create",
        "edit_current",
        "explain_current",
        "repair_invalid",
        "suggest_safer",
    ]:
        assert f'("{mode}",' in template


def test_backtests_ai_js_uses_sse_polling_snapshot_and_safe_text_rendering() -> None:
    source = (WEB_ROOT / "dist/js/pages/backtests.js").read_text(encoding="utf-8")

    assert "/api/backtests/ai-config/jobs" in source
    assert "/api/backtests/ai-config/jobs/{job_id}/events" in source
    assert "/api/backtests/ai-config/jobs/{job_id}/feedback" in source
    assert "EventSource" in source
    assert "createPoller" in source
    assert "activeAiStatusRequest" in source
    assert "current_config: buildRequestPayload(root)" in source
    assert "Idempotency-Key" in source
    for status in [
        "queued",
        "preparing_catalog",
        "generating",
        "validating",
        "repairing",
        "ready",
    ]:
        assert status in source
    assert "message.textContent" in source
    assert "button.textContent = t(\"backtests.ai.load_configuration\")" in source
    assert "assistantMessage.innerHTML" not in source
    assert "raw_model_response" not in source
    assert "chain_of_thought" not in source


def test_backtests_ai_load_configuration_does_not_auto_create_backtest_job() -> None:
    source = (WEB_ROOT / "dist/js/pages/backtests.js").read_text(encoding="utf-8")
    body = _function_body(source, "applyAiConfiguration")

    assert "seedConfigDraft" in body
    assert "recordAiFeedback(root, jobId, true)" in body
    assert "createJob(" not in body
    assert "preflight(" not in body
    assert "/api/backtests/jobs" not in body
    assert "root.dataset.jobsEndpoint" not in body


def test_backtests_ai_locale_notice_and_load_action_copy_are_bilingual() -> None:
    en = json.loads((WEB_ROOT / "locales/en.json").read_text(encoding="utf-8"))
    ru = json.loads((WEB_ROOT / "locales/ru.json").read_text(encoding="utf-8"))

    assert en["backtests.ai.load_configuration"] == "Load configuration"
    assert ru["backtests.ai.load_configuration"] == "Загрузить конфигурацию"
    assert "Prompts and AI outputs may be saved" in en["backtests.ai.notice"]
    assert "Запросы и ответы AI могут сохраняться" in ru["backtests.ai.notice"]
    assert "exchange keys" in en["backtests.ai.notice"]
    assert "exchange keys" in ru["backtests.ai.notice"]


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
