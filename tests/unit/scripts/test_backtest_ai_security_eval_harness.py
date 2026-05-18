from __future__ import annotations

from pathlib import Path

from scripts.backtest_ai.run_configurator_security_eval import (
    DEFAULT_FIXTURE_PATH,
    SecurityEvalObservation,
    load_cases,
    security_eval_blockers,
    summarize_security_eval,
)


def test_security_eval_fixture_has_required_cases_and_safe_count() -> None:
    cases = load_cases(Path(DEFAULT_FIXTURE_PATH))

    case_ids = {case.case_id for case in cases}
    safe_cases = [case for case in cases if case.kind == "safe"]

    assert len(safe_cases) == 10
    assert {
        "secrets_env_vars",
        "output_script_injection",
        "auto_run_backtest_attempt",
        "system_prompt_extraction",
        "resource_abuse_huge_prompt",
        "unsupported_values_hallucination",
    } <= case_ids
    assert any(case.category == "safe_informational" for case in safe_cases)
    assert any(case.locale == "ru" for case in safe_cases)
    assert any(case.locale == "en" for case in safe_cases)


def test_security_eval_summary_accepts_zero_unsafe_actions() -> None:
    cases = load_cases(Path(DEFAULT_FIXTURE_PATH))
    observations = [
        SecurityEvalObservation(
            case=case,
            status="ready"
            if case.expected_outcome == "safe_load_action_ready"
            else "needs_clarification",
            http_status=201,
            load_action_enabled=case.expected_outcome == "safe_load_action_ready",
            load_action_state="ready"
            if case.expected_outcome == "safe_load_action_ready"
            else "needs_clarification",
            load_action_reason=None,
            assistant_message="Safe assistant response.",
            conversation_id="conversation",
            run_id="run",
            latency_ms=1.0,
        )
        for case in cases
    ]

    summary = summarize_security_eval(observations)

    assert summary["unauthorized_actions"] == 0
    assert summary["secret_path_leakage"] == 0
    assert summary["invalid_load_action"] == 0
    assert summary["safe_prompts_blocked_literal"] == "0/10"
    assert security_eval_blockers(
        fake_worker=False,
        summary=summary,
        observations=observations,
    ) == []
