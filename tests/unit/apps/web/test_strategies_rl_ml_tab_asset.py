from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
STRATEGIES_TEMPLATE = REPO_ROOT / "apps" / "web" / "templates" / "pages" / "strategies.html"
STRATEGIES_JS = REPO_ROOT / "apps" / "web" / "dist" / "js" / "pages" / "strategies.js"
STRATEGIES_CSS = REPO_ROOT / "apps" / "web" / "dist" / "css" / "pages" / "strategies.css"


def test_strategies_template_exposes_rl_ml_tab_hooks() -> None:
    template = STRATEGIES_TEMPLATE.read_text(encoding="utf-8")

    required_literals = [
        'data-strategies-mode="rl_ml"',
        'data-strategies-mode-panel="rl_ml"',
        "data-rl-model-family",
        "data-rl-slot-rows",
        "data-rl-operator-controls",
        "data-rl-outcome-rows",
        "data-rl-risk-synthetic-exits",
        "data-rl-risk-reasons",
        "ml_agent_decision_outcomes",
    ]
    for literal in required_literals:
        assert literal in template


def test_strategies_asset_renders_rl_ml_from_backend_state_only() -> None:
    asset = STRATEGIES_JS.read_text(encoding="utf-8")

    assert "renderRlMlTab(root, summary.rl_ml)" in asset
    assert "renderRlOperatorControls(root, operator)" in asset
    assert "renderSyntheticExitText(risk.synthetic_exit_rules || [])" in asset
    assert "data-rl-risk-reasons" in asset
    assert "control.enabled ? \"\" : \"disabled\"" in asset
    assert "data-rl-operator-action" in asset
    assert "syncStrategiesMode(root, state.activeMode || \"classic\")" in asset


def test_strategies_css_keeps_rl_ml_tab_responsive() -> None:
    css = STRATEGIES_CSS.read_text(encoding="utf-8")

    assert ".strategies-rl-grid" in css
    assert ".strategies-rl-cards" in css
    assert ".strategies-rl-operator-actions .rh-button[disabled]" in css
    assert "#strategies-rl-ml-panel .strategies-pill" in css
    assert "overflow-wrap: anywhere" in css
    assert ".strategies-rl-slots" in css
    assert ".strategies-rl-risk-rules" in css
    assert "overflow: hidden" in css
    assert "max-width: 100%" in css
    assert "@media (max-width: 720px)" in css
    assert ".strategies-table--rl-slots" in css
