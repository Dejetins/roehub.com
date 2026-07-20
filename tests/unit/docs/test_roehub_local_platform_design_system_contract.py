from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_ARCHITECTURE = REPO_ROOT / "docs" / "architecture" / "apps" / "web"
SCREEN_REGISTRY = WEB_ARCHITECTURE / "roehub-local-platform-screen-registry-v1.json"
TOKEN_CONTRACT = WEB_ARCHITECTURE / "roehub-local-platform-design-token-contract-v1.json"
COMPONENT_REGISTRY = WEB_ARCHITECTURE / "roehub-local-platform-component-registry-v1.json"
DESIGN_SYSTEM_DOCUMENT = WEB_ARCHITECTURE / "roehub-local-platform-design-system-contract-v1.md"
ARCHITECTURE_INDEX = REPO_ROOT / "docs" / "architecture" / "README.md"
PROJECT_MAP = REPO_ROOT / "docs" / "architecture" / "project-map" / "project-map.json"
_LOCAL_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)#]+)(?:#[^)]+)?\)")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_luminance(hex_color: str) -> float:
    assert re.fullmatch(r"#[0-9A-Fa-f]{6}", hex_color), hex_color

    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (int(hex_color[index : index + 2], 16) for index in (1, 3, 5))
    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def _contrast_ratio(foreground: str, background: str) -> float:
    lightest, darkest = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lightest + 0.05) / (darkest + 0.05)


def test_tokens_are_exactly_scoped_and_contrast_checked() -> None:
    tokens = _load_json(TOKEN_CONTRACT)

    assert tokens["schema_id"] == "io.roehub.local-platform-design-token-contract/v1"
    assert tokens["status"] == "ready_for_product_review"
    assert tokens["scope"] == "self_hosted_local_platform_only"
    assert tokens["implementation_status"] == "contract_only_not_runtime_evidence"
    assert tokens["theme_ids"] == ["abyss", "graphite", "slate", "frost", "paper", "sand"]
    assert list(tokens["themes"]) == tokens["theme_ids"]
    assert tokens["supported_widths"] == [820, 1024, 1440]
    assert tokens["excluded_widths"] == [390]
    assert tokens["public_site_tokens_allowed"] is False
    assert set(tokens["grid"]) == {"820", "1024", "1440", "table_overflow"}
    assert tokens["grid"]["table_overflow"] == "named_data_grid_container_only"
    assert tokens["motion"]["reduced_motion"]["status_semantics_preserved"] is True
    assert tokens["focus"]["visible_on_keyboard_focus"] is True
    assert tokens["accessibility"]["locales"] == ["ru", "en"]

    requirements = tokens["contrast_requirements"]
    for theme_id, theme in tokens["themes"].items():
        colors = theme["color"]
        for foreground_key, background_key in requirements["required_pairs"]:
            minimum = (
                requirements["non_text_focus_minimum"]
                if foreground_key == "focus_ring"
                else requirements["normal_text_minimum"]
            )
            assert _contrast_ratio(colors[foreground_key], colors[background_key]) >= minimum, (
                theme_id,
                foreground_key,
                background_key,
            )


def test_component_registry_covers_every_visual_screen_and_exact_state_set() -> None:
    source = _load_json(SCREEN_REGISTRY)
    registry = _load_json(COMPONENT_REGISTRY)

    assert registry["schema_id"] == "io.roehub.local-platform-component-registry/v1"
    assert registry["status"] == "ready_for_product_review"
    assert registry["scope"] == "self_hosted_local_platform_only"
    assert registry["implementation_status"] == "contract_only_not_runtime_evidence"
    assert registry["allowed_widths"] == [820, 1024, 1440]
    assert registry["excluded_widths"] == [390]
    assert registry["theme_ids"] == ["abyss", "graphite", "slate", "frost", "paper", "sand"]
    assert registry["compatibility"] == {
        "classification": "compatible-change",
        "routes_changed": False,
        "roles_changed": False,
        "capabilities_changed": False,
        "product_code_changed": False,
    }

    visual_kinds = {
        "route_screen",
        "route_flow",
        "persistent_shell",
        "system_state_family",
        "qa_only_screen",
    }
    expected = {
        row["screen_id"]: row["required_states"]
        for row in source["screens"]
        if row["kind"] in visual_kinds
    }
    compositions = {row["screen_id"]: row for row in registry["screen_compositions"]}
    assert set(compositions) == set(expected)

    family_ids = {row["component_id"] for row in registry["component_families"]}
    assert len(family_ids) == len(registry["component_families"])
    for screen_id, required_states in expected.items():
        composition = compositions[screen_id]
        assert composition["required_states"] == required_states
        assert composition["component_ids"]
        assert set(composition["component_ids"]).issubset(family_ids)

    excluded = {
        row["screen_id"]
        for row in registry["excluded_non_visual_or_historical_registry_entries"]
    }
    expected_excluded = {
        row["screen_id"]
        for row in source["screens"]
        if row["kind"] not in visual_kinds
    }
    assert excluded == expected_excluded
    assert excluded.isdisjoint(compositions)


def test_chart_progress_accessibility_and_future_package_boundaries_are_safe() -> None:
    registry = _load_json(COMPONENT_REGISTRY)

    chart = registry["chart_contract"]
    assert chart["schema_id"] == "io.roehub.chart-spec/v1"
    assert chart["allowed_renderers"] == ["canvas", "svg"]
    assert {
        "raw_javascript_callbacks",
        "executable_formatter",
        "html_tooltip",
        "secret_bearing_tooltip",
        "arbitrary_plugin",
        "unrestricted_echarts_option",
        "arbitrary_renderer",
        "arbitrary_dataset_transform",
    }.issubset(chart["prohibited"])
    assert {
        "units",
        "timezone",
        "source",
        "freshness",
        "table_alternative",
    }.issubset(chart["required_metadata"])

    progress = registry["progress_contract"]
    assert progress["queue_and_execution_eta_are_distinct"] is True
    assert progress["measured_percent_expression"] == "completed_units / total_units * 100"
    assert progress["eta_visibility"] == {
        "high": "show",
        "medium": "show",
        "low": "suppress_duration",
        "insufficient": "suppress_duration",
    }
    assert set(progress["terminal_states"]) == {"completed", "failed", "cancelled"}
    assert progress["active_bar_hidden_after_terminal"] is True
    assert {
        "decorative_timer",
        "false_100_percent",
        "success_without_terminal_server_result",
    }.issubset(progress["prohibited"])

    packages = {row["package"]: row["owns"] for row in registry["future_package_bindings"]}
    assert set(packages) == {
        "@roehub/tokens",
        "@roehub/ui",
        "@roehub/charts",
        "@roehub/localization",
        "@roehub/web-contracts",
    }
    assert "rh.chart" in packages["@roehub/charts"]
    assert "rh.progress.job" in packages["@roehub/ui"]
    assert registry["future_design_library"]["file_name"] == "Roehub — Design System"
    assert registry["future_design_library"]["artifact_status"] == "not_created_by_this_ticket"
    assert {
        "component_does_not_grant_capability",
        "stored_secret_never_component_input",
        "no_public_site_composition",
        "no_mobile_390_variant",
        "no_runtime_implementation_claim",
    }.issubset(registry["invariants"])


def test_design_document_links_index_and_project_map_are_current() -> None:
    document = DESIGN_SYSTEM_DOCUMENT.read_text(encoding="utf-8")
    assert "Статус: `ready_for_product_review`." in document
    assert "390" in document
    assert "raw JavaScript" in document
    assert "ECharts" in document
    assert "@roehub/charts" in document
    assert "Penpot resource has been read or created" in document

    links = _LOCAL_MARKDOWN_LINK.findall(document)
    assert links
    for link in links:
        assert (DESIGN_SYSTEM_DOCUMENT.parent / link).resolve().exists(), link

    assert (
        "roehub-local-platform-design-system-contract-v1.md"
        in ARCHITECTURE_INDEX.read_text(encoding="utf-8")
    )
    project_map = _load_json(PROJECT_MAP)
    inventory_paths = {row["path"] for row in project_map["inventory"]}
    assert str(DESIGN_SYSTEM_DOCUMENT.relative_to(REPO_ROOT)) in inventory_paths
    assert str(TOKEN_CONTRACT.relative_to(REPO_ROOT)) in inventory_paths
    assert str(COMPONENT_REGISTRY.relative_to(REPO_ROOT)) in inventory_paths
