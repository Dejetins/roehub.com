from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from tools.design.validate_roehub_ui_delivery import (
    LIBRARY_FILE_KEY,
    PILOT_ACTIONS,
    PILOT_COMPONENT_IDS,
    PILOT_FIELDS,
    PILOT_STATES,
    PILOT_TICKET_ID,
    PRODUCT_FILE_KEY,
    validate_audit,
    validate_contract,
    validate_manifest,
    validate_registry,
)

COMPONENT_KEYS = {
    "backtests.toolbar": "component-key-toolbar",
    "backtests.job-row": "component-key-row",
    "backtests.detail-dock-header": "component-key-detail-header",
    "feedback.degraded-freshness": "component-key-degraded",
}
COMPONENT_SLOTS = {
    "backtests.toolbar": "toolbar",
    "backtests.job-row": "job-row",
    "backtests.detail-dock-header": "detail-dock-header",
    "feedback.degraded-freshness": "degraded-freshness",
}


def _registry_asset(stable_id: str, index: int) -> dict[str, Any]:
    return {
        "stable_id": stable_id,
        "name": stable_id.replace(".", "/"),
        "kind": "component_set",
        "page_id": "5:4",
        "node_id": f"5:{100 + index}",
        "component_key": COMPONENT_KEYS[stable_id],
        "properties": {"density": "compact"},
        "variants": ["default", "focus"],
        "slots": [COMPONENT_SLOTS[stable_id]],
        "content_limits": {"strategy": 64},
        "token_bindings": ["color.surface", "size.control.compact"],
        "text_style_bindings": ["type.control"],
        "accessibility_names": [stable_id],
        "lifecycle_status": "published_and_enabled",
    }


def _registry() -> dict[str, Any]:
    return {
        "schema_id": "io.roehub.ui.component-registry/v1",
        "registry_id": "roehub.backtests.process-pilot.registry.v1",
        "ticket_id": PILOT_TICKET_ID,
        "state": "published_and_enabled",
        "visual_standard": {
            "path": (
                "docs/architecture/ui/"
                "roehub-linear-black-authenticated-workspace-visual-standard-v1.md"
            ),
            "status": "accepted",
            "accepted_revision_sha256": "a" * 64,
        },
        "library": {
            "file_key": LIBRARY_FILE_KEY,
            "file_name": "Roehub UI Library",
            "revision_id": "revision-1",
            "publication_status": "published_and_enabled",
        },
        "assets": [
            _registry_asset(stable_id, index)
            for index, stable_id in enumerate(sorted(PILOT_COMPONENT_IDS))
        ],
    }


def _manifest() -> dict[str, Any]:
    return {
        "schema_id": "io.roehub.ui.composition-manifest/v1",
        "manifest_id": "roehub.backtests.process-pilot.v1",
        "ticket_id": PILOT_TICKET_ID,
        "state": "executing",
        "target": {
            "file_key": PRODUCT_FILE_KEY,
            "page_id": "3:3",
            "page_name": "02 Candidate",
            "parent_node_id": "3:30",
        },
        "source_contracts": [
            "docs/architecture/ui/roehub-agent-governed-figma-delivery-standard-v2.md",
            "docs/architecture/ui/roehub-backtests-process-pilot-brief-v1.md",
        ],
        "viewport": {"width": 1440, "height": 900, "state": "degraded"},
        "library": {
            "file_key": LIBRARY_FILE_KEY,
            "publication_status": "published_and_enabled",
            "approved_component_keys": sorted(COMPONENT_KEYS.values()),
        },
        "components": [
            {
                "slot": COMPONENT_SLOTS[stable_id],
                "component_key": COMPONENT_KEYS[stable_id],
                "variant_properties": {"density": "compact"},
                "text_overrides": {"label": stable_id},
            }
            for stable_id in sorted(PILOT_COMPONENT_IDS)
        ],
        "required_content": {
            "actions": sorted(PILOT_ACTIONS),
            "fields": sorted(PILOT_FIELDS),
            "states": sorted(PILOT_STATES),
        },
        "mutation_boundary": {
            "create_under_node_id": "3:30",
            "owned_node_ids": ["3:31"],
            "max_top_level_nodes_created": 1,
        },
        "raw_node_allowlist": [],
    }


def _audit(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": "io.roehub.ui.figma-audit/v1",
        "manifest_id": manifest["manifest_id"],
        "file_key": PRODUCT_FILE_KEY,
        "page_id": "3:3",
        "parent_node_id": "3:30",
        "root_node_id": "3:31",
        "top_level_nodes_created": 1,
        "detached_instance_count": 0,
        "raw_ui_node_count": 0,
        "unknown_component_keys": [],
        "missing_required_actions": [],
        "missing_required_fields": [],
        "missing_required_states": [],
        "token_binding_violations": 0,
        "text_style_binding_violations": 0,
        "clipping_or_overflow_count": 0,
        "outside_boundary_changes": [],
        "visual_review": {"status": "passed", "reviewer": "independent"},
    }


def test_repository_contract_is_valid() -> None:
    assert validate_contract(Path.cwd()) == []


def test_published_component_registry_is_valid() -> None:
    assert validate_registry(Path.cwd(), _registry()) == []


def test_registry_rejects_missing_duplicate_and_unpublished_required_assets() -> None:
    missing_registry = _registry()
    missing_registry["assets"][0]["stable_id"] = "other.component"
    missing_errors = validate_registry(Path.cwd(), missing_registry)
    assert any("missing required pilot components" in error for error in missing_errors)

    duplicate_registry = _registry()
    duplicate_registry["assets"][1]["node_id"] = duplicate_registry["assets"][0]["node_id"]
    duplicate_registry["assets"][1]["component_key"] = duplicate_registry["assets"][0][
        "component_key"
    ]
    duplicate_errors = validate_registry(Path.cwd(), duplicate_registry)
    assert "registry contains duplicate Figma node IDs" in duplicate_errors
    assert "registry contains duplicate published component keys" in duplicate_errors

    unpublished_registry = _registry()
    unpublished_registry["assets"][0]["component_key"] = None
    unpublished_errors = validate_registry(Path.cwd(), unpublished_registry)
    assert any("lacks component key" in error for error in unpublished_errors)


def test_manifest_and_registry_are_consistent() -> None:
    registry = _registry()
    manifest = _manifest()

    assert validate_manifest(Path.cwd(), manifest, registry) == []


def test_manifest_rejects_unapproved_components_and_raw_nodes_separately() -> None:
    raw_node_manifest = _manifest()
    raw_node_manifest["raw_node_allowlist"] = ["RECTANGLE"]
    raw_node_errors = validate_manifest(Path.cwd(), raw_node_manifest, _registry())
    assert any("raw_node_allowlist" in error for error in raw_node_errors)

    component_manifest = _manifest()
    component_manifest["components"][0]["component_key"] = "unknown-component"
    component_errors = validate_manifest(Path.cwd(), component_manifest, _registry())
    assert any("unapproved component" in error for error in component_errors)


def test_manifest_rejects_each_missing_pilot_content_concept() -> None:
    for category in ("actions", "fields", "states"):
        manifest = _manifest()
        manifest["required_content"][category].pop()
        errors = validate_manifest(Path.cwd(), manifest, _registry())
        assert any(f"required {category} mismatch" in error for error in errors)


def test_audit_requires_exact_candidate_root_identity() -> None:
    manifest = _manifest()
    assert validate_audit(manifest, _audit(manifest)) == []

    parent_root_audit = _audit(manifest)
    parent_root_audit["root_node_id"] = parent_root_audit["parent_node_id"]
    parent_errors = validate_audit(manifest, parent_root_audit)
    assert "audit candidate root must differ from the target parent" in parent_errors

    unowned_root_audit = _audit(manifest)
    unowned_root_audit["root_node_id"] = "3:99"
    unowned_errors = validate_audit(manifest, unowned_root_audit)
    assert "audit candidate root is not a manifest-owned node" in unowned_errors

    zero_created_audit = _audit(manifest)
    zero_created_audit["top_level_nodes_created"] = 0
    zero_errors = validate_audit(manifest, zero_created_audit)
    assert "audit must prove exactly one created top-level candidate" in zero_errors


def test_audit_requires_zero_structural_violations_and_visual_pass() -> None:
    manifest = _manifest()
    broken_audit = deepcopy(_audit(manifest))
    broken_audit["detached_instance_count"] = 1
    broken_audit["visual_review"] = {"status": "failed"}
    errors = validate_audit(manifest, broken_audit)

    assert "audit gate failed: detached_instance_count=1" in errors
    assert "independent visual review has not passed" in errors
