#!/usr/bin/env python3
"""Validate Roehub's agent-governed Figma contract, manifest, and audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

CONTRACT_PATH = Path("docs/architecture/ui/roehub-ui-agent-delivery-contract-v1.json")
REGISTRY_SCHEMA_PATH = Path("docs/architecture/ui/roehub-ui-component-registry-schema-v1.json")
MANIFEST_SCHEMA_PATH = Path("docs/architecture/ui/roehub-ui-composition-manifest-schema-v1.json")
STANDARD_PATH = Path("docs/architecture/ui/roehub-agent-governed-figma-delivery-standard-v2.md")
PILOT_TICKET_PATH = Path(".codex/tickets/2026-07-31-roehub-ui-agent-governed-pilot.md")
PILOT_BRIEF_PATH = Path("docs/architecture/ui/roehub-backtests-process-pilot-brief-v1.md")

LIBRARY_FILE_KEY = "rgbNUPCuV7q2pARG4Cml8V"
PRODUCT_FILE_KEY = "nzKVsXuCmoTbHJGckHfK3T"
HISTORICAL_FILE_KEY = "GBzmB9evtzqnAYNjp9W1sr"
PILOT_TICKET_ID = "ROEHUB-UI-AGENT-GOVERNED-PILOT-2026-07-31"
VISUAL_STANDARD_PATH = Path(
    "docs/architecture/ui/roehub-linear-black-authenticated-workspace-visual-standard-v1.md"
)

PILOT_COMPONENT_IDS = {
    "backtests.toolbar",
    "backtests.job-row",
    "backtests.detail-dock-header",
    "feedback.degraded-freshness",
}
PILOT_COMPONENT_SLOTS = {
    "toolbar",
    "job-row",
    "detail-dock-header",
    "degraded-freshness",
}
PILOT_ACTIONS = {
    "open_details",
    "manual_refresh",
    "set_autorefresh",
    "close_detail",
}
PILOT_FIELDS = {
    "text_query",
    "job_state",
    "exchange",
    "market_type",
    "symbol",
    "launched_date_range",
    "auto_refresh_preset",
    "refresh_status",
    "job_id",
    "strategy",
    "indicator_summary",
    "period",
    "direction",
    "combinations",
    "best_return_pct",
    "best_sharpe",
    "avg_drawdown_pct",
    "profit_factor",
    "win_rate_pct",
    "trades_count",
    "state",
    "progress_percent",
    "created_at",
    "last_projection_at",
}
PILOT_STATES = {"completed", "degraded"}

LIBRARY_PAGES = [
    "00 Governance",
    "01 Foundations",
    "02 Icons",
    "03 Components",
    "04 Patterns",
    "80 Audit Sandbox",
    "90 Archive",
]
PRODUCT_PAGES = [
    "00 Governance",
    "01 Direction Review",
    "02 Candidate",
    "03 Accepted",
    "80 Audit Sandbox",
    "90 Archive",
]
DEPENDENCY_ORDER = [
    "product_ui_blueprint",
    "design_contract",
    "approved_visual_direction",
    "library_assets",
    "composition_manifest",
    "isolated_figma_candidate",
    "structural_and_visual_audit",
    "product_owner_decision",
    "accepted_figma_composition",
    "implementation_ticket",
]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_contract(repo_root: Path) -> list[str]:
    """Validate the durable two-file governance contract."""

    root = repo_root.resolve()
    errors: list[str] = []
    required_paths = (
        CONTRACT_PATH,
        REGISTRY_SCHEMA_PATH,
        MANIFEST_SCHEMA_PATH,
        STANDARD_PATH,
        PILOT_TICKET_PATH,
        PILOT_BRIEF_PATH,
    )
    for path in required_paths:
        if not (root / path).is_file():
            errors.append(f"missing required UI-delivery artifact: {path}")
    if errors:
        return errors

    try:
        contract = _read_json(root / CONTRACT_PATH)
        registry_schema = _read_json(root / REGISTRY_SCHEMA_PATH)
        manifest_schema = _read_json(root / MANIFEST_SCHEMA_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot read UI-delivery contract: {exc}"]

    if contract.get("schema_id") != "io.roehub.ui.agent-delivery-contract/v1":
        errors.append("agent-delivery contract schema_id is invalid")
    if contract.get("status") != "active":
        errors.append("agent-delivery contract must be active")
    if contract.get("standard") != str(STANDARD_PATH):
        errors.append("agent-delivery contract selects the wrong standard")
    if contract.get("component_registry_schema") != str(REGISTRY_SCHEMA_PATH):
        errors.append("agent-delivery contract selects the wrong component registry schema")

    coordinator = contract.get("coordinator")
    if not isinstance(coordinator, dict):
        errors.append("coordinator policy must be an object")
    else:
        expected = {
            "owner": "codex",
            "user_interface_count": 1,
            "executor_output_trust": "untrusted_until_all_required_gates_pass",
            "max_automatic_repair_attempts": 2,
            "agent_self_acceptance": "prohibited",
        }
        if coordinator != expected:
            errors.append("coordinator policy does not enforce the accepted trust boundary")

    figma = contract.get("figma")
    if not isinstance(figma, dict):
        errors.append("figma contract must be an object")
    else:
        library = figma.get("library")
        product = figma.get("product")
        if not isinstance(library, dict) or library.get("file_key") != LIBRARY_FILE_KEY:
            errors.append("canonical library file key is invalid")
        elif library.get("pages") != LIBRARY_PAGES:
            errors.append("canonical library page order is invalid")
        elif library.get("publication_owner") != "product_owner_via_figma_ui":
            errors.append("library publication owner is invalid")
        elif library.get("publication_verification_owner") != "codex_read_only":
            errors.append("library publication verification owner is invalid")
        if not isinstance(product, dict) or product.get("file_key") != PRODUCT_FILE_KEY:
            errors.append("canonical product file key is invalid")
        elif product.get("pages") != PRODUCT_PAGES:
            errors.append("canonical product page order is invalid")
        elif product.get("max_active_candidates") != 1:
            errors.append("product file must allow exactly one active candidate")

        historical = figma.get("historical_files")
        expected_historical = [
            {
                "file_key": HISTORICAL_FILE_KEY,
                "role": "historical_only",
                "agent_input": "forbidden",
            }
        ]
        if historical != expected_historical:
            errors.append("historical Figma file is not fail-closed")

    if contract.get("dependency_order") != DEPENDENCY_ORDER:
        errors.append("UI-delivery dependency order is invalid")
    pilot = contract.get("pilot")
    if not isinstance(pilot, dict) or pilot.get("ticket_id") != PILOT_TICKET_ID:
        errors.append("pilot ticket identity is invalid")
    elif pilot.get("brief") != str(PILOT_BRIEF_PATH):
        errors.append("pilot brief identity is invalid")
    elif pilot.get("runtime_implementation_authority") != "none":
        errors.append("design pilot must not grant runtime implementation authority")
    elif pilot.get("success_requires_deliberate_negative_gate_test") is not True:
        errors.append("design pilot must require a negative gate test")

    if registry_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append("component registry must use JSON Schema 2020-12")
    Draft202012Validator.check_schema(registry_schema)
    if manifest_schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        errors.append("composition manifest must use JSON Schema 2020-12")
    Draft202012Validator.check_schema(manifest_schema)

    standard_text = (root / STANDARD_PATH).read_text(encoding="utf-8")
    ticket_text = (root / PILOT_TICKET_PATH).read_text(encoding="utf-8")
    for required_literal in (LIBRARY_FILE_KEY, PRODUCT_FILE_KEY, HISTORICAL_FILE_KEY):
        if required_literal not in standard_text:
            errors.append(f"standard is missing Figma identity: {required_literal}")
    if "status: active" not in ticket_text:
        errors.append("pilot ticket must be active while awaiting its first checkpoint")
    if "agent_self_acceptance: prohibited" not in ticket_text:
        errors.append("pilot ticket must prohibit agent self-acceptance")
    return errors


def validate_registry(repo_root: Path, registry: dict[str, Any]) -> list[str]:
    """Validate the pilot component registry and its publication lifecycle."""

    schema = _read_json(repo_root.resolve() / REGISTRY_SCHEMA_PATH)
    errors = [
        f"registry {error.json_path}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema).iter_errors(registry),
            key=lambda item: list(item.absolute_path),
        )
    ]
    if errors:
        return errors

    if registry["ticket_id"] != PILOT_TICKET_ID:
        errors.append("registry ticket_id does not select the current pilot")
    if registry["visual_standard"]["path"] != str(VISUAL_STANDARD_PATH):
        errors.append("registry selects the wrong visual standard")

    assets = registry["assets"]
    stable_ids = [asset["stable_id"] for asset in assets]
    node_ids = [asset["node_id"] for asset in assets]
    component_keys = [
        asset["component_key"] for asset in assets if asset["component_key"] is not None
    ]
    if len(stable_ids) != len(set(stable_ids)):
        errors.append("registry contains duplicate stable asset IDs")
    if len(node_ids) != len(set(node_ids)):
        errors.append("registry contains duplicate Figma node IDs")
    if len(component_keys) != len(set(component_keys)):
        errors.append("registry contains duplicate published component keys")

    missing_components = sorted(PILOT_COMPONENT_IDS - set(stable_ids))
    if missing_components:
        errors.append(
            "registry is missing required pilot components: " + ", ".join(missing_components)
        )
    for asset in assets:
        if asset["stable_id"] not in PILOT_COMPONENT_IDS:
            continue
        if asset["kind"] not in {"component", "component_set", "pattern"}:
            errors.append(f"required registry asset has invalid kind: {asset['stable_id']}")
        if not asset["token_bindings"]:
            errors.append(f"required registry asset has no token bindings: {asset['stable_id']}")
        if not asset["text_style_bindings"]:
            errors.append(
                f"required registry asset has no text-style bindings: {asset['stable_id']}"
            )

    published = registry["state"] == "published_and_enabled"
    if published:
        if registry["library"]["publication_status"] != "published_and_enabled":
            errors.append("published registry has inconsistent library publication status")
        if registry["visual_standard"]["status"] != "accepted":
            errors.append("published registry requires an accepted visual-standard revision")
        if registry["visual_standard"]["accepted_revision_sha256"] is None:
            errors.append("published registry lacks the accepted visual-standard revision hash")
        for asset in assets:
            if asset["stable_id"] in PILOT_COMPONENT_IDS and asset["component_key"] is None:
                errors.append(
                    f"published registry asset lacks component key: {asset['stable_id']}"
                )
    return errors


def validate_manifest(
    repo_root: Path,
    manifest: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> list[str]:
    """Validate one candidate manifest against schema and cross-field invariants."""

    schema = _read_json(repo_root.resolve() / MANIFEST_SCHEMA_PATH)
    errors = [
        f"manifest {error.json_path}: {error.message}"
        for error in sorted(
            Draft202012Validator(schema).iter_errors(manifest),
            key=lambda item: list(item.absolute_path),
        )
    ]
    if errors:
        return errors

    if manifest["ticket_id"] != PILOT_TICKET_ID:
        errors.append("manifest ticket_id does not select the current pilot")
    approved_keys = set(manifest["library"]["approved_component_keys"])
    component_keys = [entry["component_key"] for entry in manifest["components"]]
    unknown = sorted(set(component_keys) - approved_keys)
    if unknown:
        errors.append(f"manifest uses unapproved component keys: {', '.join(unknown)}")
    slots = [entry["slot"] for entry in manifest["components"]]
    if len(slots) != len(set(slots)):
        errors.append("manifest contains duplicate component slots")
    missing_slots = sorted(PILOT_COMPONENT_SLOTS - set(slots))
    unexpected_slots = sorted(set(slots) - PILOT_COMPONENT_SLOTS)
    if missing_slots or unexpected_slots:
        errors.append(
            "manifest component slots mismatch: "
            f"missing={missing_slots}, unexpected={unexpected_slots}"
        )
    required_content = manifest["required_content"]
    for name, expected in (
        ("actions", PILOT_ACTIONS),
        ("fields", PILOT_FIELDS),
        ("states", PILOT_STATES),
    ):
        actual = set(required_content[name])
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            errors.append(
                f"manifest required {name} mismatch: missing={missing}, unexpected={unexpected}"
            )
    if (
        manifest["mutation_boundary"]["create_under_node_id"]
        != manifest["target"]["parent_node_id"]
    ):
        errors.append("mutation boundary differs from the target parent")
    if registry is not None:
        if registry["state"] != "published_and_enabled":
            errors.append("manifest requires a published_and_enabled component registry")
        registry_keys = {
            asset["component_key"]
            for asset in registry["assets"]
            if asset["stable_id"] in PILOT_COMPONENT_IDS and asset["component_key"] is not None
        }
        if approved_keys != registry_keys:
            errors.append(
                "manifest approved component keys differ from the published registry: "
                f"manifest={sorted(approved_keys)}, registry={sorted(registry_keys)}"
            )
    return errors


def validate_audit(manifest: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    """Compare an observed post-write Figma audit to its candidate manifest."""

    required_fields = {
        "schema_id",
        "manifest_id",
        "file_key",
        "page_id",
        "parent_node_id",
        "root_node_id",
        "top_level_nodes_created",
        "detached_instance_count",
        "raw_ui_node_count",
        "unknown_component_keys",
        "missing_required_actions",
        "missing_required_fields",
        "missing_required_states",
        "token_binding_violations",
        "text_style_binding_violations",
        "clipping_or_overflow_count",
        "outside_boundary_changes",
        "visual_review",
    }
    missing = sorted(required_fields - set(audit))
    if missing:
        return [f"audit is missing fields: {', '.join(missing)}"]

    errors: list[str] = []
    if audit["schema_id"] != "io.roehub.ui.figma-audit/v1":
        errors.append("audit schema_id is invalid")
    if audit["manifest_id"] != manifest["manifest_id"]:
        errors.append("audit manifest_id mismatch")
    if audit["file_key"] != manifest["target"]["file_key"]:
        errors.append("audit file_key mismatch")
    if audit["page_id"] != manifest["target"]["page_id"]:
        errors.append("audit page_id mismatch")
    parent_node_id = manifest["target"]["parent_node_id"]
    if audit["parent_node_id"] != parent_node_id:
        errors.append("audit parent_node_id mismatch")
    owned_node_ids = set(manifest["mutation_boundary"]["owned_node_ids"])
    if audit["root_node_id"] == parent_node_id:
        errors.append("audit candidate root must differ from the target parent")
    if audit["root_node_id"] not in owned_node_ids:
        errors.append("audit candidate root is not a manifest-owned node")
    if audit["top_level_nodes_created"] != 1:
        errors.append("audit must prove exactly one created top-level candidate")

    zero_fields = (
        "detached_instance_count",
        "raw_ui_node_count",
        "token_binding_violations",
        "text_style_binding_violations",
        "clipping_or_overflow_count",
    )
    for field in zero_fields:
        if audit[field] != 0:
            errors.append(f"audit gate failed: {field}={audit[field]}")
    empty_fields = (
        "unknown_component_keys",
        "missing_required_actions",
        "missing_required_fields",
        "missing_required_states",
        "outside_boundary_changes",
    )
    for field in empty_fields:
        if audit[field] != []:
            errors.append(f"audit gate failed: {field} is not empty")
    visual_review = audit["visual_review"]
    if not isinstance(visual_review, dict) or visual_review.get("status") != "passed":
        errors.append("independent visual review has not passed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()

    errors = validate_contract(args.repo_root)
    registry: dict[str, Any] | None = None
    if args.registry:
        registry = _read_json(args.registry)
        errors.extend(validate_registry(args.repo_root, registry))
    manifest: dict[str, Any] | None = None
    if args.manifest:
        if registry is None:
            errors.append("--manifest requires --registry")
        manifest = _read_json(args.manifest)
        errors.extend(validate_manifest(args.repo_root, manifest, registry))
    if args.audit:
        if manifest is None:
            errors.append("--audit requires --manifest")
        else:
            errors.extend(validate_audit(manifest, _read_json(args.audit)))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("OK: Roehub agent-governed UI delivery contract is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
