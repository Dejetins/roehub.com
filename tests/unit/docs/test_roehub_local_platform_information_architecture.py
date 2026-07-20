from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
WEB_ARCHITECTURE = REPO_ROOT / "docs" / "architecture" / "apps" / "web"
SOURCE_REGISTRY = WEB_ARCHITECTURE / "roehub-ui-surface-registry-v1.json"
PUBLIC_REGISTRY = WEB_ARCHITECTURE / "roehub-public-site-surface-registry-v1.json"
SCREEN_REGISTRY = WEB_ARCHITECTURE / "roehub-local-platform-screen-registry-v1.json"
ACCESS_CONTRACT = WEB_ARCHITECTURE / "roehub-local-platform-access-and-route-contract-v1.json"
ARCHITECTURE_DOCUMENT = WEB_ARCHITECTURE / "roehub-local-platform-information-architecture-v1.md"
_LOCAL_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((?!https?://|#)([^)#]+)(?:#[^)]+)?\)")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_local_screen_registry_accounts_for_inventory_and_journeys() -> None:
    source = _load_json(SOURCE_REGISTRY)
    public = _load_json(PUBLIC_REGISTRY)
    screens = _load_json(SCREEN_REGISTRY)

    assert screens["schema_id"] == "io.roehub.local-platform-screen-registry/v1"
    assert screens["status"] in {
        "candidate_target_architecture",
        "ready_for_product_review",
        "accepted_target_architecture",
    }
    assert screens["scope"] == "self_hosted_local_platform_only"
    assert screens["supported_widths"] == [820, 1024, 1440]
    assert screens["excluded_widths"] == [390]
    assert screens["public_site_surfaces_allowed"] is False

    screen_rows = screens["screens"]
    screen_ids = [row["screen_id"] for row in screen_rows]
    assert len(screen_ids) == len(set(screen_ids))

    assigned_surface_ids = [
        surface_id for row in screen_rows for surface_id in row["source_surface_ids"]
    ]
    source_surface_ids = [row["surface_id"] for row in source["surfaces"]]
    assert len(assigned_surface_ids) == len(set(assigned_surface_ids))
    assert set(assigned_surface_ids) == set(source_surface_ids)

    public_surface_ids = {row["surface_id"] for row in public["surfaces"]}
    assert public_surface_ids.isdisjoint(assigned_surface_ids)
    assert not any(
        route and route.startswith("https://roehub.com")
        for route in (row["canonical_route"] for row in screen_rows)
    )

    entrypoints = screens["journey_entrypoints"]
    journey_ids = [row["journey_id"] for row in entrypoints]
    source_journey_ids = [row["journey_id"] for row in source["journeys"]]
    assert len(journey_ids) == len(set(journey_ids))
    assert set(journey_ids) == set(source_journey_ids)

    known_screen_ids = set(screen_ids)
    for row in screen_rows:
        assert row["target_status"]
        assert isinstance(row["required_states"], list)
    for row in entrypoints:
        assert row["entry_screen_id"] in known_screen_ids
        assert set(row["terminal_screen_ids"]).issubset(known_screen_ids)


def test_access_contract_is_deny_by_default_and_covers_mutations() -> None:
    source = _load_json(SOURCE_REGISTRY)
    access = _load_json(ACCESS_CONTRACT)

    assert access["schema_id"] == ("io.roehub.local-platform-access-and-route-contract/v1")
    assert access["status"] in {
        "candidate_target_architecture",
        "ready_for_product_review",
        "accepted_target_architecture",
    }
    assert access["enforcement_authority"] == "server"
    assert access["default_decision"] == "deny"

    roles = set(access["organization_roles"])
    assert roles == {"owner", "admin", "operator", "trader", "viewer"}
    assert "installation_owner" not in roles
    overlay = access["authority_overlays"][0]
    assert overlay["authority_id"] == "installation_owner"
    assert "organization_membership" in overlay["does_not_imply"]
    assert "secret_reveal" in overlay["does_not_imply"]

    delegation = access["delegation_contract"]
    assert delegation["status"] == "target_not_implemented"
    assert delegation["grantor_roles"] == ["owner"]
    assert "owner_cannot_self_grant" in delegation["rules"]
    assert "grantee_cannot_redelegate" in delegation["rules"]
    assert "missing_expired_or_revoked_grant_denies" in delegation["rules"]
    assert {
        "installation.trust.manage",
        "installation.resources.manage",
        "installation.recovery.execute",
        "connections.secret.reveal",
        "roles.manage",
    }.issubset(delegation["non_delegable_authorities"])

    safe_actions = access["operator_safe_action_policy"]
    assert safe_actions["status"] == "target_not_implemented"
    assert safe_actions["default_decision"] == "deny"
    assert {row["action_id"] for row in safe_actions["allowed_actions"]} == {
        "strategy.stop",
        "backtest.cancel",
        "backtest.retry",
        "execution.reconcile",
        "connection.recheck",
        "connection.disconnect",
        "service.diagnostics",
        "service.restart_stopped",
    }
    restart_action = next(
        row
        for row in safe_actions["allowed_actions"]
        if row["action_id"] == "service.restart_stopped"
    )
    assert "organization_owned_service" in restart_action["conditions"]
    assert "shared_installation_service_forbidden" in restart_action["conditions"]

    mutation_envelope = access["browser_mutation_envelope"]
    assert mutation_envelope["status"] == "target_not_implemented_where_absent"
    assert mutation_envelope["failure_decision"] == "deny"
    assert "fail_closed_same_origin_or_csrf" in mutation_envelope["required_checks"]
    assert "selected_organization_and_object_scope" in mutation_envelope["required_checks"]
    assert "server_role_capability_or_authority" in mutation_envelope["required_checks"]
    assert "audit_for_security_or_operational_changes" in mutation_envelope["required_checks"]
    assert "same_key_different_payload_rejected" in mutation_envelope["idempotency_rules"]

    capabilities = access["capabilities"]
    capability_ids = [row["capability_id"] for row in capabilities]
    assert len(capability_ids) == len(set(capability_ids))
    known_capability_ids = set(capability_ids)
    assert set(overlay["grants"]).issubset(known_capability_ids)
    delegable_capability_ids = set(delegation["delegable_capability_ids"])
    assert delegable_capability_ids.issubset(known_capability_ids)
    mutation_capability_ids = set(mutation_envelope["capability_ids"])
    assert mutation_capability_ids.issubset(known_capability_ids)
    assert "installation.trust.manage" in mutation_capability_ids

    grants_by_capability: dict[str, set[str]] = {}
    for row in capabilities:
        granted_roles = {role for grant in row["grants"] for role in grant["roles"]}
        denied_roles = set(row["denied_roles"])
        assert granted_roles.isdisjoint(denied_roles)
        assert granted_roles | denied_roles == roles
        assert row["requirements"]
        assert row["current_enforcement"]
        assert row["source_refs"]
        for source_ref in row["source_refs"]:
            source_path = source_ref.split("#", 1)[0].split(":", 1)[0]
            if "/" in source_path:
                assert (REPO_ROOT / source_path).exists(), source_ref
        grants_by_capability[row["capability_id"]] = granted_roles
        if row["capability_id"] in mutation_capability_ids:
            assert "browser_mutation_envelope" in row["requirements"]
        for grant in row["grants"]:
            if grant["scope"] == "delegated_organization":
                assert grant["condition"].startswith("delegation:")
                assert grant["condition"].removeprefix("delegation:") in (delegable_capability_ids)

    mutation_surface_ids = {row["surface_id"] for row in source["surfaces"] if row["mutations"]}
    policies = access["surface_policies"]
    policy_surface_ids = [row["surface_id"] for row in policies]
    assert len(policy_surface_ids) == len(set(policy_surface_ids))
    assert set(policy_surface_ids) == mutation_surface_ids
    for row in policies:
        assert set(row["capability_ids"]).issubset(known_capability_ids)
        assert row["classification"]

    forbidden_for_operator = {
        "strategies.manage",
        "strategies.run",
        "strategies.manual_trade",
        "models.manage",
        "models.promote_or_rollback",
        "backtests.manage_own",
        "backtests.promote_own",
        "connections.manage",
        "connections.secret.reveal",
        "admin.members.manage",
        "admin.plugins.manage",
        "installation.resources.manage",
        "installation.recovery.execute",
    }
    assert all(
        "operator" not in grants_by_capability[capability_id]
        for capability_id in forbidden_for_operator
    )
    assert grants_by_capability["connections.secret.reveal"] == set()


def test_routes_are_unique_and_compatibility_decisions_are_explicit() -> None:
    screens = _load_json(SCREEN_REGISTRY)
    access = _load_json(ACCESS_CONTRACT)

    routes = access["canonical_routes"]
    route_ids = [row["route_id"] for row in routes]
    route_patterns = [row["pattern"] for row in routes]
    assert len(route_ids) == len(set(route_ids))
    assert len(route_patterns) == len(set(route_patterns))

    known_screen_ids = {row["screen_id"] for row in screens["screens"]}
    known_capability_ids = {row["capability_id"] for row in access["capabilities"]}
    for row in routes:
        assert row["screen_id"] in known_screen_ids
        if row["access"].startswith("capability:"):
            assert row["access"].removeprefix("capability:") in known_capability_ids

    migrations = {row["migration_id"]: row for row in access["route_migrations"]}
    assert {
        "route.root_state_gateway",
        "route.legacy_register",
        "route.settings_market_data",
        "route.settings_default",
        "route.models_rl_ml_alias",
        "route.legacy_runbook",
        "route.fastapi_docs_collision",
    } == set(migrations)
    assert migrations["route.models_rl_ml_alias"]["to"] == "/models"
    assert migrations["route.legacy_runbook"]["to"] == ("/docs/operator/runbooks/{runbook_id}/")
    assert migrations["route.fastapi_docs_collision"]["compatibility"] == ("breaking-change")
    assert "/api/docs" in migrations["route.fastapi_docs_collision"]["from"]
    assert "/_internal/core-api/docs" in migrations["route.fastapi_docs_collision"]["to"]
    assert all(
        row["implementation_status"] == "target_not_implemented" for row in migrations.values()
    )


def test_local_architecture_markdown_links_resolve() -> None:
    document = ARCHITECTURE_DOCUMENT.read_text(encoding="utf-8")
    links = _LOCAL_MARKDOWN_LINK.findall(document)
    assert links
    for link in links:
        assert (ARCHITECTURE_DOCUMENT.parent / link).resolve().exists(), link
