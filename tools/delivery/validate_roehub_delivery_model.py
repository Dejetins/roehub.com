#!/usr/bin/env python3
"""Validate Roehub's repository-owned authenticated-platform delivery queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GRAPH_PATH = Path(".codex/delivery/graphs/roehub-authenticated-platform-delivery-v1.json")
GRAPH_ID = "ROEHUB-AUTHENTICATED-PLATFORM-DELIVERY-V1"
VALID_STATUSES = {"draft", "ready", "active", "blocked", "accepted", "superseded"}
LEGACY_GRAPH_PATHS = (
    Path(".codex/delivery/graphs/roehub-server-authorization-stream-v1.json"),
    Path(".codex/delivery/graphs/roehub-linear-workspace-ui-transition-v1.json"),
)
ACTIVE_REFERENCE_PATHS = (
    Path(".codex/AGENTS.md"),
    Path(".codex/PLANS.md"),
    Path(".codex/delivery/specs/roehub-linear-workspace-ui-transition.md"),
    Path("docs/architecture/ui/roehub-linear-ui-migration-registry-v1.json"),
)
RETIREMENT_POLICY_PATHS = ACTIVE_REFERENCE_PATHS[1:]
EXPECTED_TICKETS = {
    "ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20",
    "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20",
    "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20",
    "ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20",
    "ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20",
    "ROEHUB-PENPOT-LINEAR-VNEXT-FOUNDATIONS-2026-07-20",
    "ROEHUB-REACT-LINEAR-APPLICATION-SHELL-2026-07-20",
    "ROEHUB-AUTHZ-BACKTESTS-2026-07-20",
    "ROEHUB-BACKTESTS-LINEAR-GOLDEN-SLICE-2026-07-20",
    "ROEHUB-AUTHZ-DASHBOARD-DATA-2026-07-20",
    "ROEHUB-AUTHZ-STRATEGIES-2026-07-20",
    "ROEHUB-AUTHZ-CONNECTIONS-2026-07-20",
    "ROEHUB-AUTHZ-SETTINGS-ADMIN-2026-07-20",
    "ROEHUB-AUTHZ-OPERATIONS-INSTALLATION-2026-07-20",
    "ROEHUB-AUTHZ-INTEGRATION-PROOF-2026-07-20",
}
EXPECTED_DEPENDS_ON = {
    "ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20": [],
    "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20": ["ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20"],
    "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20": [
        "ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20"
    ],
    "ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20": [],
    "ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20": [
        "ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20"
    ],
    "ROEHUB-PENPOT-LINEAR-VNEXT-FOUNDATIONS-2026-07-20": [
        "ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20",
        "ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20",
    ],
    "ROEHUB-REACT-LINEAR-APPLICATION-SHELL-2026-07-20": [
        "ROEHUB-LINEAR-FRONTEND-ARCHITECTURE-SPIKE-2026-07-20",
        "ROEHUB-PENPOT-LINEAR-VNEXT-FOUNDATIONS-2026-07-20",
    ],
    "ROEHUB-AUTHZ-BACKTESTS-2026-07-20": [
        "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20",
        "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20",
    ],
    "ROEHUB-BACKTESTS-LINEAR-GOLDEN-SLICE-2026-07-20": [
        "ROEHUB-REACT-LINEAR-APPLICATION-SHELL-2026-07-20",
        "ROEHUB-AUTHZ-BACKTESTS-2026-07-20",
    ],
    "ROEHUB-AUTHZ-DASHBOARD-DATA-2026-07-20": [
        "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20",
        "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20",
    ],
    "ROEHUB-AUTHZ-STRATEGIES-2026-07-20": [
        "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20",
        "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20",
    ],
    "ROEHUB-AUTHZ-CONNECTIONS-2026-07-20": [
        "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20",
        "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20",
    ],
    "ROEHUB-AUTHZ-SETTINGS-ADMIN-2026-07-20": [
        "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20",
        "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20",
    ],
    "ROEHUB-AUTHZ-OPERATIONS-INSTALLATION-2026-07-20": [
        "ROEHUB-AUTHZ-SETTINGS-ADMIN-2026-07-20",
        "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20",
        "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20",
    ],
    "ROEHUB-AUTHZ-INTEGRATION-PROOF-2026-07-20": [
        "ROEHUB-AUTHZ-DASHBOARD-DATA-2026-07-20",
        "ROEHUB-AUTHZ-SETTINGS-ADMIN-2026-07-20",
        "ROEHUB-AUTHZ-STRATEGIES-2026-07-20",
        "ROEHUB-AUTHZ-BACKTESTS-2026-07-20",
        "ROEHUB-AUTHZ-CONNECTIONS-2026-07-20",
        "ROEHUB-AUTHZ-OPERATIONS-INSTALLATION-2026-07-20",
    ],
}
REQUIRED_ACCEPTED_TICKETS = {
    "ROEHUB-AUTHZ-CAPABILITY-KERNEL-2026-07-20",
    "ROEHUB-AUTHZ-DELEGATION-CORE-2026-07-20",
    "ROEHUB-AUTHZ-BROWSER-MUTATION-ENVELOPE-2026-07-20",
    "ROEHUB-LINEAR-REFERENCE-COMPLETION-2026-07-20",
}
EXPECTED_DELIVERY_AUTHORITY = {
    "ticket_status": "ticket_front_matter_only",
    "dependency_and_priority": "this_graph",
    "evidence": "ticket_evidence_files",
    "accepted_base": "main",
    "published_verification": "github_automated_checks",
    "external_tracker": "not_used",
    "linear_reference_role": "functional_structure_only",
}
EXPECTED_QUEUE_POLICY = {
    "execution_unit": "one_ready_ticket",
    "max_active_tickets": 1,
    "max_selected_ready_tickets": 1,
    "blocked_selection": "first_priority_ticket_with_accepted_dependencies",
    "external_tracker_for_status_or_order": "not_used",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _read_ticket_front_matter(path: Path) -> dict[str, Any]:
    parts = path.read_text(encoding="utf-8").split("---\n", 2)
    if len(parts) != 3 or parts[0]:
        raise ValueError(f"{path}: missing YAML front matter")
    values: dict[str, Any] = {}
    current_list: str | None = None
    for line in parts[1].splitlines():
        if line.startswith("  - ") and current_list:
            values[current_list].append(line.removeprefix("  - ").strip())
            continue
        if not line or line.startswith(" ") or ":" not in line:
            current_list = None
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if key in {"depends_on", "evidence"}:
            values[key] = []
            current_list = key if not value else None
        else:
            values[key] = value
            current_list = None
    return values


def _has_cycle(entries: dict[str, dict[str, Any]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(ticket_id: str) -> bool:
        if ticket_id in visited:
            return False
        if ticket_id in visiting:
            return True
        visiting.add(ticket_id)
        if any(
            dependency in entries and visit(dependency)
            for dependency in entries[ticket_id]["depends_on"]
        ):
            return True
        visiting.remove(ticket_id)
        visited.add(ticket_id)
        return False

    return any(visit(ticket_id) for ticket_id in entries)


def validate_delivery_model(repo_root: Path) -> list[str]:
    root = repo_root.resolve()
    graph_file = root / GRAPH_PATH
    if not graph_file.is_file():
        return [f"missing unified graph: {GRAPH_PATH}"]
    try:
        graph = _read_json(graph_file)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"cannot read unified graph: {exc}"]

    errors: list[str] = []
    if graph.get("schema_id") != "io.roehub.delivery.ticket-graph/v1":
        errors.append("unified graph schema_id is invalid")
    if graph.get("graph_id") != GRAPH_ID:
        errors.append("unified graph_id is invalid")
    if graph.get("status_authority") != "ticket_front_matter_only":
        errors.append("status authority must be ticket front matter only")
    if graph.get("delivery_authority") != EXPECTED_DELIVERY_AUTHORITY:
        errors.append("delivery_authority does not define the repository-owned model")
    if graph.get("queue_policy") != EXPECTED_QUEUE_POLICY:
        errors.append("queue_policy does not define the required single-ticket selection")
    for field in ("parallel_waves", "initial_ready_ticket_id"):
        if field in graph:
            errors.append(f"unified graph must not contain legacy field: {field}")

    raw_tickets = graph.get("tickets")
    if not isinstance(raw_tickets, list):
        return errors + ["unified graph tickets must be a list"]
    entries: dict[str, dict[str, Any]] = {}
    for entry in raw_tickets:
        if not isinstance(entry, dict) or not isinstance(entry.get("ticket_id"), str):
            errors.append("unified graph contains an invalid ticket entry")
            continue
        ticket_id = entry["ticket_id"]
        if ticket_id in entries:
            errors.append(f"duplicate ticket in unified graph: {ticket_id}")
            continue
        depends_on = entry.get("depends_on")
        if not isinstance(depends_on, list) or not all(
            isinstance(item, str) for item in depends_on
        ):
            errors.append(f"invalid depends_on for {ticket_id}")
            depends_on = []
        if "initial_status" in entry:
            errors.append(f"ticket entry must not duplicate status: {ticket_id}")
        entries[ticket_id] = {**entry, "depends_on": depends_on}
    if set(entries) != EXPECTED_TICKETS or len(entries) != len(EXPECTED_TICKETS):
        errors.append("unified graph must contain each of the 15 expected tickets exactly once")

    for ticket_id, expected_dependencies in EXPECTED_DEPENDS_ON.items():
        entry = entries.get(ticket_id)
        if not entry:
            continue
        if entry["depends_on"] != expected_dependencies:
            errors.append(f"depends_on drift for {ticket_id}")
        unknown = set(entry["depends_on"]) - set(entries)
        if unknown:
            errors.append(f"unknown dependencies for {ticket_id}: {sorted(unknown)}")
        if ticket_id in entry["depends_on"]:
            errors.append(f"self dependency for {ticket_id}")
    if entries and _has_cycle(entries):
        errors.append("unified graph contains a dependency cycle")

    tickets: dict[str, dict[str, Any]] = {}
    for ticket_id, entry in entries.items():
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            errors.append(f"missing ticket path for {ticket_id}")
            continue
        ticket_path = root / raw_path
        if not ticket_path.is_file():
            errors.append(f"missing ticket file for {ticket_id}: {raw_path}")
            continue
        try:
            metadata = _read_ticket_front_matter(ticket_path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        tickets[ticket_id] = metadata
        if metadata.get("ticket_id") != ticket_id:
            errors.append(f"ticket_id mismatch in {raw_path}")
        if metadata.get("ticket_graph") != str(GRAPH_PATH):
            errors.append(f"ticket_graph mismatch in {raw_path}")
        if metadata.get("depends_on") != entry["depends_on"]:
            errors.append(f"front-matter depends_on mismatch in {raw_path}")
        status = metadata.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"invalid status for {ticket_id}: {status!r}")
        if status == "accepted":
            evidence = metadata.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"accepted ticket has no evidence: {ticket_id}")
            elif any(not (root / item).is_file() for item in evidence):
                errors.append(f"accepted ticket has missing evidence: {ticket_id}")

    for ticket_id in REQUIRED_ACCEPTED_TICKETS:
        if tickets.get(ticket_id, {}).get("status") != "accepted":
            errors.append(f"historical accepted status changed: {ticket_id}")

    priority_queue = graph.get("priority_queue")
    if not isinstance(priority_queue, list) or not all(
        isinstance(item, str) for item in priority_queue
    ):
        errors.append("priority_queue must be a list of ticket ids")
        priority_queue = []
    if len(priority_queue) != len(set(priority_queue)):
        errors.append("priority_queue contains duplicate ticket ids")
    unfinished = {
        ticket_id
        for ticket_id, metadata in tickets.items()
        if metadata.get("status") not in {"accepted", "superseded"}
    }
    if set(priority_queue) != unfinished:
        errors.append("priority_queue must contain each unfinished graph ticket exactly once")

    ready = [
        ticket_id for ticket_id, metadata in tickets.items() if metadata.get("status") == "ready"
    ]
    active = [
        ticket_id for ticket_id, metadata in tickets.items() if metadata.get("status") == "active"
    ]
    if len(ready) > EXPECTED_QUEUE_POLICY["max_selected_ready_tickets"]:
        errors.append("more than one graph ticket is ready")
    if len(active) > EXPECTED_QUEUE_POLICY["max_active_tickets"]:
        errors.append("more than one graph ticket is active")
    if ready and active:
        errors.append("ready and active tickets cannot be selected at the same time")
    eligible = [
        ticket_id
        for ticket_id in priority_queue
        if tickets.get(ticket_id, {}).get("status") not in {"blocked", "accepted", "superseded"}
        and all(
            tickets.get(dependency, {}).get("status") == "accepted"
            for dependency in entries.get(ticket_id, {}).get("depends_on", [])
        )
    ]
    selected = ready + active
    if selected and not eligible:
        errors.append("selected ticket has no accepted dependency path")
    elif selected and selected[0] != eligible[0]:
        errors.append("selected ticket is not the first eligible priority ticket")

    legacy_names = [path.name for path in LEGACY_GRAPH_PATHS]
    for path in LEGACY_GRAPH_PATHS:
        if (root / path).exists():
            errors.append(f"replaced graph still exists: {path}")
    active_paths = [*ACTIVE_REFERENCE_PATHS, *(Path(entry["path"]) for entry in entries.values())]
    for path in active_paths:
        candidate = root / path
        if not candidate.is_file():
            continue
        content = candidate.read_text(encoding="utf-8")
        if any(legacy_name in content for legacy_name in legacy_names):
            errors.append(f"active reference retains replaced graph: {path}")
        if path in RETIREMENT_POLICY_PATHS and any(
            term in content.lower() for term in ("parallel", "worktree", "separate branch")
        ):
            errors.append(f"active delivery instruction retains retired coordination: {path}")

    registry_path = root / ACTIVE_REFERENCE_PATHS[-1]
    try:
        registry = _read_json(registry_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read UI migration registry: {exc}")
    else:
        if "status" in registry:
            errors.append("UI migration registry must not duplicate ticket status")
        clusters = registry.get("clusters")
        if isinstance(clusters, list) and any(
            isinstance(cluster, dict)
            and any(field in cluster for field in ("status", "state", "order"))
            for cluster in clusters
        ):
            errors.append(
                "UI migration registry clusters must not duplicate ticket status or priority"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    errors = validate_delivery_model(parser.parse_args().repo_root)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("OK: Roehub authenticated-platform delivery model is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
