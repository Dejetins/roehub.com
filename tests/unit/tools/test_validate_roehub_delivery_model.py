from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools.delivery.validate_roehub_delivery_model import (
    ACTIVE_REFERENCE_PATHS,
    ARCHITECTURE_SPIKE_TICKET_ID,
    DESIGN_BOUNDARY_TICKET_ID,
    FIGMA_FOUNDATIONS_TICKET_ID,
    GRAPH_PATH,
    LEGACY_PENPOT_TICKET_PATH,
    PROTOTYPE_README_PATH,
    UI_INSTRUCTIONS_COPY_TICKET_ID,
    _read_ticket_front_matter,
    validate_delivery_model,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _fixture_repo(tmp_path: Path) -> Path:
    fixture_root = tmp_path / "repo"
    graph = json.loads((REPO_ROOT / GRAPH_PATH).read_text(encoding="utf-8"))
    paths = [GRAPH_PATH, PROTOTYPE_README_PATH, *ACTIVE_REFERENCE_PATHS]
    paths.extend(Path(entry["path"]) for entry in graph["tickets"])
    for relative_path in paths:
        source = REPO_ROOT / relative_path
        target = fixture_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for entry in graph["tickets"]:
        metadata = _read_ticket_front_matter(REPO_ROOT / entry["path"])
        for evidence_path in metadata.get("evidence", []):
            source = REPO_ROOT / evidence_path
            if not source.is_file():
                continue
            target = fixture_root / evidence_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return fixture_root


def test_current_delivery_model_is_valid() -> None:
    assert validate_delivery_model(REPO_ROOT) == []


def test_duplicate_priority_entry_is_rejected(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    graph_path = fixture_root / GRAPH_PATH
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["priority_queue"].append(graph["priority_queue"][0])
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    assert "priority_queue contains duplicate ticket ids" in validate_delivery_model(fixture_root)


def test_active_reference_to_replaced_graph_is_rejected(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    plans_path = fixture_root / ".codex/PLANS.md"
    plans_path.write_text(
        plans_path.read_text(encoding="utf-8") + "\nroehub-server-authorization-stream-v1.json\n",
        encoding="utf-8",
    )

    assert any(
        error.startswith("active reference retains replaced graph")
        for error in validate_delivery_model(fixture_root)
    )


def test_registry_status_is_rejected(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    registry_path = fixture_root / ACTIVE_REFERENCE_PATHS[-1]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["status"] = "draft"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    assert "UI migration registry must not duplicate ticket status" in validate_delivery_model(
        fixture_root
    )


def test_unfinished_ticket_with_penpot_instruction_is_rejected(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    graph = json.loads((fixture_root / GRAPH_PATH).read_text(encoding="utf-8"))
    unfinished_ticket_id = graph["priority_queue"][0]
    unfinished_ticket_path = next(
        entry["path"] for entry in graph["tickets"] if entry["ticket_id"] == unfinished_ticket_id
    )
    ticket_path = fixture_root / unfinished_ticket_path
    ticket_path.write_text(
        ticket_path.read_text(encoding="utf-8") + "\nUse Penpot for the next design step.\n",
        encoding="utf-8",
    )

    assert any(
        error.startswith("unfinished ticket retains active Penpot instruction")
        for error in validate_delivery_model(fixture_root)
    )


def test_canonical_figma_identity_drift_is_rejected(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    registry_path = fixture_root / ACTIVE_REFERENCE_PATHS[-1]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["design_workspace"]["project_id"] = "wrong-project"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    assert "UI migration registry has invalid canonical Figma identity" in validate_delivery_model(
        fixture_root
    )


def test_replaced_penpot_ticket_is_rejected(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    legacy_ticket = fixture_root / LEGACY_PENPOT_TICKET_PATH
    legacy_ticket.parent.mkdir(parents=True, exist_ok=True)
    legacy_ticket.write_text("historical duplicate", encoding="utf-8")

    assert f"replaced Penpot ticket still exists: {LEGACY_PENPOT_TICKET_PATH}" in (
        validate_delivery_model(fixture_root)
    )


def test_architecture_spike_visual_rejection_is_required(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    graph = json.loads((fixture_root / GRAPH_PATH).read_text(encoding="utf-8"))
    ticket_path = next(
        entry["path"]
        for entry in graph["tickets"]
        if entry["ticket_id"] == ARCHITECTURE_SPIKE_TICKET_ID
    )
    candidate = fixture_root / ticket_path
    candidate.write_text(
        candidate.read_text(encoding="utf-8").replace(
            "visual_design_status: rejected_by_product_owner",
            "visual_design_status: approved",
        ),
        encoding="utf-8",
    )

    assert "architecture spike must record the product-owner visual rejection" in (
        validate_delivery_model(fixture_root)
    )


def test_agent_cannot_self_accept_ui_copy_or_figma_design(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    graph = json.loads((fixture_root / GRAPH_PATH).read_text(encoding="utf-8"))
    for ticket_id in (UI_INSTRUCTIONS_COPY_TICKET_ID, FIGMA_FOUNDATIONS_TICKET_ID):
        ticket_path = next(
            entry["path"] for entry in graph["tickets"] if entry["ticket_id"] == ticket_id
        )
        candidate = fixture_root / ticket_path
        candidate.write_text(
            candidate.read_text(encoding="utf-8").replace(
                "agent_self_acceptance: prohibited",
                "agent_self_acceptance: allowed",
            ),
            encoding="utf-8",
        )

    errors = validate_delivery_model(fixture_root)
    assert all(
        f"agent self-acceptance must be prohibited: {ticket_id}" in errors
        for ticket_id in (UI_INSTRUCTIONS_COPY_TICKET_ID, FIGMA_FOUNDATIONS_TICKET_ID)
    )


def test_prototype_readme_must_remain_not_a_design_source(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    readme_path = fixture_root / PROTOTYPE_README_PATH
    readme_path.write_text(
        readme_path.read_text(encoding="utf-8").replace(
            "not_a_design_source",
            "design_source",
        ),
        encoding="utf-8",
    )

    assert "prototype README must prohibit use as a design source" in (
        validate_delivery_model(fixture_root)
    )


def test_ui_copy_review_is_the_next_frontier_after_boundary_repair(tmp_path: Path) -> None:
    fixture_root = _fixture_repo(tmp_path)
    graph = json.loads((fixture_root / GRAPH_PATH).read_text(encoding="utf-8"))
    statuses = {}
    for ticket_id in (DESIGN_BOUNDARY_TICKET_ID, UI_INSTRUCTIONS_COPY_TICKET_ID):
        ticket_path = next(
            entry["path"] for entry in graph["tickets"] if entry["ticket_id"] == ticket_id
        )
        statuses[ticket_id] = _read_ticket_front_matter(fixture_root / ticket_path)["status"]

    assert statuses[DESIGN_BOUNDARY_TICKET_ID] == "accepted"
    assert statuses[UI_INSTRUCTIONS_COPY_TICKET_ID] == "ready"
    assert graph["priority_queue"][0] == UI_INSTRUCTIONS_COPY_TICKET_ID
    assert validate_delivery_model(fixture_root) == []
