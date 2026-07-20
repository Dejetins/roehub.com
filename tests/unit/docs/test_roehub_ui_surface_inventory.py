from __future__ import annotations

import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_REGISTRY_PATH = _REPO_ROOT / "docs/architecture/apps/web/roehub-ui-surface-registry-v1.json"
_PUBLIC_REGISTRY_PATH = (
    _REPO_ROOT / "docs/architecture/apps/web/roehub-public-site-surface-registry-v1.json"
)
_SURFACE_ID = re.compile(r"^[a-z][a-z0-9]*(?:[._][a-z0-9]+)+$")


def _read_registry(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_local_ui_surface_registry_has_stable_referenced_inventory() -> None:
    registry = _read_registry(_LOCAL_REGISTRY_PATH)

    assert registry["schema_version"] == "1.0"
    assert registry["scope"] == "local_platform"
    assert registry["counts"] == {
        "current_route_patterns": 28,
        "surface_records": 33,
        "journeys": 12,
    }

    routes = registry["route_inventory"]
    surfaces = registry["surfaces"]
    journeys = registry["journeys"]
    bindings = registry["api_bindings"]
    assert isinstance(routes, list)
    assert isinstance(surfaces, list)
    assert isinstance(journeys, list)
    assert isinstance(bindings, list)
    assert len(routes) == registry["counts"]["current_route_patterns"]
    assert len(surfaces) == registry["counts"]["surface_records"]
    assert len(journeys) == registry["counts"]["journeys"]

    route_ids = [route["route_id"] for route in routes]
    binding_ids = [binding["binding_id"] for binding in bindings]
    surface_ids = [surface["surface_id"] for surface in surfaces]
    assert len(route_ids) == len(set(route_ids))
    assert len(binding_ids) == len(set(binding_ids))
    assert len(surface_ids) == len(set(surface_ids))
    assert all(_SURFACE_ID.fullmatch(surface_id) for surface_id in surface_ids)

    known_statuses = set(registry["status_values"])
    for surface in surfaces:
        assert surface["status"] in known_statuses
        assert surface["source_refs"]
        assert set(surface["route_refs"]).issubset(route_ids)
        assert set(surface["api_binding_ids"]).issubset(binding_ids)
        assert "penpot_" not in json.dumps(surface, sort_keys=True).lower()
    for journey in journeys:
        assert set(journey["surface_ids"]).issubset(surface_ids)
        assert set(journey["route_refs"]).issubset(route_ids)


def test_public_site_registry_is_separate_and_target_traceable() -> None:
    registry = _read_registry(_PUBLIC_REGISTRY_PATH)

    assert registry["schema_version"] == "1.0"
    assert registry["scope"] == "public_site_roehub.com"
    assert registry["implemented_site_evidence"]["status"] == "not_found"
    surfaces = registry["surfaces"]
    assert isinstance(surfaces, list)
    assert len(surfaces) == registry["counts"]["surface_records"] == 23
    actual_route_patterns = sum(len(surface["routes"]) for surface in surfaces)
    assert actual_route_patterns == registry["counts"]["route_patterns"]

    allowed_statuses = set(registry["status_values"])
    surface_ids = [surface["surface_id"] for surface in surfaces]
    assert len(surface_ids) == len(set(surface_ids))
    assert all(_SURFACE_ID.fullmatch(surface_id) for surface_id in surface_ids)
    for surface in surfaces:
        assert surface["status"] in allowed_statuses
        assert surface["source_refs"]
        assert surface["routes"]
        assert "current_observed" not in surface["status"]
        assert "penpot_" not in json.dumps(surface, sort_keys=True).lower()
