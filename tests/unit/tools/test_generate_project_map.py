from __future__ import annotations

# Synthetic fixture lines mirror import statements and may exceed the project line limit.
# ruff: noqa: E501
import json
import subprocess
from pathlib import Path

from tools.docs.generate_project_map import ProjectMapError, build_map, run_generator

CATALOG = """
schema_version = 1
project = "Synthetic"
description = "Synthetic map"

[[areas]]
id = "domain"
title = "Domain"
purpose = "Rules"
paths = ["src/trading/contexts/"]

[descriptions.contexts]
alpha = "Alpha context"
beta = "Beta context"

[descriptions.apps]
api = "API"

[descriptions.workers]

[[runtime_nodes]]
id = "api"
title = "API"
kind = "service"
paths = ["apps/api/"]

[[flows]]
source = "api"
target = "api"
label = "self"

[[agent_routes]]
match = "API"
components = ["app:api"]
read_first = ["apps/api/"]
skills = ["backend-quality-gates"]
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _write(tmp_path / "docs/architecture/project-map/project-map.toml", CATALOG)
    _write(tmp_path / "src/trading/contexts/alpha/service.py", "from ..beta import model\n")
    _write(tmp_path / "src/trading/contexts/beta/model.py", "VALUE = 1\n")
    _write(tmp_path / "apps/api/main/app.py", "from trading.contexts.alpha import service\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    return tmp_path


def test_build_map_discovers_inventory_components_and_import_edges(tmp_path: Path) -> None:
    data = build_map(_repo(tmp_path))

    assert data["inventory_file_count"] == 4
    assert {component["id"] for component in data["components"]} == {
        "app:api",
        "context:alpha",
        "context:beta",
    }
    assert {
        (edge["source"], edge["target"]) for edge in data["dependency_edges"]
    } == {
        ("app:api", "context:alpha"),
        ("context:alpha", "context:beta"),
    }


def test_generator_is_deterministic_and_check_detects_drift(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    assert run_generator(repo, check=False) == 0
    first = (repo / "docs/architecture/project-map/project-map.json").read_text(encoding="utf-8")
    assert run_generator(repo, check=True) == 0
    assert json.loads(first)["project"] == "Synthetic"

    _write(repo / "src/trading/contexts/beta/extra.py", "VALUE = 2\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    assert run_generator(repo, check=True) == 1
    assert run_generator(repo, check=False) == 0
    assert run_generator(repo, check=True) == 0


def test_build_map_fails_closed_on_python_parse_error(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo / "src/trading/contexts/beta/broken.py", "def broken(:\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)

    try:
        build_map(repo)
    except ProjectMapError as error:
        assert "broken.py:parse-error:SyntaxError" in str(error)
    else:
        raise AssertionError("invalid Python must fail project-map analysis")


def test_build_map_validates_catalog_references(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    catalog_path = repo / "docs/architecture/project-map/project-map.toml"
    catalog_path.write_text(
        catalog_path.read_text(encoding="utf-8").replace(
            'components = ["app:api"]', 'components = ["app:missing"]'
        ),
        encoding="utf-8",
    )

    try:
        build_map(repo)
    except ProjectMapError as error:
        assert "unknown agent route component: app:missing" in str(error)
    else:
        raise AssertionError("unknown catalog references must fail project-map analysis")
