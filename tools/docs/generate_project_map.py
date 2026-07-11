from __future__ import annotations

# Generated Markdown prose and table templates intentionally contain long source lines.
# ruff: noqa: E501
import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

MAP_DIR = Path("docs/architecture/project-map")
CATALOG_PATH = MAP_DIR / "project-map.toml"
OUTPUT_PATHS = (
    MAP_DIR / "PROJECT_MAP.md",
    MAP_DIR / "project-map.mmd",
    MAP_DIR / "component-map.mmd",
    MAP_DIR / "project-map.json",
    MAP_DIR / "AGENT_GUIDE.md",
)
OWNED_DISCOVERY_PATHS = (
    CATALOG_PATH,
    *OUTPUT_PATHS,
    Path("tools/docs/generate_project_map.py"),
    Path("tests/unit/tools/test_generate_project_map.py"),
    Path(".github/workflows/update-project-map.yml"),
)


class ProjectMapError(RuntimeError):
    """Raised when repository sources or catalog references cannot be mapped safely."""


def _git_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode == 0:
        paths = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    else:
        paths = [
            path.relative_to(repo_root).as_posix()
            for path in repo_root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]
    generated = {path.as_posix() for path in OUTPUT_PATHS}
    paths = [path for path in paths if path not in generated]
    paths.extend(
        path.as_posix()
        for path in OWNED_DISCOVERY_PATHS
        if path not in OUTPUT_PATHS and (repo_root / path).exists()
    )
    return sorted(set(paths))


def _classify(path: str, areas: list[dict[str, Any]]) -> str:
    for area in areas:
        for prefix in area["paths"]:
            if path == prefix.rstrip("/") or path.startswith(prefix):
                return str(area["id"])
    return "repository-root"


def _kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix in {".md", ".rst"}:
        return "documentation"
    if suffix in {".yml", ".yaml", ".toml", ".json"}:
        return "configuration"
    if suffix in {".html", ".css", ".js", ".ts"}:
        return "frontend"
    if suffix in {".sql"}:
        return "migration"
    if suffix in {".sh", ".zsh", ".bat"}:
        return "script"
    return "other"


def _component_for_path(path: str) -> str | None:
    parts = Path(path).parts
    if len(parts) >= 4 and parts[:3] == ("src", "trading", "contexts"):
        if parts[3].endswith(".py"):
            return None
        return f"context:{parts[3]}"
    if len(parts) >= 3 and parts[:2] == ("apps", "worker"):
        return f"worker:{parts[2]}"
    if len(parts) >= 2 and parts[0] == "apps":
        if parts[1].endswith(".py"):
            return None
        return f"app:{parts[1]}"
    for name in ("shared_kernel", "platform", "integration", "fastpath"):
        if len(parts) >= 3 and parts[:3] == ("src", "trading", name):
            return f"core:{name}"
    return None


def _component_for_module(module: str) -> str | None:
    parts = module.split(".")
    if len(parts) >= 3 and parts[:2] == ["trading", "contexts"]:
        return f"context:{parts[2]}"
    if len(parts) >= 3 and parts[:2] == ["apps", "worker"]:
        return f"worker:{parts[2]}"
    if len(parts) >= 2 and parts[0] == "apps":
        return f"app:{parts[1]}"
    if len(parts) >= 2 and parts[0] == "trading" and parts[1] in {
        "shared_kernel",
        "platform",
        "integration",
        "fastpath",
    }:
        return f"core:{parts[1]}"
    return None


def _python_module(path: str) -> tuple[list[str], bool] | None:
    parts = list(Path(path).with_suffix("").parts)
    if parts[:1] == ["src"]:
        parts = parts[1:]
    if not parts or parts[0] not in {"apps", "trading"}:
        return None
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return parts, is_package


def _resolve_import_from(path: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    source = _python_module(path)
    if source is None:
        return node.module
    source_parts, is_package = source
    package = source_parts if is_package else source_parts[:-1]
    parents_to_drop = node.level - 1
    if parents_to_drop > len(package):
        return node.module
    base = package[: len(package) - parents_to_drop]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _import_edges(repo_root: Path, files: list[str]) -> list[dict[str, str]]:
    edges: set[tuple[str, str]] = set()
    errors: list[str] = []
    for relative in files:
        source = _component_for_path(relative)
        if source is None or not relative.endswith(".py"):
            continue
        path = repo_root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except OSError as error:
            errors.append(f"{relative}:read-error:{type(error).__name__}")
            continue
        except UnicodeDecodeError as error:
            errors.append(f"{relative}:decode-error:{type(error).__name__}")
            continue
        except SyntaxError as error:
            errors.append(f"{relative}:parse-error:SyntaxError:line={error.lineno or 0}")
            continue
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = _resolve_import_from(relative, node)
                if module:
                    modules.append(module)
        for module in modules:
            target = _component_for_module(module)
            if target and target != source:
                edges.add((source, target))
    if errors:
        raise ProjectMapError("Python analysis failed:\n" + "\n".join(sorted(errors)))
    return [
        {"source": source, "target": target, "kind": "python-import"}
        for source, target in sorted(edges)
    ]


def _components(
    repo_root: Path, files: list[str], catalog: dict[str, Any], edges: list[dict[str, str]]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in files:
        component = _component_for_path(path)
        if component:
            grouped[component].append(path)
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        outgoing[edge["source"]].add(edge["target"])
        incoming[edge["target"]].add(edge["source"])
    descriptions = catalog.get("descriptions", {})
    result: list[dict[str, Any]] = []
    for component_id, component_files in sorted(grouped.items()):
        kind, name = component_id.split(":", 1)
        description_group = "workers" if kind == "worker" else f"{kind}s"
        if kind == "core":
            description = "Общая техническая основа и кросс-контекстные примитивы."
        else:
            description = descriptions.get(description_group, {}).get(name, "Описание выводится из текущей структуры; уточнить при изменении ответственности.")
        entrypoints = [
            path
            for path in component_files
            if Path(path).name in {"main.py", "app.py", "__main__.py"}
        ]
        docs_prefix = f"docs/architecture/{name}/"
        docs = [path for path in files if path.startswith(docs_prefix) and path.endswith(".md")]
        result.append(
            {
                "id": component_id,
                "kind": kind,
                "name": name,
                "description": description,
                "roots": sorted({str(Path(path).parent) for path in component_files})[:12],
                "file_count": len(component_files),
                "entrypoints": entrypoints,
                "docs": docs[:25],
                "depends_on": sorted(outgoing[component_id]),
                "used_by": sorted(incoming[component_id]),
            }
        )
    return result


def _validate_catalog(data: dict[str, Any]) -> None:
    component_ids = {component["id"] for component in data["components"]}
    area_ids = {f'area:{area["id"]}' for area in data["areas"]}
    valid_route_targets = component_ids | area_ids
    runtime_ids = {node["id"] for node in data["runtime_nodes"]}
    errors: list[str] = []
    for route in data["agent_routes"]:
        for component in route["components"]:
            if component not in valid_route_targets:
                errors.append(f"unknown agent route component: {component}")
    for flow in data["runtime_flows"]:
        if flow["source"] not in runtime_ids:
            errors.append(f'unknown runtime flow source: {flow["source"]}')
        if flow["target"] not in runtime_ids:
            errors.append(f'unknown runtime flow target: {flow["target"]}')
    if errors:
        raise ProjectMapError("Catalog validation failed:\n" + "\n".join(sorted(errors)))


def build_map(repo_root: Path) -> dict[str, Any]:
    catalog_bytes = (repo_root / CATALOG_PATH).read_bytes()
    catalog = tomllib.loads(catalog_bytes.decode("utf-8"))
    files = _git_files(repo_root)
    areas = catalog["areas"]
    inventory = [
        {"path": path, "area": _classify(path, areas), "kind": _kind(path)} for path in files
    ]
    edges = _import_edges(repo_root, files)
    components = _components(repo_root, files, catalog, edges)
    structural_payload = {
        "catalog": hashlib.sha256(catalog_bytes).hexdigest(),
        "files": files,
        "edges": edges,
    }
    digest = hashlib.sha256(
        json.dumps(structural_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    data = {
        "schema_version": 1,
        "project": catalog["project"],
        "description": catalog["description"],
        "generated_by": "python -m tools.docs.generate_project_map",
        "structural_digest": digest,
        "inventory_file_count": len(files),
        "areas": [
            {
                **area,
                "file_count": sum(1 for item in inventory if item["area"] == area["id"]),
            }
            for area in areas
        ],
        "components": components,
        "dependency_edges": edges,
        "runtime_nodes": catalog["runtime_nodes"],
        "runtime_flows": catalog["flows"],
        "agent_routes": catalog["agent_routes"],
        "inventory": inventory,
    }
    _validate_catalog(data)
    return data


def _mermaid_id(value: str) -> str:
    return "n_" + "".join(character if character.isalnum() else "_" for character in value)


def render_mermaid(data: dict[str, Any]) -> str:
    lines = ["flowchart LR", "  classDef external fill:#fff3cd,stroke:#b8860b", "  classDef store fill:#e8f4ff,stroke:#3178c6", "  classDef service fill:#edf7ed,stroke:#2e7d32"]
    for node in data["runtime_nodes"]:
        node_id = _mermaid_id(node["id"])
        lines.append(f'  {node_id}["{node["title"]}"]')
        css_class = "external" if node["kind"] == "external" else "store" if node["kind"] == "data-store" else "service"
        lines.append(f"  class {node_id} {css_class}")
    for flow in data["runtime_flows"]:
        lines.append(
            f'  {_mermaid_id(flow["source"])} -->|"{flow["label"]}"| {_mermaid_id(flow["target"])}'
        )
    return "\n".join(lines) + "\n"


def render_component_mermaid(data: dict[str, Any]) -> str:
    lines = ["flowchart TB", "  classDef context fill:#e8f4ff,stroke:#3178c6", "  classDef app fill:#edf7ed,stroke:#2e7d32", "  classDef worker fill:#fff3cd,stroke:#b8860b", "  classDef core fill:#f3e8ff,stroke:#7b1fa2"]
    titles = {component["id"]: component["name"] for component in data["components"]}
    for component in data["components"]:
        component_id = _mermaid_id(component["id"])
        lines.append(f'  {component_id}["{component["id"]}"]')
        lines.append(f'  class {component_id} {component["kind"]}')
    for edge in data["dependency_edges"]:
        if edge["source"] in titles and edge["target"] in titles:
            lines.append(f'  {_mermaid_id(edge["source"])} --> {_mermaid_id(edge["target"])}')
    return "\n".join(lines) + "\n"


def _list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "—"


def render_markdown(data: dict[str, Any], mermaid: str, component_mermaid: str) -> str:
    area_rows = "\n".join(
        f'| `{area["id"]}` | {area["title"]} | {area["purpose"]} | {area["file_count"]} | {_list(area["paths"])} |'
        for area in data["areas"]
    )
    component_rows = "\n".join(
        f'| `{item["id"]}` | {item["description"]} | {item["file_count"]} | {_list(item["entrypoints"][:4])} | {_list(item["depends_on"])} |'
        for item in data["components"]
    )
    return f"""# Полная карта проекта Roehub

Этот документ — человекочитаемое представление единой карты проекта. Машиночитаемый источник для агентов — [`project-map.json`](project-map.json), семантический каталог — [`project-map.toml`](project-map.toml), правила использования — [`AGENT_GUIDE.md`](AGENT_GUIDE.md).

Карта построена детерминированно из каталога и фактического набора файлов/импортов. Generated-артефакты самой карты исключены из самоссылочного inventory. Текущий структурный digest: `{data['structural_digest']}`; учтено файлов: **{data['inventory_file_count']}**.

## Визуальная runtime-карта

```mermaid
{mermaid.rstrip()}
```

Исходник диаграммы отдельно: [`project-map.mmd`](project-map.mmd).

## Визуальная карта компонентов

Стрелка означает фактически обнаруженный Python import от источника к цели.

```mermaid
{component_mermaid.rstrip()}
```

Исходник диаграммы отдельно: [`component-map.mmd`](component-map.mmd).

## Текстовая карта репозитория

| Область | Название | Ответственность | Файлов | Корни |
|---|---|---|---:|---|
{area_rows}

## Компоненты и зависимости

Зависимости ниже вычисляются из Python imports. Это фактический статический граф, а не разрешение на новые cross-context imports.

| Компонент | Ответственность | Файлов | Точки входа | Зависит от |
|---|---|---:|---|---|
{component_rows}

## Данные, интеграции и runtime

- PostgreSQL: пользовательские, конфигурационные и операционные записи; миграции в `alembic/` и `migrations/postgres/`.
- ClickHouse: рыночные ряды, вычислительные и аналитические данные; миграции в `migrations/clickhouse/`.
- Redis: streams, команды, runtime coordination и cache; точные контракты ищутся в соответствующем контексте и runbook.
- Binance/Bybit: внешняя trust boundary; ключи и приватные payload не включаются в карту.
- Prometheus/Grafana/OpenTelemetry: метрики, dashboards и traces; эксплуатационные действия определяются runbooks.

## Как поддерживается актуальность

Локально: `python -m tools.docs.generate_project_map`.

Проверка без записи: `python -m tools.docs.generate_project_map --check`.

Workflow `.github/workflows/update-project-map.yml`:

1. на каждом branch `push` пересобирает пять generated-артефактов;
2. при изменениях коммитит только их и отправляет bot-коммит в ту же ветку;
3. на `pull_request` выполняет `--check` без записи;
4. не включает секреты, содержимое файлов или runtime payload — только пути, классификацию и import edges.

Для bot-коммита репозиторию требуется разрешение GitHub Actions `contents: write`. Защита ветки должна разрешать `github-actions[bot]` этот узкий commit path либо workflow честно завершится ошибкой.
"""


def render_agent_guide(data: dict[str, Any]) -> str:
    routes = []
    for route in data["agent_routes"]:
        routes.append(
            f'### {route["match"]}\n\n'
            f'- Компоненты: {_list(route["components"])}\n'
            f'- Читать сначала: {_list(route["read_first"])}\n'
            f'- Возможные workflow skills: {_list(route["skills"])}'
        )
    return """# Навигация агентов и субагентов по карте Roehub

Этот файл задаёт компактный маршрут чтения. Он не заменяет `AGENTS.md`, `.codex/AGENTS.md`, task prompt, ledger или локальные инструкции.

## Обязательный порядок

1. Прочитать применимый `AGENTS.md` и `.codex/AGENTS.md`.
2. Для cross-context, repository-wide или неясной задачи открыть `project-map.json` и выбрать только релевантные `areas`, `components`, `entrypoints`, `docs` и `agent_routes`.
3. Проверить указанные пути в текущем коде: карта — навигационный индекс, а не доказательство runtime-поведения.
4. Передать субагенту только нужный slice карты, точный outcome, owned paths и proof boundary. Не заставлять субагента читать весь inventory.
5. После добавления/перемещения компонентов выполнить генератор; generated-файлы вручную не редактировать.

## Машиночитаемые запросы

```bash
# Компонент и его зависимости
jq '.components[] | select(.id == "context:backtest")' docs/architecture/project-map/project-map.json

# Маршруты агента
jq '.agent_routes[]' docs/architecture/project-map/project-map.json

# Все файлы области
jq -r '.inventory[] | select(.area == "operations") | .path' docs/architecture/project-map/project-map.json
```

## Маршруты по типу работы

""" + "\n\n".join(routes) + "\n"


def expected_outputs(repo_root: Path) -> dict[Path, str]:
    data = build_map(repo_root)
    mermaid = render_mermaid(data)
    component_mermaid = render_component_mermaid(data)
    return {
        MAP_DIR / "PROJECT_MAP.md": render_markdown(data, mermaid, component_mermaid),
        MAP_DIR / "project-map.mmd": mermaid,
        MAP_DIR / "component-map.mmd": component_mermaid,
        MAP_DIR / "project-map.json": json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        MAP_DIR / "AGENT_GUIDE.md": render_agent_guide(data),
    }


def run_generator(repo_root: Path, *, check: bool) -> int:
    try:
        outputs = expected_outputs(repo_root)
    except ProjectMapError as error:
        print(str(error))
        return 2
    drift: list[str] = []
    for relative, content in outputs.items():
        target = repo_root / relative
        current = target.read_text(encoding="utf-8") if target.exists() else None
        if current != content:
            drift.append(relative.as_posix())
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
    if check and drift:
        print("Project map is out-of-date: " + ", ".join(drift))
        return 1
    action = "up-to-date" if check else "generated"
    print(f"OK: project map {action} ({len(outputs)} artifacts)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Roehub text, visual and agent project maps.")
    parser.add_argument("--check", action="store_true", help="Fail if generated artifacts drift.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_generator(args.repo_root.resolve(), check=args.check)


if __name__ == "__main__":
    sys.exit(main())
