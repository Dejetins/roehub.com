from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import jsonschema
import yaml

RUNBOOK_SCHEMA = Path("schemas/ops/runbook.schema.json")
LOCALE_SCHEMA = Path("schemas/ops/runbook-locale.schema.json")
CAPABILITIES = Path("schemas/ops/action-capabilities.json")
CANONICAL_DIR = Path("docs/runbooks/ops")
RU_LOCALE_DIR = Path("docs/runbooks/locales/ru")
GENERATED_RU_DIR = Path("docs/runbooks/generated/ru")
INDEX_PATH = Path("docs/runbooks/runbooks.json")
ALERT_RULE_DIRS = (
    Path("infra/monitoring/rules"),
)
GENERATED_ALERT_RULE_FILES = (
    Path("configs/installation/generated/base/observability/alerts.yml"),
)

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}"),
    re.compile(r"\b(?:ghp|github_pat|xoxb)-?[A-Za-z0-9_]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{12,}\b"),
    re.compile(r"(?i)\b(?:postgres(?:ql)?|redis)://[^\s:/]+:[^\s@]+@"),
)
FORBIDDEN_OPERATION_KEYS = {"argv", "command", "environment", "script", "shell"}
APPROVAL_ORDER = {"none": 0, "operator": 1, "installation_owner": 2}


class RunbookError(RuntimeError):
    """Raised when the runbook contract or generated outputs are unsafe."""


@dataclass(frozen=True)
class AlertRecord:
    name: str
    source: str
    runbook_link: str | None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunbookError(f"YAML root must be an object: {path}")
    return payload


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _schema_errors(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'/'.join(str(item) for item in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    ]


def _walk(value: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, (*path, str(index)))


def _validate_no_secret_or_shell_fields(payload: dict[str, Any], source: Path) -> None:
    for path, value in _walk(payload):
        if path and path[-1].lower() in FORBIDDEN_OPERATION_KEYS:
            raise RunbookError(
                f"arbitrary operation field is forbidden in {source}: {'.'.join(path)}"
            )
        if not isinstance(value, str):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(value):
                raise RunbookError(
                    f"secret-shaped value is forbidden in {source}: {'.'.join(path)}"
                )


def _narrative_ids(items: list[dict[str, str]]) -> set[str]:
    identifiers = [item["id"] for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise RunbookError(f"duplicate narrative ids: {identifiers}")
    return set(identifiers)


def _expected_translation_ids(runbook: dict[str, Any]) -> dict[str, set[str]]:
    spec = runbook["spec"]
    return {
        "prerequisites": _narrative_ids(spec["prerequisites"]),
        "symptoms": _narrative_ids(spec["symptoms"]),
        "diagnostics": {item["id"] for item in spec["diagnostics"]},
        "allowed_actions": {item["id"] for item in spec["allowed_actions"]},
        "rollback": {item["id"] for item in spec["rollback"]},
        "evidence_collect": _narrative_ids(spec["evidence"]["collect"]),
        "success_conditions": _narrative_ids(spec["evidence"]["success_conditions"]),
        "failure_conditions": _narrative_ids(spec["evidence"]["failure_conditions"]),
        "redaction_rules": _narrative_ids(spec["secret_redaction"]["rules"]),
        "monitoring_gaps": _narrative_ids(spec["monitoring_gaps"]),
        "warnings": _narrative_ids(spec["safety"]["warnings"]),
        "stop_conditions": _narrative_ids(spec["safety"]["stop_conditions"]),
    }


def _validate_locale_coverage(runbook: dict[str, Any], locale: dict[str, Any]) -> None:
    runbook_id = runbook["metadata"]["id"]
    if locale["metadata"]["runbook_id"] != runbook_id:
        raise RunbookError(f"locale/runbook id mismatch: {runbook_id}")
    translations = locale["translations"]
    for section, expected in _expected_translation_ids(runbook).items():
        actual = set(translations[section])
        if actual != expected:
            raise RunbookError(
                f"Russian locale coverage mismatch for {runbook_id}.{section}; "
                f"missing={sorted(expected - actual)}, stale={sorted(actual - expected)}"
            )
    for path, value in _walk(translations):
        if isinstance(value, str) and not CYRILLIC_RE.search(value):
            raise RunbookError(
                f"Russian locale narrative lacks Cyrillic text: {runbook_id}.{'.'.join(path)}"
            )


def _validate_canonical_language(runbook: dict[str, Any], source: Path) -> None:
    for path, value in _walk(runbook["spec"]):
        if isinstance(value, str) and CYRILLIC_RE.search(value):
            raise RunbookError(f"canonical narrative must be English: {source}:{'.'.join(path)}")


def _validate_capabilities(runbook: dict[str, Any], catalog: dict[str, Any]) -> None:
    runbook_id = runbook["metadata"]["id"]
    spec = runbook["spec"]
    diagnostic_ids: set[str] = set()
    for diagnostic in spec["diagnostics"]:
        identifier = diagnostic["id"]
        if identifier in diagnostic_ids:
            raise RunbookError(f"duplicate diagnostic id in {runbook_id}: {identifier}")
        diagnostic_ids.add(identifier)
        capability = diagnostic["capability"]
        if capability not in catalog["diagnostics"]:
            raise RunbookError(f"unknown diagnostic capability in {runbook_id}: {capability}")

    actions: dict[str, dict[str, Any]] = {}
    for action in spec["allowed_actions"]:
        identifier = action["id"]
        if identifier in actions:
            raise RunbookError(f"duplicate action id in {runbook_id}: {identifier}")
        actions[identifier] = action
        capability = action["capability"]
        policy = catalog["actions"].get(capability)
        if policy is None:
            raise RunbookError(f"unknown action capability in {runbook_id}: {capability}")
        if action["effect"] != policy["effect"]:
            raise RunbookError(f"action effect differs from catalog: {runbook_id}.{identifier}")
        if APPROVAL_ORDER[action["approval"]] < APPROVAL_ORDER[policy["minimum_approval"]]:
            raise RunbookError(f"action approval is weaker than catalog: {runbook_id}.{identifier}")

    for rollback in spec["rollback"]:
        if rollback["action_ref"] not in actions:
            raise RunbookError(
                f"rollback references unknown action: {runbook_id}.{rollback['action_ref']}"
            )


def _github_anchor(title: str) -> str:
    lowered = title.strip().lower()
    lowered = re.sub(r"[^\w\- ]", "", lowered, flags=re.UNICODE)
    return re.sub(r" +", "-", lowered)


def _markdown_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            continue
        anchor = _github_anchor(line.lstrip("#").strip())
        count = counts.get(anchor, 0)
        counts[anchor] = count + 1
        anchors.add(anchor if count == 0 else f"{anchor}-{count}")
    return anchors


def _alert_records(repo_root: Path) -> dict[str, AlertRecord]:
    records: dict[str, AlertRecord] = {}
    paths: list[Path] = []
    for relative_dir in ALERT_RULE_DIRS:
        directory = repo_root / relative_dir
        if not directory.exists():
            continue
        paths.extend(sorted(directory.glob("*.yml")))
    paths.extend(
        repo_root / relative
        for relative in GENERATED_ALERT_RULE_FILES
        if (repo_root / relative).is_file()
    )
    for path in paths:
        payload = _load_yaml(path)
        for group in payload.get("groups", []):
            for rule in group.get("rules", []):
                name = rule.get("alert")
                if not name:
                    continue
                if name in records:
                    raise RunbookError(f"duplicate alert id: {name}")
                link = rule.get("annotations", {}).get("runbook")
                records[name] = AlertRecord(
                    name=name,
                    source=path.relative_to(repo_root).as_posix(),
                    runbook_link=link,
                )
    return records


def _validate_alert_links(repo_root: Path, alerts: dict[str, AlertRecord]) -> None:
    for alert in alerts.values():
        if not alert.runbook_link:
            continue
        if alert.runbook_link == "/runbooks/{{ $labels.runbook_id }}":
            continue
        if alert.runbook_link.startswith("/runbooks/"):
            runbook_id = alert.runbook_link.removeprefix("/runbooks/")
            target = repo_root / CANONICAL_DIR / f"{runbook_id}.yaml"
            if not target.is_file():
                raise RunbookError(
                    f"alert {alert.name} has missing canonical runbook: {runbook_id}"
                )
            continue
        path_text, separator, anchor = alert.runbook_link.partition("#")
        target = repo_root / path_text
        if not target.is_file():
            raise RunbookError(f"alert {alert.name} has missing runbook path: {path_text}")
        if separator and anchor not in _markdown_anchors(target):
            raise RunbookError(f"alert {alert.name} has missing runbook anchor: {anchor}")


def _narrative_map(items: list[dict[str, str]]) -> dict[str, str]:
    return {item["id"]: item["text"] for item in items}


def _render_ru(runbook: dict[str, Any], locale: dict[str, Any]) -> str:
    metadata = runbook["metadata"]
    spec = runbook["spec"]
    tr = locale["translations"]
    lines = [
        f"# {tr['title']}",
        "",
        f"> Инструкция: `{metadata['id']}` · ревизия `{metadata['revision']}` · "
        f"критичность `{spec['severity']}`",
        "",
        tr["summary"],
        "",
        "## Строгие предупреждения",
        "",
    ]
    lines.extend(f"- {tr['warnings'][item['id']]}" for item in spec["safety"]["warnings"])
    lines.extend(["", "## Условия обязательной остановки", ""])
    lines.extend(
        f"- {tr['stop_conditions'][item['id']]}" for item in spec["safety"]["stop_conditions"]
    )
    lines.extend(["", "## Симптомы", ""])
    lines.extend(f"- {tr['symptoms'][item['id']]}" for item in spec["symptoms"])
    lines.extend(["", "## Предварительные условия", ""])
    lines.extend(
        f"- {tr['prerequisites'][item['id']]}" for item in spec["prerequisites"]
    )
    lines.extend(["", "## Диагностика только разрешёнными возможностями", ""])
    for item in spec["diagnostics"]:
        translated = tr["diagnostics"][item["id"]]
        lines.extend(
            [
                f"### `{item['id']}` · `{item['capability']}`",
                "",
                translated["instruction"],
                "",
                f"Ожидаемое доказательство: {translated['expected_evidence']}",
                "",
            ]
        )
    lines.extend(["## Разрешённые действия", ""])
    for item in spec["allowed_actions"]:
        lines.extend(
            [
                f"### `{item['id']}` · `{item['capability']}`",
                "",
                tr["allowed_actions"][item["id"]],
                "",
                f"Разрешение: `{item['approval']}`. Эффект: `{item['effect']}`.",
                "",
            ]
        )
    lines.extend(["## Откат", ""])
    for item in spec["rollback"]:
        translated = tr["rollback"][item["id"]]
        lines.extend(
            [
                f"- Через действие `{item['action_ref']}`: {translated['instruction']}",
                f"  Доказательство: {translated['expected_evidence']}",
            ]
        )
    lines.extend(["", "## Необходимые доказательства", ""])
    lines.extend(
        f"- {tr['evidence_collect'][item['id']]}" for item in spec["evidence"]["collect"]
    )
    lines.extend(["", "Успех:", ""])
    lines.extend(
        f"- {tr['success_conditions'][item['id']]}"
        for item in spec["evidence"]["success_conditions"]
    )
    lines.extend(["", "Неуспех:", ""])
    lines.extend(
        f"- {tr['failure_conditions'][item['id']]}"
        for item in spec["evidence"]["failure_conditions"]
    )
    lines.extend(["", "## Удаление чувствительных данных", ""])
    lines.append(
        "Запрещённые ключи доказательств: "
        + ", ".join(f"`{item}`" for item in spec["secret_redaction"]["forbidden_evidence_keys"])
        + "."
    )
    lines.append("")
    lines.extend(
        f"- {tr['redaction_rules'][item['id']]}"
        for item in spec["secret_redaction"]["rules"]
    )
    lines.extend(["", "## Владение и связи", ""])
    lines.append(
        f"Команда-владелец: `{spec['owner']['team']}`; роль эскалации: "
        f"`{spec['owner']['escalation_role']}`."
    )
    lines.append("")
    if spec["related_alerts"]:
        alert_list = ", ".join(f"`{item}`" for item in spec["related_alerts"])
        lines.append(f"Связанные alerts: {alert_list}.")
    else:
        lines.append("Связанные alerts отсутствуют; это явно зафиксированный пробел мониторинга.")
    if spec["monitoring_gaps"]:
        lines.append("")
        lines.extend(
            f"- Пробел мониторинга: {tr['monitoring_gaps'][item['id']]}"
            for item in spec["monitoring_gaps"]
        )
    lines.extend(
        [
            "",
            f"Исходная инструкция: `{metadata['source_legacy']}`.",
            "",
            "Этот документ сгенерирован из `ops.roehub.io/v1`; ручные изменения будут отклонены.",
            "",
        ]
    )
    return "\n".join(lines)


def _index_payload(
    runbooks: list[dict[str, Any]],
    locales: dict[str, dict[str, Any]],
    legacy_unmigrated: list[str],
) -> dict[str, Any]:
    problem_index: dict[str, str] = {}
    alert_index: dict[str, list[str]] = {}
    records = []
    for runbook in runbooks:
        runbook_id = runbook["metadata"]["id"]
        spec = runbook["spec"]
        locale = locales[runbook_id]
        problems = []
        for symptom in spec["symptoms"]:
            problem_id = f"{runbook_id}/{symptom['id']}"
            problem_index[problem_id] = runbook_id
            problems.append(problem_id)
        for alert in spec["related_alerts"]:
            alert_index.setdefault(alert, []).append(runbook_id)
        records.append(
            {
                "allowed_action_capabilities": sorted(
                    {item["capability"] for item in spec["allowed_actions"]}
                ),
                "canonical_path": f"docs/runbooks/ops/{runbook_id}.yaml",
                "component_ids": spec["component_ids"],
                "generated_ru_path": f"docs/runbooks/generated/ru/{runbook_id}.md",
                "id": runbook_id,
                "monitoring_gap_ids": [item["id"] for item in spec["monitoring_gaps"]],
                "problems": problems,
                "related_alerts": spec["related_alerts"],
                "revision": runbook["metadata"]["revision"],
                "severity": spec["severity"],
                "source_legacy": runbook["metadata"]["source_legacy"],
                "title": spec["title"],
                "title_ru": locale["translations"]["title"],
            }
        )
    return {
        "alert_index": {key: sorted(value) for key, value in sorted(alert_index.items())},
        "apiVersion": "ops.roehub.io/v1",
        "kind": "RunbookIndex",
        "legacy_unmigrated": [
            {"path": path, "target_stages": ["17", "20"]} for path in legacy_unmigrated
        ],
        "problem_index": dict(sorted(problem_index.items())),
        "runbooks": records,
    }


def expected_outputs(repo_root: Path) -> dict[Path, bytes]:
    runbook_schema = _load_json(repo_root / RUNBOOK_SCHEMA)
    locale_schema = _load_json(repo_root / LOCALE_SCHEMA)
    catalog = _load_json(repo_root / CAPABILITIES)
    jsonschema.Draft202012Validator.check_schema(runbook_schema)
    jsonschema.Draft202012Validator.check_schema(locale_schema)
    if catalog.get("apiVersion") != "ops.roehub.io/v1" or catalog.get("kind") != (
        "ActionCapabilityCatalog"
    ):
        raise RunbookError("invalid action capability catalog identity")
    alerts = _alert_records(repo_root)
    _validate_alert_links(repo_root, alerts)

    runbook_paths = sorted((repo_root / CANONICAL_DIR).glob("*.yaml"))
    if not runbook_paths:
        raise RunbookError("no canonical runbooks found")
    runbooks: list[dict[str, Any]] = []
    locales: dict[str, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    migrated_sources: set[str] = set()
    for path in runbook_paths:
        runbook = _load_yaml(path)
        errors = _schema_errors(runbook, runbook_schema)
        if errors:
            raise RunbookError(f"schema errors in {path}: {'; '.join(errors)}")
        runbook_id = runbook["metadata"]["id"]
        if path.stem != runbook_id:
            raise RunbookError(f"runbook filename/id mismatch: {path}")
        if runbook_id in seen_ids:
            raise RunbookError(f"duplicate runbook id: {runbook_id}")
        seen_ids.add(runbook_id)
        source_legacy = runbook["metadata"]["source_legacy"]
        if not (repo_root / source_legacy).is_file():
            raise RunbookError(f"missing legacy source for {runbook_id}: {source_legacy}")
        migrated_sources.add(source_legacy)
        _validate_no_secret_or_shell_fields(runbook, path)
        _validate_canonical_language(runbook, path)
        _validate_capabilities(runbook, catalog)
        for alert in runbook["spec"]["related_alerts"]:
            if alert not in alerts:
                raise RunbookError(f"unknown alert reference in {runbook_id}: {alert}")
        if not runbook["spec"]["related_alerts"] and not runbook["spec"]["monitoring_gaps"]:
            raise RunbookError(f"runbook lacks alerts and an explicit monitoring gap: {runbook_id}")

        locale_path = repo_root / RU_LOCALE_DIR / f"{runbook_id}.yaml"
        if not locale_path.is_file():
            raise RunbookError(f"missing Russian locale: {locale_path}")
        locale = _load_yaml(locale_path)
        locale_errors = _schema_errors(locale, locale_schema)
        if locale_errors:
            raise RunbookError(f"schema errors in {locale_path}: {'; '.join(locale_errors)}")
        _validate_no_secret_or_shell_fields(locale, locale_path)
        _validate_locale_coverage(runbook, locale)
        locales[runbook_id] = locale
        runbooks.append(runbook)

    stale_locales = sorted(
        path.stem
        for path in (repo_root / RU_LOCALE_DIR).glob("*.yaml")
        if path.stem not in seen_ids
    )
    if stale_locales:
        raise RunbookError(f"Russian locales without canonical runbooks: {stale_locales}")

    all_legacy = sorted(
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "docs/runbooks").glob("*.md")
    )
    legacy_unmigrated = [path for path in all_legacy if path not in migrated_sources]
    outputs = {
        GENERATED_RU_DIR / f"{runbook['metadata']['id']}.md": _render_ru(
            runbook, locales[runbook["metadata"]["id"]]
        ).encode()
        for runbook in runbooks
    }
    outputs[INDEX_PATH] = _json_bytes(_index_payload(runbooks, locales, legacy_unmigrated))
    return outputs


def run_generator(repo_root: Path, *, check: bool) -> int:
    try:
        outputs = expected_outputs(repo_root)
    except (
        json.JSONDecodeError,
        jsonschema.SchemaError,
        OSError,
        RunbookError,
        yaml.YAMLError,
    ) as error:
        print(f"Runbook generation failed: {error}", file=sys.stderr)
        return 2
    expected_generated = {
        path for path in outputs if path.parent == GENERATED_RU_DIR
    }
    current_generated = {
        path.relative_to(repo_root)
        for path in (repo_root / GENERATED_RU_DIR).glob("*.md")
    }
    drift = sorted(current_generated - expected_generated)
    for relative, content in outputs.items():
        target = repo_root / relative
        current = target.read_bytes() if target.exists() else None
        if current != content:
            drift.append(relative)
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    if not check:
        for relative in current_generated - expected_generated:
            (repo_root / relative).unlink()
    if check and drift:
        print("Runbook artifacts are out-of-date: " + ", ".join(str(item) for item in drift))
        return 1
    action = "up-to-date" if check else "generated"
    print(f"OK: runbooks {action} ({len(outputs) - 1} Russian docs + JSON index)")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and validate ops.roehub.io/v1 runbooks")
    parser.add_argument("--check", action="store_true", help="fail on generated artifact drift")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_generator(args.repo_root.resolve(), check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
