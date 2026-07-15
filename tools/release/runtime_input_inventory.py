#!/usr/bin/env python3
"""Generate a value-free inventory of legacy runtime environment consumers."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "configs" / "installation" / "runtime-input-inventory.json"
SCAN_ROOTS = (Path("apps"), Path("src"), Path("infra"))
TEXT_SUFFIXES = {".bash", ".env", ".example", ".sh", ".yaml", ".yml", ".zsh"}
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,127}$")
INTERPOLATION_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]{1,127})(?::[-?][^}]*)?\}")
ASSIGNMENT_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]{1,127})\s*=")
DOCKER_ENV_RE = re.compile(r"^\s*ENV\s+([A-Z][A-Z0-9_]{1,127})(?:\s+|=)")
CONFIG_PATH_RE = re.compile(r"configs/[A-Za-z0-9_./-]+\.(?:csv|json|ya?ml)")


class RuntimeInputInventoryError(RuntimeError):
    """Raised when the inventory cannot be generated deterministically."""


def _git_visible_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    files: list[Path] = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode())
        if any(relative.is_relative_to(root) for root in SCAN_ROOTS):
            files.append(relative)
    return sorted(path for path in files if (ROOT / path).is_file())


def _string_constant(node: ast.AST, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


class _PythonEnvironmentVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.constants: dict[str, str] = {}
        self.names: set[str] = set()

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        value = _string_constant(node.value, self.constants)
        if value is not None:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    self.constants[target.id] = value
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if isinstance(node.target, ast.Name) and node.target.id.isupper() and node.value:
            value = _string_constant(node.value, self.constants)
            if value is not None:
                self.constants[node.target.id] = value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if node.args and self._is_environment_read(node.func):
            name = _string_constant(node.args[0], self.constants)
            if name and ENV_NAME_RE.fullmatch(name):
                self.names.add(name)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        if self._is_environ_object(node.value):
            name = _string_constant(node.slice, self.constants)
            if name and ENV_NAME_RE.fullmatch(name):
                self.names.add(name)
        self.generic_visit(node)

    @staticmethod
    def _attribute_chain(node: ast.AST) -> tuple[str, ...]:
        values: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            values.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            values.append(current.id)
        return tuple(reversed(values))

    def _is_environ_object(self, node: ast.AST) -> bool:
        chain = self._attribute_chain(node)
        return bool(chain and chain[-1] in {"environ", "environment"})

    def _is_environment_read(self, node: ast.AST) -> bool:
        chain = self._attribute_chain(node)
        if not chain:
            return False
        if chain[-1] == "getenv":
            return True
        return (
            len(chain) >= 2
            and chain[-1] in {"get", "pop", "setdefault"}
            and chain[-2] in {"environ", "environment"}
        )


def _python_env_names(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise RuntimeInputInventoryError(f"cannot parse Python input {path}: {error}") from error
    visitor = _PythonEnvironmentVisitor()
    visitor.visit(tree)
    return visitor.names


def _text_env_names(text: str) -> set[str]:
    names = set(INTERPOLATION_RE.findall(text))
    for line in text.splitlines():
        assignment = ASSIGNMENT_RE.match(line)
        if assignment:
            names.add(assignment.group(1))
        docker_env = DOCKER_ENV_RE.match(line)
        if docker_env:
            names.add(docker_env.group(1))
    return names


def _classification(name: str) -> tuple[str, str]:
    if re.search(
        r"(?:PASSWORD|PASS_PHRASE|TOKEN|SECRET|API_KEY|PRIVATE_KEY|CREDENTIAL|DSN)",
        name,
    ):
        return "openbao_secret_reference", "08"
    if re.search(
        r"(?:^|_)(?:HOST|PORT|URL|ADDR|ADDRESS|DOMAIN|PATH|DIR|ROOT|TLS|SSL|"
        r"VERIFY|SECURE|DATABASE|DB|PROFILE|ENV)(?:$|_)",
        name,
    ):
        return "installation_generated_runtime", "03,17"
    if name.startswith(
        (
            "BACKTEST_",
            "EXCHANGE_",
            "IDENTITY_",
            "MARKET_DATA_",
            "NOTIFICATION_",
            "RL_",
            "ROEHUB_BACKTEST_",
            "ROEHUB_RL_",
            "STRATEGY_",
            "TELEGRAM_",
        )
    ):
        return "product_config_postgresql", "04-16"
    if name.startswith(("CUDA_", "LOG_", "NUMBA_", "OTEL_", "PYTHON", "UVICORN_")):
        return "internal_runtime_tuning", "17"
    return "explicit_legacy_runtime_handoff", "17"


def inventory_payload() -> dict[str, Any]:
    environment_sources: dict[str, set[str]] = defaultdict(set)
    file_input_sources: dict[str, set[str]] = defaultdict(set)
    for relative in _git_visible_files():
        path = ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise RuntimeInputInventoryError(
                f"cannot read runtime input {path}: {error}"
            ) from error
        if path.suffix == ".py":
            names = _python_env_names(path)
        elif path.suffix in TEXT_SUFFIXES or path.name.endswith(".env.example"):
            names = _text_env_names(text)
        else:
            names = set()
        for name in names:
            environment_sources[name].add(relative.as_posix())
        for config_path in CONFIG_PATH_RE.findall(text):
            file_input_sources[config_path].add(relative.as_posix())

    entries = []
    counts: dict[str, int] = defaultdict(int)
    for name in sorted(environment_sources):
        classification, owner_stage = _classification(name)
        counts[classification] += 1
        entries.append(
            {
                "classification": classification,
                "key": name,
                "owner_stage": owner_stage,
                "sources": sorted(environment_sources[name]),
            }
        )
    file_inputs = [
        {"path": path, "sources": sorted(file_input_sources[path])}
        for path in sorted(file_input_sources)
    ]
    return {
        "contract": {
            "legacy_env_is_user_contract": False,
            "raw_values_collected": False,
            "secret_values_allowed": False,
            "user_installation_input": "configs/installation/roehub.yaml",
        },
        "counts": {name: counts[name] for name in sorted(counts)},
        "entries": entries,
        "file_input_total": len(file_inputs),
        "file_inputs": file_inputs,
        "scan_roots": [path.as_posix() for path in SCAN_ROOTS],
        "schema": "io.roehub.runtime-input-inventory/v1alpha1",
        "total": len(entries),
    }


def inventory_bytes() -> bytes:
    return (json.dumps(inventory_payload(), indent=2, sort_keys=True) + "\n").encode()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        expected = inventory_bytes()
        if args.write:
            if not OUTPUT_PATH.exists() or OUTPUT_PATH.read_bytes() != expected:
                OUTPUT_PATH.write_bytes(expected)
        elif not OUTPUT_PATH.exists() or OUTPUT_PATH.read_bytes() != expected:
            raise RuntimeInputInventoryError(
                f"runtime input inventory is missing or stale: {OUTPUT_PATH.relative_to(ROOT)}"
            )
    except (OSError, RuntimeInputInventoryError, subprocess.CalledProcessError) as error:
        print(f"runtime input inventory failed: {error}", file=sys.stderr)
        return 1
    print(
        "runtime input inventory passed: "
        f"mode={'write' if args.write else 'check'}, total={inventory_payload()['total']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
