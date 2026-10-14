"""Source-only S05 caller inventory; not a runtime reachability proof."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).parent
roots = ("apps", "src/trading", "scripts")
rows = []
for root in roots:
    for path in sorted((ROOT / root).rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [n.name for n in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for name in names:
                if "clickhouse" not in name.lower():
                    continue
                if (
                    "/persistence/clickhouse/" in relative
                    or relative == "apps/cli/wiring/db/clickhouse.py"
                ):
                    disposition = "S07: disconnected legacy adapter/module removal"
                elif relative.startswith(("apps/migrations/", "apps/monitoring/")):
                    disposition = "S06/S07: operational migration/probe/exporter replacement"
                else:
                    disposition = "UNEXPECTED_BUSINESS_CALLER"
                rows.append(
                    dict(path=relative, line=node.lineno, imported=name, disposition=disposition)
                )
result = dict(
    scope=list(roots),
    method="Python AST Import/ImportFrom inventory",
    limits=(
        "Dynamic imports/HTTP and generated installation config require S07 residual audit; "
        "source-only, no package-absence claim."
    ),
    rows=rows,
    unexpected=sum(r["disposition"] == "UNEXPECTED_BUSINESS_CALLER" for r in rows),
)
(OUT / "caller-audit.json").write_text(json.dumps(result, indent=2) + "\n")
assert not result["unexpected"], result
print("Caller imports classified:", len(rows), "unexpected business callers:", result["unexpected"])
