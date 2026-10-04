"""Portable source/registration checks, not a test of agent adherence."""
from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from tools.ci.route_changes import ALL_SHARDS, classify_ci

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "hook_config", ROOT / ".codex/hooks/validate_config.py"
)
assert _spec and _spec.loader
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)


class AgentPolicyTests(unittest.TestCase):
    def test_registration(self):
        _config.validate_config(ROOT / ".codex/hooks.json")

    def test_malformed_registration_fails(self):
        valid = json.loads((ROOT / ".codex/hooks.json").read_text())
        cases = ["{", "[]"]
        for key, value in (("command", "missing.py"), ("type", "prompt"), ("timeout", 0)):
            broken = copy.deepcopy(valid)
            broken["hooks"]["PreToolUse"][0]["hooks"][0][key] = value
            cases.append(json.dumps(broken))
        broken = copy.deepcopy(valid)
        broken["hooks"]["PreToolUse"][0]["matcher"] = "exec_command"
        cases.append(json.dumps(broken))
        broken = copy.deepcopy(valid)
        broken["hooks"]["Stop"] = []
        cases.append(json.dumps(broken))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hooks.json"
            (Path(tmp) / "hooks").mkdir()
            (Path(tmp) / "hooks/roehub_hook_router.py").touch()
            for text in cases:
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(ValueError):
                        _config.validate_config(path)
            path.write_text(json.dumps(valid))
            (Path(tmp) / "hooks/roehub_hook_router.py").unlink()
            with self.assertRaisesRegex(ValueError, "Missing hook router"):
                _config.validate_config(path)

    def test_policy_only_paths_skip_backend(self):
        for path in ("AGENTS.md", ".codex/AGENTS.md", ".codex/hooks.json",
                     ".codex/config.toml", ".codex/hooks/README.md",
                     ".codex/hooks/roehub_hook_router.py", ".codex/hooks/tests/run_tests.py",
                     "tools/ci/test_agent_policy.py"):
            with self.subTest(path=path):
                result = classify_ci([path])
                self.assertEqual(result["agent_policy"], "true")
                self.assertEqual(result["code"], "false")
                self.assertEqual(result["has_tests"], "false")
                self.assertEqual(result["run_migrations"], "false")

    def test_policy_dependencies_keep_existing_gates(self):
        for path in ("tools/ci/route_changes.py", "tests/unit/tools/test_ci_route_changes.py",
                     ".github/workflows/ci.yml", "pyproject.toml", "uv.lock", ".python-version"):
            with self.subTest(path=path):
                result = classify_ci([path])
                self.assertEqual(result["agent_policy"], "true")
                self.assertEqual(result["code"], "true")
                self.assertEqual(result["has_tests"], "true")
        all_changes = classify_ci([], all_changes=True)
        self.assertEqual(all_changes["agent_policy"], "true")
        names = {v["name"] for v in json.loads(all_changes["test_matrix"])["include"]}
        self.assertEqual(names, set(ALL_SHARDS))

    def test_mixed_changes_preserve_backend_route(self):
        path = "src/trading/contexts/backtest/domain/example.py"
        before = classify_ci([path])
        after = classify_ci([path, ".codex/hooks.json"])
        self.assertEqual(after, {**before, "agent_policy": "true"})
        self.assertEqual(classify_ci(["apps/web/dist/js/example.js"])["agent_policy"], "false")

    def test_root_is_self_contained(self):
        root = (ROOT / "AGENTS.md").read_text()
        self.assertNotIn("/Users/", root)
        self.assertIn("authoritative repository entrypoint", root)
        for contract in ("apps/platform-web", "apps/web", "apps/navigator-web",
                         "shared snapshot", "focused self-review", "one independent review",
                         "publication authority", "shipped-no-runtime", "always Russian"):
            self.assertIn(contract, root.replace("\n  ", " ").replace("\n", " "))
        self.assertIn("stop mutations", root)
        self.assertIn("../AGENTS.md", (ROOT / ".codex/AGENTS.md").read_text())


if __name__ == "__main__":
    unittest.main()
