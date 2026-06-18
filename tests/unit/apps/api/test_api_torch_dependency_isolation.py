from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_api_app_import_does_not_import_torch() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    code = r'''
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("ROEHUB_NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("STRATEGY_PG_DSN", "postgresql://user:pass@localhost:5432/roehub")

original_read_text = Path.read_text

def _safe_read_text(self, *args, **kwargs):
    if self == Path("/etc/roehub/roehub.env"):
        return ""
    return original_read_text(self, *args, **kwargs)

Path.read_text = _safe_read_text
importlib.import_module("apps.api.main.app")
print("torch" in sys.modules)
'''
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(repo_root))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "False"
