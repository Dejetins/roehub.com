#!/usr/bin/env python3
"""Load file-backed runtime inputs before replacing this process."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    redis_path = os.environ.get("ROEHUB_REDIS_PASSWORD_FILE", "").strip()
    if redis_path:
        try:
            redis_value = Path(redis_path).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise SystemExit("Redis password file is not readable") from error
        if not redis_value:
            raise SystemExit("Redis password file is empty")
        os.environ.setdefault("ROEHUB_REDIS_PASSWORD", redis_value)
        os.environ.setdefault("ROEHUB_STORAGE_REDIS_PASSWORD", redis_value)
    clickhouse_path = os.environ.get("ROEHUB_CLICKHOUSE_PASSWORD_FILE", "").strip()
    if clickhouse_path:
        try:
            clickhouse_value = Path(clickhouse_path).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise SystemExit("ClickHouse password file is not readable") from error
        if not clickhouse_value:
            raise SystemExit("ClickHouse password file is empty")
        os.environ.setdefault("CH_PASSWORD", clickhouse_value)
        os.environ.setdefault("ROEHUB_STORAGE_CLICKHOUSE_PASSWORD", clickhouse_value)
    if len(sys.argv) < 2:
        raise SystemExit("runtime command is required")
    os.execvpe(sys.argv[1], sys.argv[1:], os.environ)


if __name__ == "__main__":
    main()
