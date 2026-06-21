#!/usr/bin/env python3
"""Summarize Roehub hook observe logs without reading raw hook payloads."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _iter_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _finding_parts(finding: Any) -> tuple[str, str, str]:
    if isinstance(finding, dict):
        return (
            str(finding.get("severity") or "UNKNOWN"),
            str(finding.get("validator") or "unknown"),
            str(finding.get("target") or "-"),
        )
    if isinstance(finding, str):
        severity = finding.split(":", 1)[0] if ":" in finding else "UNKNOWN"
        validator = "legacy_line"
        return severity, validator, "-"
    return "UNKNOWN", "unknown", "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_path", help="Path from ROEHUB_HOOK_OBSERVE_LOG")
    args = parser.parse_args()

    path = Path(args.log_path).expanduser()
    records = _iter_records(path)
    event_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    validator_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()

    for record in records:
        event_counts[str(record.get("event") or "unknown")] += 1
        for finding in record.get("findings") or []:
            severity, validator, target = _finding_parts(finding)
            severity_counts[severity] += 1
            validator_counts[validator] += 1
            target_counts[target] += 1

    print(f"records: {len(records)}")
    print("events:")
    for key, count in event_counts.most_common():
        print(f"  {key}: {count}")
    print("severities:")
    for key, count in severity_counts.most_common():
        print(f"  {key}: {count}")
    print("validators:")
    for key, count in validator_counts.most_common():
        print(f"  {key}: {count}")
    print("targets:")
    for key, count in target_counts.most_common():
        print(f"  {key}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
