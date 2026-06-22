from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.rl_trading.domain.dataset_refresh_manifest import (  # noqa: E402
    Stage04BArtifactRef,
    build_dataset_refresh_manifest_v1,
    hash_dataset_refresh_payload_v1,
    render_dataset_refresh_manifest_json_v1,
)

DEFAULT_STAGE04B_ROOT = Path(
    "/opt/roehub/state/rl_trading/stage04b_binance_futures_full_current_repair"
)
DEFAULT_SOURCE_WINDOW_MANIFEST = (
    DEFAULT_STAGE04B_ROOT / "stage04b_full_current_repair_manifest_first_kline.json"
)
DEFAULT_COVERAGE_REPORT = (
    DEFAULT_STAGE04B_ROOT / "stage04b_full_current_coverage_report_first_kline.json"
)
DEFAULT_OUTPUT_JSON = (
    Path("/opt/roehub/state/rl_trading/stage04c_dataset_refresh_manifest")
    / "stage04c_dataset_refresh_manifest.json"
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_manifest = _read_json(args.source_window_manifest)
    coverage_report = _read_json(args.coverage_report)
    generated_at_utc = (
        _parse_utc(args.generated_at_utc) if args.generated_at_utc is not None else _now_utc()
    )
    manifest = build_dataset_refresh_manifest_v1(
        stage04b_source_window_manifest=source_manifest,
        stage04b_coverage_report=coverage_report,
        source_window_artifact=Stage04BArtifactRef(
            path=str(args.source_window_manifest),
            sha256=_file_sha256_hex(args.source_window_manifest),
        ),
        coverage_artifact=Stage04BArtifactRef(
            path=str(args.coverage_report),
            sha256=_file_sha256_hex(args.coverage_report),
        ),
        runtime_manifest_path=str(args.output_json),
        generated_at_utc=generated_at_utc,
    )
    _atomic_write_json(args.output_json, manifest)
    file_sha256 = _file_sha256_hex(args.output_json)
    print(
        json.dumps(
            {
                "dataset_refresh_manifest": str(args.output_json),
                "manifest_file_sha256": file_sha256,
                "manifest_payload_sha256": hash_dataset_refresh_payload_v1(manifest),
                "acceptance_status": manifest["acceptance_status"],
                "market": manifest["market"],
                "universe_symbols_count": manifest["universe"]["symbols_count"],
                "dataset_versions": [
                    {
                        "dataset_version": item["dataset_version"],
                        "status": item["status"],
                        "included_symbols_count": item["included_symbols_count"],
                        "excluded_symbols_count": item["excluded_symbols_count"],
                    }
                    for item in cast(list[Mapping[str, Any]], manifest["dataset_versions"])
                ],
                "stage05_input_manifest_path": manifest["stage05_handoff"][
                    "input_manifest_path"
                ],
                "stage05_input_manifest_sha256": file_sha256,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage 04C dataset refresh manifest.")
    parser.add_argument(
        "--source-window-manifest",
        type=Path,
        default=DEFAULT_SOURCE_WINDOW_MANIFEST,
    )
    parser.add_argument("--coverage-report", type=Path, default=DEFAULT_COVERAGE_REPORT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--generated-at-utc", type=str, default=None)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_dataset_refresh_manifest_json_v1(payload) + "\n", encoding="utf-8")
    tmp.replace(path)


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(UTC).replace(microsecond=0)


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


if __name__ == "__main__":
    raise SystemExit(main())
