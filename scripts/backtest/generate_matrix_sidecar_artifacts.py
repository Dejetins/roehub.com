from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.backtest.adapters.outbound.artifacts_fs import (  # noqa: E402
    BacktestArtifactPathBuilderV2,
)
from trading.contexts.backtest.application.services.v2.matrix_backend.bitsets import (  # noqa: E402,E501
    build_matrix_sidecar_artifacts,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (  # noqa: E402,E501
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (  # noqa: E402,E501
    ArtifactCoordinatesV2,
)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    indicators = tuple(dict.fromkeys(args.indicator))
    if not indicators:
        raise ValueError("at least one --indicator is required")

    builder = BacktestArtifactPathBuilderV2(root=args.artifact_root)
    loader = YamlBacktestArtifactLoaderV2(path_resolver=builder)
    coordinates = ArtifactCoordinatesV2(
        exchange=args.exchange,
        market_type=args.market_type,
        symbol=args.symbol,
    )
    current = loader.load_current_pointer(coordinates)
    slot = args.slot or current.active_slot
    slot_manifest_path = builder.slot_manifest_path(coordinates, slot)
    slot_manifest = loader.load_manifest_from_path(slot_manifest_path, slot=slot)

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    items: list[dict[str, Any]] = []
    for indicator_id in indicators:
        signal_paths = builder.signal_paths(
            coordinates,
            slot,
            args.timeframe,
            indicator_id,
        )
        signal_manifest = loader.load_signal_manifest(
            coordinates,
            slot,
            args.timeframe,
            indicator_id,
        )
        signal_matrix = np.load(signal_paths.signals, mmap_mode="r")
        item_started = time.perf_counter()
        sidecar = build_matrix_sidecar_artifacts(
            signal_matrix=signal_matrix,
            source_manifest_path=signal_paths.manifest,
            source_signals_path=signal_paths.signals,
            output_dir=output_root / _safe_indicator_path(indicator_id),
            identity={
                "exchange": coordinates.exchange,
                "market_type": coordinates.market_type,
                "symbol": coordinates.symbol,
                "slot": slot,
                "slot_generation": slot_manifest.slot_generation,
                "artifact_asof_date": slot_manifest.asof_date,
                "root_manifest_sha256": current.manifest_sha256,
                "timeframe": signal_manifest.timeframe,
                "indicator_id": signal_manifest.indicator_id,
            },
        )
        items.append(
            {
                "indicator_id": indicator_id,
                "rows": int(sidecar.manifest["source_signal_shape"][0]),
                "signal_length": int(sidecar.manifest["source_signal_shape"][1]),
                "word_count": int(sidecar.manifest["word_count"]),
                "sidecar_generate_ms": (time.perf_counter() - item_started) * 1000.0,
                "sidecar_dir": str(output_root / _safe_indicator_path(indicator_id)),
                "source_signals_sha256": sidecar.manifest["source_signals_sha256"],
                "source_manifest_sha256": sidecar.manifest["source_manifest_sha256"],
                "unique_signal_row_count": int(sidecar.unique_signal_row_ids.shape[0]),
                "duplicate_signal_row_count": int(sidecar.duplicate_signal_row_ids.shape[0]),
            }
        )

    report = {
        "schema": "backtest_matrix_sidecar_generation_report_v1",
        "artifact_root": str(args.artifact_root),
        "output_dir": str(output_root),
        "identity": {
            "exchange": coordinates.exchange,
            "market_type": coordinates.market_type,
            "symbol": coordinates.symbol,
            "slot": slot,
            "slot_generation": slot_manifest.slot_generation,
            "artifact_asof_date": slot_manifest.asof_date,
            "root_manifest_sha256": current.manifest_sha256,
            "timeframe": args.timeframe,
        },
        "sidecar_generate_ms": (time.perf_counter() - started) * 1000.0,
        "indicators": items,
    }
    report_path = output_root / "sidecar_generation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {report_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate benchmark/test matrix sidecar bitset artifacts."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--market-type", default="spot")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--slot", default=None)
    parser.add_argument(
        "--indicator",
        action="append",
        default=[],
        help="Indicator id to generate; repeat for each benchmark indicator.",
    )
    return parser


def _safe_indicator_path(indicator_id: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in indicator_id)
    return safe or "indicator"


if __name__ == "__main__":
    raise SystemExit(main())
