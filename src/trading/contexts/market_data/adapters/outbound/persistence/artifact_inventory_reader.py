from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_fs import (
    BacktestArtifactPathBuilderV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.artifact_manifest_loader import (
    YamlBacktestArtifactLoaderV2,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    artifact_coordinates_from_market_id_v2,
)
from trading.shared_kernel.primitives import InstrumentId


@dataclass(frozen=True, slots=True)
class FileSystemActiveArtifactInventoryReader:
    """Report bytes in the validated active slot without traversing outside its root."""

    artifact_root: Path

    def __post_init__(self) -> None:
        if not self.artifact_root.is_absolute():
            raise ValueError("artifact_root must be absolute")

    def active_slot_bytes(self, *, instrument_id: InstrumentId) -> int:
        coordinates = artifact_coordinates_from_market_id_v2(
            market_id=instrument_id.market_id.value,
            symbol=str(instrument_id.symbol),
        )
        resolver = BacktestArtifactPathBuilderV2(root=self.artifact_root)
        loader = YamlBacktestArtifactLoaderV2(path_resolver=resolver)
        try:
            current = loader.load_current_pointer(coordinates)
        except FileNotFoundError:
            return 0
        slot_root = resolver.slot_root(coordinates, current.active_slot)
        safe_root = self.artifact_root.resolve(strict=False)
        resolved_slot = slot_root.resolve(strict=False)
        if safe_root != resolved_slot and safe_root not in resolved_slot.parents:
            raise ValueError("artifact slot escapes configured artifact_root")
        if not resolved_slot.is_dir():
            return 0
        total_bytes = 0
        for path in resolved_slot.rglob("*"):
            if path.is_symlink():
                raise ValueError("artifact slot must not contain symlinks")
            if path.is_file():
                total_bytes += path.stat().st_size
        return total_bytes
