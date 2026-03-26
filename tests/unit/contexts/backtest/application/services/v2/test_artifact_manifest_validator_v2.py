from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from tests.unit.contexts.backtest.application.services.v2.artifact_testkit_v2 import (
    build_synthetic_artifact_store_v2,
)
from trading.contexts.backtest.application.services import BacktestArtifactManifestValidatorV2


def test_backtest_artifact_manifest_validator_v2_accepts_valid_strict_slot(
    tmp_path: Path,
) -> None:
    """
    Verify strict validator accepts a fully valid inactive slot and exposes typed metadata.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Valid slot manifests and arrays should produce no diagnostics before publish switch.
    Raises:
        AssertionError: If a valid strict slot is rejected.
    Side Effects:
        Creates and reads a synthetic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert result.slot_manifest is not None
    assert result.slot_manifest.slot_generation == 5
    assert result.manifest_sha256 is not None
    assert len(result.signal_manifests) == 1
    assert result.signal_manifests[0].indicator_id == "ma.ema"
    assert result.hit_times_manifest is not None
    assert result.hit_times_manifest.timeline_bar_count == 4
    assert result.diagnostics == ()


def test_backtest_artifact_manifest_validator_v2_rejects_root_manifest_schema_drift(
    tmp_path: Path,
) -> None:
    """
    Verify strict validator rejects root manifests with unsupported extra keys.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Root manifest schema drift must fail before any deeper array validation.
    Raises:
        AssertionError: If unsupported extra keys are accepted.
    Side Effects:
        Creates and mutates a synthetic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    root_manifest_path = store.builder.slot_manifest_path(store.coordinates, store.inactive_slot)
    payload = yaml.safe_load(root_manifest_path.read_text(encoding="utf-8"))
    payload["unexpected"] = "drift"
    root_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)
    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert result.slot_manifest is None
    assert result.manifest_sha256 is None
    assert len(result.diagnostics) == 1
    assert result.diagnostics[0].code == "root_manifest_invalid"
    assert "unexpected keys" in result.diagnostics[0].message


def test_backtest_artifact_manifest_validator_v2_rejects_mapping_price_correspondence_mismatch(
    tmp_path: Path,
) -> None:
    """
    Verify validator reports a stable diagnostic when mapping indexes no longer match price
    timelines.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Bounds and monotonicity may still be valid while exact `prices/<tf>` correspondence fails.
    Raises:
        AssertionError: If correspondence drift is not reported with the documented error code.
    Side Effects:
        Creates and reads a synthetic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        inactive_mapping_open_idx=np.array([1, 2], dtype=np.uint32),
    )
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert tuple(diagnostic.code for diagnostic in result.diagnostics) == (
        "mapping_open_time_correspondence_mismatch",
    )


def test_backtest_artifact_manifest_validator_v2_orders_multiple_diagnostics_deterministically(
    tmp_path: Path,
) -> None:
    """
    Verify validator emits stable diagnostics ordering across multiple simultaneous violations.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Diagnostics ordering must stay deterministic by artifact family and validation stage.
    Raises:
        AssertionError: If diagnostics order is unstable or misses expected violations.
    Side Effects:
        Creates and reads synthetic invalid artifact trees under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
    """
    store = build_synthetic_artifact_store_v2(
        tmp_path=tmp_path,
        inactive_signal_values=np.array([[-1, 2], [1, 0]], dtype=np.int8),
        inactive_mapping_close_idx=np.array([1, 4], dtype=np.uint32),
        inactive_long_tp=np.array([[1, 3, 4, 4], [1, 2, 4, 4]], dtype=np.uint32),
    )
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    first_result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )
    second_result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    first_codes = tuple(diagnostic.code for diagnostic in first_result.diagnostics)
    second_codes = tuple(diagnostic.code for diagnostic in second_result.diagnostics)

    assert first_codes == second_codes
    assert first_codes == (
        "mapping_close_indexes_out_of_bounds",
        "signal_values_out_of_set",
        "hit_times_table_not_monotone",
    )


def test_backtest_artifact_manifest_validator_v2_rejects_signal_manifest_reference_hash_drift(
    tmp_path: Path,
) -> None:
    """
    Verify validator reports root-catalog drift when one signal manifest hash no longer matches.

    Args:
        tmp_path: pytest temporary path fixture.
    Returns:
        None.
    Assumptions:
        Signal manifest file remains valid while only the root reference hash is corrupted.
    Raises:
        AssertionError: If hash drift is not reported with the documented diagnostic code.
    Side Effects:
        Creates and mutates a synthetic artifact tree under `tmp_path`.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/backtest/backtest-precompute-runner-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_validator.py
      - tests/unit/contexts/backtest/application/services/v2/artifact_testkit_v2.py
    """
    store = build_synthetic_artifact_store_v2(tmp_path=tmp_path)
    root_manifest_path = store.builder.slot_manifest_path(store.coordinates, store.inactive_slot)
    payload = yaml.safe_load(root_manifest_path.read_text(encoding="utf-8"))
    payload["signals"]["manifests"][0]["manifest_sha256"] = "0" * 64
    root_manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    validator = BacktestArtifactManifestValidatorV2(artifact_loader=store.loader)

    result = validator.validate_slot(
        coordinates=store.coordinates,
        slot=store.inactive_slot,
        validation_spec=store.validation_spec,
        expected_asof_date="2026-03-26",
        expected_slot_generation=5,
    )

    assert tuple(diagnostic.code for diagnostic in result.diagnostics) == (
        "signal_manifest_reference_hash_mismatch",
    )
