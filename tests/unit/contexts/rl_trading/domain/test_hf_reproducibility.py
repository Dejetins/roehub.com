from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import (
    FEATURE_NAMES_V1,
    HF_TRAIN_CARD_SESSION_COUNT_DELTA_V1,
    HfDatasetSplitSpec,
    HfReproducibilityError,
    HfReproducibilityRunConfig,
    compute_file_sha256,
    expected_hf_dataset_manifest_hash_v1,
    expected_hf_dataset_manifest_payload_v1,
    inspect_hf_split_file_v1,
    render_json_payload_v1,
    run_config_hash_v1,
    run_hf_reproducibility_smoke_v1,
    select_deterministic_sample_keys_v1,
)


def test_expected_hf_manifest_records_binance_futures_hashes_and_count_mismatch() -> None:
    payload = expected_hf_dataset_manifest_payload_v1()
    splits = {row["split_name"]: row for row in payload["splits"]}  # type: ignore[index]

    assert payload["dataset_repo"] == "ResearchRL/open-rl-trading-binance-dataset"
    assert payload["channel_order_observed"] == list(FEATURE_NAMES_V1)
    assert payload["dataset_format"]["source_market"] == "binance:futures"  # type: ignore[index]
    assert payload["attribution"]["external_repo"]["id"] == (  # type: ignore[index]
        "YuriyKolesnikov/rl-trading-binance"
    )
    assert payload["attribution"]["dataset"]["license"] == "MIT License"  # type: ignore[index]
    assert payload["train_count_mismatch"]["observed_minus_card"] == (  # type: ignore[index]
        HF_TRAIN_CARD_SESSION_COUNT_DELTA_V1
    )
    assert splits["train"]["expected_sha256"] == (
        "1c5cdf179777f0a68a81da915749f50d97826282e1419a5314a67b170e9cb14d"
    )
    assert len(expected_hf_dataset_manifest_hash_v1()) == 64


def test_sample_key_selection_is_stable_without_iteration_order_dependence() -> None:
    keys = ("fetcher_10", "fetcher_2", "fetcher_1", "fetcher_7")

    selected = select_deterministic_sample_keys_v1(
        keys=keys,
        sample_size=3,
        seed=240604,
        split_name="train",
    )
    selected_again = select_deterministic_sample_keys_v1(
        keys=tuple(reversed(keys)),
        sample_size=3,
        seed=240604,
        split_name="train",
    )

    assert selected == selected_again
    assert selected == tuple(sorted(selected, key=lambda key: int(key.rsplit("_", 1)[-1])))


def test_hf_reproducibility_smoke_is_deterministic_and_sanitized(tmp_path: Path) -> None:
    split_specs = _write_fixture_dataset(tmp_path)
    config = HfReproducibilityRunConfig(
        seed=240604,
        trainer="numpy_centroid",
        train_sample_size=6,
        evaluation_sample_size=4,
        backtest_sample_size=4,
    )

    first = run_hf_reproducibility_smoke_v1(
        dataset_dir=tmp_path,
        config=config,
        split_specs=split_specs,
    )
    second = run_hf_reproducibility_smoke_v1(
        dataset_dir=tmp_path,
        config=config,
        split_specs=split_specs,
    )

    assert first == second
    assert first["run_config_hash"] == run_config_hash_v1(config)
    assert first["limits"]["raw_arrays_in_report"] is False  # type: ignore[index]
    assert first["smoke"]["training_smoke"]["sample_size"] == 6  # type: ignore[index]
    assert first["smoke"]["evaluation_smoke"]["sample_size"] == 4  # type: ignore[index]
    assert first["smoke"]["backtest_smoke"]["sample_size"] == 4  # type: ignore[index]
    rendered = render_json_payload_v1(first)
    assert "array(" not in rendered
    assert "raw_provider_payload" not in rendered


def test_inspection_rejects_hash_mismatch_before_loading_npz(tmp_path: Path) -> None:
    split_specs = _write_fixture_dataset(tmp_path)
    spec = split_specs[0]
    bad_spec = HfDatasetSplitSpec(
        split_name=spec.split_name,
        file_name=spec.file_name,
        expected_sha256="0" * 64,
        card_sessions=spec.card_sessions,
        observed_sessions=spec.observed_sessions,
        observed_unique_symbols=spec.observed_unique_symbols,
        observed_period_start_utc=spec.observed_period_start_utc,
        observed_period_end_utc=spec.observed_period_end_utc,
        dtype_summary=spec.dtype_summary,
    )

    with pytest.raises(HfReproducibilityError) as exc_info:
        inspect_hf_split_file_v1(split_spec=bad_spec, dataset_dir=tmp_path)

    assert exc_info.value.reason == "hf_split_hash_mismatch"
    assert exc_info.value.field == spec.file_name


def _write_fixture_dataset(root: Path) -> tuple[HfDatasetSplitSpec, ...]:
    return (
        _write_split(root, split_name="train", file_name="train_data.npz", session_count=8),
        _write_split(root, split_name="validation", file_name="val_data.npz", session_count=5),
        _write_split(root, split_name="test", file_name="test_data.npz", session_count=5),
        _write_split(root, split_name="backtest", file_name="backtest_data.npz", session_count=5),
    )


def _write_split(
    root: Path,
    *,
    split_name: str,
    file_name: str,
    session_count: int,
) -> HfDatasetSplitSpec:
    arrays: dict[str, Any] = {}
    keys_map: dict[str, tuple[str, str]] = {}
    for idx in range(session_count):
        key = f"fetcher_{idx}"
        arrays[key] = _session_array(idx)
        keys_map[key] = (f"SYM{idx % 3}USDT", f"2025-01-01 00:{idx:02d}")
    arrays["_keys_map_"] = np.array(keys_map, dtype=object)
    path = root / file_name
    np.savez(path, **arrays)
    return HfDatasetSplitSpec(
        split_name=split_name,  # type: ignore[arg-type]
        file_name=file_name,
        expected_sha256=compute_file_sha256(path),
        card_sessions=session_count,
        observed_sessions=session_count,
        observed_unique_symbols=3,
        observed_period_start_utc="2025-01-01 00:00",
        observed_period_end_utc=f"2025-01-01 00:{session_count - 1:02d}",
        dtype_summary="fixture arrays float64, each shaped (150, 7)",
    )


def _session_array(idx: int) -> np.ndarray:
    base = 100.0 + idx
    close = np.linspace(base, base + 1.0, 150, dtype=np.float64)
    if idx % 2 == 0:
        close[89:] = np.linspace(base + 0.2, base + 3.0, 61, dtype=np.float64)
    else:
        close[89:] = np.linspace(base + 0.2, base - 2.0, 61, dtype=np.float64)

    array = np.zeros((150, 7), dtype=np.float64)
    array[:, 0] = close - 0.05
    array[:, 1] = close + 0.25
    array[:, 2] = close + 0.02
    array[:, 3] = close - 0.25
    array[:, 4] = close
    array[:, 5] = np.linspace(10.0 + idx, 20.0 + idx, 150, dtype=np.float64)
    array[:, 6] = np.linspace(5.0 + idx, 15.0 + idx, 150, dtype=np.float64)
    return copy.deepcopy(array)
