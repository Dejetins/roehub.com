from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from .action_state_reward_contract import RlTrainingState, apply_training_reward_step_v1
from .feature_contract import FEATURE_NAMES_V1

HF_REPRODUCIBILITY_SCHEMA_VERSION_V1 = 1
HF_DATASET_REPO_ID_V1 = "ResearchRL/open-rl-trading-binance-dataset"
HF_DATASET_URL_V1 = (
    "https://huggingface.co/datasets/ResearchRL/open-rl-trading-binance-dataset"
)
HF_DATASET_RESOLVE_BASE_URL_V1 = (
    "https://huggingface.co/datasets/ResearchRL/open-rl-trading-binance-dataset"
    "/resolve/main"
)
HF_EXTERNAL_REPO_URL_V1 = "https://github.com/YuriyKolesnikov/rl-trading-binance"
HF_EXTERNAL_REPO_ID_V1 = "YuriyKolesnikov/rl-trading-binance"
HF_RUNTIME_ARTIFACT_ROOT_V1 = "/opt/roehub/state/rl_trading/"
HF_SESSION_SHAPE_V1 = (150, 7)
HF_PRE_SIGNAL_LEN_V1 = 90
HF_POST_SIGNAL_LEN_V1 = 60
HF_TRAIN_CARD_SESSION_COUNT_DELTA_V1 = -18

SmokeTrainerName = Literal["numpy_centroid", "torch_logistic"]
SplitName = Literal["train", "validation", "test", "backtest"]
PredictionSplitName = Literal["train", "validation", "backtest"]


class HfReproducibilityError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class HfDatasetSplitSpec:
    split_name: SplitName
    file_name: str
    expected_sha256: str
    card_sessions: int
    observed_sessions: int
    observed_unique_symbols: int
    observed_period_start_utc: str
    observed_period_end_utc: str
    dtype_summary: str

    def as_payload(self) -> dict[str, object]:
        return {
            "card_sessions": self.card_sessions,
            "dtype_summary": self.dtype_summary,
            "expected_sha256": self.expected_sha256,
            "file_name": self.file_name,
            "observed_period_end_utc": self.observed_period_end_utc,
            "observed_period_start_utc": self.observed_period_start_utc,
            "observed_sessions": self.observed_sessions,
            "observed_unique_symbols": self.observed_unique_symbols,
            "session_count_delta_vs_card": self.observed_sessions - self.card_sessions,
            "split_name": self.split_name,
        }


@dataclass(frozen=True, slots=True)
class HfSplitInspection:
    spec: HfDatasetSplitSpec
    file_path: Path
    file_size_bytes: int
    sha256: str
    fetcher_key_count: int
    keys_map_count: int
    unique_symbols: int
    first_signal_utc: str | None
    last_signal_utc: str | None

    def as_payload(self) -> dict[str, object]:
        return {
            "expected_sha256": self.spec.expected_sha256,
            "fetcher_key_count": self.fetcher_key_count,
            "file_name": self.spec.file_name,
            "file_path": str(self.file_path),
            "file_size_bytes": self.file_size_bytes,
            "first_signal_utc": self.first_signal_utc,
            "hash_matches_expected": self.sha256 == self.spec.expected_sha256,
            "keys_map_count": self.keys_map_count,
            "last_signal_utc": self.last_signal_utc,
            "observed_sessions_expected": self.spec.observed_sessions,
            "observed_unique_symbols_expected": self.spec.observed_unique_symbols,
            "sha256": self.sha256,
            "split_name": self.spec.split_name,
            "unique_symbols": self.unique_symbols,
        }


@dataclass(frozen=True, slots=True)
class HfSessionExample:
    key: str
    symbol: str
    signal_time_utc: str
    features: tuple[float, ...]
    label: int
    signal_close: float
    final_close: float
    future_return: float

    def as_sanitized_payload(self) -> dict[str, object]:
        return {
            "final_close": _round_float(self.final_close),
            "future_return": _round_float(self.future_return),
            "key": self.key,
            "label": self.label,
            "signal_close": _round_float(self.signal_close),
            "signal_time_utc": self.signal_time_utc,
            "symbol": self.symbol,
        }


@dataclass(frozen=True, slots=True)
class HfReproducibilityRunConfig:
    seed: int = 240604
    trainer: SmokeTrainerName = "torch_logistic"
    train_sample_size: int = 32
    evaluation_sample_size: int = 16
    backtest_sample_size: int = 16
    pre_signal_len: int = HF_PRE_SIGNAL_LEN_V1
    post_signal_len: int = HF_POST_SIGNAL_LEN_V1
    torch_epochs: int = 16
    torch_learning_rate: float = 0.05
    initial_balance: float = 100.0
    slippage: float = 0.0
    transaction_fee: float = 0.001
    inaction_penalty_ratio: float = 0.0
    device: str = "cpu"

    def as_payload(self) -> dict[str, object]:
        return {
            "backtest_sample_size": self.backtest_sample_size,
            "channel_order": list(FEATURE_NAMES_V1),
            "device": self.device,
            "evaluation_sample_size": self.evaluation_sample_size,
            "initial_balance": self.initial_balance,
            "inaction_penalty_ratio": self.inaction_penalty_ratio,
            "post_signal_len": self.post_signal_len,
            "pre_signal_len": self.pre_signal_len,
            "seed": self.seed,
            "slippage": self.slippage,
            "torch_epochs": self.torch_epochs,
            "torch_learning_rate": self.torch_learning_rate,
            "train_sample_size": self.train_sample_size,
            "trainer": self.trainer,
            "transaction_fee": self.transaction_fee,
        }


@dataclass(frozen=True, slots=True)
class TrainedSmokeModel:
    trainer: SmokeTrainerName
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    training_summary: dict[str, object]
    centroid_by_label: dict[int, tuple[float, ...]] | None = None
    torch_weights: tuple[float, ...] | None = None
    torch_bias: float | None = None


@dataclass(frozen=True, slots=True)
class PredictionResult:
    predictions: tuple[int, ...]
    probabilities: tuple[float, ...]


_EXPECTED_HF_SPLIT_SPECS_V1: tuple[HfDatasetSplitSpec, ...] = (
    HfDatasetSplitSpec(
        split_name="train",
        file_name="train_data.npz",
        expected_sha256="1c5cdf179777f0a68a81da915749f50d97826282e1419a5314a67b170e9cb14d",
        card_sessions=24104,
        observed_sessions=24086,
        observed_unique_symbols=309,
        observed_period_start_utc="2020-01-14 14:28",
        observed_period_end_utc="2024-08-30 18:33",
        dtype_summary="24,085 float64 arrays and 1 float32 array, each shaped (150, 7)",
    ),
    HfDatasetSplitSpec(
        split_name="validation",
        file_name="val_data.npz",
        expected_sha256="1e1e347bd4f842680f8a1781bc1e51f790f5e5865796e9ef3bd69548e20c51f4",
        card_sessions=1377,
        observed_sessions=1377,
        observed_unique_symbols=280,
        observed_period_start_utc="2024-09-01 06:02",
        observed_period_end_utc="2024-11-30 22:46",
        dtype_summary="all arrays float64, each shaped (150, 7)",
    ),
    HfDatasetSplitSpec(
        split_name="test",
        file_name="test_data.npz",
        expected_sha256="ff72d998fbf7d507b3db46e543aae324bece368a50ad043c057217ec2c744b1b",
        card_sessions=3400,
        observed_sessions=3400,
        observed_unique_symbols=362,
        observed_period_start_utc="2024-12-01 00:16",
        observed_period_end_utc="2025-02-28 22:53",
        dtype_summary="all arrays float64, each shaped (150, 7)",
    ),
    HfDatasetSplitSpec(
        split_name="backtest",
        file_name="backtest_data.npz",
        expected_sha256="dce732fda8fe1d33e92617d12f0defa3e202013617b91bb34df4d0b65aa023ee",
        card_sessions=3186,
        observed_sessions=3186,
        observed_unique_symbols=321,
        observed_period_start_utc="2025-03-01 00:15",
        observed_period_end_utc="2025-05-31 22:47",
        dtype_summary="all arrays float64, each shaped (150, 7)",
    ),
)


def expected_hf_split_specs_v1() -> tuple[HfDatasetSplitSpec, ...]:
    return _EXPECTED_HF_SPLIT_SPECS_V1


def hf_attribution_payload_v1() -> dict[str, object]:
    return {
        "dataset": {
            "id": HF_DATASET_REPO_ID_V1,
            "license": "MIT License",
            "url": HF_DATASET_URL_V1,
        },
        "external_repo": {
            "id": HF_EXTERNAL_REPO_ID_V1,
            "license": "MIT License",
            "url": HF_EXTERNAL_REPO_URL_V1,
        },
        "import_policy": {
            "external_code_vendored": False,
            "large_artifact_root": HF_RUNTIME_ARTIFACT_ROOT_V1,
            "notes": (
                "Concepts only: NPZ session format, 90/60 split, action/reward semantics, "
                "and baseline train/eval/backtest lifecycle are adapted to Roehub code style."
            ),
        },
    }


def expected_hf_dataset_manifest_payload_v1() -> dict[str, object]:
    return {
        "attribution": hf_attribution_payload_v1(),
        "channel_order_observed": list(FEATURE_NAMES_V1),
        "dataset_format": {
            "metadata_key": "_keys_map_",
            "session_key_pattern": "fetcher_N",
            "session_shape": list(HF_SESSION_SHAPE_V1),
            "source_market": "binance:futures",
        },
        "dataset_repo": HF_DATASET_REPO_ID_V1,
        "schema_version": HF_REPRODUCIBILITY_SCHEMA_VERSION_V1,
        "splits": [spec.as_payload() for spec in _EXPECTED_HF_SPLIT_SPECS_V1],
        "train_count_mismatch": {
            "card_sessions": 24104,
            "observed_sessions": 24086,
            "observed_minus_card": HF_TRAIN_CARD_SESSION_COUNT_DELTA_V1,
        },
    }


def expected_hf_dataset_manifest_hash_v1() -> str:
    return _sha256_text(_canonical_json(expected_hf_dataset_manifest_payload_v1()))


def run_config_hash_v1(config: HfReproducibilityRunConfig) -> str:
    return _sha256_text(_canonical_json(config.as_payload()))


def inspect_hf_split_file_v1(
    *,
    split_spec: HfDatasetSplitSpec,
    dataset_dir: Path,
) -> HfSplitInspection:
    file_path = dataset_dir / split_spec.file_name
    if not file_path.exists():
        raise HfReproducibilityError(reason="missing_hf_split_file", field=split_spec.file_name)

    sha256 = compute_file_sha256(file_path)
    if sha256 != split_spec.expected_sha256:
        raise HfReproducibilityError(reason="hf_split_hash_mismatch", field=split_spec.file_name)

    with np.load(file_path, allow_pickle=True) as archive:
        keys_map = _load_keys_map(archive)
        fetcher_keys = _fetcher_keys(archive.files)

    symbols = {_metadata_symbol(value) for value in keys_map.values()}
    signal_times = sorted(_metadata_signal_time(value) for value in keys_map.values())

    if len(fetcher_keys) != split_spec.observed_sessions:
        raise HfReproducibilityError(
            reason="hf_split_fetcher_count_mismatch",
            field=split_spec.file_name,
        )
    if len(keys_map) != split_spec.observed_sessions:
        raise HfReproducibilityError(
            reason="hf_split_keys_map_count_mismatch",
            field=split_spec.file_name,
        )

    return HfSplitInspection(
        spec=split_spec,
        file_path=file_path,
        file_size_bytes=file_path.stat().st_size,
        sha256=sha256,
        fetcher_key_count=len(fetcher_keys),
        keys_map_count=len(keys_map),
        unique_symbols=len(symbols),
        first_signal_utc=signal_times[0] if signal_times else None,
        last_signal_utc=signal_times[-1] if signal_times else None,
    )


def run_hf_reproducibility_smoke_v1(
    *,
    dataset_dir: Path,
    config: HfReproducibilityRunConfig,
    split_specs: Sequence[HfDatasetSplitSpec] = _EXPECTED_HF_SPLIT_SPECS_V1,
) -> dict[str, object]:
    specs_by_name = {spec.split_name: spec for spec in split_specs}
    required = ("train", "validation", "test", "backtest")
    missing_specs = [name for name in required if name not in specs_by_name]
    if missing_specs:
        raise HfReproducibilityError(reason="missing_split_spec", field=",".join(missing_specs))

    inspections = [
        inspect_hf_split_file_v1(split_spec=specs_by_name[name], dataset_dir=dataset_dir)
        for name in required
    ]
    samples = {
        "train": load_hf_sample_examples_v1(
            dataset_dir=dataset_dir,
            split_spec=specs_by_name["train"],
            sample_size=config.train_sample_size,
            seed=config.seed,
        ),
        "validation": load_hf_sample_examples_v1(
            dataset_dir=dataset_dir,
            split_spec=specs_by_name["validation"],
            sample_size=config.evaluation_sample_size,
            seed=config.seed,
        ),
        "backtest": load_hf_sample_examples_v1(
            dataset_dir=dataset_dir,
            split_spec=specs_by_name["backtest"],
            sample_size=config.backtest_sample_size,
            seed=config.seed,
        ),
    }
    train_model = train_smoke_model_v1(samples["train"], config=config)

    split_metrics = {
        split_name: evaluate_smoke_split_v1(
            examples=examples,
            model=train_model,
            config=config,
        )
        for split_name, examples in samples.items()
    }
    payload = {
        "attribution": hf_attribution_payload_v1(),
        "dataset_manifest": {
            "expected_manifest_hash": expected_hf_dataset_manifest_hash_v1(),
            "inspected_splits": [inspection.as_payload() for inspection in inspections],
            "source_market": "binance:futures",
        },
        "limits": {
            "large_artifacts_committed_to_git": False,
            "production_approval": False,
            "raw_arrays_in_report": False,
            "research_only": True,
        },
        "run_config": config.as_payload(),
        "run_config_hash": run_config_hash_v1(config),
        "runtime": _runtime_payload(config.trainer),
        "schema_version": HF_REPRODUCIBILITY_SCHEMA_VERSION_V1,
        "smoke": {
            "backtest_smoke": split_metrics["backtest"],
            "evaluation_smoke": split_metrics["validation"],
            "sample_keys": {
                split_name: [example.key for example in examples]
                for split_name, examples in samples.items()
            },
            "training_smoke": {
                **train_model.training_summary,
                "train_split_metrics": split_metrics["train"],
            },
        },
    }
    return cast(dict[str, object], _sanitize_payload(payload))


def load_hf_sample_examples_v1(
    *,
    dataset_dir: Path,
    split_spec: HfDatasetSplitSpec,
    sample_size: int,
    seed: int,
) -> tuple[HfSessionExample, ...]:
    if sample_size <= 0:
        raise HfReproducibilityError(reason="non_positive_sample_size", field=split_spec.split_name)

    file_path = dataset_dir / split_spec.file_name
    sha256 = compute_file_sha256(file_path)
    if sha256 != split_spec.expected_sha256:
        raise HfReproducibilityError(reason="hf_split_hash_mismatch", field=split_spec.file_name)

    with np.load(file_path, allow_pickle=True) as archive:
        keys_map = _load_keys_map(archive)
        selected_keys = select_deterministic_sample_keys_v1(
            keys=tuple(keys_map.keys()),
            sample_size=sample_size,
            seed=seed,
            split_name=split_spec.split_name,
        )
        examples = []
        for key in selected_keys:
            if key not in archive.files:
                raise HfReproducibilityError(reason="missing_fetcher_array", field=key)
            metadata = keys_map[key]
            array = np.asarray(archive[key], dtype=np.float64)
            examples.append(_build_session_example(key=key, metadata=metadata, array=array))
    return tuple(examples)


def select_deterministic_sample_keys_v1(
    *,
    keys: Sequence[str],
    sample_size: int,
    seed: int,
    split_name: str,
) -> tuple[str, ...]:
    if sample_size <= 0:
        raise HfReproducibilityError(reason="non_positive_sample_size", field="sample_size")
    if not keys:
        raise HfReproducibilityError(reason="empty_split_keys", field=split_name)
    ranked = sorted(
        set(keys),
        key=lambda value: (
            _sha256_text(f"{seed}:{split_name}:{value}"),
            _fetcher_key_sort_value(value),
        ),
    )
    selected = ranked[: min(sample_size, len(ranked))]
    return tuple(sorted(selected, key=_fetcher_key_sort_value))


def train_smoke_model_v1(
    examples: Sequence[HfSessionExample],
    *,
    config: HfReproducibilityRunConfig,
) -> TrainedSmokeModel:
    features, labels = _feature_matrix_and_labels(examples)
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
    normalized = (features - feature_mean) / feature_scale

    if config.trainer == "numpy_centroid":
        return _train_numpy_centroid(
            normalized=normalized,
            labels=labels,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            config=config,
        )
    if config.trainer == "torch_logistic":
        return _train_torch_logistic(
            normalized=normalized,
            labels=labels,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            config=config,
        )
    raise HfReproducibilityError(reason="unsupported_trainer", field=config.trainer)


def predict_smoke_model_v1(
    *,
    model: TrainedSmokeModel,
    examples: Sequence[HfSessionExample],
) -> PredictionResult:
    features, _ = _feature_matrix_and_labels(examples)
    normalized = (features - np.asarray(model.feature_mean)) / np.asarray(model.feature_scale)
    if model.trainer == "numpy_centroid":
        if model.centroid_by_label is None:
            raise HfReproducibilityError(reason="missing_centroids", field="model")
        distances = []
        for row in normalized:
            distance_to_zero = np.linalg.norm(
                row - np.asarray(model.centroid_by_label.get(0, model.centroid_by_label[1]))
            )
            distance_to_one = np.linalg.norm(
                row - np.asarray(model.centroid_by_label.get(1, model.centroid_by_label[0]))
            )
            distances.append((float(distance_to_zero), float(distance_to_one)))
        probabilities = tuple(
            _stable_sigmoid(distance_to_zero - distance_to_one)
            for distance_to_zero, distance_to_one in distances
        )
        predictions = tuple(1 if probability >= 0.5 else 0 for probability in probabilities)
        return PredictionResult(predictions=predictions, probabilities=probabilities)

    if model.trainer == "torch_logistic":
        if model.torch_weights is None or model.torch_bias is None:
            raise HfReproducibilityError(reason="missing_torch_weights", field="model")
        logits = normalized @ np.asarray(model.torch_weights) + model.torch_bias
        probabilities = tuple(_stable_sigmoid(float(logit)) for logit in logits)
        predictions = tuple(1 if probability >= 0.5 else 0 for probability in probabilities)
        return PredictionResult(predictions=predictions, probabilities=probabilities)

    raise HfReproducibilityError(reason="unsupported_trainer", field=model.trainer)


def evaluate_smoke_split_v1(
    *,
    examples: Sequence[HfSessionExample],
    model: TrainedSmokeModel,
    config: HfReproducibilityRunConfig,
) -> dict[str, object]:
    if not examples:
        raise HfReproducibilityError(reason="empty_evaluation_examples")

    labels = tuple(example.label for example in examples)
    majority_label = 1 if sum(labels) >= len(labels) / 2 else 0
    baseline_predictions = tuple(majority_label for _ in labels)
    result = predict_smoke_model_v1(model=model, examples=examples)
    backtest = _simulate_directional_backtest(
        examples=examples,
        predictions=result.predictions,
        config=config,
    )
    return {
        "accuracy": _round_float(_accuracy(result.predictions, labels)),
        "action_counts": {
            "open_long": sum(1 for value in result.predictions if value == 1),
            "open_short": sum(1 for value in result.predictions if value == 0),
        },
        "baseline_accuracy": _round_float(_accuracy(baseline_predictions, labels)),
        "label_counts": {
            "down_or_flat": sum(1 for value in labels if value == 0),
            "up": sum(1 for value in labels if value == 1),
        },
        "mean_probability_up": _round_float(sum(result.probabilities) / len(result.probabilities)),
        "sample_size": len(examples),
        "simulated_directional_backtest": backtest,
    }


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_json_payload_v1(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _train_numpy_centroid(
    *,
    normalized: NDArray[np.float64],
    labels: NDArray[np.int_],
    feature_mean: NDArray[np.float64],
    feature_scale: NDArray[np.float64],
    config: HfReproducibilityRunConfig,
) -> TrainedSmokeModel:
    label_values = set(int(value) for value in labels)
    if not label_values:
        raise HfReproducibilityError(reason="empty_training_labels")
    centroid_by_label = {}
    for label in (0, 1):
        if label in label_values:
            centroid = np.mean(normalized[labels == label], axis=0)
        else:
            centroid = np.mean(normalized, axis=0)
        centroid_by_label[label] = tuple(float(value) for value in centroid)

    model = TrainedSmokeModel(
        trainer=config.trainer,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        training_summary={
            "class_balance": _label_counts(labels),
            "deterministic_seed": config.seed,
            "sample_size": int(labels.size),
            "trainer": config.trainer,
        },
        centroid_by_label=centroid_by_label,
    )
    train_predictions = predict_smoke_model_v1(
        model=model,
        examples=[
            HfSessionExample(
                key=f"training_{idx}",
                symbol="SYNTH",
                signal_time_utc="training",
                features=tuple(float(value) for value in features),
                label=int(label),
                signal_close=1.0,
                final_close=1.0,
                future_return=0.0,
            )
            for idx, (features, label) in enumerate(
                zip(_denormalize(normalized, feature_mean, feature_scale), labels, strict=True)
            )
        ],
    )
    return TrainedSmokeModel(
        trainer=model.trainer,
        feature_mean=model.feature_mean,
        feature_scale=model.feature_scale,
        training_summary={
            **model.training_summary,
            "train_accuracy": _round_float(
                _accuracy(train_predictions.predictions, tuple(int(value) for value in labels))
            ),
        },
        centroid_by_label=model.centroid_by_label,
    )


def _train_torch_logistic(
    *,
    normalized: NDArray[np.float64],
    labels: NDArray[np.int_],
    feature_mean: NDArray[np.float64],
    feature_scale: NDArray[np.float64],
    config: HfReproducibilityRunConfig,
) -> TrainedSmokeModel:
    torch = importlib.import_module("torch")
    torch.manual_seed(config.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    x_train = torch.tensor(normalized, dtype=torch.float32, device=config.device)
    y_train = torch.tensor(labels.reshape(-1, 1), dtype=torch.float32, device=config.device)
    model = torch.nn.Linear(normalized.shape[1], 1).to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.torch_learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses: list[float] = []
    for _ in range(config.torch_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    with torch.no_grad():
        train_probabilities = torch.sigmoid(model(x_train)).detach().cpu().numpy().reshape(-1)
    train_predictions = tuple(1 if float(value) >= 0.5 else 0 for value in train_probabilities)
    weight = model.weight.detach().cpu().numpy().reshape(-1)
    bias = float(model.bias.detach().cpu().numpy().reshape(-1)[0])

    return TrainedSmokeModel(
        trainer=config.trainer,
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        training_summary={
            "class_balance": _label_counts(labels),
            "deterministic_seed": config.seed,
            "device": config.device,
            "final_loss": _round_float(losses[-1]),
            "initial_loss": _round_float(losses[0]),
            "sample_size": int(labels.size),
            "torch_epochs": config.torch_epochs,
            "torch_learning_rate": config.torch_learning_rate,
            "train_accuracy": _round_float(
                _accuracy(train_predictions, tuple(int(value) for value in labels))
            ),
            "trainer": config.trainer,
        },
        torch_weights=tuple(float(value) for value in weight),
        torch_bias=bias,
    )


def _simulate_directional_backtest(
    *,
    examples: Sequence[HfSessionExample],
    predictions: Sequence[int],
    config: HfReproducibilityRunConfig,
) -> dict[str, object]:
    total_pnl_change = 0.0
    total_reward = 0.0
    winning_trades = 0
    for example, prediction in zip(examples, predictions, strict=True):
        action_id = 1 if prediction == 1 else 2
        opened = apply_training_reward_step_v1(
            state=RlTrainingState(balance=config.initial_balance),
            action_id=action_id,
            price=example.signal_close,
            initial_balance=config.initial_balance,
            slippage=config.slippage,
            transaction_fee=config.transaction_fee,
            inaction_penalty_ratio=config.inaction_penalty_ratio,
        )
        closed = apply_training_reward_step_v1(
            state=opened.state,
            action_id=3,
            price=example.final_close,
            initial_balance=config.initial_balance,
            slippage=config.slippage,
            transaction_fee=config.transaction_fee,
            inaction_penalty_ratio=config.inaction_penalty_ratio,
            is_last_step=True,
        )
        trade_pnl = opened.pnl_change + closed.pnl_change
        total_pnl_change += trade_pnl
        total_reward += opened.reward + closed.reward
        if trade_pnl > 0.0:
            winning_trades += 1
    sample_size = len(examples)
    return {
        "mean_reward": _round_float(total_reward / sample_size),
        "sample_size": sample_size,
        "total_pnl_change": _round_float(total_pnl_change),
        "total_pnl_ratio": _round_float(total_pnl_change / (config.initial_balance * sample_size)),
        "win_rate": _round_float(winning_trades / sample_size),
    }


def _build_session_example(
    *,
    key: str,
    metadata: object,
    array: NDArray[np.float64],
) -> HfSessionExample:
    if array.shape != HF_SESSION_SHAPE_V1:
        raise HfReproducibilityError(reason="invalid_session_shape", field=key)
    if not np.all(np.isfinite(array)):
        raise HfReproducibilityError(reason="non_finite_session_values", field=key)

    close_idx = FEATURE_NAMES_V1.index("close")
    high_idx = FEATURE_NAMES_V1.index("high")
    low_idx = FEATURE_NAMES_V1.index("low")
    vwap_idx = FEATURE_NAMES_V1.index("volume_weighted_average")
    volume_idx = FEATURE_NAMES_V1.index("volume")
    trades_idx = FEATURE_NAMES_V1.index("num_trades")

    close_window = array[:HF_PRE_SIGNAL_LEN_V1, close_idx]
    if np.any(close_window <= 0.0):
        raise HfReproducibilityError(reason="non_positive_close", field=key)
    signal_close = float(array[HF_PRE_SIGNAL_LEN_V1 - 1, close_idx])
    final_close = float(array[-1, close_idx])
    if signal_close <= 0.0 or final_close <= 0.0:
        raise HfReproducibilityError(reason="non_positive_close", field=key)

    log_close = np.log(close_window)
    log_return_pre = float(log_close[-1] - log_close[0])
    volatility_pre = float(np.std(np.diff(log_close)))
    high_window = array[:HF_PRE_SIGNAL_LEN_V1, high_idx]
    low_window = array[:HF_PRE_SIGNAL_LEN_V1, low_idx]
    range_ratio = np.mean((high_window - low_window) / close_window)
    vwap_gap = np.mean((array[:HF_PRE_SIGNAL_LEN_V1, vwap_idx] - close_window) / close_window)
    volume_log_mean = np.mean(np.log1p(np.maximum(array[:HF_PRE_SIGNAL_LEN_V1, volume_idx], 0.0)))
    trades_log_mean = np.mean(np.log1p(np.maximum(array[:HF_PRE_SIGNAL_LEN_V1, trades_idx], 0.0)))
    future_return = (final_close - signal_close) / signal_close

    return HfSessionExample(
        key=key,
        symbol=_metadata_symbol(metadata),
        signal_time_utc=_metadata_signal_time(metadata),
        features=(
            log_return_pre,
            volatility_pre,
            float(range_ratio),
            float(vwap_gap),
            float(volume_log_mean),
            float(trades_log_mean),
        ),
        label=1 if future_return > 0.0 else 0,
        signal_close=signal_close,
        final_close=final_close,
        future_return=future_return,
    )


def _feature_matrix_and_labels(
    examples: Sequence[HfSessionExample],
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    if not examples:
        raise HfReproducibilityError(reason="empty_examples")
    features = np.asarray([example.features for example in examples], dtype=np.float64)
    labels = np.asarray([example.label for example in examples], dtype=np.int_)
    if features.ndim != 2 or features.shape[1] == 0:
        raise HfReproducibilityError(reason="invalid_feature_matrix")
    return features, labels


def _load_keys_map(archive: Any) -> Mapping[str, object]:
    if "_keys_map_" not in archive.files:
        raise HfReproducibilityError(reason="missing_keys_map", field="_keys_map_")
    raw = archive["_keys_map_"]
    raw_object = raw.item() if getattr(raw, "shape", ()) == () else raw.tolist()
    if not isinstance(raw_object, Mapping):
        raise HfReproducibilityError(reason="invalid_keys_map", field="_keys_map_")
    return {str(key): value for key, value in raw_object.items()}


def _fetcher_keys(keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            (key for key in keys if key.startswith("fetcher_")),
            key=_fetcher_key_sort_value,
        )
    )


def _fetcher_key_sort_value(key: str) -> tuple[int, str]:
    try:
        return int(key.rsplit("_", maxsplit=1)[-1]), key
    except ValueError:
        return sys.maxsize, key


def _metadata_symbol(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, str) and value:
        return str(value[0]).upper()
    raise HfReproducibilityError(reason="invalid_metadata_symbol")


def _metadata_signal_time(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) >= 2:
        return str(value[1])
    raise HfReproducibilityError(reason="invalid_metadata_signal_time")


def _accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    if len(predictions) != len(labels) or not labels:
        raise HfReproducibilityError(reason="invalid_prediction_lengths")
    matches = sum(
        1 for pred, label in zip(predictions, labels, strict=True) if pred == label
    )
    return matches / len(labels)


def _label_counts(labels: NDArray[np.int_]) -> dict[str, int]:
    return {
        "down_or_flat": int(np.sum(labels == 0)),
        "up": int(np.sum(labels == 1)),
    }


def _denormalize(
    normalized: NDArray[np.float64],
    feature_mean: NDArray[np.float64],
    feature_scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (normalized * feature_scale) + feature_mean


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _runtime_payload(trainer: SmokeTrainerName) -> dict[str, object]:
    payload: dict[str, object] = {
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "trainer": trainer,
    }
    if trainer == "torch_logistic":
        torch = importlib.import_module("torch")
        payload["torch_mps_available"] = bool(torch.backends.mps.is_available())
        payload["torch_version"] = str(torch.__version__)
    return payload


def _sanitize_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_payload(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, float):
        return _round_float(value)
    return value


def _round_float(value: float) -> float:
    if not math.isfinite(value):
        raise HfReproducibilityError(reason="non_finite_output_float")
    return round(float(value), 10)


def _canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
