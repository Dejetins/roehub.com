from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from trading.contexts.rl_trading.domain import stage08k_monitor_policy as policy_module
from trading.contexts.rl_trading.domain.action_state_reward_contract import RlTrainingState
from trading.contexts.rl_trading.domain.feature_contract import FEATURE_NAMES_V1, RlFeatureCandle
from trading.contexts.rl_trading.domain.raw_feature_dataset import RawFeatureSlab
from trading.contexts.rl_trading.domain.sessionized_dataset import (
    SessionSplitWindow,
    select_article_future_impulse_session_candidates_v1,
)
from trading.contexts.rl_trading.domain.stage08k_monitor_policy import (
    STAGE08K_MONITOR_REQUIRED_CANDIDATE_SHA256_V1,
    STAGE08K_MONITOR_REQUIRED_CHECKPOINT_SHA256_V1,
    STAGE08K_MONITOR_REQUIRED_EVALUATION_SHA256_V1,
    STAGE08K_MONITOR_REQUIRED_NORMALIZATION_FILE_SHA256_V1,
    Stage08kArtifactContract,
    Stage08kMonitorPolicyConfig,
    Stage08kMonitorPolicyError,
    Stage08kPreloadedMonitorPolicy,
    preload_stage08k_monitor_policy_v1,
    score_stage08k_live_signal_v1,
)
from trading.contexts.rl_trading.domain.upstream_methodology import (
    UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
    NormalizationStats,
    UpstreamAlphaConfig,
    build_upstream_entry_state_from_history_v1,
    build_upstream_state_v1,
)


def test_entry_state_uses_exact_training_step_zero_representation() -> None:
    config = UpstreamAlphaConfig(torch_num_threads=1, torch_num_interop_threads=1)
    session = _session_matrix()
    stats = _normalization_stats()

    training_state = build_upstream_state_v1(
        session=session,
        step_idx=0,
        action_history=[None] * config.action_history_len,
        training_state=RlTrainingState(balance=config.initial_balance),
        normalization_stats=stats,
        config=config,
    )
    live_state = build_upstream_entry_state_from_history_v1(
        history=session[: config.pre_signal_len],
        normalization_stats=stats,
        config=config,
    )

    np.testing.assert_array_equal(live_state, training_state)


def test_article_live_signal_uses_only_closed_past_candles() -> None:
    candles = list(_candles(count=90))
    candles[-1] = _candle(open_=100.0, close=106.0)

    signal = score_stage08k_live_signal_v1(candles)

    assert signal.eligible is True
    assert signal.reason == "article_signal_eligible"
    assert signal.event_return == pytest.approx(0.06)
    assert signal.volatility_score == pytest.approx(0.06)
    assert signal.contrast_max_abs_return == 0.0


def test_article_live_signal_matches_dataset_builder_at_same_signal_boundary() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    matrix = _session_matrix()
    matrix[:90, :] = _session_matrix()[:90, :]
    matrix[:90, FEATURE_NAMES_V1.index("open")] = 100.0
    matrix[:90, FEATURE_NAMES_V1.index("close")] = 100.0
    matrix[89, FEATURE_NAMES_V1.index("close")] = 106.0
    matrix[:90, FEATURE_NAMES_V1.index("high")] = np.maximum(
        matrix[:90, FEATURE_NAMES_V1.index("open")],
        matrix[:90, FEATURE_NAMES_V1.index("close")],
    )
    matrix[:90, FEATURE_NAMES_V1.index("low")] = np.minimum(
        matrix[:90, FEATURE_NAMES_V1.index("open")],
        matrix[:90, FEATURE_NAMES_V1.index("close")],
    )
    open_times = np.asarray(
        [int((start + timedelta(minutes=index)).timestamp() * 1_000) for index in range(150)],
        dtype=np.int64,
    )
    slab = RawFeatureSlab(
        open_time_ms=open_times,
        close_time_ms=open_times + 60_000,
        features_f32=np.ascontiguousarray(matrix, dtype=np.float32),
    )
    signal_time = start + timedelta(minutes=90)
    candidates = select_article_future_impulse_session_candidates_v1(
        slab=slab,
        split_window=SessionSplitWindow(
            dataset_version="golden",
            split="test",
            signal_start_utc=signal_time.isoformat(),
            signal_end_utc=(signal_time + timedelta(minutes=1)).isoformat(),
            source_start_utc=start.isoformat(),
            source_end_utc=(start + timedelta(minutes=150)).isoformat(),
        ),
        symbol="BTCUSDT",
    )
    candles = tuple(
        RlFeatureCandle(
            open=float(row[FEATURE_NAMES_V1.index("open")]),
            high=float(row[FEATURE_NAMES_V1.index("high")]),
            low=float(row[FEATURE_NAMES_V1.index("low")]),
            close=float(row[FEATURE_NAMES_V1.index("close")]),
            volume_base=float(row[FEATURE_NAMES_V1.index("volume")]),
            volume_quote=float(row[FEATURE_NAMES_V1.index("volume_weighted_average")])
            * float(row[FEATURE_NAMES_V1.index("volume")]),
            trades_count=int(row[FEATURE_NAMES_V1.index("num_trades")]),
        )
        for row in matrix[:90]
    )

    live_signal = score_stage08k_live_signal_v1(candles)

    assert len(candidates) == 1
    assert live_signal.eligible is True
    assert live_signal.volatility_score == pytest.approx(candidates[0].volatility_score)
    assert live_signal.contrast_max_abs_return == pytest.approx(
        candidates[0].article_contrast_max_abs_return
    )


def test_monitor_policy_allows_long_and_blocks_short() -> None:
    candles = list(_candles(count=90))
    candles[-1] = _candle(open_=100.0, close=106.0)
    config = UpstreamAlphaConfig(torch_num_threads=1, torch_num_interop_threads=1)

    long_policy = Stage08kPreloadedMonitorPolicy(
        agent=cast(Any, _FakeAgent([0.0, 0.2, 0.1, 0.05])),
        alpha=config,
        normalization_stats=_normalization_stats(),
        policy_config=Stage08kMonitorPolicyConfig(),
    )
    short_policy = Stage08kPreloadedMonitorPolicy(
        agent=cast(Any, _FakeAgent([0.0, 0.1, 0.2, 0.05])),
        alpha=config,
        normalization_stats=_normalization_stats(),
        policy_config=Stage08kMonitorPolicyConfig(),
    )

    long_decision = long_policy.decide(candles)
    short_decision = short_policy.decide(candles)

    assert long_decision.requested_action_name == "open_long"
    assert long_decision.action_name == "open_long"
    assert long_decision.policy_reason == "model_action_allowed"
    assert short_decision.requested_action_name == "open_short"
    assert short_decision.action_name == "hold"
    assert short_decision.policy_reason == "short_blocked_by_monitor_policy"


def test_trusted_loader_rejects_unexpected_hash_and_path_outside_root() -> None:
    artifacts = _artifact_contract()
    with pytest.raises(Stage08kMonitorPolicyError, match="candidate_manifest"):
        preload_stage08k_monitor_policy_v1(
            artifacts=replace(artifacts, candidate_manifest_sha256="0" * 64),
            policy_config=Stage08kMonitorPolicyConfig(),
            torch_num_threads=1,
            torch_num_interop_threads=1,
        )
    with pytest.raises(Stage08kMonitorPolicyError, match="artifact_outside_root"):
        preload_stage08k_monitor_policy_v1(
            artifacts=replace(
                artifacts,
                candidate_manifest_path=Path("/tmp/candidate.json"),
            ),
            policy_config=Stage08kMonitorPolicyConfig(),
            torch_num_threads=1,
            torch_num_interop_threads=1,
        )


def test_trusted_loader_uses_weights_only_and_validates_lineage(monkeypatch) -> None:
    artifacts = _artifact_contract()
    stats = _normalization_stats()
    candidate = _candidate_payload(artifacts)
    evaluation = _evaluation_payload(stats_hash=stats.stats_hash())
    normalization = {
        "feature_names": list(stats.feature_names),
        "means": stats.means,
        "sequence_count": stats.sequence_count,
        "source_split": stats.source_split,
        "stds": stats.stds,
    }
    payloads = {
        "candidate_manifest": candidate,
        "evaluation_manifest": evaluation,
        "normalization_stats": normalization,
    }
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        policy_module,
        "_load_trusted_json",
        lambda _path, _sha, *, root, field: payloads[field],
    )
    monkeypatch.setattr(
        policy_module,
        "_trusted_path",
        lambda path, *, expected_sha256, root, field: path,
    )
    monkeypatch.setattr(
        policy_module,
        "TorchD3qnPerAgent",
        lambda **kwargs: _FakeLoadedAgent(calls=calls),
    )

    loaded = preload_stage08k_monitor_policy_v1(
        artifacts=artifacts,
        policy_config=Stage08kMonitorPolicyConfig(),
        torch_num_threads=1,
        torch_num_interop_threads=1,
    )

    assert loaded.model_version_id == "stage08k_roehub_native_best_3e033951"
    assert calls == [{"map_location": "cpu", "weights_only": True}]

    payloads["evaluation_manifest"] = {**evaluation, "status": "blocked"}
    with pytest.raises(Stage08kMonitorPolicyError, match="evaluation_research_lineage"):
        preload_stage08k_monitor_policy_v1(
            artifacts=artifacts,
            policy_config=Stage08kMonitorPolicyConfig(),
            torch_num_threads=1,
            torch_num_interop_threads=1,
        )
    payloads["evaluation_manifest"] = {
        **evaluation,
        "normalization_stats_hash": "0" * 64,
    }
    with pytest.raises(Stage08kMonitorPolicyError, match="normalization_stats_hash"):
        preload_stage08k_monitor_policy_v1(
            artifacts=artifacts,
            policy_config=Stage08kMonitorPolicyConfig(),
            torch_num_threads=1,
            torch_num_interop_threads=1,
        )
    payloads["evaluation_manifest"] = evaluation
    monkeypatch.setattr(
        policy_module,
        "TorchD3qnPerAgent",
        lambda **kwargs: _FakeLoadedAgent(
            calls=[],
            checkpoint_overrides={"architecture_id": "unexpected"},
        ),
    )
    with pytest.raises(Stage08kMonitorPolicyError, match="checkpoint_architecture"):
        preload_stage08k_monitor_policy_v1(
            artifacts=artifacts,
            policy_config=Stage08kMonitorPolicyConfig(),
            torch_num_threads=1,
            torch_num_interop_threads=1,
        )


class _FakeAgent:
    def __init__(self, q_values: list[float]) -> None:
        self._q_values = np.asarray(q_values, dtype=np.float32)

    def predict_q_values(self, state: np.ndarray) -> np.ndarray:
        assert state.shape == (219,)
        return self._q_values.copy()


class _FakeNet:
    def load_state_dict(self, state: object) -> None:
        assert state == {"weight": 1}

    def eval(self) -> None:
        return


class _FakeTorch:
    def __init__(
        self,
        *,
        calls: list[dict[str, object]],
        checkpoint_overrides: dict[str, object] | None = None,
    ) -> None:
        self._calls = calls
        self._checkpoint_overrides = checkpoint_overrides or {}

    def load(
        self,
        _path: str,
        *,
        map_location: object,
        weights_only: bool,
    ) -> dict[str, object]:
        self._calls.append(
            {"map_location": map_location, "weights_only": weights_only}
        )
        return {
            "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
            "config_hash": "cfg",
            "policy_state": {"weight": 1},
            "stage": "08E",
            "target_state": {"weight": 1},
            **self._checkpoint_overrides,
        }


class _FakeLoadedAgent:
    def __init__(
        self,
        *,
        calls: list[dict[str, object]],
        checkpoint_overrides: dict[str, object] | None = None,
    ) -> None:
        self.device = "cpu"
        self.policy_net = _FakeNet()
        self.target_net = _FakeNet()
        self.torch = _FakeTorch(
            calls=calls,
            checkpoint_overrides=checkpoint_overrides,
        )


def _artifact_contract() -> Stage08kArtifactContract:
    root = Path("/opt/roehub/state/rl_trading")
    return Stage08kArtifactContract(
        artifact_root=root,
        candidate_manifest_path=root / "candidate.json",
        candidate_manifest_sha256=STAGE08K_MONITOR_REQUIRED_CANDIDATE_SHA256_V1,
        evaluation_manifest_path=root / "evaluation.json",
        evaluation_manifest_sha256=STAGE08K_MONITOR_REQUIRED_EVALUATION_SHA256_V1,
        checkpoint_path=root / "best.pth",
        checkpoint_sha256=STAGE08K_MONITOR_REQUIRED_CHECKPOINT_SHA256_V1,
        normalization_stats_path=root / "normalization.json",
        normalization_stats_file_sha256=(
            STAGE08K_MONITOR_REQUIRED_NORMALIZATION_FILE_SHA256_V1
        ),
    )


def _candidate_payload(artifacts: Stage08kArtifactContract) -> dict[str, object]:
    return {
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "artifact_hashes": {
            "best_checkpoint": {
                "path": str(artifacts.checkpoint_path),
                "sha256": artifacts.checkpoint_sha256,
            },
            "normalization_stats": {
                "path": str(artifacts.normalization_stats_path),
                "sha256": artifacts.normalization_stats_file_sha256,
            },
        },
        "checkpoint_policy": {"default_evaluation_checkpoint": "best"},
        "config_hash": "cfg",
        "stage": "08E",
        "status": "completed",
    }


def _evaluation_payload(*, stats_hash: str) -> dict[str, object]:
    return {
        "candidate_dependency": {
            "manifest_sha256": STAGE08K_MONITOR_REQUIRED_CANDIDATE_SHA256_V1
        },
        "config": {"alpha_config": {}},
        "feature_contract_hash": policy_module.FEATURE_CONTRACT_HASH_V1,
        "normalization_stats_hash": stats_hash,
        "research_candidate_save_allowed": True,
        "stage": "08F",
        "status": "accepted_for_research",
    }


def _normalization_stats() -> NormalizationStats:
    return NormalizationStats(
        means={name: 0.0 for name in FEATURE_NAMES_V1},
        stds={name: 1.0 for name in FEATURE_NAMES_V1},
        source_split="train",
        sequence_count=1,
    )


def _session_matrix() -> np.ndarray:
    rows = []
    for index in range(150):
        close = 100.0 + (index * 0.01)
        rows.append([close, close + 0.2, close, close - 0.2, close, 10.0, 5.0])
    return np.asarray(rows, dtype=np.float32)


def _candles(*, count: int) -> tuple[RlFeatureCandle, ...]:
    return tuple(_candle(open_=100.0, close=100.0) for _ in range(count))


def _candle(*, open_: float, close: float) -> RlFeatureCandle:
    return RlFeatureCandle(
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        volume_base=10.0,
        volume_quote=close * 10.0,
        trades_count=5,
    )
