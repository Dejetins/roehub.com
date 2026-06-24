from __future__ import annotations

import math
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from .action_state_reward_contract import (
    ACTION_NAMES_BY_ID_V1,
    ACTION_STATE_REWARD_CONTRACT_HASH_V1,
)
from .feature_contract import FEATURE_CONTRACT_HASH_V1, FEATURE_NAMES_V1
from .raw_feature_dataset import hash_json_payload_v1, render_raw_feature_json_payload_v1
from .upstream_methodology import (
    UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
    UPSTREAM_METHODOLOGY_PARITY_ID_V1,
    UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    NormalizationStats,
    TorchD3qnPerAgent,
    UpstreamAlphaConfig,
    UpstreamTradingEnvironment,
    compute_train_only_normalization_stats_v1,
    default_upstream_alpha_config_v1,
    select_checkpoint_policy_v1,
)

STAGE08C_SCHEMA_VERSION_V1 = 1
STAGE08C_ORIGINAL_HF_RUN_KIND_V1 = "rl_trading_stage08c_original_hf_training_run"
STAGE08C_PROGRESS_KIND_V1 = "rl_trading_stage08c_training_progress"
STAGE08C_CANDIDATE_MANIFEST_KIND_V1 = "rl_trading_stage08c_hf_original_candidate_manifest"
STAGE08C_CANDIDATE_LEVEL_V1 = "hf_original_candidate"
STAGE08C_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08c_original_hf_full_training_run_v1"

DevicePolicy = Literal["cpu_only_deterministic", "mps_preferred_cpu_fallback"]
TrainingStatus = Literal["starting", "running", "completed", "failed", "interrupted"]


class HfOriginalTrainingError(ValueError):
    def __init__(self, *, reason: str, field: str | None = None) -> None:
        self.reason = reason
        self.field = field
        message = reason if field is None else f"{reason}: {field}"
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class HfOriginalTrainingConfig:
    alpha: UpstreamAlphaConfig = field(default_factory=default_upstream_alpha_config_v1)
    planned_episodes: int = 55_000
    validation_every_episodes: int = 1_000
    checkpoint_every_episodes: int = 1_000
    progress_emit_every_episodes: int = 100
    progress_emit_every_sec: int = 300
    validation_max_sessions: int | None = None
    device_policy: DevicePolicy = "mps_preferred_cpu_fallback"

    def __post_init__(self) -> None:
        _positive_int(self.planned_episodes, "planned_episodes")
        _positive_int(self.validation_every_episodes, "validation_every_episodes")
        _positive_int(self.checkpoint_every_episodes, "checkpoint_every_episodes")
        _positive_int(self.progress_emit_every_episodes, "progress_emit_every_episodes")
        _positive_int(self.progress_emit_every_sec, "progress_emit_every_sec")
        if self.validation_max_sessions is not None:
            _positive_int(self.validation_max_sessions, "validation_max_sessions")
        if self.device_policy not in {"cpu_only_deterministic", "mps_preferred_cpu_fallback"}:
            raise HfOriginalTrainingError(reason="unsupported_device_policy")

    @property
    def planned_env_steps(self) -> int:
        return self.planned_episodes * self.alpha.agent_session_len

    def as_payload(self) -> dict[str, object]:
        return {
            "alpha_config": self.alpha.as_payload(),
            "alpha_config_hash": self.alpha.config_hash(),
            "checkpoint_every_episodes": self.checkpoint_every_episodes,
            "device_policy": self.device_policy,
            "planned_env_steps": self.planned_env_steps,
            "planned_episodes": self.planned_episodes,
            "progress_emit_every_episodes": self.progress_emit_every_episodes,
            "progress_emit_every_sec": self.progress_emit_every_sec,
            "stage": "08C",
            "validation_every_episodes": self.validation_every_episodes,
            "validation_max_sessions": self.validation_max_sessions,
        }

    def config_hash(self) -> str:
        return hash_json_payload_v1(self.as_payload())


def default_hf_original_training_config_v1() -> HfOriginalTrainingConfig:
    return HfOriginalTrainingConfig()


def run_stage08c_hf_original_training_v1(
    *,
    train_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    dataset_dependency: Mapping[str, Any],
    output_root: Path,
    run_id: str,
    config: HfOriginalTrainingConfig | None = None,
    generated_at_utc: datetime | None = None,
    code_version: Mapping[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    selected_config = default_hf_original_training_config_v1() if config is None else config
    train = _validate_sequences(train_sequences, config=selected_config.alpha, field="train")
    validation = _validate_sequences(
        validation_sequences,
        config=selected_config.alpha,
        field="validation",
    )
    _validate_dataset_dependency(dataset_dependency)

    torch = _import_torch()
    generated = generated_at_utc or datetime.now(UTC).replace(microsecond=0)
    run_dir = output_root / run_id
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "best_checkpoint": checkpoints_dir / "best.pth",
        "candidate_manifest": run_dir / "hf_original_candidate_manifest.json",
        "candidate_report": run_dir / "hf_original_training_report.json",
        "final_checkpoint": checkpoints_dir / "final.pth",
        "latest_checkpoint": run_dir / "latest_checkpoint.json",
        "latest_status": run_dir / "latest_status.json",
        "normalization_stats": run_dir / "train_only_normalization_stats.json",
        "progress": run_dir / "progress.jsonl",
        "resume_checkpoint": checkpoints_dir / "latest_resume.pth",
        "training_config": run_dir / "training_config.json",
    }

    stats = compute_train_only_normalization_stats_v1(train, config=selected_config.alpha)
    _atomic_write_json(
        paths["normalization_stats"],
        {
            **stats.as_payload(),
            "normalization_stats_hash": stats.stats_hash(),
            "stage": "08C",
        },
    )
    source_payload = {} if code_version is None else dict(code_version)
    training_config_payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "code_version": source_payload,
        "config": selected_config.as_payload(),
        "config_hash": selected_config.config_hash(),
        "dataset_dependency": dict(dataset_dependency),
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "generated_at_utc": _format_utc(generated),
        "methodology_parity_id": UPSTREAM_METHODOLOGY_PARITY_ID_V1,
        "run_id": run_id,
        "schema_version": STAGE08C_SCHEMA_VERSION_V1,
        "stage": "08C",
        "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    }
    training_config_payload = {
        **training_config_payload,
        "training_config_hash": hash_json_payload_v1(training_config_payload),
    }
    _atomic_write_json(paths["training_config"], training_config_payload)

    train_env = UpstreamTradingEnvironment(
        sequences=train,
        normalization_stats=stats,
        config=selected_config.alpha,
    )
    agent = TorchD3qnPerAgent(
        config=selected_config.alpha,
        device_policy=selected_config.device_policy,
    )
    writer = _Stage08cProgressWriter(
        run_id=run_id,
        progress_path=paths["progress"],
        latest_status_path=paths["latest_status"],
        planned_episodes=selected_config.planned_episodes,
        planned_env_steps=selected_config.planned_env_steps,
        device=str(agent.device_payload["selected_device"]),
    )

    completed_episodes = 0
    completed_env_steps = 0
    best_metric: float | None = None
    best_episode: int | None = None
    train_curve: list[dict[str, object]] = []
    validation_curve: list[dict[str, object]] = []
    counters = _empty_counters()
    loss_window: list[float] = []
    reward_window: list[float] = []
    resume_payload: dict[str, object] | None = None
    if resume and paths["latest_checkpoint"].exists():
        resume_payload = _load_latest_resume_checkpoint(
            torch=torch,
            latest_checkpoint_path=paths["latest_checkpoint"],
            device=agent.device,
        )
        if resume_payload is not None:
            completed_episodes = _payload_int(
                resume_payload.get("completed_episodes"),
                "completed_episodes",
            )
            completed_env_steps = _payload_int(
                resume_payload.get("completed_env_steps"),
                "completed_env_steps",
            )
            best_metric_payload = resume_payload.get("best_metric")
            best_metric = None if best_metric_payload is None else _payload_float(
                best_metric_payload,
                "best_metric",
            )
            best_episode_payload = resume_payload.get("best_episode")
            best_episode = None if best_episode_payload is None else _payload_int(
                best_episode_payload,
                "best_episode",
            )
            train_curve = _payload_list_of_dicts(resume_payload.get("train_curve"))
            validation_curve = _payload_list_of_dicts(resume_payload.get("validation_curve"))
            counters = _payload_counters(resume_payload.get("counters"))
            loss_window = [
                _payload_float(value, "loss_window")
                for value in _payload_sequence(resume_payload.get("loss_window"))
            ]
            reward_window = [
                _payload_float(value, "reward_window")
                for value in _payload_sequence(resume_payload.get("reward_window"))
            ]
            _restore_agent_state(agent=agent, payload=resume_payload)

    writer.emit(
        status="starting" if completed_episodes == 0 else "running",
        completed_episodes=completed_episodes,
        completed_env_steps=completed_env_steps,
        details={
            "checkpoint_resume": resume_payload is not None,
            "config_hash": selected_config.config_hash(),
            "normalization_stats_hash": stats.stats_hash(),
            "train_session_count": int(train.shape[0]),
            "validation_session_count": int(validation.shape[0]),
        },
    )

    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    last_emit_episode = completed_episodes
    last_emit_wall = time.perf_counter()
    status: TrainingStatus = "running"
    try:
        for episode in range(completed_episodes + 1, selected_config.planned_episodes + 1):
            episode_result = _run_training_episode(
                environment=train_env,
                agent=agent,
                episode_index=episode - 1,
                counters=counters,
            )
            completed_episodes = episode
            completed_env_steps += _payload_int(episode_result["env_steps"], "env_steps")
            reward_window.append(
                _payload_float(episode_result["episode_reward"], "episode_reward")
            )
            loss_value = episode_result.get("latest_loss")
            if loss_value is not None:
                loss_window.append(_payload_float(loss_value, "latest_loss"))

            should_validate = (
                episode == selected_config.planned_episodes
                or episode % selected_config.validation_every_episodes == 0
            )
            if should_validate:
                train_point = _training_curve_point(
                    completed_episodes=completed_episodes,
                    completed_env_steps=completed_env_steps,
                    counters=counters,
                    loss_window=loss_window,
                    reward_window=reward_window,
                )
                validation_point = evaluate_stage08c_validation_v1(
                    agent=agent,
                    validation_sequences=validation,
                    normalization_stats=stats,
                    completed_episodes=completed_episodes,
                    completed_env_steps=completed_env_steps,
                    config=selected_config,
                )
                train_curve.append(train_point)
                validation_curve.append(validation_point)
                current_metric = _payload_float(
                    validation_point["Validation_mean_pnl"],
                    "Validation_mean_pnl",
                )
                if best_metric is None or current_metric > best_metric:
                    best_metric = current_metric
                    best_episode = completed_episodes
                    _save_model_checkpoint(
                        torch=torch,
                        path=paths["best_checkpoint"],
                        agent=agent,
                        checkpoint_name="best.pth",
                        completed_episodes=completed_episodes,
                        completed_env_steps=completed_env_steps,
                        config_hash=selected_config.config_hash(),
                        dataset_dependency=dict(dataset_dependency),
                        validation_point=validation_point,
                    )
                loss_window = []
                reward_window = []

            should_checkpoint = (
                episode == selected_config.planned_episodes
                or should_validate
                or episode % selected_config.checkpoint_every_episodes == 0
            )
            if should_checkpoint:
                _save_resume_checkpoint(
                    torch=torch,
                    path=paths["resume_checkpoint"],
                    latest_checkpoint_path=paths["latest_checkpoint"],
                    agent=agent,
                    completed_episodes=completed_episodes,
                    completed_env_steps=completed_env_steps,
                    run_id=run_id,
                    config_hash=selected_config.config_hash(),
                    dataset_dependency=dict(dataset_dependency),
                    train_curve=train_curve,
                    validation_curve=validation_curve,
                    counters=counters,
                    best_metric=best_metric,
                    best_episode=best_episode,
                    loss_window=loss_window,
                    reward_window=reward_window,
                )

            wall_now = time.perf_counter()
            should_emit = (
                episode == 1
                or episode == selected_config.planned_episodes
                or (episode - last_emit_episode) >= selected_config.progress_emit_every_episodes
                or (wall_now - last_emit_wall) >= selected_config.progress_emit_every_sec
            )
            if should_emit:
                writer.emit(
                    status="running",
                    completed_episodes=completed_episodes,
                    completed_env_steps=completed_env_steps,
                    details={
                        "best_episode": best_episode,
                        "best_validation_metric": None
                        if best_metric is None
                        else _round_float(best_metric),
                        "learn_update_count": counters["learn_update_count"],
                        "train_curve_points": len(train_curve),
                        "validation_curve_points": len(validation_curve),
                    },
                )
                last_emit_episode = episode
                last_emit_wall = wall_now
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        writer.emit(
            status="interrupted",
            completed_episodes=completed_episodes,
            completed_env_steps=completed_env_steps,
            details={"reason": "keyboard_interrupt"},
        )
        raise
    except Exception as exc:
        status = "failed"
        writer.emit(
            status="failed",
            completed_episodes=completed_episodes,
            completed_env_steps=completed_env_steps,
            details={"reason": type(exc).__name__},
        )
        raise

    _synchronize_agent(agent)
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    _save_model_checkpoint(
        torch=torch,
        path=paths["final_checkpoint"],
        agent=agent,
        checkpoint_name="final.pth",
        completed_episodes=completed_episodes,
        completed_env_steps=completed_env_steps,
        config_hash=selected_config.config_hash(),
        dataset_dependency=dict(dataset_dependency),
        validation_point=validation_curve[-1] if validation_curve else {},
    )
    checkpoint_policy = select_checkpoint_policy_v1(validation_curve)
    writer.emit(
        status=status,
        completed_episodes=completed_episodes,
        completed_env_steps=completed_env_steps,
        details={
            "best_checkpoint_path": str(paths["best_checkpoint"]),
            "final_checkpoint_path": str(paths["final_checkpoint"]),
            "train_curve_points": len(train_curve),
            "validation_curve_points": len(validation_curve),
        },
    )
    resource_usage = _resource_usage_payload(
        agent=agent,
        start_usage=start_usage,
        end_usage=end_usage,
        progress_writer=writer,
        completed_episodes=completed_episodes,
        completed_env_steps=completed_env_steps,
    )
    artifact_hashes = {
        "best_checkpoint": _file_payload(paths["best_checkpoint"]),
        "final_checkpoint": _file_payload(paths["final_checkpoint"]),
        "latest_checkpoint": _file_payload(paths["latest_checkpoint"]),
        "latest_status": _file_payload(paths["latest_status"]),
        "normalization_stats": _file_payload(paths["normalization_stats"]),
        "progress_jsonl": _file_payload(paths["progress"]),
        "resume_checkpoint": _file_payload(paths["resume_checkpoint"]),
        "training_config": _file_payload(paths["training_config"]),
    }
    metrics = {
        "completed_env_steps": completed_env_steps,
        "completed_episodes": completed_episodes,
        "progress_pct": 100.0,
        "scripted_transition_sequence_used": False,
        "selection_mode_counts": counters["selection_mode_counts"],
        "target_sync_count": agent.target_sync_count,
        "throughput_env_steps_per_sec": _round_float(
            completed_env_steps / max(writer.elapsed_sec(), 1e-9)
        ),
        "throughput_episodes_per_sec": _round_float(
            completed_episodes / max(writer.elapsed_sec(), 1e-9)
        ),
        "train_curve": train_curve,
        "training_used_environment_rollout": True,
        "validation_curve": validation_curve,
    }
    report = _build_report_payload(
        generated_at_utc=generated,
        finished_at_utc=datetime.now(UTC).replace(microsecond=0),
        run_id=run_id,
        run_dir=run_dir,
        config=selected_config,
        dataset_dependency=dict(dataset_dependency),
        code_version=source_payload,
        normalization_stats_hash=stats.stats_hash(),
        checkpoint_policy=checkpoint_policy,
        metrics=metrics,
        resource_usage=resource_usage,
        artifact_hashes=artifact_hashes,
    )
    _atomic_write_json(paths["candidate_report"], report)
    artifact_hashes = {
        **artifact_hashes,
        "candidate_report": _file_payload(paths["candidate_report"]),
    }
    manifest = _build_manifest_payload(
        generated_at_utc=generated,
        run_id=run_id,
        run_dir=run_dir,
        config=selected_config,
        dataset_dependency=dict(dataset_dependency),
        code_version=source_payload,
        normalization_stats_hash=stats.stats_hash(),
        checkpoint_policy=checkpoint_policy,
        metrics=metrics,
        resource_usage=resource_usage,
        artifact_hashes=artifact_hashes,
    )
    _atomic_write_json(paths["candidate_manifest"], manifest)
    manifest = {
        **manifest,
        "artifact_hashes": {
            **artifact_hashes,
            "candidate_manifest": _file_payload(paths["candidate_manifest"]),
        },
        "candidate_manifest_path": str(paths["candidate_manifest"]),
    }
    manifest = _finalize_manifest(manifest)
    _atomic_write_json(paths["candidate_manifest"], manifest)
    return manifest


def evaluate_stage08c_validation_v1(
    *,
    agent: TorchD3qnPerAgent,
    validation_sequences: np.ndarray,
    normalization_stats: NormalizationStats,
    completed_episodes: int,
    completed_env_steps: int,
    config: HfOriginalTrainingConfig,
) -> dict[str, object]:
    validation = _validate_sequences(
        validation_sequences,
        config=config.alpha,
        field="validation",
    )
    limit = int(validation.shape[0])
    if config.validation_max_sessions is not None:
        limit = min(limit, config.validation_max_sessions)
    if limit <= 0:
        raise HfOriginalTrainingError(reason="validation_sessions_required")
    env = UpstreamTradingEnvironment(
        sequences=validation[:limit],
        normalization_stats=normalization_stats,
        config=config.alpha,
    )
    pnls: list[float] = []
    rewards: list[float] = []
    action_counts = {ACTION_NAMES_BY_ID_V1[action_id]: 0 for action_id in ACTION_NAMES_BY_ID_V1}
    for session_idx in range(limit):
        state, _ = env.reset(forced_index=session_idx)
        done = False
        latest_info: Mapping[str, object] = {}
        total_reward = 0.0
        while not done:
            selection = agent.select_action_with_details(
                state,
                training=False,
                valid_actions=env.valid_actions(),
            )
            next_state, reward, done, _, info = env.step(selection.action_id)
            effective_action_id = _payload_int(info["effective_action_id"], "effective_action_id")
            action_counts[ACTION_NAMES_BY_ID_V1[effective_action_id]] += 1
            total_reward += float(reward)
            state = next_state
            latest_info = info
        pnls.append(_payload_float(latest_info.get("episode_realized_pnl", 0.0), "pnl"))
        rewards.append(total_reward)
    mean_pnl = float(np.mean(np.asarray(pnls, dtype=np.float64)))
    return {
        "Validation_mean_pnl": _round_float(mean_pnl),
        "Validation_mean_reward": _round_float(float(np.mean(np.asarray(rewards)))),
        "Validation_pnl_std": _round_float(float(np.std(np.asarray(pnls, dtype=np.float64)))),
        "action_counts": action_counts,
        "completed_env_steps": completed_env_steps,
        "completed_episodes": completed_episodes,
        "completed_training_steps": completed_env_steps,
        "sampled_validation_sessions": limit,
        "total_validation_sessions": int(validation.shape[0]),
    }


def _run_training_episode(
    *,
    environment: UpstreamTradingEnvironment,
    agent: TorchD3qnPerAgent,
    episode_index: int,
    counters: dict[str, Any],
) -> dict[str, object]:
    state, _ = environment.reset(forced_index=episode_index % len(environment.sequences))
    done = False
    env_steps = 0
    total_reward = 0.0
    latest_loss: float | None = None
    latest_info: Mapping[str, object] = {}
    while not done:
        selection = agent.select_action_with_details(
            state,
            training=True,
            valid_actions=environment.valid_actions(),
        )
        next_state, reward, done, _, info = environment.step(selection.action_id)
        agent.store_experience(state, selection.action_id, reward, next_state, done)
        learning_result = agent.learn()
        if learning_result is not None:
            counters["learn_update_count"] += 1
            counters["target_sync_count"] = agent.target_sync_count
            latest_loss = learning_result.loss
            counters["latest_mean_abs_td_error"] = learning_result.mean_abs_td_error
        agent.increment_step()
        counters["completed_env_steps_observed"] += 1
        counters["selection_mode_counts"][selection.mode] += 1
        requested_name = ACTION_NAMES_BY_ID_V1[selection.action_id]
        counters["requested_action_counts"][requested_name] += 1
        effective_action_id = _payload_int(info["effective_action_id"], "effective_action_id")
        effective_name = ACTION_NAMES_BY_ID_V1[effective_action_id]
        counters["effective_action_counts"][effective_name] += 1
        audit_reason = str(info["audit_reason"])
        counters["audit_reason_counts"][audit_reason] = (
            counters["audit_reason_counts"].get(audit_reason, 0) + 1
        )
        total_reward += float(reward)
        env_steps += 1
        latest_info = info
        state = next_state
    return {
        "env_steps": env_steps,
        "episode_realized_pnl": latest_info.get("episode_realized_pnl", 0.0),
        "episode_reward": _round_float(total_reward),
        "latest_loss": latest_loss,
    }


def _training_curve_point(
    *,
    completed_episodes: int,
    completed_env_steps: int,
    counters: Mapping[str, Any],
    loss_window: Sequence[float],
    reward_window: Sequence[float],
) -> dict[str, object]:
    return {
        "completed_env_steps": completed_env_steps,
        "completed_episodes": completed_episodes,
        "completed_training_steps": completed_env_steps,
        "learn_update_count": _payload_int(counters["learn_update_count"], "learn_update_count"),
        "loss_window_mean": None
        if not loss_window
        else _round_float(float(np.mean(np.asarray(loss_window, dtype=np.float64)))),
        "loss_window_size": len(loss_window),
        "mean_abs_td_error_last": counters.get("latest_mean_abs_td_error"),
        "reward_window_mean": None
        if not reward_window
        else _round_float(float(np.mean(np.asarray(reward_window, dtype=np.float64)))),
        "reward_window_size": len(reward_window),
        "target_sync_count": _payload_int(counters["target_sync_count"], "target_sync_count"),
    }


def _save_model_checkpoint(
    *,
    torch: Any,
    path: Path,
    agent: TorchD3qnPerAgent,
    checkpoint_name: str,
    completed_episodes: int,
    completed_env_steps: int,
    config_hash: str,
    dataset_dependency: Mapping[str, Any],
    validation_point: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "checkpoint_name": checkpoint_name,
        "completed_env_steps": completed_env_steps,
        "completed_episodes": completed_episodes,
        "config_hash": config_hash,
        "dataset_dependency": dict(dataset_dependency),
        "policy_state": _cpu_state_dict(agent.policy_net),
        "schema_version": STAGE08C_SCHEMA_VERSION_V1,
        "stage": "08C",
        "target_state": _cpu_state_dict(agent.target_net),
        "validation_point": dict(validation_point),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_resume_checkpoint(
    *,
    torch: Any,
    path: Path,
    latest_checkpoint_path: Path,
    agent: TorchD3qnPerAgent,
    completed_episodes: int,
    completed_env_steps: int,
    run_id: str,
    config_hash: str,
    dataset_dependency: Mapping[str, Any],
    train_curve: Sequence[Mapping[str, object]],
    validation_curve: Sequence[Mapping[str, object]],
    counters: Mapping[str, Any],
    best_metric: float | None,
    best_episode: int | None,
    loss_window: Sequence[float],
    reward_window: Sequence[float],
) -> None:
    payload = {
        "agent_learn_steps": agent.learn_steps,
        "agent_target_sync_count": agent.target_sync_count,
        "agent_total_steps": agent.total_steps,
        "best_episode": best_episode,
        "best_metric": best_metric,
        "completed_env_steps": completed_env_steps,
        "completed_episodes": completed_episodes,
        "config_hash": config_hash,
        "counters": _json_safe(counters),
        "dataset_dependency": dict(dataset_dependency),
        "loss_window": list(loss_window),
        "optimizer_state": agent.optimizer.state_dict(),
        "policy_state": _cpu_state_dict(agent.policy_net),
        "replay_buffer_state": agent.replay_buffer.state_payload(),
        "reward_window": list(reward_window),
        "run_id": run_id,
        "schema_version": STAGE08C_SCHEMA_VERSION_V1,
        "stage": "08C",
        "target_state": _cpu_state_dict(agent.target_net),
        "train_curve": list(train_curve),
        "validation_curve": list(validation_curve),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    _atomic_write_json(
        latest_checkpoint_path,
        {
            "checkpoint": _file_payload(path),
            "completed_env_steps": completed_env_steps,
            "completed_episodes": completed_episodes,
            "run_id": run_id,
            "stage": "08C",
        },
    )


def _load_latest_resume_checkpoint(
    *,
    torch: Any,
    latest_checkpoint_path: Path,
    device: Any,
) -> dict[str, object] | None:
    if not latest_checkpoint_path.exists():
        return None
    latest = _read_json_payload(latest_checkpoint_path)
    checkpoint = latest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise HfOriginalTrainingError(reason="latest_checkpoint_payload_invalid")
    path_value = checkpoint.get("path")
    if not isinstance(path_value, str):
        raise HfOriginalTrainingError(reason="latest_checkpoint_path_missing")
    checkpoint_path = Path(path_value)
    if not checkpoint_path.exists():
        raise HfOriginalTrainingError(reason="latest_checkpoint_file_missing", field=path_value)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise HfOriginalTrainingError(reason="resume_checkpoint_payload_invalid")
    return cast(dict[str, object], payload)


def _restore_agent_state(*, agent: TorchD3qnPerAgent, payload: Mapping[str, object]) -> None:
    policy_state = payload.get("policy_state")
    target_state = payload.get("target_state")
    optimizer_state = payload.get("optimizer_state")
    replay_state = payload.get("replay_buffer_state")
    if not isinstance(policy_state, Mapping) or not isinstance(target_state, Mapping):
        raise HfOriginalTrainingError(reason="checkpoint_model_state_invalid")
    if not isinstance(optimizer_state, Mapping):
        raise HfOriginalTrainingError(reason="checkpoint_optimizer_state_invalid")
    if not isinstance(replay_state, Mapping):
        raise HfOriginalTrainingError(reason="checkpoint_replay_state_invalid")
    agent.policy_net.load_state_dict(policy_state)
    agent.target_net.load_state_dict(target_state)
    agent.optimizer.load_state_dict(optimizer_state)
    _optimizer_state_to_device(agent=agent)
    agent.replay_buffer.restore_state_payload(replay_state)
    agent.total_steps = _payload_int(payload.get("agent_total_steps"), "agent_total_steps")
    agent.learn_steps = _payload_int(payload.get("agent_learn_steps"), "agent_learn_steps")
    agent.target_sync_count = _payload_int(
        payload.get("agent_target_sync_count"),
        "agent_target_sync_count",
    )


def _optimizer_state_to_device(*, agent: TorchD3qnPerAgent) -> None:
    for state in agent.optimizer.state.values():
        if not isinstance(state, dict):
            continue
        for key, value in list(state.items()):
            if hasattr(value, "to"):
                state[key] = value.to(agent.device)


def _build_report_payload(
    *,
    generated_at_utc: datetime,
    finished_at_utc: datetime,
    run_id: str,
    run_dir: Path,
    config: HfOriginalTrainingConfig,
    dataset_dependency: Mapping[str, Any],
    code_version: Mapping[str, Any],
    normalization_stats_hash: str,
    checkpoint_policy: Mapping[str, object],
    metrics: Mapping[str, Any],
    resource_usage: Mapping[str, Any],
    artifact_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "artifact_hashes": dict(artifact_hashes),
        "artifact_kind": STAGE08C_ORIGINAL_HF_RUN_KIND_V1,
        "candidate_level": STAGE08C_CANDIDATE_LEVEL_V1,
        "checkpoint_policy": dict(checkpoint_policy),
        "code_version": dict(code_version),
        "config": config.as_payload(),
        "config_hash": config.config_hash(),
        "dataset_dependency": dict(dataset_dependency),
        "dependency_isolation": {
            "default_api_runtime_requires_torch": False,
            "torch_extra": "rl-ml",
        },
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "finished_at_utc": _format_utc(finished_at_utc),
        "generated_at_utc": _format_utc(generated_at_utc),
        "metrics": dict(metrics),
        "normalization_stats_hash": normalization_stats_hash,
        "resource_usage": dict(resource_usage),
        "run_dir": str(run_dir),
        "run_id": run_id,
        "safety": _safety_payload(),
        "schema_version": STAGE08C_SCHEMA_VERSION_V1,
        "stage": "08C",
        "status": "completed",
        "upstream_methodology_parity": True,
        "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    }
    return {**payload, "candidate_report_hash": hash_json_payload_v1(payload)}


def _build_manifest_payload(
    *,
    generated_at_utc: datetime,
    run_id: str,
    run_dir: Path,
    config: HfOriginalTrainingConfig,
    dataset_dependency: Mapping[str, Any],
    code_version: Mapping[str, Any],
    normalization_stats_hash: str,
    checkpoint_policy: Mapping[str, object],
    metrics: Mapping[str, Any],
    resource_usage: Mapping[str, Any],
    artifact_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "action_state_reward_contract_hash": ACTION_STATE_REWARD_CONTRACT_HASH_V1,
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "artifact_hashes": dict(artifact_hashes),
        "artifact_kind": STAGE08C_CANDIDATE_MANIFEST_KIND_V1,
        "candidate_level": STAGE08C_CANDIDATE_LEVEL_V1,
        "checkpoint_policy": dict(checkpoint_policy),
        "code_version": dict(code_version),
        "config_hash": config.config_hash(),
        "dataset_dependency": dict(dataset_dependency),
        "feature_contract_hash": FEATURE_CONTRACT_HASH_V1,
        "generated_at_utc": _format_utc(generated_at_utc),
        "metrics_summary": {
            "completed_env_steps": metrics["completed_env_steps"],
            "completed_episodes": metrics["completed_episodes"],
            "progress_pct": metrics["progress_pct"],
            "scripted_transition_sequence_used": metrics["scripted_transition_sequence_used"],
            "throughput_env_steps_per_sec": metrics["throughput_env_steps_per_sec"],
            "throughput_episodes_per_sec": metrics["throughput_episodes_per_sec"],
            "train_curve_points": len(cast(Sequence[Any], metrics["train_curve"])),
            "training_used_environment_rollout": metrics["training_used_environment_rollout"],
            "validation_curve_points": len(cast(Sequence[Any], metrics["validation_curve"])),
        },
        "next_stage_handoff": {
            "stage08d_allowed": True,
            "stage08d_input": "hf_original_candidate_manifest_path",
        },
        "normalization_stats_hash": normalization_stats_hash,
        "resource_summary": dict(resource_usage),
        "run_dir": str(run_dir),
        "run_id": run_id,
        "safety": _safety_payload(),
        "schema_version": STAGE08C_SCHEMA_VERSION_V1,
        "stage": "08C",
        "status": "completed",
        "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    }
    return _finalize_manifest(payload)


def _finalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "candidate_manifest_hash"
    }
    return {**payload, "candidate_manifest_hash": hash_json_payload_v1(payload)}


class _Stage08cProgressWriter:
    def __init__(
        self,
        *,
        run_id: str,
        progress_path: Path,
        latest_status_path: Path,
        planned_episodes: int,
        planned_env_steps: int,
        device: str,
    ) -> None:
        self.run_id = run_id
        self.progress_path = progress_path
        self.latest_status_path = latest_status_path
        self.planned_episodes = planned_episodes
        self.planned_env_steps = planned_env_steps
        self.device = device
        self.started_at_wall = time.perf_counter()
        self.started_at_utc = datetime.now(UTC).replace(microsecond=0)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

    def elapsed_sec(self) -> float:
        return _round_float(time.perf_counter() - self.started_at_wall)

    def emit(
        self,
        *,
        status: TrainingStatus,
        completed_episodes: int,
        completed_env_steps: int,
        details: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        elapsed = self.elapsed_sec()
        progress_pct = _round_float((completed_episodes / self.planned_episodes) * 100.0)
        eta_sec: float | None = None
        if status in {"starting", "running"} and completed_episodes > 0:
            remaining = self.planned_episodes - completed_episodes
            eta_sec = _round_float((elapsed / completed_episodes) * remaining)
        elif status == "completed":
            eta_sec = 0.0
        event = {
            "artifact_kind": STAGE08C_PROGRESS_KIND_V1,
            "completed_env_steps": completed_env_steps,
            "completed_episodes": completed_episodes,
            "details": {} if details is None else dict(details),
            "device": self.device,
            "elapsed_sec": elapsed,
            "eta_sec": eta_sec,
            "planned_env_steps": self.planned_env_steps,
            "planned_episodes": self.planned_episodes,
            "progress_pct": progress_pct,
            "resource_snapshot": _resource_snapshot_payload(),
            "run_id": self.run_id,
            "stage": "08C",
            "started_at_utc": _format_utc(self.started_at_utc),
            "status": status,
            "timestamp": _format_utc(datetime.now(UTC).replace(microsecond=0)),
        }
        rendered = _render_json_line_payload(event)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        latest = {
            **event,
            "latest_status_path": str(self.latest_status_path),
            "progress_path": str(self.progress_path),
        }
        _atomic_write_json(self.latest_status_path, latest)
        return event


def _empty_counters() -> dict[str, Any]:
    return {
        "audit_reason_counts": {},
        "completed_env_steps_observed": 0,
        "effective_action_counts": {
            ACTION_NAMES_BY_ID_V1[action_id]: 0 for action_id in ACTION_NAMES_BY_ID_V1
        },
        "latest_mean_abs_td_error": None,
        "learn_update_count": 0,
        "requested_action_counts": {
            ACTION_NAMES_BY_ID_V1[action_id]: 0 for action_id in ACTION_NAMES_BY_ID_V1
        },
        "selection_mode_counts": {"cache_hit": 0, "epsilon_random": 0, "greedy": 0},
        "target_sync_count": 0,
    }


def _payload_counters(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return _empty_counters()
    counters = _empty_counters()
    for key in counters:
        if key in value:
            counters[key] = value[key]
    return counters


def _safety_payload() -> dict[str, object]:
    return {
        "allow_browser_runtime_verification": False,
        "allow_exchange_side_effects": False,
        "allow_mainnet_submit": False,
        "candidate_level": STAGE08C_CANDIDATE_LEVEL_V1,
        "contains_raw_provider_payloads": False,
        "contains_secrets": False,
        "hf_original_training_only": True,
        "register_promote_activate_trade": False,
        "stage06_roehub_native_data_used": False,
        "stage08d_evaluation_run": False,
    }


def _resource_usage_payload(
    *,
    agent: TorchD3qnPerAgent,
    start_usage: Any,
    end_usage: Any,
    progress_writer: _Stage08cProgressWriter,
    completed_episodes: int,
    completed_env_steps: int,
) -> dict[str, object]:
    elapsed = progress_writer.elapsed_sec()
    return {
        "completed_env_steps": completed_env_steps,
        "completed_episodes": completed_episodes,
        "cpu_system_seconds_delta": _round_float(end_usage.ru_stime - start_usage.ru_stime),
        "cpu_user_seconds_delta": _round_float(end_usage.ru_utime - start_usage.ru_utime),
        "mps_available": bool(agent.device_payload["mps_available"]),
        "mps_built": bool(agent.device_payload["mps_built"]),
        "process_threads_observed": _process_thread_count(),
        "rss_mb_after": _rss_mb(),
        "selected_device": agent.device_payload["selected_device"],
        "torch_num_interop_threads": int(agent.torch.get_num_interop_threads()),
        "torch_num_threads": int(agent.torch.get_num_threads()),
        "wall_seconds": elapsed,
    }


def _validate_dataset_dependency(value: Mapping[str, Any]) -> None:
    if value.get("stage") != "04":
        raise HfOriginalTrainingError(reason="unexpected_dataset_dependency_stage")
    if value.get("source_market") != "binance:futures":
        raise HfOriginalTrainingError(reason="unexpected_dataset_dependency_market")


def _validate_sequences(
    sequences: np.ndarray,
    *,
    config: UpstreamAlphaConfig,
    field: str,
) -> np.ndarray:
    value = np.asarray(sequences, dtype=np.float32)
    expected_shape = (config.full_seq_len, len(FEATURE_NAMES_V1))
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_shape:
        raise HfOriginalTrainingError(reason="sequence_shape_mismatch", field=field)
    if value.shape[0] <= 0:
        raise HfOriginalTrainingError(reason="sequence_empty", field=field)
    if not np.all(np.isfinite(value)):
        raise HfOriginalTrainingError(reason="sequence_non_finite", field=field)
    close_idx = FEATURE_NAMES_V1.index("close")
    if np.any(value[:, :, close_idx] <= 0.0):
        raise HfOriginalTrainingError(reason="sequence_non_positive_close", field=field)
    return np.ascontiguousarray(value, dtype=np.float32)


def _cpu_state_dict(module: Any) -> dict[str, Any]:
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def _synchronize_agent(agent: TorchD3qnPerAgent) -> None:
    device_type = str(agent.device.type)
    if device_type == "mps" and hasattr(agent.torch, "mps"):
        agent.torch.mps.synchronize()


def _import_torch() -> Any:
    try:
        import importlib

        return importlib.import_module("torch")
    except Exception as exc:
        raise HfOriginalTrainingError(reason="torch_import_failed", field=str(exc)) from exc


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_raw_feature_json_payload_v1(_json_safe(payload)) + "\n", encoding="utf-8")
    tmp.replace(path)


def _file_payload(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    import hashlib

    return {"bytes": len(data), "path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def _read_json_payload(path: Path) -> dict[str, Any]:
    import json

    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _render_json_line_payload(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _resource_snapshot_payload() -> dict[str, object]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "cpu_system_seconds": _round_float(usage.ru_stime),
        "cpu_user_seconds": _round_float(usage.ru_utime),
        "process_threads_observed": _process_thread_count(),
        "rss_mb": _rss_mb(),
    }


def _process_thread_count() -> int | None:
    try:
        import os

        status_path = Path(f"/proc/{os.getpid()}/status")
        if not status_path.exists():
            return None
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("Threads:"):
                return int(line.split(":", 1)[1].strip())
    except Exception:
        return None
    return None


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = float(usage.ru_maxrss)
    if value > 10_000_000:
        value = value / (1024.0 * 1024.0)
    else:
        value = value / 1024.0
    return _round_float(value)


def _format_utc(value: datetime) -> str:
    selected = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return selected.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _payload_sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return value


def _payload_list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    out = []
    for item in value:
        if isinstance(item, Mapping):
            out.append(dict(item))
    return out


def _payload_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HfOriginalTrainingError(reason="payload_int_invalid", field=field)
    return int(value)


def _payload_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HfOriginalTrainingError(reason="payload_float_invalid", field=field)
    out = float(value)
    if not math.isfinite(out):
        raise HfOriginalTrainingError(reason="payload_float_non_finite", field=field)
    return out


def _positive_int(value: int, field: str) -> int:
    if isinstance(value, bool) or int(value) <= 0:
        raise HfOriginalTrainingError(reason="non_positive_int", field=field)
    return int(value)


def _round_float(value: float) -> float:
    return round(float(value), 10)
