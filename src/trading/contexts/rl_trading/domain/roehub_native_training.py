from __future__ import annotations

import resource
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from .action_state_reward_contract import ACTION_STATE_REWARD_CONTRACT_HASH_V1
from .feature_contract import FEATURE_CONTRACT_HASH_V1
from .hf_original_training import (
    HfOriginalTrainingConfig,
    HfOriginalTrainingError,
    TrainingStatus,
    _atomic_write_json,
    _cpu_state_dict,
    _empty_counters,
    _file_payload,
    _format_utc,
    _import_torch,
    _json_safe,
    _load_latest_resume_checkpoint,
    _payload_counters,
    _payload_float,
    _payload_int,
    _payload_list_of_dicts,
    _payload_sequence,
    _process_thread_count,
    _resource_snapshot_payload,
    _restore_agent_state,
    _round_float,
    _rss_mb,
    _run_training_episode,
    _training_curve_point,
    _validate_sequences,
    evaluate_stage08c_validation_v1,
)
from .raw_feature_dataset import hash_json_payload_v1
from .upstream_methodology import (
    UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
    UPSTREAM_METHODOLOGY_PARITY_ID_V1,
    UPSTREAM_SOURCE_SHA_STAGE08A_V1,
    TorchD3qnPerAgent,
    UpstreamTradingEnvironment,
    compute_train_only_normalization_stats_v1,
    select_checkpoint_policy_v1,
)

STAGE08E_SCHEMA_VERSION_V1 = 1
STAGE08E_ROEHUB_NATIVE_RUN_KIND_V1 = "rl_trading_stage08e_roehub_native_training_run"
STAGE08E_PROGRESS_KIND_V1 = "rl_trading_stage08e_training_progress"
STAGE08E_CANDIDATE_MANIFEST_KIND_V1 = "rl_trading_stage08e_roehub_native_candidate_manifest"
STAGE08E_CANDIDATE_LEVEL_V1 = "roehub_native_candidate"
STAGE08E_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08e_roehub_native_full_training_run_v1"

RoehubNativeTrainingConfig = HfOriginalTrainingConfig


class RoehubNativeTrainingError(HfOriginalTrainingError):
    pass


def default_roehub_native_training_config_v1() -> RoehubNativeTrainingConfig:
    return RoehubNativeTrainingConfig(stage="08E")


def run_stage08e_roehub_native_training_v1(
    *,
    train_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    dataset_dependency: Mapping[str, Any],
    output_root: Path,
    run_id: str,
    config: RoehubNativeTrainingConfig | None = None,
    generated_at_utc: datetime | None = None,
    code_version: Mapping[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    selected_config = default_roehub_native_training_config_v1() if config is None else config
    if selected_config.stage != "08E":
        raise RoehubNativeTrainingError(
            reason="unexpected_training_stage",
            field=selected_config.stage,
        )
    train = _validate_sequences(train_sequences, config=selected_config.alpha, field="train")
    validation = _validate_sequences(
        validation_sequences,
        config=selected_config.alpha,
        field="validation",
    )
    _validate_stage06_dataset_dependency(dataset_dependency)

    torch = _import_torch()
    generated = generated_at_utc or datetime.now(UTC).replace(microsecond=0)
    run_dir = output_root / run_id
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "best_checkpoint": checkpoints_dir / "best.pth",
        "candidate_manifest": run_dir / "roehub_native_candidate_manifest.json",
        "candidate_report": run_dir / "roehub_native_training_report.json",
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
            "stage": "08E",
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
        "schema_version": STAGE08E_SCHEMA_VERSION_V1,
        "stage": "08E",
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
    writer = _Stage08eProgressWriter(
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
                    _save_stage08e_model_checkpoint(
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
                agent.release_device_cache()
                loss_window = []
                reward_window = []

            should_checkpoint = (
                episode == selected_config.planned_episodes
                or should_validate
                or episode % selected_config.checkpoint_every_episodes == 0
            )
            if should_checkpoint:
                _save_stage08e_resume_checkpoint(
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
                agent.release_device_cache()

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

    agent.release_device_cache()
    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    _save_stage08e_model_checkpoint(
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


def _save_stage08e_model_checkpoint(
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
        "schema_version": STAGE08E_SCHEMA_VERSION_V1,
        "stage": "08E",
        "target_state": _cpu_state_dict(agent.target_net),
        "validation_point": dict(validation_point),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _save_stage08e_resume_checkpoint(
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
        "schema_version": STAGE08E_SCHEMA_VERSION_V1,
        "stage": "08E",
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
            "stage": "08E",
        },
    )


def _build_report_payload(
    *,
    generated_at_utc: datetime,
    finished_at_utc: datetime,
    run_id: str,
    run_dir: Path,
    config: RoehubNativeTrainingConfig,
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
        "adaptation_diff_from_hf_original": _adaptation_diff_payload(),
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "artifact_hashes": dict(artifact_hashes),
        "artifact_kind": STAGE08E_ROEHUB_NATIVE_RUN_KIND_V1,
        "candidate_level": STAGE08E_CANDIDATE_LEVEL_V1,
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
        "safety": _safety_payload(dataset_stage=str(dataset_dependency.get("stage", "06"))),
        "schema_version": STAGE08E_SCHEMA_VERSION_V1,
        "stage": "08E",
        "status": "completed",
        "upstream_methodology_parity": True,
        "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
        "warning_register_from_stage08d": _stage08d_warning_register_payload(),
    }
    return {**payload, "candidate_report_hash": hash_json_payload_v1(payload)}


def _build_manifest_payload(
    *,
    generated_at_utc: datetime,
    run_id: str,
    run_dir: Path,
    config: RoehubNativeTrainingConfig,
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
        "adaptation_diff_from_hf_original": _adaptation_diff_payload(),
        "architecture_id": UPSTREAM_METHODOLOGY_ARCHITECTURE_ID_V1,
        "artifact_hashes": dict(artifact_hashes),
        "artifact_kind": STAGE08E_CANDIDATE_MANIFEST_KIND_V1,
        "candidate_level": STAGE08E_CANDIDATE_LEVEL_V1,
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
            "stage08f_allowed": True,
            "stage08f_input": "roehub_native_candidate_manifest_path",
        },
        "normalization_stats_hash": normalization_stats_hash,
        "resource_summary": dict(resource_usage),
        "run_dir": str(run_dir),
        "run_id": run_id,
        "safety": _safety_payload(dataset_stage=str(dataset_dependency.get("stage", "06"))),
        "schema_version": STAGE08E_SCHEMA_VERSION_V1,
        "stage": "08E",
        "status": "completed",
        "upstream_source_sha": UPSTREAM_SOURCE_SHA_STAGE08A_V1,
        "warning_register_from_stage08d": _stage08d_warning_register_payload(),
    }
    return _finalize_manifest(payload)


def _finalize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "candidate_manifest_hash"
    }
    return {**payload, "candidate_manifest_hash": hash_json_payload_v1(payload)}


class _Stage08eProgressWriter:
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
            "artifact_kind": STAGE08E_PROGRESS_KIND_V1,
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
            "stage": "08E",
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


def _resource_usage_payload(
    *,
    agent: TorchD3qnPerAgent,
    start_usage: Any,
    end_usage: Any,
    progress_writer: _Stage08eProgressWriter,
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


def _validate_stage06_dataset_dependency(value: Mapping[str, Any]) -> None:
    if value.get("stage") not in {"06", "08J"}:
        raise RoehubNativeTrainingError(reason="unexpected_dataset_dependency_stage")
    if value.get("source_market") != "binance:futures":
        raise RoehubNativeTrainingError(reason="unexpected_dataset_dependency_market")
    if value.get("sessionized_manifest_status") != "accepted":
        raise RoehubNativeTrainingError(reason="stage06_manifest_not_accepted")
    splits = value.get("splits")
    if not isinstance(splits, Mapping) or "train" not in splits or "validation" not in splits:
        raise RoehubNativeTrainingError(reason="stage06_train_validation_splits_required")


def _adaptation_diff_payload() -> list[dict[str, object]]:
    return [
        {
            "surface": "dataset_source",
            "hf_original": "Stage 04 external HF train_data.npz and val_data.npz",
            "roehub_native": (
                "Stage 06 sessionized binance:futures train and validation split artifacts"
            ),
        },
        {
            "surface": "symbol_universe",
            "hf_original": "HF original split symbols",
            "roehub_native": "accepted Stage 06 Roehub Binance Futures USDT perpetual sessions",
        },
        {
            "surface": "normalization",
            "hf_original": "train-only stats from HF train split",
            "roehub_native": "train-only stats from Stage 06 train split only",
        },
        {
            "surface": "methodology",
            "hf_original": "upstream alpha.py CNN dueling D3QN environment rollout profile",
            "roehub_native": "same profile unless explicit CLI flags change the run config",
        },
        {
            "surface": "evaluation",
            "hf_original": "Stage 08D owns HF test/backtest evaluation",
            "roehub_native": (
                "Stage 08F owns native test/backtest evaluation; Stage 08E does not score"
            ),
        },
    ]


def _stage08d_warning_register_payload() -> list[dict[str, object]]:
    return [
        {
            "warning": "weak_untuned_hf_demo_profitability",
            "value": "candidate net PnL after costs 2064.37744919",
        },
        {
            "warning": "simple_baseline_outperformed_hf_candidate",
            "value": "simple baseline net PnL after costs 4508.37753925",
        },
        {
            "warning": "low_positive_session_ratio",
            "value": "0.0324699",
        },
        {
            "warning": "missing_optuna_tuning",
            "value": "no Optuna/tuning in Stage 08D",
        },
        {
            "warning": "demo_profile_30_10",
            "value": "agent_history_len=30 and agent_session_len=10",
        },
    ]


def _safety_payload(*, dataset_stage: str = "06") -> dict[str, object]:
    return {
        "allow_browser_runtime_verification": False,
        "allow_exchange_side_effects": False,
        "allow_mainnet_submit": False,
        "candidate_level": STAGE08E_CANDIDATE_LEVEL_V1,
        "contains_raw_provider_payloads": False,
        "contains_secrets": False,
        "hf_original_data_used": False,
        "register_promote_activate_trade": False,
        "stage06_roehub_native_data_used": dataset_stage == "06",
        "stage08j_article_selector_data_used": dataset_stage == "08J",
        "stage08f_evaluation_run": False,
    }


def _render_json_line_payload(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(_json_safe(payload), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
