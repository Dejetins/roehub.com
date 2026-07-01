from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.rl_trading.domain import (  # noqa: E402
    FEATURE_NAMES_V1,
    HF_DATASET_REPO_ID_V1,
    STAGE07A_RUNTIME_ARTIFACT_ROOT_V1,
    HfDatasetSplitSpec,
    HfOriginalEvaluationConfig,
    HfOriginalEvaluationError,
    HfOriginalSplitData,
    QValueCache,
    UpstreamAlphaConfig,
    UpstreamTradingEnvironment,
    compute_file_sha256,
    expected_hf_dataset_manifest_hash_v1,
    expected_hf_split_specs_v1,
    hash_json_payload_v1,
    session_close_price_v1,
)
from trading.contexts.rl_trading.domain.action_state_reward_contract import (  # noqa: E402
    ACTION_NAMES_BY_ID_V1,
    RlTrainingState,
    apply_training_reward_step_v1,
)
from trading.contexts.rl_trading.domain.hf_original_evaluation import (  # noqa: E402
    _BacktestRiskManagementState,
    _alpha_config_from_training_config_payload,
    _artifact_path,
    _load_checkpoint_agent,
    _load_normalization_stats,
    _mask_q_values,
    _position_fraction_alpha,
    _risk_management_action_override_v1,
    _update_risk_management_state_after_step_v1,
)
from trading.contexts.rl_trading.domain.raw_feature_dataset import (  # noqa: E402
    render_raw_feature_json_payload_v1,
)
from trading.contexts.rl_trading.domain.upstream_methodology import (  # noqa: E402
    FilteredBacktestPolicy,
    NormalizationStats,
    TorchD3qnPerAgent,
    build_upstream_state_v1,
    valid_upstream_training_actions_v1,
)

STAGE08I_RUNTIME_ARTIFACT_SUBDIR_V1 = "stage08i_upstream_evaluator_session_parity_forensic_v1"
DEFAULT_HF_DATASET_DIR = (
    Path("/opt/roehub/state/rl_trading/hf_reproducibility/dataset")
    / "ResearchRL"
    / "open-rl-trading-binance-dataset"
)
DEFAULT_CANDIDATE_MANIFEST = (
    Path("/opt/roehub/state/rl_trading/training_runs")
    / "stage08c_original_hf_full_training_run_v1"
    / "full"
    / "stage08c_hf_original_full"
    / "hf_original_candidate_manifest.json"
)
DEFAULT_CANDIDATE_MANIFEST_SHA256 = (
    "4382389a45abff070681bf0bb07c2d3aee601d7813777675ad909d3258a9d5e8"
)
DEFAULT_OUTPUT_ROOT = (
    Path(STAGE07A_RUNTIME_ARTIFACT_ROOT_V1)
    / "evaluation_runs"
    / STAGE08I_RUNTIME_ARTIFACT_SUBDIR_V1
)
UPSTREAM_COMMIT = "f71130903f8237351164f4b875494185465bf1ea"
UPSTREAM_SOURCE_HASHES = {
    "agent.py": {"bytes": 9236, "sha256": "49ef8faaba845eb31207704fae23a73a9f784af0a4b6aef9323fd8be769e2fab"},
    "backtest_engine.py": {
        "bytes": 16710,
        "sha256": "d05e426fdad3acb24df4c74fce17536d584e56a0b9e528160c5cb9762e179892",
    },
    "config.py": {"bytes": 7191, "sha256": "65bfc4b8fa0722defe75ecf38dbb0ce92c53d5edc2e96b8b5fe0d849fc6219d6"},
    "configs/alpha.py": {
        "bytes": 2600,
        "sha256": "c8f0348379ed4deaf7dc306bbab039203e22e4039321ab294caedd2f5f698f9e",
    },
    "model.py": {"bytes": 2942, "sha256": "042f406b0c35222bb79d659883d935454b12f42f4551daa06dc95e3a08a396cc"},
    "optimize_cfg.py": {
        "bytes": 5266,
        "sha256": "f6b2c542958cdce4c1cec6096cdae619304f67740b79098e136bf8dbfbe646dd",
    },
    "trading_environment.py": {
        "bytes": 16247,
        "sha256": "c38154ee416f1fb3de59c2f7085092d0237216c7854e70ba89863d9676920c8c",
    },
    "utils.py": {"bytes": 11982, "sha256": "38d00c1bbdafa0201f219e530544c70ac47dc0d143b503b158d52c8c96db2f25"},
}
TRACE_COMPARE_FIELDS = (
    "state_hash",
    "q_values_hash",
    "raw_argmax_action",
    "masked_q_values_hash",
    "selected_action",
    "effective_action",
    "position_side",
    "entry_price",
    "pnl_change",
    "reward",
    "balance_or_equity",
    "audit_reason",
)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = run_forensic(args)
    except HfOriginalEvaluationError as exc:
        print(_render_status({"field": exc.field, "reason": exc.reason, "status": "blocked"}))
        return 2
    print(_render_status(payload))
    return 0 if payload["status"] == "accepted" else 2


def run_forensic(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    candidate_manifest_sha256 = _file_sha256_hex(args.candidate_manifest)
    if (
        args.expected_candidate_manifest_sha256
        and candidate_manifest_sha256 != args.expected_candidate_manifest_sha256
    ):
        raise HfOriginalEvaluationError(
            reason="candidate_manifest_sha256_mismatch",
            field=str(args.candidate_manifest),
        )
    candidate_manifest = _read_json(args.candidate_manifest)
    training_config = _read_json(_artifact_path(candidate_manifest, "training_config"))
    alpha = _alpha_config_from_training_config_payload(training_config)
    alpha = replace(
        alpha,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
    )
    config = HfOriginalEvaluationConfig(
        alpha=alpha,
        checkpoint_name="best",
        selection_strategy="advantage_based_filter",
        device_policy=args.device_policy,
        backtest_max_sessions=args.max_backtest_sessions,
    )
    normalization_stats = _load_normalization_stats(
        _artifact_path(candidate_manifest, "normalization_stats"),
        expected_hash=str(candidate_manifest.get("normalization_stats_hash", "")),
    )
    split = load_hf_split(
        dataset_dir=args.dataset_dir,
        split_name="backtest",
        max_sessions=args.max_backtest_sessions,
        allow_fixture_hashes=args.allow_fixture_hashes,
    )
    source_schedule = source_backtest_schedule_v1(
        split.signal_times_utc,
        max_parallel_sessions=alpha.max_parallel_sessions,
        agent_session_len=alpha.agent_session_len,
    )
    roehub_schedule = roehub_grouped_schedule_v1(
        split.signal_times_utc,
        max_parallel_sessions=alpha.max_parallel_sessions,
    )
    first_selection_diff = first_schedule_diff_v1(
        source_schedule,
        roehub_schedule,
        compare_limit=args.compare_session_count,
    )
    trace_indices = tuple(row["session_idx"] for row in source_schedule[: args.trace_session_count])
    source_agent, _ = _load_checkpoint_agent(
        candidate_manifest=candidate_manifest,
        checkpoint_name="best",
        config=config,
    )
    roehub_agent, _ = _load_checkpoint_agent(
        candidate_manifest=candidate_manifest,
        checkpoint_name="best",
        config=config,
    )
    source_trace = build_source_backtest_trace_v1(
        split=split,
        session_indices=trace_indices,
        normalization_stats=normalization_stats,
        agent=source_agent,
        alpha=alpha,
    )
    roehub_trace = build_roehub_backtest_trace_v1(
        split=split,
        session_indices=trace_indices,
        normalization_stats=normalization_stats,
        agent=roehub_agent,
        alpha=alpha,
    )
    first_step_diff = first_trace_diff_v1(source_trace, roehub_trace)
    material_diff = first_selection_diff or first_step_diff
    status = "blocked" if material_diff is not None else "accepted"
    run_id = args.run_id or default_run_id_v1(
        candidate_manifest_sha256=candidate_manifest_sha256,
        split_payload=split.source_payload,
        trace_indices=trace_indices,
        alpha=alpha,
    )
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_payload = {
        "first_selection_diff": first_selection_diff,
        "roehub_first_selected": roehub_schedule[: args.compare_session_count],
        "source_first_selected": source_schedule[: args.compare_session_count],
    }
    first_diff_payload = {
        "first_material_diff": material_diff,
        "first_selection_diff": first_selection_diff,
        "first_step_diff": first_step_diff,
        "status": status,
    }
    source_trace_path = run_dir / "source_derived_upstream_trace.jsonl"
    roehub_trace_path = run_dir / "roehub_current_trace.jsonl"
    selection_path = run_dir / "selection_comparison.json"
    first_diff_path = run_dir / "first_material_diff.json"
    _write_jsonl(source_trace_path, source_trace)
    _write_jsonl(roehub_trace_path, roehub_trace)
    _write_json(selection_path, selection_payload)
    _write_json(first_diff_path, first_diff_payload)
    artifact_hashes = {
        "first_material_diff": _file_payload(first_diff_path),
        "roehub_current_trace": _file_payload(roehub_trace_path),
        "selection_comparison": _file_payload(selection_path),
        "source_derived_upstream_trace": _file_payload(source_trace_path),
    }
    manifest = {
        "artifact_hashes": artifact_hashes,
        "candidate_manifest_path": str(args.candidate_manifest),
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "config_hash": config.config_hash(),
        "dataset": dict(split.source_payload),
        "duration_seconds": _round(time.perf_counter() - started),
        "first_material_diff": material_diff,
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "normalization_stats_hash": normalization_stats.stats_hash(),
        "proof_boundary": "target_host_non_production_forensic_pre_main",
        "run_dir": str(run_dir),
        "run_id": run_id,
        "schema_version": 1,
        "source_execution_mode": "source_derived_from_pinned_backtest_engine_without_external_repo_checkout",
        "stage": "08I",
        "status": status,
        "trace_session_count": len(trace_indices),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_hashes": UPSTREAM_SOURCE_HASHES,
    }
    manifest = {**manifest, "manifest_hash": hash_json_payload_v1(manifest)}
    manifest_path = run_dir / "stage08i_trace_manifest.json"
    manifest = {**manifest, "manifest_path": str(manifest_path)}
    _write_json(manifest_path, manifest)
    return {
        "first_material_diff": material_diff,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _file_sha256_hex(manifest_path),
        "run_dir": str(run_dir),
        "run_id": run_id,
        "status": status,
    }


def load_hf_split(
    *,
    dataset_dir: Path,
    split_name: str,
    max_sessions: int | None,
    allow_fixture_hashes: bool,
) -> HfOriginalSplitData:
    specs = {spec.split_name: spec for spec in expected_hf_split_specs_v1()}
    split_spec = specs[split_name]
    return _load_hf_split_by_spec(
        dataset_dir=dataset_dir,
        split_spec=split_spec,
        max_sessions=max_sessions,
        allow_fixture_hashes=allow_fixture_hashes,
    )


def source_backtest_schedule_v1(
    signal_times: Sequence[str | None],
    *,
    max_parallel_sessions: int,
    agent_session_len: int,
) -> list[dict[str, Any]]:
    groups: dict[datetime, list[int]] = {}
    for idx, value in enumerate(signal_times):
        signal_dt = _parse_signal_time(value, fallback_idx=idx)
        groups.setdefault(signal_dt, []).append(idx)
    selected: list[dict[str, Any]] = []
    open_sessions: list[dict[str, datetime]] = []
    for signal_dt in sorted(groups):
        open_sessions = [row for row in open_sessions if row["end_time"] > signal_dt]
        free_slots = max_parallel_sessions - len(open_sessions)
        if free_slots <= 0:
            continue
        for session_idx in groups[signal_dt][:free_slots]:
            selected.append(
                {
                    "selected_order": len(selected),
                    "session_idx": session_idx,
                    "signal_time": _format_dt(signal_dt),
                    "source_rule": "rolling_open_sessions_cap",
                }
            )
            open_sessions.append({"end_time": signal_dt + timedelta(minutes=agent_session_len)})
    return selected


def roehub_grouped_schedule_v1(
    signal_times: Sequence[str | None],
    *,
    max_parallel_sessions: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for idx, signal_time in enumerate(signal_times):
        key = signal_time if signal_time is not None else f"__missing_signal_time_{idx}"
        groups.setdefault(str(key), []).append(idx)
    selected: list[dict[str, Any]] = []
    for key in sorted(groups):
        for session_idx in groups[key][:max_parallel_sessions]:
            selected.append(
                {
                    "selected_order": len(selected),
                    "session_idx": session_idx,
                    "signal_time": key,
                    "source_rule": "exact_signal_time_group_cap",
                }
            )
    return selected


def first_schedule_diff_v1(
    source_schedule: Sequence[Mapping[str, Any]],
    roehub_schedule: Sequence[Mapping[str, Any]],
    *,
    compare_limit: int,
) -> dict[str, Any] | None:
    limit = min(compare_limit, len(source_schedule), len(roehub_schedule))
    for idx in range(limit):
        source_idx = int(source_schedule[idx]["session_idx"])
        roehub_idx = int(roehub_schedule[idx]["session_idx"])
        if source_idx != roehub_idx:
            return {
                "diff_type": "session_selection_order",
                "material": True,
                "reason": "upstream_uses_rolling_open_sessions_but_roehub_caps_only_exact_signal_time_groups",
                "selected_order": idx,
                "source": dict(source_schedule[idx]),
                "roehub": dict(roehub_schedule[idx]),
            }
    if len(source_schedule) != len(roehub_schedule):
        return {
            "diff_type": "session_selection_count",
            "material": True,
            "reason": "selected_session_count_differs",
            "roehub_count": len(roehub_schedule),
            "source_count": len(source_schedule),
        }
    return None


def first_trace_diff_v1(
    source_trace: Sequence[Mapping[str, Any]],
    roehub_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    limit = min(len(source_trace), len(roehub_trace))
    for idx in range(limit):
        source = source_trace[idx]
        roehub = roehub_trace[idx]
        for field in TRACE_COMPARE_FIELDS:
            if source.get(field) != roehub.get(field):
                return {
                    "diff_type": "step_trace_field",
                    "field": field,
                    "material": field not in {"reward", "audit_reason"},
                    "roehub": _trace_identity(roehub, field),
                    "source": _trace_identity(source, field),
                    "trace_row": idx,
                }
    if len(source_trace) != len(roehub_trace):
        return {
            "diff_type": "step_trace_length",
            "material": True,
            "roehub_count": len(roehub_trace),
            "source_count": len(source_trace),
        }
    return None


def build_source_backtest_trace_v1(
    *,
    split: HfOriginalSplitData,
    session_indices: Sequence[int],
    normalization_stats: NormalizationStats,
    agent: TorchD3qnPerAgent,
    alpha: UpstreamAlphaConfig,
) -> list[dict[str, Any]]:
    agent.q_value_cache = QValueCache()
    filter_policy = FilteredBacktestPolicy.from_config(alpha)
    global_balance = float(alpha.initial_balance)
    rows: list[dict[str, Any]] = []
    for selected_order, session_idx in enumerate(session_indices):
        session = split.sequences[session_idx]
        symbol = split.symbols[session_idx]
        signal_dt = _parse_signal_time(split.signal_times_utc[session_idx], fallback_idx=session_idx)
        position_size = global_balance * alpha.position_fraction
        session_alpha = replace(alpha, initial_balance=position_size)
        state = RlTrainingState(balance=position_size)
        action_history: list[int | None] = [None] * alpha.action_history_len
        trade_realized_since_open = 0.0
        risk_state = _BacktestRiskManagementState()
        for step_idx in range(alpha.agent_session_len):
            obs = build_upstream_state_v1(
                session=session,
                step_idx=step_idx,
                action_history=action_history,
                training_state=state,
                normalization_stats=normalization_stats,
                config=session_alpha,
            )
            q_values = agent.q_value_cache.get_or_compute(
                (symbol, signal_dt + timedelta(minutes=step_idx)),
                lambda obs=obs: agent.predict_q_values(obs),
            )
            raw_argmax_action = int(np.argmax(q_values))
            masked_q_values = _mask_q_values(
                q_values,
                valid_actions=valid_upstream_training_actions_v1(
                    position_side=state.position_side,
                    is_last_step=step_idx == alpha.agent_session_len - 1,
                ),
            )
            decision = filter_policy.select_from_q_values(q_values)
            forced_action, risk_reason = _risk_management_action_override_v1(
                state=state,
                session=session,
                step_idx=step_idx,
                config=session_alpha,
                risk_state=risk_state,
            )
            action_for_environment = (
                decision.effective_action_id if forced_action is None else forced_action
            )
            price = session_close_price_v1(session, step_idx=step_idx, config=session_alpha)
            before = state
            result = apply_training_reward_step_v1(
                state=state,
                action_id=action_for_environment,
                price=price,
                initial_balance=session_alpha.initial_balance,
                slippage=session_alpha.slippage,
                transaction_fee=session_alpha.transaction_fee,
                inaction_penalty_ratio=session_alpha.inaction_penalty_ratio,
                is_last_step=step_idx == alpha.agent_session_len - 1,
            )
            state = result.state
            _update_risk_management_state_after_step_v1(
                risk_state=risk_state,
                state_before=before,
                state_after=state,
                closed_position=result.closed_position,
                config=session_alpha,
            )
            action_history = [*action_history[1:], result.effective_action_id]
            trade_realized_since_open += float(result.pnl_change)
            if result.closed_position:
                global_balance += trade_realized_since_open
                trade_realized_since_open = 0.0
            rows.append(
                _trace_row(
                    implementation="upstream_source_derived",
                    selected_order=selected_order,
                    session_idx=int(session_idx),
                    symbol=symbol,
                    signal_time=_format_dt(signal_dt),
                    step_idx=step_idx,
                    price=price,
                    state_hash=_hash_array(obs),
                    q_values_hash=_hash_array(q_values),
                    raw_argmax_action=raw_argmax_action,
                    masked_q_values_hash=_hash_array(masked_q_values),
                    selected_action=decision.effective_action_id,
                    effective_action=result.effective_action_id,
                    position_side=state.position_side,
                    entry_price=state.entry_price,
                    pnl_change=result.pnl_change,
                    reward=0.0,
                    balance_or_equity=global_balance,
                    audit_reason=risk_reason or result.audit_reason,
                )
            )
    return rows


def build_roehub_backtest_trace_v1(
    *,
    split: HfOriginalSplitData,
    session_indices: Sequence[int],
    normalization_stats: NormalizationStats,
    agent: TorchD3qnPerAgent,
    alpha: UpstreamAlphaConfig,
) -> list[dict[str, Any]]:
    agent.q_value_cache = QValueCache()
    backtest_alpha = _position_fraction_alpha(alpha)
    filter_policy = FilteredBacktestPolicy.from_config(alpha)
    sequences = split.sequences[np.asarray(session_indices, dtype=np.int64)]
    environment = UpstreamTradingEnvironment(
        sequences=sequences,
        normalization_stats=normalization_stats,
        config=backtest_alpha,
    )
    equity = alpha.initial_balance * float(len(session_indices))
    rows: list[dict[str, Any]] = []
    for selected_order, session_idx in enumerate(session_indices):
        state_obs, _ = environment.reset(forced_index=selected_order)
        done = False
        risk_state = _BacktestRiskManagementState()
        while not done:
            step_idx = environment.step_idx
            symbol = split.symbols[session_idx]
            signal_time = split.signal_times_utc[session_idx]
            q_values = agent.q_value_cache.get_or_compute(
                (
                    symbol,
                    signal_time,
                    step_idx,
                    environment.state.position_side,
                    tuple(environment.action_history),
                ),
                lambda state_obs=state_obs: agent.predict_q_values(state_obs),
            )
            raw_argmax_action = int(np.argmax(q_values))
            masked_q_values = _mask_q_values(q_values, valid_actions=environment.valid_actions())
            decision = filter_policy.select_from_q_values(masked_q_values)
            before = environment.state
            forced_action, risk_reason = _risk_management_action_override_v1(
                state=before,
                session=environment.sequences[selected_order],
                step_idx=step_idx,
                config=backtest_alpha,
                risk_state=risk_state,
            )
            action_for_environment = (
                decision.effective_action_id if forced_action is None else forced_action
            )
            price = session_close_price_v1(
                environment.sequences[selected_order],
                step_idx=step_idx,
                config=backtest_alpha,
            )
            next_state, reward, done, _, info = environment.step(action_for_environment)
            pnl_change = float(info["pnl_change"])
            equity += pnl_change
            _update_risk_management_state_after_step_v1(
                risk_state=risk_state,
                state_before=before,
                state_after=environment.state,
                closed_position=bool(info.get("closed_position", False)),
                config=backtest_alpha,
            )
            rows.append(
                _trace_row(
                    implementation="roehub_current",
                    selected_order=selected_order,
                    session_idx=int(session_idx),
                    symbol=symbol,
                    signal_time=signal_time,
                    step_idx=step_idx,
                    price=price,
                    state_hash=_hash_array(state_obs),
                    q_values_hash=_hash_array(q_values),
                    raw_argmax_action=raw_argmax_action,
                    masked_q_values_hash=_hash_array(masked_q_values),
                    selected_action=decision.effective_action_id,
                    effective_action=int(info["effective_action_id"]),
                    position_side=environment.state.position_side,
                    entry_price=environment.state.entry_price,
                    pnl_change=pnl_change,
                    reward=float(reward),
                    balance_or_equity=equity,
                    audit_reason=risk_reason or str(info["audit_reason"]),
                )
            )
            state_obs = next_state
    return rows


def default_run_id_v1(
    *,
    candidate_manifest_sha256: str,
    split_payload: Mapping[str, Any],
    trace_indices: Sequence[int],
    alpha: UpstreamAlphaConfig,
) -> str:
    digest = hash_json_payload_v1(
        {
            "candidate_manifest_sha256": candidate_manifest_sha256,
            "split": dict(split_payload),
            "stage": "08I",
            "trace_indices": list(trace_indices),
            "upstream_commit": UPSTREAM_COMMIT,
            "alpha_config_hash": alpha.config_hash(),
        }
    )
    return f"stage08i_forensic_{digest[:20]}"


def _load_hf_split_by_spec(
    *,
    dataset_dir: Path,
    split_spec: HfDatasetSplitSpec,
    max_sessions: int | None,
    allow_fixture_hashes: bool,
) -> HfOriginalSplitData:
    file_path = dataset_dir / split_spec.file_name
    if not file_path.exists():
        raise HfOriginalEvaluationError(reason="missing_hf_split_file", field=str(file_path))
    sha256 = compute_file_sha256(file_path)
    if sha256 != split_spec.expected_sha256 and not allow_fixture_hashes:
        raise HfOriginalEvaluationError(reason="hf_split_hash_mismatch", field=split_spec.file_name)
    with np.load(file_path, allow_pickle=True) as archive:
        keys = sorted(
            (key for key in archive.files if key.startswith("fetcher_")),
            key=_fetcher_key_sort_value,
        )
        if sha256 == split_spec.expected_sha256 and len(keys) != split_spec.observed_sessions:
            raise HfOriginalEvaluationError(
                reason="hf_split_session_count_mismatch",
                field=split_spec.file_name,
            )
        selected_keys = keys if max_sessions is None else keys[:max_sessions]
        key_map = _keys_map(archive)
        features = np.empty((len(selected_keys), 150, len(FEATURE_NAMES_V1)), dtype=np.float32)
        symbols: list[str] = []
        signal_times: list[str | None] = []
        for row_idx, key in enumerate(selected_keys):
            features[row_idx] = np.asarray(archive[key], dtype=np.float32)
            symbol, signal_time = _metadata_for_key(key_map, key)
            symbols.append(symbol)
            signal_times.append(signal_time)
    return HfOriginalSplitData(
        split_name=split_spec.split_name,
        sequences=np.ascontiguousarray(features, dtype=np.float32),
        symbols=tuple(symbols),
        signal_times_utc=tuple(signal_times),
        source_payload={
            "allow_fixture_hashes": bool(allow_fixture_hashes),
            "dataset_dir": str(dataset_dir),
            "dataset_manifest_hash": expected_hf_dataset_manifest_hash_v1(),
            "dataset_repo_id": HF_DATASET_REPO_ID_V1,
            "expected_sha256": split_spec.expected_sha256,
            "file_name": split_spec.file_name,
            "file_path": str(file_path),
            "hash_matches_expected": sha256 == split_spec.expected_sha256,
            "selected_session_count": len(selected_keys),
            "sha256": sha256,
            "split_name": split_spec.split_name,
            "total_session_count": len(keys),
        },
    )


def _trace_row(
    *,
    implementation: str,
    selected_order: int,
    session_idx: int,
    symbol: str,
    signal_time: str | None,
    step_idx: int,
    price: float,
    state_hash: str,
    q_values_hash: str,
    raw_argmax_action: int,
    masked_q_values_hash: str,
    selected_action: int,
    effective_action: int,
    position_side: str | None,
    entry_price: float | None,
    pnl_change: float,
    reward: float,
    balance_or_equity: float,
    audit_reason: str,
) -> dict[str, Any]:
    return {
        "audit_reason": audit_reason,
        "balance_or_equity": _round(balance_or_equity),
        "effective_action": effective_action,
        "effective_action_name": ACTION_NAMES_BY_ID_V1[effective_action],
        "entry_price": None if entry_price is None else _round(entry_price),
        "implementation": implementation,
        "masked_q_values_hash": masked_q_values_hash,
        "pnl_change": _round(pnl_change),
        "position_side": position_side,
        "price": _round(price),
        "q_values_hash": q_values_hash,
        "raw_argmax_action": raw_argmax_action,
        "raw_argmax_action_name": ACTION_NAMES_BY_ID_V1[raw_argmax_action],
        "reward": _round(reward),
        "selected_action": selected_action,
        "selected_action_name": ACTION_NAMES_BY_ID_V1[selected_action],
        "selected_order": selected_order,
        "session_idx": session_idx,
        "signal_time": signal_time,
        "state_hash": state_hash,
        "step_idx": step_idx,
        "symbol": symbol,
    }


def _trace_identity(row: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {
        "audit_reason": row.get("audit_reason"),
        "field": field,
        "field_value": row.get(field),
        "implementation": row.get("implementation"),
        "selected_order": row.get("selected_order"),
        "session_idx": row.get("session_idx"),
        "signal_time": row.get("signal_time"),
        "step_idx": row.get("step_idx"),
        "symbol": row.get("symbol"),
    }


def _metadata_for_key(key_map: Mapping[str, Any], key: str) -> tuple[str, str | None]:
    value = key_map.get(key)
    if isinstance(value, tuple) and len(value) >= 2:
        symbol = str(value[0]).upper()
        signal_time = value[1]
        if hasattr(signal_time, "astimezone"):
            text = signal_time.astimezone(UTC).replace(microsecond=0).isoformat()
            return symbol, text.replace("+00:00", "Z")
        return symbol, str(signal_time)
    return "UNKNOWN", None


def _keys_map(archive: Any) -> Mapping[str, Any]:
    if "_keys_map_" not in archive.files:
        return {}
    value = archive["_keys_map_"]
    try:
        item = value.item()
    except Exception:
        return {}
    return item if isinstance(item, Mapping) else {}


def _fetcher_key_sort_value(value: str) -> tuple[int, str]:
    try:
        return int(value.split("_", 1)[1]), value
    except Exception:
        return sys.maxsize, value


def _parse_signal_time(value: str | None, *, fallback_idx: int) -> datetime:
    if value is None:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(minutes=fallback_idx)
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(text).astimezone(UTC).replace(microsecond=0)


def _format_dt(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _hash_array(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _file_sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_payload(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "path": str(path), "sha256": _file_sha256_hex(path)}


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(render_raw_feature_json_payload_v1(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _round(value: float) -> float:
    return round(float(value), 10)


def _render_status(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage 08I upstream evaluator/session parity forensic trace."
    )
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument(
        "--expected-candidate-manifest-sha256",
        type=str,
        default=DEFAULT_CANDIDATE_MANIFEST_SHA256,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_HF_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--allow-fixture-hashes", action="store_true")
    parser.add_argument("--max-backtest-sessions", type=int, default=None)
    parser.add_argument("--trace-session-count", type=int, default=20)
    parser.add_argument("--compare-session-count", type=int, default=50)
    parser.add_argument(
        "--device-policy",
        choices=("cpu_only_deterministic", "mps_preferred_cpu_fallback"),
        default="cpu_only_deterministic",
    )
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
