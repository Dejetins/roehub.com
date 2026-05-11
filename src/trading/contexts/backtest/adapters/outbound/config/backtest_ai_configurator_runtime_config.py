from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from trading.contexts.backtest.application.ai_configurator import (
    BacktestAiQuotaConfig,
    BacktestAiTierQuota,
)

_ENV_NAME_KEY = "ROEHUB_ENV"
_ALLOWED_ENVS = ("dev", "prod", "test")
_CONFIG_PATH_KEY = "ROEHUB_BACKTEST_AI_CONFIGURATOR_CONFIG"
_CONFIG_VERSION = 1


@dataclass(frozen=True, slots=True)
class BacktestAiConfiguratorQueueRuntimeConfig:
    max_queue_size: int
    lease_seconds: int
    job_timeout_seconds: int
    repair_attempts: int
    max_active_generations: int
    request_timeout_sec: int
    queue_timeout_sec: int
    estimated_wait_seconds: int = 8

    def __post_init__(self) -> None:
        for field_name, value in (
            ("max_queue_size", self.max_queue_size),
            ("lease_seconds", self.lease_seconds),
            ("job_timeout_seconds", self.job_timeout_seconds),
            ("repair_attempts", self.repair_attempts),
            ("max_active_generations", self.max_active_generations),
            ("request_timeout_sec", self.request_timeout_sec),
            ("queue_timeout_sec", self.queue_timeout_sec),
            ("estimated_wait_seconds", self.estimated_wait_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class BacktestAiConfiguratorRuntimeConfig:
    enabled: bool
    queue: BacktestAiConfiguratorQueueRuntimeConfig
    tier_quotas: Mapping[str, BacktestAiTierQuota]

    def to_quota_config(self) -> BacktestAiQuotaConfig:
        return BacktestAiQuotaConfig(
            tier_quotas=self.tier_quotas,
            max_queue_size=self.queue.max_queue_size,
            estimated_wait_seconds=self.queue.estimated_wait_seconds,
        )


def resolve_backtest_ai_configurator_config_path(*, environ: Mapping[str, str]) -> Path:
    raw_path = environ.get(_CONFIG_PATH_KEY, "").strip()
    if raw_path:
        return Path(raw_path)
    env_name = _resolve_env_name(environ=environ)
    return Path("configs") / env_name / "backtest_ai_configurator.yaml"


def load_backtest_ai_configurator_runtime_config(
    config_path: str | Path,
) -> BacktestAiConfiguratorRuntimeConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"backtest AI configurator config not found: {path}")
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, Mapping):
        raise ValueError("backtest AI configurator config must be a mapping")
    version = raw_payload.get("version")
    if version != _CONFIG_VERSION:
        raise ValueError(
            f"backtest AI configurator config version must be {_CONFIG_VERSION}"
        )
    root = _required_mapping(raw_payload, "backtest_ai_configurator")
    queue = _queue_config(_required_mapping(root, "queue"))
    quotas_payload = _required_mapping(root, "quotas")
    return BacktestAiConfiguratorRuntimeConfig(
        enabled=_required_bool(root, "enabled"),
        queue=queue,
        tier_quotas=_tier_quotas(quotas_payload),
    )


def _resolve_env_name(*, environ: Mapping[str, str]) -> str:
    raw_value = environ.get(_ENV_NAME_KEY, "").strip().lower()
    if raw_value not in _ALLOWED_ENVS:
        raise ValueError(f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}; got {raw_value!r}")
    return raw_value


def _queue_config(payload: Mapping[str, Any]) -> BacktestAiConfiguratorQueueRuntimeConfig:
    return BacktestAiConfiguratorQueueRuntimeConfig(
        max_queue_size=_required_int(payload, "max_queue_size"),
        lease_seconds=_required_int(payload, "lease_seconds"),
        job_timeout_seconds=_required_int(payload, "job_timeout_seconds"),
        repair_attempts=_required_int(payload, "repair_attempts"),
        max_active_generations=_required_int(payload, "max_active_generations"),
        request_timeout_sec=_required_int(payload, "request_timeout_sec"),
        queue_timeout_sec=_required_int(payload, "queue_timeout_sec"),
        estimated_wait_seconds=_optional_int(payload, "estimated_wait_seconds", default=8),
    )


def _tier_quotas(payload: Mapping[str, Any]) -> Mapping[str, BacktestAiTierQuota]:
    quotas: dict[str, BacktestAiTierQuota] = {}
    for tier in ("free", "base", "pro", "ultra"):
        tier_payload = _required_mapping(payload, tier)
        quotas[tier] = BacktestAiTierQuota(
            requests_per_5h=_required_int(tier_payload, "requests_per_5h"),
            requests_per_week=_required_int(tier_payload, "requests_per_week"),
            max_queued_per_user=_required_int(tier_payload, "max_queued_per_user"),
            max_active_user_jobs=_required_int(tier_payload, "max_active_user_jobs"),
        )
    return quotas


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"backtest AI configurator config field {key!r} must be mapping")
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"backtest AI configurator config field {key!r} must be boolean")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"backtest AI configurator config field {key!r} must be integer")
    return value


def _optional_int(payload: Mapping[str, Any], key: str, *, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"backtest AI configurator config field {key!r} must be integer")
    return value


__all__ = [
    "BacktestAiConfiguratorQueueRuntimeConfig",
    "BacktestAiConfiguratorRuntimeConfig",
    "load_backtest_ai_configurator_runtime_config",
    "resolve_backtest_ai_configurator_config_path",
]
