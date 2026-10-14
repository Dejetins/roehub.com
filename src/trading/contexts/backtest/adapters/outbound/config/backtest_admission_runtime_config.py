from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from trading.contexts.backtest.application.services.v2 import (
    BacktestAdmissionConfig,
    BacktestResourcePolicy,
)

_ENV_NAME_KEY = "ROEHUB_ENV"
_ALLOWED_ENVS = ("dev", "prod", "test")
_CONFIG_PATH_KEY = "ROEHUB_BACKTEST_ADMISSION_CONFIG"
_CONFIG_VERSION = 2


def resolve_backtest_admission_config_path(*, environ: Mapping[str, str]) -> Path:
    raw_path = environ.get(_CONFIG_PATH_KEY, "").strip()
    if raw_path:
        return Path(raw_path)
    env_name = _resolve_env_name(environ=environ)
    return Path("configs") / env_name / "backtest_admission.yaml"


def load_backtest_admission_config(config_path: str | Path) -> BacktestAdmissionConfig:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"backtest admission config not found: {path}")
    raw_payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, Mapping):
        raise ValueError("backtest admission config must be a mapping")
    version = raw_payload.get("version")
    if version != _CONFIG_VERSION:
        raise ValueError(f"backtest admission config version must be {_CONFIG_VERSION}")
    root = _required_mapping(raw_payload, "backtest_admission")
    global_payload = _required_mapping(root, "global")
    resource_payload = _required_mapping(root, "resources")
    return BacktestAdmissionConfig(
        max_active_full_jobs_global=_required_int(
            global_payload,
            "max_active_full_jobs",
        ),
        max_active_lazy_detail_tasks_global=_required_int(
            global_payload,
            "max_active_lazy_detail_tasks",
        ),
        retry_after_seconds=_required_int(root, "retry_after_seconds"),
        resource_policy=BacktestResourcePolicy(**resource_payload),
    )


def _resolve_env_name(*, environ: Mapping[str, str]) -> str:
    raw_value = environ.get(_ENV_NAME_KEY, "").strip().lower()
    if raw_value not in _ALLOWED_ENVS:
        raise ValueError(f"{_ENV_NAME_KEY} must be one of {_ALLOWED_ENVS}; got {raw_value!r}")
    return raw_value



def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"backtest admission config field {key!r} must be mapping")
    return value


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"backtest admission config field {key!r} must be integer")
    return value



__all__ = [
    "load_backtest_admission_config",
    "resolve_backtest_admission_config_path",
]
