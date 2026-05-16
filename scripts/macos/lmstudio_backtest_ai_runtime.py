#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from trading.contexts.backtest.adapters.outbound import (
    BacktestAiConfiguratorModelRuntimeConfig,
    load_backtest_ai_configurator_runtime_config,
)

DEFAULT_CONFIG_PATH = Path("/opt/roehub/app/configs/prod/backtest_ai_configurator.yaml")
DEFAULT_LMS_PATH = Path("/Users/daniildegtyarev/.lmstudio/bin/lms")
DEFAULT_MODEL_KEY = "gemma-4-e2b-it"
DEFAULT_ARTIFACT_PATH = Path(
    "/opt/roehub/state/backtest_ai_configurator/lmstudio_runtime_smoke.json"
)
DEFAULT_READINESS_ATTEMPTS = 6
DEFAULT_READINESS_RETRY_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class RuntimeTarget:
    base_url: str
    host: str
    port: int
    model_key: str
    model_identifier: str
    context_length: int
    parallel: int


class RuntimeCheckError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ensure or smoke the Roehub LM Studio backtest AI runtime."
    )
    parser.add_argument("command", choices=("ensure", "smoke", "status", "stop"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--lms", type=Path, default=DEFAULT_LMS_PATH)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    started = time.time()
    try:
        result = _run_command(args)
    except Exception as error:  # noqa: BLE001
        result = {
            "accepted": False,
            "blocking_reason": str(error),
            "next_prompt_allowed": False,
            "command": args.command,
            "timestamp_unix": time.time(),
            "duration_seconds": round(time.time() - started, 3),
        }
        _emit(result, json_output=args.json)
        return 1

    result.setdefault("accepted", True)
    result.setdefault("blocking_reason", None)
    result.setdefault("next_prompt_allowed", True)
    result["duration_seconds"] = round(time.time() - started, 3)
    _emit(result, json_output=args.json)
    if args.command in {"ensure", "smoke", "status"}:
        _write_artifact(result, args.artifact)
    return 0 if result["accepted"] else 1


def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    runtime_config = load_backtest_ai_configurator_runtime_config(args.config)
    target = _target_from_model_config(
        runtime_config.model,
        model_key=args.model_key,
    )
    lms = Path(args.lms)
    if args.command == "stop":
        stopped = _stop_server(lms=lms)
        return {
            "accepted": True,
            "blocking_reason": None,
            "next_prompt_allowed": True,
            "command": "stop",
            "target": _target_payload(target),
            "stop": stopped,
            "timestamp_unix": time.time(),
        }

    if args.command == "ensure":
        return _ensure_runtime(lms=lms, target=target, config_path=args.config)
    if args.command == "smoke":
        return _smoke_runtime(lms=lms, target=target, config_path=args.config)
    return _status_runtime(lms=lms, target=target, config_path=args.config)


def _ensure_runtime(
    *,
    lms: Path,
    target: RuntimeTarget,
    config_path: Path,
) -> dict[str, Any]:
    _require_lms(lms)
    before_preflight = _port_preflight(lms=lms, target=target)
    daemon_up = _run_lms_json(lms, "daemon", "up", "--json")
    server_status = _server_status(lms)
    if not _server_running_on_target(server_status, target):
        _run_lms(
            lms,
            "server",
            "start",
            "--port",
            str(target.port),
            "--bind",
            target.host,
        )
    after_preflight = _port_preflight(lms=lms, target=target)
    ps_before_load = _lms_ps(lms)
    load_result: Mapping[str, Any] | str = "already_loaded"
    if not _model_loaded(ps_before_load, target):
        load_result = _run_lms(
            lms,
            "load",
            target.model_key,
            "--identifier",
            target.model_identifier,
            "--context-length",
            str(target.context_length),
            "--parallel",
            str(target.parallel),
        ).stdout.strip()
    smoke = _smoke_runtime(lms=lms, target=target, config_path=config_path)
    return {
        "accepted": smoke["accepted"],
        "blocking_reason": smoke["blocking_reason"],
        "next_prompt_allowed": smoke["next_prompt_allowed"],
        "command": "ensure",
        "config_path": str(config_path),
        "target": _target_payload(target),
        "daemon_up": daemon_up,
        "server_status": _server_status(lms),
        "port_preflight_before": before_preflight,
        "port_preflight_after": after_preflight,
        "load_result": load_result,
        "smoke": smoke,
        "timestamp_unix": time.time(),
    }


def _smoke_runtime(
    *,
    lms: Path,
    target: RuntimeTarget,
    config_path: Path,
) -> dict[str, Any]:
    _require_lms(lms)
    last_error: RuntimeCheckError | None = None
    for attempt in range(1, DEFAULT_READINESS_ATTEMPTS + 1):
        try:
            result = _smoke_runtime_once(
                lms=lms,
                target=target,
                config_path=config_path,
            )
            result["readiness_attempts"] = attempt
            return result
        except RuntimeCheckError as error:
            if _is_non_retryable_readiness_error(error):
                raise
            last_error = error
            if attempt >= DEFAULT_READINESS_ATTEMPTS:
                break
            time.sleep(DEFAULT_READINESS_RETRY_SECONDS)
    if last_error is None:
        raise RuntimeCheckError("LM Studio runtime smoke failed without diagnostics")
    raise last_error


def _smoke_runtime_once(
    *,
    lms: Path,
    target: RuntimeTarget,
    config_path: Path,
) -> dict[str, Any]:
    port_preflight = _port_preflight(lms=lms, target=target)
    server_status = _server_status(lms)
    if not _server_running_on_target(server_status, target):
        raise RuntimeCheckError(
            f"LM Studio server is not running on configured {target.host}:{target.port}"
        )
    ps = _lms_ps(lms)
    if not _model_loaded(ps, target):
        raise RuntimeCheckError(
            f"lms ps --json does not show loaded {target.model_identifier}"
        )
    api_models = _http_json(f"{target.base_url}/api/v1/models", timeout=10.0)
    if not _api_models_has_loaded_instance(api_models, target.model_identifier):
        raise RuntimeCheckError(
            f"/api/v1/models does not show loaded instance {target.model_identifier}"
        )
    return {
        "accepted": True,
        "blocking_reason": None,
        "next_prompt_allowed": True,
        "command": "smoke",
        "config_path": str(config_path),
        "target": _target_payload(target),
        "port_preflight": port_preflight,
        "server_status": server_status,
        "lms_ps": ps,
        "api_v1_models_loaded_instance": True,
        "single_shot_chat_probe": "removed",
        "tool_agent_contract": "pending",
        "timestamp_unix": time.time(),
    }


def _is_non_retryable_readiness_error(error: RuntimeCheckError) -> bool:
    message = str(error)
    return (
        message.startswith("port preflight failed:")
        or "must not bind publicly" in message
        or "occupied by another service" in message
    )


def _status_runtime(
    *,
    lms: Path,
    target: RuntimeTarget,
    config_path: Path,
) -> dict[str, Any]:
    _require_lms(lms)
    return {
        "accepted": True,
        "blocking_reason": None,
        "next_prompt_allowed": True,
        "command": "status",
        "config_path": str(config_path),
        "target": _target_payload(target),
        "daemon_status": _run_lms_json(lms, "daemon", "status", "--json"),
        "server_status": _server_status(lms),
        "lms_ps": _lms_ps(lms),
        "port_preflight": _port_preflight(lms=lms, target=target),
        "timestamp_unix": time.time(),
    }


def _target_from_model_config(
    config: BacktestAiConfiguratorModelRuntimeConfig,
    *,
    model_key: str,
) -> RuntimeTarget:
    parsed = urlparse(config.base_url)
    if parsed.scheme != "http":
        raise RuntimeCheckError("LM Studio base_url must use http on loopback")
    host = (parsed.hostname or "").strip().lower()
    if host == "localhost":
        host = "127.0.0.1"
    if host != "127.0.0.1":
        raise RuntimeCheckError("LM Studio base_url must resolve to 127.0.0.1")
    if parsed.port is None:
        raise RuntimeCheckError("LM Studio base_url must include an explicit port")
    if parsed.path not in {"", "/"}:
        raise RuntimeCheckError("LM Studio base_url must not include a path")
    return RuntimeTarget(
        base_url=f"http://{host}:{parsed.port}",
        host=host,
        port=parsed.port,
        model_key=model_key,
        model_identifier=config.model_id,
        context_length=config.context_window_tokens,
        parallel=config.active_generations,
    )


def _port_preflight(*, lms: Path, target: RuntimeTarget) -> dict[str, Any]:
    listeners = _port_listeners(target.port)
    server_status = _server_status(lms)
    running_on_target = _server_running_on_target(server_status, target)
    if listeners and not running_on_target:
        raise RuntimeCheckError(
            "port preflight failed: configured port "
            f"{target.port} is occupied by another service: {listeners[0]}"
        )
    for listener in listeners:
        name = listener.get("name", "")
        if "0.0.0.0" in name or "*:" in name:
            raise RuntimeCheckError(
                f"port preflight failed: LM Studio must not bind publicly: {name}"
            )
    return {
        "accepted": True,
        "blocking_reason": None,
        "next_prompt_allowed": True,
        "port": target.port,
        "host": target.host,
        "listeners": listeners,
        "server_running_on_configured_port": running_on_target,
    }


def _port_listeners(port: int) -> list[dict[str, str]]:
    result = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in {0, 1}:
        raise RuntimeCheckError(
            f"port preflight failed: lsof returned {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    listeners: list[dict[str, str]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 9:
            listeners.append({"raw": line, "name": line})
            continue
        listeners.append(
            {
                "command": parts[0],
                "pid": parts[1],
                "user": parts[2],
                "name": " ".join(parts[8:]),
            }
        )
    return listeners


def _model_loaded(ps_payload: object, target: RuntimeTarget) -> bool:
    if not isinstance(ps_payload, list):
        return False
    for item in ps_payload:
        if not isinstance(item, Mapping):
            continue
        if item.get("identifier") != target.model_identifier:
            continue
        if item.get("contextLength") != target.context_length:
            continue
        if item.get("parallel") != target.parallel:
            continue
        return True
    return False


def _api_models_has_loaded_instance(payload: object, identifier: str) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key == "loaded_instances" and isinstance(value, list):
                if any(_loaded_instance_matches(item, identifier) for item in value):
                    return True
            if _api_models_has_loaded_instance(value, identifier):
                return True
    if isinstance(payload, list):
        return any(_api_models_has_loaded_instance(item, identifier) for item in payload)
    return False


def _loaded_instance_matches(item: object, identifier: str) -> bool:
    if not isinstance(item, Mapping):
        return False
    return item.get("id") == identifier or item.get("identifier") == identifier


def _server_running_on_target(payload: object, target: RuntimeTarget) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return payload.get("running") is True and payload.get("port") == target.port


def _server_status(lms: Path) -> object:
    return _run_lms_json(lms, "server", "status", "--json", "--quiet")


def _lms_ps(lms: Path) -> object:
    return _run_lms_json(lms, "ps", "--json")


def _stop_server(*, lms: Path) -> dict[str, Any]:
    _require_lms(lms)
    result = _run_lms(lms, "server", "stop")
    return {
        "stdout": result.stdout.strip(),
        "server_status": _server_status(lms),
    }


def _run_lms_json(lms: Path, *args: str) -> object:
    result = _run_lms(lms, *args)
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeCheckError(
            f"lms {' '.join(args)} did not return JSON: {result.stdout[:200]}"
        ) from error


def _run_lms(lms: Path, *args: str) -> subprocess.CompletedProcess[str]:
    command = [str(lms), *args]
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeCheckError(
            f"{' '.join(command)} failed with exit {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _require_lms(lms: Path) -> None:
    if not lms.exists():
        raise RuntimeCheckError(f"lms binary not found: {lms}")
    if not lms.is_file():
        raise RuntimeCheckError(f"lms path is not a file: {lms}")


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
    timeout: float,
) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeCheckError(
            f"{method} {url} returned HTTP {error.code}: {body[:300]}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeCheckError(f"{method} {url} failed: {error}") from error
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeCheckError(
            f"{method} {url} returned non-JSON body: {body[:300]}"
        ) from error


def _target_payload(target: RuntimeTarget) -> dict[str, Any]:
    return {
        "base_url": target.base_url,
        "host": target.host,
        "port": target.port,
        "model_key": target.model_key,
        "model_identifier": target.model_identifier,
        "context_length": target.context_length,
        "parallel": target.parallel,
    }


def _write_artifact(payload: Mapping[str, Any], artifact: Path) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _emit(payload: Mapping[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(payload, sort_keys=True))
        return
    accepted = payload.get("accepted")
    reason = payload.get("blocking_reason")
    print(f"accepted={accepted} blocking_reason={reason}")


if __name__ == "__main__":
    sys.exit(main())
