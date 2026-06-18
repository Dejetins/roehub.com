from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ARTIFACT_ROOT = Path("/opt/roehub/state/rl_trading")


@dataclass(frozen=True)
class SmokeConfig:
    matrix_size: int
    matmul_iterations: int
    train_steps: int
    torch_num_threads: int
    torch_num_interop_threads: int
    artifact_root: Path
    output_json: Path | None


def _parse_args() -> SmokeConfig:
    parser = argparse.ArgumentParser(
        description="Run a sanitized PyTorch CPU/MPS smoke for Roehub RL Stage 03."
    )
    parser.add_argument("--matrix-size", type=int, default=512)
    parser.add_argument("--matmul-iterations", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=12)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=int(os.environ.get("ROEHUB_RL_TORCH_NUM_THREADS", "4")),
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        default=int(os.environ.get("ROEHUB_RL_TORCH_NUM_INTEROP_THREADS", "1")),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(os.environ.get("ROEHUB_RL_ARTIFACT_ROOT", str(DEFAULT_ARTIFACT_ROOT))),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    return SmokeConfig(
        matrix_size=args.matrix_size,
        matmul_iterations=args.matmul_iterations,
        train_steps=args.train_steps,
        torch_num_threads=args.torch_num_threads,
        torch_num_interop_threads=args.torch_num_interop_threads,
        artifact_root=args.artifact_root,
        output_json=args.output_json,
    )


def _rss_mb() -> float | None:
    try:
        output = subprocess.check_output(
            ["/bin/ps", "-o", "rss=", "-p", str(os.getpid())],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return round(int(output.strip()) / 1024, 3)
    except Exception:
        return None


def _process_thread_count() -> int | None:
    if platform.system() == "Darwin":
        try:
            output = subprocess.check_output(
                ["/bin/ps", "-M", str(os.getpid())],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            return max(len([line for line in output.splitlines() if line.strip()]) - 1, 0)
        except Exception:
            return None
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/task"))
    except Exception:
        return None


def _synchronize_if_needed(torch: Any, device_type: str) -> None:
    if device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _run_device_smoke(torch: Any, device_type: str, config: SmokeConfig) -> dict[str, Any]:
    device = torch.device(device_type)
    torch.manual_seed(17)
    start_usage = resource.getrusage(resource.RUSAGE_SELF)
    start_wall = time.perf_counter()

    x = torch.randn((config.matrix_size, config.matrix_size), device=device)
    weight = torch.randn((config.matrix_size, config.matrix_size), device=device)
    _synchronize_if_needed(torch, device_type)

    matmul_start = time.perf_counter()
    for _ in range(config.matmul_iterations):
        x = torch.relu(x @ weight)
    _synchronize_if_needed(torch, device_type)
    matmul_wall = time.perf_counter() - matmul_start

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 4),
    ).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    inputs = torch.randn((128, 32), device=device)
    targets = torch.randn((128, 4), device=device)
    loss_value = 0.0

    train_start = time.perf_counter()
    for _ in range(config.train_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
    _synchronize_if_needed(torch, device_type)
    train_wall = time.perf_counter() - train_start

    end_usage = resource.getrusage(resource.RUSAGE_SELF)
    wall = time.perf_counter() - start_wall
    return {
        "device": device_type,
        "ok": True,
        "matrix_size": config.matrix_size,
        "matmul_iterations": config.matmul_iterations,
        "train_steps": config.train_steps,
        "matmul_wall_seconds": round(matmul_wall, 6),
        "train_wall_seconds": round(train_wall, 6),
        "total_wall_seconds": round(wall, 6),
        "cpu_user_seconds_delta": round(end_usage.ru_utime - start_usage.ru_utime, 6),
        "cpu_system_seconds_delta": round(end_usage.ru_stime - start_usage.ru_stime, 6),
        "rss_mb_after": _rss_mb(),
        "process_threads_observed": _process_thread_count(),
        "final_loss": round(loss_value, 8),
    }


def main() -> int:
    config = _parse_args()
    config.artifact_root.mkdir(parents=True, exist_ok=True)

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        print(f"PyTorch import failed: {exc}", file=sys.stderr)
        return 2

    torch.set_num_threads(config.torch_num_threads)
    torch.set_num_interop_threads(config.torch_num_interop_threads)

    mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    devices = ["cpu"]
    if mps_available:
        devices.append("mps")

    results = {
        "schema": "roehub.rl_trading.torch_device_smoke.v1",
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "dependency_isolation": {
            "extra": "rl-ml",
            "default_api_runtime_requires_torch": False,
        },
        "runtime_paths": {
            "artifact_root": str(config.artifact_root),
            "large_artifacts_committed_to_git": False,
        },
        "torch": {
            "version": torch.__version__,
            "mps_built": mps_built,
            "mps_available": mps_available,
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        },
        "accepted_device_policy": "mps_preferred_cpu_fallback"
        if mps_available
        else "cpu_fallback_mps_unavailable",
        "smokes": [_run_device_smoke(torch, device_type, config) for device_type in devices],
    }

    payload = json.dumps(results, indent=2, sort_keys=True)
    if config.output_json is not None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
