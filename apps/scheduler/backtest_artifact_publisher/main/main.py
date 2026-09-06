from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal

from apps.scheduler.backtest_artifact_publisher.wiring.modules import (
    build_backtest_artifact_publisher_app,
)


def _configure_logging() -> None:
    """
    Configure process-wide logging defaults for the artifact publisher scheduler process.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Logging is configured exactly once at process startup.
    Raises:
        None.
    Side Effects:
        Configures the root logging handler and message format.
    Docs:
      - docs/runbooks/mac-studio-native-backend-operations.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/main/main.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the backtest artifact publisher scheduler process.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured parser instance.
    Assumptions:
        Defaults target local development wiring while launchd passes explicit prod/test paths.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/runbooks/mac-studio-native-backend-operations.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - configs/installation/runtime-service-manifest.json
    """
    parser = argparse.ArgumentParser(prog="backtest-artifact-publisher")
    parser.add_argument(
        "--config",
        default="configs/dev/backtest_artifacts.yaml",
        help="Path to backtest_artifacts.yaml",
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9203,
        help="Prometheus metrics HTTP port",
    )
    parser.add_argument(
        "--lock-path",
        default=None,
        help="Optional host-level lock file path override",
    )
    return parser


def _install_signal_handlers(stop_event: asyncio.Event) -> None:
    """
    Install SIGTERM and SIGINT handlers that trigger cooperative shutdown.

    Args:
        stop_event: Shared shutdown event used by the scheduler runtime.
    Returns:
        None.
    Assumptions:
        Called from the main asyncio event loop.
    Raises:
        None.
    Side Effects:
        Registers process signal handlers.
    Docs:
      - docs/runbooks/mac-studio-native-backend-operations.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/main/main.py
    """
    loop = asyncio.get_running_loop()

    def _mark_stop() -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _mark_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_args: _mark_stop())


async def _run_async(
    *,
    config_path: str,
    metrics_port: int,
    lock_path: str | None,
) -> int:
    """
    Build and run the long-lived scheduler runtime until termination is requested.

    Args:
        config_path: Runtime artifact config path.
        metrics_port: HTTP port for the Prometheus `/metrics` endpoint.
        lock_path: Optional host-level lock file path override.
    Returns:
        int: Process exit code.
    Assumptions:
        ClickHouse and Postgres settings are supplied through environment variables.
    Raises:
        Exception: Propagates fatal wiring/runtime exceptions to the caller.
    Side Effects:
        Starts the scheduler runtime and metrics endpoint.
    Docs:
      - docs/runbooks/mac-studio-native-backend-operations.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - apps/scheduler/backtest_artifact_publisher/wiring/modules/backtest_artifact_publisher.py
    """
    stop_event = asyncio.Event()
    _install_signal_handlers(stop_event)
    app = build_backtest_artifact_publisher_app(
        config_path=config_path,
        environ=os.environ,
        metrics_port=metrics_port,
        lock_path=lock_path,
    )
    await app.run(stop_event)
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Entrypoint for the backtest artifact publisher scheduler process.

    Args:
        argv: Optional command-line arguments excluding the program name.
    Returns:
        int: Process exit code.
    Assumptions:
        This function runs in a standalone long-lived process context.
    Raises:
        None. Fatal runtime errors are logged and converted into a non-zero exit code.
    Side Effects:
        Configures logging and runs the asyncio event loop.
    Docs:
      - docs/runbooks/mac-studio-native-backend-operations.md
      - docs/runbooks/backtest-artifacts-rebuild.md
    Related:
      - configs/installation/runtime-service-manifest.json
    """
    _configure_logging()
    args = _build_parser().parse_args(argv)
    try:
        return asyncio.run(
            _run_async(
                config_path=args.config,
                metrics_port=args.metrics_port,
                lock_path=args.lock_path,
            )
        )
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).exception("backtest-artifact-publisher failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
