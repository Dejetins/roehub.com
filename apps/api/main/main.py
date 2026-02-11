"""
CLI entrypoint for running Roehub FastAPI service.
"""

from __future__ import annotations

import argparse

import uvicorn


def _build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser for API process.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured parser.
    Assumptions:
        Defaults are suitable for local development.
    Raises:
        None.
    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(prog="roehub-api")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Run API process using uvicorn.

    Args:
        argv: Optional command arguments without program name.
    Returns:
        int: Process exit code.
    Assumptions:
        Import path `apps.api.main.app:app` is available in PYTHONPATH.
    Raises:
        None.
    Side Effects:
        Starts HTTP server loop.
    """
    args = _build_parser().parse_args(argv)
    uvicorn.run("apps.api.main.app:app", host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
