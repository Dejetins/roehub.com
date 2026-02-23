"""CLI entrypoint for running Roehub Web SSR service."""

from __future__ import annotations

import argparse

import uvicorn


def _build_parser() -> argparse.ArgumentParser:
    """
    Build command-line parser for web SSR process.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Parser with network bind options.
    Assumptions:
        Defaults are suitable for local development.
    Raises:
        None.
    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(prog="roehub-web")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8010, help="Bind port")
    return parser



def main(argv: list[str] | None = None) -> int:
    """
    Run web SSR process using uvicorn.

    Args:
        argv: Optional command arguments without program name.
    Returns:
        int: Process exit code.
    Assumptions:
        Import path `apps.web.main.app:create_app` resolves and passes fail-fast config checks.
    Raises:
        None.
    Side Effects:
        Starts HTTP server loop.
    """
    args = _build_parser().parse_args(argv)
    # Developer mode example:
    # WEB_API_BASE_URL=http://127.0.0.1:8000 uv run python -m apps.web.main.main --port 8010
    uvicorn.run("apps.web.main.app:create_app", host=args.host, port=args.port, factory=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
