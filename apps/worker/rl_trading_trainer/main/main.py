from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from scripts.rl_trading.stage07a_training_runner_smoke import main as smoke_main

    return smoke_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
