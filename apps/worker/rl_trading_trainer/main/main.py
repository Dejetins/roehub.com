from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    selected_argv = [] if argv is None else list(argv)
    if selected_argv[:1] == ["stage07b"]:
        from scripts.rl_trading.stage07b_full_candidate_training_run import main as candidate_main

        return candidate_main(selected_argv[1:])

    from scripts.rl_trading.stage07a_training_runner_smoke import main as smoke_main

    return smoke_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
