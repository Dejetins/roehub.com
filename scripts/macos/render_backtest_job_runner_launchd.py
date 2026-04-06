from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Sequence

from trading.contexts.backtest.adapters.outbound import load_backtest_runtime_config

_DEFAULT_LAUNCH_AGENTS_DIR = Path("/Users/daniildegtyarev/Library/LaunchAgents")
_DEFAULT_LOGS_DIR = Path("/Users/daniildegtyarev/Library/Logs/roehub")


@dataclass(frozen=True, slots=True)
class LaunchdProfileDefaults:
    """
    Store deterministic launchd defaults for one operational profile.

    Args:
        profile: Stable runtime profile name (`dev`, `test`, `prod`).
        label_prefix: Launchd label prefix without trailing instance suffix.
        config_path: Runtime config path used by rendered launchd services.
        source_config_path: Repository config path used to read `worker_processes`.
        env_file: Host env file sourced by launchd.
        base_metrics_port: Fleet-wide base metrics port passed to each worker process.
        ch_port: ClickHouse HTTP port exposed on the target host.
        log_prefix: Prefix for per-instance log filenames.
    Returns:
        None.
    Assumptions:
        One launchd service maps to one worker process and one `instance_index`.
    Raises:
        None.
    Side Effects:
        None.
    """

    profile: str
    label_prefix: str
    config_path: Path
    source_config_path: Path
    env_file: Path
    base_metrics_port: int
    ch_port: int
    log_prefix: str


def _build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for launchd materialization helper.

    Args:
        None.
    Returns:
        argparse.ArgumentParser: Configured CLI parser.
    Assumptions:
        Renderer is executed from a checkout that contains the template and config files.
    Raises:
        None.
    Side Effects:
        None.
    """
    parser = argparse.ArgumentParser(
        prog="render_backtest_job_runner_launchd",
        description="Render deterministic launchd plists for backtest-job-runner fleet.",
    )
    parser.add_argument(
        "--profile",
        choices=("dev", "test", "prod"),
        required=True,
        help="Runtime profile used to load backtest worker_processes and render labels.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root that contains configs/ and infra/macos/launchd template.",
    )
    parser.add_argument(
        "--launch-agents-dir",
        type=Path,
        default=_DEFAULT_LAUNCH_AGENTS_DIR,
        help="Target launchd directory where per-instance plists are written.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove stale managed worker plists not present in the rendered fleet.",
    )
    return parser


def _resolve_profile_defaults(*, profile: str, repo_root: Path) -> LaunchdProfileDefaults:
    """
    Resolve deterministic launchd defaults for one profile.

    Args:
        profile: Runtime profile name (`dev`, `test`, `prod`).
        repo_root: Repository root used to locate source config files.
    Returns:
        LaunchdProfileDefaults: Stable defaults for label naming, logs, and ports.
    Assumptions:
        Configs live under `configs/<profile>/backtest.yaml`.
    Raises:
        ValueError: If profile is unsupported.
    Side Effects:
        None.
    """
    runtime_root = Path("/opt/roehub/app") / "configs"
    profile_defaults = {
        "dev": LaunchdProfileDefaults(
            profile="dev",
            label_prefix="com.roehub.dev.backtest-job-runner",
            config_path=runtime_root / "dev" / "backtest.yaml",
            source_config_path=repo_root / "configs" / "dev" / "backtest.yaml",
            env_file=Path("/Users/daniildegtyarev/.config/roehub/roehub.dev.env"),
            base_metrics_port=29_204,
            ch_port=8_123,
            log_prefix="dev-backtest-job-runner",
        ),
        "test": LaunchdProfileDefaults(
            profile="test",
            label_prefix="com.roehub.test.backtest-job-runner",
            config_path=runtime_root / "test" / "backtest.yaml",
            source_config_path=repo_root / "configs" / "test" / "backtest.yaml",
            env_file=Path("/Users/daniildegtyarev/.config/roehub/roehub.test.env"),
            base_metrics_port=19_204,
            ch_port=18_124,
            log_prefix="test-backtest-job-runner",
        ),
        "prod": LaunchdProfileDefaults(
            profile="prod",
            label_prefix="com.roehub.backtest-job-runner",
            config_path=runtime_root / "prod" / "backtest.yaml",
            source_config_path=repo_root / "configs" / "prod" / "backtest.yaml",
            env_file=Path("/Users/daniildegtyarev/.config/roehub/roehub.env"),
            base_metrics_port=9_204,
            ch_port=8_123,
            log_prefix="backtest-job-runner",
        ),
    }
    if profile not in profile_defaults:
        raise ValueError(f"Unsupported profile for launchd materialization: {profile}")
    return profile_defaults[profile]


def _load_template(*, repo_root: Path) -> Template:
    """
    Load the launchd plist template used for all worker instances.

    Args:
        repo_root: Repository root that contains the launchd template file.
    Returns:
        Template: Parsed string template for per-instance plist rendering.
    Assumptions:
        Template file is committed under `infra/macos/launchd`.
    Raises:
        FileNotFoundError: If the template file is missing.
    Side Effects:
        Reads template content from disk.
    """
    template_path = (
        repo_root / "infra" / "macos" / "launchd" / "com.roehub.backtest-job-runner@.plist.template"
    )
    return Template(template_path.read_text(encoding="utf-8"))


def _resolve_worker_instances(*, source_config_path: Path) -> int:
    """
    Resolve desired worker fleet size from runtime config.

    Args:
        source_config_path: Repository-local `backtest.yaml` path used for config loading.
    Returns:
        int: Number of launchd worker instances to materialize.
    Assumptions:
        Disabled jobs should materialize zero worker services.
    Raises:
        ValueError: Propagated from runtime config loader on invalid config.
    Side Effects:
        Reads runtime config file from disk.
    """
    runtime_config = load_backtest_runtime_config(source_config_path)
    if not runtime_config.jobs.enabled:
        return 0
    return runtime_config.jobs.worker_processes


def _render_plist_content(
    *,
    template: Template,
    defaults: LaunchdProfileDefaults,
    instance_index: int,
) -> str:
    """
    Render one launchd plist body for a worker instance.

    Args:
        template: Loaded launchd template.
        defaults: Profile-specific launchd defaults.
        instance_index: Deterministic worker instance index.
    Returns:
        str: Fully rendered launchd plist XML.
    Assumptions:
        Each service label ends with `.instance_index`.
    Raises:
        ValueError: If `instance_index` is negative.
    Side Effects:
        None.
    """
    if instance_index < 0:
        raise ValueError("instance_index must be >= 0")
    label = f"{defaults.label_prefix}.{instance_index}"
    stdout_path = _DEFAULT_LOGS_DIR / f"{defaults.log_prefix}.{instance_index}.out.log"
    stderr_path = _DEFAULT_LOGS_DIR / f"{defaults.log_prefix}.{instance_index}.err.log"
    return template.substitute(
        label=label,
        env_file=str(defaults.env_file),
        roehub_env=defaults.profile,
        ch_port=str(defaults.ch_port),
        config_path=str(defaults.config_path),
        metrics_port=str(defaults.base_metrics_port),
        instance_index=str(instance_index),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _collect_managed_plists(
    *,
    launch_agents_dir: Path,
    label_prefix: str,
) -> list[Path]:
    """
    Collect already materialized worker plists for one profile.

    Args:
        launch_agents_dir: LaunchAgents directory that stores generated plists.
        label_prefix: Profile-specific worker label prefix.
    Returns:
        list[Path]: Sorted managed plist paths.
    Assumptions:
        Managed worker filenames follow `<label_prefix>.<instance_index>.plist`.
    Raises:
        None.
    Side Effects:
        Reads directory entries from disk.
    """
    managed_paths = sorted(launch_agents_dir.glob(f"{label_prefix}.*.plist"))
    return [path for path in managed_paths if path.name[len(label_prefix) + 1 : -6].isdigit()]


def materialize_worker_launchd_plists(
    *,
    profile: str,
    repo_root: Path,
    launch_agents_dir: Path,
    clean: bool,
) -> list[Path]:
    """
    Render per-instance launchd plists for the configured worker fleet.

    Args:
        profile: Runtime profile (`dev`, `test`, `prod`).
        repo_root: Repository root that contains configs and template file.
        launch_agents_dir: Target directory for generated launchd plists.
        clean: Whether to delete stale managed plists outside the desired fleet.
    Returns:
        list[Path]: Deterministically ordered plist paths that should be bootstrapped.
    Assumptions:
        Fleet size is controlled by `backtest.jobs.worker_processes` when jobs are enabled.
    Raises:
        ValueError: If profile is unsupported or config is invalid.
        OSError: If filesystem writes/removals fail.
    Side Effects:
        Creates or overwrites launchd plists and optionally removes stale ones.
    """
    defaults = _resolve_profile_defaults(profile=profile, repo_root=repo_root)
    template = _load_template(repo_root=repo_root)
    launch_agents_dir.mkdir(parents=True, exist_ok=True)

    worker_instances = _resolve_worker_instances(source_config_path=defaults.source_config_path)
    rendered_paths: list[Path] = []
    for instance_index in range(worker_instances):
        plist_path = launch_agents_dir / f"{defaults.label_prefix}.{instance_index}.plist"
        plist_path.write_text(
            _render_plist_content(
                template=template,
                defaults=defaults,
                instance_index=instance_index,
            ),
            encoding="utf-8",
        )
        plist_path.chmod(0o644)
        rendered_paths.append(plist_path)

    if clean:
        desired_paths = {path.resolve() for path in rendered_paths}
        for existing_path in _collect_managed_plists(
            launch_agents_dir=launch_agents_dir,
            label_prefix=defaults.label_prefix,
        ):
            if existing_path.resolve() not in desired_paths:
                existing_path.unlink()

    return rendered_paths


def main(argv: Sequence[str] | None = None) -> int:
    """
    Execute launchd materialization CLI for backtest-job-runner fleet.

    Args:
        argv: Optional command-line arguments without program name.
    Returns:
        int: Process exit code.
    Assumptions:
        Standard output is consumed by shell scripts that may capture rendered filenames.
    Raises:
        None.
    Side Effects:
        Writes generated plist filenames to standard output.
    """
    args = _build_parser().parse_args(argv)
    rendered_paths = materialize_worker_launchd_plists(
        profile=args.profile,
        repo_root=args.repo_root.resolve(),
        launch_agents_dir=args.launch_agents_dir.resolve(),
        clean=args.clean,
    )
    for rendered_path in rendered_paths:
        print(rendered_path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
