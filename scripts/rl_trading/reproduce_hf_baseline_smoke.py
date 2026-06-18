from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trading.contexts.rl_trading.domain.hf_reproducibility import (  # noqa: E402
    HF_DATASET_RESOLVE_BASE_URL_V1,
    HF_RUNTIME_ARTIFACT_ROOT_V1,
    HfDatasetSplitSpec,
    HfReproducibilityError,
    HfReproducibilityRunConfig,
    compute_file_sha256,
    expected_hf_split_specs_v1,
    render_json_payload_v1,
    run_hf_reproducibility_smoke_v1,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Stage 04 HF reproducibility train/eval/backtest smoke."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path(HF_RUNTIME_ARTIFACT_ROOT_V1),
        help="Runtime artifact root outside git.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Directory containing HF NPZ split files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path for sanitized smoke evidence JSON.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing or hash-mismatched HF NPZ files into dataset-dir.",
    )
    parser.add_argument("--seed", type=int, default=240604)
    parser.add_argument("--train-sample-size", type=int, default=32)
    parser.add_argument("--evaluation-sample-size", type=int, default=16)
    parser.add_argument("--backtest-sample-size", type=int, default=16)
    parser.add_argument("--torch-epochs", type=int, default=16)
    parser.add_argument("--torch-learning-rate", type=float, default=0.05)
    parser.add_argument(
        "--trainer",
        choices=("torch_logistic", "numpy_centroid"),
        default="torch_logistic",
    )
    args = parser.parse_args(argv)

    dataset_dir = args.dataset_dir or (
        args.artifact_root
        / "hf_reproducibility"
        / "dataset"
        / "ResearchRL"
        / "open-rl-trading-binance-dataset"
    )
    output_json = args.output_json or (
        args.artifact_root / "hf_reproducibility" / "stage04_hf_reproducibility_smoke.json"
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    split_specs = expected_hf_split_specs_v1()
    if args.download:
        ensure_hf_dataset_files(dataset_dir=dataset_dir, split_specs=split_specs)
    else:
        missing = [
            spec.file_name
            for spec in split_specs
            if not (dataset_dir / spec.file_name).exists()
        ]
        if missing:
            raise HfReproducibilityError(
                reason="missing_hf_split_file_download_not_enabled",
                field=",".join(missing),
            )

    config = HfReproducibilityRunConfig(
        seed=args.seed,
        trainer=args.trainer,
        train_sample_size=args.train_sample_size,
        evaluation_sample_size=args.evaluation_sample_size,
        backtest_sample_size=args.backtest_sample_size,
        torch_epochs=args.torch_epochs,
        torch_learning_rate=args.torch_learning_rate,
    )
    payload = run_hf_reproducibility_smoke_v1(dataset_dir=dataset_dir, config=config)
    rendered = render_json_payload_v1(payload)
    output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


def ensure_hf_dataset_files(
    *,
    dataset_dir: Path,
    split_specs: tuple[HfDatasetSplitSpec, ...],
) -> None:
    for split_spec in split_specs:
        target = dataset_dir / split_spec.file_name
        if target.exists() and compute_file_sha256(target) == split_spec.expected_sha256:
            continue
        if target.exists():
            target.unlink()
        _download_split_file(split_spec=split_spec, target=target)


def _download_split_file(*, split_spec: HfDatasetSplitSpec, target: Path) -> None:
    url = f"{HF_DATASET_RESOLVE_BASE_URL_V1}/{split_spec.file_name}"
    temp_path = target.with_suffix(target.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "roehub-stage04-hf-reproducibility/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with temp_path.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
    observed_sha256 = compute_file_sha256(temp_path)
    if observed_sha256 != split_spec.expected_sha256:
        temp_path.unlink(missing_ok=True)
        raise HfReproducibilityError(
            reason="downloaded_hf_split_hash_mismatch",
            field=split_spec.file_name,
        )
    temp_path.replace(target)


if __name__ == "__main__":
    raise SystemExit(main())
