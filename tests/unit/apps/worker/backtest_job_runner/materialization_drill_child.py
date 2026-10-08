"""Fault injection at native numerical/write boundaries; real full-job child composition."""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path


def main() -> int:
    from apps.worker.backtest_job_runner.main.full_job_child import main as run_child
    from trading.contexts.backtest.application.services.v2.no_risk_exact import (
        BacktestNoRiskExactScoringService,
    )
    from trading.contexts.backtest_artifacts.application.services.v2 import (
        artifact_precompute_runner as native,
    )

    mode = os.environ["S04_DRILL_MODE"]
    marker = Path(os.environ["S04_DRILL_MARKER"])

    def pause() -> None:
        marker.write_text(mode)
        if mode == "crash":
            os._exit(9)
        while True:
            time.sleep(0.05)

    if mode in ("signal", "crash"):
        original = native._execute_signal_chunk_job_v2

        def signal(*args, **kwargs):
            value = original(*args, **kwargs)
            pause()
            return value

        native._execute_signal_chunk_job_v2 = signal
    elif mode in ("hit_times", "timeout"):
        original_hit = native.materialize_hit_times_from_ohlcv_v2

        def hit(*args, **kwargs):
            value = original_hit(*args, **kwargs)
            pause()
            return value

        native.materialize_hit_times_from_ohlcv_v2 = hit
    elif mode == "scoring":
        original_score = BacktestNoRiskExactScoringService.execute

        def score(*args, **kwargs):
            value = original_score(*args, **kwargs)
            pause()
            return value

        BacktestNoRiskExactScoringService.execute = score
    elif mode == "disk_full":

        def disk_full(*args, **kwargs):
            marker.write_text(mode)
            raise OSError(errno.ENOSPC, "S04 injected full attempt disk")

        native._write_signal_features_v2 = disk_full
    elif mode == "ready_manifest":
        original_yaml = native._write_yaml_atomically_v2

        def yaml_write(*, path, payload):
            if payload.get("manifest_kind") == "derived_artifacts":
                marker.write_text(mode)
                raise OSError("S04 injected failure before ready manifest")
            return original_yaml(path=path, payload=payload)

        native._write_yaml_atomically_v2 = yaml_write
    return run_child()


if __name__ == "__main__":
    raise SystemExit(main())
