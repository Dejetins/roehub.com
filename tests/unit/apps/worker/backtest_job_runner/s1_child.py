"""Real disposable IPC fixture: deliberately writes partial payloads on failures."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    args = dict(zip(sys.argv[1::2], sys.argv[2::2], strict=True))
    output = Path(args["--output-json"])
    record = Path(os.environ["S1_CHILD_RECORD"])
    assert "numba" not in sys.modules
    preimport_threads = os.environ["NUMBA_NUM_THREADS"]
    import numba as nb

    from apps.worker.backtest_job_runner.main import full_job_child

    assert getattr(nb.config, "NUMBA_NUM_THREADS") == int(preimport_threads)
    assert nb.get_num_threads() == int(preimport_threads)
    from trading.contexts.backtest.application.services.v2.job_scheduling import (
        backtest_numba_environ,
    )

    os.environ.update(
        backtest_numba_environ(
            environ=os.environ,
            scheduling_class="heavy",
            inherited=True,
        )
    )
    data = dict(
        pid=os.getpid(),
        output=str(output),
        module=full_job_child.__file__,
        threads=os.environ["NUMBA_NUM_THREADS"],
        source=os.environ["ROEHUB_BACKTEST_EFFECTIVE_NUMBA_THREAD_SOURCE"],
        numba_preimport=False,
        actual_threads=nb.get_num_threads(),
    )
    record.write_text(json.dumps(data))
    mode = os.environ["S1_CHILD_MODE"]
    if mode in {"timeout", "cancel", "failure", "malformed"}:
        output.write_text('{"status":"succeeded","top_variants":[')
    if mode in {"timeout", "cancel"}:
        time.sleep(30)
    if mode == "failure":
        return 7
    if mode == "malformed":
        return 0
    output.write_text(
        json.dumps(
            dict(
                status="succeeded",
                top_variants=[],
                stage_timings={},
                summary_hash="c" * 64,
                cleanup_evidence=data,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
