"""One normal managed cycle; emit exception class/frames without credentials."""

import json
import os
import traceback

from apps.cli.wiring.db.postgres import close_market_gateways
from apps.scheduler.backtest_artifact_publisher.wiring.modules import (
    build_backtest_artifact_publisher_app,
)

app = build_backtest_artifact_publisher_app(
    config_path="/etc/roehub/backtest-artifacts.yaml",
    environ=os.environ,
    metrics_port=9297,
    lock_path=None,
)


def wrap(target, name):
    original = getattr(target, name)

    def observed(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as error:
            chain = []
            while error is not None:
                chain.append(
                    {
                        "type": type(error).__name__,
                        "message": str(error)[:200],
                        "frames": [
                            (f.filename, f.lineno, f.name)
                            for f in traceback.extract_tb(error.__traceback__)
                        ],
                    }
                )
                error = error.__cause__
            print(json.dumps({"boundary": name, "chain": chain}), flush=True)
            raise

    setattr(target, name, observed)


wrap(app.worker.repository, "claim_group")
wrap(app.worker, "check_storage")
wrap(app.worker, "execute")
# Use the normal lock; only the long-lived scheduler is briefly paused while idle.
try:
    print(json.dumps({"cycle": app.cycle()}), flush=True)
except Exception as error:
    print(
        json.dumps(
            {
                "exception": type(error).__name__,
                "message": str(error)[:200]
                if not any(
                    x in str(error).lower()
                    for x in ["password", "dsn", "postgres://", "postgresql://"]
                )
                else "redacted",
                "frames": [
                    (f.filename, f.lineno, f.name)
                    for f in traceback.extract_tb(error.__traceback__)
                ],
            }
        ),
        flush=True,
    )
finally:
    close_market_gateways()
