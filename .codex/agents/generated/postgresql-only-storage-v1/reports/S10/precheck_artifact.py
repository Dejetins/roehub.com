"""Read-only reproduction of retained-manifest bootstrap conflict."""

import json
import os

from apps.cli.wiring.db.postgres import close_market_gateways
from apps.scheduler.backtest_artifact_publisher.wiring.modules import (
    build_backtest_artifact_publisher_app,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
)

app = build_backtest_artifact_publisher_app(
    config_path="/etc/roehub/backtest-artifacts.yaml",
    environ=os.environ,
    metrics_port=9298,
    lock_path=None,
)
try:
    app.worker.publisher.publisher.precheck_publish(
        ArtifactCoordinatesV2("binance", "spot", "AAVEUSDT")
    )
except ValueError as error:
    assert str(error).startswith(
        "bootstrap requires missing current.yaml and no pre-existing slot manifests"
    )
    print(
        json.dumps(
            {
                "precheck_refused": "retained_manifests_without_current_pointer",
                "before_build_and_publish": True,
                "error_type": type(error).__name__,
            }
        )
    )
else:
    raise AssertionError("expected conflict")
finally:
    close_market_gateways()
