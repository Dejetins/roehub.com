from datetime import UTC, datetime, timedelta

import pytest

from apps.api.worker_control import worker_commands
from apps.monitoring.operational_health import OperationalStatus


@pytest.mark.parametrize('name,action,age,expected', [
    ('backtest-artifact-publisher','manage_worker',0,('start','stop')),
    ('market-data-ws','manage_worker',29,('start','stop')),
    ('market-data-scheduler','manage_worker',31,()),
    ('api','manage_worker',0,()),
    ('postgresql','manage_worker',0,()),
    ('backtest-artifact-publisher','diagnostics',0,()),
    ('backtest-artifact-publisher','manage_worker',-1,()),
])
def test_worker_commands_fail_closed(name, action, age, expected):
    now = datetime.now(UTC)
    status = OperationalStatus(service_id=name, capability='research.artifacts', state='unknown',
        detail_code='probe.reachable_no_readiness', runbook_id='runtime.service-degraded',
        action_ref=action, required=False, observed_at=now-timedelta(seconds=age))
    assert worker_commands(status, now) == expected
