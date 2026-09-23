from pathlib import Path

import pytest

from apps.worker.backtest_job_runner.wiring.modules import compute_resources as module


def _mount(tmp_path, monkeypatch, membership):
    proc = tmp_path / 'cgroup-membership'
    proc.write_text(membership)
    root = tmp_path / 'mount'
    root.mkdir()
    def mapped(path):
        if path == '/proc/self/cgroup':
            return proc
        if path == '/sys/fs/cgroup':
            return root
        if path == '/sys/fs/cgroup/cpu':
            return root / 'cpu'
        return Path(path)
    monkeypatch.setattr(module, 'Path', mapped)
    return root


def test_unified_quota_intersects_ancestors_and_fractional_capacity(tmp_path, monkeypatch):
    root = _mount(tmp_path, monkeypatch, '0::/group/child\n')
    (root / 'group/child').mkdir(parents=True)
    (root / 'cpu.max').write_text('1200000 100000')
    (root / 'group/cpu.max').write_text('250000 100000')
    (root / 'group/child/cpu.max').write_text('max 100000')
    assert module._linux_cpu_quota() == 2


def test_legacy_quota_intersects_ancestors(tmp_path, monkeypatch):
    root = _mount(tmp_path, monkeypatch, '2:cpu,cpuacct:/child\n') / 'cpu'
    (root / 'child').mkdir(parents=True)
    for folder, quota in [(root, '400000'), (root / 'child', '600000')]:
        (folder / 'cpu.cfs_quota_us').write_text(quota)
        (folder / 'cpu.cfs_period_us').write_text('100000')
    assert module._linux_cpu_quota() == 4


def test_unreadable_quota_remains_unknown(tmp_path, monkeypatch):
    root = _mount(tmp_path, monkeypatch, '0::/\n')
    (root / 'cpu.max').write_text('invalid')
    assert module._linux_cpu_quota() is None


@pytest.mark.parametrize('cap', ['0', '-1', 'bad'])
def test_invalid_operator_cap_is_rejected(cap):
    with pytest.raises(ValueError):
        module.discover_cpu_capacity({module.OPERATOR_CAP: cap})
