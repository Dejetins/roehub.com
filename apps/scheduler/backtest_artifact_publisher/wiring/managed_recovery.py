"""Local process-lifetime proof. Unknown pointer outcomes remain quarantined, without TTL."""

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path

from trading.contexts.backtest_artifacts.application.services.management import ManagementError


@contextmanager
def lifetime(root: Path, build):
    directory = root / ".managed-attempts" / str(build["owner_token"])
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or directory.parent.is_symlink():
        raise ManagementError("untrusted_root")
    descriptor = os.open(directory / "lifetime.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield directory
    finally:
        os.close(descriptor)


def save_receipt(directory: Path, build, publication, coverage):
    payload = dict(
        build_id=str(build["build_id"]),
        owner_token=str(build["owner_token"]),
        parent_incarnation=str(build["parent_incarnation"]),
        attempt=build["attempt"],
        epoch=build["epoch"],
        publication=publication,
        coverage=coverage,
    )
    target = directory / "receipt.json"
    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as file:
        json.dump(payload, file, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
