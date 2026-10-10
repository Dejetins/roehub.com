"""Installation-owner storage service; no browser paths, no timed ownership reclaim."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from psycopg.types.json import Jsonb

from trading.contexts.backtest.adapters.outbound.persistence.postgres.backtest_job_repository import (  # noqa: E501
    PostgresBacktestJobRepository,
)
from trading.contexts.backtest_artifacts.adapters.outbound.artifacts_fs.path_builder import (
    BacktestArtifactPathBuilderV2,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    ManagementError,
    digest,
)
from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCoordinatesV2,
    ArtifactCurrentPointerV2,
)
from trading.contexts.operations.artifact_storage_contracts import StorageResult


@contextmanager
def directory(path):
    """Open every component without following links; retain descriptor for actual IO."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ManagementError("untrusted_root")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            info = os.fstat(child)
            if info.st_uid not in (0, os.geteuid()) or (
                info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX
            ):
                os.close(child)
                raise ManagementError("untrusted_root")
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid not in (0, os.geteuid()) or info.st_mode & 0o022:
            raise ManagementError("untrusted_root")
        yield fd
    finally:
        os.close(fd)


def identity(fd):
    s = os.fstat(fd)
    return [s.st_dev, s.st_ino]


def inventory(fd, *, remove=False, remaining=None):
    """Bounded no-follow traversal; reject nonregular objects and hardlinked payloads."""
    remaining = [100000] if remaining is None else remaining
    size = 0
    for name in os.listdir(fd):
        remaining[0] -= 1
        if remaining[0] < 0:
            raise ManagementError("storage_inventory_unavailable")
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                size += inventory(child, remove=remove, remaining=remaining)
            finally:
                os.close(child)
            if remove:
                os.rmdir(name, dir_fd=fd)
        elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
            size += info.st_size
            if remove:
                os.unlink(name, dir_fd=fd)
        else:
            raise ManagementError("untrusted_root")
    if remove:
        os.fsync(fd)
    return size


def probe(fd):
    name = ".storage-probe-" + uuid4().hex
    file = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    try:
        if os.write(file, b"roehub-storage-probe\n") != 21:
            raise ManagementError("storage_probe_failed")
        os.fsync(file)
    finally:
        os.close(file)
        os.unlink(name, dir_fd=fd)
        os.fsync(fd)


class ArtifactStorageService:
    def __init__(self, *, repository, roots, default_schedule, installation_id):
        self.installation_id = installation_id
        self.repo = repository
        self.roots = dict(roots)
        self.default_schedule = default_schedule
        self.jobs = PostgresBacktestJobRepository(gateway=repository.db)

    def execute(self, envelope):
        try:
            return self._execute(envelope)
        except OSError as error:
            import errno

            reason = "disk_full" if error.errno == errno.ENOSPC else "storage_access_unavailable"
            raise ManagementError(reason) from error

    def _execute(self, envelope):
        r, c = self.repo, envelope.command
        cleanup = None
        with r.db.transaction():
            installation = r._installation_owner(envelope.organization_id, envelope.actor_id)
            if installation != self.installation_id:
                raise ManagementError("installation_forbidden")
            r._lock(installation)
            settings = r._one(
                "SELECT * FROM backtest_artifact_installation_settings WHERE "
                "installation_id=%(i)s FOR UPDATE",
                i=installation,
            )
            version = settings["control_version"] if settings else 0
            revision = settings["root_revision"] if settings else 1
            if c.root_id not in self.roots:
                raise ManagementError("untrusted_root")
            root = self.roots[c.root_id]
            args = dict(i=installation, a=envelope.actor_id, k=envelope.idempotency_key)
            fingerprint = digest(c.model_dump(mode="json"))
            old = r._one(
                "SELECT body_hash,response FROM backtest_artifact_storage_commands WHERE "
                "installation_id=%(i)s AND actor_id=%(a)s AND idempotency_key=%(k)s",
                **args,
            )
            if old:
                if old["body_hash"] != fingerprint:
                    raise ManagementError("idempotency_conflict")
                if old["response"] is None:
                    return StorageResult(
                        state="unknown",
                        root_id=c.root_id,
                        control_version=version,
                        root_revision=revision,
                    )
                return StorageResult.model_validate(old["response"])
            if c.action != "inspect" and version != c.expected_version:
                raise ManagementError("stale_version")
            with directory(root) as fd:
                root_identity = identity(fd)
                used = inventory(fd)
                space = os.fstatvfs(fd)
                free = space.f_bavail * space.f_frsize
                base: dict[str, Any] = dict(
                    root_id=c.root_id,
                    control_version=version,
                    root_revision=revision,
                    path=str(root),
                    used_bytes=used,
                    free_bytes=free,
                    budget_bytes=settings["budget_bytes"] if settings else None,
                    schedule=settings["schedule"] if settings else self.default_schedule,
                )
                if c.action == "inspect":
                    return StorageResult(state="ready", **base)
                if c.action == "validate":
                    probe(fd)
                    token, expires = self._preview(
                        envelope, installation, root_identity, {}, "validate"
                    )
                    result = StorageResult(
                        state="validated", validation_id=token, expires_at=expires, **base
                    )
                elif c.action == "configure":
                    self._token(envelope, installation, c.validation_id, root_identity, "validate")
                    probe(fd)
                    relocating = settings and settings["root_id"] != c.root_id
                    busy = r._one(
                        """SELECT 1 FROM backtest_artifact_requests WHERE
                installation_id=%(i)s LIMIT 1""",
                        i=installation,
                    )
                    pins = r._one("SELECT 1 FROM backtest_artifact_slot_readers LIMIT 1")
                    owners = r._one(
                        "SELECT 1 FROM backtest_artifact_slot_ownership WHERE generation IS "
                        "NOT NULL OR state!='available' LIMIT 1"
                    )
                    old_used = 0
                    if relocating or settings is None:
                        old_path = self.roots.get(
                            settings["root_id"] if settings else "backtest-primary"
                        )
                        if old_path is None:
                            raise ManagementError("untrusted_root")
                        with directory(old_path) as old_fd:
                            old_used = len(os.listdir(old_fd))
                    if (relocating or settings is None) and (
                        used or old_used or busy or pins or owners
                    ):
                        result = StorageResult(state="relocation_required", **base)
                    else:
                        if free < 1073741824:
                            raise ManagementError("disk_full")
                        if settings is None:
                            r.initialize_settings(
                                organization_id=envelope.organization_id,
                                actor_id=envelope.actor_id,
                                root_id=c.root_id,
                                schedule=c.schedule,
                            )
                        r._exec(
                            """UPDATE backtest_artifact_installation_settings SET root_id=%(root)s,
                            root_revision=%(revision)s,control_version=%(version)s,budget_bytes=%(budget)s,
                            schedule=%(schedule)s,updated_at=now() WHERE installation_id=%(i)s""",
                            i=installation,
                            root=c.root_id,
                            revision=revision + int(bool(relocating)),
                            version=version + 1,
                            budget=c.budget_bytes,
                            schedule=Jsonb(c.schedule.model_dump(mode="json")),
                        )
                        base.update(
                            control_version=version + 1,
                            root_revision=revision + int(bool(relocating)),
                            budget_bytes=c.budget_bytes,
                            schedule=c.schedule,
                        )
                        result = StorageResult(state="configured", **base)
                elif c.action == "cleanup_preview":
                    if settings is None or settings["root_id"] != c.root_id:
                        raise ManagementError("stale_version")
                    target = self._target(root, c.coordinate, c.slot)
                    self._check_db(c.coordinate, c.slot, target)
                    token, expires = self._preview(
                        envelope, installation, root_identity, target, "cleanup_preview"
                    )
                    result = StorageResult(
                        state="preview",
                        preview_id=token,
                        expires_at=expires,
                        eligible_bytes=target["bytes"],
                        **base,
                    )
                else:
                    if settings is None or settings["root_id"] != c.root_id:
                        raise ManagementError("stale_version")
                    saved = self._token(
                        envelope, installation, c.preview_id, root_identity, "cleanup_preview"
                    )
                    # Durable intent: a crash never replays cleanup blindly.
                    r._exec(
                        """INSERT INTO
                backtest_artifact_storage_commands(installation_id,actor_id,idempotency_key,body_hash)
                        VALUES (%(i)s,%(a)s,%(k)s,%(h)s)""",
                        **args,
                        h=fingerprint,
                    )
                    # Commit intent before writer admission and destructive IO.
                    cleanup = (saved["target"], base, args, fingerprint, installation)
                    result = None
                if result is not None:
                    r._exec(
                        """INSERT INTO
                backtest_artifact_storage_commands(installation_id,actor_id,idempotency_key,body_hash,response)
                        VALUES (%(i)s,%(a)s,%(k)s,%(h)s,%(response)s)""",
                        **args,
                        h=fingerprint,
                        response=Jsonb(result.model_dump(mode="json")),
                    )
                    return result
        assert cleanup is not None
        target, base, args, fingerprint, installation = cleanup
        return self._cleanup(envelope, root, target, base, args, installation)

    def _preview(self, envelope, installation, root_identity, target, kind):
        token, expires = uuid4(), datetime.now(UTC) + timedelta(minutes=5)
        self.repo._exec(
            """INSERT INTO backtest_artifact_storage_previews
            (preview_id,installation_id,actor_id,kind,control_version,root_id,root_identity,target,expires_at)
            VALUES
                (%(p)s,%(i)s,%(a)s,%(kind)s,%(v)s,%(root)s,%(identity)s,%(target)s,%(expires)s)""",
            p=token,
            i=installation,
            a=envelope.actor_id,
            kind=kind,
            v=envelope.command.expected_version,
            root=envelope.command.root_id,
            identity=Jsonb(root_identity),
            target=Jsonb(target),
            expires=expires,
        )
        return token, expires

    def _token(self, envelope, installation, token, root_identity, kind):
        row = self.repo._one(
            """SELECT * FROM backtest_artifact_storage_previews WHERE preview_id=%(p)s
            AND installation_id=%(i)s AND actor_id=%(a)s AND kind=%(kind)s AND NOT consumed
            AND expires_at>now() FOR UPDATE""",
            p=token,
            i=installation,
            a=envelope.actor_id,
            kind=kind,
        )
        if (
            row is None
            or row["control_version"] != envelope.command.expected_version
            or row["root_id"] != envelope.command.root_id
            or row["root_identity"] != root_identity
        ):
            raise ManagementError("preview_stale")
        self.repo._exec(
            "UPDATE backtest_artifact_storage_previews SET consumed=TRUE WHERE preview_id=%(p)s",
            p=token,
        )
        return row

    def _target(self, root, coordinate, slot):
        coords = ArtifactCoordinatesV2(*coordinate.key)
        path = BacktestArtifactPathBuilderV2(root=root).slot_manifest_path(coords, slot)
        with directory(path.parent.parent) as parent:
            current = _read_yaml(parent, "current.yaml")
            try:
                ArtifactCurrentPointerV2(
                    path=path.parent.parent / "current.yaml", raw_payload=current, **current
                )
                current_fd = os.open(
                    current["active_slot"],
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent,
                )
                try:
                    active_bytes = _read_bytes(current_fd, "manifest.yaml")
                finally:
                    os.close(current_fd)
                if hashlib.sha256(active_bytes).hexdigest() != current["manifest_sha256"]:
                    raise ValueError("current_manifest_changed")
                active_manifest = yaml.safe_load(active_bytes)
                if active_manifest["slot_generation"] != current["slot_generation"]:
                    raise ValueError("current_generation_changed")
            except (ValueError, TypeError, KeyError) as error:
                raise ManagementError("preview_stale") from error
            child = os.open(slot, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            try:
                size = inventory(child)
                info = identity(child)
                manifest_bytes = _read_bytes(child, path.name)
                manifest = yaml.safe_load(manifest_bytes)
                if (
                    not isinstance(manifest, dict)
                    or type(manifest.get("slot_generation")) is not int
                    or manifest["slot_generation"] < 1
                ):
                    raise ManagementError("preview_stale")
            finally:
                os.close(child)
            if current["active_slot"] == slot:
                raise ManagementError("cleanup_current")
            manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        return dict(
            coordinate=coordinate.model_dump(),
            slot=slot,
            generation=manifest["slot_generation"],
            manifest_sha256=manifest_hash,
            directory_identity=info,
            bytes=size,
            current_slot=current["active_slot"],
            current_generation=current["slot_generation"],
            current_hash=current["manifest_sha256"],
        )

    def _check_db(self, c, slot, target):
        row = self.repo._one(
            """SELECT * FROM backtest_artifact_slot_ownership WHERE exchange=%(exchange)s
            AND market_type=%(market_type)s AND symbol=%(symbol)s AND slot=%(slot)s""",
            **c.model_dump(),
            slot=slot,
        )
        # Cleanup never adopts an unregistered on-disk slot.
        if (
            row is None
            or row["state"] != "available"
            or row["generation"] != target["generation"]
            or row["manifest_sha256"] != target["manifest_sha256"]
        ):
            raise ManagementError("cleanup_busy")
        pin = self.repo._one(
            """SELECT 1 FROM backtest_artifact_slot_readers WHERE exchange=%(exchange)s
            AND market_type=%(market_type)s AND symbol=%(symbol)s AND slot=%(slot)s LIMIT 1""",
            **c.model_dump(),
            slot=slot,
        )
        if pin:
            raise ManagementError("cleanup_pinned")

    def _cleanup(self, envelope, root, target, base, args, installation):
        from trading.contexts.backtest_artifacts.application.services.management import Coordinate

        c = Coordinate.model_validate(target["coordinate"])
        writer = None
        try:
            with self.repo.db.transaction():
                self.repo._installation_owner(envelope.organization_id, envelope.actor_id)
                self.repo._lock(installation)
                settings = self.repo._settings(installation, active=False)
                if (
                    settings["control_version"] != envelope.command.expected_version
                    or settings["root_id"] != envelope.command.root_id
                ):
                    raise ManagementError("preview_stale")
                self._check_db(c, target["slot"], target)
                writer = self.jobs.reserve_artifact_writer(
                    **c.model_dump(),
                    slot=target["slot"],
                    expected_generation=target["generation"],
                    expected_manifest_sha256=target["manifest_sha256"],
                    owner_token=uuid4(),
                    attempt=1,
                    parent_incarnation=uuid4(),
                )
                if self._target(root, c, target["slot"]) != target:
                    raise ManagementError("preview_stale")
            # Durable writer ownership fences every reader/publisher until completion or quarantine.
            with self.repo.db.transaction():
                self.repo._installation_owner(envelope.organization_id, envelope.actor_id)
                self.repo._lock(installation)
                settings = self.repo._settings(installation, active=False)
                if (
                    settings["control_version"] != envelope.command.expected_version
                    or settings["root_id"] != envelope.command.root_id
                ):
                    raise ManagementError("preview_stale")
                preview = self.repo._one(
                    "SELECT root_identity FROM backtest_artifact_storage_previews WHERE "
                    "preview_id=%(p)s",
                    p=envelope.command.preview_id,
                )
                with directory(root) as root_fd:
                    if preview is None or identity(root_fd) != preview["root_identity"]:
                        raise ManagementError("preview_stale")
                self.jobs.verify_artifact_writer(writer=writer)
                if self._target(root, c, target["slot"]) != target:
                    raise ManagementError("preview_stale")
                path = (
                    BacktestArtifactPathBuilderV2(root=root)
                    .slot_manifest_path(ArtifactCoordinatesV2(*c.key), target["slot"])
                    .parent
                )
                with directory(path) as fd:
                    if identity(fd) != target["directory_identity"]:
                        raise ManagementError("preview_stale")
                    inventory(fd, remove=True)
                with directory(path.parent) as parent:
                    os.rmdir(path.name, dir_fd=parent)
                    os.fsync(parent)
                self.jobs.complete_artifact_writer(
                    writer=writer, generation=None, manifest_sha256=None
                )
                result = StorageResult(state="cleaned", eligible_bytes=target["bytes"], **base)
                self.repo._exec(
                    """UPDATE backtest_artifact_storage_commands SET response=%(r)s
                    WHERE installation_id=%(i)s AND actor_id=%(a)s AND idempotency_key=%(k)s""",
                    **args,
                    r=Jsonb(result.model_dump(mode="json")),
                )
                return result
        except BaseException:
            if writer is not None and self.jobs.owns_artifact_writer(writer=writer):
                self.jobs.quarantine_artifact_writer(writer=writer)
            raise


def _read_bytes(fd, name):
    child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
    try:
        info = os.fstat(child)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > 4 * 1024 * 1024:
            raise ManagementError("untrusted_root")
        with os.fdopen(child, "rb", closefd=False) as stream:
            return stream.read(4 * 1024 * 1024 + 1)
    finally:
        os.close(child)


def _read_yaml(fd, name):
    value = yaml.safe_load(_read_bytes(fd, name))
    if not isinstance(value, dict):
        raise ManagementError("untrusted_root")
    return value
