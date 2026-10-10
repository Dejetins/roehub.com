"""Automatic base projection on the additive S02 schema; no new physical lock protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from psycopg.types.json import Jsonb

from trading.contexts.backtest_artifacts.adapters.outbound.persistence.postgres.management_repository import (  # noqa: E501
    PostgresArtifactManagementRepository,
)
from trading.contexts.backtest_artifacts.application.services.management import (
    Coordinate,
    ManagementError,
    SourceTarget,
)


class PostgresAutomaticBaseRepository(PostgresArtifactManagementRepository):
    def base_request(self, *, installation_id: UUID, request_id: UUID) -> Mapping[str, Any]:
        row = self._one(
            """SELECT * FROM backtest_artifact_requests
            WHERE installation_id=%(i)s AND request_id=%(r)s AND kind='base'""",
            i=installation_id,
            r=request_id,
        )
        if row is None:
            raise ManagementError("invalid_base_request")
        return row

    def observe_source(
        self,
        *,
        installation_id: UUID,
        coordinate: Coordinate,
        source: SourceTarget,
        observed_at: datetime,
    ) -> UUID | None:
        """Coalesce every hint to latest desired state, enqueue only the first usable source.

        Later source changes remain durable for S05 cadence; never subscribe compute to
        every streaming minute. Failed initial work does not create timer retry storms.
        """
        coordinate = Coordinate.model_validate(coordinate.model_dump())
        source = SourceTarget.model_validate(source.model_dump())
        with self.db.transaction():
            self._lock(installation_id)
            previous = self._one(
                """SELECT desired_target FROM backtest_artifact_source_state
                WHERE exchange=%(exchange)s AND market_type=%(market_type)s AND symbol=%(symbol)s
                FOR UPDATE""",
                **coordinate.model_dump(),
            )
            if (
                previous
                and previous["desired_target"].get("observed_at", "") > observed_at.isoformat()
            ):
                return None
            request = self._one(
                """SELECT request_id FROM backtest_artifact_requests
                WHERE installation_id=%(i)s AND exchange=%(exchange)s
                AND market_type=%(market_type)s
                AND symbol=%(symbol)s AND kind='base' AND snapshot->'source' != 'null'::jsonb
                ORDER BY created_at LIMIT 1""",
                i=installation_id,
                **coordinate.model_dump(),
            )
            result = None
            if request is None:
                # enqueue_system preserves the source obligation on capacity rejection.
                try:
                    result = self.enqueue_system(
                        installation_id=installation_id, coordinate=coordinate, source=source
                    )
                except ManagementError as error:
                    if str(error) != "quota_exceeded":
                        raise
                    # Keep the obligation and let a later bounded sweep admit it.
            target = {
                "source": source.model_dump(mode="json"),
                "observed_at": observed_at.isoformat(),
            }
            self._exec(
                """INSERT INTO backtest_artifact_source_state
                (exchange,market_type,symbol,desired_target) VALUES
                (%(exchange)s,%(market_type)s,%(symbol)s,%(target)s)
                ON CONFLICT (exchange,market_type,symbol) DO UPDATE
                SET desired_target=EXCLUDED.desired_target""",
                **coordinate.model_dump(),
                target=Jsonb(target),
            )
            return result

    def finish_base(
        self,
        *,
        build: Mapping[str, Any],
        publication: Mapping[str, Any],
        coverage: Mapping[str, Any],
    ) -> None:
        with self.db.transaction():
            self.finish(build=build, outcome="ready", publication=publication)
            # Request readiness means committed eligible inventory, not all-timeframe coverage.
            # Explicit partial/insufficient state remains in the source projection.
            self._exec(
                """UPDATE backtest_artifact_source_state SET published=published || %(data)s
                WHERE exchange=%(e)s AND market_type=%(m)s AND symbol=%(s)s""",
                data=Jsonb(dict(coverage)),
                e=build["exchange"],
                m=build["market_type"],
                s=build["symbol"],
            )
