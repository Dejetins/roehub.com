from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from trading.contexts.market_data.adapters.outbound.persistence.clickhouse.gateway import (
    ClickHouseGateway,
)
from trading.contexts.market_data.application.dto import FundingInstrument, FundingRateRecord
from trading.contexts.market_data.application.ports.stores.funding_instrument_universe_store import (  # noqa: E501
    FundingInstrumentUniverseStore,
)
from trading.contexts.market_data.application.ports.stores.funding_rate_coverage_reader import (
    FundingRateArtifactRecord,
    FundingRateCoverageReader,
    FundingRateCoverageSnapshot,
)
from trading.contexts.market_data.application.ports.stores.funding_rate_writer import (
    FundingRateWriter,
)
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp


class ClickHouseFundingRateStore(
    FundingRateWriter,
    FundingInstrumentUniverseStore,
    FundingRateCoverageReader,
):
    def __init__(self, *, gateway: ClickHouseGateway, database: str = "market_data") -> None:
        if gateway is None:  # type: ignore[truthy-bool]
            raise ValueError("ClickHouseFundingRateStore requires gateway")
        if not database.strip():
            raise ValueError("ClickHouseFundingRateStore requires non-empty database")
        self._gw = gateway
        self._db = database.strip()

    def upsert_funding_instruments(self, rows: Sequence[FundingInstrument]) -> None:
        payload = [self._universe_payload(row) for row in rows]
        self._gw.insert_rows(f"{self._db}.funding_instrument_universe", payload)

    def list_tradable_funding_instruments(
        self,
        *,
        market_ids: Sequence[MarketId],
    ) -> Sequence[FundingInstrument]:
        if not market_ids:
            return ()
        q = f"""
        SELECT
            market_id,
            symbol,
            argMax(instrument_key, updated_at) AS instrument_key,
            argMax(exchange, updated_at) AS exchange,
            argMax(market_type, updated_at) AS market_type,
            argMax(status, updated_at) AS status,
            argMax(is_tradable, updated_at) AS is_tradable,
            argMax(base_asset, updated_at) AS base_asset,
            argMax(quote_asset, updated_at) AS quote_asset,
            argMax(funding_interval_minutes, updated_at) AS funding_interval_minutes,
            argMax(funding_interval_source, updated_at) AS funding_interval_source,
            argMax(funding_cap, updated_at) AS funding_cap,
            argMax(funding_floor, updated_at) AS funding_floor,
            max(updated_at) AS latest_updated_at
        FROM {self._db}.funding_instrument_universe
        WHERE market_id IN %(market_ids)s
        GROUP BY market_id, symbol
        HAVING is_tradable = 1
        ORDER BY market_id, symbol
        """
        rows = self._gw.select(q, {"market_ids": tuple(int(m.value) for m in market_ids)})
        return tuple(self._universe_from_row(row) for row in rows)

    def get_funding_instrument(self, instrument_id: InstrumentId) -> FundingInstrument | None:
        rows = self.list_tradable_funding_instruments(market_ids=(instrument_id.market_id,))
        symbol = str(instrument_id.symbol).upper()
        for row in rows:
            if str(row.instrument_id.symbol).upper() == symbol:
                return row
        return None

    def write_funding_rates(self, rows: Sequence[FundingRateRecord]) -> None:
        materialized = list(rows)
        if not materialized:
            return
        canonical = [self._canonical_payload(row) for row in materialized]
        self._gw.insert_rows(f"{self._db}.canonical_funding_rates", canonical)

        binance = [
            self._binance_raw_payload(row)
            for row in materialized
            if int(row.instrument_id.market_id.value) == 2
        ]
        bybit = [
            self._bybit_raw_payload(row)
            for row in materialized
            if int(row.instrument_id.market_id.value) == 4
        ]
        self._gw.insert_rows(f"{self._db}.raw_binance_funding_rates", binance)
        self._gw.insert_rows(f"{self._db}.raw_bybit_funding_rates", bybit)

    def latest_funding_time(self, instrument_id: InstrumentId) -> UtcTimestamp | None:
        q = f"""
        SELECT maxOrNull(toUnixTimestamp64Milli(funding_time)) AS funding_time_ms
        FROM {self._db}.canonical_funding_rates
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
        """
        rows = self._gw.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
            },
        )
        if not rows:
            return None
        value = rows[0].get("funding_time_ms")
        if value is None:
            return None
        return UtcTimestamp(_dt_from_epoch_ms(int(value)))

    def read_funding_coverage(
        self,
        *,
        instrument_id: InstrumentId,
        time_range: TimeRange,
    ) -> FundingRateCoverageSnapshot:
        q = f"""
        SELECT
            market_id,
            symbol,
            argMax(instrument_key, ingested_at) AS instrument_key,
            toUnixTimestamp64Milli(funding_time) AS funding_time_ms,
            argMax(funding_rate, ingested_at) AS funding_rate,
            CAST(NULL AS Nullable(Float64)) AS mark_price,
            argMax(funding_interval_minutes, ingested_at) AS funding_interval_minutes,
            1 AS data_quality
        FROM {self._db}.canonical_funding_rates
        WHERE market_id = %(market_id)s
          AND symbol = %(symbol)s
          AND funding_time >= %(start)s
          AND funding_time < %(end)s
        GROUP BY market_id, symbol, funding_time
        ORDER BY funding_time ASC
        """
        rows = self._gw.select(
            q,
            {
                "market_id": int(instrument_id.market_id.value),
                "symbol": str(instrument_id.symbol),
                "start": _ensure_utc(time_range.start.value),
                "end": _ensure_utc(time_range.end.value),
            },
        )
        records = tuple(self._artifact_record_from_row(row) for row in rows)
        return _coverage_snapshot_from_records(
            instrument_id=instrument_id,
            time_range=time_range,
            records=records,
        )

    def _universe_payload(self, row: FundingInstrument) -> Mapping[str, Any]:
        return {
            "market_id": int(row.instrument_id.market_id.value),
            "symbol": str(row.instrument_id.symbol),
            "instrument_key": row.instrument_key,
            "exchange": row.exchange,
            "market_type": row.market_type,
            "status": row.status,
            "is_tradable": int(row.is_tradable),
            "base_asset": row.base_asset,
            "quote_asset": row.quote_asset,
            "funding_interval_minutes": row.funding_interval_minutes,
            "funding_interval_source": row.funding_interval_source,
            "funding_cap": row.funding_cap,
            "funding_floor": row.funding_floor,
            "updated_at": _ensure_utc(row.updated_at.value),
        }

    def _universe_from_row(self, row: Mapping[str, Any]) -> FundingInstrument:
        market_id = MarketId(int(row["market_id"]))
        symbol = Symbol(str(row["symbol"]))
        return FundingInstrument(
            instrument_id=InstrumentId(market_id, symbol),
            instrument_key=str(row["instrument_key"]),
            exchange=str(row["exchange"]),
            market_type=str(row["market_type"]),
            status=str(row["status"]),
            is_tradable=int(row["is_tradable"]),
            base_asset=_optional_str(row.get("base_asset")),
            quote_asset=_optional_str(row.get("quote_asset")),
            funding_interval_minutes=_optional_int(row.get("funding_interval_minutes")),
            funding_interval_source=_optional_str(row.get("funding_interval_source")),
            funding_cap=_optional_float(row.get("funding_cap")),
            funding_floor=_optional_float(row.get("funding_floor")),
            updated_at=UtcTimestamp(
                _clickhouse_datetime_utc(row.get("updated_at") or row["latest_updated_at"])
            ),
        )

    def _canonical_payload(self, row: FundingRateRecord) -> Mapping[str, Any]:
        return {
            "market_id": int(row.instrument_id.market_id.value),
            "symbol": str(row.instrument_id.symbol),
            "instrument_key": row.instrument_key,
            "funding_time": _ensure_utc(row.funding_time.value),
            "funding_rate": float(row.funding_rate),
            "funding_interval_minutes": int(row.funding_interval_minutes),
            "funding_interval_source": row.funding_interval_source,
            "source": row.source,
            "ingested_at": _ensure_utc(row.ingested_at.value),
            "ingest_id": row.ingest_id,
        }

    def _binance_raw_payload(self, row: FundingRateRecord) -> Mapping[str, Any]:
        payload = dict(self._canonical_payload(row))
        payload["mark_price"] = row.mark_price
        return payload

    def _bybit_raw_payload(self, row: FundingRateRecord) -> Mapping[str, Any]:
        payload = dict(self._canonical_payload(row))
        payload["category"] = row.bybit_category or "linear"
        return payload

    def _artifact_record_from_row(self, row: Mapping[str, Any]) -> FundingRateArtifactRecord:
        instrument_id = InstrumentId(
            MarketId(int(row["market_id"])),
            Symbol(str(row["symbol"])),
        )
        return FundingRateArtifactRecord(
            instrument_id=instrument_id,
            instrument_key=str(row["instrument_key"]),
            funding_time=UtcTimestamp(_dt_from_epoch_ms(int(row["funding_time_ms"]))),
            funding_rate=float(row["funding_rate"]),
            mark_price=_optional_float(row.get("mark_price")),
            funding_interval_minutes=int(row["funding_interval_minutes"]),
            data_quality=int(row["data_quality"]),
        )


def _coverage_snapshot_from_records(
    *,
    instrument_id: InstrumentId,
    time_range: TimeRange,
    records: tuple[FundingRateArtifactRecord, ...],
) -> FundingRateCoverageSnapshot:
    if not records:
        return FundingRateCoverageSnapshot(
            status="unavailable",
            coverage_policy="unavailable",
            requested_range=time_range,
            available_start=None,
            available_end=None,
            expected_event_count=0,
            observed_event_count=0,
            missing_event_count=0,
            reason_codes=("no_funding_rows",),
            records=(),
        )
    interval_ms = int(records[0].funding_interval_minutes) * 60 * 1000
    start_ms = _epoch_ms(time_range.start.value)
    end_ms = _epoch_ms(time_range.end.value)
    times_ms = tuple(_epoch_ms(record.funding_time.value) for record in records)
    missing_event_count = 0
    reason_codes: list[str] = []
    first_ms = times_ms[0]
    last_ms = times_ms[-1]
    if first_ms > start_ms + interval_ms:
        missing_event_count += 1
        reason_codes.append("missing_leading_coverage")
    for previous_ms, current_ms in zip(times_ms, times_ms[1:]):
        interval_gap_count = max(0, round((current_ms - previous_ms) / interval_ms) - 1)
        if interval_gap_count > 0:
            missing_event_count += interval_gap_count
            if "funding_interval_gap" not in reason_codes:
                reason_codes.append("funding_interval_gap")
    if last_ms < end_ms - interval_ms:
        missing_event_count += 1
        reason_codes.append("missing_trailing_coverage")
    status = "ready" if missing_event_count == 0 else "degraded"
    return FundingRateCoverageSnapshot(
        status=status,
        coverage_policy="ready" if status == "ready" else "degraded_with_warning",
        requested_range=time_range,
        available_start=records[0].funding_time,
        available_end=records[-1].funding_time,
        expected_event_count=len(records) + missing_event_count,
        observed_event_count=len(records),
        missing_event_count=missing_event_count,
        reason_codes=tuple(reason_codes),
        records=records,
    )


def _epoch_ms(value: datetime) -> int:
    return int(_ensure_utc(value).timestamp() * 1000)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc)


def _clickhouse_datetime_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _dt_from_epoch_ms(value: int) -> datetime:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
