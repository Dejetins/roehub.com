"""Generation-first publication/recovery fence over canonical source revisions."""

from contextlib import contextmanager
from datetime import timedelta

from trading.contexts.backtest_artifacts.application.services.management import ManagementError

_MARKETS = {
    ("binance", "spot"): 1,
    ("binance", "futures"): 2,
    ("bybit", "spot"): 3,
    ("bybit", "futures"): 4,
}


class PostgresPublicationFence:
    def __init__(self, gateway):
        self.db = gateway

    def _read(self, coordinate):
        row = self.db.fetch_one(
            query="""SELECT g.generation::text AS source_generation,
            g.content_revision,g.funding_revision,g.last_correction_revision,
            g.append_safe_revision,s.last_ts FROM market_data.instrument_generations g
            JOIN market_data.candle_coverage_summary s USING(market_id,symbol,generation)
            WHERE g.market_id=%(m)s AND g.symbol=%(s)s AND g.index_state='ready'
            AND s.actual_count>0 FOR SHARE OF g""",
            parameters={
                "m": _MARKETS[(coordinate.exchange, coordinate.market_type)],
                "s": coordinate.symbol,
            },
        )
        if row is None:
            raise ManagementError("source_unavailable")
        return dict(row)

    def capture(self, coordinate, *, frozen_end=None):
        row = self._read(coordinate)
        result = {
            key: row[key] for key in ("source_generation", "content_revision", "funding_revision")
        }
        if frozen_end is not None:
            result["append_safe_prefix"] = frozen_end <= row["last_ts"] + timedelta(minutes=1)
            result["frozen_end"] = frozen_end.isoformat()
        return result

    @contextmanager
    def guard(self, coordinate, expected):
        with self.db.transaction():
            current = self._read(coordinate)
            if expected is None or not (
                current["source_generation"] == expected.get("source_generation")
                and current["funding_revision"] == expected.get("funding_revision")
                and isinstance(expected.get("content_revision"), int)
                and (
                    current["content_revision"] == expected["content_revision"]
                    or (
                        expected.get("append_safe_prefix") is True
                        and current["append_safe_revision"] == current["content_revision"]
                        and current["last_correction_revision"]
                        <= expected["content_revision"]
                        <= current["content_revision"]
                    )
                )
            ):
                raise ManagementError("source_revision_changed")
            yield
