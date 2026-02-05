from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

from apps.cli.wiring.clients.parquet import ParquetSourceFactory
from apps.cli.wiring.db.clickhouse import ClickHouseWriterFactory
from trading.contexts.market_data.application.ports.clock.clock import Clock
from trading.platform.time.system_clock import SystemClock

if TYPE_CHECKING:
    # Только для типов (чтобы модуль CLI импортировался без импорта use-case на runtime)
    from trading.contexts.market_data.application.use_cases.backfill_1m_candles import (
        Backfill1mCommand,
        Backfill1mReport,
    )


class Backfill1mUseCase(Protocol):
    def run(self, command: "Backfill1mCommand") -> "Backfill1mReport":
        ...


@dataclass(frozen=True, slots=True)
class MarketDataBackfill1mWiring:
    """
    Composition root для CLI backfill-1m.

    Env — источник правды.
    """

    environ: Mapping[str, str]

    def use_case(self, *, parquet_paths: Sequence[str], batch_size: int | None) -> Backfill1mUseCase:  # noqa: E501
        clock: Clock = SystemClock()

        source = ParquetSourceFactory(clock=clock).source(paths=parquet_paths, batch_size=batch_size)  # noqa: E501
        writer = ClickHouseWriterFactory(environ=self.environ).writer()

        # use-case у тебя именно здесь:
        from trading.contexts.market_data.application.use_cases.backfill_1m_candles import (
            Backfill1mCandlesUseCase,
        )

        # ВАЖНО: твой use-case принимает int, поэтому None преобразуем в "один батч"
        effective_batch_size = batch_size if batch_size is not None else 2_147_483_647

        return Backfill1mCandlesUseCase(
            source=source,
            writer=writer,
            clock=clock,
            batch_size=effective_batch_size,
        )
