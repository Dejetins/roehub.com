from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

from apps.cli.wiring.clients.parquet import ParquetSourceFactory
from apps.cli.wiring.db.clickhouse import ClickHouseWriterFactory
from trading.contexts.market_data.adapters.outbound.config.runtime_config import (
    load_market_data_runtime_config,
)
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
    market_data_config_path: str

    def use_case(self, *, parquet_paths: Sequence[str], batch_size: int | None) -> Backfill1mUseCase:  # noqa: E501
        """
        Build a fully wired backfill use-case instance for CLI command execution.

        Parameters:
        - parquet_paths: list of parquet inputs used by source scanner.
        - batch_size: optional writer batch size requested by CLI.

        Returns:
        - Ready-to-run `Backfill1mUseCase`.

        Assumptions/Invariants:
        - `market_data_config_path` points to a readable runtime config file.
        - ClickHouse settings are supplied through environment.

        Errors/Exceptions:
        - Propagates config parsing, source creation, and writer wiring errors.

        Side effects:
        - Loads runtime config from filesystem.
        """
        clock: Clock = SystemClock()
        cfg = load_market_data_runtime_config(Path(self.market_data_config_path))

        source = ParquetSourceFactory(clock=clock, cfg=cfg).source(
            paths=parquet_paths,
            batch_size=batch_size,
        )
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
