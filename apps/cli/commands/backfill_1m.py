from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Protocol, Sequence

from apps.cli.wiring.modules.market_data import MarketDataBackfill1mWiring
from trading.shared_kernel.primitives import InstrumentId, MarketId, Symbol, TimeRange, UtcTimestamp

logger = logging.getLogger(__name__)


class UtcTimestampParser(Protocol):
    def parse(self, value: str) -> UtcTimestamp:
        ...


class BatchSizeParser(Protocol):
    def parse(self, value: str | None) -> int | None:
        ...


class ReportPresenter(Protocol):
    def render(self, report: object) -> str:
        ...


class IsoUtcTimestampParser(UtcTimestampParser):
    """
    Парсер ISO timestamps.

    Требование CLI-дока: ISO UTC с 'Z'.
    Реализация:
    - принимает ...Z и ...+00:00
    - запрещает naive datetime
    - приводит к UTC через UtcTimestamp (он сам нормализует UTC+ms)
    """

    def parse(self, value: str) -> UtcTimestamp:
        text = value.strip()
        if not text:
            raise ValueError("timestamp must be non-empty")

        # datetime.fromisoformat не понимает 'Z'
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"

        dt = datetime.fromisoformat(text)

        if dt.tzinfo is None or dt.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware UTC (naive datetime is forbidden)")

        return UtcTimestamp(dt.astimezone(timezone.utc))


class OptionalIntBatchSizeParser(BatchSizeParser):
    """
    --batch-size:
    - None / отсутствует => None
    - строка 'None' (любой регистр) => None
    - int > 0 => int
    """

    def parse(self, value: str | None) -> int | None:
        if value is None:
            return None

        text = value.strip()
        if not text:
            return None

        if text.lower() == "none":
            return None

        n = int(text)
        if n <= 0:
            raise ValueError("batch_size must be a positive integer or None")
        return n


class TextReportPresenter(ReportPresenter):
    def render(self, report: object) -> str:
        # v1: без требований к формату отчёта use-case -> логируем repr
        return repr(report)


class JsonReportPresenter(ReportPresenter):
    def render(self, report: object) -> str:
        # v1: пытаемся сериализовать через as_dict(); иначе repr
        as_dict = getattr(report, "as_dict", None)
        if callable(as_dict):
            payload = as_dict()
            return json.dumps(payload, ensure_ascii=False, default=str)

        return json.dumps({"report": repr(report)}, ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Backfill1mCliArgs:
    market_id: int
    symbol: str
    start: UtcTimestamp
    end: UtcTimestamp
    parquet_paths: tuple[str, ...]
    batch_size: int | None
    report_format: str

    def instrument_id(self) -> InstrumentId:
        return InstrumentId(MarketId(self.market_id), Symbol(self.symbol))

    def time_range(self) -> TimeRange:
        return TimeRange(self.start, self.end)


class Backfill1mCli:
    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        ts_parser: UtcTimestampParser | None = None,
        batch_parser: BatchSizeParser | None = None,
    ) -> None:
        self._environ = dict(environ) if environ is not None else None
        self._ts_parser = ts_parser if ts_parser is not None else IsoUtcTimestampParser()
        self._batch_parser = batch_parser if batch_parser is not None else OptionalIntBatchSizeParser()  # noqa: E501

    def run(self, argv: Sequence[str]) -> int:
        ns = self._build_arg_parser().parse_args(list(argv))
        try:
            args = self._to_args(ns)
        except Exception as e:  # noqa: BLE001
            logger.error("Invalid arguments: %s", e)
            return 2

        presenter = self._presenter(args.report_format)

        wiring = MarketDataBackfill1mWiring(environ=self._effective_environ())

        try:
            use_case = wiring.use_case(parquet_paths=args.parquet_paths, batch_size=args.batch_size)

            # Use-case contract: run(command) -> report
            from trading.contexts.market_data.application.use_cases.backfill_1m_candles import (
                Backfill1mCommand,
            )

            command = Backfill1mCommand(
                instrument_id=args.instrument_id(),
                time_range=args.time_range(),
            )

            report = use_case.run(command)

        except Exception as e:  # noqa: BLE001
            logger.exception("Backfill failed: %s", e)
            return 1

        logger.info("%s", presenter.render(report))
        return 0

    def _effective_environ(self) -> Mapping[str, str]:
        if self._environ is not None:
            return self._environ
        import os

        return os.environ

    def _build_arg_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(prog="backfill-1m", add_help=True)

        p.add_argument("--market-id", required=True, type=int, help="MarketId (1..4)")
        p.add_argument("--symbol", required=True, type=str, help="Symbol, e.g. BTCUSDT")
        p.add_argument(
            "--start",
            required=True,
            type=str,
            help="UTC ISO timestamp, e.g. 2026-02-01T00:00:00Z",
        )
        p.add_argument(
            "--end",
            required=True,
            type=str,
            help="UTC ISO timestamp, e.g. 2026-02-02T00:00:00Z",
        )
        p.add_argument(
            "--parquet",
            required=True,
            action="append",
            help="Path to parquet file or directory. Can be passed multiple times.",
        )

        p.add_argument("--batch-size", default='10000', type=str, help="Positive int or None (default None)") # noqa: E501
        p.add_argument(
            "--report-format",
            default="text",
            choices=("text", "json"),
            help="Report format for logging",
        )

        return p

    def _to_args(self, ns: argparse.Namespace) -> Backfill1mCliArgs:
        market_id = int(ns.market_id)
        if market_id < 1 or market_id > 4:
            raise ValueError("market_id must be in 1..4")

        symbol = str(ns.symbol).strip()
        if not symbol:
            raise ValueError("symbol must be non-empty")

        start = self._ts_parser.parse(str(ns.start))
        end = self._ts_parser.parse(str(ns.end))
        if start.value >= end.value:
            raise ValueError("start must be < end")

        parquet_paths = tuple(str(x) for x in (ns.parquet or []))
        if not parquet_paths:
            raise ValueError("--parquet must be provided at least once")

        batch_size = self._batch_parser.parse(ns.batch_size)

        return Backfill1mCliArgs(
            market_id=market_id,
            symbol=symbol,
            start=start,
            end=end,
            parquet_paths=parquet_paths,
            batch_size=batch_size,
            report_format=str(ns.report_format),
        )

    def _presenter(self, report_format: str) -> ReportPresenter:
        if report_format == "json":
            return JsonReportPresenter()
        return TextReportPresenter()
