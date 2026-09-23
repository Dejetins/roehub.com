"""Disposable-child diagnostic observers. No adapter installation in product code.

Adapted from inspected study research_score_audit.py/details.py. All temporary
bindings restore in finally, including failing service calls. Timing mode never
enters these contexts. Serialization/capture overhead is not performance evidence.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
from contextlib import contextmanager
from typing import Any

import numpy as np


@contextmanager
def score_capture():
    records: list[dict[str, Any]] = []
    originals = []
    try:
        for name, class_name, hook in (
            (
                "no_risk_exact",
                "BacktestNoRiskExactScoringService",
                "_update_heap_from_score_values",
            ),
            ("tp_sl_exact", "BacktestTpSlExactScoringService", "_update_heap_ranked"),
        ):
            module = importlib.import_module(
                "trading.contexts.backtest.application.services.v2." + name
            )
            cls = getattr(module, class_name)
            original = cls.execute
            originals.append((cls, original))
            cls.execute = _observer(module, original, hook, name, records)
        yield records
    finally:
        for cls, original in reversed(originals):
            cls.execute = original


def _observer(module, execute, hook, name, records):
    @functools.wraps(execute)
    def observed(self, **kwargs):
        original = getattr(module, hook)
        record: dict[str, Any] = {
            "service": name,
            "evaluated": 0,
            "eligible": 0,
            "row_identities": [],
            "batches": [],
        }

        def capture(**kw):
            context, buffers = kw["top_k_context"], kw["buffers"]
            selected = kw["selected_rows_by_indicator"]
            rows = np.vstack(
                [
                    ids[selected[indicator]]
                    for indicator, ids in zip(
                        context.indicator_ids, context.row_ids_by_pos, strict=True
                    )
                ]
            )
            # Both current call sites supply the requested ranking metric explicitly.
            # Fail closed if the seam changes rather than silently recording return.
            scores = kw["score_values"]
            arrays = {
                "rows": rows.copy(),
                "score": scores.copy(),
                "trade_count": buffers.trade_count.copy(),
                "eligible": (buffers.trade_count >= kw["min_closed_trades"]).copy(),
            }
            for metric in dataclasses.fields(buffers):
                value = getattr(buffers, metric.name)
                if isinstance(value, np.ndarray):
                    arrays["metric_" + metric.name] = value.copy()
            for key in ("best_tp_idx", "best_sl_idx"):
                if hasattr(buffers, key):
                    arrays[key] = getattr(buffers, key).copy()
            record["batches"].append(
                {
                    "indicators": list(context.indicator_ids),
                    "minimum_trades": kw["min_closed_trades"],
                    "multiplier": kw["score_multiplier"],
                    "arrays": arrays,
                }
            )
            record["row_identities"].extend(rows.T.tolist())
            record["evaluated"] += len(scores)
            record["eligible"] += int(np.count_nonzero(arrays["eligible"]))
            return original(**kw)

        setattr(module, hook, capture)
        try:
            result = execute(self, **kwargs)
            records.append(record)
            return result
        finally:
            setattr(module, hook, original)

    return observed
