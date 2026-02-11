from __future__ import annotations


class MissingInputSeriesError(ValueError):
    """
    Raised when required candle input series cannot be loaded for compute.

    Docs: docs/architecture/indicators/indicators-application-ports-walking-skeleton-v1.md
    Related: ...application.dto.candle_arrays, ...application.ports.feeds.candle_feed
    """
