from __future__ import annotations

from .missing_input_series_error import MissingInputSeriesError


class MissingRequiredSeries(MissingInputSeriesError):
    """
    Raised when compute request lacks one or more required input series.

    Docs: docs/architecture/indicators/indicators-compute-engine-core.md
    Related: trading.contexts.indicators.application.dto.candle_arrays,
      trading.contexts.indicators.adapters.outbound.compute_numba.engine
    """
