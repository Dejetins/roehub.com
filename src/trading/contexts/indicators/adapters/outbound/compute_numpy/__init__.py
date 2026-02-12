"""
Numpy oracle adapters for indicators compute validation.

Docs: docs/architecture/indicators/indicators-ma-compute-numba-v1.md
Related: trading.contexts.indicators.adapters.outbound.compute_numba.kernels.ma
"""

from .ma import compute_ma_grid_f32, is_supported_ma_indicator

__all__ = [
    "compute_ma_grid_f32",
    "is_supported_ma_indicator",
]
