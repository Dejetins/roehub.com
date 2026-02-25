# Notebooks

These notebooks are intended as runnable regression/performance checks for the backtest subsystem.

They are not part of `pytest`.

Recommended environment:

- Python 3.12
- project deps installed (notably `numba`)

Run:

- `jupyter lab` (or `jupyter notebook`)
- open and execute the notebooks in order

Notes:

- The notebooks default to a small synthetic candle dataset, so they do not require database access.
- If `numba` is not installed, the regression notebook can fall back to a minimal NumPy-only SMA compute for basic sanity, but golden checks require the real Numba compute engine.
